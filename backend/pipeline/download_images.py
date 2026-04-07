"""download_images.py – Stage 3a of the Met Retrieval pipeline.

Downloads artifact images from remote URLs to a local cache directory using a
thread-pool for concurrency.  Produces ``data/images_manifest.parquet`` that
maps each artifact's DataFrame index to its local JPEG path (or ``None`` when
the download fails).  This manifest is the sole input expected by
``features.py`` for GPU feature extraction.

Pipeline order
--------------
ingest.py → geocode.py → download_images.py → features.py

Usage
-----
    uv run backend/pipeline/download_images.py [--workers N] [--timeout S]

Dependencies added via ``uv add`` are the same ones already present in the
project; no new packages are required.
"""

import argparse
import logging
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

DATA_DIR = Path("data")
INPUT_PATH = DATA_DIR / "aic_geocoded.parquet"
IMAGES_DIR = DATA_DIR / "images"
OUTPUT_MANIFEST_PATH = DATA_DIR / "images_manifest.parquet"

# Reasonable JPEG quality for storage; PIL will still open any source format.
JPEG_QUALITY = 90


def _download_one(
    idx: int,
    url: str,
    dest: Path,
    timeout: int,
) -> tuple[int, str | None]:
    """Fetch *url*, save as JPEG to *dest*, and return ``(idx, path_str)``.

    Returns ``(idx, None)`` on any failure so the caller can track missing
    images without crashing.
    """
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        image.save(dest, format="JPEG", quality=JPEG_QUALITY)
        return idx, str(dest)
    except Exception as exc:  # noqa: BLE001
        logging.debug("Failed to download idx=%d url=%s: %s", idx, url, exc)
        return idx, None


def download_all(
    df: pd.DataFrame,
    images_dir: Path,
    *,
    workers: int = 32,
    timeout: int = 15,
) -> dict[int, str | None]:
    """Download all images referenced in *df* and return an index→path mapping.

    Parameters
    ----------
    df:
        DataFrame produced by ``geocode.py``.  Must contain a ``Primary Image``
        column with HTTP(S) URLs.
    images_dir:
        Directory where downloaded JPEG files will be stored.
    workers:
        Number of parallel download threads.  Higher values are bounded by
        network bandwidth and remote server rate limits.
    timeout:
        Per-request HTTP timeout in seconds.

    Returns
    -------
    dict[int, str | None]
        Mapping from DataFrame positional index (0-based) to the local JPEG
        path string, or ``None`` if the download failed or no URL was present.
    """
    images_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[int, str, Path]] = []
    for idx, row in df.iterrows():
        url = row.get("Primary Image")
        if pd.notna(url) and str(url).startswith("http"):
            dest = images_dir / f"{idx}.jpg"
            tasks.append((idx, str(url), dest))

    logging.info(
        "Scheduling %d image downloads across %d threads (timeout=%ds).",
        len(tasks),
        workers,
        timeout,
    )

    results: dict[int, str | None] = {i: None for i in range(len(df))}
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_one, idx, url, dest, timeout): idx
            for idx, url, dest in tasks
        }
        for future in as_completed(futures):
            idx, path = future.result()
            results[idx] = path
            completed += 1
            if completed % 500 == 0 or completed == len(tasks):
                ok = sum(1 for v in results.values() if v is not None)
                logging.info(
                    "Progress: %d / %d done  (%d succeeded so far).",
                    completed,
                    len(tasks),
                    ok,
                )

    return results


def main(workers: int = 32, timeout: int = 15) -> None:
    if not INPUT_PATH.exists():
        logging.error(
            "Input file %s not found. Run geocode.py first.", INPUT_PATH
        )
        return

    df = pd.read_parquet(INPUT_PATH)
    logging.info("Loaded %d rows from %s.", len(df), INPUT_PATH)

    results = download_all(df, IMAGES_DIR, workers=workers, timeout=timeout)

    # Build manifest aligned with the DataFrame's positional index.
    manifest = pd.DataFrame(
        {
            "df_index": list(results.keys()),
            "local_path": list(results.values()),
        }
    ).sort_values("df_index")

    n_ok = manifest["local_path"].notna().sum()
    n_fail = len(manifest) - n_ok
    logging.info(
        "Download complete: %d succeeded, %d failed / missing.",
        n_ok,
        n_fail,
    )

    manifest.to_parquet(OUTPUT_MANIFEST_PATH, index=False)
    logging.info("Manifest written to %s.", OUTPUT_MANIFEST_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Met artifact images to a local cache."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of parallel download threads (default: 32).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="Per-request HTTP timeout in seconds (default: 15).",
    )
    args = parser.parse_args()
    main(workers=args.workers, timeout=args.timeout)
