"""features.py – Stage 3b of the Met Retrieval pipeline.

Extracts DINOv2 image embeddings and Nomic text embeddings from pre-downloaded
local images.  Assumes ``download_images.py`` has already populated
``data/images/`` and written ``data/images_manifest.parquet``.

Because all I/O is local disk access, the DataLoader can safely use multiple
workers (``--num-workers``) so the GPU stays saturated between batches.

Pipeline order
--------------
ingest.py → geocode.py → download_images.py → features.py

Usage
-----
    uv run backend/pipeline/features.py [--batch-size N] [--num-workers N]
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

DATA_DIR = Path("data")
INPUT_GEOCODED_PATH = DATA_DIR / "aic_geocoded.parquet"
INPUT_MANIFEST_PATH = DATA_DIR / "images_manifest.parquet"
OUTPUT_METADATA_PATH = DATA_DIR / "metadata_index.parquet"
OUTPUT_IMAGES_PATH = DATA_DIR / "images_unprojected.pt"
OUTPUT_TEXTS_PATH = DATA_DIR / "text_unprojected.pt"


class LocalImageArtifactDataset(Dataset):
    """Dataset that loads pre-downloaded JPEG images from disk.

    Parameters
    ----------
    df:
        The geocoded artifact DataFrame.
    manifest:
        DataFrame with columns ``df_index`` (int) and ``local_path``
        (str | None) produced by ``download_images.py``.
    """

    _TRANSFORM = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    def __init__(self, df: pd.DataFrame, manifest: pd.DataFrame) -> None:
        self.df = df.reset_index(drop=True)
        # Build a fast positional-index → local_path lookup.
        self._path_map: dict[int, str | None] = dict(
            zip(manifest["df_index"], manifest["local_path"])
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        row = self.df.iloc[idx]
        local_path = self._path_map.get(idx)
        text: str = "search_document: " + str(row.get("text_serialized", ""))

        # Zero tensor is the safe fallback for missing / corrupt images.
        image_tensor = torch.zeros((3, 224, 224))

        if local_path is not None:
            path = Path(local_path)
            if path.exists():
                try:
                    image = Image.open(path).convert("RGB")
                    image_tensor = self._TRANSFORM(image)
                except Exception as exc:  # noqa: BLE001
                    logging.debug("Could not load image %s: %s", path, exc)

        return image_tensor, text


def _mean_pool(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Attention-mask-weighted mean pooling over the token dimension."""
    mask_expanded = (
        attention_mask.unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )
    return torch.sum(token_embeddings * mask_expanded, dim=1) / torch.clamp(
        mask_expanded.sum(dim=1), min=1e-9
    )


def main(batch_size: int = 64, num_workers: int = 4) -> None:
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not INPUT_GEOCODED_PATH.exists():
        logging.error(f"Input {INPUT_GEOCODED_PATH} missing. Run geocode.py first.")
        return

    df = pd.read_parquet(INPUT_GEOCODED_PATH)
    
    if INPUT_MANIFEST_PATH.exists():
        manifest = pd.read_parquet(INPUT_MANIFEST_PATH)
        logging.info(f"Using manifest from {INPUT_MANIFEST_PATH}")
    else:
        logging.warning(f"Manifest {INPUT_MANIFEST_PATH} not found. Auto-scanning {DATA_DIR / 'images'}...")
        image_dir = DATA_DIR / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # Scan for files matching {index}.jpg
        found_paths = {}
        for img_path in image_dir.glob("*.jpg"):
            try:
                idx = int(img_path.stem)
                found_paths[idx] = str(img_path)
            except ValueError:
                continue
        
        manifest = pd.DataFrame([
            {"df_index": i, "local_path": found_paths.get(i)}
            for i in range(len(df))
        ])
        logging.info(f"Auto-scan found {len(found_paths)} valid images.")

    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info("Using device: %s", device)


    logging.info(
        "Loaded %d artifacts; manifest covers %d entries (%d with images).",
        len(df),
        len(manifest),
        manifest["local_path"].notna().sum(),
    )

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    logging.info("Loading DINOv2-small for image embeddings…")
    image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")  # noqa: F841
    image_model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)

    logging.info("Loading nomic-embed-text-v1.5 for text embeddings…")
    tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
    text_model = AutoModel.from_pretrained(
        "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
    ).to(device)

    # ------------------------------------------------------------------
    # Build DataLoader
    # ------------------------------------------------------------------
    dataset = LocalImageArtifactDataset(df, manifest)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    logging.info(
        "DataLoader: %d items, batch_size=%d, num_workers=%d.",
        len(dataset),
        batch_size,
        num_workers,
    )

    # ------------------------------------------------------------------
    # Feature extraction (GPU-only inner loop)
    # ------------------------------------------------------------------
    all_image_features: list[torch.Tensor] = []
    all_text_features: list[torch.Tensor] = []

    image_model.eval()
    text_model.eval()

    n_batches = len(dataloader)
    logging.info("Starting feature extraction over %d batches…", n_batches)

    with torch.no_grad():
        for i, (images, texts) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)

            # Image: CLS token of DINOv2.
            img_out = image_model(pixel_values=images)
            img_feat = img_out.last_hidden_state[:, 0, :]
            all_image_features.append(img_feat.cpu())

            # Text: attention-masked mean pooling over Nomic embeddings.
            enc = tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            txt_out = text_model(**enc)
            txt_feat = _mean_pool(
                txt_out.last_hidden_state, enc["attention_mask"]
            )
            all_text_features.append(txt_feat.cpu())

            if (i + 1) % 10 == 0 or (i + 1) == n_batches:
                logging.info("Processed batch %d / %d.", i + 1, n_batches)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    image_tensor_full = torch.cat(all_image_features, dim=0)
    text_tensor_full = torch.cat(all_text_features, dim=0)

    logging.info(
        "Saving image features %s → %s", image_tensor_full.shape, OUTPUT_IMAGES_PATH
    )
    torch.save(image_tensor_full, OUTPUT_IMAGES_PATH)

    logging.info(
        "Saving text features %s → %s", text_tensor_full.shape, OUTPUT_TEXTS_PATH
    )
    torch.save(text_tensor_full, OUTPUT_TEXTS_PATH)

    # Stamp has_image into the metadata so retrieval layers can mask
    # image-less items without re-reading the manifest at query time.
    has_image_series = (
        manifest.set_index("df_index")["local_path"].notna().reindex(range(len(df))).fillna(False)
    )
    df = df.copy()
    df["has_image"] = has_image_series.values

    logging.info(
        "has_image: %d / %d items have a downloaded image.",
        df["has_image"].sum(),
        len(df),
    )
    logging.info("Saving metadata index → %s", OUTPUT_METADATA_PATH)
    df.to_parquet(OUTPUT_METADATA_PATH, index=False)

    logging.info("Feature extraction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 + Nomic embeddings from local image cache."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for model forward passes (default: 64).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes for parallel image loading (default: 4).",
    )
    args = parser.parse_args()
    main(batch_size=args.batch_size, num_workers=args.num_workers)
