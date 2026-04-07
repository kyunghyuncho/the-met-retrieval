"""
Art Institute of Chicago — Data Ingestion Pipeline

Downloads artwork metadata via the AIC public API (paginated, 100 items/page),
filters to public-domain items with valid IIIF image IDs, constructs
deterministic image URLs, augments text via Wikipedia, and saves as Parquet.
"""
import asyncio
import aiohttp
import pandas as pd
import requests
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_DIR = Path("data")
OUTPUT_PATH = DATA_DIR / "aic_cleaned.parquet"
IIIF_BASE = "https://www.artic.edu/iiif/2"

AIC_API_BASE = "https://api.artic.edu/api/v1/artworks"
AIC_FIELDS = ",".join([
    "id", "title", "artist_display", "date_display",
    "medium_display", "image_id", "place_of_origin",
    "classification_title", "is_public_domain",
    "department_title", "style_title",
])


# ---------------------------------------------------------------------------
# 1. Paginated metadata download from AIC API
# ---------------------------------------------------------------------------

def download_metadata() -> pd.DataFrame:
    """Fetch all artworks from the AIC API, page by page."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    page = 1
    total_pages = None

    while True:
        url = f"{AIC_API_BASE}?page={page}&limit=100&fields={AIC_FIELDS}"
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    break
                time.sleep(2 * (attempt + 1))
            except requests.RequestException:
                time.sleep(2 * (attempt + 1))
        else:
            logging.warning(f"Skipping page {page} after 3 failures.")
            page += 1
            if total_pages and page > total_pages:
                break
            continue

        data = response.json()
        if total_pages is None:
            total_pages = data["pagination"]["total_pages"]
            logging.info(f"Total pages to fetch: {total_pages}")

        all_records.extend(data["data"])

        if page % 50 == 0 or page == total_pages:
            logging.info(f"Fetched page {page}/{total_pages} ({len(all_records)} records)")

        if page >= total_pages:
            break
        page += 1
        time.sleep(0.1)  # gentle pacing

    df = pd.DataFrame(all_records)
    logging.info(f"Downloaded {len(df)} total records from AIC API.")
    return df


# ---------------------------------------------------------------------------
# 2. Wikipedia augmentation (reused from prior pipeline)
# ---------------------------------------------------------------------------

async def fetch_wikipedia_extract(session, row, semaphore):
    """Search Wikipedia for a matching article and return the extract."""
    title = str(row.get("Title", ""))
    if len(title) <= 8 or title.lower() == "nan":
        return None

    origin = str(row.get("Country", ""))
    query = title
    if origin and origin.lower() != "nan" and origin.lower() != "none":
        query += f" {origin}"

    async with semaphore:
        try:
            search_url = (
                "https://en.wikipedia.org/w/rest.php/v1/search/page"
                f"?q={requests.utils.quote(query)}&limit=1"
            )
            async with session.get(search_url, timeout=5) as search_res:
                if search_res.status == 200:
                    search_data = await search_res.json()
                    pages = search_data.get("pages", [])
                    if pages:
                        page_title = pages[0].get("title", "")
                        # Fuzzy match guard
                        if (
                            title.lower()[:10] in page_title.lower()
                            or page_title.lower()[:10] in title.lower()
                        ):
                            summary_url = (
                                "https://en.wikipedia.org/api/rest_v1/page/summary/"
                                f"{requests.utils.quote(page_title)}"
                            )
                            async with session.get(summary_url, timeout=5) as wiki_res:
                                if wiki_res.status == 200:
                                    wiki_data = await wiki_res.json()
                                    return wiki_data.get("extract")
        except Exception:
            pass
    return None


async def augment_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Starting Wikipedia augmentation...")
    semaphore = asyncio.Semaphore(15)
    rows = df.to_dict("records")
    results: list[str | None] = []

    async with aiohttp.ClientSession() as session:
        chunk_size = 2000
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i : i + chunk_size]
            tasks = [fetch_wikipedia_extract(session, row, semaphore) for row in chunk]
            res = await asyncio.gather(*tasks)
            results.extend(res)
            logging.info(
                f"Augmented {min(i + chunk_size, len(rows))} / {len(rows)} records..."
            )

    df["Wikipedia Extract"] = results

    def resolve_text(row):
        if pd.notna(row.get("Wikipedia Extract")):
            return row["Wikipedia Extract"]
        return row["base_string"]

    df["text_serialized"] = df.apply(resolve_text, axis=1)
    df = df.drop(columns=["Wikipedia Extract"])
    logging.info("Wikipedia augmentation completed.")
    return df


# ---------------------------------------------------------------------------
# 3. Text serialisation
# ---------------------------------------------------------------------------

def generate_base_string(row) -> str:
    parts = []
    if pd.notna(row.get("Object Name")):
        parts.append(f"Artifact: {row['Object Name']}")
    if pd.notna(row.get("Title")):
        parts.append(f"Title: {row['Title']}")
    if pd.notna(row.get("Country")):
        parts.append(f"Origin: {row['Country']}")
    if pd.notna(row.get("Object Date")):
        parts.append(f"Period: {row['Object Date']}")
    if pd.notna(row.get("Medium")):
        parts.append(f"Medium: {row['Medium']}")
    if pd.notna(row.get("Artist Display Name")):
        parts.append(f"Artist: {row['Artist Display Name']}")
    return ". ".join(parts) + "." if parts else ""


# ---------------------------------------------------------------------------
# 4. Main pipeline
# ---------------------------------------------------------------------------

async def main():
    # Step 1 — Download metadata
    df = download_metadata()

    # Step 2 — Filter: public domain + has image
    df = df[df["is_public_domain"] == True]
    df = df[df["image_id"].notna() & (df["image_id"] != "")]
    df = df.drop_duplicates(subset=["id"])
    logging.info(f"Retained {len(df)} public-domain artworks with images.")

    # Step 3 — Construct deterministic image URLs (zero API calls)
    df["Primary Image"] = df["image_id"].apply(
        lambda iid: f"{IIIF_BASE}/{iid}/full/843,/0/default.jpg"
    )

    # Step 4 — Rename columns for downstream compatibility
    df = df.rename(columns={
        "id": "Object ID",
        "title": "Title",
        "place_of_origin": "Country",
        "classification_title": "Object Name",
        "date_display": "Object Date",
        "medium_display": "Medium",
        "artist_display": "Artist Display Name",
    })

    # Step 5 — Text serialisation
    df["base_string"] = df.apply(generate_base_string, axis=1)
    df["text_serialized"] = df["base_string"]

    # Step 6 — Wikipedia augmentation
    df = await augment_dataframe(df)

    # Step 7 — Save
    logging.info(f"Saving cleaned dataset ({len(df)} rows) to {OUTPUT_PATH}")
    df.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Ingestion pipeline completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
