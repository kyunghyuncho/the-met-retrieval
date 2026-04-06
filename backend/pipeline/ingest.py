import os
import asyncio
import aiohttp
import pandas as pd
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MET_CSV_URL = "https://media.githubusercontent.com/media/metmuseum/openaccess/master/MetObjects.csv"
DATA_DIR = Path("data")
MET_CSV_PATH = DATA_DIR / "MetObjects.csv"
OUTPUT_PATH = DATA_DIR / "met_cleaned.parquet"

def download_csv():
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True)
    if not MET_CSV_PATH.exists():
        logging.info(f"Downloading {MET_CSV_URL} to {MET_CSV_PATH}")
        response = requests.get(MET_CSV_URL, stream=True)
        response.raise_for_status()
        with open(MET_CSV_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info("Download completed.")

async def fetch_wikipedia_extract(session, row, semaphore):
    # Try Wikidata URL first if it exists
    url = row.get("Wikidata URL") if "Wikidata URL" in row else None
    if isinstance(url, str) and url.startswith("http"):
        entity_id = url.split('/')[-1]
        wikidata_api = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
        
        async with semaphore:
            try:
                async with session.get(wikidata_api, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        entities = data.get("entities", {})
                        if entity_id in entities:
                            sitelinks = entities[entity_id].get("sitelinks", {})
                            if "enwiki" in sitelinks:
                                title = sitelinks["enwiki"].get("title")
                                if title:
                                    wiki_res = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
                                    async with session.get(wiki_res, timeout=5) as wiki_response:
                                        if wiki_response.status == 200:
                                            wiki_data = await wiki_response.json()
                                            return wiki_data.get("extract")
            except Exception:
                pass

    # Fallback to programmatic search
    title = str(row.get('Title', ''))
    if len(title) > 8 and title.lower() != "nan":
        culture = str(row.get('Culture', ''))
        query = f"{title}"
        if culture and culture.lower() != "nan":
            query += f" {culture}"
        
        async with semaphore:
            try:
                search_url = f"https://en.wikipedia.org/w/rest.php/v1/search/page?q={requests.utils.quote(query)}&limit=1"
                async with session.get(search_url, timeout=5) as search_res:
                    if search_res.status == 200:
                        search_data = await search_res.json()
                        pages = search_data.get("pages", [])
                        if pages:
                            first_page = pages[0]
                            page_title = first_page.get("title")
                            
                            # Simple fuzzy matched check
                            if title.lower()[:10] in page_title.lower() or page_title.lower()[:10] in title.lower():
                                summary_res = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(page_title)}"
                                async with session.get(summary_res, timeout=5) as wiki_response:
                                    if wiki_response.status == 200:
                                        wiki_data = await wiki_response.json()
                                        return wiki_data.get("extract")
            except Exception:
                pass
                
    return None


async def augment_dataframe(df):
    logging.info("Starting Wikipedia augmentation (URL parsing + programmatic fallback search)...")
    semaphore = asyncio.Semaphore(15)
    
    rows = df.to_dict('records')
    results = []
    
    async with aiohttp.ClientSession() as session:
        chunk_size = 5000
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i:i+chunk_size]
            tasks = [fetch_wikipedia_extract(session, row, semaphore) for row in chunk]
            res = await asyncio.gather(*tasks)
            results.extend(res)
            logging.info(f"Augmented {min(i+chunk_size, len(rows))} / {len(rows)} records...")

    df['Wikipedia Extract'] = results
    
    # Overwrite the base string if extract is not none
    def resolve_text(row):
        if pd.notna(row.get('Wikipedia Extract')):
            return row['Wikipedia Extract']
        return row['base_string']

    df['text_serialized'] = df.apply(resolve_text, axis=1)
    
    df = df.drop(columns=['Wikipedia Extract'])
    logging.info("Wikipedia augmentation completed.")
    return df

async def fetch_image_url(session, object_id, semaphore, max_retries=3):
    """Fetch the primary image URL for a single Met object, with retry and pacing."""
    for attempt in range(max_retries):
        async with semaphore:
            try:
                api_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
                timeout = aiohttp.ClientTimeout(total=30)
                async with session.get(api_url, timeout=timeout) as res:
                    if res.status == 200:
                        data = await res.json()
                        url = data.get('primaryImageSmall') or data.get('primaryImage')
                        await asyncio.sleep(0.05)  # gentle pacing
                        return url
                    elif res.status == 429:
                        wait = 5 * (attempt + 1)
                        logging.warning(f"Rate limited (429) on ID={object_id}, waiting {wait}s...")
                        await asyncio.sleep(wait)
                        continue
                    elif res.status in (404, 400, 403):
                        # Permanent failures — do not retry
                        return None
                    else:
                        # 5xx or other transient — retry
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
            except asyncio.TimeoutError:
                await asyncio.sleep(2 * (attempt + 1))
                continue
            except aiohttp.ClientError:
                await asyncio.sleep(2 * (attempt + 1))
                continue
            except Exception as e:
                logging.warning(f"ID={object_id}: unexpected error {type(e).__name__}: {e}")
                break
    return None

CHECKPOINT_PATH = DATA_DIR / "image_urls_checkpoint.json"

async def augment_image_urls(df):
    logging.info("Starting Met API Image URL harvesting...")

    object_ids = df['Object ID'].tolist()

    # --- Resume from checkpoint if available ---
    import json
    cached_results = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, 'r') as f:
            cached_results = json.load(f)
        logging.info(f"Resuming from checkpoint: {len(cached_results)} cached entries found.")

    # Identify which IDs still need fetching
    ids_to_fetch = [oid for oid in object_ids if str(oid) not in cached_results]
    logging.info(f"Total: {len(object_ids)}, cached: {len(cached_results)}, remaining: {len(ids_to_fetch)}")

    # --- Fetch in small batches with conservative rate limiting ---
    semaphore = asyncio.Semaphore(5)
    connector = aiohttp.TCPConnector(limit=5, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        chunk_size = 500
        for i in range(0, len(ids_to_fetch), chunk_size):
            chunk = ids_to_fetch[i:i+chunk_size]
            tasks = [fetch_image_url(session, oid, semaphore) for oid in chunk]
            res = await asyncio.gather(*tasks)

            for oid, url in zip(chunk, res):
                cached_results[str(oid)] = url

            resolved = sum(1 for r in res if r is not None)
            failed = sum(1 for r in res if r is None)
            total_done = len(cached_results)
            logging.info(
                f"Batch {i+len(chunk)}/{len(ids_to_fetch)}: "
                f"{resolved} resolved, {failed} failed "
                f"(total cached: {total_done})"
            )

            # Checkpoint to disk every batch
            with open(CHECKPOINT_PATH, 'w') as f:
                json.dump(cached_results, f)

            # If an entire batch fails, the API might be throttling — pause
            if resolved == 0 and failed > 0:
                logging.warning("Entire batch failed. Pausing 30s before continuing...")
                await asyncio.sleep(30)

    # --- Reconstruct the Primary Image column from cache ---
    df['Primary Image'] = [cached_results.get(str(oid)) for oid in object_ids]

    initial_len = len(df)
    df = df.dropna(subset=['Primary Image'])
    df = df[df['Primary Image'] != ""]
    logging.info(
        f"Image harvesting complete. "
        f"Retained {len(df)} / {initial_len} artifacts with valid images."
    )
    return df


def generate_base_string(row):
    # Format: "Artifact: {Object Name}. Title: {Title}. Origin: {Culture}. Period: {Object Date}. Medium: {Medium}."
    parts = []
    if pd.notna(row.get('Object Name')):
        parts.append(f"Artifact: {row['Object Name']}")
    if pd.notna(row.get('Title')):
        parts.append(f"Title: {row['Title']}")
    if pd.notna(row.get('Culture')):
        parts.append(f"Origin: {row['Culture']}")
    if pd.notna(row.get('Object Date')):
        parts.append(f"Period: {row['Object Date']}")
    if pd.notna(row.get('Medium')):
        parts.append(f"Medium: {row['Medium']}")
    
    return ". ".join(parts) + "." if parts else ""

async def main():
    download_csv()
    logging.info("Loading CSV into Pandas...")
    df = pd.read_csv(MET_CSV_PATH, low_memory=False)
    
    # Data Cleaning Sequence
    bool_col = "Is Public Domain" if "Is Public Domain" in df.columns else "Is Public Domain"
    
    df = df[(df[bool_col] == True) & (df['Object Name'].notna())]
    
    df = df.drop_duplicates(subset=['Object ID'])
    
    # Asynchronous Image Harvesting via Met API
    df = await augment_image_urls(df)
    
    logging.info(f"Loaded {len(df)} public domain artifacts with images.")
    
    df['base_string'] = df.apply(generate_base_string, axis=1)
    df['text_serialized'] = df['base_string']
    
    # Asynchronous augmentation
    df = await augment_dataframe(df)
    
    logging.info(f"Saving cleaned dataset to {OUTPUT_PATH}")
    df.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Ingestion pipeline completed successfully.")

if __name__ == "__main__":
    asyncio.run(main())
