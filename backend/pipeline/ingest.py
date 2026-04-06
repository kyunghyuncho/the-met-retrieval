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

async def fetch_wikidata_summary(session, url, semaphore):
    # Extracts the Wikipedia or Wikidata summary
    # Wikidata URL: http://www.wikidata.org/entity/Q12345
    # For now, if the wiki data URL is present, we could just ping it.
    # But PLAN says: "retrieve the linked English Wikipedia article title, and subsequently query the Wikipedia REST API"
    if not isinstance(url, str) or not url.startswith("http"):
        return None

    entity_id = url.split('/')[-1]
    
    # query wikidata for english wikipedia sitelink title
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
                                # Query wikipedia REST API
                                wiki_res = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
                                async with session.get(wiki_res, timeout=5) as wiki_response:
                                    if wiki_response.status == 200:
                                        wiki_data = await wiki_response.json()
                                        return wiki_data.get("extract")
        except Exception as e:
            # logging.debug(f"Error fetching {url}: {e}")
            pass
    return None

async def augment_dataframe(df):
    logging.info("Starting Wikipedia augmentation...")
    semaphore = asyncio.Semaphore(10)
    
    # Determine the wikidata column name, it's usually "Wikidata URL"
    wiki_col = "Wikidata URL" if "Wikidata URL" in df.columns else None
    
    if not wiki_col:
        logging.warning("No 'Wikidata URL' column found, skipping augmentation.")
        return df

    urls = df[wiki_col].tolist()
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_wikidata_summary(session, url, semaphore) for url in urls]
        results = await asyncio.gather(*tasks)

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
    
    p_img_col = "Primary Image" if "Primary Image" in df.columns else None
    if p_img_col:
        df = df[df[p_img_col].notna() & (df[p_img_col] != '')]
        
    df = df.drop_duplicates(subset=['Object ID'])
    
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
