import pandas as pd
import time
from pathlib import Path
from geopy.geocoders import Nominatim, GoogleV3
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_DIR = Path("data")
INPUT_PATH = DATA_DIR / "met_cleaned.parquet"
OUTPUT_PATH = DATA_DIR / "met_geocoded.parquet"

def get_location_string(row):
    parts = []
    if pd.notna(row.get('City')):
        parts.append(str(row['City']))
    if pd.notna(row.get('State')):
        parts.append(str(row['State']))
    if pd.notna(row.get('Country')):
        parts.append(str(row['Country']))
    elif pd.notna(row.get('Geography')):
        parts.append(str(row['Geography']))
        
    return ", ".join(parts) if parts else ""

def geocode_unique_locations(unique_locations):
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        logging.error("GOOGLE_MAPS_API_KEY environment variable is not set. Switching back to slow Nominatim...")
        geolocator = Nominatim(user_agent="met_antiquities_pedagogical_app", timeout=10)
        sleep_time = 1.1
    else:
        logging.info("Using Google Maps Geocoding API for high-speed resolution.")
        geolocator = GoogleV3(api_key=api_key, timeout=10)
        sleep_time = 0.02  # Maximum 50 QPS support

    mapping = {}
    total = len(unique_locations)
    logging.info(f"Found {total} unique locations to geocode.")
    
    for i, loc in enumerate(unique_locations):
        if not loc:
            mapping[loc] = (0.0, 0.0)
            continue
            
        try:
            time.sleep(sleep_time)
            location = geolocator.geocode(loc)
            if location:
                mapping[loc] = (location.latitude, location.longitude)
            else:
                mapping[loc] = (0.0, 0.0)
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            logging.error(f"Error geocoding {loc}: {e}")
            mapping[loc] = (0.0, 0.0)
            
        if (i+1) % 100 == 0:
            logging.info(f"Geocoded {i+1}/{total} locations.")
            
    return mapping

def main():
    if not INPUT_PATH.exists():
        logging.error(f"Input file {INPUT_PATH} not found. Run ingest.py first.")
        return
        
    logging.info(f"Loading {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    
    # Create location string
    df['location_query'] = df.apply(get_location_string, axis=1)
    
    unique_locs = df['location_query'].unique()
    
    # Check if we already have some cached locations
    mapping = geocode_unique_locations(unique_locs)
    
    def apply_lat(loc):
        return mapping.get(loc, (0.0, 0.0))[0]
        
    def apply_lon(loc):
        return mapping.get(loc, (0.0, 0.0))[1]
        
    df['Latitude'] = df['location_query'].apply(apply_lat)
    df['Longitude'] = df['location_query'].apply(apply_lon)
    
    df = df.drop(columns=['location_query'])
    
    logging.info(f"Saving geocoded dataset to {OUTPUT_PATH}")
    df.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Geocoding pipeline completed.")

if __name__ == "__main__":
    main()
