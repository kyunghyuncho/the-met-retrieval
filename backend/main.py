import os
# Fix for OMP: Error #15 on macOS when using faiss + torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, Depends, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import pandas as pd
from pathlib import Path
import logging
from contextlib import asynccontextmanager
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from PIL import Image
import io

from backend.models.lit_model import MetContrastiveModel
from backend.api.persistence import load_state
from backend.api.train import router as train_router

logging.basicConfig(level=logging.INFO)

DATA_DIR = Path("data")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Starting up FastAPI application...")
    
    # Check if files exist
    metadata_path = DATA_DIR / "metadata_index.parquet"
    images_path = DATA_DIR / "images_unprojected.pt"
    texts_path = DATA_DIR / "text_unprojected.pt"
    
    if metadata_path.exists():
        logging.info("Loading metadata index...")
        app.state.metadata_df = pd.read_parquet(metadata_path)
        app.state.metadata_records = app.state.metadata_df.to_dict('records')
        app.state.metadata_dict = {str(r.get("Object ID")): r for r in app.state.metadata_records if r.get("Object ID")}
        # Build a boolean mask aligned with the metadata rows.
        if "has_image" in app.state.metadata_df.columns:
            app.state.has_image_mask = torch.tensor(
                app.state.metadata_df["has_image"].values, dtype=torch.bool
            )
        else:
            # Fallback for older metadata files: assume all items have images.
            logging.warning(
                "'has_image' column missing from metadata; assuming all items "
                "have images.  Re-run features.py to populate this column."
            )
            app.state.has_image_mask = torch.ones(
                len(app.state.metadata_df), dtype=torch.bool
            )
    else:
        logging.warning("Metadata index not found.")
        app.state.metadata_records = []
        app.state.metadata_dict = {}
        app.state.has_image_mask = torch.zeros(0, dtype=torch.bool)
        
    if images_path.exists() and images_path.stat().st_size > 1000:
        logging.info("Loading unprojected images tensor...")
        app.state.images_tensor = torch.load(images_path, weights_only=False)
    else:
        logging.warning("Images tensor not found or empty. Run features.py first.")
        app.state.images_tensor = None
        
    if texts_path.exists() and texts_path.stat().st_size > 1000:
        logging.info("Loading unprojected texts tensor...")
        app.state.texts_tensor = torch.load(texts_path, weights_only=False)
    else:
        logging.warning("Texts tensor not found or empty. Run features.py first.")
        app.state.texts_tensor = None
        
    app.state.index_images = None
    app.state.index_texts = None
    app.state.image_row_ids = []   # list[int] mapping FAISS pos → global row
    app.state.model = None
    
    # Attempt to load persistent state early
    load_state(app.state, MetContrastiveModel)
    
    # Load inference models safely
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    app.state.device = device
    
    try:
        app.state.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        app.state.image_model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
        app.state.tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        app.state.text_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True).to(device).eval()
    except Exception as e:
        logging.error(f"Error loading models for inference: {e}")
        
    yield
    logging.info("Shutting down application...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(train_router)

@app.get("/")
async def root():
    """System health and data status summary."""
    n_total = len(app.state.metadata_records)
    n_with_images = app.state.metadata_df["has_image"].sum() if "has_image" in app.state.metadata_df.columns else 0
    n_with_coords = (app.state.metadata_df["Latitude"] != 0).sum() if "Latitude" in app.state.metadata_df.columns else 0
    
    return {
        "status": "online",
        "data_summary": {
            "total_artifacts": n_total,
            "with_images": int(n_with_images),
            "with_coordinates": int(n_with_coords)
        },
        "docs": "/docs"
    }

class TextSearchQuery(BaseModel):
    query: str
    k: int = 20

@app.post("/api/search/text")
async def search_text(query: TextSearchQuery):
    """Text-query → descriptions (all N items) and/or images (image-having items)."""
    if app.state.index_texts is None or app.state.model is None:
        raise HTTPException(status_code=400, detail="Model not trained or indices not built yet.")

    text_prefix = "search_query: " + query.query

    inputs = app.state.tokenizer(text_prefix, return_tensors="pt").to(app.state.device)
    with torch.no_grad():
        outputs = app.state.text_model(**inputs)
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        text_features = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        device_model = app.state.model.to(app.state.device)
        z_text = torch.nn.functional.normalize(device_model.W_text(text_features), p=2, dim=1).cpu().numpy()

    # index_texts covers all N rows; FAISS position == global metadata row index.
    distances, indices = app.state.index_texts.search(z_text, query.k)

    results = []
    for idx_tuple, dist_tuple in zip(indices, distances):
        for idx, dist in zip(idx_tuple, dist_tuple):
            if idx != -1 and idx < len(app.state.metadata_records):
                record = app.state.metadata_records[idx].copy()
                record['similarity'] = float(dist)
                record['similarity_percentage'] = round(((float(dist) + 1.0) / 2.0) * 100, 2)
                results.append(record)

    return results

@app.post("/api/search/image")
async def search_image(file: UploadFile = File(...), k: int = Form(20)):
    """Image-query → descriptions (all N, via text index) and images (image-only)."""
    if app.state.index_images is None or app.state.model is None:
        raise HTTPException(status_code=400, detail="Model not trained or indices not built yet.")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    inputs = app.state.image_processor(images=image, return_tensors="pt").to(app.state.device)
    with torch.no_grad():
        outputs = app.state.image_model(**inputs)
        img_features = outputs.last_hidden_state[:, 0, :]

        device_model = app.state.model.to(app.state.device)
        z_image = torch.nn.functional.normalize(device_model.W_image(img_features), p=2, dim=1).cpu().numpy()

    # index_images covers only M image-having rows.
    # image_row_ids[faiss_pos] → global metadata row index.
    distances, indices = app.state.index_images.search(z_image, k)
    row_ids = app.state.image_row_ids

    results = []
    for idx_tuple, dist_tuple in zip(indices, distances):
        for faiss_pos, dist in zip(idx_tuple, dist_tuple):
            if faiss_pos != -1 and faiss_pos < len(row_ids):
                global_idx = row_ids[faiss_pos]
                if global_idx < len(app.state.metadata_records):
                    record = app.state.metadata_records[global_idx].copy()
                    record['similarity'] = float(dist)
                    record['similarity_percentage'] = round(((float(dist) + 1.0) / 2.0) * 100, 2)
                    results.append(record)

    return results

@app.get("/api/metadata/locations")
async def get_locations():
    if not app.state.metadata_records:
        return []
    
    # Return minimal payload
    minimal_records = [
        {
            "id": r.get("Object ID"),
            "latitude": r.get("Latitude", 0.0),
            "longitude": r.get("Longitude", 0.0),
            "age": r.get("Object Date", "Unknown") # Simplified for now
        }
        for r in app.state.metadata_records
        if "Latitude" in r and "Longitude" in r and pd.notna(r["Latitude"])
    ]
    return minimal_records

@app.get("/api/metadata/item/{item_id}")
async def get_item(item_id: str):
    if not hasattr(app.state, 'metadata_dict'):
        raise HTTPException(status_code=500, detail="Metadata dictionary not initialized")
    
    item = app.state.metadata_dict.get(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Artifact not found")
        
    return item
