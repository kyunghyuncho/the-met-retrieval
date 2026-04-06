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
    else:
        logging.warning("Metadata index not found.")
        app.state.metadata_records = []
        
    if images_path.exists():
        logging.info("Loading unprojected images tensor...")
        app.state.images_tensor = torch.load(images_path)
    else:
        app.state.images_tensor = None
        
    if texts_path.exists():
        logging.info("Loading unprojected texts tensor...")
        app.state.texts_tensor = torch.load(texts_path)
    else:
        app.state.texts_tensor = None
        
    app.state.index_images = None
    app.state.index_texts = None
    app.state.model = None
    
    # Load inference models safely
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    app.state.device = device
    
    try:
        app.state.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-vits14")
        app.state.image_model = AutoModel.from_pretrained("facebook/dinov2-vits14").to(device).eval()
        app.state.tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
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

class TextSearchQuery(BaseModel):
    query: str
    k: int = 20

@app.post("/api/search/text")
async def search_text(query: TextSearchQuery):
    if app.state.index_images is None or app.state.model is None:
        raise HTTPException(status_code=400, detail="Model not trained or indices not built yet.")
        
    text_prefix = "search_query: " + query.query
    
    inputs = app.state.tokenizer(text_prefix, return_tensors="pt").to(app.state.device)
    with torch.no_grad():
        outputs = app.state.text_model(**inputs)
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        text_features = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Apply projection
        device_model = app.state.model.to(app.state.device)
        z_text = torch.nn.functional.normalize(device_model.W_text(text_features), p=2, dim=1).cpu().numpy()
        
    distances, indices = app.state.index_images.search(z_text, query.k)
    
    results = []
    for idx_tuple, dist_tuple in zip(indices, distances):
        for idx, dist in zip(idx_tuple, dist_tuple):
            if idx != -1 and idx < len(app.state.metadata_records):
                record = app.state.metadata_records[idx].copy()
                record['similarity'] = float(dist)
                # Map Inner-product roughly to percentage. Assuming L2 normalized, IP is cosine sim [-1, 1]
                record['similarity_percentage'] = round(((float(dist) + 1.0) / 2.0) * 100, 2)
                results.append(record)
                
    return results

@app.post("/api/search/image")
async def search_image(file: UploadFile = File(...), k: int = Form(20)):
    if app.state.index_texts is None or app.state.model is None:
        raise HTTPException(status_code=400, detail="Model not trained or indices not built yet.")
        
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    inputs = app.state.image_processor(images=image, return_tensors="pt").to(app.state.device)
    with torch.no_grad():
        outputs = app.state.image_model(**inputs)
        img_features = outputs.last_hidden_state[:, 0, :]
        
        device_model = app.state.model.to(app.state.device)
        z_image = torch.nn.functional.normalize(device_model.W_image(img_features), p=2, dim=1).cpu().numpy()
        
    distances, indices = app.state.index_texts.search(z_image, k)
    
    results = []
    for idx_tuple, dist_tuple in zip(indices, distances):
        for idx, dist in zip(idx_tuple, dist_tuple):
            if idx != -1 and idx < len(app.state.metadata_records):
                record = app.state.metadata_records[idx].copy()
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
