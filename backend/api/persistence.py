import os
import torch
import faiss
import numpy as np
import logging
from pathlib import Path

# Paths relative to project root
MODEL_DIR = Path("data/model")
MODEL_PATH = MODEL_DIR / "met_model.pt"
INDEX_TEXTS_PATH = MODEL_DIR / "index_texts.faiss"
INDEX_IMAGES_PATH = MODEL_DIR / "index_images.faiss"
ROW_IDS_PATH = MODEL_DIR / "image_row_ids.npy"

def save_state(app_state, model):
    """Persists model weights, indices, and mapping to disk."""
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        # 1. Save model weights
        torch.save(model.state_dict(), MODEL_PATH)
        
        # 2. Save FAISS indices
        if hasattr(app_state, 'index_texts') and app_state.index_texts is not None:
            faiss.write_index(app_state.index_texts, str(INDEX_TEXTS_PATH))
            
        if hasattr(app_state, 'index_images') and app_state.index_images is not None:
            faiss.write_index(app_state.index_images, str(INDEX_IMAGES_PATH))
            
        # 3. Save entry mapping
        if hasattr(app_state, 'image_row_ids') and app_state.image_row_ids is not None:
            np.save(str(ROW_IDS_PATH), np.array(app_state.image_row_ids))
            
        logging.info(f"Persistence: State saved to {MODEL_DIR}")
    except Exception as e:
        logging.error(f"Persistence: Error saving state: {e}")

def load_state(app_state, model_class):
    """Loads weights and indices into app_state if they exist."""
    if not MODEL_PATH.exists():
        logging.info("Persistence: No existing model found on disk.")
        return False
        
    try:
        # 1. Load Model
        # Need to know the architecture parameters from app_state or hardcoded
        # For our case, we'll try to reconstruct from state dict
        # We assume model_class is MetContrastiveModel
        
        # First, load the weights to check dims
        state_dict = torch.load(MODEL_PATH, weights_only=True, map_location='cpu')
        
        # Infer d_joint, d_image, d_text from state_dict keys
        # W_image.weight: [d_joint, d_image]
        # W_text.weight: [d_joint, d_text]
        d_joint, d_image = state_dict['W_image.weight'].shape
        _, d_text = state_dict['W_text.weight'].shape
        
        model = model_class(d_image=d_image, d_text=d_text, d_joint=d_joint)
        model.load_state_dict(state_dict)
        app_state.model = model.eval()
        
        # 2. Load FAISS indices
        if INDEX_TEXTS_PATH.exists():
            app_state.index_texts = faiss.read_index(str(INDEX_TEXTS_PATH))
            
        if INDEX_IMAGES_PATH.exists():
            app_state.index_images = faiss.read_index(str(INDEX_IMAGES_PATH))
            
        # 3. Load mapping
        if ROW_IDS_PATH.exists():
            app_state.image_row_ids = np.load(str(ROW_IDS_PATH)).tolist()
            
        logging.info(f"Persistence: Successfully loaded model and indices from {MODEL_DIR}")
        return True
    except Exception as e:
        logging.error(f"Persistence: Failed to load state: {e}")
        return False
