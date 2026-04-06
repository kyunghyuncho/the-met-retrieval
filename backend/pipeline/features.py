import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
import logging
from torch.utils.data import Dataset, DataLoader
import requests
import io
from safetensors.numpy import save_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_DIR = Path("data")
INPUT_PATH = DATA_DIR / "met_geocoded.parquet"
OUTPUT_METADATA_PATH = DATA_DIR / "metadata_index.parquet"
OUTPUT_IMAGES_PATH = DATA_DIR / "images_unprojected.pt"
OUTPUT_TEXTS_PATH = DATA_DIR / "text_unprojected.pt"

class MetArtifactDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_url = row.get("Primary Image")
        text = row.get("text_serialized")
        
        image_tensor = torch.zeros((3, 224, 224))
        
        if pd.notna(image_url) and str(image_url).startswith("http"):
            try:
                response = requests.get(image_url, timeout=5)
                if response.status_code == 200:
                    image = Image.open(io.BytesIO(response.content)).convert("RGB")
                    image_tensor = self.transform(image)
            except Exception as e:
                pass # Return zero tensor on failure
        
        return image_tensor, "search_document: " + str(text)

def main():
    if not INPUT_PATH.exists():
        logging.error(f"Input file {INPUT_PATH} not found. Run geocode.py first.")
        return
        
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    logging.info(f"Using device: {device}")
    
    df = pd.read_parquet(INPUT_PATH)
    
    # Initialize models
    logging.info("Loading dinov2-small for images...")
    image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    image_model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
    
    logging.info("Loading nomic-embed-text-v1.5 for text...")
    tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
    # nomic requires trust_remote_code=True
    text_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True).to(device)
    
    dataset = MetArtifactDataset(df)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=0, shuffle=False)
    
    all_image_features = []
    all_text_features = []
    
    logging.info(f"Starting feature extraction for {len(dataset)} items...")
    
    image_model.eval()
    text_model.eval()
    
    with torch.no_grad():
        for i, (images, texts) in enumerate(dataloader):
            images = images.to(device)
            
            # Extract Image Features
            img_outputs = image_model(pixel_values=images)
            img_features = img_outputs.last_hidden_state[:, 0, :] # CLS token
            all_image_features.append(img_features.cpu())
            
            # Extract Text Features
            inputs = tokenizer(list(texts), padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            text_outputs = text_model(**inputs)
            # mean pooling or cls pooling depending on model, nomic usually uses mean pooling over active tokens
            attention_mask = inputs['attention_mask']
            token_embeddings = text_outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            text_features = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            # Layer normalization according to some nomic recommendations, but we'll stick to raw features
            # L2 normalization comes later in the projection layer.
            all_text_features.append(text_features.cpu())
            
            if (i+1) % 10 == 0:
                logging.info(f"Processed batch {i+1} / {len(dataloader)}")
                
    image_tensor_full = torch.cat(all_image_features, dim=0)
    text_tensor_full = torch.cat(all_text_features, dim=0)
    
    logging.info(f"Saving images tensor: {image_tensor_full.shape} to {OUTPUT_IMAGES_PATH}")
    torch.save(image_tensor_full, OUTPUT_IMAGES_PATH)
    
    logging.info(f"Saving texts tensor: {text_tensor_full.shape} to {OUTPUT_TEXTS_PATH}")
    torch.save(text_tensor_full, OUTPUT_TEXTS_PATH)
    
    logging.info(f"Saving final metadata index to {OUTPUT_METADATA_PATH}")
    df.to_parquet(OUTPUT_METADATA_PATH, index=False)

if __name__ == "__main__":
    main()
