# Cross-Modal Antiquities Retrieval System

This repository contains a pedagogical implementation of a cross-modal retrieval system utilizing PyTorch, FastAPI, and a React frontend. The application demonstrates asynchronous data ingestion, contrastive representation learning, and high-performance vector search on the Metropolitan Museum of Art Open Access dataset.

## Architecture Highlights

*   **Backend:** FastAPI for asynchronous HTTP handling and WebSocket streaming, enabling non-blocking serving of heavy PyTorch workloads.
*   **Modeling:** A custom PyTorch Lightning model mapping `dinov2-vits14` and `nomic-embed-text-v1.5` embeddings into a unified joint latent space ($d_{joint}=512$) via InfoNCE contrastive loss.
*   **Vector Search:** `faiss-cpu` utilizing `IndexFlatIP` since embeddings are $L_2$ normalized before querying.
*   **Frontend:** Vite/React 18 structured with Tailwind CSS v4, utilizing Zustand for global websocket telemetry mapping on Recharts, and Deck.gl for geospatial plotting (Nominatim/OpenStreetMap coordinates).

## Managing the Execution Environment

This directory strictly uses `uv` for python virtual environments.

### 1. Environment Activation
```bash
uv venv
source .venv/bin/activate
uv sync
```
*Note: Due to our robust dependency hygiene, environments strictly align with pinned architectures.*

### 2. Frontend Initialization
```bash
cd frontend
npm install
npm run dev
```

## Running the Data Engineering Pipeline

To establish the backend vector representations, execute the offline data pipeline. Note that querying Wikimedia extracts and Nominatim for the entire catalog takes considerable elapsed time.

```bash
# Execute sequentially from the project root
python -m backend.pipeline.ingest
python -m backend.pipeline.geocode
python -m backend.pipeline.features
```

## Running the Server API

```bash
uvicorn backend.main:app --reload --port 8000
```