# Cross-Modal Antiquities Retrieval System

This repository contains a pedagogical implementation of a cross-modal retrieval system utilizing PyTorch, FastAPI, and a React frontend. The application demonstrates asynchronous data ingestion, contrastive representation learning, and high-performance vector search on the **Art Institute of Chicago (AIC) Open Access** dataset.

## Architecture Highlights

*   **Backend:** FastAPI for asynchronous HTTP handling and WebSocket streaming, enabling non-blocking serving of heavy PyTorch workloads.
*   **Modeling:** A custom PyTorch Lightning model mapping `dinov2-small` and `nomic-embed-text-v1.5` embeddings into a unified joint latent space ($d_{joint}=512$) via InfoNCE contrastive loss.
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

The offline pipeline is split into four stages to maximize hardware utilisation.
Image downloading is I/O-bound and runs on CPU; feature extraction is
compute-bound and runs exclusively on GPU (or MPS on Apple Silicon).

```bash
# Stage 1 – AIC metadata ingestion (~8 min, paginated REST API)
uv run backend/pipeline/ingest.py

# Stage 2 – Geocode place-of-origin strings
uv run backend/pipeline/geocode.py

# Stage 3a – Download images to local cache (CPU / bandwidth, run on any machine)
#   --workers  parallel HTTP threads (default: 32)
#   --timeout  per-request timeout in seconds (default: 15)
uv run backend/pipeline/download_images.py --workers 32

# Stage 3b – GPU feature extraction (reads from local cache, zero network I/O)
#   --batch-size  model forward-pass batch size (default: 64)
#   --num-workers DataLoader worker processes (default: 4)
uv run backend/pipeline/features.py --batch-size 64 --num-workers 4
```

Stages 3a and 3b can be run on different machines: copy `data/images/` and
`data/images_manifest.parquet` to your GPU host before running stage 3b.
The manifest records which images were successfully downloaded so the feature
extractor can emit a safe zero-tensor fallback for any missing entries.

## Running the Server API

```bash
uvicorn backend.main:app --reload --port 8000
```