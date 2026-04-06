# PLAN.md: Comprehensive Specification for Cross-Modal Antiquities Retrieval System

## 1. System Architecture and Technology Stack

The system requires a strict separation of concerns to handle the asynchronous nature of deep learning training alongside a synchronous, responsive user interface.

### 1.1 Frontend (Client-Side)
* **Framework:** React 18 with TypeScript.
* **Build Tool:** Vite (optimized for fast Hot Module Replacement during local Apple Silicon development).
* **State Management:** Zustand (preferred over Redux for lightweight, boilerplate-free global state, specifically for managing the active WebSocket connection and training status).
* **Routing:** React Router v6 (Client-side routing for `/dashboard`, `/map`, `/search`).
* **Styling:** Tailwind CSS (utility-first) combined with Radix UI primitives for accessible, unstyled interactive components (sliders for hyperparameters, dropdowns).
* **Data Visualization:** * Training Telemetry: Recharts (optimized for real-time appending of timeseries data).
    * Geospatial: `deck.gl` (WebGL-powered, highly performant for rendering $N > 100,000$ coordinate points simultaneously).

### 1.2 Backend (Server-Side & Compute)
* **Web Framework:** FastAPI (Python 3.11+). Chosen for native `asyncio` support, critical for non-blocking HTTP endpoints while background PyTorch tasks execute.
* **Deep Learning Framework:** PyTorch and PyTorch Lightning.
* **Vector Search Engine:** FAISS (`faiss-cpu` for the FastAPI runtime). The exact index type will be `IndexFlatIP` (Inner Product) since embeddings will be strictly $L_2$ normalized, rendering inner product mathematically equivalent to cosine similarity.

## 2. Phase 1: Offline Data Engineering Pipeline

This phase executes strictly outside the web application lifecycle. It produces the static assets required for the FastAPI server to initialize.

### 2.1 Data Ingestion and Deterministic Serialization
* **Target:** The Metropolitan Museum of Art Open Access CSV.
* **Data Cleaning Sequence:**
    1.  Load via `pandas`.
    2.  Filter: `df = df[(df['Is Public Domain'] == True) & (df['Object Name'].notna())]`.
    3.  Drop duplicates based on the `Object ID` column.
* **REST API Image Harvester:** Asynchronously map over `https://collectionapi.metmuseum.org/public/collection/v1/objects/{ObjectID}`. Because the static CSV strictly lacks the `.jpg` strings, the ingestion pipeline proactively fetches `primaryImageSmall` and filters out instances completely lacking a valid visual component prior to saving.
* **Hybrid Text Serialization Algorithm:**
    * **Step A (Deterministic Base):** Construct a base string for all rows. Format: `"Artifact: {Object Name}. Title: {Title}. Origin: {Culture}. Period: {Object Date}. Medium: {Medium}."` Missing values must be dynamically omitted rather than injecting "NaN" strings.
    * **Step B (Asynchronous Augmentation):** Isolate rows containing a valid `Wikidata URL`. Utilize `aiohttp` with a concurrency limit (e.g., `asyncio.Semaphore(10)`) to query the Wikidata SPARQL endpoint, retrieve the linked English Wikipedia article title, and subsequently query the Wikipedia REST API (`/api/rest_v1/page/summary/{title}`).
    * **Step C (Resolution):** Overwrite the deterministic string with the Wikipedia `extract` where the API response is HTTP 200.

### 2.2 Geospatial Coordinate Resolution (Geocoding)
* Isolate unique strings from the dataset's `Geography`, `City`, `State`, and `Country` columns.
* Construct a geographically hierarchical query string (e.g., `"{City}, {State}, {Country}"`).
* Utilize the Google Maps Geocoding API via a rate-limited Python script.
* **Output:** Append two strictly typed float columns to the dataset: `Latitude` and `Longitude`. Rows failing resolution default to `0.0, 0.0` or a predefined "Unknown" coordinate block.

### 2.3 Feature Extraction and Storage
* **Image Processing (`dinov2-small`):** Define a standard `torchvision.transforms` pipeline (Resize to 256, CenterCrop to 224, Normalize using ImageNet statistics). Process the downloaded images in batches of $B=64$ (or maximum VRAM capacity).
* **Text Processing (`nomic-embed-text-v1.5`):** Tokenize the final serialized text columns with `truncation=True, max_length=512`. Prefix all strings with `search_document:`.
* **Serialization Format:** Save the extracted features as monolithic `.safetensors` or `.pt` files to enable memory-mapping. The files must be named `images_unprojected.pt` ($N \times 384$) and `text_unprojected.pt` ($N \times 768$).
* **Metadata Index:** Save the enriched `pandas` DataFrame (containing Object IDs, URLs, resolved text, and lat/lon) as `metadata_index.parquet` to preserve data types and enable rapid loading in FastAPI.

## 3. Phase 2: Backend Implementation Specifications

### 3.1 Application State and Memory Management
FastAPI must hold the following in application state (`app.state`) upon startup:
1.  The unprojected PyTorch tensors.
2.  The `metadata_index.parquet` file (loaded as a Dictionary or list of Dicts for $O(1)$ lookup).
3.  Two FAISS indices (`index_images`, `index_texts`). These remain empty until a training cycle completes.
4.  A reference to the current projection layers ($W_{text}$, $W_{image}$).

### 3.2 The Dynamic Training Endpoint (`POST /api/train`)
* **Payload Schema (Pydantic):**
    ```python
    class TrainingConfig(BaseModel):
        learning_rate: float = 1e-4
        batch_size: int = 256
        d_joint: int = 512
        max_epochs: int = 50
        temperature_init: float = 0.07
    ```
* **Execution Flow:**
    1.  The endpoint receives the payload and immediately returns a `202 Accepted` response with a unique `session_id`, delegating the training to `fastapi.BackgroundTasks`.
    2.  The PyTorch Lightning `DataModule` segments the tensors (80/10/10 split).
    3.  **Hardware Dispatch:** The backend reads `os.environ.get("DEV_MODE")`. If `1`, `trainer = Trainer(accelerator="mps", devices=1)`. If `0`, `trainer = Trainer(accelerator="cuda", devices=1)`.
* **WebSocket Telemetry (`WS /ws/telemetry/{session_id}`):**
    * A custom PyTorch Lightning Callback hooks into `on_train_batch_end` and `on_validation_epoch_end`.
    * It serializes the metrics into JSON (e.g., `{"epoch": 1, "train_loss": 2.4, "val_loss": 2.1, "val_r_at_5": 0.45}`) and pushes them into an `asyncio.Queue`.
    * The WebSocket endpoint continuously consumes this queue and broadcasts to the connected React client.
* **Post-Training Hook:** Upon completion, the backend applies the newly learned $W_{image}$ and $W_{text}$ to the entire tensor database, $L_2$ normalizes the outputs, and reconstructs the `faiss.IndexFlatIP` instances in memory, enabling immediate inference.

### 3.3 Inference Endpoints
* `POST /api/search/text`:
    * Accepts `{"query": "ancient egyptian pharaoh", "k": 20}`.
    * Prepends `search_query:`, passes through `nomic-embed`, applies $W_{text}$, and normalizes.
    * Executes `index_images.search(query_vector, k)`.
    * Maps the resulting integer indices back to the `metadata_index` and returns the complete JSON records.
* `POST /api/search/image`:
    * Accepts `multipart/form-data` containing an image file.
    * Transforms the image, passes through `dinov2`, applies $W_{image}$, and normalizes.
    * Executes `index_texts.search(image_vector, k)`.

## 4. Phase 3: Frontend Implementation Specifications

### 4.1 Training Configuration View (`/train`)
* **Layout:** Form inputs on the left panel (sliders for LR, Batch Size, $d_{joint}$).
* **Chart Panel:** The right panel mounts a `Recharts` component. The local state maintains an array of epoch data. The WebSocket `onmessage` event listener pushes new data points into this array, causing the chart to smoothly animate the progression of the loss curves.
* **Control Mechanisms:** Include an explicit "Abort Training" button that sends a `DELETE /api/train/{session_id}` request, halting the PyTorch Lightning trainer gracefully via a shared event flag.

### 4.2 Geospatial Exploratory Data Analysis (`/map`)
* **Map Initialization:** Instantiate `DeckGL` with a `Mapbox` basemap.
* **Data Layer:** Render a `ScatterplotLayer`.
    * `data`: Fetched from a dedicated `GET /api/metadata/locations` endpoint (returning only ID, Lat, Lon to minimize payload size).
    * `getPosition`: `d => [d.longitude, d.latitude]`.
    * `getFillColor`: Dynamically mapped based on the object's age (e.g., older objects map to a darker red, newer to a bright yellow).
* **Interaction:** Clicking a scatterplot node triggers a state update, fetching the full artifact details via its ID and sliding in a side-panel containing the high-resolution image and the serialized textual description.

### 4.3 Retrieval View (`/search`)
* **Query Input:** A unified search bar that detects input type. If text is typed, it defaults to Text-to-Image. If a file is dropped into the zone utilizing `react-dropzone`, it switches to Image-to-Text.
* **Debouncing:** Implement a 500ms debounce on the text input to prevent flooding the backend with embedding requests during rapid typing.
* **Result Rendering:** Iterate through the JSON response array. Render images utilizing `loading="lazy"` attributes. Display the mathematically computed similarity score (converted from the inner product range $[-1, 1]$ to a percentage $[0, 100\%]$ for user comprehension).

## 5. Staged Execution Protocol

### 5.1 Local Development Environment (Apple Silicon)
1.  **Data Truncation:** Execute the Phase 1 scripts locally, but modify the initial pandas load: `df = df.head(5000)`. This generates a lightweight $N=5000$ dataset.
2.  **Environment Setup:** Define `export DEV_MODE=1` in the terminal.
3.  **Backend Startup:** Run `uvicorn main:app --reload --port 8000`. Verify PyTorch successfully binds to the `mps` backend in the console logs.
4.  **Frontend Startup:** Run `npm run dev` in the React directory.
5.  **Validation:** Initiate a training cycle from the UI. Ensure the WebSocket telemetry graphs the loss curve without significant latency.

### 5.2 Remote Production Deployment (GPU Server)
1.  **Full Pipeline Execution:** Sync the Python scripts to the remote server. Execute the Phase 1 pipeline over the entire $N \approx 406,000$ dataset utilizing CUDA for maximum throughput.
2.  **System configuration:** Define `export DEV_MODE=0`.
3.  **Reverse Proxy:** Configure Nginx to route external traffic to the FastAPI uvicorn workers (managed via `gunicorn` with `uvicorn.workers.UvicornWorker` class). Ensure Nginx is configured to correctly proxy WebSocket upgrade requests (`Connection: Upgrade`, `Upgrade: websocket`).
4.  **Static Serving:** Build the React application (`npm run build`) and configure Nginx to serve the resulting static files from the `/dist` directory.