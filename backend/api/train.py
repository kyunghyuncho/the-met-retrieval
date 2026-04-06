import os
import uuid
import asyncio
from fastapi import APIRouter, BackgroundTasks, WebSocket, WebSocketDisconnect, Request, HTTPException
from pydantic import BaseModel
import lightning.pytorch as pl
import torch
import faiss
import numpy as np

from backend.models.lit_model import MetContrastiveModel, MetDataModule, TelemetryCallback

router = APIRouter()

# Global state for train sessions
active_sessions = {}

class TrainingConfig(BaseModel):
    learning_rate: float = 1e-4
    batch_size: int = 256
    d_joint: int = 512
    max_epochs: int = 50
    temperature_init: float = 0.07

def build_faiss_indices(app_state, model, device):
    images_tensor = app_state.images_tensor.to(device)
    texts_tensor = app_state.texts_tensor.to(device)
    
    with torch.no_grad():
        z_image_proj = torch.nn.functional.normalize(model.W_image(images_tensor), p=2, dim=1).cpu().numpy()
        z_text_proj = torch.nn.functional.normalize(model.W_text(texts_tensor), p=2, dim=1).cpu().numpy()
        
    d = z_image_proj.shape[1]
    
    index_images = faiss.IndexFlatIP(d)
    index_images.add(z_image_proj)
    
    index_texts = faiss.IndexFlatIP(d)
    index_texts.add(z_text_proj)
    
    app_state.index_images = index_images
    app_state.index_texts = index_texts
    app_state.model = model.cpu() # ensure the model state is kept on CPU for inference routing usually or whatever inference uses

def run_training_loop(session_id: str, config: TrainingConfig, app_state, queue: asyncio.Queue):
    dev_mode = int(os.environ.get("DEV_MODE", "1"))
    accelerator = "mps" if dev_mode == 1 and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MetContrastiveModel(
        d_image=app_state.images_tensor.shape[1],
        d_text=app_state.texts_tensor.shape[1],
        d_joint=config.d_joint,
        learning_rate=config.learning_rate,
        temperature_init=config.temperature_init
    )
    
    datamodule = MetDataModule(
        app_state.images_tensor, 
        app_state.texts_tensor, 
        batch_size=config.batch_size
    )
    
    telemetry = TelemetryCallback(queue)
    
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=config.max_epochs,
        callbacks=[telemetry],
        enable_checkpointing=False,
        logger=False
    )
    
    active_sessions[session_id]['trainer'] = trainer
    
    try:
        trainer.fit(model, datamodule=datamodule)
        queue.put_nowait('{"type": "status", "status": "completed"}')
    except Exception as e:
        queue.put_nowait(f'{{"type": "error", "message": "{str(e)}"}}')
        
    # Rebuild indices
    build_faiss_indices(app_state, model, 'cpu') # Use CPU for building generic indices or accelerator if memory permits
    
    if session_id in active_sessions:
        del active_sessions[session_id]

@router.post("/api/train", status_code=202)
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks, request: Request):
    if not hasattr(request.app.state, 'images_tensor') or request.app.state.images_tensor is None:
        raise HTTPException(status_code=400, detail="Data tensors not loaded. Run ingest/features pipeline first.")
        
    session_id = str(uuid.uuid4())
    queue = asyncio.Queue()
    active_sessions[session_id] = {
        'queue': queue,
        'trainer': None
    }
    
    background_tasks.add_task(run_training_loop, session_id, config, request.app.state, queue)
    
    return {"session_id": session_id, "status": "accepted"}

@router.delete("/api/train/{session_id}")
async def abort_training(session_id: str):
    if session_id in active_sessions:
        trainer = active_sessions[session_id].get('trainer')
        if trainer:
            trainer.should_stop = True
        return {"status": "aborted"}
    raise HTTPException(status_code=404, detail="Session not found")

@router.websocket("/ws/telemetry/{session_id}")
async def telemetry_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    if session_id not in active_sessions:
        await websocket.send_text('{"type": "error", "message": "Session not found"}')
        await websocket.close()
        return
        
    queue = active_sessions[session_id]['queue']
    
    try:
        while True:
            msg = await queue.get()
            await websocket.send_text(msg)
            if "status" in msg and "completed" in msg:
                break
            if "error" in msg:
                break
    except WebSocketDisconnect:
        pass
    finally:
        pass # Optional cleanup
