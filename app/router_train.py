from fastapi import APIRouter, HTTPException
from .models import TrainRequest, TrainResponse
from .ml.pipeline import ModelManager

router = APIRouter()
model_manager = ModelManager()

@router.post("/train", response_model=TrainResponse)
async def train(payload: TrainRequest):
    samples = [(s.text.strip(), s.label) for s in payload.samples if s.text and s.text.strip()]
    if not samples:
        raise HTTPException(status_code=400, detail="No valid samples.")
    metrics = model_manager.train(samples)
    return TrainResponse(status="ok", trained_on=len(samples), metrics=metrics)
