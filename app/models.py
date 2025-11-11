from pydantic import BaseModel, Field
from typing import List, Optional, Literal

Label = Literal["toxic", "neutral"]

class AnalyzeRequest(BaseModel):
    texts: List[str] = Field(..., description="List of messages to score")

class AnalyzeResult(BaseModel):
    text: str
    toxicity: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    label: Label

class AnalyzeResponse(BaseModel):
    results: List[AnalyzeResult]

class TrainSample(BaseModel):
    text: str
    label: Label

class TrainRequest(BaseModel):
    samples: List[TrainSample]

class TrainResponse(BaseModel):
    status: str
    trained_on: int
    metrics: dict
