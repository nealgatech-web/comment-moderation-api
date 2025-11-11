from fastapi import APIRouter, HTTPException
from .models import AnalyzeRequest, AnalyzeResponse, AnalyzeResult
from .deps import get_redis, get_settings
from .ml.pipeline import ModelManager
import hashlib, json, asyncio

router = APIRouter()
model_manager = ModelManager()

def cache_key(texts):
    j = json.dumps(texts, ensure_ascii=False, sort_keys=True)
    return "analyze:" + hashlib.sha256(j.encode("utf-8")).hexdigest()

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest):
    texts = [t.strip() for t in payload.texts if t and t.strip()]
    if not texts:
        raise HTTPException(status_code=400, detail="No non-empty texts provided.")

    r = get_redis()
    ttl = get_settings()["CACHE_TTL_SECONDS"]
    if r:
        cached = await r.get(cache_key(texts))
        if cached:
            data = json.loads(cached)
            return AnalyzeResponse(**data)

    probs = model_manager.predict_proba(texts)
    results = []
    for text, (p_toxic, p_neutral) in zip(texts, probs):
        toxicity = float(p_toxic)
        confidence = float(max(p_toxic, p_neutral))
        label = "toxic" if toxicity >= 0.5 else "neutral"
        results.append(AnalyzeResult(text=text, toxicity=toxicity, confidence=confidence, label=label))

    resp = AnalyzeResponse(results=results)
    if r:
        await r.setex(cache_key(texts), ttl, json.dumps(resp.model_dump()))
    return resp
