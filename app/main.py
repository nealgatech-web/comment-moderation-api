from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from . import router_analyze, router_train

app = FastAPI(
    title="Comment Moderation API",
    version="0.1.0",
    description="FastAPI service for real-time toxicity scoring and lightweight model training."
)

# CORS (open by default; tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(router_analyze.router)
app.include_router(router_train.router)

@app.get("/healthz")
async def healthz():
    return {"ok": True}
