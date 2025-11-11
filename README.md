# Comment Moderation API

A lightweight, developer‑friendly REST service for real‑time toxicity scoring, built with FastAPI.  
You can deploy it as‑is, fine‑tune your own moderation model, and plug it into Discord, Telegram, or in‑game chat.

## Highlights

- `POST /analyze` → returns `toxicity` (0–1) and `confidence` for one or many messages.
- `POST /train` → fine‑tune a simple TF‑IDF + Linear model on your labeled data.
- Optional Redis caching and IP‑based rate limiting.
- Drop‑in integration examples for **Discord**, **Telegram**, and **game chats**.
- Dockerfile + GitHub Actions to build and test.

---

## Quick Start

### 1) Setup (Python 3.10+)

```bash
pip3 install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open http://127.0.0.1:8000/docs for the interactive API.

### 2) Analyze Text

```bash
curl -X POST http://127.0.0.1:8000/analyze   -H "Content-Type: application/json"   -d '{"texts": ["I hate you", "Have a great day!"]}'
```

Response:

```json
{
  "results": [
    {"text":"I hate you","toxicity":0.93,"confidence":0.93,"label":"toxic"},
    {"text":"Have a great day!","toxicity":0.03,"confidence":0.97,"label":"neutral"}
  ]
}
```

> Out of the box, the service falls back to a **heuristic baseline** (regex + token stats).  
> Train a small supervised model via `/train` for better quality on your domain.

### 3) Optional: Redis Cache + Rate Limiting

Create an `.env` file (or export env vars):

```bash
REDIS_URL=redis://localhost:6379/0
RATE_LIMIT=60/minute    # or 100/hour etc.
CACHE_TTL_SECONDS=600
```

Run a local Redis (example via Docker):

```bash
docker run -p 6379:6379 redis:7
```

### 4) Train Your Model

Send labeled data to `/train`. Labels are strings: `"toxic"` or `"neutral"` (you can extend this).

```bash
curl -X POST http://127.0.0.1:8000/train   -H "Content-Type: application/json"   -d '{
    "samples": [
      {"text":"you are stupid","label":"toxic"},
      {"text":"let’s play tonight","label":"neutral"},
      {"text":"go to hell","label":"toxic"},
      {"text":"thanks for the help","label":"neutral"}
    ]
  }'
```

Response includes basic metrics and saves a model to `models/model.joblib`:

```json
{
  "status": "ok",
  "trained_on": 4,
  "metrics": {"accuracy": 0.75, "f1_macro": 0.73}
}
```

Restart is **not** needed—models hot‑reload automatically when saved.

---

## Endpoints

### `POST /analyze`
Request:
```json
{ "texts": ["string", "..."] }
```
Response:
```json
{
  "results": [
    {"text":"...","toxicity":0.42,"confidence":0.84,"label":"neutral"}
  ]
}
```

### `POST /train`
Request:
```json
{
  "samples": [
    {"text":"...", "label":"toxic|neutral"}
  ]
}
```
Response:
```json
{ "status":"ok", "trained_on": N, "metrics": { "accuracy": ..., "f1_macro": ... } }
```

---

## Integrations

- **Discord:** `examples/discord_bot.py`
- **Telegram:** `examples/telegram_bot.py`
- **Game Chat / generic:** `examples/game_chat.py`

Each example reads `API_BASE` from env (default `http://127.0.0.1:8000`).

---

## Docker

```bash
docker build -t comment-moderation-api .
docker run -p 8000:8000 --env-file .env comment-moderation-api
```

---

## Project Layout

```
app/
  main.py            # FastAPI app + middleware + routing
  models.py          # Pydantic schemas
  deps.py            # Redis / rate limit utilities
  router_analyze.py  # /analyze endpoint
  router_train.py    # /train endpoint
  ml/
    pipeline.py      # ModelManager: load/save/predict; heuristic fallback
examples/
  discord_bot.py
  telegram_bot.py
  game_chat.py
models/              # persisted models (gitignored)
tests/
  test_analyze.py
```

---

## Notes

- This project intentionally keeps training simple (TF‑IDF + Linear model). You can swap in your favorite deep model—just expose a `predict_proba(texts)` in `ModelManager`.
- Safety & fairness: always test models on your real data slices and consider human‑in‑the‑loop review for edge cases.
