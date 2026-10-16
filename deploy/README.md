# Deployment — 6-module architecture (paper Section IX)

This deploys the paper's minimal-viable architecture plus the existing
product, wired together by Docker Compose:

```
docker-compose.yml
├── embedding-service   # FastAPI /encode — multilingual sentence encoder
├── faiss-service       # FastAPI /search — per-language FAISS indexes
├── reranker-service    # FastAPI /rerank — cross-encoder second stage
├── orchestrator        # FastAPI /ask — detect→encode→search→rerank→generate→score
├── api-gateway         # Flask backend (gunicorn) — public REST API
└── frontend            # nginx — built React SPA, proxies /api
```

## Quick start

```bash
cp .env.example .env   # set SECRET_KEY and JWT_SECRET_KEY (>= 32 chars);
                       # optional: GROQ_API_KEY, ADMIN_TOKEN
docker compose up --build -d
```

Compose refuses to start without `SECRET_KEY`/`JWT_SECRET_KEY`, and the
backend rejects placeholder or short values in production.

- UI:      http://localhost:3000
- API:     http://localhost:8080/api/health
- QA flow: POST http://localhost:3000/api/v1/qa/ask   (needs GROQ_API_KEY)
- QA flow: POST http://localhost:8080/api/v1/qa/ask   (needs GROQ_API_KEY)

First build downloads torch/transformers/sentence-transformers (~3-5 GB);
the first request on api-gateway cold-loads NLLB (~15-30 s).

## The paper's /ask pipeline (orchestrator)

```
POST http://localhost:3000/api/v1/qa/ask
{"question": "What does the bank offer for students?", "top_k": 5, "use_reranker": false}
```

1. Language detection   — script-based detector (in-process)
2. /encode              — embedding-service
3. /search              — faiss-service (per-language index, top-K)
4. /rerank              — reranker-service (optional, cross-encoder)
5. Generate             — Llama-3.1-8B via Groq, grounded prompt
6. Confidence           — model-free consistency heuristic + degenerate-output
                          flags; low confidence is surfaced, never blocks

## Without GROQ_API_KEY

Translation, auth, history, TM, retrieval and the evaluation suite all work.
`/api/v1/qa/ask` returns 503 with instructions; `/api/v1/qa/status` reports
`enabled: false`.

## Health checks

Every service exposes `/health` (api-gateway: `/api/health`). Compose uses
them for `depends_on` ordering — the orchestrator only starts after the
three model services report healthy.

## Configuration

| Env var (host .env)   | Purpose                                   | Default            |
| --------------------- | ----------------------------------------- | ------------------ |
| `GROQ_API_KEY`        | Enables generative QA (orchestrator)      | —                  |
| `GROQ_MODEL`          | LLM for answering                         | llama-3.1-8b-instant |
| `SECRET_KEY`          | Flask secret (production)                 | required           |
| `JWT_SECRET_KEY`      | JWT signing secret                        | required           |
| `ADMIN_TOKEN`         | `X-Admin-Token` for /evaluate/run, /tm/*  | empty = disabled   |
| `NLLB_NUM_BEAMS`      | Beam size for the translator              | 2                  |

## Local (non-Docker) equivalents

- The Flask backend alone: `cd backend && python wsgi.py`
- The orchestrator's three microservices run in-process inside the backend
  (RAG pipeline), so the standalone API is a drop-in replacement.

## Extending

- Swap `EMBEDDING_MODEL` for a fine-tuned encoder (see
  `backend/scripts/finetune_embedder.py`) and rebuild the FAISS indexes with
  `backend/scripts/build_faiss_index.py --encoder ...`. Indexes are
  otherwise built on first use into the `rag-indexes` volume.
- Tune beam size per service via `NLLB_NUM_BEAMS` (latency vs quality).
