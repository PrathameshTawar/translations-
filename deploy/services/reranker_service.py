"""
reranker-service — the paper's third deployment module.

Serves the cross-encoder reranker (ms-marco-MiniLM-L-6-v2 via the
existing app.services.reranker wrapper). Degrades gracefully when the
model cannot be loaded (returns available=false, input order unchanged).

    POST /rerank  {"query": "...", "candidates": [dict|str], "top_k": 3}
                  -> {"reranked": [...], "scores": [...], "available": bool}
    GET  /health
"""

import logging

from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title='reranker-service', version='1.0.0')

_reranker = None


class RerankRequest(BaseModel):
    query: str
    candidates: list  # list of dicts (term objects) or plain strings
    top_k: int = 3


def get_reranker():
    global _reranker
    if _reranker is None:
        from app.services.reranker import CrossEncoderReranker
        _reranker = CrossEncoderReranker()
    return _reranker


@app.get('/health')
def health():
    return {'status': 'ok', 'service': 'reranker-service'}


@app.post('/rerank')
def rerank(req: RerankRequest):
    reranker = get_reranker()
    if not req.candidates:
        return {'reranked': [], 'scores': [], 'available': reranker.available}
    try:
        out = reranker.rerank(req.query, req.candidates, top_k=req.top_k)
        return {
            'reranked': out.get('reranked', req.candidates),
            'scores': out.get('scores', []),
            'available': out.get('available', False),
            'model': reranker.model_name,
        }
    except Exception as e:
        logger.exception('Rerank failed; returning input order')
        return {
            'reranked': req.candidates[:req.top_k],
            'scores': [],
            'available': False,
            'error': str(e),
        }
