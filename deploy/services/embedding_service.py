"""
embedding-service — the paper's first deployment module.

Exposes /encode for the multilingual sentence encoder used by the RAG
pipeline. The model loads lazily on first request so the container can
start before the (large) weights are touched.

    POST /encode  {"texts": ["...", ...]}  ->  {"vectors": [[...]], "dim": 384}
    GET  /health                          ->  {"status": "ok"}
"""

import logging
import os

from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title='embedding-service', version='1.0.0')

_encoder = None
ENCODER_MODEL = os.getenv(
    'EMBEDDING_MODEL',
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
)


class EncodeRequest(BaseModel):
    texts: list[str]


class EncodeResponse(BaseModel):
    vectors: list
    dim: int
    model: str


def get_encoder():
    """Lazy singleton encoder (first call downloads/loads the weights)."""
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f'Loading encoder {ENCODER_MODEL}...')
        _encoder = SentenceTransformer(ENCODER_MODEL)
        logger.info('Encoder ready')
    return _encoder


@app.get('/health')
def health():
    return {'status': 'ok', 'service': 'embedding-service'}


@app.post('/encode', response_model=EncodeResponse)
def encode(req: EncodeRequest):
    texts = [t for t in req.texts if t.strip()]
    if not texts:
        return EncodeResponse(vectors=[], dim=0, model=ENCODER_MODEL)
    vecs = get_encoder().encode(texts, convert_to_numpy=True)
    return EncodeResponse(
        vectors=vecs.tolist(),
        dim=int(vecs.shape[1]),
        model=ENCODER_MODEL,
    )
