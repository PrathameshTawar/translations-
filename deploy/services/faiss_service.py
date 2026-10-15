"""
faiss-service — the paper's second deployment module.

Serves the per-language FAISS indexes ({lang}_index.faiss + {lang}_mapping.json).
The orchestrator encodes the query with embedding-service, then POSTs the
vector here for ANN search; /search-text is a convenience that encodes
in-process when the embedding service is unreachable.

Index files resolve relative to INDEX_DIR (default: /data/indexes, a named
volume in docker-compose). Missing indexes are built on first use from the
glossaries baked into the image (backend/data/terminology).

    POST /search       {"lang": "hindi", "vector": [...], "top_k": 5}
                       -> {"terms": [{term, translation, context, ...}], "backend": "faiss"}
    POST /search-text  {"lang": "hindi", "query": "...", "top_k": 5}  (encodes locally)
    GET  /health
"""

import json
import logging
import os

from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title='faiss-service', version='1.0.0')

INDEX_DIR = os.getenv('INDEX_DIR', '/data/indexes')

_indices = {}
_mappings = {}
_encoder = None


class SearchRequest(BaseModel):
    lang: str
    vector: list[float] | None = None
    query: str | None = None
    top_k: int = 5


def _load(lang: str):
    """Lazily load the per-language index + mapping (built on demand)."""
    if lang in _indices:
        return True
    from app.services.term_index import load_or_build
    index, mapping = load_or_build(lang, _local_encoder(), index_dir=INDEX_DIR)
    if index is None:
        logger.warning(f'No glossary/index for lang={lang} in {INDEX_DIR}')
        return False
    _indices[lang] = index
    _mappings[lang] = mapping
    logger.info(f'Loaded index for lang={lang} '
                f'({index.ntotal} vectors, dim {index.d})')
    return True


def _local_encoder():
    """Lazy in-process encoder for the /search-text convenience path."""
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer(
            os.getenv('EMBEDDING_MODEL',
                      'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'))
    return _encoder


@app.get('/health')
def health():
    return {'status': 'ok', 'service': 'faiss-service', 'index_dir': INDEX_DIR}


@app.post('/search')
def search(req: SearchRequest):
    import faiss
    import numpy as np
    if not _load(req.lang):
        return {'terms': [], 'backend': 'no-index'}

    if req.vector is None and req.query:
        req.vector = _local_encoder().encode([req.query])[0].tolist()
    if not req.vector:
        return {'terms': [], 'backend': 'faiss', 'error': 'vector or query required'}

    vec = np.asarray(req.vector, dtype='float32').reshape(1, -1)
    if vec.shape[1] != _indices[req.lang].d:
        return {'terms': [], 'backend': 'faiss',
                'error': f'vector dim {vec.shape[1]} != index dim {_indices[req.lang].d}'}

    _, idxs = _indices[req.lang].search(vec, req.top_k)
    mapping = _mappings.get(req.lang, [])
    terms = []
    for idx in idxs[0]:
        if idx != -1 and idx < len(mapping):
            terms.append(mapping[idx])
    return {'terms': terms, 'backend': 'faiss', 'num_results': len(terms)}


@app.post('/search-text')
def search_text(req: SearchRequest):
    """Convenience: encode the query locally, then search."""
    if not req.query:
        return {'terms': [], 'backend': 'no-query'}
    import numpy as np
    vec = _local_encoder().encode([req.query])[0].tolist()
    return search(SearchRequest(lang=req.lang, vector=vec, top_k=req.top_k))
