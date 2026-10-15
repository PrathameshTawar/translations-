"""
orchestrator — the paper's langchain-orchestrator module.

Implements the /ask flow by wiring the three model microservices over HTTP:

    detect language  ->  embedding-service /encode  ->  faiss-service /search
    ->  reranker-service /rerank (optional)  ->  generate (Groq Llama-3.1-8B)
    ->  consistency score (model-free heuristic)

Service URLs come from EMBEDDING_SERVICE_URL / FAISS_SERVICE_URL /
RERANKER_SERVICE_URL (docker-compose network names by default). The HTTP
client is stdlib urllib — no extra dependency.

    POST /ask     {"question": "...", "lang"?, "top_k"?, "use_reranker"?}
    GET  /status  {"groq_configured": bool, "services": {...}}
    GET  /health
"""

import json
import logging
import os
import time
import urllib.error
import urllib.request

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title='orchestrator', version='1.0.0')

EMBEDDING_URL = os.getenv('EMBEDDING_SERVICE_URL', 'http://embedding-service:8000')
FAISS_URL = os.getenv('FAISS_SERVICE_URL', 'http://faiss-service:8000')
RERANKER_URL = os.getenv('RERANKER_SERVICE_URL', 'http://reranker-service:8000')
GROQ_MODEL = os.getenv('GROQ_MODEL') or 'llama-3.1-8b-instant'
HTTP_TIMEOUT = 30

_LANG_CODES = {
    'english', 'hindi', 'marathi', 'bengali', 'gujarati',
    'tamil', 'telugu', 'kannada',
}


class AskRequest(BaseModel):
    question: str
    lang: str | None = None
    top_k: int = 5
    use_reranker: bool = False


class AskResponse(BaseModel):
    answer: str
    confidence: float
    low_confidence: bool
    language: str
    sources: list
    context_used: bool
    model: str
    timings: dict


def _post(url, payload):
    """Small urllib POST helper returning the parsed JSON.

    Raises HTTPException(503) when the upstream microservice is unreachable
    so a missing/inflating service yields a clean response instead of 500.
    """
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        raise HTTPException(status_code=502, detail=f'upstream {url} returned {e.code}')
    except urllib.error.URLError as e:
        raise HTTPException(status_code=503, detail=f'upstream {url} unreachable: {e.reason}')


def _detect_lang(question):
    from app.services.language_detection import LanguageDetector
    try:
        det = LanguageDetector().detect_with_scores(question)
        lang = (det.get('detected_language') or 'english').lower()
        return lang if lang in _LANG_CODES else 'english'
    except Exception as e:
        logger.warning(f'Language detection failed: {e}')
        return 'english'


def _generate(question, context_text, lang):
    """Generate a grounded answer with Groq Llama-3.1-8B."""
    from app.services.generative_qa import _call_groq, _lang_native_name
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail='Generation is not configured: set GROQ_API_KEY to enable '
                   'Llama-3.1-8B answering.',
        )
    prompt = (
        f'Context:\n{context_text}\n\n'
        f'Question: {question}\n\n'
        f'Answer in {_lang_native_name(lang)}. Only answer using the context '
        f'above; if the information is not present, say exactly: '
        f'"The context does not contain this information."'
    )
    return _call_groq(prompt, api_key, model=GROQ_MODEL, timeout=HTTP_TIMEOUT)


@app.get('/health')
def health():
    return {'status': 'ok', 'service': 'orchestrator'}


@app.get('/status')
def status():
    return {
        'groq_configured': bool(os.getenv('GROQ_API_KEY')),
        'services': {
            'embedding': EMBEDDING_URL,
            'faiss': FAISS_URL,
            'reranker': RERANKER_URL,
        },
    }


@app.post('/ask', response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail='question is required')
    if req.top_k < 1 or req.top_k > 10:
        raise HTTPException(status_code=400, detail='top_k must be 1..10')
    lang = (req.lang or '').lower() or None
    if lang is not None and lang not in _LANG_CODES:
        raise HTTPException(status_code=400, detail=f'unsupported lang: {lang}')

    timings = {}

    # 1. language detection (in-process, script-based)
    t0 = time.time()
    lang = lang or _detect_lang(req.question)
    timings['detect_ms'] = round((time.time() - t0) * 1000, 2)

    # 2. encode
    t0 = time.time()
    enc = _post(f'{EMBEDDING_URL}/encode', {'texts': [req.question]})
    vector = enc['vectors'][0] if enc.get('vectors') else []
    timings['encode_ms'] = round((time.time() - t0) * 1000, 2)

    # 3. search (FAISS per-language index)
    t0 = time.time()
    search = _post(f'{FAISS_URL}/search',
                   {'lang': lang, 'vector': vector, 'top_k': req.top_k})
    terms = search.get('terms', [])
    timings['search_ms'] = round((time.time() - t0) * 1000, 2)

    # 4. rerank (optional second stage)
    reranked = False
    if req.use_reranker and terms:
        try:
            t0 = time.time()
            rr = _post(f'{RERANKER_URL}/rerank',
                       {'query': req.question, 'candidates': terms,
                        'top_k': req.top_k})
            terms = rr.get('reranked', terms)
            reranked = bool(rr.get('available'))
            timings['rerank_ms'] = round((time.time() - t0) * 1000, 2)
        except Exception as e:
            logger.warning(f'Reranker call failed (continuing): {e}')

    # 5. context + sources
    context_lines = [
        f"{t.get('term', '')} = {t.get('translation', '')}"
        + (f" ({t.get('context', '')})" if t.get('context') else '')
        for t in terms
    ]
    context_text = '\n'.join(context_lines)
    sources = [
        {
            'term': t.get('term'),
            'translation': t.get('translation'),
            'context': (t.get('context') or '')[:200],
        }
        for t in terms
    ]

    # 6. generate
    t0 = time.time()
    answer = _generate(req.question, context_text or '(no context retrieved)', lang)
    timings['generate_ms'] = round((time.time() - t0) * 1000, 2)

    # 7. consistency score (model-free heuristic)
    from app.services.generative_qa import consistency_score
    t0 = time.time()
    confidence = consistency_score(answer, context_text or '')
    timings['score_ms'] = round((time.time() - t0) * 1000, 2)
    timings['total_ms'] = round(sum(
        v for v in timings.values() if isinstance(v, (int, float))), 2)

    return AskResponse(
        answer=answer,
        confidence=confidence,
        low_confidence=confidence < 0.5,
        language=lang,
        sources=sources,
        context_used=bool(context_text),
        model=GROQ_MODEL,
        timings=timings,
    )
