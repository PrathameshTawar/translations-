"""
Generative QA — grounded question answering with LLaMA-3-8B via the Groq
API (the paper's Section V-D retrieval-generation pipeline, low-latency
LLM config).

Enabled only when ``GROQ_API_KEY`` is set; otherwise ``is_available()`` is
False and callers should surface a clear "not configured" message. The Groq
chat-completions endpoint is called directly over HTTPS with ``urllib``
(stdlib) so no extra dependency (langchain/requests) is required.

Flow (mirrors the paper's /ask sketch):
  1. language detection   — script-based detector (existing service)
  2. retrieval            — IndicRAGPipeline.retrieve_context (terminology
                             grounding; dense FAISS when available)
  3. prompt               — structured context + strict grounding instruction
                             ("Only answer using the provided context")
  4. generation           — llama3-8b-8192, temperature 0.2, max 200 tokens
  5. confidence           — model-free heuristic: content-word coverage of the
                             answer in the retrieved context, penalized by
                             degenerate-output signatures (repetition). Low
                             confidence is surfaced as a warning, never used
                             to block generation (paper Section V-E).
"""

import json
import logging
import os
import re
import time
import urllib.request

logger = logging.getLogger(__name__)

GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions'
DEFAULT_MODEL = 'llama3-8b-8192'
MAX_QUESTION_CHARS = 2000

_ANSWER_ABSENT_MARKER = 'the context does not contain this information'

_WORD_RE = re.compile(r'[A-Za-z\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F'
                      r'\u0C80-\u0CFF\u0D00-\u0D7F\u0B00-\u0B7F]+')


def is_available() -> bool:
    """Whether Groq credentials are configured (generation is enabled)."""
    return bool(os.getenv('GROQ_API_KEY'))


def _lang_native_name(lang: str) -> str:
    names = {
        'english': 'English', 'hindi': 'Hindi', 'marathi': 'Marathi',
        'bengali': 'Bengali', 'gujarati': 'Gujarati', 'tamil': 'Tamil',
        'telugu': 'Telugu', 'kannada': 'Kannada',
    }
    return names.get((lang or '').lower(), (lang or 'English').capitalize())


def _call_groq(prompt: str, api_key: str, model: str = DEFAULT_MODEL,
               temperature: float = 0.2, max_tokens: int = 200,
               timeout: int = 30) -> str:
    """Call the Groq chat-completions endpoint. Raises on HTTP errors."""
    body = {
        'model': model,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'messages': [
            {'role': 'system',
             'content': 'You are a precise question-answering assistant. '
                        'Always ground your answer in the provided context. '
                        f'If the information is not in the context, reply with '
                        f'exactly: "{_ANSWER_ABSENT_MARKER}."'},
            {'role': 'user', 'content': prompt},
        ],
    }
    req = urllib.request.Request(
        GROQ_URL,
        data=json.dumps(body).encode('utf-8'),
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        },
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    content = (data.get('choices') or [{}])[0].get('message') or {}
    answer = (content.get('content') or '').strip()
    if not answer:
        raise RuntimeError('Groq returned an empty completion')
    return answer


def consistency_score(answer: str, context_text: str) -> float:
    """
    Model-free grounding heuristic (0..1).

    base = fraction of the answer's content words that appear in the
    retrieved context; an answer paraphrasing the context scores high, an
    invented answer scores ~0. Degenerate/repetitive output is penalized
    (hallucination signature). The paper's NLI-based scorer is the ideal
    upgrade path; this keeps the pipeline offline and deterministic.
    """
    if not answer or not answer.strip():
        return 0.0
    tokens = _WORD_RE.findall(answer.lower())
    if not tokens:
        return 0.0
    ctx = (context_text or '').lower()
    covered = sum(1 for t in tokens if t in ctx)
    base = covered / len(tokens)

    from app.evaluation.hallucination import HallucinationDetector
    rep = HallucinationDetector.repetition_ratio(answer)
    if rep >= 0.35:
        base *= 0.4
    return round(min(max(base, 0.0), 1.0), 2)


class GenerativeQA:
    """Singleton grounded-QA service (lazy; no models loaded here)."""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.model = DEFAULT_MODEL

    def ask(self, question: str, lang: str = None, top_k: int = 5,
            use_reranker: bool = False, timeout: int = 30) -> dict:
        """
        Answer `question` grounded in the retrieved terminology context.

        Returns a dict with answer, confidence, low_confidence, language,
        sources, model, and timings. Raises RuntimeError when Groq is not
        configured or the upstream call fails.
        """
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise RuntimeError('GROQ_API_KEY is not set — generative QA '
                               'is not configured')

        timings = {}

        # 1. Language detection (script-based, model-free)
        t0 = time.time()
        detected = lang or 'english'
        if not lang:
            try:
                from app.services.language_detection import LanguageDetector
                det = LanguageDetector().detect_with_scores(question)
                detected = (det.get('detected_language') or 'english').lower()
                if detected not in _LANG_CODES:
                    detected = 'english'
            except Exception as e:
                logger.debug(f'Language detection failed (defaulting): {e}')
                detected = 'english'
        timings['detect_ms'] = round((time.time() - t0) * 1000, 2)

        # 2. Retrieval (terminology grounding)
        t0 = time.time()
        from app.services.rag_service import IndicRAGPipeline
        rag = IndicRAGPipeline.get_instance()
        rag_result = rag.retrieve_context(
            question, source_lang='english', target_lang=detected,
            top_k=top_k, use_reranker=use_reranker,
        )
        context_text = rag_result.get('context')
        terms = rag_result.get('terms', [])
        timings['retrieve_ms'] = round((time.time() - t0) * 1000, 2)

        sources = [
            {
                'term': t.get('term'),
                'translation': t.get('translation'),
                'context': (t.get('context') or '')[:200],
            }
            for t in terms
        ]

        # 3. Grounded prompt
        if context_text:
            context_block = '\n'.join(
                f'- {t.get("term")}: {t.get("translation")}'
                + (f' ({t.get("context")})' if t.get('context') else '')
                for t in terms
            )
            prompt = (
                f'Context:\n{context_block}\n\n'
                f'Question: {question}\n\n'
                f'Answer in {_lang_native_name(detected)}. '
                f'Use only the context above.'
            )
        else:
            prompt = (
                f'Question: {question}\n\n'
                f'Answer in {_lang_native_name(detected)}. '
                f'No context was retrieved — state that you cannot answer '
                f'without context.'
            )

        # 4. Generation
        t0 = time.time()
        answer = _call_groq(prompt, api_key, model=self.model, timeout=timeout)
        timings['generate_ms'] = round((time.time() - t0) * 1000, 2)

        # 5. Confidence (model-free grounding heuristic)
        t0 = time.time()
        confidence = consistency_score(answer, context_text or '')
        low_confidence = confidence < 0.5
        timings['score_ms'] = round((time.time() - t0) * 1000, 2)

        return {
            'answer': answer,
            'confidence': confidence,
            'low_confidence': low_confidence,
            'language': detected,
            'language_name': _lang_native_name(detected),
            'sources': sources,
            'context_used': bool(context_text),
            'reranked': bool(rag_result.get('reranked')),
            'reranker_available': bool(rag_result.get('reranker_available')),
            'model': self.model,
            'timings': timings,
        }


_LANG_CODES = {
    'english', 'hindi', 'marathi', 'bengali', 'gujarati',
    'tamil', 'telugu', 'kannada',
}
