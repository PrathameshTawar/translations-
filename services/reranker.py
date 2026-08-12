"""
Cross-encoder reranker for the terminology RAG pipeline.

After the bi-encoder (sentence-transformer) + FAISS step retrieves candidate
terms, this service rescorses them with a cross-encoder — the same model
family the paper's reranker ablation uses (cross-encoder/ms-marco-MiniLM-L-6-v2).
Cross-encoders attend to the (query, candidate) pair jointly, so they are
stronger (and slower) than the bi-encoder — the classic retrieval/rerank
trade-off.

Design mirrors the COMETkiwi pattern in services/quality.py:

- The ~430 MB model is loaded lazily (module-level singleton) on first use,
  and only when a caller explicitly asks to rerank.
- Every public entry point degrades gracefully: if the model cannot be
  loaded or scoring fails, the input order is returned unchanged with
  `available=False` — the RAG pipeline keeps working without it.

Usage:
    reranker = CrossEncoderReranker()
    out = reranker.rerank(query, candidate_terms, top_k=3)
    # out = {'reranked': [...], 'scores': [...], 'available': True}
"""

import logging
import threading
from typing import List, Optional

logger = logging.getLogger(__name__)

# Guards the lazy singleton load: the throughput/concurrency path may call
# retrieve_context (and thus the reranker) from several threads at once, and
# two simultaneous loads of the ~430 MB model would be wasteful.
_LOAD_LOCK = threading.Lock()

# Matches the paper's reranker (Section V-D: ms-marco MiniLM cross-encoder).
CROSS_ENCODER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# Module-level lazy singleton (like COMETkiwi in services/quality.py):
# _RERANKER = None | model; _RERANKER_AVAILABLE = None | True | False.
_RERANKER = None
_RERANKER_AVAILABLE = None  # None = not probed yet, False = probed & missing


def _load_reranker_model():
    """Lazily download + load the cross-encoder (module singleton)."""
    global _RERANKER, _RERANKER_AVAILABLE
    if _RERANKER is not None:
        return _RERANKER
    if _RERANKER_AVAILABLE is False:
        return None
    with _LOAD_LOCK:
        # Re-check inside the lock: another thread may have loaded meanwhile.
        if _RERANKER is not None:
            return _RERANKER
        if _RERANKER_AVAILABLE is False:
            return None
        try:
            from sentence_transformers import CrossEncoder
            _RERANKER = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
            _RERANKER_AVAILABLE = True
        except Exception as exc:  # pragma: no cover - env dependent
            _RERANKER_AVAILABLE = False
            logger.error(f'Cross-encoder reranker could not be loaded: {exc}')
            return None
    return _RERANKER


def reset_reranker_singleton():
    """Drop the cached model (used by tests)."""
    global _RERANKER, _RERANKER_AVAILABLE
    _RERANKER = None
    _RERANKER_AVAILABLE = None


class CrossEncoderReranker:
    """
    Rerank retrieved candidates by cross-encoder similarity to the query.

    Candidates are dicts (term records with a 'term' field, as produced by
    the RAG glossary) or plain strings; scoring is always against the
    candidate's text representation.
    """

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL_NAME,
                 batch_size: int = 16):
        self._model_name = model_name
        self.batch_size = batch_size

    @property
    def available(self) -> bool:
        return _load_reranker_model() is not None

    @property
    def model_name(self) -> str:
        return self._model_name

    @staticmethod
    def _candidate_text(candidate) -> str:
        if isinstance(candidate, dict):
            return candidate.get('term') or candidate.get('text') or ''
        return str(candidate)

    def rerank(self, query: str, candidates: List,
               top_k: Optional[int] = None,
               score_key: str = 'term') -> dict:
        """
        Rescore `candidates` by cross-encoder relevance to `query`.

        Args:
            query: The query text (e.g. the source sentence being translated).
            candidates: Candidate term records to rerank.
            top_k: Keep only the top-k after reranking (default: all).
            score_key: Candidate dict field to score against the query.

        Returns:
            dict with:
              reranked — candidates in new order (input order if unavailable)
              scores   — cross-encoder scores aligned with `reranked`
              available — whether the reranker actually ran
        """
        if not candidates:
            return {'reranked': list(candidates), 'scores': [],
                    'available': False}

        model = _load_reranker_model()
        if model is None:
            logger.warning('Reranker unavailable — returning input order')
            return {'reranked': list(candidates), 'scores': [],
                    'available': False}

        try:
            pairs = [
                (query or '', self._candidate_text(c))
                for c in candidates
            ]
            scores = model.predict(pairs, batch_size=self.batch_size)
            if isinstance(scores, list) and scores and isinstance(scores[0], list):
                # Some backends return one score per pair; others return
                # [[score]] per pair. Normalize to flat floats.
                scores = [s[0] if isinstance(s, (list, tuple)) else s
                          for s in scores]
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1],
                            reverse=True)
            if top_k:
                ranked = ranked[:top_k]
            return {
                'reranked': [c for c, _ in ranked],
                'scores': [round(float(s), 4) for _, s in ranked],
                'available': True,
            }
        except Exception as exc:
            logger.warning(f'Reranking failed ({exc}); returning input order')
            return {'reranked': list(candidates), 'scores': [],
                    'available': False}

    def __repr__(self):
        return (f'CrossEncoderReranker(model={self._model_name!r}, '
                f'available={self.available})')
