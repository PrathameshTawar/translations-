"""
Open-vocabulary passage-retrieval benchmark with a reranker on/off ablation.

The glossary benchmark (retrieval.py) runs on tiny per-language glossaries
(15 terms) where the dense bi-encoder sits at ceiling — so the cross-encoder
reranker has nothing to improve. This benchmark evaluates the reranker where
it is *supposed* to help: open-vocabulary passage retrieval, i.e. finding a
passage in a large corpus given a query that is a paraphrase, not a copy.

Task construction
-----------------
* Corpus: unique English passages from the in-project parallel data (the
  42 curated sentences + the 25 code-mixed English references). Optionally
  replace/augment with FLORES-200 when it can be loaded (gated repo — it
  degrades gracefully to the in-project corpus otherwise).
* Queries (one per passage, ground truth = the passage it derives from):
    - roundtrip  — NLLB en->hi->en round trip. The back-translated English
                   is a paraphrase of the passage, so retrieval must match on
                   meaning, not exact words. This is the realistic
                   open-vocabulary regime. Needs the live NLLB model.
    - masked     — deterministic offline fallback: the query is the passage
                   with its most distinctive content word(s) blanked out
                   (highest corpus idf). Also non-trivial — the bi-encoder
                   cannot lean on that word.

Evaluation
----------
Dense-only: bi-encoder (paraphrase-multilingual-MiniLM-L12-v2) retrieves the
top-10 passages; the top-5 is the baseline. Dense+rerank: the same top-10
pool is re-scored by the cross-encoder (ms-marco-MiniLM-L-6-v2) and the top-5
of the re-ranked list is taken. Recall@K / MRR / nDCG@K are reported for both
plus the delta — a positive delta on masked/roundtrip queries is the
paper's "+3.9 Recall@5 on open-vocabulary passage retrieval" claim, measured
honestly on this corpus.

Usage (from backend/):
    python -m app.evaluation.passage_retrieval               # offline (masked queries)
    python -m app.evaluation.passage_retrieval --reranker    # + cross-encoder ablation
    python -m app.evaluation.passage_retrieval --live        # NLLB round-trip queries
    python -m app.evaluation.passage_retrieval --live --reranker
"""

import json
import logging
import math
import os
import re
import time
from collections import Counter

logger = logging.getLogger(__name__)

# Corpus source: in-project parallel data (always available, no downloads)
from app.evaluation.datasets import (
    CODE_MIXED_ITEMS,
    ENGLISH_SENTENCES,
    load_flores_dataset,
)

DEFAULT_TOP_KS = (1, 3, 5)

# Dense-pool size the reranker re-ranks: standard "retrieve 10, rerank, keep 5".
RERANK_POOL = 10


def _log2(n):
    return math.log2(n)


# ---------------------------------------------------------------------------
# Corpus + query builders (pure, testable)
# ---------------------------------------------------------------------------

def build_passage_corpus(flores: bool = False, max_passages: int = None) -> dict:
    """
    Build the open-vocabulary passage corpus.

    Returns dict with 'passages' (list[str], unique, non-empty), 'source'
    (label), 'num_passages'. FLORES is attempted only when `flores=True`;
    any failure degrades to the in-project corpus.
    """
    passages = list(dict.fromkeys(
        s.strip() for s in ENGLISH_SENTENCES if s and s.strip()
    ))
    for item in CODE_MIXED_ITEMS:
        ref = (item.get('reference') or '').strip()
        if ref:
            passages.append(ref)
    passages = list(dict.fromkeys(passages))

    source = 'in-project parallel corpus (curated + code-mixed references)'
    if flores:
        try:
            flores_data = load_flores_dataset(
                target_langs=['hindi'], max_sentences=max_passages or 200,
            )
            if 'error' not in flores_data:
                sources = [
                    s.strip() for s in flores_data.get('sources', [])
                    if s and s.strip()
                ]
                # Keep only FLORES passages (dedupe against in-project).
                known = set(passages)
                flores_passages = [s for s in sources if s not in known]
                if flores_passages:
                    passages.extend(flores_passages)
                    source = (
                        'FLORES-200 devtest (hindi) + in-project parallel corpus'
                    )
        except Exception as e:  # pragma: no cover - network/env dependent
            logger.warning(f'FLORES unavailable for passage corpus: {e}')

    if max_passages:
        passages = passages[:max_passages]
    return {'passages': passages, 'source': source,
            'num_passages': len(passages)}


def _content_words(text: str):
    """Lowercased alphabetic tokens of length >= 4 (drops stopwords/short)."""
    return [w for w in re.findall(r"[A-Za-z']+", text.lower()) if len(w) >= 4]


def build_masked_queries(passages, blank_words=1) -> list:
    """
    Deterministic offline queries: blank the passage's most distinctive
    content word(s) (highest corpus idf) so the query is a partial version of
    the passage. Ground truth = the passage index.
    """
    if not passages:
        return []
    freq = Counter()
    per_passage = []
    for p in passages:
        words = _content_words(p)
        per_passage.append(words)
        freq.update(words)

    n = len(passages)
    queries = []
    for i, words in enumerate(per_passage):
        if not words:
            queries.append({'query': passages[i], 'relevant': [i]})
            continue
        # Most distinctive = lowest corpus frequency (ties broken by order).
        ranked = sorted(set(words), key=lambda w: (freq[w], w))
        to_blank = set(ranked[:blank_words])
        tokens = re.split(r"([A-Za-z']+)", passages[i])
        masked = ''.join(
            '____' if part.lower() in to_blank else part for part in tokens
        )
        queries.append({'query': masked.strip(), 'relevant': [i]})
    return queries


def build_roundtrip_queries(passages, translate_fn) -> list:
    """
    Live queries: en -> hi -> en round trip via `translate_fn(text, src, tgt)
    -> str`. The back-translated English is a paraphrase; the ground truth is
    the original passage. Passages that fail to round-trip keep the passage
    itself as the query (dense will be at ceiling for those — the honest
    lower bound of the reranker's usefulness).
    """
    queries = []
    for i, passage in enumerate(passages):
        query = passage
        try:
            fwd = translate_fn(passage, 'english', 'hindi')
            back = translate_fn(fwd, 'hindi', 'english')
            if back and back.strip():
                query = back.strip()
        except Exception as e:
            logger.debug(f'round-trip failed for passage {i}: {e}')
        queries.append({'query': query, 'relevant': [i]})
    return queries


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class PassageRetrievalBenchmark:
    """
    Dense-only vs dense+rerank retrieval over an open-vocabulary passage
    corpus, with per-query latency for both modes.
    """

    def __init__(self, pipeline=None, use_encoder=True):
        self.use_encoder = use_encoder
        if pipeline is None:
            import app.services.rag_service as rag_mod
            if not use_encoder:
                orig = rag_mod.IndicRAGPipeline._try_load_encoder
                rag_mod.IndicRAGPipeline._try_load_encoder = lambda self: None
                try:
                    pipeline = rag_mod.IndicRAGPipeline.get_instance()
                finally:
                    rag_mod.IndicRAGPipeline._try_load_encoder = orig
            else:
                pipeline = rag_mod.IndicRAGPipeline.get_instance()
        self.pipeline = pipeline

    # -- metrics -----------------------------------------------------------

    @staticmethod
    def _mean(values):
        values = list(values)
        return sum(values) / max(len(values), 1)

    @classmethod
    def _metric_dict(cls, ranked_lists, relevant_list, top_ks):
        from app.evaluation.retrieval import RetrievalMetrics
        n = len(ranked_lists)
        out = {}
        for k in top_ks:
            out[f'recall_at_{k}'] = round(cls._mean(
                RetrievalMetrics.recall_at_k(ranked_lists[i], relevant_list[i], k)
                for i in range(n)
            ), 4)
            out[f'ndcg_at_{k}'] = round(cls._mean(
                RetrievalMetrics.ndcg_at_k(ranked_lists[i], relevant_list[i], k)
                for i in range(n)
            ), 4)
        out['mrr'] = round(RetrievalMetrics.mrr(ranked_lists, relevant_list), 4)
        return out

    @staticmethod
    def _diff(new, old):
        return {k: round((new.get(k) or 0.0) - (old.get(k) or 0.0), 4)
                for k in new}

    # -- retrieval core ----------------------------------------------------

    def _embed(self, texts):
        """Embed texts with the pipeline encoder (or a no-op if unavailable)."""
        if self.pipeline.encoder is None:
            raise RuntimeError(
                'Sentence-transformer encoder unavailable — cannot embed '
                'passages. Run with use_encoder=True (live mode).'
            )
        import numpy as np
        vecs = self.pipeline.encoder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    @staticmethod
    def _dense_topk(query_vec, index_mat, k):
        """Cosine-similarity top-k passage indices for one query vector."""
        import numpy as np
        sims = index_mat @ query_vec  # both L2-normalized
        return np.argsort(sims)[::-1][:k].tolist()

    def _retrieve(self, queries, passages, top_ks, rerank=False,
                  pool_size=RERANK_POOL, reranker=None):
        """
        Run retrieval over all queries.

        Returns (metrics, mean_latency_ms). When rerank=True the dense top-
        `pool_size` pool is re-scored by the cross-encoder and truncated to
        max(top_ks).
        """
        import numpy as np
        index_mat = self._embed(passages)
        keep = max(top_ks)
        ranked_lists = []
        relevant_list = []
        latencies = []
        for q in queries:
            t0 = time.time()
            qvec = self._embed([q['query']])[0]
            dense_top = self._dense_topk(qvec, index_mat, pool_size)
            if rerank:
                pool_texts = [passages[i] for i in dense_top]
                out = reranker.rerank(q['query'], pool_texts, top_k=keep)
                ranked = [dense_top[pool_texts.index(t)] for t in out['reranked']]
            else:
                ranked = dense_top[:keep]
            latencies.append((time.time() - t0) * 1000)
            ranked_lists.append(ranked)
            relevant_list.append(q['relevant'])

        metrics = self._metric_dict(ranked_lists, relevant_list, top_ks)
        mean_lat = (sum(latencies) / max(len(latencies), 1)) if latencies else None
        return metrics, round(mean_lat, 2) if mean_lat is not None else None

    # -- main --------------------------------------------------------------

    def run(self, queries_mode='masked', use_reranker=False, top_ks=DEFAULT_TOP_KS,
            flores=False, max_passages=None, translate_fn=None,
            passages=None) -> dict:
        """
        Run the passage-retrieval benchmark.

        Args:
            queries_mode: 'masked' (offline) or 'roundtrip' (needs translate_fn).
            use_reranker: run the dense+rerank ablation (loads the ~430MB
                          cross-encoder on first use).
            flores: attempt to load FLORES-200 as the passage source.
            translate_fn: (text, src, tgt) -> str, used for roundtrip queries.
            passages: optional pre-built passage list (tests / quick runs).
        """
        corpus = ({
            'passages': passages, 'source': 'injected', 'num_passages': len(passages),
        } if passages is not None else
        build_passage_corpus(flores=flores, max_passages=max_passages))
        passages = corpus['passages']
        if not passages:
            return {'error': 'no passages to retrieve', 'corpus': corpus}

        if queries_mode == 'roundtrip':
            if translate_fn is None:
                return {'error': 'roundtrip queries need a live translate_fn '
                                 '(use --live)'}
            queries = build_roundtrip_queries(passages, translate_fn)
        else:
            queries = build_masked_queries(passages)

        # Dense-only baseline
        dense, dense_lat = self._retrieve(
            queries, passages, top_ks, rerank=False)
        result = {
            'corpus': corpus,
            'queries_mode': queries_mode,
            'num_queries': len(queries),
            'top_ks': list(top_ks),
            'dense': dense,
            'dense_query_latency_ms': dense_lat,
            'reranker': {'available': False},
        }

        # Reranker ablation
        if use_reranker:
            from app.services.reranker import CrossEncoderReranker
            reranker = CrossEncoderReranker()
            if reranker.available:
                reranked, reranked_lat = self._retrieve(
                    queries, passages, top_ks, rerank=True, reranker=reranker)
                result['reranker'] = {
                    'available': True,
                    'model': reranker.model_name,
                    'reranked': reranked,
                    'delta': self._diff(reranked, dense),
                    'reranked_query_latency_ms': reranked_lat,
                    'rerank_overhead_ms': (
                        round(reranked_lat - dense_lat, 2)
                        if dense_lat is not None and reranked_lat is not None
                        else None
                    ),
                }
            else:
                result['reranker'] = {
                    'available': False,
                    'model': reranker.model_name,
                    'error': 'cross-encoder model unavailable',
                }

        note = (
            f'Corpus: {corpus["num_passages"]} unique English passages. '
        )
        if queries_mode == 'roundtrip':
            note += ('Queries are NLLB round-trip paraphrases (en->hi->en) — '
                     'retrieval must match on meaning, which is the regime '
                     'where re-ranking can help.')
        else:
            note += ('Queries are masked passages (most distinctive content '
                     'word blanked) — a deterministic, offline stand-in for '
                     'paraphrase retrieval.')
        result['note'] = note
        return result


def run_passage_retrieval_benchmark(use_encoder=True, use_reranker=False,
                                    queries_mode='masked', flores=False,
                                    max_passages=None) -> dict:
    """Module-level convenience entry point (offline-safe)."""
    translate_fn = None
    if queries_mode == 'roundtrip':
        from app.services.model_registry import ModelRegistry
        registry = ModelRegistry.get_instance()

        def translate_fn(text, source_lang, target_lang):
            return registry.translate(
                text=text, source_lang=source_lang, target_lang=target_lang,
                model='nllb-200-distilled-600M', use_tm=False,
            ).translated

    bench = PassageRetrievalBenchmark(use_encoder=use_encoder)
    return bench.run(queries_mode=queries_mode, use_reranker=use_reranker,
                     flores=flores, max_passages=max_passages,
                     translate_fn=translate_fn)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Open-vocabulary passage retrieval (dense vs rerank)')
    parser.add_argument('--live', action='store_true',
                        help='Use NLLB round-trip paraphrase queries (loads NLLB)')
    parser.add_argument('--reranker', action='store_true',
                        help='Run the cross-encoder on/off ablation')
    parser.add_argument('--queries', default=None, choices=['masked', 'roundtrip'],
                        help='Query mode (default: masked, or roundtrip with --live)')
    parser.add_argument('--flores', action='store_true',
                        help='Attempt FLORES-200 as the passage source')
    parser.add_argument('--max-passages', type=int, default=None)
    args = parser.parse_args()

    mode = args.queries or ('roundtrip' if args.live else 'masked')
    # Retrieval needs the encoder to embed passages in both modes; --live
    # additionally enables the NLLB round-trip query generation.
    result = run_passage_retrieval_benchmark(
        use_encoder=True, use_reranker=args.reranker,
        queries_mode=mode, flores=args.flores, max_passages=args.max_passages,
    )
    if 'error' in result:
        print('ERROR:', result['error'])
        raise SystemExit(1)

    print(f"\nPassage retrieval ({result['queries_mode']} queries)")
    print(f"Corpus: {result['corpus']['num_passages']} passages "
          f"({result['corpus']['source']})")
    print(f"Queries: {result['num_queries']}")
    print(f"\n  dense   : {result['dense']}")
    print(f"  latency : {result['dense_query_latency_ms']} ms/query")
    rer = result['reranker']
    if rer.get('available'):
        print(f"  reranked: {rer['reranked']}")
        print(f"  delta   : {rer['delta']}")
        print(f"  latency : dense={result['dense_query_latency_ms']} ms "
              f"reranked={rer['reranked_query_latency_ms']} ms "
              f"overhead={rer['rerank_overhead_ms']} ms/query")
    print(f"\nNote: {result['note']}")
