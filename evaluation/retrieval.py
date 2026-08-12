"""
Retrieval metrics and terminology-RAG retrieval benchmark.

Implements the plan's "Benchmark #12 — Benchmark Your RAG Component":

  * Recall@K  — fraction of queries where a relevant term appears in top-K
  * MRR       — mean reciprocal rank of the first relevant result
  * nDCG@K    — discounted cumulative gain (binary relevance here)
  * P@K       — precision at K

Ground truth (qrels) is built from the terminology JSON files: for every
term we issue a query built from the term + its context, and the relevant
result is that term itself. The benchmark measures the *existing* pipeline
(FAISS semantic search when the index + encoder are available, keyword
overlap otherwise) and reports which backend was active — this is the
"dense only" baseline row for the paper.

RERANKER ABLATION (Section VIII on/off comparison)
--------------------------------------------------
When `use_reranker=True` and the cross-encoder is available, the same
evaluation is repeated with the reranker switched on and the deltas are
reported:

  * term_queries    — the standard qrels (term + context queries)
  * context_queries — a harder variant where the query is the term's domain
                      context with the term itself removed, so the retriever
                      must match on semantics rather than lexical overlap
                      (this is where re-ranking can actually change the
                      ranking on the small glossaries)

Latency of both modes is measured so the ablation shows the reranker's
speed cost (the paper's "+0.4s latency" side of the trade-off).

Usage (from translator-backend/ — FAISS index files must be in CWD):
    python -m app.evaluation.retrieval [--lang hindi,marathi] [--top-k 1,3,5,10]
    python -m app.evaluation.retrieval --reranker   # includes on/off ablation
"""

import json
import logging
import math
import os
import time

logger = logging.getLogger(__name__)

TERMINOLOGY_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'terminology'
)
TERMINOLOGY_DIR = os.path.abspath(TERMINOLOGY_DIR)


def _log2(n):
    return math.log2(n)


class RetrievalMetrics:
    """Pure metric functions over ranked result lists."""

    @staticmethod
    def recall_at_k(ranked, relevant, k):
        """Fraction of relevant items present in the top-k."""
        if not relevant:
            return 0.0
        top_k = set(ranked[:k])
        hits = sum(1 for r in relevant if r in top_k)
        return hits / len(relevant)

    @staticmethod
    def precision_at_k(ranked, relevant, k):
        """Precision of the top-k results."""
        top_k = ranked[:k]
        if not top_k:
            return 0.0
        relevant_set = set(relevant)
        hits = sum(1 for r in top_k if r in relevant_set)
        return hits / len(top_k)

    @staticmethod
    def reciprocal_rank(ranked, relevant):
        """1/rank of the first relevant result (0 if none retrieved)."""
        relevant_set = set(relevant)
        for i, item in enumerate(ranked):
            if item in relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def ndcg_at_k(ranked, relevant, k):
        """
        nDCG@K with binary relevance.
        Gain = 1 for relevant, discounted by log2(rank + 1).
        """
        relevant_set = set(relevant)
        dcg = 0.0
        for i, item in enumerate(ranked[:k]):
            if item in relevant_set:
                dcg += 1.0 / _log2(i + 2)

        # Ideal DCG: all relevant items at the top of the ranking
        ideal = sum(1.0 / _log2(i + 2) for i in range(min(k, len(relevant_set))))
        return dcg / ideal if ideal > 0 else 0.0

    @staticmethod
    def mrr(ranked_lists, relevant_per_query):
        """Mean reciprocal rank across queries."""
        if not ranked_lists:
            return 0.0
        total = sum(
            RetrievalMetrics.reciprocal_rank(ranked_lists[i], relevant_per_query[i])
            for i in range(len(ranked_lists))
        )
        return total / len(ranked_lists)


class TerminologyRetrievalBenchmark:
    """
    Recall@K / MRR / nDCG for the terminology RAG pipeline.

    Runs against IndicRAGPipeline.retrieve_context with ground-truth qrels
    derived from the terminology files. Reports which retrieval backend was
    actually used (faiss-semantic vs keyword-fallback) and, when requested,
    the cross-encoder reranker on/off ablation.
    """

    def __init__(self, pipeline=None, use_encoder=True):
        """
        Args:
            pipeline: Pre-built IndicRAGPipeline (default: build one).
            use_encoder: Load the sentence-transformer encoder (FAISS semantic
                         search). Set False for fast/offline runs that use the
                         keyword-fallback backend.
        """
        self.use_encoder = use_encoder
        if pipeline is None:
            import app.services.rag_service as rag_mod
            if not use_encoder:
                # Build the singleton without loading the encoder (fast, offline)
                orig = rag_mod.IndicRAGPipeline._try_load_encoder
                rag_mod.IndicRAGPipeline._try_load_encoder = lambda self: None
                try:
                    pipeline = rag_mod.IndicRAGPipeline.get_instance()
                finally:
                    rag_mod.IndicRAGPipeline._try_load_encoder = orig
            else:
                pipeline = rag_mod.IndicRAGPipeline.get_instance()
        self.pipeline = pipeline

    # -- helpers ------------------------------------------------------------

    def _load_terms(self, lang):
        path = os.path.join(TERMINOLOGY_DIR, f'{lang}_terms.json')
        if not os.path.exists(path):
            return []
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return list(data.values()) if isinstance(data, dict) else data

    def _build_qrels(self, lang, query_mode='term'):
        """
        Return list of {'query', 'relevant': [term]} built from terminology.

        query_mode:
          'term'    — query contains the term itself (standard qrels)
          'context' — query is the term's domain context with the term
                      removed; the retriever must match on semantics
                      (harder — this is where reranking can bite)
        """
        terms = self._load_terms(lang)
        if not terms:
            return []
        qrels = []
        for t in terms:
            term = (t.get('term') or '').strip()
            context = (t.get('context') or '').strip()
            if not term:
                continue
            if query_mode == 'context':
                if not context or context.lower() == term.lower() or len(context) < 3:
                    continue
                qrels.append({'query': context, 'relevant': [term]})
                continue
            if context and term.lower() not in context.lower():
                query = f"{term} — {context}"
            else:
                query = f"Please translate the term {term}."
            qrels.append({'query': query, 'relevant': [term]})
        return qrels

    @staticmethod
    def _extract_ranked(result, top_k):
        """Convert a retrieve_context result into a ranked list of term texts."""
        ranked = []
        for t in (result.get('terms') or []):
            term = (t.get('term') or '').strip()
            if term and term not in ranked:
                ranked.append(term)
        return ranked[:top_k]

    @staticmethod
    def _mean(values):
        values = list(values)
        return sum(values) / max(len(values), 1)

    # -- evaluation core ------------------------------------------------------

    def _evaluate_langs(self, langs, top_ks, max_queries, rerank=False,
                        query_mode='term'):
        """
        Run retrieval over the given languages.

        Returns (results, num_queries_total, query_latency_ms).
        `rerank=True` turns the pipeline's cross-encoder reranker on.
        """
        results = {}
        num_queries_total = 0
        latency_ms = []
        for lang in langs:
            qrels = self._build_qrels(lang, query_mode=query_mode)
            if not qrels:
                results[lang] = {'error': f'no terminology data for {lang}'}
                continue
            num_terms = len(self._load_terms(lang))
            if max_queries:
                qrels = qrels[:max_queries]

            ranked_lists = []
            relevant_list = []
            max_k = max(top_ks)
            for q in qrels:
                t0 = time.time()
                # Glossary lookup is by target language: the English term is
                # queried against {lang}_terms.json
                res = self.pipeline.retrieve_context(
                    q['query'], source_lang='english', target_lang=lang,
                    top_k=max_k, use_reranker=rerank,
                )
                latency_ms.append((time.time() - t0) * 1000)
                ranked_lists.append(self._extract_ranked(res, max_k))
                relevant_list.append(q['relevant'])

            n = len(ranked_lists)
            recall = {k: self._mean(
                RetrievalMetrics.recall_at_k(ranked_lists[i], relevant_list[i], k)
                for i in range(n)
            ) for k in top_ks}
            precision = {k: self._mean(
                RetrievalMetrics.precision_at_k(ranked_lists[i], relevant_list[i], k)
                for i in range(n)
            ) for k in top_ks}
            ndcg = {k: self._mean(
                RetrievalMetrics.ndcg_at_k(ranked_lists[i], relevant_list[i], k)
                for i in range(n)
            ) for k in top_ks}

            results[lang] = {
                'num_queries': n,
                'num_terms': num_terms,
                'recall_at_k': {k: round(v, 4) for k, v in recall.items()},
                'precision_at_k': {k: round(v, 4) for k, v in precision.items()},
                'ndcg_at_k': {k: round(v, 4) for k, v in ndcg.items()},
                'mrr': round(RetrievalMetrics.mrr(ranked_lists, relevant_list), 4),
            }
            num_queries_total += n
        return results, num_queries_total, latency_ms

    @staticmethod
    def _macro_metrics(languages: dict, top_ks) -> dict:
        """Macro-average (mean over languages) of the retrieval metrics."""
        lang_results = [m for m in languages.values() if 'error' not in m]
        if not lang_results:
            return {}
        out = {}
        for k in top_ks:
            out[f'recall_at_{k}'] = TerminologyRetrievalBenchmark._mean(
                m['recall_at_k'][k] for m in lang_results)
            out[f'ndcg_at_{k}'] = TerminologyRetrievalBenchmark._mean(
                m['ndcg_at_k'][k] for m in lang_results)
        out['mrr'] = TerminologyRetrievalBenchmark._mean(
            m['mrr'] for m in lang_results)
        return {k: round(v, 4) for k, v in out.items()}

    @staticmethod
    def _metric_diff(new: dict, old: dict) -> dict:
        return {k: round((new.get(k) or 0.0) - (old.get(k) or 0.0), 4)
                for k in new}

    @staticmethod
    def _mean_ms(values) -> float:
        return round(sum(values) / max(len(values), 1), 2) if values else None

    # -- main ---------------------------------------------------------------

    def run(self, langs=None, top_ks=(1, 3, 5, 10), max_queries=None,
            use_reranker=False) -> dict:
        """
        Run the retrieval benchmark.

        Args:
            langs: Languages to test (default: all that have terminology files).
            top_ks: K values for Recall@K / nDCG@K.
            max_queries: Cap queries per language (default: all).
            use_reranker: Also run the cross-encoder on/off ablation
                          (downloads ~430MB on first use).

        Returns:
            dict with per-language metrics, overall aggregates, and — when
            `use_reranker` — a `reranker` block comparing dense-only vs
            dense + rerank on term queries and the harder context queries.
        """
        if langs is None:
            langs = [
                fname.replace('_terms.json', '')
                for fname in os.listdir(TERMINOLOGY_DIR)
                if fname.endswith('_terms.json')
            ]
        elif isinstance(langs, str):
            langs = [langs]

        backend = 'keyword-fallback'
        if self.pipeline.encoder is not None:
            if any(os.path.exists(f'{l}_index.faiss') for l in langs):
                backend = 'faiss-semantic'
            else:
                backend = 'encoder-loaded-but-no-index'

        languages, num_queries_total, latency_ms = self._evaluate_langs(
            langs, top_ks, max_queries, rerank=False)

        glossary_sizes = {
            lang: m.get('num_terms', 0)
            for lang, m in languages.items()
        }

        note = (
            f'Glossary sizes (terms per language): {glossary_sizes}. '
            'Because each glossary is small, perfect Recall@1/MRR is '
            'trivially reachable; treat these as sanity numbers, not '
            'evidence of strong retrieval on open vocabulary.'
        )

        if use_reranker:
            from app.services.reranker import CrossEncoderReranker
            reranker = CrossEncoderReranker()
            if reranker.available:
                reranked_langs, _, reranked_lat = self._evaluate_langs(
                    langs, top_ks, max_queries, rerank=True)
                ctx_dense, _, _ = self._evaluate_langs(
                    langs, top_ks, max_queries, rerank=False,
                    query_mode='context')
                ctx_reranked, _, _ = self._evaluate_langs(
                    langs, top_ks, max_queries, rerank=True,
                    query_mode='context')
                result = self._build_reranker_block(
                    top_ks, languages, reranked_langs,
                    ctx_dense, ctx_reranked,
                    latency_ms, reranked_lat, reranker.model_name,
                )
                result['available'] = True
                note += (' Reranker comparison (cross-encoder on/off) ran '
                         f'with {reranker.model_name}; see the "reranker" '
                         'block for deltas and latency cost.')
            else:
                result = {
                    'available': False,
                    'model': reranker.model_name,
                    'error': 'cross-encoder model unavailable',
                }
                note += (' Reranker comparison skipped: the cross-encoder '
                         'model could not be loaded.')
        else:
            result = {'available': False,
                      'note': 'Reranker comparison not requested (use '
                              '--reranker).'}

        return {
            'backend': backend,
            'top_ks': list(top_ks),
            'languages': languages,
            'num_queries_total': num_queries_total,
            'glossary_sizes': glossary_sizes,
            'reranker': result,
            'note': note,
        }

    def _build_reranker_block(self, top_ks, dense_langs, reranked_langs,
                              ctx_dense_langs, ctx_reranked_langs,
                              dense_latency_ms, reranked_latency_ms,
                              model_name) -> dict:
        """Aggregate the dense-vs-reranked deltas for both query variants."""
        dense = self._macro_metrics(dense_langs, top_ks)
        reranked = self._macro_metrics(reranked_langs, top_ks)
        ctx_dense = self._macro_metrics(ctx_dense_langs, top_ks)
        ctx_reranked = self._macro_metrics(ctx_reranked_langs, top_ks)

        dense_lat = self._mean_ms(dense_latency_ms)
        reranked_lat = self._mean_ms(reranked_latency_ms)

        return {
            'model': model_name,
            'term_queries': {
                'dense': dense,
                'reranked': reranked,
                'delta': self._metric_diff(reranked, dense),
            },
            'context_queries': {
                'dense': ctx_dense,
                'reranked': ctx_reranked,
                'delta': self._metric_diff(ctx_reranked, ctx_dense),
            },
            'dense_query_latency_ms': dense_lat,
            'reranked_query_latency_ms': reranked_lat,
            'rerank_overhead_ms': (
                round(reranked_lat - dense_lat, 2)
                if dense_lat is not None and reranked_lat is not None else None
            ),
        }


def run_retrieval_benchmark(langs=None, top_ks=(1, 3, 5, 10), use_encoder=True,
                            use_reranker=False) -> dict:
    """Module-level convenience entry point."""
    return TerminologyRetrievalBenchmark(use_encoder=use_encoder).run(
        langs=langs, top_ks=top_ks, use_reranker=use_reranker
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Terminology RAG retrieval benchmark')
    parser.add_argument('--lang', default=None, help='Comma-separated languages')
    parser.add_argument('--top-k', default='1,3,5,10', help='Comma-separated K values')
    parser.add_argument('--reranker', action='store_true',
                        help='Also run the cross-encoder on/off ablation '
                             '(downloads ~430MB on first use)')
    args = parser.parse_args()

    langs = args.lang.split(',') if args.lang else None
    top_ks = tuple(int(k) for k in args.top_k.split(','))

    result = run_retrieval_benchmark(langs=langs, top_ks=top_ks,
                                     use_reranker=args.reranker)
    print(f"\nRetrieval benchmark (backend: {result['backend']})")
    print(f"Queries: {result['num_queries_total']}")
    for lang, m in result['languages'].items():
        if 'error' in m:
            print(f"  {lang}: {m['error']}")
            continue
        print(f"\n  {lang}:")
        print(f"    Recall@K : {m['recall_at_k']}")
        print(f"    P@K      : {m['precision_at_k']}")
        print(f"    nDCG@K   : {m['ndcg_at_k']}")
        print(f"    MRR      : {m['mrr']}")

    rer = result.get('reranker') or {}
    if rer.get('available'):
        print("\nReranker ablation (cross-encoder on/off):")
        for variant in ('term_queries', 'context_queries'):
            block = rer[variant]
            print(f"\n  {variant}:")
            print(f"    dense   : {block['dense']}")
            print(f"    reranked: {block['reranked']}")
            print(f"    delta   : {block['delta']}")
        print(f"\n  latency: dense={rer['dense_query_latency_ms']} ms "
              f"reranked={rer['reranked_query_latency_ms']} ms "
              f"overhead={rer['rerank_overhead_ms']} ms/query")
    print(f"\nNote: {result['note']}")
