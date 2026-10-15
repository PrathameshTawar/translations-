"""
Retrieval metrics and the terminology-RAG retrieval benchmark.

Metrics: Recall@K, Precision@K, MRR, nDCG@K (binary relevance).

Protocol (replaces the earlier one, which queried a 15-entry glossary with
its own 15 entries and trivially scored 100%):

  * Queries come from data/eval/terminology_queries.jsonl:
      sentence    natural sentences that contain the term
      paraphrase  the concept without the term's surface form
      distractor  no glossary term (incl. near-misses such as "riverbank")
  * Retrieval metrics are computed on sentence and paraphrase queries,
    reported separately (paraphrases are where dense retrieval has to work).
  * Enforcement metrics mirror the translate route: retrieved candidates plus
    exact glossary matches, verified against the query text. On distractors
    any enforced term is a false positive.
  * The glossaries hold 15 entries. ``pad_terms`` adds unrelated entries so
    the index has a realistic size; results carry ``index_size`` and
    ``valid`` (index_size >= MIN_INDEX_SIZE). Small-index results are sanity
    checks, not evidence.

Usage (from backend/; FAISS indexes are built on demand):
    python -m app.evaluation.retrieval [--lang hindi] [--pad-index terms.txt]
    python -m app.evaluation.retrieval --no-encoder      # keyword backend
    python -m app.evaluation.retrieval --reranker        # + reranker ablation
"""

import json
import logging
import math
import os
import time

logger = logging.getLogger(__name__)

from app.services.term_index import TERMINOLOGY_DIR

QUERY_FILE = os.path.join(TERMINOLOGY_DIR, '..', 'eval', 'terminology_queries.jsonl')
MIN_INDEX_SIZE = 1000


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


def load_queries(path: str = None) -> list:
    """Load the JSONL query set ({id, type, text, relevant})."""
    path = os.path.abspath(path or QUERY_FILE)
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load_pad_terms(path: str) -> list:
    """
    Load index padding: a JSON glossary (list or dict of {term, ...}) or a
    plain text file with one English term per line.
    """
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read()
    if path.endswith('.json'):
        data = json.loads(raw)
        items = list(data.values()) if isinstance(data, dict) else data
        return [t for t in items if isinstance(t, dict) and t.get('term')]
    return [{'term': line.strip(), 'translation': f'<pad:{line.strip()}>',
             'context': '', 'category': 'padding'}
            for line in raw.splitlines() if line.strip()]


class TerminologyRetrievalBenchmark:
    """
    Recall@K / MRR / nDCG plus enforcement precision for the terminology RAG
    pipeline, on sentence / paraphrase / distractor queries.
    """

    def __init__(self, pipeline=None, use_encoder=True, query_file=None,
                 pad_terms=None):
        """
        Args:
            pipeline: Pre-built IndicRAGPipeline-like object (default: a new
                      pipeline, so padding never touches the app singleton).
            use_encoder: Load the sentence encoder (FAISS semantic search).
                         False runs the keyword-fallback backend offline.
            query_file: JSONL query set (default data/eval/...).
            pad_terms: extra glossary entries added to every language's
                       index for this run only.
        """
        self.use_encoder = use_encoder
        self.queries = load_queries(query_file)
        self.pad_terms = list(pad_terms or [])
        if pipeline is None:
            import app.services.rag_service as rag_mod
            cls = rag_mod.IndicRAGPipeline
            if not use_encoder:
                orig = cls._try_load_encoder
                cls._try_load_encoder = lambda self: None
                try:
                    pipeline = cls()
                finally:
                    cls._try_load_encoder = orig
            else:
                pipeline = cls()
        self.pipeline = pipeline

    # -- glossary / index setup -------------------------------------------

    def _prepare_lang(self, lang):
        """
        Install the (optionally padded) glossary for ``lang`` on the pipeline,
        in memory only. Returns the index size, or None when the pipeline
        does not expose its glossary (test doubles).
        """
        db = getattr(self.pipeline, 'terminology_db', None)
        if db is None:
            return None
        base = [t for t in db.get(lang, []) if t.get('category') != 'padding']
        known = {(t.get('term') or '').lower() for t in base}
        pad = [t for t in self.pad_terms if t['term'].lower() not in known]
        terms = base + pad
        db[lang] = terms
        encoder = getattr(self.pipeline, 'encoder', None)
        if encoder is not None and hasattr(self.pipeline, 'indices'):
            if pad:
                from app.services.term_index import build_index
                self.pipeline.indices[lang] = build_index(terms, encoder)
                self.pipeline.mappings[lang] = terms
            else:
                self.pipeline._load_faiss_index(lang)
        return len(terms)

    def _backend(self, langs):
        if getattr(self.pipeline, 'encoder', None) is None:
            return 'keyword-fallback'
        indices = getattr(self.pipeline, 'indices', None)
        if indices is None or any(l in indices for l in langs):
            return 'faiss-semantic'
        return 'encoder-loaded-but-no-index'

    # -- helpers ------------------------------------------------------------

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

    def _enforced_terms(self, query, retrieved, lang):
        """Terms the translate route would enforce for this query."""
        from app.services.terminology import match_terms
        lexical = getattr(self.pipeline, 'glossary_matches', None)
        candidates = list(retrieved) + (lexical(query, lang) if lexical else [])
        return {m.term for m in match_terms(query, candidates)}

    def _rank_metrics(self, ranked_lists, relevant_list, top_ks):
        n = len(ranked_lists)
        if not n:
            return {'num_queries': 0}
        rng = range(n)
        return {
            'num_queries': n,
            'recall_at_k': {k: round(self._mean(RetrievalMetrics.recall_at_k(
                ranked_lists[i], relevant_list[i], k) for i in rng), 4) for k in top_ks},
            'precision_at_k': {k: round(self._mean(RetrievalMetrics.precision_at_k(
                ranked_lists[i], relevant_list[i], k) for i in rng), 4) for k in top_ks},
            'ndcg_at_k': {k: round(self._mean(RetrievalMetrics.ndcg_at_k(
                ranked_lists[i], relevant_list[i], k) for i in rng), 4) for k in top_ks},
            'mrr': round(RetrievalMetrics.mrr(ranked_lists, relevant_list), 4),
        }

    # -- evaluation core ------------------------------------------------------

    def _evaluate_lang(self, lang, top_ks, max_queries, rerank=False):
        """Return (metrics, query_latency_ms list) for one language."""
        max_k = max(top_ks)
        queries = self.queries[:max_queries] if max_queries else self.queries
        per_type = {}
        latency_ms = []
        fp = n_distractor = 0
        enforced_hits = n_sentence = 0
        enforced_total = enforced_correct = 0
        for q in queries:
            t0 = time.time()
            # Glossaries are per target language: the English query is
            # looked up in {lang}_terms.json.
            res = self.pipeline.retrieve_context(
                q['text'], source_lang='english', target_lang=lang,
                top_k=max_k, use_reranker=rerank)
            latency_ms.append((time.time() - t0) * 1000)

            enforced = self._enforced_terms(q['text'], res.get('terms') or [], lang)
            relevant = set(q.get('relevant') or [])
            enforced_total += len(enforced)
            enforced_correct += len(enforced & relevant)
            if q['type'] == 'distractor':
                n_distractor += 1
                fp += bool(enforced)
                continue
            if q['type'] == 'sentence':
                n_sentence += 1
                enforced_hits += bool(enforced & relevant)
            bucket = per_type.setdefault(q['type'], ([], []))
            bucket[0].append(self._extract_ranked(res, max_k))
            bucket[1].append(list(relevant))

        all_ranked = [r for b in per_type.values() for r in b[0]]
        all_relevant = [r for b in per_type.values() for r in b[1]]
        metrics = self._rank_metrics(all_ranked, all_relevant, top_ks)
        metrics['by_type'] = {t: self._rank_metrics(b[0], b[1], top_ks)
                              for t, b in sorted(per_type.items())}
        metrics['enforcement'] = {
            'sentence_term_recall': round(enforced_hits / n_sentence, 4) if n_sentence else None,
            'distractor_false_positive_rate': round(fp / n_distractor, 4) if n_distractor else None,
            'enforced_precision': (round(enforced_correct / enforced_total, 4)
                                   if enforced_total else None),
            'num_distractors': n_distractor,
        }
        return metrics, latency_ms

    @staticmethod
    def _macro(languages: dict, top_ks, qtype=None) -> dict:
        """Macro-average over languages (optionally for one query type)."""
        blocks = []
        for m in languages.values():
            if 'error' in m:
                continue
            block = m['by_type'].get(qtype) if qtype else m
            if block and block.get('num_queries'):
                blocks.append(block)
        if not blocks:
            return {}
        mean = TerminologyRetrievalBenchmark._mean
        out = {}
        for k in top_ks:
            out[f'recall_at_{k}'] = mean(b['recall_at_k'][k] for b in blocks)
            out[f'ndcg_at_{k}'] = mean(b['ndcg_at_k'][k] for b in blocks)
        out['mrr'] = mean(b['mrr'] for b in blocks)
        return {k: round(v, 4) for k, v in out.items()}

    @staticmethod
    def _metric_diff(new: dict, old: dict) -> dict:
        return {k: round((new.get(k) or 0.0) - (old.get(k) or 0.0), 4) for k in new}

    @staticmethod
    def _mean_ms(values):
        return round(sum(values) / len(values), 2) if values else None

    # -- main ---------------------------------------------------------------

    def run(self, langs=None, top_ks=(1, 3, 5, 10), max_queries=None,
            use_reranker=False) -> dict:
        if langs is None:
            langs = sorted(f[:-len('_terms.json')] for f in os.listdir(TERMINOLOGY_DIR)
                           if f.endswith('_terms.json'))
        elif isinstance(langs, str):
            langs = [langs]

        index_sizes = {lang: self._prepare_lang(lang) for lang in langs}
        backend = self._backend(langs)

        languages, latency_ms = {}, []
        for lang in langs:
            if index_sizes[lang] == 0:
                languages[lang] = {'error': f'no terminology data for {lang}'}
                continue
            languages[lang], lat = self._evaluate_lang(lang, top_ks, max_queries)
            languages[lang]['index_size'] = index_sizes[lang]
            latency_ms += lat

        known_sizes = [n for n in index_sizes.values() if n]
        min_index = min(known_sizes) if known_sizes else None
        valid = min_index is not None and min_index >= MIN_INDEX_SIZE
        enforcement = [m['enforcement'] for m in languages.values() if 'error' not in m]

        def _mean_of(key):
            vals = [e[key] for e in enforcement if e.get(key) is not None]
            return round(self._mean(vals), 4) if vals else None

        result = {
            'backend': backend,
            'top_ks': list(top_ks),
            'languages': languages,
            'num_queries_total': sum(m.get('num_queries', 0) + m.get('enforcement', {})
                                     .get('num_distractors', 0)
                                     for m in languages.values()),
            'index_size': min_index,
            'valid': valid,
            'aggregate': {
                'all': self._macro(languages, top_ks),
                'sentence': self._macro(languages, top_ks, 'sentence'),
                'paraphrase': self._macro(languages, top_ks, 'paraphrase'),
                'sentence_term_recall': _mean_of('sentence_term_recall'),
                'distractor_false_positive_rate': _mean_of('distractor_false_positive_rate'),
                'enforced_precision': _mean_of('enforced_precision'),
            },
            'query_latency_ms': self._mean_ms(latency_ms),
        }
        result['note'] = (
            f'Index size {min_index} entries per language'
            + ('' if valid else
               f' (< {MIN_INDEX_SIZE}: a sanity check, not evidence of retrieval '
               'quality; pad the index with --pad-index)')
            + '. Paraphrase queries do not contain the term, so lexical '
              'backends cannot retrieve them.')

        if use_reranker:
            result['reranker'] = self._reranker_ablation(langs, top_ks, max_queries,
                                                         languages, latency_ms)
        else:
            result['reranker'] = {'available': False,
                                  'note': 'Reranker comparison not requested (--reranker).'}
        return result

    def _reranker_ablation(self, langs, top_ks, max_queries, dense_langs, dense_lat):
        from app.services.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        if not reranker.available:
            return {'available': False, 'model': reranker.model_name,
                    'error': 'cross-encoder model unavailable'}
        reranked_langs, reranked_lat = {}, []
        for lang in langs:
            if 'error' in dense_langs.get(lang, {}):
                continue
            reranked_langs[lang], lat = self._evaluate_lang(lang, top_ks, max_queries,
                                                            rerank=True)
            reranked_lat += lat
        out = {'available': True, 'model': reranker.model_name}
        for qtype in ('sentence', 'paraphrase'):
            dense = self._macro(dense_langs, top_ks, qtype)
            reranked = self._macro(reranked_langs, top_ks, qtype)
            out[f'{qtype}_queries'] = {'dense': dense, 'reranked': reranked,
                                       'delta': self._metric_diff(reranked, dense)}
        d, r = self._mean_ms(dense_lat), self._mean_ms(reranked_lat)
        out['dense_query_latency_ms'] = d
        out['reranked_query_latency_ms'] = r
        out['rerank_overhead_ms'] = round(r - d, 2) if d is not None and r is not None else None
        return out


def run_retrieval_benchmark(langs=None, top_ks=(1, 3, 5, 10), use_encoder=True,
                            use_reranker=False, pad_index=None) -> dict:
    """Module-level convenience entry point."""
    pad = load_pad_terms(pad_index) if pad_index else None
    return TerminologyRetrievalBenchmark(use_encoder=use_encoder, pad_terms=pad).run(
        langs=langs, top_ks=top_ks, use_reranker=use_reranker)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Terminology RAG retrieval benchmark')
    parser.add_argument('--lang', default=None, help='Comma-separated languages')
    parser.add_argument('--top-k', default='1,3,5,10', help='Comma-separated K values')
    parser.add_argument('--no-encoder', action='store_true',
                        help='Keyword-fallback backend only (offline, no downloads)')
    parser.add_argument('--pad-index', default=None,
                        help='Glossary JSON or term-per-line file added to the index')
    parser.add_argument('--reranker', action='store_true',
                        help='Also run the cross-encoder on/off ablation')
    parser.add_argument('--json', default=None, help='Write the full result here')
    args = parser.parse_args()

    result = run_retrieval_benchmark(
        langs=args.lang.split(',') if args.lang else None,
        top_ks=tuple(int(k) for k in args.top_k.split(',')),
        use_encoder=not args.no_encoder, use_reranker=args.reranker,
        pad_index=args.pad_index)
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    agg = result['aggregate']
    print(f"\nRetrieval benchmark (backend: {result['backend']}, "
          f"index size: {result['index_size']}, valid: {result['valid']})")
    print(f"Queries: {result['num_queries_total']}")
    for qtype in ('sentence', 'paraphrase'):
        print(f"  {qtype:<10}: {agg[qtype]}")
    print(f"  sentence term recall (enforcement): {agg['sentence_term_recall']}")
    print(f"  distractor false-positive rate     : {agg['distractor_false_positive_rate']}")
    print(f"  enforced precision                 : {agg['enforced_precision']}")
    print(f"\nNote: {result['note']}")
