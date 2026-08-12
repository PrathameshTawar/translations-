"""
Translation Memory benchmark — with vs without.

Implements the plan's "Benchmark #11 — Benchmark Translation Memory":

  * model latency      — translating with the model directly (no TM)
  * TM-hit latency     — latency when the TM answers from cache
  * exact hit rate     — identical sentences served from cache
  * semantic hit rate  — paraphrases served from cache (needs encoder)
  * latency reduction  % — (model - tm-hit) / model

Uses an *isolated* TranslationMemory instance (never the app singleton) so
the user's live translation memory is never modified. Ground truth is built
by priming the TM with the model's own translations.

Usage (from translator-backend/):
    python -m app.evaluation.tm_benchmark --mock
    python -m app.evaluation.tm_benchmark            # live model
"""

import logging
import os
import statistics
import tempfile
import time

logger = logging.getLogger(__name__)

# 8 representative sentences per language pair (keep the benchmark quick)
DEFAULT_SENTENCES = [
    "The weather is beautiful today and I want to go for a walk in the park.",
    "Please tell me where I can find the nearest hospital.",
    "I would like to order two cups of tea and some snacks.",
    "The server processes data using a neural network architecture.",
    "The patient should take this medicine twice a day after meals.",
    "The bank offers a low interest rate on education loans.",
    "Students must submit their assignments before the end of this week.",
    "Could you please send me the document tomorrow?",
]

# Near-duplicates of the sentences above, used for the semantic-hit check
PARAPHRASE_VARIANTS = [
    "The weather is lovely today and I want to stroll in the park.",
    "Could you kindly tell me where the nearest hospital is?",
    "I would like two cups of tea and some snacks, please.",
    "The server uses a neural network to process data.",
    "The patient should take this medicine twice daily after meals.",
    "The bank gives a low interest rate on student loans.",
    "Students need to submit their assignments before this week ends.",
    "Could you please send the document to me tomorrow?",
]


def _pct(values):
    values = [v for v in values if v > 0]
    if len(values) < 2:
        return {'p50_ms': round(statistics.median(values), 2) if values else 0.0,
                'p95_ms': round(values[-1], 2) if values else 0.0}
    s = sorted(values)
    return {
        'p50_ms': round(statistics.median(s), 2),
        'p95_ms': round(s[int(len(s) * 0.95)], 2),
    }


class TranslationMemoryBenchmark:
    """Compare model-only translation against TM-accelerated translation."""

    def run(self, model_fn, sentences=None, source_lang='english',
            target_lang='hindi', paraphrases=None, threshold=0.92,
            enable_encoder=True) -> dict:
        """
        Run the TM benchmark.

        Args:
            model_fn: callable (text, source_lang, target_lang) -> str.
                      The translation model (or a stub for tests).
            sentences: Sentences to translate (default: DEFAULT_SENTENCES).
            source_lang / target_lang: Language pair.
            paraphrases: Paraphrase variants for the semantic-hit check.
            threshold: TM similarity threshold.
            enable_encoder: Load the sentence-transformer encoder for the
                      semantic-hit phase. Set False for fast/offline runs
                      (only exact-match hits are then measured).

        Returns:
            dict with hit rates, latency stats and latency reduction.
        """
        from app.services.translation_memory import TranslationMemory

        sentences = sentences or DEFAULT_SENTENCES
        paraphrases = paraphrases or PARAPHRASE_VARIANTS

        # Isolated TM instance — point persistence at a throwaway temp file
        tmp_path = os.path.join(tempfile.gettempdir(), 'tm_benchmark.json')
        old_env = os.environ.get('TM_DATA_FILE')
        os.environ['TM_DATA_FILE'] = tmp_path
        try:
            tm = TranslationMemory(similarity_threshold=threshold)
            if not enable_encoder:
                # Block encoder loading: keeps runs fast, offline and deterministic
                tm._encoder_failed = True
            tm.clear()
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

            # ---- Phase 1: model-only baseline ----------------------------
            model_latencies = []
            for s in sentences:
                t0 = time.time()
                model_fn(s, source_lang=source_lang, target_lang=target_lang)
                model_latencies.append((time.time() - t0) * 1000)

            # ---- Phase 2: prime the TM -----------------------------------
            prime_latencies = []
            for s in sentences:
                t0 = time.time()
                entry = tm.lookup(s, source_lang, target_lang)
                if entry is None:
                    out = model_fn(s, source_lang=source_lang, target_lang=target_lang)
                    tm.store(s, out, source_lang, target_lang, model='benchmark')
                prime_latencies.append((time.time() - t0) * 1000)

            # ---- Phase 3: exact-match hits -------------------------------
            exact_hits = 0
            exact_latencies = []
            for s in sentences:
                t0 = time.time()
                entry = tm.lookup(s, source_lang, target_lang)
                exact_latencies.append((time.time() - t0) * 1000)
                if entry is not None:
                    exact_hits += 1

            # ---- Phase 4: semantic (paraphrase) hits ---------------------
            semantic_hits = 0
            semantic_latencies = []
            for p in paraphrases[:len(sentences)]:
                t0 = time.time()
                entry = tm.lookup(p, source_lang, target_lang)
                semantic_latencies.append((time.time() - t0) * 1000)
                if entry is not None:
                    semantic_hits += 1

            stats = tm.get_stats()
            n = max(len(sentences), 1)

            model_lat = _pct(model_latencies)
            tm_lat = _pct(exact_latencies)
            mean_model = statistics.mean([v for v in model_latencies if v > 0]) \
                if any(v > 0 for v in model_latencies) else 0.0
            mean_tm = statistics.mean(exact_latencies) if exact_latencies else 0.0

            return {
                'lang_pair': f'{source_lang}->{target_lang}',
                'num_sentences': n,
                'model': {
                    'mean_ms': round(mean_model, 2),
                    'percentiles': model_lat,
                },
                'tm_hit': {
                    'mean_ms': round(mean_tm, 2),
                    'percentiles': tm_lat,
                },
                'exact_hit_rate': round(exact_hits / n, 4),
                'semantic_hit_rate': round(
                    semantic_hits / max(len(semantic_latencies), 1), 4),
                'latency_reduction_pct': round(
                    (mean_model - mean_tm) / max(mean_model, 1e-9) * 100, 2),
                'encoder_available': bool(stats.get('encoder_loaded')),
                'tm_stats': {
                    'hits': stats.get('hits'),
                    'misses': stats.get('misses'),
                    'total_entries': stats.get('total_entries'),
                    'similarity_threshold': stats.get('similarity_threshold'),
                },
                'note': (
                    'Semantic hits require the sentence-transformer encoder '
                    '(downloads on first use). Without it, only exact matches '
                    'are served from the TM.'
                ),
            }
        finally:
            if old_env is None:
                os.environ.pop('TM_DATA_FILE', None)
            else:
                os.environ['TM_DATA_FILE'] = old_env


def run_tm_benchmark(model_fn=None, **kwargs) -> dict:
    """
    Module-level entry point.

    With no model_fn, uses the real ModelRegistry (live model). Pass a stub
    for fast/deterministic runs (tests, CI).
    """
    if model_fn is None:
        from app.services.model_registry import ModelRegistry

        registry = ModelRegistry.get_instance()

        def live_model_fn(text, source_lang, target_lang):
            result = registry.translate(
                text=text, source_lang=source_lang, target_lang=target_lang,
                use_tm=False,
            )
            return result.translated

        model_fn = live_model_fn

    return TranslationMemoryBenchmark().run(model_fn, **kwargs)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Translation Memory benchmark')
    parser.add_argument('--mock', action='store_true',
                        help='Use a fast echo stub instead of the real model')
    parser.add_argument('--source', default='english')
    parser.add_argument('--target', default='hindi')
    args = parser.parse_args()

    def stub(text, source_lang, target_lang):
        time.sleep(0.005)
        return f'[{source_lang}->{target_lang}] {text}'

    fn = stub if args.mock else None
    result = run_tm_benchmark(fn, source_lang=args.source, target_lang=args.target)

    print("\nTranslation Memory Benchmark")
    print("-" * 40)
    print(f"Pair: {result['lang_pair']}  sentences: {result['num_sentences']}")
    print(f"Encoder available: {result['encoder_available']}")
    print(f"Exact hit rate   : {result['exact_hit_rate'] * 100:.1f}%")
    print(f"Semantic hit rate: {result['semantic_hit_rate'] * 100:.1f}%")
    print(f"Model latency    : {result['model']}")
    print(f"TM-hit latency   : {result['tm_hit']}")
    print(f"Latency reduction: {result['latency_reduction_pct']}%")
    print(f"TM stats         : {result['tm_stats']}")
    print(f"Note: {result['note']}")
