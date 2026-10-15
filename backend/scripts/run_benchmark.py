"""
Master benchmark runner — the single command that produces the scorecard.

    python scripts/run_benchmark.py                 # auto: model-free suites
    python scripts/run_benchmark.py --mock          # everything with stubs
    python scripts/run_benchmark.py --live          # everything with real models (slow)
    python scripts/run_benchmark.py --live --langs hindi,marathi --metrics bleu,chrf
    python scripts/run_benchmark.py --live --flores 100     # quality matrix on FLORES-200
    python scripts/run_benchmark.py --metrics all   # add bertscore/comet (~1-2GB downloads)
    python scripts/run_benchmark.py --skip-retrieval --output-dir evaluation/reports
    python scripts/run_benchmark.py --resume        # continue from the periodic checkpoint

Checkpointing
-------------
The run atomically writes partial results to <output-dir>/benchmark_checkpoint.json
after every section and after every matrix language. If the process dies (e.g.
OOM), re-run the SAME command with --resume to skip the work already completed
and keep going. The checkpoint is removed after a fully successful run.

Sections
--------
  functional           API/functional test matrix (mock by default)
  security             hostile-input robustness suite (mock by default)
  e2e                  full pipeline lifecycle test (mock by default)
  unicode              Unicode/script robustness (mock by default)
  quality_estimation   good/bad/partial detection by the heuristic scorer
  retrieval            Recall@K / MRR / nDCG on the terminology RAG
  translation_memory   with vs without TM (mock: stub; live: model)
  throughput           latency percentiles + concurrency stress (mock: 2ms stub)
  hallucination        degenerate-output detection (mock: good stub)
  translation          per-language quality matrix (needs --live or --mock)
  code_mixed           code-mixed Indic->English quality matrix (--live/--mock)
  backtranslation      round-trip diagnostic (needs --live or --mock)
  flores               optional FLORES-200 quality matrix (--live --flores N)

NOTE: the FAISS index files ({lang}_index.faiss) are resolved relative to
the CWD, so run this from the backend/ directory.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('run_benchmark')


def parse_args():
    p = argparse.ArgumentParser(
        description='Indic Translator — Master Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--mock', action='store_true',
                      help='Use deterministic stubs everywhere (fast pipeline test)')
    mode.add_argument('--live', action='store_true',
                      help='Use real translation models (slow; quality matrix runs)')
    p.add_argument('--langs', default='all',
                   help='Comma-separated target languages or "all" (default: all)')
    p.add_argument('--models', default=None, metavar='NAMES',
                   help='Comma-separated model names to evaluate in the quality '
                        'matrix (default: all registered, e.g. indictrans2, '
                        'nllb-200-distilled-600M, m2m100_418M). Use this to avoid '
                        'downloading heavy models like IndicTrans2.')
    p.add_argument('--metrics', default='bleu,chrf,quality',
                   help='Comma-separated metrics or "all". Default: bleu,chrf,quality '
                        '(light, no model downloads). "all" adds bertscore/comet '
                        '(downloads ~1-2GB models on first use).')
    p.add_argument('--continue-from', default=None, metavar='JSON',
                   help='Load a previous benchmark results JSON and merge this run\'s '
                        'results into it (useful for staged/continued runs). New '
                        'sections overwrite; per-language results are merged.')
    p.add_argument('--resume', action='store_true',
                   help='Resume from the periodic checkpoint written to '
                        '--output-dir (see module docstring). Sections and '
                        'matrix languages already completed in the checkpoint '
                        'are skipped. Use the same flags as the crashed run.')
    p.add_argument('--flores', type=int, default=None, metavar='N',
                   help='Also run the translation quality matrix on N FLORES-200 '
                        'sentences per language (requires --live and internet)')
    p.add_argument('--output-dir', default='evaluation/reports',
                   help='Directory for report files (default: evaluation/reports)')
    p.add_argument('--max-sentences', type=int, default=None, metavar='N',
                   help='Cap the quality matrix at N sentences per language '
                        '(handy for quick runs and ablations)')
    p.add_argument('--run-latency', action='store_true',
                   help='Run latency benchmarks in the translation matrix (live mode)')
    p.add_argument('--bertscore', action='store_true',
                   help='Also compute BERTScore in back-translation (downloads a '
                        '~570MB model on first use)')
    p.add_argument('--skip-functional', action='store_true')
    p.add_argument('--skip-security', action='store_true',
                   help='Skip the hostile-input security/robustness suite')
    p.add_argument('--skip-e2e', action='store_true',
                   help='Skip the end-to-end pipeline lifecycle test')
    p.add_argument('--skip-unicode', action='store_true')
    p.add_argument('--skip-quality', action='store_true')
    p.add_argument('--qe-model', default='heuristic',
                   choices=['heuristic', 'cometkiwi', 'both'],
                   help='Quality-estimation engine: heuristic (default, '
                        'model-free), cometkiwi (reference-free neural QE, '
                        'downloads ~2.7GB on first use), or both (head-to-head '
                        'comparison in the report).')
    p.add_argument('--skip-language', action='store_true',
                   help='Skip the language detection benchmark')
    p.add_argument('--skip-retrieval', action='store_true')
    p.add_argument('--reranker', action='store_true',
                   help='Run the cross-encoder reranker on/off ablation in the '
                        'retrieval benchmark (downloads ~430MB on first use; '
                        'only meaningful with --live, which loads the encoder)')
    p.add_argument('--passage-retrieval', action='store_true',
                   help='Run the open-vocabulary passage-retrieval benchmark '
                        '(dense vs rerank on NLLB round-trip paraphrase '
                        'queries). Requires --live (encoder + NLLB).')
    p.add_argument('--skip-tm', action='store_true')
    p.add_argument('--skip-throughput', action='store_true',
                   help='Skip the latency/throughput + concurrency benchmark')
    p.add_argument('--skip-hallucination', action='store_true',
                   help='Skip the hallucination/degenerate-output benchmark')
    p.add_argument('--skip-translation', action='store_true')
    p.add_argument('--skip-code-mixed', action='store_true',
                   help='Skip the code-mixed Indic->English quality matrix')
    p.add_argument('--skip-backtranslation', action='store_true')
    p.add_argument('--num-beams', type=int, default=2,
                   help='Beam size passed to model.generate() for live models '
                        '(default 2 — latency/quality sweet spot on CPU; '
                        'beam=4 ~2x slower with no BLEU gain). Use this for '
                        'num_beams ablations.')
    p.add_argument('--keep-metrics-loaded', action='store_true',
                   help='Keep COMET/BERTScore models resident between matrix '
                        'languages instead of unloading after each metric '
                        '(faster — saves ~10min of reloads — but peak memory '
                        'grows by several GB; only use on machines with '
                        'plenty of RAM).')
    return p.parse_args()


def parse_langs(args):
    if args.langs.lower() == 'all':
        return None
    return [l.strip().lower() for l in args.langs.split(',')]


def parse_metrics(args):
    """
    Resolve metric selection.

    'all' -> None (Evaluator uses every registered metric, including heavy
    bertscore/comet — opt-in because they download ~1-2GB of weights).
    """
    if args.metrics.lower() == 'all':
        return None
    return [m.strip().lower() for m in args.metrics.split(',')]


# ---------------------------------------------------------------------------
# Checkpointing (partial-result persistence across crashes)
# ---------------------------------------------------------------------------

CHECKPOINT_FILENAME = 'benchmark_checkpoint.json'


def checkpoint_path(args) -> str:
    """Path of the periodic checkpoint file for this run."""
    return os.path.join(args.output_dir, CHECKPOINT_FILENAME)


def _normalized_command() -> str:
    """The CLI invocation without --resume, for checkpoint mismatch checks."""
    return ' '.join(a for a in sys.argv[1:] if a != '--resume')


def has_per_language_completion(completed, prefix):
    """
    Whether the checkpoint holds per-language completions under a prefix.

    The live translation matrix records 'translation.<lang>' keys while the
    auto/mock modes record a single 'translation' section key. Resuming across
    modes must not confuse the two, so marker-writing branches check this
    before overwriting real per-language results.
    """
    return any(k.startswith(f'{prefix}.') for k in completed)


def save_checkpoint(results, completed, args, label=''):
    """
    Atomically persist partial results to the checkpoint file.

    Called after every section and after every matrix language. Writes to a
    temp file then os.replace()s it into place, so a kill mid-write (OOM etc.)
    can never corrupt the previous checkpoint.
    """
    path = checkpoint_path(args)
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        payload = {
            'checkpoint_version': 1,
            'saved_at': datetime.now(timezone.utc).isoformat(),
            'command': _normalized_command(),
            'completed': sorted(completed),
            'results': results,
        }
        tmp = f'{path}.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp, path)
        logger.info(f'Checkpoint saved ({label or "section"}) — '
                    f'{len(completed)} completed -> {path}')
    except Exception as e:
        logger.warning(f'Checkpoint save failed ({label}): {e}')


def load_checkpoint(args):
    """
    Load a prior checkpoint for --resume.

    Returns:
        (results, completed_set) on success, or (None, set()) when the
        checkpoint is missing, from another format version, or unreadable
        (the caller then starts fresh).
    """
    path = checkpoint_path(args)
    if not os.path.exists(path):
        logger.info(f'No checkpoint found at {path} — starting fresh')
        return None, set()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            cp = json.load(f)
        if cp.get('checkpoint_version') != 1:
            raise ValueError(f'unsupported checkpoint version '
                             f'{cp.get("checkpoint_version")}')
        results = cp.get('results')
        completed = set(cp.get('completed', []))
        if not isinstance(results, dict):
            raise ValueError('checkpoint missing a results dict')
        # Warn when the resume invocation differs from the crashed run —
        # completed sections keep their checkpoint data even if the flags now
        # ask for something else (e.g. --metrics all).
        stored_cmd = cp.get('command')
        current_cmd = _normalized_command()
        if stored_cmd and stored_cmd != current_cmd:
            logger.warning('Resume flags differ from the checkpoint run — '
                           'completed sections keep their checkpoint data.\n'
                           f'  checkpoint run: {stored_cmd}\n'
                           f'  now           : {current_cmd}')
        shown = ', '.join(sorted(completed)[:10])
        more = f' (+{len(completed) - 10} more)' if len(completed) > 10 else ''
        logger.info(f'Resumed from checkpoint: {len(completed)} completed '
                    f'({shown}{more}) -> {path}')
        return results, completed
    except Exception as e:
        logger.warning(f'Could not load checkpoint {path} ({e}) — starting fresh')
        return None, set()


# ---------------------------------------------------------------------------
# Model function factories
# ---------------------------------------------------------------------------

def make_live_model_fns(target_lang, registry, model_names=None, num_beams=2,
                        source_lang='english'):
    """
    Translate functions pinned to specific registry models (no TM).

    Models whose supported_languages do not cover the source->target pair are
    excluded. Without this, a model that simply cannot translate a language
    (e.g. m2m100 -> telugu, which is not in M2M100's 100 languages) would be
    run anyway and fail on every sentence, polluting the matrix with errors.
    """

    def make(name):
        def fn(text, source_lang=source_lang, target_lang=target_lang):
            result = registry.translate(
                text=text, source_lang=source_lang, target_lang=target_lang,
                model=name, use_tm=False, num_beams=num_beams,
            )
            return result.to_dict()
        return fn

    model_names = model_names or ['indictrans2', 'nllb-200-distilled-600M', 'm2m100_418M']
    fns = {}
    for name in model_names:
        model = registry.get_model(name)
        if not (model and model.available):
            logger.warning(f"Model '{name}' not available — skipped")
            continue
        codes = model.supported_languages
        if source_lang not in codes or target_lang not in codes:
            logger.info(f"Model '{name}' does not support "
                        f"{source_lang}->{target_lang} — skipped")
            continue
        fns[name] = make(name)
    return fns


def make_mock_model_fns(target_lang):
    """Echo stubs so the whole matrix runs without any model."""

    def make():
        def fn(text, source_lang='english', target_lang=target_lang):
            return {
                'original': text,
                'translated': f'[{source_lang}->{target_lang}] {text}',
                'source_lang': source_lang,
                'target_lang': target_lang,
                'error': None,
                'metadata': {'model': 'mock-echo', 'error': None},
            }
        return fn

    return {'mock-echo': make()}


def make_mock_text_fn():
    def fn(text, source_lang='english', target_lang='hindi'):
        return f'[{source_lang}->{target_lang}] {text}'
    return fn


def make_live_text_fn(registry, model_name=None, num_beams=2):
    """
    Generic live translate function pinned to one registry model (no TM),
    used by throughput and hallucination benchmarks.
    """
    model_name = model_name or 'nllb-200-distilled-600M'

    def fn(text, source_lang='english', target_lang='hindi'):
        return registry.translate(
            text=text, source_lang=source_lang, target_lang=target_lang,
            model=model_name, use_tm=False, num_beams=num_beams,
        ).to_dict()
    return fn


def run_code_mixed_matrix(model_fns, items, metrics, run_latency=False,
                          model_fns_factory=None, unload_neural_after_each=True):
    """
    Run the code-mixed quality matrix: each Indic source language's code-mixed
    sentences are translated to English and scored against the English
    references (BLEU / chrF++ / quality).

    ``model_fns_factory(lang)`` (optional) returns the per-language model
    functions, letting callers filter out models that don't support the
    source language (e.g. m2m100 for telugu code-mixed input). When omitted,
    the same ``model_fns`` dict is used for every language.

    ``unload_neural_after_each``: forwarded to BenchmarkRunner (see
    --keep-metrics-loaded).
    """
    from app.evaluation.benchmark import BenchmarkRunner

    # Group items by source language (each sentence has its own source)
    by_lang = {}
    for item in items:
        by_lang.setdefault(item['source_lang'], []).append(item)

    runner = BenchmarkRunner(metrics=metrics,
                             unload_neural_after_each=unload_neural_after_each)
    results = {}
    for lang, group in sorted(by_lang.items()):
        fns = model_fns_factory(lang) if model_fns_factory else model_fns
        if not fns:
            logger.warning(f'No available models support code-mixed '
                           f'{lang} -> english — skipped')
            continue
        sources = [i['source'] for i in group]
        references = {'english': [i['reference'] for i in group]}
        logger.info(f"Code-mixed quality: {lang} -> english "
                    f"({len(sources)} sentences, {list(fns.keys())})")
        results[lang] = runner.run(
            model_fns=fns,
            sources=sources,
            references=references,
            source_lang=lang,
            target_lang='english',
            run_latency=run_latency,
            warmup_runs=0,
        )
    return results


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def run_translation_matrix(model_fns, target_langs, metrics, dataset,
                           run_latency=False, unload_neural_after_each=True):
    """
    Run the per-language quality matrix (curated or FLORES dataset).

    ``unload_neural_after_each``: forwarded to BenchmarkRunner (see
    --keep-metrics-loaded).
    """
    from app.evaluation.benchmark import BenchmarkRunner

    sources = dataset['sources']
    references = dataset['references']
    runner = BenchmarkRunner(metrics=metrics,
                             unload_neural_after_each=unload_neural_after_each)
    results = {}
    for lang in target_langs:
        if lang not in references or not references.get(lang):
            logger.warning(f'No references for {lang}, skipping')
            continue
        logger.info(f"Quality matrix: en -> {lang} "
                    f"({len(sources)} sentences, {list(model_fns.keys())})")
        results[lang] = runner.run(
            model_fns=model_fns,
            sources=sources,
            references=references,
            source_lang='english',
            target_lang=lang,
            domains=dataset.get('domains'),
            run_latency=run_latency,
            warmup_runs=1,
        )
    return results


def main():
    # Windows consoles default to cp1252, which cannot print characters like
    # '→' (U+2192) used in the report. Force UTF-8 output.
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    args = parse_args()
    model_names = (
        [m.strip() for m in args.models.split(',') if m.strip()]
        if args.models else None
    )

    # Mode semantics:
    #   auto (default) — run every model-free suite; skip the model-based
    #                    quality matrix and back-translation (they need real
    #                    models and would otherwise produce misleading numbers)
    #   --mock          — run EVERYTHING with deterministic stubs (pipeline test)
    #   --live          — run EVERYTHING with the real translation models
    if args.mock:
        mode = 'mock'
    elif args.live:
        mode = 'live'
    else:
        mode = 'auto'
    use_mock = mode != 'live'
    run_model_matrix = mode in ('mock', 'live')

    results = {'metadata': {'mode': mode}}
    completed = set()
    if args.resume:
        loaded, completed = load_checkpoint(args)
        if loaded is not None:
            results = loaded
            results.setdefault('metadata', {})['mode'] = mode
    start = time.time()

    # ------------------------------------------------------------------
    # 1. Functional
    # ------------------------------------------------------------------
    if not args.skip_functional and 'functional' not in completed:
        from app.evaluation.functional import FunctionalTestSuite
        logger.info('Running functional/API contract tests...')
        results['functional'] = FunctionalTestSuite(use_mock=use_mock).run()
        completed.add('functional')
        save_checkpoint(results, completed, args, 'functional')

    # ------------------------------------------------------------------
    # 1b. Security & robustness (hostile inputs)
    # ------------------------------------------------------------------
    if not args.skip_security and 'security' not in completed:
        from app.evaluation.security import SecurityTestSuite
        logger.info('Running security/robustness test suite...')
        results['security'] = SecurityTestSuite(use_mock=use_mock).run()
        completed.add('security')
        save_checkpoint(results, completed, args, 'security')

    # ------------------------------------------------------------------
    # 1c. End-to-end pipeline lifecycle
    # ------------------------------------------------------------------
    if not args.skip_e2e and 'e2e' not in completed:
        from app.evaluation.e2e import EndToEndTest
        logger.info('Running end-to-end pipeline lifecycle test...')
        results['e2e'] = EndToEndTest(use_mock=use_mock).run()
        completed.add('e2e')
        save_checkpoint(results, completed, args, 'e2e')

    # ------------------------------------------------------------------
    # 2. Unicode
    # ------------------------------------------------------------------
    if not args.skip_unicode and 'unicode' not in completed:
        from app.evaluation.functional import EchoTranslator
        from app.evaluation.unicode import UnicodeTestSuite
        logger.info('Running Unicode/script tests...')
        stub = lambda text, source_lang, target_lang: (  # noqa: E731
            EchoTranslator().translate(text, source_lang, target_lang)['translated']
        )
        results['unicode'] = UnicodeTestSuite().run(translate_fn=stub)
        completed.add('unicode')
        save_checkpoint(results, completed, args, 'unicode')

    # ------------------------------------------------------------------
    # 3. Quality estimation
    # ------------------------------------------------------------------
    if not args.skip_quality and 'quality_estimation' not in completed:
        from app.evaluation.quality_estimation import QualityEstimationBenchmark
        logger.info(f'Running quality estimation validation ({args.qe_model})...')
        results['quality_estimation'] = QualityEstimationBenchmark(
            scorer=args.qe_model).run()
        completed.add('quality_estimation')
        save_checkpoint(results, completed, args, 'quality_estimation')

    # ------------------------------------------------------------------
    # 3b. Language detection
    # ------------------------------------------------------------------
    if not args.skip_language and 'language_detection' not in completed:
        from app.evaluation.language_detection import (
            run_language_detection_benchmark,
        )
        logger.info('Running language detection benchmark...')
        results['language_detection'] = run_language_detection_benchmark()
        completed.add('language_detection')
        save_checkpoint(results, completed, args, 'language_detection')

    # ------------------------------------------------------------------
    # 4. Retrieval
    # ------------------------------------------------------------------
    if not args.skip_retrieval and 'retrieval' not in completed:
        from app.evaluation.retrieval import run_retrieval_benchmark
        logger.info('Running terminology RAG retrieval benchmark...')
        # Encoder only in live mode: keeps default runs fast and offline.
        # The reranker ablation is opt-in (--reranker) — it downloads the
        # cross-encoder model on first use.
        results['retrieval'] = run_retrieval_benchmark(
            langs=parse_langs(args), use_encoder=not use_mock,
            use_reranker=args.reranker,
        )
        completed.add('retrieval')
        save_checkpoint(results, completed, args, 'retrieval')

    # ------------------------------------------------------------------
    # 4b. Passage retrieval (open-vocabulary, rerank ablation)
    # ------------------------------------------------------------------
    if args.passage_retrieval and 'passage_retrieval' not in completed:
        from app.evaluation.passage_retrieval import (
            run_passage_retrieval_benchmark,
        )
        logger.info('Running open-vocabulary passage retrieval '
                    '(NLLB round-trip queries, rerank ablation)...')
        if not args.live:
            results['passage_retrieval'] = {
                'error': 'Run with --live (loads the encoder + NLLB for '
                         'round-trip paraphrase queries).',
            }
            logger.warning('Passage retrieval skipped in auto mode '
                           '(use --live --passage-retrieval)')
        else:
            results['passage_retrieval'] = run_passage_retrieval_benchmark(
                use_encoder=True, use_reranker=args.reranker,
                queries_mode='roundtrip', flores=False,
            )
        completed.add('passage_retrieval')
        save_checkpoint(results, completed, args, 'passage_retrieval')

    # ------------------------------------------------------------------
    # 5. Translation memory
    # ------------------------------------------------------------------
    if not args.skip_tm and 'translation_memory' not in completed:
        from app.evaluation.tm_benchmark import run_tm_benchmark
        logger.info('Running translation memory benchmark...')
        if use_mock:
            import time as _time
            def stub(text, source_lang, target_lang):
                _time.sleep(0.005)
                return f'[{source_lang}->{target_lang}] {text}'
            results['translation_memory'] = run_tm_benchmark(
                stub, enable_encoder=False,
            )
        else:
            results['translation_memory'] = run_tm_benchmark(enable_encoder=True)
        completed.add('translation_memory')
        save_checkpoint(results, completed, args, 'translation_memory')

    # ------------------------------------------------------------------
    # 5b. Throughput & concurrency (latency percentiles + stress)
    # ------------------------------------------------------------------
    if not args.skip_throughput and 'throughput' not in completed:
        from app.evaluation.throughput import ThroughputBenchmark
        logger.info('Running throughput & concurrency benchmark...')
        if use_mock:
            # 2ms stub — measures harness overhead, not model inference
            results['throughput'] = ThroughputBenchmark().run()
        else:
            from app.services.model_registry import ModelRegistry
            registry = ModelRegistry.get_instance()
            live_fn = make_live_text_fn(registry, num_beams=args.num_beams)
            try:
                results['throughput'] = ThroughputBenchmark(
                    translate_fn=live_fn).run(rounds=1, warmup=1)
            except Exception as e:
                logger.warning(f'Throughput benchmark failed: {e}')
                results['throughput'] = {'status': 'skipped', 'reason': str(e)}
        completed.add('throughput')
        save_checkpoint(results, completed, args, 'throughput')

    # ------------------------------------------------------------------
    # 5c. Hallucination / degenerate-output detection
    # ------------------------------------------------------------------
    if not args.skip_hallucination and 'hallucination' not in completed:
        from app.evaluation.hallucination import HallucinationBenchmark
        logger.info('Running hallucination/degenerate-output benchmark...')
        if use_mock:
            # Good stub preserves entities; detector calibration still proves
            # the heuristics separate good vs degenerate outputs.
            results['hallucination'] = HallucinationBenchmark().run()
        else:
            from app.services.model_registry import ModelRegistry
            registry = ModelRegistry.get_instance()
            live_fn = make_live_text_fn(registry, num_beams=args.num_beams)
            try:
                results['hallucination'] = HallucinationBenchmark(
                    translate_fn=live_fn).run()
            except Exception as e:
                logger.warning(f'Hallucination benchmark failed: {e}')
                results['hallucination'] = {'status': 'skipped', 'reason': str(e)}
        completed.add('hallucination')
        save_checkpoint(results, completed, args, 'hallucination')

    # ------------------------------------------------------------------
    # 6. Translation quality matrix
    # ------------------------------------------------------------------
    if not args.skip_translation:
        target_langs = parse_langs(args)
        from app.evaluation.datasets import get_benchmark_dataset, REFERENCES
        if target_langs is None:
            target_langs = list(REFERENCES.keys())
        dataset = get_benchmark_dataset(target_langs=target_langs)

        if args.max_sentences:
            dataset['sources'] = dataset['sources'][:args.max_sentences]
            dataset['references'] = {
                k: v[:args.max_sentences] for k, v in dataset['references'].items()
            }
            if dataset.get('domains'):
                dataset['domains'] = dataset['domains'][:args.max_sentences]
            logger.info(f'Quality matrix capped at {args.max_sentences} sentences/language')

        if not run_model_matrix:
            # Auto mode: record the skip marker once; a resumed run keeps it.
            # Guard: never overwrite real per-language matrix results from a
            # live checkpoint resumed in a different mode.
            if ('translation' not in completed
                    and not has_per_language_completion(completed, 'translation')):
                results['translation'] = {
                    'status': 'skipped',
                    'reason': 'Run with --live (real models) or --mock (stub pipeline test) '
                              'to execute the quality matrix.',
                }
                logger.info('Quality matrix skipped in auto mode (use --live or --mock)')
                completed.add('translation')
                save_checkpoint(results, completed, args, 'translation')
        elif use_mock:
            if ('translation' not in completed
                    and not has_per_language_completion(completed, 'translation')):
                logger.info('Quality matrix (mock echo — pipeline test only, numbers are NOT real)')
                model_fns = make_mock_model_fns('hindi')
                results['translation'] = run_translation_matrix(
                    model_fns, target_langs, parse_metrics(args), dataset,
                    run_latency=False,
                    unload_neural_after_each=not args.keep_metrics_loaded,
                )
                completed.add('translation')
                save_checkpoint(results, completed, args, 'translation')
        else:
            from app.services.model_registry import ModelRegistry
            registry = ModelRegistry.get_instance()
            # True if any language already completed (resume) or will run now.
            any_model = any(f'translation.{lang}' in completed for lang in target_langs)
            for lang in target_langs:
                key = f'translation.{lang}'
                if key in completed:
                    logger.info(f'[resume] {key} already completed — skipping')
                    continue
                fns = make_live_model_fns(lang, registry, model_names=model_names,
                                          num_beams=args.num_beams)
                if fns:
                    any_model = True
                    results.setdefault('translation', {})[lang] = run_translation_matrix(
                        fns, [lang], parse_metrics(args), dataset,
                        run_latency=args.run_latency,
                        unload_neural_after_each=not args.keep_metrics_loaded,
                    )[lang]
                    completed.add(key)
                    save_checkpoint(results, completed, args, key)
            if not any_model:
                logger.warning('No translation models available — quality matrix skipped. '
                               'Run with --live once models are installed.')

    # ------------------------------------------------------------------
    # 6b. Code-mixed quality matrix (Indic script + English words -> English)
    # ------------------------------------------------------------------
    if not args.skip_code_mixed:
        from app.evaluation.datasets import get_code_mixed_dataset
        if not run_model_matrix:
            if 'code_mixed' not in completed:
                results['code_mixed'] = {
                    'status': 'skipped',
                    'reason': 'Run with --live (real models) or --mock (stub pipeline test) '
                              'to execute the code-mixed quality matrix.',
                }
                completed.add('code_mixed')
                save_checkpoint(results, completed, args, 'code_mixed')
        elif 'code_mixed' not in completed:
            cm_data = get_code_mixed_dataset()
            logger.info(f'Running code-mixed quality matrix '
                        f'({cm_data["metadata"]["num_sentences"]} sentences, '
                        f'{len(cm_data["metadata"]["languages"])} languages)...')
            if use_mock:
                model_fns = make_mock_model_fns('english')
                results['code_mixed'] = run_code_mixed_matrix(
                    model_fns, cm_data['items'], parse_metrics(args),
                    unload_neural_after_each=not args.keep_metrics_loaded,
                )
            else:
                from app.services.model_registry import ModelRegistry
                registry = ModelRegistry.get_instance()
                model_fns = make_live_model_fns(
                    'english', registry, model_names=model_names,
                    num_beams=args.num_beams,
                )
                if model_fns:
                    # Build per-source-language fns so models that cannot
                    # read a given Indic script (e.g. m2m100 for telugu) are
                    # excluded from that language's code-mixed matrix.
                    results['code_mixed'] = run_code_mixed_matrix(
                        model_fns, cm_data['items'], parse_metrics(args),
                        run_latency=args.run_latency,
                        model_fns_factory=lambda lang: make_live_model_fns(
                            'english', registry, model_names=model_names,
                            num_beams=args.num_beams, source_lang=lang,
                        ),
                        unload_neural_after_each=not args.keep_metrics_loaded,
                    )
                else:
                    results['code_mixed'] = {
                        'status': 'skipped',
                        'reason': 'No translation models available.',
                    }
            completed.add('code_mixed')
            save_checkpoint(results, completed, args, 'code_mixed')

    # ------------------------------------------------------------------
    # 7. Back-translation
    # ------------------------------------------------------------------
    if not args.skip_backtranslation and 'backtranslation' not in completed:
        from app.evaluation.backtranslation import run_backtranslation_test
        logger.info('Running back-translation diagnostic...')
        target = 'hindi'
        # BERTScore only when explicitly requested (heavy download otherwise)
        if not run_model_matrix:
            results['backtranslation'] = {
                'status': 'skipped',
                'reason': 'Run with --live (real models) or --mock (stub) to execute '
                          'the back-translation diagnostic.',
            }
        elif use_mock:
            results['backtranslation'] = run_backtranslation_test(
                make_mock_text_fn(), use_bertscore=False,
            )
        else:
            results['backtranslation'] = run_backtranslation_test(
                target_lang=target, use_bertscore=args.bertscore,
            )
        completed.add('backtranslation')
        save_checkpoint(results, completed, args, 'backtranslation')

    # ------------------------------------------------------------------
    # 8. FLORES-200 (optional, live only)
    # ------------------------------------------------------------------
    if args.flores:
        from app.evaluation.datasets import load_flores_dataset
        logger.info(f'Loading FLORES-200 (max {args.flores} sentences/lang)...')
        target_langs = parse_langs(args)
        flores = load_flores_dataset(
            target_langs=target_langs, split='devtest', max_sentences=args.flores,
        )
        if 'error' in flores:
            # A load failure is NOT "completed" — a resume retries it (the
            # failure may be transient, e.g. network).
            logger.warning(f'FLORES-200 load failed: {flores["error"]}')
            results['flores'] = flores
            save_checkpoint(results, completed, args, 'flores')
        else:
            results['flores'] = {
                'status': 'loaded',
                'num_sentences': flores['metadata']['num_sentences'],
                'target_langs': flores['metadata']['target_langs'],
            }
            if not use_mock:
                from app.services.model_registry import ModelRegistry
                registry = ModelRegistry.get_instance()
                for lang in flores['metadata']['target_langs']:
                    key = f'flores_translation.{lang}'
                    if key in completed:
                        logger.info(f'[resume] {key} already completed — skipping')
                        continue
                    fns = make_live_model_fns(lang, registry, model_names=model_names,
                                              num_beams=args.num_beams)
                    if not fns:
                        continue
                    results.setdefault('flores_translation', {})[lang] = \
                        run_translation_matrix(
                            fns, [lang], parse_metrics(args), flores,
                            run_latency=args.run_latency,
                            unload_neural_after_each=not args.keep_metrics_loaded,
                        )[lang]
                    completed.add(key)
                    save_checkpoint(results, completed, args, key)
            completed.add('flores')
            save_checkpoint(results, completed, args, 'flores')

    # ------------------------------------------------------------------
    # Report + scorecard
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Merge with a previous run (staged/continued benchmarks)
    # ------------------------------------------------------------------
    if args.continue_from and os.path.exists(args.continue_from):
        try:
            from app.evaluation.report import deep_merge
            with open(args.continue_from, 'r', encoding='utf-8') as f:
                previous = json.load(f)
            results = deep_merge(previous, results)
            logger.info(f'Merged this run into {args.continue_from} '
                        f'({len(previous)} sections loaded)')
        except Exception as e:
            logger.warning(f'Could not merge {args.continue_from}: {e}')

    # Cumulative across resume segments (fresh runs have no prior total).
    results['metadata']['total_time_sec'] = round(
        (results['metadata'].get('total_time_sec') or 0) + (time.time() - start), 2)
    results['metadata']['num_beams'] = args.num_beams
    results['metadata']['targets'] = {
        'functional_pass_rate': 99.0,
        'security_pass_rate': 99.0,
        'e2e_pass_rate': 100.0,
        'unicode_pass_rate': 100.0,
        'retrieval_recall_at_5': 90.0,
        'tm_hit_p95_ms': 300.0,
        'translation_p95_ms': 3000.0,
        'quality_estimation_f1': 0.95,
        'hallucination_rate_pct': 5.0,
    }

    from app.evaluation.report import build_scorecard, render_markdown, save_reports
    scorecard = build_scorecard(results)

    print()
    print(render_markdown(scorecard, results))
    print(f"\nTotal time: {results['metadata']['total_time_sec']}s")

    paths = save_reports(results, args.output_dir)
    print(f"\nReports saved:")
    for label, path in paths.items():
        print(f"  {label:<14}: {path}")

    # Console summary line
    s = scorecard['summary']
    print(f"\nScorecard: {s['pass']} PASS / {s['fail']} FAIL / "
          f"{s['not_measured']} not measured")
    if args.live:
        print("Remember: report ACTUAL measured values against baselines — "
              "targets are engineering goals, not results.")

    # A fully successful run already wrote the timestamped report, so the
    # transient checkpoint is no longer needed (avoids stale resumes).
    cp = checkpoint_path(args)
    if os.path.exists(cp):
        try:
            os.remove(cp)
            logger.info(f'Checkpoint removed after successful run: {cp}')
        except Exception as e:
            logger.warning(f'Could not remove checkpoint {cp}: {e}')


if __name__ == '__main__':
    main()
