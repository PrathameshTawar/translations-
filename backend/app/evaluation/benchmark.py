"""
Benchmark — batch evaluation runner for model comparison.

Runs translations through multiple models and evaluates
them against reference translations using all configured metrics.
"""

import time
import logging
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, asdict

from .evaluator import Evaluator
from .metrics import LatencyBenchmark

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Batch benchmark runner for comparing multiple translation models.

    Usage:
        runner = BenchmarkRunner()
        results = runner.run(
            model_fns={
                'nllb-200': nllb_translate_fn,
                'indic-trans2': indic_translate_fn,
            },
            sources=['Hello world', 'How are you?'],
            references=[['नमस्ते दुनिया'], ['आप कैसे हैं?']],
            target_lang='hindi',
        )
    """

    def __init__(self, metrics: Optional[List[str]] = None,
                 unload_neural_after_each: bool = True):
        """
        Args:
            metrics: Metrics to compute. Defaults to all available.
            unload_neural_after_each: Whether the evaluator releases
                COMET/BERTScore models right after each scoring call (default
                True) so matrix runs stay within memory on CPU machines.
                Set False to keep them resident for speed.
        """
        self.evaluator = Evaluator(metrics=metrics,
                                   unload_neural_after_each=unload_neural_after_each)
        self.latency_bench = LatencyBenchmark()

    def run(
        self,
        model_fns: Dict[str, Callable],
        sources: List[str],
        references: Dict[str, List[str]],
        source_lang: str = 'english',
        target_lang: str = 'hindi',
        domains: Optional[List[str]] = None,
        run_latency: bool = True,
        warmup_runs: int = 2,
    ) -> Dict:
        """
        Run benchmark comparing multiple translation models.

        Args:
            model_fns: dict of model_name -> translate_fn(text, source_lang, target_lang) -> dict
            sources: List of source sentences.
            references: dict of lang_code -> [reference_translations].
            source_lang: Source language code.
            target_lang: Target language code.
            domains: Optional domain labels for domain-wise analysis.
            run_latency: Whether to run latency benchmarks.
            warmup_runs: Number of warmup sentences before benchmarking.

        Returns:
            dict with per-model results and comparison.
        """
        target_references = references.get(target_lang, [])
        if not target_references:
            return {'error': f'No references found for target language: {target_lang}'}

        if len(sources) != len(target_references):
            return {
                'error': f'Mismatch: {len(sources)} sources vs {len(target_references)} references',
            }

        results = {}
        start_time = time.time()

        for model_name, translate_fn in model_fns.items():
            logger.info(f"Benchmarking model: {model_name}")

            model_result = {
                'model': model_name,
                'lang_pair': f'{source_lang}->{target_lang}',
                'num_sentences': len(sources),
                'translations': [],
            }

            # Run translations
            hypotheses = []
            translation_times = []
            errors = []

            for i, src in enumerate(sources):
                try:
                    # monotonic: wall-clock (time.time) absorbs system suspend
                    # into a single sample and inflates per-sentence times.
                    t0 = time.monotonic()
                    translation_result = translate_fn(
                        text=src,
                        source_lang=source_lang,
                        target_lang=target_lang,
                    )
                    elapsed = (time.monotonic() - t0) * 1000

                    translated = translation_result.get('translated', '')
                    hypotheses.append(translated)
                    translation_times.append(elapsed)

                    model_result['translations'].append({
                        'index': i,
                        'source': src,
                        'hypothesis': translated,
                        'reference': target_references[i] if i < len(target_references) else '',
                        'time_ms': round(elapsed, 2),
                    })

                    # Wrappers report failures two ways: by raising, or by
                    # returning an 'error' field (TranslationResult.to_dict).
                    # Without this check, a caught-but-returned error (e.g. an
                    # unsupported language pair like m2m100->telugu) would
                    # silently surface as an empty hypothesis with no error
                    # record in the report.
                    err = translation_result.get('error')
                    if err:
                        logger.warning(
                            f"Translation error for model {model_name}, "
                            f"sentence {i}: {err}"
                        )
                        errors.append({'index': i, 'error': str(err)})
                except Exception as e:
                    logger.error(f"Translation failed for model {model_name}, sentence {i}: {e}")
                    hypotheses.append('')
                    translation_times.append(0.0)
                    errors.append({'index': i, 'error': str(e)})

            # Run metrics evaluation
            refs_for_eval = [[r] for r in target_references]
            eval_result = self.evaluator.evaluate(
                hypotheses=hypotheses,
                references=refs_for_eval,
                sources=sources,
                domains=domains,
                model_name=model_name,
                lang_pair=f'{source_lang}->{target_lang}',
            )

            model_result['metrics'] = eval_result.get('metrics', {})
            model_result['summary'] = eval_result.get('summary', {})
            model_result['domain_analysis'] = eval_result.get('domain_analysis', {})
            model_result['errors'] = errors

            # Latency benchmarks
            if run_latency and hypotheses:
                # Only benchmark on successfully translated sentences
                valid_indices = [i for i, h in enumerate(hypotheses) if h.strip()]
                valid_sentences = [sources[i] for i in valid_indices]

                if valid_sentences:
                    # Re-run for proper latency measurement. 30 sentences so
                    # the reported P95 is a real percentile: with <=10 samples
                    # any percentile method collapses onto the maximum, which
                    # makes the scorecard's P95 latency a single-request
                    # outlier instead of a distribution statistic.
                    latency_result = self.latency_bench.benchmark(
                        translate_fn=translate_fn,
                        sentences=valid_sentences[:min(len(valid_sentences), 30)],
                        source_lang=source_lang,
                        target_lang=target_lang,
                        warmup_runs=warmup_runs,
                    )
                    model_result['latency'] = latency_result
                else:
                    model_result['latency'] = {'error': 'No valid sentences for latency benchmark'}
            else:
                model_result['latency'] = {
                    'mean_ms': sum(translation_times) / max(len(translation_times), 1),
                    'total_time_ms': sum(translation_times),
                }

            # System info
            model_result['system_info'] = self.latency_bench.get_system_info()

            results[model_name] = model_result

        # Model comparison
        comparison = self.evaluator.compare_models(results)

        # Safety net: free any COMET/BERTScore models still resident before
        # the next language's run() call (evaluate() already unloads them per
        # metric, so this is normally a no-op). Keeps peak memory bounded on
        # CPU machines running --metrics all across many languages.
        self.evaluator.unload_neural_metrics()

        return {
            'benchmark': {
                'source_lang': source_lang,
                'target_lang': target_lang,
                'num_sentences': len(sources),
                'num_models': len(model_fns),
                'total_time_ms': round((time.time() - start_time) * 1000, 2),
            },
            'models': results,
            'comparison': comparison,
        }

    def run_all_languages(
        self,
        model_fns: Dict[str, Callable],
        sources: List[str],
        references: Dict[str, List[str]],
        source_lang: str = 'english',
        target_langs: Optional[List[str]] = None,
        domains: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run benchmark across multiple target languages.

        Args:
            model_fns: dict of model_name -> translate_fn
            sources: Source sentences.
            references: dict of lang_code -> [reference_translations].
            source_lang: Source language.
            target_langs: List of target languages to evaluate.
            domains: Optional domain labels.

        Returns:
            dict with results per language.
        """
        if target_langs is None:
            target_langs = [k for k in references.keys() if k != source_lang]

        all_results = {}
        for lang in target_langs:
            logger.info(f"\n{'='*60}\nBenchmarking language: {lang}\n{'='*60}")
            lang_result = self.run(
                model_fns=model_fns,
                sources=sources,
                references=references,
                source_lang=source_lang,
                target_lang=lang,
                domains=domains,
                run_latency=(lang == target_langs[0]),  # Only run latency for first lang to save time
                warmup_runs=2,
            )
            all_results[lang] = lang_result

        return {
            'languages': target_langs,
            'results': all_results,
            'metadata': {
                'source_lang': source_lang,
                'num_languages': len(target_langs),
                'num_models': len(model_fns),
                'num_sentences': len(sources),
                'metrics': self.evaluator.get_available_metrics(),
            },
        }
