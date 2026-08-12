"""
Evaluator — the central orchestrator for running evaluation metrics.

Runs a configurable set of metrics on hypothesis-reference pairs,
aggregates results, and produces structured evaluation reports.
"""

import logging
import time
from typing import List, Dict, Optional, Callable

from .metrics import (
    BLEUScorer,
    ChrFScorer,
    BERTScoreScorer,
    COMETScorer,
    LatencyBenchmark,
    QualityScorerWrapper,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Central evaluation orchestrator.

    Usage:
        evaluator = Evaluator(metrics=['bleu', 'chrf', 'bertscore'])
        results = evaluator.evaluate(
            hypotheses=['translation 1', 'translation 2'],
            references=[['ref 1a'], ['ref 2a']],
        )
    """

    AVAILABLE_METRICS = {
        'bleu': BLEUScorer,
        'chrf': ChrFScorer,
        'bertscore': BERTScoreScorer,
        'comet': COMETScorer,
        'quality': QualityScorerWrapper,
    }

    # Heavy neural metrics: their models are unloaded right after each score
    # (and on demand via unload_neural_metrics()) so a matrix run over many
    # language pairs never holds COMET + BERTScore + MT models at once.
    _UNLOAD_AFTER_SCORE = frozenset({'comet', 'bertscore'})

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        unload_neural_after_each: bool = True,
    ):
        """
        Args:
            metrics: List of metric names to compute.
                     Defaults to all available: ['bleu', 'chrf', 'bertscore', 'quality'].
            unload_neural_after_each: If True (default), the COMET/BERTScore
                     models are released immediately after each scoring call so
                     peak memory stays bounded on CPU machines (reload costs
                     ~10-20s per metric). Set False to keep them resident when
                     evaluating a single batch repeatedly.
        """
        self.metrics = metrics or list(self.AVAILABLE_METRICS.keys())
        self._unload_neural_after_each = unload_neural_after_each
        self._scorers = {}
        self._init_scorers()

    def _init_scorers(self):
        for name in self.metrics:
            if name in self.AVAILABLE_METRICS:
                try:
                    self._scorers[name] = self.AVAILABLE_METRICS[name]()
                    logger.info(f"Initialized metric: {name} (available={self._scorers[name].available})")
                except Exception as e:
                    logger.warning(f"Failed to initialize metric '{name}': {e}")
            else:
                logger.warning(f"Unknown metric '{name}'. Available: {list(self.AVAILABLE_METRICS.keys())}")

    def evaluate(
        self,
        hypotheses: List[str],
        references: List[List[str]],
        sources: Optional[List[str]] = None,
        domains: Optional[List[str]] = None,
        model_name: str = 'unknown',
        lang_pair: str = 'en->unknown',
        **kwargs,
    ) -> Dict:
        """
        Run all configured metrics on the provided data.

        Args:
            hypotheses: List of translated sentences.
            references: List of reference translations (each is a list of reference strings).
            sources: Optional source sentences (used by some metrics like COMET).
            domains: Optional domain labels for domain-wise analysis.
            model_name: Identifier for the model being evaluated.
            lang_pair: Language pair string (e.g., 'en->hi').

        Returns:
            dict with metrics results, metadata, and summary.
        """
        if not hypotheses or not references:
            return {'error': 'Empty hypotheses or references', 'model': model_name}

        if len(hypotheses) != len(references):
            return {
                'error': f'Mismatch: {len(hypotheses)} hypotheses vs {len(references)} references',
                'model': model_name,
            }

        start_time = time.time()

        results = {
            'model': model_name,
            'lang_pair': lang_pair,
            'num_sentences': len(hypotheses),
            'metrics': {},
            'metadata': {
                'evaluation_time_ms': 0,
                'metrics_computed': [],
                'metrics_failed': [],
            },
        }

        # Run each metric
        for name, scorer in self._scorers.items():
            if not scorer.available:
                results['metrics'][name] = {
                    'error': f'metric \'{name}\' unavailable (dependency not installed)',
                }
                results['metadata']['metrics_failed'].append(name)
                continue
            try:
                metric_start = time.time()

                if name == 'bleu':
                    score_result = scorer.score(hypotheses, references)
                elif name == 'chrf':
                    score_result = scorer.score(hypotheses, references)
                elif name == 'bertscore':
                    score_result = scorer.score(hypotheses, references)
                elif name == 'comet':
                    # COMET needs the source sentences for scoring
                    score_result = scorer.score(
                        hypotheses, references, sources=sources or hypotheses
                    )
                elif name == 'quality':
                    score_result = scorer.score(hypotheses, references, **kwargs)
                else:
                    score_result = scorer.score(hypotheses, references)

                metric_time = (time.time() - metric_start) * 1000

                results['metrics'][name] = {
                    **score_result,
                    'computation_time_ms': round(metric_time, 2),
                }
                results['metadata']['metrics_computed'].append(name)

                # Free heavy neural models as soon as their numbers are stored:
                # with --metrics all a matrix language otherwise holds COMET +
                # BERTScore + both MT models simultaneously (the original OOM).
                if self._unload_neural_after_each and name in self._UNLOAD_AFTER_SCORE:
                    self._unload_metric(name)

            except Exception as e:
                logger.error(f"Metric '{name}' failed: {e}")
                results['metrics'][name] = {'error': str(e)}
                results['metadata']['metrics_failed'].append(name)

        # Domain-wise analysis (if domains provided)
        if domains and len(domains) == len(hypotheses):
            results['domain_analysis'] = self._domain_analysis(hypotheses, references, domains)

        # Summary statistics
        results['metadata']['evaluation_time_ms'] = round((time.time() - start_time) * 1000, 2)
        results['summary'] = self._compute_summary(results)

        return results

    def _domain_analysis(
        self,
        hypotheses: List[str],
        references: List[List[str]],
        domains: List[str],
    ) -> Dict:
        """Compute metrics per domain."""
        domain_indices = {}
        for i, d in enumerate(domains):
            if d not in domain_indices:
                domain_indices[d] = []
            domain_indices[d].append(i)

        domain_results = {}
        for domain, indices in domain_indices.items():
            domain_hyp = [hypotheses[i] for i in indices]
            domain_ref = [references[i] for i in indices]
            domain_results[domain] = {
                'num_sentences': len(indices),
                'indices': indices,
            }
            # Run BLEU on domain subset (fastest metric)
            if 'bleu' in self._scorers:
                try:
                    bleu_result = self._scorers['bleu'].score(domain_hyp, domain_ref)
                    domain_results[domain]['bleu'] = bleu_result.get('bleu', 0.0)
                except Exception:
                    domain_results[domain]['bleu'] = 0.0

        return domain_results

    def _unload_metric(self, name: str):
        """Release a metric's model weights if the scorer supports it."""
        scorer = self._scorers.get(name)
        unload = getattr(scorer, 'unload', None)
        if callable(unload):
            try:
                unload()
            except Exception as e:
                logger.warning(f"Failed to unload metric '{name}': {e}")

    def unload_neural_metrics(self):
        """
        Unload heavy neural metric models (COMET/BERTScore) to free memory.

        Safe to call between matrix languages / evaluations: the scorers lazily
        reload on their next use, so already-collected results are unaffected.
        With ``unload_neural_after_each=True`` (default) this is normally a
        no-op because each metric already unloads after scoring; it exists for
        explicit callers and as a safety net at the end of a run.
        """
        for name in self._UNLOAD_AFTER_SCORE:
            self._unload_metric(name)

    def _compute_summary(self, results: Dict) -> Dict:
        """Compute summary statistics from all metrics."""
        summary = {
            'bleu': None,
            'chrf_plus': None,
            'bertscore_f1': None,
            'comet_score': None,
            'quality_score': None,
        }

        if 'bleu' in results['metrics'] and 'bleu' in results['metrics']['bleu']:
            summary['bleu'] = round(results['metrics']['bleu']['bleu'], 2)
        if 'chrf' in results['metrics'] and 'chrf_plus' in results['metrics']['chrf']:
            summary['chrf_plus'] = round(results['metrics']['chrf']['chrf_plus'], 2)
        if 'bertscore' in results['metrics'] and 'f1' in results['metrics']['bertscore']:
            summary['bertscore_f1'] = round(results['metrics']['bertscore']['f1'], 4)
        if 'comet' in results['metrics'] and 'comet_score' in results['metrics']['comet']:
            summary['comet_score'] = round(results['metrics']['comet']['comet_score'], 4)
        if 'quality' in results['metrics'] and 'quality_score' in results['metrics']['quality']:
            summary['quality_score'] = round(results['metrics']['quality']['quality_score'], 4)

        return summary

    def compare_models(
        self,
        model_results: Dict[str, Dict],
    ) -> Dict:
        """
        Compare results from multiple models.

        Args:
            model_results: dict of model_name -> evaluate() output

        Returns:
            dict with comparison table and rankings per metric.
        """
        comparison = {
            'models': list(model_results.keys()),
            'metrics_comparison': {},
            'rankings': {},
        }

        # Collect all metrics across models
        all_metrics = set()
        for name, result in model_results.items():
            if 'summary' in result:
                all_metrics.update(result['summary'].keys())

        # Build comparison per metric
        for metric in all_metrics:
            comparison['metrics_comparison'][metric] = {}
            scores = []
            for name, result in model_results.items():
                if 'summary' in result and result['summary'].get(metric) is not None:
                    comparison['metrics_comparison'][metric][name] = result['summary'][metric]
                    scores.append((name, result['summary'][metric]))

            # Rank (higher is better for all our metrics)
            scores.sort(key=lambda x: -x[1])
            comparison['rankings'][metric] = [
                {'rank': i + 1, 'model': name, 'score': score}
                for i, (name, score) in enumerate(scores)
            ]

        # Overall ranking (mean of all metric ranks)
        if all_metrics:
            model_avg_rank = {}
            for name in model_results:
                ranks = []
                for metric in all_metrics:
                    for entry in comparison['rankings'].get(metric, []):
                        if entry['model'] == name:
                            ranks.append(entry['rank'])
                model_avg_rank[name] = sum(ranks) / max(len(ranks), 1)

            sorted_models = sorted(model_avg_rank.items(), key=lambda x: x[1])
            comparison['overall_ranking'] = [
                {'rank': i + 1, 'model': name}
                for i, (name, _) in enumerate(sorted_models)
            ]

        comparison['num_models'] = len(model_results)
        comparison['num_metrics'] = len(all_metrics)
        return comparison

    def get_available_metrics(self) -> List[str]:
        """List metrics that are available (dependencies installed)."""
        return [name for name, scorer in self._scorers.items() if scorer.available]

    def __repr__(self):
        return f"Evaluator(metrics={list(self._scorers.keys())})"
