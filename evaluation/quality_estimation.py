"""
Quality Estimation (QE) validation benchmark.

Validates reference-free quality estimation by presenting it with a labeled
set of translations:

  GOOD     — curated reference translations (from the benchmark datasets)
  BAD      — word-scrambled or wrong-language "translations"
  PARTIAL  — truncated / half-translated outputs

The estimator must separate good translations from bad ones. We report
accuracy, precision, recall, F1 (good = positive class) plus a confusion
matrix — exactly the plan's "Benchmark #15 — Translation Quality
Estimation" (AUROC is reported when scikit-learn is available).

Two engines are supported, and can be compared head-to-head:

  heuristic  — the lightweight QualityScorer (script consistency,
               sentence-end punctuation, Devanagari word-order sanity,
               length ratio, term overlap). Model-free, runs in seconds.
  cometkiwi  — reference-free neural QE via Unbabel's COMETkiwi
               (wmt22-cometkiwi-da). Downloads ~2.7GB of weights on
               first use, so it is opt-in.

Usage (from translator-backend/):
    python -m app.evaluation.quality_estimation                    # heuristic
    python -m app.evaluation.quality_estimation --scorer cometkiwi
    python -m app.evaluation.quality_estimation --scorer both      # head-to-head
"""

import logging
import random

logger = logging.getLogger(__name__)

SCORER_HEURISTIC = 'heuristic'
SCORER_COMETKIWI = 'cometkiwi'


class QualityEstimationBenchmark:
    """Evaluate how well a reference-free scorer separates good from bad."""

    # overall_score >= threshold => predicted GOOD. Calibrated against the
    # improved heuristic scorer (script + sentence-end features); see README
    # notes. COMETkiwi uses its own DEFAULT_THRESHOLD (raw 0 boundary).
    DEFAULT_THRESHOLD = 0.80

    def __init__(self, threshold: float = DEFAULT_THRESHOLD, seed: int = 42,
                 target_lang: str = 'hindi', scorer: str = SCORER_HEURISTIC):
        """
        Args:
            threshold: Prediction boundary for the heuristic scorer.
            seed: Deterministic scrambling for the BAD samples.
            target_lang: Target language for script-consistency checks.
            scorer: 'heuristic' (default, model-free), 'cometkiwi' (neural,
                downloads weights on first use), or 'both' (head-to-head).
        """
        if scorer not in (SCORER_HEURISTIC, SCORER_COMETKIWI, 'both'):
            raise ValueError(f"scorer must be one of {{'heuristic', 'cometkiwi', 'both'}}, "
                             f"got {scorer!r}")
        self.threshold = threshold
        self.seed = seed
        self.target_lang = target_lang
        self.scorer = scorer
        self._scorers = {}

    # -- engine loading ----------------------------------------------------

    def _load_scorer(self, which: str):
        if which not in self._scorers:
            if which == SCORER_HEURISTIC:
                from app.services.quality import QualityScorer
                self._scorers[which] = QualityScorer()
            else:
                from app.services.quality import COMETkiwiQualityScorer
                self._scorers[which] = COMETkiwiQualityScorer()
        return self._scorers[which]

    # -- labeled sample construction ----------------------------------------

    def build_samples(self, max_good=60, max_bad=60, max_partial=40):
        """
        Build labeled (source, translation, label) samples from the curated
        datasets. Deterministic (seeded) scrambling.

        Returns:
            list of {'source', 'translation', 'label'} where label in
            {'good', 'bad', 'partial'}
        """
        from app.evaluation.datasets import CURATED_ITEMS

        rng = random.Random(self.seed)
        samples = []

        # GOOD: reference translations
        for domain, category, source, refs in CURATED_ITEMS[:max_good]:
            ref = refs.get('hindi') or refs.get('marathi') or ''
            if ref:
                samples.append({'source': source, 'translation': ref, 'label': 'good'})

        # BAD: word-scrambled references (preserve length ratio)
        for domain, category, source, refs in CURATED_ITEMS[:max_bad]:
            ref = refs.get('hindi') or refs.get('marathi') or ''
            if not ref:
                continue
            words = ref.split()
            scrambled = words[:]
            for _ in range(3):
                rng.shuffle(scrambled)
            samples.append({
                'source': source,
                'translation': ' '.join(scrambled),
                'label': 'bad',
            })

        # BAD: wrong-language translation (German/Chinese text for an English source)
        wrong_lang = [
            'Deutschland ist ein vielfältiges Land in Europa.',
            '天气今天很好，我想去公园散步。',
            'El tiempo es agradable hoy y quiero pasear por el parque.',
            'আজ আবহাওয়া খুব সুন্দর।',  # Bengali (right script, wrong language)
            'ನಾನು ನಾಳೆ ಕಾಲೇಜಿಗೆ ಹೋಗುತ್ತಿದ್ದೇನೆ.',  # Kannada
        ]
        for i, item in enumerate(CURATED_ITEMS[:5]):
            ref = item[3].get('hindi') or ''
            if ref:
                samples.append({
                    'source': item[2],
                    'translation': wrong_lang[i % len(wrong_lang)],
                    'label': 'bad',
                })

        # PARTIAL: truncated references (first 40% of characters)
        for domain, category, source, refs in CURATED_ITEMS[:max_partial]:
            ref = refs.get('hindi') or refs.get('marathi') or ''
            if not ref:
                continue
            cut = max(1, int(len(ref) * 0.4))
            samples.append({
                'source': source,
                'translation': ref[:cut].rstrip() + '…',
                'label': 'partial',
            })

        return samples

    # -- scoring helpers ----------------------------------------------------

    def _score_all(self, scorer, samples) -> list:
        """Score every sample. Uses batched inference when the scorer offers it."""
        sources = [s['source'] for s in samples]
        translations = [s['translation'] for s in samples]

        if hasattr(scorer, 'score_batch'):
            out = scorer.score_batch(sources, translations)
            if out.get('error'):
                raise RuntimeError(f"{getattr(scorer, '__class__', None).__name__}: "
                                   f"{out['error']}")
            results = out['results']
        else:
            results = [
                scorer.score(source_text=src, translated_text=tgt,
                             target_lang=self.target_lang)
                for src, tgt in zip(sources, translations)
            ]
        return results

    def _evaluate_with(self, scorer, samples, threshold) -> dict:
        """
        Score all samples with one engine and compute classification metrics.

        Returns:
            dict with metrics + 'predictions' (list of per-sample dicts).
        """
        results = self._score_all(scorer, samples)

        predictions = []
        for s, r in zip(samples, results):
            predictions.append({
                **s,
                'score': r['overall_score'],
                'confidence': r.get('confidence'),
                'hallucination': r.get('hallucination_detected', False),
                'word_order_suspicious': r.get('word_order_suspicious', False),
                'pred_good': r['overall_score'] >= threshold,
                'true_good': s['label'] == 'good',
            })

        # Binary classification (good = positive)
        tp = sum(1 for p in predictions if p['pred_good'] and p['true_good'])
        fp = sum(1 for p in predictions if p['pred_good'] and not p['true_good'])
        fn = sum(1 for p in predictions if not p['pred_good'] and p['true_good'])
        tn = sum(1 for p in predictions if not p['pred_good'] and not p['true_good'])

        accuracy = (tp + tn) / max(len(predictions), 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        # Per-label mean scores
        by_label = {}
        for p in predictions:
            by_label.setdefault(p['label'], {'count': 0, 'score_sum': 0.0, 'conf': {}})
            by_label[p['label']]['count'] += 1
            by_label[p['label']]['score_sum'] += p['score']
            by_label[p['label']]['conf'][p['confidence']] = \
                by_label[p['label']]['conf'].get(p['confidence'], 0) + 1
        for label, data in by_label.items():
            data['mean_score'] = round(data['score_sum'] / max(data['count'], 1), 4)

        # Optional AUROC
        auroc = None
        try:
            from sklearn.metrics import roc_auc_score  # noqa: F401
            y_true = [1 if p['true_good'] else 0 for p in predictions]
            y_score = [p['score'] for p in predictions]
            auroc = round(roc_auc_score(y_true, y_score), 4)
        except Exception:
            pass  # scikit-learn not installed — optional

        return {
            'threshold': threshold,
            'confusion_matrix': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'auroc': auroc,
            'per_label': by_label,
            'predictions': predictions,
        }

    # -- evaluation ---------------------------------------------------------

    def run(self, samples=None) -> dict:
        """
        Run the QE benchmark with the configured scorer(s).

        Returns:
            dict with top-level metrics for the primary scorer plus a
            'scorers' head-to-head block when scorer='both'.
        """
        samples = samples if samples is not None else self.build_samples()

        if self.scorer == SCORER_COMETKIWI:
            primaries = [SCORER_COMETKIWI]
        elif self.scorer == 'both':
            primaries = [SCORER_HEURISTIC, SCORER_COMETKIWI]
        else:
            primaries = [SCORER_HEURISTIC]

        scored = {}
        for which in primaries:
            scorer = self._load_scorer(which)
            if which == SCORER_COMETKIWI and not scorer.available:
                scored[which] = {'available': False, 'error': 'COMETkiwi unavailable'}
                logger.warning('COMETkiwi scorer unavailable — check that '
                               'unbabel-comet is installed and the model can '
                               'be downloaded.')
                continue
            threshold = self.threshold if which == SCORER_HEURISTIC \
                else scorer.DEFAULT_THRESHOLD
            try:
                metrics = self._evaluate_with(scorer, samples, threshold)
                scored[which] = {'available': True, **metrics}
            except Exception as exc:
                logger.error(f'{which} scorer failed: {exc}')
                scored[which] = {'available': False, 'error': str(exc)}

        # Primary engine = the one the top-level metrics report.
        primary = primaries[0]
        if primary not in scored and len(primaries) > 1:
            primary = primaries[1]  # heuristic fallback if cometkiwi failed
        if primary not in scored or not scored[primary].get('available'):
            return {'error': 'no scorer available', 'scorers': scored,
                    'num_samples': len(samples)}

        top = scored[primary]
        result = {
            'scorer': primary,
            'num_samples': len(samples),
            'label_counts': {label: sum(1 for s in samples if s['label'] == label)
                             for label in ('good', 'bad', 'partial')},
            'confusion_matrix': top['confusion_matrix'],
            'accuracy': top['accuracy'],
            'precision': top['precision'],
            'recall': top['recall'],
            'f1': top['f1'],
            'auroc': top['auroc'],
            'per_label': top['per_label'],
            'samples': top['predictions'],
        }
        # How many samples the Devanagari word-order sanity rule flagged
        # (diagnostic for the report: scrambled output caught at zero model
        # cost on top of the score signals).
        result['word_order_flagged'] = sum(
            1 for p in top['predictions'] if p.get('word_order_suspicious')
        )
        result['threshold'] = top['threshold']

        if self.scorer == 'both' and len(scored) > 1:
            comparison = {}
            for which, m in scored.items():
                if m.get('available'):
                    comparison[which] = {
                        'available': True,
                        'threshold': m['threshold'],
                        'confusion_matrix': m['confusion_matrix'],
                        'accuracy': m['accuracy'],
                        'precision': m['precision'],
                        'recall': m['recall'],
                        'f1': m['f1'],
                        'auroc': m['auroc'],
                        'per_label': m['per_label'],
                    }
                else:
                    comparison[which] = {
                        'available': False,
                        'error': m.get('error', 'unknown'),
                    }
            candidates = {k: v['f1'] for k, v in comparison.items()
                          if v.get('available')}
            winner = max(candidates, key=candidates.get) if candidates else None
            result['scorers'] = comparison
            result['winner'] = winner
            result['f1_delta'] = round(
                candidates.get(winner, 0.0) - comparison.get(
                    SCORER_HEURISTIC, {}).get('f1', 0.0), 4) if winner else None

        return result


def run_quality_estimation_benchmark(scorer: str = SCORER_HEURISTIC) -> dict:
    """Module-level convenience entry point."""
    return QualityEstimationBenchmark(scorer=scorer).run()


def _print_result(result: dict):
    print("\nQuality Estimation Benchmark")
    print("-" * 40)
    if result.get('error'):
        print(f"ERROR: {result['error']}")
        return
    print(f"Scorer   : {result['scorer']}")
    print(f"Samples  : {result['num_samples']} (labels: {result['label_counts']})")
    print(f"Threshold: {result['threshold']}")
    print(f"Confusion: {result['confusion_matrix']}")
    print(f"Accuracy : {result['accuracy']}")
    print(f"Precision: {result['precision']}")
    print(f"Recall   : {result['recall']}")
    print(f"F1       : {result['f1']}")
    print(f"AUROC    : {result['auroc']}")
    print("\nPer-label mean scores:")
    for label, data in result['per_label'].items():
        print(f"  {label:<8} n={data['count']:<3} mean_score={data['mean_score']} "
              f"confidence={data['conf']}")

    scorers = result.get('scorers')
    if scorers:
        print("\n" + "=" * 40)
        print("HEAD-TO-HEAD")
        print("=" * 40)
        print(f"{'metric':<12} {'heuristic':>10} {'cometkiwi':>10}")
        print("-" * 36)
        for metric in ('f1', 'accuracy', 'precision', 'recall', 'auroc'):
            cells = []
            for which in ('heuristic', 'cometkiwi'):
                m = scorers.get(which) or {}
                cells.append(f"{m.get(metric, 'n/a') if m.get('available') else 'unavail':>10}")
            print(f"{metric:<12} {' '.join(cells)}")
        print(f"\nWinner by F1: {result.get('winner')} "
              f"(Δ vs heuristic: {result.get('f1_delta')})")


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Quality Estimation benchmark')
    p.add_argument('--scorer', default=SCORER_HEURISTIC,
                   choices=[SCORER_HEURISTIC, SCORER_COMETKIWI, 'both'],
                   help='Which QE engine(s) to evaluate (default: heuristic). '
                        'cometkiwi downloads ~2.7GB of weights on first use.')
    args = p.parse_args()

    result = run_quality_estimation_benchmark(scorer=args.scorer)
    _print_result(result)
