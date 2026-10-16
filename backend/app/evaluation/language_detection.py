"""
Language detection benchmark.

Implements the plan's "Benchmark #10 — Benchmark Language Detection":

    Accuracy · Precision · Recall · F1 · Confusion Matrix

Test set: the curated English sources + the 7 Indic reference translations
(42 sentences per language) plus the code-mixed sentences. The Devanagari
model (Hindi vs Marathi — words + suffixes over a Laplace-smoothed
multinomial NB) is trained on a deterministic half of the data and
evaluated on the other half, so the reported numbers are honest
generalization estimates rather than self-consistent scores.

The confusion matrix for related languages (e.g. hi/mr, te/kn) is exactly
the kind of figure the research paper needs.

Usage (from backend/):
    python -m app.evaluation.language_detection
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

BENCHMARK_LANGUAGES = [
    'english', 'hindi', 'marathi', 'bengali',
    'gujarati', 'tamil', 'telugu', 'kannada',
]


# ---------------------------------------------------------------------------
# Metric helpers (pure python — no sklearn dependency)
# ---------------------------------------------------------------------------

def _per_class_metrics(pairs: List[Tuple[str, str]], labels: List[str]) -> Dict:
    """precision / recall / f1 / support per class (prediction correctness)."""
    metrics = {}
    for label in labels:
        tp = sum(1 for a, p in pairs if a == label and p == label)
        fp = sum(1 for a, p in pairs if a != label and p == label)
        fn = sum(1 for a, p in pairs if a == label and p != label)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        metrics[label] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'support': sum(1 for a, _ in pairs if a == label),
        }
    return metrics


def _macro_f1(per_class: Dict) -> float:
    f1s = [m['f1'] for m in per_class.values() if m['support'] > 0]
    return sum(f1s) / max(len(f1s), 1)


def _weighted_f1(per_class: Dict) -> float:
    total = sum(m['support'] for m in per_class.values())
    if total == 0:
        return 0.0
    return sum(m['f1'] * m['support'] for m in per_class.values()) / total


def _confusion_matrix(pairs: List[Tuple[str, str]], labels: List[str]) -> Dict:
    cm = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for actual, predicted in pairs:
        actual_key = actual if actual in cm else 'unknown'
        predicted_key = predicted if predicted in cm else 'unknown'
        cm.setdefault(actual_key, {}).setdefault(predicted_key, 0)
        cm[actual_key][predicted_key] += 1
    return cm


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class LanguageDetectionBenchmark:
    """Evaluate LanguageDetector accuracy / precision / recall / F1."""

    def run(self, include_code_mixed: bool = True) -> Dict:
        """
        Run the benchmark.

        Args:
            include_code_mixed: Also test the code-mixed sentences (their
                                gold label is the Indic language of the script).

        Returns:
            dict with accuracy, per-class metrics, macro/weighted F1 and the
            confusion matrix (plus the detector's script/mode used).
        """
        from app.evaluation.datasets import (
            CODE_MIXED_ITEMS, CURATED_ITEMS, ENGLISH_SENTENCES,
        )
        from app.services.language_detection import LanguageDetector

        # ---- labeled samples --------------------------------------------
        samples: List[Tuple[str, str]] = []  # (text, gold_language)
        for src in ENGLISH_SENTENCES:
            samples.append((src, 'english'))
        for item in CURATED_ITEMS:
            refs = item[3]
            for lang in BENCHMARK_LANGUAGES:
                if lang == 'english':
                    continue
                ref = refs.get(lang, '')
                if ref:
                    samples.append((ref, lang))
        if include_code_mixed:
            for item in CODE_MIXED_ITEMS:
                samples.append((item['source'], item['source_lang']))

        # ---- deterministic train/test split (half-and-half by index) ----
        train_texts = {'hindi': [], 'marathi': []}
        test_pairs: List[Tuple[str, str]] = []
        for i, (text, lang) in enumerate(samples):
            if i % 2 == 0:
                if lang in train_texts:
                    train_texts[lang].append(text)
            else:
                test_pairs.append((text, lang))

        # ---- train + predict --------------------------------------------
        detector = LanguageDetector(train_texts=train_texts)
        predictions: List[Tuple[str, str]] = []  # (gold, predicted)
        for text, gold in test_pairs:
            predictions.append((gold, detector.detect(text)))

        # ---- metrics -----------------------------------------------------
        per_class = _per_class_metrics(predictions, BENCHMARK_LANGUAGES)
        accuracy = sum(1 for a, p in predictions if a == p) / max(len(predictions), 1)

        # The learned component is only the Devanagari (hi/mr) disambiguation;
        # the other six languages are deterministic script-range checks.
        dev_f1s = [per_class[lang]['f1'] for lang in ('hindi', 'marathi')]

        return {
            'num_samples': len(samples),
            'num_test': len(test_pairs),
            'train_split': 'even-indexed (deterministic half)',
            'detector': 'unicode-script + devanagari word/2-5-suffix model '
                        '(char n-gram fallback)',
            'accuracy': round(accuracy, 4),
            'macro_f1': round(_macro_f1(per_class), 4),
            'weighted_f1': round(_weighted_f1(per_class), 4),
            'devanagari_macro_f1': round(sum(dev_f1s) / len(dev_f1s), 4),
            'per_class': per_class,
            'confusion_matrix': _confusion_matrix(predictions, BENCHMARK_LANGUAGES),
            'note': (
                'English/Bengali/Gujarati/Tamil/Telugu/Kannada are deterministic '
                'script-range checks; only the Devanagari (hi/mr) disambiguation '
                'is learned (words + 2-5-char suffixes, multinomial NB with '
                'Laplace smoothing, plus language-marker characters and a char '
                'n-gram fallback). The Devanagari model is trained on the '
                'held-out-even split, so its F1 is an honest generalization '
                'estimate; the production detector trains on the full curated set. '
                'Related-language confusions (hi/mr) appear in the confusion matrix.'
            ),
        }


def run_language_detection_benchmark() -> Dict:
    """Module-level convenience entry point."""
    return LanguageDetectionBenchmark().run()


def _render_confusion(cm: Dict, labels: List[str]) -> List[str]:
    """Human-readable confusion matrix rows for the report."""
    width = max(len(l) for l in labels) + 2
    header = ' ' * width + ''.join(f'{l:>{width}}' for l in labels)
    rows = [header]
    for actual in labels:
        row = f'{actual:<{width}}' + ''.join(
            f'{cm.get(actual, {}).get(predicted, 0):>{width}}'
            for predicted in labels
        )
        rows.append(row)
    return rows


if __name__ == '__main__':
    result = run_language_detection_benchmark()
    print("\nLanguage Detection Benchmark")
    print("-" * 50)
    print(f"Samples: {result['num_samples']}  test: {result['num_test']}")
    print(f"Accuracy    : {result['accuracy']}")
    print(f"Macro F1    : {result['macro_f1']}")
    print(f"Weighted F1 : {result['weighted_f1']}")
    print("\nPer-class:")
    print(f"  {'language':<12} {'P':>6} {'R':>6} {'F1':>6} {'n':>4}")
    for lang, m in result['per_class'].items():
        print(f"  {lang:<12} {m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['f1']:>6.3f} {m['support']:>4}")
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    for row in _render_confusion(result['confusion_matrix'], BENCHMARK_LANGUAGES):
        print('  ' + row)
    print(f"\nNote: {result['note']}")
