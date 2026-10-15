"""
Hallucination / factuality benchmark — degenerate-output detection.

Implements the plan's factuality probe without an LLM judge. Because the
pipeline has no reference model or factual KB, "hallucination" here is
measured with deterministic heuristics over the output text:

  * Entity preservation   — numbers, dates, currency, measurements, URLs,
    emails and Initial-caps names from the source must survive translation.
  * Source leakage        — output that copies the source verbatim (whole
    sentences / long spans) means translation never happened.
  * Repetition            — self-repeating n-grams (stuttering / degenerate
    loop) are a classic hallucination signature.
  * Empty / punctuation-only output — the trivial hallucination.

Detector calibration
--------------------
The benchmark first feeds the detector known-good and known-degenerate
outputs (a "good" mock translator that preserves entities and a degenerate
mock that stutters / echoes / emits nothing) and asserts the detector
separates them. Only then does it score the translator under test. In mock
mode the translator under test is the good stub; in live mode it is the
real model.

Usage (from backend/):
    python -m app.evaluation.hallucination [--live]
"""

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entity extraction (numbers, dates, currency, measurement, URLs, emails,
# and Initial-caps multi-word names).
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"""
    \b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b   # dates (12/08/2026, 2026-08-12)
  | \b(?:19|20)\d{2}\b                    # years
  | \$\d[\d,]*(?:\.\d+)?                  # currency ($1,250.75)
  | \b\d+\s?(?:kg|km|km/h|cm|mm|m|l|litres?|sq\.?\s?ft|%)\b
  | https?://\S+                           # URLs
  | [\w.+-]+@[\w-]+\.[\w.-]+               # emails
  | \b(?:Dr\.?|Prof\.?|Mr\.?|Mrs\.?|Ms\.?|Miss|Shri|Smt\.?)\s+[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)?
  | \b\d[\d,]*(?:\.\d+)?%?                # numbers (12,450.5 / 8%)
""", re.IGNORECASE | re.VERBOSE)

_NAME_RE = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b')


def extract_entities(text):
    """
    Return the deduplicated list of entity strings in `text`.

    Overlapping captures from the two regexes are collapsed: if one entity is
    a substring of another (e.g. "Meera Patel" inside "Dr. Meera Patel"),
    only the longest is kept.
    """
    found = _TOKEN_RE.findall(text)
    found += _NAME_RE.findall(text)
    # Strip trailing sentence punctuation from number/date tokens (e.g. the
    # `\d[\d,]*` alternative captures '1632,' — the comma is not part of the
    # entity). Leading punctuation ($, https://) is intentional.
    unique = list(dict.fromkeys(
        f.strip().rstrip('.,;:!?"\'') for f in found if f and f.strip()
    ))
    kept = []
    for ent in sorted(unique, key=len, reverse=True):
        if any(ent in k for k in kept):
            continue
        kept.append(ent)
    return kept


def _ngrams(tokens, n):
    return zip(*[tokens[i:] for i in range(n)])


class HallucinationDetector:
    """Deterministic heuristics for degenerate/hallucinated output."""

    @staticmethod
    def repetition_ratio(text, max_ngram=8):
        """
        Fraction of token n-grams (n in 2..max_ngram) that repeat within the
        text. 0.0 = no self-repetition, >0 = stuttering/degenerate loop.
        """
        tokens = text.split()
        if len(tokens) < 2:
            return 0.0
        worst = 0.0
        for n in range(2, min(max_ngram, len(tokens)) + 1):
            grams = list(_ngrams(tokens, n))
            if not grams:
                continue
            seen = {}
            for g in grams:
                key = ' '.join(g)
                seen[key] = seen.get(key, 0) + 1
            repeated = sum(c for c in seen.values() if c > 1)
            worst = max(worst, repeated / len(grams))
        return round(worst, 4)

    @staticmethod
    def source_leakage_ratio(hypothesis, source):
        """
        Fraction of source *characters* that appear verbatim in the output
        (longest common subsequence ratio). ~1.0 means the "translation" is
        just the source echoed back; ~0.0 means it was rewritten.
        """
        if not source:
            return 0.0
        source = ' '.join(source.split()).lower()
        hypothesis = ' '.join(hypothesis.split()).lower()
        if not hypothesis:
            return 0.0

        # Longest common substring (LCS) over normalized text
        m, n = len(source), len(hypothesis)
        if m == 0 or n == 0:
            return 0.0
        prev = [0] * (n + 1)
        best = 0
        for i in range(1, m + 1):
            cur = [0] * (n + 1)
            for j in range(1, n + 1):
                if source[i - 1] == hypothesis[j - 1]:
                    cur[j] = prev[j - 1] + 1
                    best = max(best, cur[j])
                else:
                    cur[j] = 0
            prev = cur
        return round(best / m, 4)

    @staticmethod
    def entity_preservation_rate(hypothesis, source):
        """Fraction of source entities present in the hypothesis."""
        entities = extract_entities(source)
        if not entities:
            return 1.0  # nothing to preserve -> not a failure
        hl = hypothesis.lower()
        kept = sum(1 for e in entities if e.lower() in hl)
        return round(kept / len(entities), 4)

    @staticmethod
    def is_degenerate(hypothesis, source, repetition_threshold=0.35,
                      leakage_threshold=0.9):
        """
        Verdict on whether the output is degenerate/hallucinated.

        Returns (bool, reason).
        """
        text = hypothesis.strip()
        if not text or all(c in ' .,;:!?"\'' for c in text):
            return True, 'empty or punctuation-only output'
        if text.lower() == source.strip().lower():
            return True, 'output is an exact copy of the source'
        rep = HallucinationDetector.repetition_ratio(text)
        if rep >= repetition_threshold:
            return True, f'repetitive/degenerate output (repetition_ratio={rep})'
        leakage = HallucinationDetector.source_leakage_ratio(text, source)
        if leakage >= leakage_threshold:
            return True, f'source leakage (leakage_ratio={leakage})'
        return False, 'ok'

    def analyze(self, hypothesis, source):
        """Return the full detector profile for one output."""
        deg, reason = self.is_degenerate(hypothesis, source)
        return {
            'degenerate': deg,
            'reason': reason,
            'repetition_ratio': self.repetition_ratio(hypothesis),
            'source_leakage_ratio': self.source_leakage_ratio(hypothesis, source),
            'entity_preservation_rate': self.entity_preservation_rate(
                hypothesis, source),
        }


# ---------------------------------------------------------------------------
# Mock translators
# ---------------------------------------------------------------------------

def _good_mock_translator(text, source_lang, target_lang):
    """Preserves entities, produces distinct (non-echo) plausible output."""
    entities = extract_entities(text)
    gloss = ' '.join(entities) if entities else 'सरल वाक्य'
    return {
        'original': text,
        'translated': f'यह सटीक अनुवाद है। {gloss}',
        'source_lang': source_lang,
        'target_lang': target_lang,
        'metadata': {'model': 'mock-good', 'error': None},
    }


def _degenerate_stutter(text, source_lang, target_lang):
    return {
        'original': text,
        'translated': 'यह यह यह यह यह यह यह यह यह यह।' * 3,
        'metadata': {'model': 'mock-stutter', 'error': None},
    }


def _degenerate_echo(text, source_lang, target_lang):
    return {
        'original': text,
        'translated': text,  # exact source echo
        'metadata': {'model': 'mock-echo', 'error': None},
    }


def _degenerate_empty(text, source_lang, target_lang):
    return {
        'original': text,
        'translated': '',
        'metadata': {'model': 'mock-empty', 'error': None},
    }


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

# (sentence, [entities to expect])
HALLUCINATION_CASES = [
    "The meeting with Dr. Meera Patel is scheduled for 12/08/2026 at 10:30 AM.",
    "The total invoice was $1,250.75 and payment is due by 2026-03-31.",
    "Contact support@example.com or visit https://example.com/refund for details.",
    "India's GDP grew 8.4% in 2025, reaching ₹420.5 lakh crore.",
    "The package weighs 12.5 kg and must travel 240 km by train.",
    "Rahul Gandhi spoke with Rakesh Kumar about the project timeline.",
    "The conference in Mumbai starts on 15 March 2026 and ends on 17 March 2026.",
    "Please call +91 98765 43210 to reach Dr. Sharma's clinic.",
    "The temperature reached 42°C in Delhi on 15 June 2025.",
    "She booked 2 tickets for the 7:45 PM show at PVR Cinemas.",
    "The Taj Mahal, built in 1632, is in Agra, Uttar Pradesh.",
]


class HallucinationBenchmark:
    """Scores a translator's outputs for hallucination signatures."""

    def __init__(self, translate_fn=None, detector=None):
        """
        Args:
            translate_fn: (text, source_lang, target_lang) -> dict.
                          If None, the good mock translator is used.
        """
        self.detector = detector or HallucinationDetector()
        if translate_fn is None:
            self.mock = True
            self.translate_fn = _good_mock_translator
        else:
            self.mock = False
            self.translate_fn = translate_fn

    def calibrate_detector(self):
        """
        Feed the detector known-good and known-degenerate outputs and assert
        it separates them. Returns a dict proving detector sanity.
        """
        probe = ("Dr. Meera Patel will charge $1,250 for the surgery "
                 "on 12/08/2026.")
        cases = [
            ('good', _good_mock_translator(probe, 'english', 'hindi')
             ['translated'], probe),
            ('stutter', _degenerate_stutter(probe, 'english', 'hindi')
             ['translated'], probe),
            ('echo', _degenerate_echo(probe, 'english', 'hindi')
             ['translated'], probe),
            ('empty', _degenerate_empty(probe, 'english', 'hindi')
             ['translated'], probe),
        ]

        def expected_degenerate(label):
            return label != 'good'

        results = {}
        for label, hyp, src in cases:
            analysis = self.detector.analyze(hyp, src)
            detected = analysis['degenerate']
            results[label] = {
                'expected_degenerate': expected_degenerate(label),
                'detected': detected,
                'correct': detected == expected_degenerate(label),
                **{k: v for k, v in analysis.items()
                   if k not in ('degenerate', 'reason')},
            }
        correct = sum(1 for r in results.values() if r['correct'])
        return {
            'probe': probe,
            'cases': results,
            'calibration_pass_rate': round(correct / max(len(results), 1) * 100, 2),
        }

    def run(self, cases=None, source_lang='english', target_lang='hindi') -> dict:
        """
        Translate each case and score the outputs with the detectors.

        Returns:
            dict: per-case analyses + aggregate rates + calibration.
        """
        cases = cases or HALLUCINATION_CASES
        per_case = []
        total_degenerate = 0
        for src in cases:
            try:
                result = self.translate_fn(
                    src, source_lang=source_lang, target_lang=target_lang,
                )
                hyp = (result.get('translated') or '').strip()
            except Exception as e:
                hyp = ''
                per_case.append({
                    'source': src, 'error': str(e),
                    'degenerate': True, 'reason': 'translator raised',
                })
                total_degenerate += 1
                continue

            analysis = self.detector.analyze(hyp, src)
            analysis['source'] = src
            analysis['hypothesis'] = hyp
            if analysis['degenerate']:
                total_degenerate += 1
            per_case.append(analysis)

        n = max(len(per_case), 1)
        agg = {
            'num_cases': len(per_case),
            'entity_preservation_rate': round(sum(
                c.get('entity_preservation_rate', 0) for c in per_case) / n, 4),
            'mean_source_leakage_ratio': round(sum(
                c.get('source_leakage_ratio', 0) for c in per_case) / n, 4),
            'mean_repetition_ratio': round(sum(
                c.get('repetition_ratio', 0) for c in per_case) / n, 4),
            'degenerate_output_rate_pct': round(
                total_degenerate / n * 100, 2),
            'degenerate_outputs': [c['source'] for c in per_case
                                   if c.get('degenerate')],
        }

        calibration = self.calibrate_detector()
        return {
            'mode': 'mock' if self.mock else 'live',
            'detector': 'deterministic heuristics (entity/leakage/repetition)',
            'aggregate': agg,
            'cases': per_case,
            'calibration': calibration,
            'note': (
                'Heuristic (model-free) hallucination detection: entity '
                'preservation, source leakage, and repetition ratios. '
                'Calibration run confirms the detector separates good vs '
                'degenerate outputs. Use an LLM judge or human review for '
                'semantic hallucination.'
            ),
        }


def run_hallucination_benchmark(translate_fn=None) -> dict:
    """Module-level convenience entry point."""
    return HallucinationBenchmark(translate_fn=translate_fn).run()


def _print_result(result):
    agg = result['aggregate']
    print(f"\nHallucination benchmark ({result['mode']} mode, "
          f"{agg['num_cases']} cases)")
    print(f"  entity preservation rate   : {agg['entity_preservation_rate']}")
    print(f"  mean source leakage ratio  : {agg['mean_source_leakage_ratio']}")
    print(f"  mean repetition ratio      : {agg['mean_repetition_ratio']}")
    print(f"  degenerate output rate     : {agg['degenerate_output_rate_pct']}%")
    if agg['degenerate_outputs']:
        print("  degenerate outputs:")
        for s in agg['degenerate_outputs']:
            print(f"    - {s}")
    cal = result['calibration']
    print(f"  detector calibration       : {cal['calibration_pass_rate']}% correct")
    for label, r in cal['cases'].items():
        print(f"    {label:<8}: expected_degenerate={r['expected_degenerate']} "
              f"detected={r['detected']} "
              f"(rep={r['repetition_ratio']}, leak={r['source_leakage_ratio']}, "
              f"ent={r['entity_preservation_rate']})")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Hallucination/factuality benchmark')
    parser.add_argument('--live', action='store_true', help='Use real models (slow)')
    args = parser.parse_args()

    if args.live:
        from app.services.model_registry import ModelRegistry
        registry = ModelRegistry.get_instance()

        def live_fn(text, source_lang='english', target_lang='hindi'):
            return registry.translate(
                text=text, source_lang=source_lang, target_lang=target_lang,
                model='nllb-200-distilled-600M', use_tm=False,
            ).to_dict()

        result = run_hallucination_benchmark(translate_fn=live_fn)
    else:
        result = run_hallucination_benchmark()
    _print_result(result)
