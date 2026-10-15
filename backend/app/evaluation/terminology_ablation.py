"""
Terminology ablation: does enforcing glossary terms help, and at what cost?

Protocol (Dinu et al., ACL 2019, "Training NMT to Apply Terminology
Constraints"): from a parallel test set, keep the sentence pairs where a
glossary term occurs in the source AND its glossary translation occurs in
the reference. On that term-bearing subset the reference agrees with the
glossary, so chrF++ does not penalise correct term usage, and the term
accuracy has a well-defined target.

Every strategy in app/services/terminology.STRATEGIES is run on the same
sentences with the same model. Reported per strategy:

  term_accuracy    required glossary translations present in the output
  chrf_plus, bleu  corpus scores vs the references
  delta_chrf       chrF++ minus the 'none' baseline, with a paired-bootstrap
                   95% CI and p-value (sacrebleu.significance)
  applied_rate     share of sentences where the strategy changed decoding
  fallback_rate    share where it had to fall back to plain translation
  p50/p95 latency

A subset smaller than ``min_sentences`` (default 200) is reported with
``underpowered: true`` instead of being silently used.
"""

import logging
import math
import time
from typing import Callable, Dict, List, Optional

from app.services.terminology import STRATEGIES, match_terms, term_accuracy, translate_with_terms

logger = logging.getLogger(__name__)

MIN_SENTENCES = 200


def select_term_bearing(sources: List[str], references: List[str],
                        glossary: List[Dict]) -> List[Dict]:
    """
    Keep pairs where a glossary term is in the source and its translation is
    in the reference. Returns [{index, source, reference, required: [...]}].
    """
    selected = []
    for i, (src, ref) in enumerate(zip(sources, references)):
        if not src or not ref:
            continue
        required = []
        for m in match_terms(src, glossary):
            if m.translation in ref and m.translation not in required:
                required.append(m.translation)
        if required:
            selected.append({'index': i, 'source': src, 'reference': ref,
                             'required': required})
    return selected


def _percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def corpus_scores(hypotheses: List[str], references: List[str]) -> Dict:
    from sacrebleu.metrics import BLEU, CHRF
    chrf = CHRF(word_order=2).corpus_score(hypotheses, [references]).score
    try:  # spBLEU is the FLORES-200 convention; needs the flores200 SPM model
        bleu_metric, tok = BLEU(tokenize='flores200'), 'flores200'
        bleu = bleu_metric.corpus_score(hypotheses, [references]).score
    except Exception:
        bleu_metric, tok = BLEU(), '13a'
        bleu = bleu_metric.corpus_score(hypotheses, [references]).score
    return {'chrf_plus': round(chrf, 2), 'bleu': round(bleu, 2), 'bleu_tokenize': tok}


def paired_chrf_test(baseline: List[str], system: List[str], references: List[str],
                     n_samples: int = 1000) -> Dict:
    """Paired bootstrap resampling for chrF++ (system vs baseline)."""
    from sacrebleu.metrics import CHRF
    from sacrebleu.significance import PairedTest
    test = PairedTest([('baseline', baseline), ('system', system)],
                      {'chrF++': CHRF(word_order=2)}, [references],
                      test_type='bs', n_samples=n_samples)
    signatures, results = test()
    # Results are keyed by the metric's display name (e.g. 'chrF2++'), next to
    # a 'System' column; take the one metric we passed.
    key = next(iter(signatures))
    base, sys_ = results[key][0], results[key][1]
    return {
        'delta_chrf': round(sys_.score - base.score, 2),
        'ci95_halfwidth': round(float(sys_.ci), 2) if sys_.ci is not None else None,
        'p_value': round(float(sys_.p_value), 4) if sys_.p_value is not None else None,
    }


def run_strategy(translate_fn: Callable, subset: List[Dict], glossary: List[Dict],
                 strategy: str, **gen_kwargs) -> Dict:
    """Translate the subset with one strategy and collect per-sentence data."""
    hyps, latencies = [], []
    applied = fallback = errors = 0
    hits = total = 0
    for row in subset:
        t0 = time.time()
        try:
            result, info = translate_with_terms(
                translate_fn, row['source'], glossary, strategy,
                use_tm=False, **gen_kwargs)
            hyp = result.translated or ''
            if result.error:
                errors += 1
        except Exception as e:  # keep going; a crash is a data point
            logger.warning(f'{strategy}: translation failed: {e}')
            hyp, info = '', None
            errors += 1
        latencies.append((time.time() - t0) * 1000)
        hyps.append(hyp)
        if info is not None:
            applied += info.applied
            fallback += info.fallback is not None
        h, t = term_accuracy(hyp, row['required'])
        hits, total = hits + h, total + t
    n = max(len(subset), 1)
    return {
        'hypotheses': hyps,
        'term_accuracy': round(hits / total, 4) if total else None,
        'terms_found': hits,
        'terms_required': total,
        'applied_rate': round(applied / n, 4),
        'fallback_rate': round(fallback / n, 4),
        'errors': errors,
        'p50_ms': round(_percentile(latencies, 50) or 0, 1),
        'p95_ms': round(_percentile(latencies, 95) or 0, 1),
    }


def run_ablation(translate_fn: Callable, sources: List[str], references: List[str],
                 glossary: List[Dict], strategies=STRATEGIES,
                 min_sentences: int = MIN_SENTENCES, max_sentences: Optional[int] = None,
                 n_bootstrap: int = 1000, **gen_kwargs) -> Dict:
    """
    Run every strategy on the term-bearing subset of (sources, references).

    ``translate_fn(text, **kwargs)`` must return a TranslationResult-like
    object; kwargs include use_tm/context/force_words/num_beams.
    """
    strategies = list(strategies)
    if 'none' not in strategies:
        strategies.insert(0, 'none')
    subset = select_term_bearing(sources, references, glossary)
    if max_sentences:
        subset = subset[:max_sentences]
    out = {
        'num_candidates': len(sources),
        'num_sentences': len(subset),
        'num_required_terms': sum(len(r['required']) for r in subset),
        'glossary_size': len(glossary),
        'underpowered': len(subset) < min_sentences,
        'min_sentences': min_sentences,
        'strategies': {},
    }
    if not subset:
        out['error'] = 'no term-bearing sentences: the glossary does not occur in this test set'
        return out
    if out['underpowered']:
        logger.warning(f'Only {len(subset)} term-bearing sentences (< {min_sentences}); '
                       'results are underpowered. Use a larger glossary.')

    refs = [r['reference'] for r in subset]
    runs = {s: run_strategy(translate_fn, subset, glossary, s, **gen_kwargs)
            for s in strategies}
    baseline = runs['none']['hypotheses']
    for name, run in runs.items():
        entry = {k: v for k, v in run.items() if k != 'hypotheses'}
        entry.update(corpus_scores(run['hypotheses'], refs))
        if name != 'none':
            entry.update(paired_chrf_test(baseline, run['hypotheses'], refs,
                                          n_samples=n_bootstrap))
        out['strategies'][name] = entry
    out['examples'] = [
        {'source': subset[i]['source'], 'reference': subset[i]['reference'],
         'required': subset[i]['required'],
         **{name: runs[name]['hypotheses'][i] for name in runs}}
        for i in range(min(5, len(subset)))
    ]
    return out


def render_markdown(results: Dict) -> str:
    """Markdown tables for a {lang: run_ablation(...)} result."""
    lines = ['## Terminology ablation (term-bearing subset)', '']
    for lang, res in results.items():
        if not isinstance(res, dict) or 'strategies' not in res:
            continue
        flag = ' — **underpowered**' if res.get('underpowered') else ''
        lines.append(f"### en→{lang}: {res['num_sentences']} sentences, "
                     f"{res['num_required_terms']} required terms, "
                     f"glossary {res['glossary_size']} entries{flag}")
        lines.append('')
        if res.get('error'):
            lines.append(f"_{res['error']}_")
            lines.append('')
            continue
        lines.append('| Strategy | Term acc. | chrF++ | Δ chrF++ (95% CI) | p | BLEU '
                     '| Applied | Fallback | P95 ms |')
        lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
        for name, s in res['strategies'].items():
            acc = f"{s['term_accuracy'] * 100:.1f}%" if s.get('term_accuracy') is not None else 'n/a'
            delta = ('—' if name == 'none' else
                     f"{s['delta_chrf']:+.2f} (±{s['ci95_halfwidth']})")
            p = '—' if name == 'none' else f"{s['p_value']}"
            lines.append(f"| {name} | {acc} | {s['chrf_plus']} | {delta} | {p} | {s['bleu']} "
                         f"| {s['applied_rate'] * 100:.0f}% | {s['fallback_rate'] * 100:.0f}% "
                         f"| {s['p95_ms']} |")
        lines.append('')
    return '\n'.join(lines)
