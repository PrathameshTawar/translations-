"""
Benchmark report builder — scorecard + JSON/Markdown rendering.

Produces the "master scorecard" from the plan:

    | Category                  | Target          | Measured | Status |
    |---------------------------|-----------------|----------|--------|
    | API functional tests      | >= 99% pass     | ...      | PASS   |
    | Unicode test cases        | 100% pass       | ...      | PASS   |
    | Retrieval Recall@5        | >= 90% target   | ...      | ...    |
    | Translation-memory hit    | < 300 ms        | ...      | ...    |
    | P95 translation latency   | < 2-3 s         | ...      | ...    |

Translation quality metrics (BLEU / chrF++ / BERTScore / COMET) are
benchmark-dependent — reported as-is, never asserted against invented
targets. The paper should report the actual measured values.
"""

import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


DEFAULT_TARGETS = {
    'functional_pass_rate': {'target': 99.0, 'op': '>=', 'unit': '%',
                             'label': 'API functional tests'},
    'security_pass_rate': {'target': 99.0, 'op': '>=', 'unit': '%',
                           'label': 'Security/robustness tests'},
    'e2e_pass_rate': {'target': 100.0, 'op': '>=', 'unit': '%',
                      'label': 'End-to-end pipeline tests'},
    'unicode_pass_rate': {'target': 100.0, 'op': '==', 'unit': '%',
                          'label': 'Unicode test cases'},
    'retrieval_recall_at_5': {'target': 90.0, 'op': '>=', 'unit': '%',
                              'label': 'Retrieval Recall@5 (target)'},
    'tm_hit_p95_ms': {'target': 300.0, 'op': '<=', 'unit': 'ms',
                      'label': 'TM hit latency (P95)'},
    'translation_p95_ms': {'target': 3000.0, 'op': '<=', 'unit': 'ms',
                           'label': 'P95 translation latency'},
    'language_detection_f1': {'target': 0.95, 'op': '>=', 'unit': '',
                              'label': 'Language detection F1 (macro)'},
    'quality_estimation_f1': {'target': 0.95, 'op': '>=', 'unit': '',
                              'label': 'Quality estimation F1'},  # COMETkiwi reference-free target
    'hallucination_rate_pct': {'target': 5.0, 'op': '<=', 'unit': '%',
                               'label': 'Degenerate/hallucinated outputs'},
}


def deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge override into base (new values win; dicts merge by key).

    Used for staged/continued benchmark runs: a later run's sections overwrite
    earlier ones, while per-language translation results accumulate.
    """
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _check(value, op, target):
    if value is None:
        return None
    if op == '>=':
        return value >= target
    if op == '<=':
        return value <= target
    if op == '==':
        return value == target
    return None


def build_scorecard(results: dict, targets: dict = None) -> dict:
    """
    Build the master scorecard from benchmark results.

    Args:
        results: dict produced by the master benchmark runner with keys like
                 'functional', 'unicode', 'quality_estimation', 'retrieval',
                 'translation_memory', 'translation' (per-language), etc.
        targets: Target table (defaults to DEFAULT_TARGETS).

    Returns:
        dict: {entries: [...], passes, failures, total_targets}
    """
    targets = targets or DEFAULT_TARGETS

    def _get(path, default=None):
        node = results
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    functional = _get(['functional'])
    unicode_res = _get(['unicode'])
    qe = _get(['quality_estimation'])
    retrieval = _get(['retrieval'])
    tm = _get(['translation_memory'])
    translation = _get(['translation']) or {}

    entries = []

    # -- functional ---------------------------------------------------------
    func_rate = functional.get('pass_rate') if isinstance(functional, dict) else None
    entries.append({
        'key': 'functional_pass_rate',
        'label': 'API functional tests',
        'measured': func_rate,
        'display': f'{func_rate:.1f}%' if func_rate is not None else 'n/a',
        'target': targets['functional_pass_rate'],
    })

    # -- unicode ------------------------------------------------------------
    uni_rate = unicode_res.get('pass_rate') if isinstance(unicode_res, dict) else None
    entries.append({
        'key': 'unicode_pass_rate',
        'label': 'Unicode test cases',
        'measured': uni_rate,
        'display': f'{uni_rate:.1f}%' if uni_rate is not None else 'n/a',
        'target': targets['unicode_pass_rate'],
    })

    # -- security / robustness ----------------------------------------------
    security = _get(['security'])
    sec_rate = security.get('pass_rate') if isinstance(security, dict) else None
    entries.append({
        'key': 'security_pass_rate',
        'label': 'Security/robustness tests',
        'measured': sec_rate,
        'display': f'{sec_rate:.1f}%' if sec_rate is not None else 'n/a',
        'target': targets['security_pass_rate'],
    })

    # -- end-to-end pipeline ------------------------------------------------
    e2e = _get(['e2e'])
    e2e_rate = e2e.get('pass_rate') if isinstance(e2e, dict) else None
    entries.append({
        'key': 'e2e_pass_rate',
        'label': 'End-to-end pipeline tests',
        'measured': e2e_rate,
        'display': f'{e2e_rate:.1f}%' if e2e_rate is not None else 'n/a',
        'target': targets['e2e_pass_rate'],
    })

    # -- hallucination / degenerate outputs ---------------------------------
    hal = _get(['hallucination'])
    hal_rate = None
    if isinstance(hal, dict):
        hal_rate = hal.get('aggregate', {}).get('degenerate_output_rate_pct')
    entries.append({
        'key': 'hallucination_rate_pct',
        'label': 'Degenerate/hallucinated outputs',
        'measured': hal_rate,
        'display': f'{hal_rate:.1f}%' if hal_rate is not None else 'n/a',
        'target': targets['hallucination_rate_pct'],
    })

    # -- quality estimation -------------------------------------------------
    qe_f1 = qe.get('f1') if isinstance(qe, dict) else None
    entries.append({
        'key': 'quality_estimation_f1',
        'label': 'Quality estimation F1',
        'measured': qe_f1,
        'display': f'{qe_f1:.2f}' if qe_f1 is not None else 'n/a',
        'target': targets['quality_estimation_f1'],
    })

    # -- retrieval ----------------------------------------------------------
    ret_recall5 = None
    if isinstance(retrieval, dict):
        for lang, m in retrieval.get('languages', {}).items():
            if isinstance(m, dict) and 'recall_at_k' in m:
                # Keys may be int (in-memory) or str (after a JSON round-trip
                # in staged/continued runs) — resolve both.
                ret_recall5 = _get_ratio_value(m['recall_at_k'], 5)
                break
    # recall is stored as a 0-1 ratio; targets are in percent
    ret_recall5_pct = ret_recall5 * 100 if ret_recall5 is not None else None
    entries.append({
        'key': 'retrieval_recall_at_5',
        'label': 'Retrieval Recall@5 (target)',
        'measured': ret_recall5_pct,
        'display': f'{ret_recall5_pct:.1f}%' if ret_recall5_pct is not None else 'n/a',
        'target': targets['retrieval_recall_at_5'],
    })

    # -- translation memory -------------------------------------------------
    tm_p95 = None
    if isinstance(tm, dict):
        tm_p95 = tm.get('tm_hit', {}).get('percentiles', {}).get('p95_ms')
    entries.append({
        'key': 'tm_hit_p95_ms',
        'label': 'TM hit latency (P95)',
        'measured': tm_p95,
        'display': f'{tm_p95:.0f} ms' if tm_p95 is not None else 'n/a',
        'target': targets['tm_hit_p95_ms'],
    })

    # -- language detection ------------------------------------------------
    langdet = _get(['language_detection'])
    lang_f1 = langdet.get('macro_f1') if isinstance(langdet, dict) else None
    entries.append({
        'key': 'language_detection_f1',
        'label': 'Language detection F1 (macro)',
        'measured': lang_f1,
        'display': f'{lang_f1:.2f}' if lang_f1 is not None else 'n/a',
        'target': targets['language_detection_f1'],
    })

    # -- translation latency (max P95 across languages) ---------------------
    trans_p95 = None
    for lang, lang_res in translation.items():
        if not isinstance(lang_res, dict):
            continue
        for model_name, model_res in (lang_res.get('models') or {}).items():
            lat = model_res.get('latency') or {}
            p95 = lat.get('p95_ms')
            if p95 is not None:
                trans_p95 = max(trans_p95 or 0.0, p95)
    entries.append({
        'key': 'translation_p95_ms',
        'label': 'P95 translation latency (worst pair)',
        'measured': trans_p95,
        'display': f'{trans_p95:.0f} ms' if trans_p95 is not None else 'n/a',
        'target': targets['translation_p95_ms'],
    })

    # -- status -------------------------------------------------------------
    passes = 0
    failures = 0
    not_measured = 0
    for e in entries:
        t = e['target']
        if e['measured'] is None:
            e['status'] = 'NOT MEASURED'
            not_measured += 1
            continue
        ok = _check(e['measured'], t['op'], t['target'])
        e['status'] = 'PASS' if ok else 'FAIL'
        if ok:
            passes += 1
        else:
            failures += 1

    return {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'entries': entries,
        'summary': {
            'pass': passes,
            'fail': failures,
            'not_measured': not_measured,
            'total_targets': len(entries),
        },
    }


def render_markdown(scorecard: dict, results: dict = None) -> str:
    """Render the scorecard (and translation quality table) as Markdown."""
    lines = []
    lines.append('=' * 62)
    lines.append('              INDIC TRANSLATOR — MASTER BENCHMARK')
    lines.append('=' * 62)
    lines.append('')

    # Scorecard table
    lines.append('## Scorecard (acceptance criteria)')
    lines.append('')
    lines.append('| Category | Target | Measured | Status |')
    lines.append('| --- | ---: | ---: | --- |')
    for e in scorecard['entries']:
        t = e['target']
        target_str = f"{t['op']} {t['target']:g}{t['unit']}"
        lines.append(f"| {e['label']} | {target_str} | {e['display']} | {e['status']} |")
    s = scorecard['summary']
    lines.append('')
    lines.append(f"Summary: {s['pass']} PASS / {s['fail']} FAIL / "
                 f"{s['not_measured']} not measured")
    lines.append('')

    # Translation quality table (benchmark-dependent, reported as measured)
    translation = (results or {}).get('translation') or {}
    # Only render when real per-language results exist (auto mode may store a
    # {'status': 'skipped', ...} marker instead of a results dict)
    model_entries = {
        lang: r for lang, r in translation.items()
        if isinstance(r, dict) and isinstance(r.get('models'), dict)
    }
    if model_entries:
        lines.append('## Translation quality (per language pair)')
        lines.append('')
        lines.append('| Lang pair | Model | BLEU | chrF++ | BERTScore-F1 | COMET | P95 (ms) |')
        lines.append('| --- | --- | ---: | ---: | ---: | ---: | ---: |')
        for lang, lang_res in model_entries.items():
            for model_name, m in (lang_res.get('models') or {}).items():
                summ = m.get('summary') or {}
                lat = m.get('latency') or {}
                lines.append(
                    f"| en→{lang} | {model_name} | "
                    f"{_fmt(summ.get('bleu'))} | {_fmt(summ.get('chrf_plus'))} | "
                    f"{_fmt(summ.get('bertscore_f1'))} | {_fmt(summ.get('comet_score'))} | "
                    f"{_fmt(lat.get('p95_ms'), 'ms')} |"
                )
        lines.append('')

    # Code-mixed quality table (Indic script + English words -> English)
    code_mixed = (results or {}).get('code_mixed')
    cm_entries = {
        lang: r for lang, r in (code_mixed or {}).items()
        if isinstance(r, dict) and isinstance(r.get('models'), dict)
    }
    if cm_entries:
        lines.append('## Code-mixed translation (Indic + English words -> English)')
        lines.append('')
        lines.append('| Lang | Model | BLEU | chrF++ | BERTScore-F1 | P95 (ms) |')
        lines.append('| --- | --- | ---: | ---: | ---: | ---: |')
        for lang, lang_res in cm_entries.items():
            for model_name, m in (lang_res.get('models') or {}).items():
                summ = m.get('summary') or {}
                lat = m.get('latency') or {}
                lines.append(
                    f"| {lang}→en | {model_name} | "
                    f"{_fmt(summ.get('bleu'))} | {_fmt(summ.get('chrf_plus'))} | "
                    f"{_fmt(summ.get('bertscore_f1'))} | "
                    f"{_fmt(lat.get('p95_ms'), 'ms')} |"
                )
        lines.append('')

    # FLORES-200 quality table (research-grade references)
    flores_translation = (results or {}).get('flores_translation') or {}
    flores_entries = {
        lang: r for lang, r in flores_translation.items()
        if isinstance(r, dict) and isinstance(r.get('models'), dict)
    }
    if flores_entries:
        lines.append('## Translation quality — FLORES-200 devtest (research-grade refs)')
        lines.append('')
        lines.append('| Lang pair | Model | BLEU | chrF++ | P95 (ms) |')
        lines.append('| --- | --- | ---: | ---: | ---: |')
        for lang, lang_res in flores_entries.items():
            for model_name, m in (lang_res.get('models') or {}).items():
                summ = m.get('summary') or {}
                lat = m.get('latency') or {}
                lines.append(
                    f"| en→{lang} | {model_name} | "
                    f"{_fmt(summ.get('bleu'))} | {_fmt(summ.get('chrf_plus'))} | "
                    f"{_fmt(lat.get('p95_ms'), 'ms')} |"
                )
        lines.append('')

    # Retrieval table
    retrieval = (results or {}).get('retrieval')
    if isinstance(retrieval, dict) and retrieval.get('languages'):
        lines.append('## Retrieval (terminology RAG)')
        lines.append('')
        lines.append(f"Backend: **{retrieval.get('backend', 'unknown')}** — "
                     f"{retrieval.get('num_queries_total', 0)} queries")
        lines.append('')
        lines.append('| Lang | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |')
        lines.append('| --- | ---: | ---: | ---: | ---: | ---: |')
        for lang, m in retrieval['languages'].items():
            if 'error' in m:
                lines.append(f"| {lang} | {m['error']} | | | | |")
                continue
            lines.append(
                f"| {lang} | {_ratio_pct(m.get('recall_at_k'), 1)} | "
                f"{_ratio_pct(m.get('recall_at_k'), 3)} | "
                f"{_ratio_pct(m.get('recall_at_k'), 5)} | "
                f"{_fmt(m.get('mrr'))} | {_ratio_pct(m.get('ndcg_at_k'), 5)} |"
            )
        lines.append('')

    # Reranker ablation (cross-encoder on/off)
    reranker = (retrieval or {}).get('reranker') or {}
    if reranker.get('available'):
        lines.append('### Reranker ablation (cross-encoder on/off)')
        lines.append('')
        lines.append(f"Model: **{reranker.get('model')}** · rerank overhead: "
                     f"{_fmt(reranker.get('rerank_overhead_ms'))} ms/query")
        lines.append('')
        lines.append('| Query variant | Metric | Dense | Dense + rerank | Δ |')
        lines.append('| --- | --- | ---: | ---: | ---: |')
        metric_order = ('recall_at_1', 'recall_at_3', 'recall_at_5',
                        'recall_at_10', 'mrr', 'ndcg_at_5')
        for variant, label in (
                ('term_queries', 'Term + context queries'),
                ('context_queries', 'Context-only queries (harder)'),
        ):
            block = reranker.get(variant) or {}
            dense = block.get('dense') or {}
            reranked = block.get('reranked') or {}
            delta = block.get('delta') or {}
            for metric in metric_order:
                if metric not in dense:
                    continue
                d = dense.get(metric)
                r = reranked.get(metric)
                dl = delta.get(metric)
                sign = '+' if (dl is not None and dl >= 0) else ''
                lines.append(
                    f"| {label} | {metric} | {_fmt(d)} | {_fmt(r)} | "
                    f"{sign}{_fmt(dl)} |"
                )
        lines.append(
            f"| — | Query latency mean (ms) | "
            f"{_fmt(reranker.get('dense_query_latency_ms'))} | "
            f"{_fmt(reranker.get('reranked_query_latency_ms'))} | "
            f"+{_fmt(reranker.get('rerank_overhead_ms'))} |"
        )
        lines.append('')
        lines.append('*Deltas are macro-averaged over languages. On these small '
                     'glossaries the dense bi-encoder baseline is often at '
                     'ceiling, so the reranker mostly adds latency — its '
                     'Recall@5 gain in the paper (+3.9 pts) is specific to '
                     'the open-vocabulary passage-retrieval task, not to '
                     'ceiling-level glossary lookup.*')
        lines.append('')

    # Open-vocabulary passage retrieval (reranker where it should help)
    passage = (results or {}).get('passage_retrieval')
    if isinstance(passage, dict) and not passage.get('error') and passage.get('dense'):
        lines.append('## Passage retrieval (open-vocabulary, rerank ablation)')
        lines.append('')
        lines.append(f"Corpus: **{passage['corpus']['num_passages']} passages** "
                     f"({passage['corpus']['source']}) · queries: "
                     f"{passage['num_queries']} ({passage['queries_mode']})")
        lines.append('')
        lines.append('| Metric | Dense | Dense + rerank | Δ |')
        lines.append('| --- | ---: | ---: | ---: |')
        metric_order = ('recall_at_1', 'recall_at_3', 'recall_at_5', 'mrr')
        rer = passage.get('reranker') or {}
        reranked = rer.get('reranked') or {}
        delta = rer.get('delta') or {}
        for metric in metric_order:
            if metric not in passage['dense']:
                continue
            d = passage['dense'].get(metric)
            r = reranked.get(metric)
            dl = delta.get(metric)
            sign = '+' if (dl is not None and dl >= 0) else ''
            lines.append(
                f"| {metric} | {_fmt(d)} | {_fmt(r)} | {sign}{_fmt(dl)} |"
            )
        if rer.get('available'):
            lines.append(
                f"| — | Query latency mean (ms) | "
                f"{_fmt(passage.get('dense_query_latency_ms'))} | "
                f"{_fmt(rer.get('reranked_query_latency_ms'))} | "
                f"+{_fmt(rer.get('rerank_overhead_ms'))} |"
            )
            lines.append('')
            lines.append(f"*Reranker: {rer.get('model')} re-scores the dense "
                         'top-10 pool. Queries are paraphrases (NLLB round-trip '
                         'en→hi→en) or masked passages, so the bi-encoder is '
                         'off ceiling — this is the open-vocabulary regime '
                         "where the paper's +3.9 Recall@5 claim applies.*")
        else:
            lines.append('')
            lines.append('*Reranker unavailable — dense-only numbers shown.*')
        lines.append('')

    # TM table
    tm = (results or {}).get('translation_memory')
    if isinstance(tm, dict) and tm.get('lang_pair'):
        lines.append('## Translation memory (with vs without)')
        lines.append('')
        lines.append(f"Pair: **{tm['lang_pair']}** · sentences: {tm['num_sentences']} "
                     f"· encoder: {tm['encoder_available']}")
        lines.append('')
        lines.append('| Metric | Value |')
        lines.append('| --- | ---: |')
        lines.append(f"| Model latency mean | {tm['model']['mean_ms']:.0f} ms |")
        lines.append(f"| TM-hit latency mean | {tm['tm_hit']['mean_ms']:.0f} ms |")
        lines.append(f"| Exact hit rate | {tm['exact_hit_rate'] * 100:.1f}% |")
        lines.append(f"| Semantic hit rate | {tm['semantic_hit_rate'] * 100:.1f}% |")
        lines.append(f"| Latency reduction | {tm['latency_reduction_pct']:.1f}% |")
        lines.append('')

    # Quality estimation
    qe = (results or {}).get('quality_estimation')
    if isinstance(qe, dict) and qe.get('num_samples'):
        lines.append('## Quality estimation (good vs bad detection)')
        lines.append('')
        lines.append(f"| Metric | Value |")
        lines.append(f"| --- | ---: |")
        lines.append(f"| Scorer | {qe.get('scorer', 'heuristic')} |")
        lines.append(f"| Samples | {qe['num_samples']} |")
        lines.append(f"| Accuracy | {qe['accuracy']} |")
        lines.append(f"| Precision | {qe['precision']} |")
        lines.append(f"| Recall | {qe['recall']} |")
        lines.append(f"| F1 | {qe['f1']} |")
        lines.append(f"| AUROC | {qe['auroc'] if qe['auroc'] is not None else 'n/a'} |")
        if 'word_order_flagged' in qe:
            lines.append(f"| Word-order violations flagged | "
                         f"{qe['word_order_flagged']} |")
        lines.append('')

        scorers = qe.get('scorers')
        if scorers:
            lines.append('Reference-free QE engines — head-to-head:')
            lines.append('')
            lines.append('| Metric | Heuristic | COMETkiwi |')
            lines.append('| --- | ---: | ---: |')
            for metric in ('f1', 'accuracy', 'precision', 'recall', 'auroc'):
                cells = []
                for which in ('heuristic', 'cometkiwi'):
                    m = (scorers.get(which) or {})
                    if not m.get('available'):
                        cells.append('unavail')
                    else:
                        v = m.get(metric)
                        cells.append(f'{v:g}' if isinstance(v, float) else 'n/a')
                lines.append(f"| {metric} | {cells[0]} | {cells[1]} |")
            lines.append(f"| Winner (F1) | — | **{qe.get('winner')}** |")
            if qe.get('f1_delta') is not None:
                lines.append(f"| Δ F1 vs heuristic | — | +{qe['f1_delta']} |")
            lines.append('')
            lines.append('*COMETkiwi is reference-free neural QE '
                         '(Unbabel/wmt22-cometkiwi-da).*')
            lines.append('')

    # Language detection
    langdet = (results or {}).get('language_detection')
    if isinstance(langdet, dict) and langdet.get('per_class'):
        lines.append('## Language detection (Unicode-script detector)')
        lines.append('')
        lines.append(f"Detector: {langdet.get('detector')} · samples: "
                     f"{langdet.get('num_samples')} · test: {langdet.get('num_test')}")
        lines.append('')
        lines.append('| Metric | Value |')
        lines.append('| --- | ---: |')
        lines.append(f"| Accuracy | {langdet.get('accuracy')} |")
        lines.append(f"| Macro F1 | {langdet.get('macro_f1')} |")
        lines.append(f"| Weighted F1 | {langdet.get('weighted_f1')} |")
        if langdet.get('devanagari_macro_f1') is not None:
            lines.append(f"| Devanagari (hi/mr) F1 — learned part | "
                         f"{langdet['devanagari_macro_f1']} |")
        lines.append('')
        lines.append('Per-class (P / R / F1):')
        lines.append('')
        lines.append('| Language | P | R | F1 | n |')
        lines.append('| --- | ---: | ---: | ---: | ---: |')
        for lang, m in langdet.get('per_class', {}).items():
            lines.append(f"| {lang} | {m['precision']} | {m['recall']} | "
                         f"{m['f1']} | {m['support']} |")
        lines.append('')
        cm = langdet.get('confusion_matrix', {})
        labels = list(langdet.get('per_class', {}).keys())
        if labels:
            width = max(len(l) for l in labels) + 2
            header = ' '*width + ''.join(f'{l:>{width}}' for l in labels)
            lines.append('Confusion matrix (rows=actual, cols=predicted):')
            lines.append('')
            lines.append('```')
            lines.append(header)
            for actual in labels:
                row = f'{actual:<{width}}' + ''.join(
                    f"{cm.get(actual, {}).get(p, 0):>{width}}" for p in labels
                )
                lines.append(row)
            lines.append('```')
        lines.append('')
        lines.append('*Note:* english/bengali/gujarati/tamil/telugu/kannada are '
                     'deterministic script-range checks (hence F1 1.0 on clean '
                     'text); only the Devanagari hi/mr disambiguation is learned. '
                     'The benchmark trains the word/suffix model on a held-out '
                     'half of the data (honest generalization); the production '
                     'detector trains on the full curated set.')
        lines.append('')

    # Security & robustness
    security = (results or {}).get('security')
    if isinstance(security, dict) and security.get('cases'):
        lines.append('## Security & robustness (hostile inputs)')
        lines.append('')
        lines.append(f"Mode: **{security.get('mode', 'unknown')}** · "
                     f"{security['passed']}/{security['total']} cases "
                     f"({security['pass_rate']}%)")
        lines.append('')
        lines.append('| Category | Passed | Total | Rate |')
        lines.append('| --- | ---: | ---: | ---: |')
        for cat, m in security.get('categories', {}).items():
            lines.append(f"| {cat} | {m['passed']} | {m['total']} | "
                         f"{m['pass_rate']}% |")
        failed_cases = [c for c in security['cases'] if not c['passed']]
        if failed_cases:
            lines.append('')
            lines.append('Failing cases:')
            for c in failed_cases:
                lines.append(f"- **{c['name']}** ({c['category']}) — "
                             f"status {c['status']}, expected "
                             f"{c['expected']}; {c.get('detail') or ''}")
        lines.append('')
        lines.append(f"*{security.get('note', '')}*")
        lines.append('')

    # End-to-end pipeline
    e2e = (results or {}).get('e2e')
    if isinstance(e2e, dict) and e2e.get('stages'):
        lines.append('## End-to-end pipeline lifecycle')
        lines.append('')
        lines.append(f"Mode: **{e2e.get('mode', 'unknown')}** · "
                     f"{e2e['passed']}/{e2e['total']} stages "
                     f"({e2e['pass_rate']}%)")
        lines.append('')
        lines.append('| Stage | Status | Latency | Detail |')
        lines.append('| --- | --- | ---: | --- |')
        for s in e2e['stages']:
            status = 'PASS' if s['passed'] else 'FAIL'
            lines.append(f"| {s['name']} | {status} | {s['latency_ms']} ms | "
                         f"{s.get('detail') or ''} |")
        lines.append('')

    # Throughput & concurrency
    throughput = (results or {}).get('throughput')
    if isinstance(throughput, dict) and throughput.get('latency'):
        lines.append('## Throughput & concurrency')
        lines.append('')
        lat = throughput['latency']
        lines.append(f"Mode: **{throughput.get('mode', 'unknown')}**")
        lines.append('')
        lines.append('Sequential latency profile:')
        lines.append('')
        lines.append('| Metric | Value |')
        lines.append('| --- | ---: |')
        lines.append(f"| Calls | {lat['num_calls']} |")
        lines.append(f"| Mean | {lat['mean_ms']} ms |")
        lines.append(f"| Median (P50) | {lat['median_ms']} ms |")
        lines.append(f"| P95 | {lat['p95_ms']} ms |")
        lines.append(f"| P99 | {lat['p99_ms']} ms |")
        lines.append(f"| Min / Max | {lat['min_ms']} / {lat['max_ms']} ms |")
        lines.append(f"| Throughput | {lat['throughput_sentences_per_sec']} sents/s |")
        if lat.get('by_length'):
            lines.append('')
            lines.append('By length bucket (mean / P95):')
            for bucket, m in lat['by_length'].items():
                lines.append(f"- {bucket}: n={m['n']} mean={m['mean_ms']} ms "
                             f"p95={m['p95_ms']} ms")
        lines.append('')
        lines.append('Concurrency stress test:')
        lines.append('')
        lines.append('| Concurrency | Requests | Wall (s) | req/s | mean (ms) | '
                     'P95 (ms) | err% |')
        lines.append('| ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
        for lv in throughput['concurrency']['levels']:
            lines.append(
                f"| {lv['concurrency']} | {lv['total_requests']} | "
                f"{lv['wall_time_sec']} | {lv['throughput_req_per_sec']} | "
                f"{lv['mean_ms']} | {lv['p95_ms']} | {lv['error_rate_pct']} |")
        lines.append('')
        lines.append(f"Peak throughput: "
                     f"**{throughput['concurrency']['peak_req_per_sec']} req/s**")
        if throughput.get('peak_memory_mb'):
            lines.append(f"Peak RSS: {throughput['peak_memory_mb']} MB")
        lines.append('')
        lines.append(f"*{throughput.get('note', '')}*")
        lines.append('')

    # Hallucination / degenerate outputs
    hallucination = (results or {}).get('hallucination')
    if isinstance(hallucination, dict) and hallucination.get('aggregate'):
        lines.append('## Hallucination & degenerate outputs')
        lines.append('')
        agg = hallucination['aggregate']
        lines.append(f"Mode: **{hallucination.get('mode', 'unknown')}** · "
                     f"detector: {hallucination.get('detector')}")
        lines.append('')
        lines.append('| Metric | Value |')
        lines.append('| --- | ---: |')
        lines.append(f"| Cases | {agg.get('num_cases', 'n/a')} |")
        lines.append(f"| Entity preservation | "
                     f"{agg.get('entity_preservation_rate', 'n/a')} |")
        lines.append(f"| Mean source leakage | "
                     f"{agg.get('mean_source_leakage_ratio', 'n/a')} |")
        lines.append(f"| Mean repetition | "
                     f"{agg.get('mean_repetition_ratio', 'n/a')} |")
        lines.append(f"| Degenerate output rate | "
                     f"{agg.get('degenerate_output_rate_pct', 'n/a')}% |")
        if agg.get('degenerate_outputs'):
            lines.append('')
            lines.append('Degenerate outputs:')
            for s in agg['degenerate_outputs']:
                lines.append(f"- {s}")
        cal = hallucination.get('calibration', {})
        if cal.get('cases'):
            lines.append('')
            lines.append(f"Detector calibration: **{cal['calibration_pass_rate']}%** "
                         'correct on known-good vs known-degenerate outputs '
                         f"(probe: `{cal.get('probe', '')}`)")
        lines.append('')
        lines.append(f"*{hallucination.get('note', '')}*")
        lines.append('')

    # Notes
    lines.append('## Notes')
    lines.append('')
    lines.append('- Translation quality targets are benchmark-dependent; '
                 'report actual measured values against baselines.')
    lines.append('- Curated references are machine-generated and lightly '
                 'edited — verify with native speakers or use FLORES-200 '
                 'before publication.')
    return '\n'.join(lines)


def _fmt(value, suffix=''):
    if value is None:
        return '—'
    if isinstance(value, float):
        return f'{value:g}{suffix}'
    return f'{value}{suffix}'


def _get_ratio_value(mapping, key):
    """
    Fetch a value whose key may be stored as an int OR a string.

    JSON serialization converts int dict keys to strings, so staged/continued
    benchmark runs (which merge loaded JSON) present e.g. recall_at_k as
    {'5': 1.0} instead of {5: 1.0}. Look up both spellings.
    """
    if not isinstance(mapping, dict):
        return None
    if key in mapping:
        return mapping[key]
    return mapping.get(str(key))


def _ratio_pct(mapping, key):
    """Format a 0-1 ratio stored under `key` as a percentage, robust to gaps."""
    value = _get_ratio_value(mapping, key)
    if value is None:
        return '—'
    return f'{value * 100:.1f}%'


def save_reports(results: dict, output_dir: str) -> dict:
    """
    Save the benchmark results and markdown report to disk.

    Args:
        results: Master benchmark results dict.
        output_dir: Directory to write reports into.

    Returns:
        dict with json_path, md_path, scorecard_path.
    """
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = os.path.join(output_dir, f'benchmark_{stamp}')

    scorecard = build_scorecard(results)

    json_path = f'{base}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    md_path = f'{base}.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(render_markdown(scorecard, results))

    scorecard_path = f'{base}_scorecard.json'
    with open(scorecard_path, 'w', encoding='utf-8') as f:
        json.dump(scorecard, f, indent=2, ensure_ascii=False, default=str)

    return {
        'json_path': json_path,
        'md_path': md_path,
        'scorecard_path': scorecard_path,
    }
