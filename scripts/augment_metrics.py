"""
augment_metrics.py — add BERTScore/COMET to an existing benchmark report.

The full `--metrics all` run OOM-killed on this machine because NLLB +
BERTScore + COMET cannot fit in RAM simultaneously (15.6 GB total, ~1 GB
free). This script sidesteps that entirely: it loads a completed benchmark
JSON, re-scores the STORED translations (source/hypothesis/reference) with
BERTScore and/or COMET — loading only ONE neural metric model at a time —
and merges the results back into the report.

Usage:
    python scripts/augment_metrics.py --base PATH.json --metric bertscore
    python scripts/augment_metrics.py --base PATH.json --metric comet
    python scripts/augment_metrics.py --base PATH.json --metric all   # both

The script mirrors the Evaluator's exact call signatures so the numbers
match what a genuine `--metrics all` run would have produced.
"""

import argparse
import gc
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('augment_metrics')


# Sections that contain per-language per-model results with stored translations
SECTIONS = ('translation', 'code_mixed', 'flores_translation')


def load_report(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_report(results, path):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)
    logger.info('Saved %s', path)


def collect_sections(results):
    """Yield (section, lang, model_name, model_result) tuples in a stable order."""
    for section in SECTIONS:
        block = results.get(section)
        if not isinstance(block, dict):
            continue
        for lang, lang_res in block.items():
            if not isinstance(lang_res, dict):
                continue
            models = lang_res.get('models')
            if not isinstance(models, dict):
                continue
            for model_name, model_res in models.items():
                yield section, lang, model_name, model_res


def get_pairs(model_res):
    """Rebuild (sources, references, hypotheses) lists from stored translations."""
    translations = model_res.get('translations') or []
    sources, refs, hyps = [], [], []
    for t in translations:
        if not isinstance(t, dict):
            continue
        sources.append(t.get('source', ''))
        refs.append(t.get('reference', ''))
        hyps.append(t.get('hypothesis', ''))
    return sources, [[r] for r in refs], hyps


def run_metric(model_res, metric, scorer):
    """Score one model result with the given metric; returns (ok, detail)."""
    sources, refs, hyps = get_pairs(model_res)
    if not hyps:
        return False, 'no stored translations'

    metrics = model_res.setdefault('metrics', {})
    summary = model_res.setdefault('summary', {})

    t0 = time.time()
    if metric == 'bertscore':
        score_result = scorer.score(hyps, refs)
        summary_key = 'bertscore_f1'
        value = score_result.get('f1')
    elif metric == 'comet':
        score_result = scorer.score(hyps, refs, sources=sources)
        summary_key = 'comet_score'
        value = score_result.get('comet_score')
    else:
        return False, f'unknown metric {metric}'
    elapsed = round((time.time() - t0) * 1000, 2)

    if 'error' in score_result:
        metrics[metric] = {'error': score_result['error']}
        return False, f"scoring error: {score_result['error']}"

    metrics[metric] = {**score_result, 'computation_time_ms': elapsed}
    if value is not None:
        summary[summary_key] = round(value, 4)
    logger.info('  %s %s: %s=%s (%.1fs)',
                model_res.get('lang_pair', '?'), metric,
                summary_key, summary.get(summary_key), elapsed / 1000)
    return True, 'ok'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True, metavar='JSON',
                   help='Benchmark results JSON to augment (edited in place)')
    p.add_argument('--metric', required=True, choices=['bertscore', 'comet', 'all'])
    args = p.parse_args()

    results = load_report(args.base)

    metrics = (['bertscore', 'comet'] if args.metric == 'all'
               else [args.metric])

    for metric in metrics:
        if metric == 'bertscore':
            from app.evaluation.metrics.bertscore import BERTScoreScorer
            scorer = BERTScoreScorer()
        else:
            from app.evaluation.metrics.comet import COMETScorer
            scorer = COMETScorer()

        if not scorer.available:
            logger.error('%s unavailable — aborting', metric)
            sys.exit(1)

        logger.info('Scoring %s across stored translations...', metric)
        done, skipped, failed = 0, 0, 0
        for section, lang, model_name, model_res in collect_sections(results):
            pair = f'{section}:{lang}:{model_name}'
            logger.info('[%s] %s', metric, pair)
            ok, detail = run_metric(model_res, metric, scorer)
            if detail == 'ok':
                done += 1
            elif detail.startswith('already') or detail.startswith('no stored'):
                skipped += 1
            else:
                failed += 1
                logger.warning('  %s FAILED: %s', pair, detail)

        logger.info('%s done: %d scored, %d skipped, %d failed',
                    metric, done, skipped, failed)

        # Free the big model before the next metric
        del scorer
        gc.collect()

    save_report(results, args.base)
    logger.info('Done. Report updated in place: %s', args.base)


if __name__ == '__main__':
    main()
