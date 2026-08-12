"""
Ablation runner — sweeps pipeline configuration and appends one row per run
to `ablation_results.csv` (reproduces the paper's Section VIII config
sweeps, Table IV style).

Knobs that actually change the pipeline in this system:

  --top-k          terminology terms retrieved (1..10)
  --reranker       cross-encoder reranker on/off
  --num-beams      beam size (1 greedy / 2 / 4)
  --rag            include terminology context (on) or translate plain (off)
  --lang           target language
  --max-sentences  cap the benchmark sentences (quick runs / smoke tests)

Each run measures:
  * retrieval     — Recall@K / MRR from a single top_k=5 retrieval
                    (FAISS ranking is K-invariant, so Recall@K for K<=5 is
                    exact even though the route only keeps top_k terms)
  * quality       — BLEU / chrF++ against the curated references
  * latency       — P50/P95/mean of total_ms, translate_ms and rag_ms as
                    measured end-to-end by the real /api/v1/translate/text
                    route (real NLLB model, real RAG, no stubs)

and appends a row to the CSV with all config + metrics, so sweeps can be
merged into a single table.

Example sweep (from translator-backend/, FAISS indexes resolve to CWD):

    for k in 1 2 3 5; do
        python scripts/run_ablation.py --top-k $k --num-beams 2 \
            --lang hindi --max-sentences 40
    done

    python scripts/run_ablation.py --reranker --num-beams 2 \
        --lang hindi --max-sentences 40     # reranker on vs off
    python scripts/run_ablation.py --no-rag --num-beams 1 \
        --lang hindi --max-sentences 40     # RAG on vs off, greedy vs beam
"""

import argparse
import csv
import json
import os
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np  # noqa: E402

CSV_COLUMNS = [
    'timestamp', 'lang', 'top_k', 'num_beams', 'reranker', 'rag',
    'num_sentences', 'errors',
    'recall_at_1', 'recall_at_5', 'mrr',
    'bleu', 'chrf_plus',
    'total_mean_ms', 'total_p50_ms', 'total_p95_ms', 'total_p99_ms',
    'translate_p95_ms', 'rag_p95_ms',
]


def _pct(values, p):
    return float(np.percentile(values, p)) if values else None


def _summarize(values):
    return {
        'n': len(values),
        'mean_ms': round(statistics.mean(values), 1) if values else None,
        'p50_ms': round(_pct(values, 50), 1),
        'p95_ms': round(_pct(values, 95), 1),
        'p99_ms': round(_pct(values, 99), 1),
        'max_ms': round(max(values), 1) if values else None,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Config ablation runner — appends a row to ablation_results.csv',
    )
    parser.add_argument('--top-k', type=int, default=2,
                        help='Terminology terms retrieved (1-10); default 2')
    parser.add_argument('--num-beams', type=int, default=2,
                        help='Beam size for model.generate(); default 2')
    parser.add_argument('--reranker', action='store_true',
                        help='Rescore retrieved terms with the cross-encoder '
                             '(downloads ~430MB on first use)')
    parser.add_argument('--no-rag', action='store_true',
                        help='Disable RAG context (translate plain)')
    parser.add_argument('--lang', default='hindi', help='Target language')
    parser.add_argument('--max-sentences', type=int, default=0,
                        help='Cap benchmark sentences (0 = all)')
    parser.add_argument('--csv', default='evaluation/reports/ablation_results.csv',
                        help='CSV output path (appends a row)')
    return parser.parse_args()


def main():
    args = parse_args()

    if not 1 <= args.top_k <= 10:
        print('FATAL: top_k must be in 1..10')
        sys.exit(1)
    if not 1 <= args.num_beams <= 10:
        print('FATAL: num_beams must be in 1..10')
        sys.exit(1)

    rag = not args.no_rag

    # ------------------------------------------------------------------
    # App + temp DB + test client (real model, real RAG)
    # ------------------------------------------------------------------
    from app import create_app
    from app.extensions import db

    app = create_app('development')
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_path.replace('\\\\', '/')
    app.config['TESTING'] = True
    with app.app_context():
        db.drop_all()
        db.create_all()
    client = app.test_client()

    from app.evaluation.datasets import get_benchmark_dataset
    ds = get_benchmark_dataset(target_langs=[args.lang])
    sentences = ds['sources']
    references = ds['references'].get(args.lang, [])
    if args.max_sentences:
        sentences = sentences[:args.max_sentences]
        references = references[:args.max_sentences]

    # ------------------------------------------------------------------
    # Retrieval metrics (only meaningful when RAG context is on)
    # ------------------------------------------------------------------
    recall_at_1 = recall_at_5 = mrr = None
    if rag:
        from app.evaluation.retrieval import TerminologyRetrievalBenchmark
        ret = TerminologyRetrievalBenchmark(use_encoder=True).run(
            langs=[args.lang], top_ks=(1, 5), use_reranker=args.reranker)
        lang_ret = ret['languages'].get(args.lang, {})
        if 'error' not in lang_ret:
            recall_at_1 = lang_ret['recall_at_k'].get(1)
            recall_at_5 = lang_ret['recall_at_k'].get(5)
            mrr = lang_ret.get('mrr')
        print(f"  retrieval backend={ret['backend']} "
              f"recall@1={recall_at_1} recall@5={recall_at_5} mrr={mrr}")

    # ------------------------------------------------------------------
    # Translation + latency at this config (real route)
    # ------------------------------------------------------------------
    hypotheses, refs, used_sources = [], [], []
    total_ms, translate_ms, rag_ms = [], [], []
    errors = 0
    for i, s in enumerate(sentences):
        body = {
            'text': s,
            'source_lang': 'english',
            'target_lang': args.lang,
            'model': 'nllb-200-distilled-600M',
            'num_beams': args.num_beams,
            'include_context': rag,
        }
        if rag:
            body['top_k'] = args.top_k
            body['use_reranker'] = args.reranker
        r = client.post('/api/v1/translate/text', json=body)
        if r.status_code != 200:
            errors += 1
            continue
        body_resp = r.get_json()
        hypotheses.append(body_resp.get('translated_text', ''))
        refs.append([references[i]])
        used_sources.append(s)
        t = body_resp.get('timings', {})
        total_ms.append(t.get('total_ms'))
        translate_ms.append(t.get('translate_ms'))
        rag_ms.append(t.get('rag_ms'))

    # ------------------------------------------------------------------
    # Translation quality
    # ------------------------------------------------------------------
    from app.evaluation.evaluator import Evaluator
    evaluator = Evaluator(metrics=['bleu', 'chrf'])
    quality = evaluator.evaluate(
        hypotheses=hypotheses,
        references=refs,
        sources=used_sources,
        model_name='nllb-200-distilled-600M',
        lang_pair=f'english->{args.lang}',
    )
    summary = quality.get('summary') or {}
    bleu = summary.get('bleu')
    chrf = summary.get('chrf_plus')

    s_total = _summarize(total_ms)
    s_translate = _summarize(translate_ms)
    s_rag = _summarize(rag_ms)

    # ------------------------------------------------------------------
    # Append the CSV row
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    write_header = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
    row = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'lang': args.lang,
        'top_k': args.top_k,
        'num_beams': args.num_beams,
        'reranker': int(args.reranker),
        'rag': int(rag),
        'num_sentences': len(sentences),
        'errors': errors,
        'recall_at_1': recall_at_1,
        'recall_at_5': recall_at_5,
        'mrr': mrr,
        'bleu': bleu,
        'chrf_plus': chrf,
        'total_mean_ms': s_total['mean_ms'],
        'total_p50_ms': s_total['p50_ms'],
        'total_p95_ms': s_total['p95_ms'],
        'total_p99_ms': s_total['p99_ms'],
        'translate_p95_ms': s_translate['p95_ms'],
        'rag_p95_ms': s_rag['p95_ms'],
    }
    with open(args.csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"config: top_k={args.top_k} beams={args.num_beams} "
          f"reranker={args.reranker} rag={rag} lang={args.lang}")
    print(f"  sentences={len(sentences)} errors={errors}")
    print(f"  quality: bleu={bleu} chrF++={chrf}")
    print(f"  total:   mean={s_total['mean_ms']} p50={s_total['p50_ms']} "
          f"p95={s_total['p95_ms']} p99={s_total['p99_ms']}")
    print(f"  translate p95={s_translate['p95_ms']}  rag p95={s_rag['p95_ms']}")
    print(f"Row appended: {args.csv}")
    try:
        os.remove(db_path)
    except OSError:
        pass


if __name__ == '__main__':
    main()
