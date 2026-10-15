"""
top_k ablation for the terminology RAG pipeline (plan Phase 4).

For a single top_k value, translates the curated benchmark sentences through
the real /api/v1/translate/text route (RAG context at that top_k, beam
configurable, default 2) and reports:

  * retrieval metrics   — Recall@K / MRR / P@K / nDCG@K for K=1..5, computed
                          from a single retrieval with top_k=5. Because FAISS
                          ranking is K-invariant, Recall@K from that list
                          equals what the pipeline would get if it only kept
                          the top-K terms.
  * translation quality — BLEU / chrF++ / BERTScore (xlm-roberta-large)
                          against the curated references.
  * latency             — mean/P50/P95/P99/max of total_ms, translate_ms and
                          rag_ms as measured end-to-end by the route.

Run once per top_k, then merge the outputs into the ablation table:

    python scripts/ablate_topk.py --top-k 1 --out evaluation/reports/topk1.json
    python scripts/ablate_topk.py --top-k 2 --out evaluation/reports/topk2.json
    ... --top-k 3 / --top-k 5

Run from backend/ (FAISS index files resolve relative to CWD).
"""

import argparse
import json
import os
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np  # noqa: E402


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
        'min_ms': round(min(values), 1) if values else None,
    }


def main():
    parser = argparse.ArgumentParser(description='top_k ablation (Phase 4)')
    parser.add_argument('--top-k', type=int, required=True,
                        help='Number of terminology terms to retrieve (1-10)')
    parser.add_argument('--num-beams', type=int, default=2,
                        help='Beam size for model.generate() (default 2)')
    parser.add_argument('--lang', default='hindi', help='Target language')
    parser.add_argument('--max-sentences', type=int, default=0,
                        help='Cap benchmark sentences (0 = all)')
    parser.add_argument('--out', required=True, help='Output JSON')
    args = parser.parse_args()

    if not 1 <= args.top_k <= 10:
        print('FATAL: top_k must be in 1..10')
        sys.exit(1)

    # ------------------------------------------------------------------
    # App + temp DB + test client (real model, real RAG)
    # ------------------------------------------------------------------
    from app import create_app
    from app.extensions import db

    app = create_app('development')
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_path.replace('\\', '/')
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
    # Retrieval metrics (K-invariant; one retrieval with top_k=5)
    # ------------------------------------------------------------------
    from app.evaluation.retrieval import TerminologyRetrievalBenchmark
    ret = TerminologyRetrievalBenchmark(use_encoder=True).run(
        langs=[args.lang], top_ks=(1, 2, 3, 5))

    # ------------------------------------------------------------------
    # Translation + latency at this top_k (beam fixed at 4)
    # ------------------------------------------------------------------
    hypotheses, refs, srcs = [], [], []
    total_ms, translate_ms, rag_ms = [], [], []
    errors = 0
    for i, s in enumerate(sentences):
        r = client.post('/api/v1/translate/text', json={
            'text': s,
            'source_lang': 'english',
            'target_lang': args.lang,
            'model': 'nllb-200-distilled-600M',
            'num_beams': args.num_beams,
            'include_context': True,
            'top_k': args.top_k,
        })
        if r.status_code != 200:
            errors += 1
            continue
        body = r.get_json()
        hypotheses.append(body.get('translated_text', ''))
        refs.append([references[i]])
        srcs.append(s)
        t = body.get('timings', {})
        total_ms.append(t.get('total_ms'))
        translate_ms.append(t.get('translate_ms'))
        rag_ms.append(t.get('rag_ms'))

    # ------------------------------------------------------------------
    # Translation quality
    # ------------------------------------------------------------------
    from app.evaluation.evaluator import Evaluator
    evaluator = Evaluator(metrics=['bleu', 'chrf', 'bertscore'])
    quality = evaluator.evaluate(
        hypotheses=hypotheses,
        references=refs,
        sources=srcs,
        model_name='nllb-200-distilled-600M',
        lang_pair=f'english->{args.lang}',
    )

    report = {
        'top_k': args.top_k,
        'lang': args.lang,
        'num_sentences': len(sentences),
        'errors': errors,
        'retrieval': ret,
        'quality_summary': quality.get('summary'),
        'metrics': {k: v for k, v in quality.get('metrics', {}).items()},
        'latency': {
            'total_ms': _summarize(total_ms),
            'translate_ms': _summarize(translate_ms),
            'rag_ms': _summarize(rag_ms),
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    s = report['latency']['total_ms']
    r_rag = report['latency']['rag_ms']
    lang_ret = ret['languages'].get(args.lang, {})
    print(f"top_k={args.top_k} lang={args.lang} sentences={len(sentences)} errors={errors}")
    print(f"  total:  mean={s['mean_ms']} p50={s['p50_ms']} p95={s['p95_ms']} p99={s['p99_ms']} max={s['max_ms']}")
    print(f"  rag:    mean={r_rag['mean_ms']} p95={r_rag['p95_ms']}")
    print(f"  quality: {quality.get('summary')}")
    print(f"  retrieval({args.lang}): recall_at_k={lang_ret.get('recall_at_k')} mrr={lang_ret.get('mrr')} backend={ret['backend']}")
    print(f"Report saved: {args.out}")
    try:
        os.remove(db_path)
    except OSError:
        pass


if __name__ == '__main__':
    main()
