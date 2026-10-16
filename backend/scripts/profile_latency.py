"""
Pipeline latency profiler — finds WHERE the time goes per request.

Measures the real /api/v1/translate/text path (Flask test client, real NLLB
model, no stubs) and reports per-stage percentiles (P50/P90/P95/P99, mean,
min, max):

    validation_ms, rag_ms, translate_ms (tm_lookup + tokenize + generate +
    decode + tm_store), quality_ms, persist_ms, serialize_ms, total_ms

plus the model-internal breakdown that comes back in response.timings.model.

Also measures:
    * cold-start latency      (time to first model load in this process)
    * warm P50/P90/P95/P99    for every stage
    * TM-hit latency          (2nd request of an identical sentence)
    * language detection      (/detect endpoint — NOT part of the /text
                               translate path, which trusts client-supplied
                               source_lang; profiled separately for honesty)
    * model environment       (device, dtype, num_beams, max_length)

Usage (from backend/):
    python scripts/profile_latency.py [--n 100] [--lang hindi] [--out evaluation/reports/profile.json]
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
    if not values:
        return None
    return float(np.percentile(values, p))


def _r(value):
    """Round a float, returning None when the input is None (empty sample)."""
    return round(value, 1) if value is not None else None


def _summarize(values):
    return {
        'n': len(values),
        'mean_ms': round(statistics.mean(values), 1) if values else None,
        'p50_ms': _r(_pct(values, 50)),
        'p90_ms': _r(_pct(values, 90)),
        'p95_ms': _r(_pct(values, 95)),
        'p99_ms': _r(_pct(values, 99)),
        'max_ms': round(max(values), 1) if values else None,
        'min_ms': round(min(values), 1) if values else None,
    }


def load_sentences(lang, max_sentences):
    """Reuse the curated benchmark sources (English)."""
    from app.evaluation.datasets import get_benchmark_dataset
    ds = get_benchmark_dataset(target_langs=[lang])
    sources = ds['sources']
    if max_sentences and len(sources) > max_sentences:
        sources = sources[:max_sentences]
    # Pad to reach the requested count using short, varied fillers
    filler = [
        'The weather is pleasant today.',
        'Please send me the report.',
        'The hospital is near the station.',
        'I am going to college tomorrow.',
        'Could you help me with this document?',
        'The train leaves at six in the evening.',
        'She works at the university library.',
        'We need to open a bank account.',
    ]
    i = 0
    while len(sources) < max_sentences:
        sources = sources + [filler[i % len(filler)]]
        i += 1
    return sources[:max_sentences]


def _model_env(model):
    """Report how the loaded model is actually running (no inference)."""
    import torch as _torch
    hf = getattr(model, '_model', None)
    dtype = str(getattr(hf, 'dtype', 'unknown')) if hf is not None else 'unknown'
    device_idx = getattr(model, '_device', -1)
    max_len = getattr(model, 'max_length', None)
    return {
        'device': 'cuda' if device_idx >= 0 else 'cpu',
        'device_index': device_idx,
        'dtype': dtype,
        'torch_threads': _torch.get_num_threads(),
        'num_beams': getattr(model, '_num_beams', 4),
        'max_length': max_len,
        'max_new_tokens': 'dynamic',  # resolved per request from the input length
        'note': 'num_beams resolved as wrapper default (NLLB_NUM_BEAMS env or '
                'constructor arg) unless overridden per request; max_new_tokens is '
                'resolved per request by resolve_max_new_tokens() (NLLB_MAX_NEW_TOKENS* '
                'env vars, default 2x input tokens clamped to 16..256); max_length is '
                'the input truncation limit only.',
    }


def main():
    parser = argparse.ArgumentParser(description='Pipeline latency profiler')
    parser.add_argument('--n', type=int, default=100, help='Number of warm requests')
    parser.add_argument('--lang', default='hindi', help='Target language')
    parser.add_argument('--out', default='evaluation/reports/profile.json', help='Output JSON')
    parser.add_argument('--no-rag', action='store_true', help='include_context=false')
    parser.add_argument('--num-beams', type=int, default=2,
                        help='Beam size passed to model.generate() (default 2 — '
                             'wrapper default, latency/quality sweet spot)')
    parser.add_argument('--no-num-beams', action='store_true',
                        help='Do not send num_beams in the request body; use the '
                             'wrapper-configured default (NLLB_NUM_BEAMS env or '
                             'constructor arg)')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # App + DB (temp sqlite, real model)
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

    sentences = load_sentences(args.lang, args.n)

    def translate(text, **overrides):
        body = {
            'text': text,
            'source_lang': 'english',
            'target_lang': args.lang,
            'model': 'nllb-200-distilled-600M',
        }
        if not args.no_num_beams:
            body['num_beams'] = args.num_beams
        if args.no_rag:
            body['include_context'] = False
        body.update(overrides)
        return client.post('/api/v1/translate/text', json=body)

    # ------------------------------------------------------------------
    # Cold start: measure first model load
    # ------------------------------------------------------------------
    print('\n[1/4] Loading real model (cold start)...')
    t0 = time.time()
    from app.services.model_registry import ModelRegistry
    registry = ModelRegistry.get_instance()
    model = registry.get_model('nllb-200-distilled-600M')
    cold_ms = (time.time() - t0) * 1000
    if model is None or not model.available:
        print('FATAL: NLLB model did not load. Cannot profile.')
        sys.exit(1)
    env = _model_env(model)
    if args.no_num_beams:
        env['num_beams'] = getattr(model, '_num_beams', 4)
        env['note'] = (f'num_beams NOT sent in the request body; wrapper default applied '
                       f'(NLLB_NUM_BEAMS env or constructor arg, resolved to '
                       f'{env["num_beams"]}). max_new_tokens is resolved per request '
                       '(2x input tokens, 16..256).')
    else:
        env['num_beams'] = args.num_beams
        env['note'] = (f'num_beams={args.num_beams} passed explicitly by the profiler; '
                       'max_new_tokens is resolved per request (2x input tokens, 16..256).')
    print(f'  cold-start model load: {cold_ms:.1f} ms')
    print(f'  env: device={env["device"]} dtype={env["dtype"]} '
          f'num_beams={env["num_beams"]} max_length={env["max_length"]} '
          f'max_new_tokens={env["max_new_tokens"]} torch_threads={env["torch_threads"]}')

    # ------------------------------------------------------------------
    # Warmup (also primes RAG encoder + TM encoder)
    # ------------------------------------------------------------------
    print(f'[2/4] Warmup (2 requests)...')
    for s in sentences[:2]:
        r = translate(s)
        if r.status_code != 200:
            print(f'  warmup failed {r.status_code}: {r.get_json()}')
            sys.exit(1)
    print('  warmup OK')

    # ------------------------------------------------------------------
    # Warm benchmark
    # ------------------------------------------------------------------
    print(f'[3/4] Profiling {len(sentences)} warm requests...')
    stages = {
        'validation_ms': [], 'rag_ms': [], 'translate_ms': [],
        'quality_ms': [], 'persist_ms': [], 'serialize_ms': [], 'total_ms': [],
    }
    model_stages = {
        'tm_lookup_ms': [], 'tm_store_ms': [],
        'tokenize_ms': [], 'generate_ms': [], 'decode_ms': [],
    }
    all_latency = []
    errors = 0
    per_request = []
    for i, s in enumerate(sentences):
        r = translate(s)
        if r.status_code != 200:
            errors += 1
            continue
        body = r.get_json()
        t = body.get('timings', {})
        for k in stages:
            if k in t:
                stages[k].append(t[k])
        mt = t.get('model') or {}
        for k in model_stages:
            if k in mt:
                model_stages[k].append(mt[k])
        all_latency.append(t.get('total_ms'))
        per_request.append({
            'idx': i,
            'src_len_words': len(s.split()),
            'total_ms': t.get('total_ms'),
            'translate_ms': t.get('translate_ms'),
            'generate_ms': mt.get('generate_ms'),
            'rag_ms': t.get('rag_ms'),
            'tm_store_ms': mt.get('tm_store_ms'),
        })

    print(f'\n  requests: {len(all_latency)}  errors: {errors}')

    # ------------------------------------------------------------------
    # Language detection: profile /detect separately. NOTE: the /text
    # translate path does NOT run the language detector — source_lang is
    # client-supplied — so this stage is measured in isolation.
    # ------------------------------------------------------------------
    print('[4/4] Profiling /detect language detection...')
    # Prime the lazy LanguageDetector singleton (n-gram training) so the first
    # measured request is warm, mirroring the translate-path warmup.
    client.post('/api/v1/translate/detect', json={'text': 'warmup'})
    detect_ms = []
    for s in sentences:
        t0 = time.time()
        r = client.post('/api/v1/translate/detect', json={'text': s})
        if r.status_code == 200:
            detect_ms.append((time.time() - t0) * 1000)

    # ------------------------------------------------------------------
    # TM fast path: exact-hit then semantic-hit, measured WITHOUT RAG
    # (the /text route skips TM whenever RAG context is present)
    # ------------------------------------------------------------------
    tm_exact_ms = None
    tm_semantic_ms = None
    try:
        probe = 'The quarterly report was approved by the board yesterday.'
        translate(probe, include_context=False)  # first: model, stored in TM
        t0 = time.time()
        r = translate(probe, include_context=False)  # exact hit
        if r.status_code == 200:
            tm_exact_ms = round((time.time() - t0) * 1000, 1)
        t0 = time.time()
        r = translate('The quarterly report was approved by the board yesterday.',
                      include_context=False)  # same -> exact
        if r.status_code == 200:
            tm_semantic_ms = round((time.time() - t0) * 1000, 1)
    except Exception as e:
        print(f'  TM probe failed: {e}')

    print(f'  TM exact-hit latency (no RAG): {tm_exact_ms} ms')
    print(f'  TM repeat-hit latency (no RAG): {tm_semantic_ms} ms')

    print('\n=== PIPELINE BREAKDOWN (warm requests) ===')
    order = ['validation_ms', 'rag_ms', 'translate_ms', 'quality_ms',
             'persist_ms', 'serialize_ms', 'total_ms']
    print(f'  {"stage":<16}{"mean":>9}{"p50":>9}{"p90":>9}{"p95":>9}{"p99":>9}{"max":>9}{"min":>8}')
    for k in order:
        s = _summarize(stages[k])
        print(f'  {k:<16}{s["mean_ms"] or 0:>9.1f}{s["p50_ms"] or 0:>9.1f}'
              f'{s["p90_ms"] or 0:>9.1f}{s["p95_ms"] or 0:>9.1f}'
              f'{s["p99_ms"] or 0:>9.1f}{s["max_ms"] or 0:>9.1f}{s["min_ms"] or 0:>8.1f}')

    # Derived "postprocessing" stage = tokenizer decode + response serialization
    post_ms = [a + b for a, b in zip(model_stages['decode_ms'], stages['serialize_ms'])]

    print('\n=== MODEL-INTERNAL BREAKDOWN ===')
    print(f'  {"stage":<16}{"mean":>9}{"p50":>9}{"p90":>9}{"p95":>9}{"p99":>9}{"max":>9}{"min":>8}')
    for k in ['tm_lookup_ms', 'tokenize_ms', 'generate_ms', 'decode_ms', 'tm_store_ms']:
        s = _summarize(model_stages[k])
        print(f'  {k:<16}{s["mean_ms"] or 0:>9.1f}{s["p50_ms"] or 0:>9.1f}'
              f'{s["p90_ms"] or 0:>9.1f}{s["p95_ms"] or 0:>9.1f}'
              f'{s["p99_ms"] or 0:>9.1f}{s["max_ms"] or 0:>9.1f}{s["min_ms"] or 0:>8.1f}')
    ps = _summarize(post_ms)
    print(f'  {"postprocess_ms":<16}{ps["mean_ms"] or 0:>9.1f}'
          f'{ps["p50_ms"] or 0:>9.1f}{ps["p90_ms"] or 0:>9.1f}'
          f'{ps["p95_ms"] or 0:>9.1f}{ps["p99_ms"] or 0:>9.1f}'
          f'{ps["max_ms"] or 0:>9.1f}{ps["min_ms"] or 0:>8.1f}  (decode + serialize)')

    print('\n=== LANGUAGE DETECTION (/detect endpoint, standalone) ===')
    s = _summarize(detect_ms)
    print(f'  {"stage":<16}{"mean":>9}{"p50":>9}{"p90":>9}{"p95":>9}{"p99":>9}{"max":>9}{"min":>8}')
    print(f'  {"detect_ms":<16}{s["mean_ms"] or 0:>9.1f}{s["p50_ms"] or 0:>9.1f}'
          f'{s["p90_ms"] or 0:>9.1f}{s["p95_ms"] or 0:>9.1f}'
          f'{s["p99_ms"] or 0:>9.1f}{s["max_ms"] or 0:>9.1f}{s["min_ms"] or 0:>8.1f}')
    print('  (not part of the /text translate path — source_lang is client-supplied)')

    report = {
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'mode': 'live-profile',
        'lang': args.lang,
        'num_requests': len(all_latency),
        'errors': errors,
        'cold_start_model_load_ms': round(cold_ms, 1),
        'model_env': env,
        'tm_exact_hit_total_ms': tm_exact_ms,
        'tm_semantic_hit_total_ms': tm_semantic_ms,
        'per_request': per_request,
        'stages': {k: _summarize(v) for k, v in stages.items()},
        'model_stages': {k: _summarize(v) for k, v in model_stages.items()},
        'postprocess_ms': _summarize(post_ms),
        'language_detection_ms': _summarize(detect_ms),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f'\nReport saved: {args.out}')
    try:
        os.remove(db_path)
    except OSError:
        pass


if __name__ == '__main__':
    main()
