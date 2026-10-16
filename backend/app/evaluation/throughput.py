"""
Throughput & concurrency benchmark — latency percentiles + stress test.

Implements the plan's latency/throughput probes:

  * Sequential latency profile — mean/median/P95/P99 + min/max across length
    buckets (short / medium / long sentences) and overall requests/sec.
  * Concurrency stress test — ThreadPoolExecutor sweeps across concurrency
    levels (1, 2, 4, 8, 16…); per level reports wall time, req/s, mean and
    P95 latency, and error rate (how the service saturates).

The translate_fn signature matches the rest of the framework:
    translate_fn(text, source_lang, target_lang) -> dict

Two modes:
  * mock (default) — a deterministic 2ms stub translator. This measures the
    benchmark harness / framework overhead and validates the pipeline; the
    numbers are NOT model inference latency.
  * live — the real ModelRegistry is used (real inference latency; slow).

Usage (from backend/):
    python -m app.evaluation.throughput [--live] [--concurrency 1,2,4,8,16]
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

MOCK_SLEEP_MS = 2.0


def make_mock_translate_fn(sleep_ms=MOCK_SLEEP_MS):
    """Deterministic stub: sleeps `sleep_ms` then returns an echo-ish dict."""
    def fn(text, source_lang='english', target_lang='hindi'):
        time.sleep(sleep_ms / 1000.0)
        return {
            'original': text,
            'translated': f'[{source_lang}->{target_lang}] {text}',
            'source_lang': source_lang,
            'target_lang': target_lang,
            'metadata': {'model': 'mock-sleep', 'error': None},
        }
    return fn


def _percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    idx = int(n * p)
    idx = min(idx, n - 1)
    return sorted_vals[idx]


def _length_bucket(text):
    n = len(text)
    if n < 60:
        return 'short'
    if n < 200:
        return 'medium'
    return 'long'


class ThroughputBenchmark:
    """Sequential latency profile + concurrency stress test."""

    def __init__(self, translate_fn=None, source_lang='english',
                 target_lang='hindi'):
        """
        Args:
            translate_fn: (text, source_lang, target_lang) -> dict.
                          If None, a mock sleep stub is used.
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        if translate_fn is None:
            self.mock = True
            self.translate_fn = make_mock_translate_fn()
        else:
            self.mock = False
            self.translate_fn = translate_fn

    # -- sequential latency profile ----------------------------------------

    def latency_profile(self, sentences, rounds=1, warmup=0) -> dict:
        """
        Sequential per-sentence latency statistics.

        Args:
            sentences: source sentences to translate.
            rounds: repeat count (each round re-runs all sentences).
            warmup: warmup sentences (run once, excluded from stats).

        Returns:
            dict: aggregate percentiles + throughput + per-length buckets.
        """
        latencies = []
        by_length = {}

        for s in sentences[:warmup]:
            try:
                self.translate_fn(s, source_lang=self.source_lang,
                                  target_lang=self.target_lang)
            except Exception:
                pass

        for _ in range(rounds):
            for s in sentences:
                t0 = time.time()
                try:
                    self.translate_fn(s, source_lang=self.source_lang,
                                      target_lang=self.target_lang)
                    elapsed_ms = (time.time() - t0) * 1000
                except Exception as e:
                    logger.debug(f"latency profile call failed: {e}")
                    elapsed_ms = 0.0
                latencies.append(elapsed_ms)
                bucket = _length_bucket(s)
                by_length.setdefault(bucket, []).append(elapsed_ms)

        valid = [l for l in latencies if l > 0]
        sorted_lat = sorted(valid)
        total_time_sec = sum(latencies) / 1000.0

        result = {
            'num_calls': len(latencies),
            'mean_ms': round(sum(valid) / max(len(valid), 1), 2),
            'median_ms': round(_percentile(sorted_lat, 0.5), 2),
            'p95_ms': round(_percentile(sorted_lat, 0.95), 2),
            'p99_ms': round(_percentile(sorted_lat, 0.99), 2),
            'min_ms': round(sorted_lat[0], 2) if sorted_lat else 0.0,
            'max_ms': round(sorted_lat[-1], 2) if sorted_lat else 0.0,
            'throughput_sentences_per_sec': round(
                len(latencies) / max(total_time_sec, 1e-9), 2),
        }
        result['by_length'] = {
            bucket: {
                'n': len(vals),
                'mean_ms': round(sum(vals) / max(len(vals), 1), 2),
                'p95_ms': round(_percentile(sorted(vals), 0.95), 2),
            }
            for bucket, vals in sorted(by_length.items())
        }
        return result

    # -- concurrency stress test -------------------------------------------

    def concurrency_test(self, sentences, levels=(1, 2, 4, 8, 16),
                         iters_per_worker=8) -> dict:
        """
        Fire requests concurrently and measure throughput / latency / errors.

        For each concurrency level C, run C workers each issuing
        `iters_per_worker` requests over the shared sentence list. Reports
        wall time, req/s, mean & P95 latency and error rate at that level.

        Args:
            sentences: pool of source sentences (cycled across workers).
            levels: concurrency levels to sweep.
            iters_per_worker: requests per worker at each level.

        Returns:
            dict: {'levels': [...], 'peak_req_per_sec': float}
        """
        if not sentences:
            return {'levels': [], 'peak_req_per_sec': 0.0}

        levels_out = []
        peak = 0.0
        for c in levels:
            latencies = []
            errors = 0

            def worker(worker_id):
                base = (worker_id * 7) % len(sentences)
                for i in range(iters_per_worker):
                    s = sentences[(base + i) % len(sentences)]
                    t0 = time.time()
                    try:
                        self.translate_fn(
                            s, source_lang=self.source_lang,
                            target_lang=self.target_lang,
                        )
                        latencies.append((time.time() - t0) * 1000)
                    except Exception:
                        latencies.append(0.0)
                        return 'error'
                return 'ok'

            wall_start = time.time()
            with ThreadPoolExecutor(max_workers=c) as pool:
                futures = [pool.submit(worker, w) for w in range(c)]
                for fut in as_completed(futures):
                    try:
                        if fut.result() != 'ok':
                            errors += 1
                    except Exception:
                        errors += 1
            wall_time_sec = time.time() - wall_start

            total_reqs = c * iters_per_worker
            valid = [l for l in latencies if l > 0]
            sorted_lat = sorted(valid)
            req_per_sec = total_reqs / max(wall_time_sec, 1e-9)
            peak = max(peak, req_per_sec)

            levels_out.append({
                'concurrency': c,
                'iterations': iters_per_worker,
                'total_requests': total_reqs,
                'wall_time_sec': round(wall_time_sec, 3),
                'throughput_req_per_sec': round(req_per_sec, 2),
                'mean_ms': round(sum(valid) / max(len(valid), 1), 2),
                'p95_ms': round(_percentile(sorted_lat, 0.95), 2),
                'error_count': errors,
                'error_rate_pct': round(errors / max(total_reqs, 1) * 100, 2),
            })

        return {'levels': levels_out, 'peak_req_per_sec': round(peak, 2)}

    # -- combined -----------------------------------------------------------

    def run(self, sentences=None, rounds=1, warmup=0,
            levels=(1, 2, 4, 8, 16), iters_per_worker=8,
            track_memory=True) -> dict:
        """
        Run the full benchmark (latency profile + concurrency stress).

        Args:
            sentences: sources to use. Defaults to the curated English set.
            track_memory: measure peak RSS with psutil if available.

        Returns:
            dict with latency, concurrency, system_info, note.
        """
        if sentences is None:
            from app.evaluation.datasets import get_benchmark_dataset
            ds = get_benchmark_dataset(target_langs=['hindi'])
            sentences = ds['sources']
            # Take the first 21 to keep default runs snappy
            sentences = sentences[:21]

        latency = self.latency_profile(sentences, rounds=rounds, warmup=warmup)
        concurrency = self.concurrency_test(
            sentences, levels=levels, iters_per_worker=iters_per_worker,
        )

        from app.evaluation.metrics.latency import LatencyBenchmark
        system_info = LatencyBenchmark().get_system_info()

        peak_memory_mb = None
        if track_memory:
            try:
                import psutil
                import os
                peak_memory_mb = round(
                    psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)
            except Exception:
                pass

        return {
            'mode': 'mock' if self.mock else 'live',
            'latency': latency,
            'concurrency': concurrency,
            'peak_memory_mb': peak_memory_mb,
            'system_info': system_info,
            'note': (
                'Mock mode uses a deterministic 2ms stub — it measures '
                'framework/harness overhead, NOT model inference latency. '
                'Run with --live for real inference numbers.'
                if self.mock else
                'Live mode: real ModelRegistry inference. Numbers are '
                'single-machine and depend on the installed models.'
            ),
        }


def run_throughput_benchmark(translate_fn=None, sentences=None,
                             levels=(1, 2, 4, 8, 16)) -> dict:
    """Module-level convenience entry point."""
    return ThroughputBenchmark(translate_fn=translate_fn).run(
        sentences=sentences, levels=levels,
    )


def _print_result(result):
    print(f"\nThroughput benchmark ({result['mode']} mode)")
    print(f"Sequential latency profile:")
    lat = result['latency']
    print(f"  calls={lat['num_calls']}  mean={lat['mean_ms']}ms  "
          f"median={lat['median_ms']}ms  p95={lat['p95_ms']}ms  "
          f"p99={lat['p99_ms']}ms  min={lat['min_ms']}  max={lat['max_ms']}")
    print(f"  throughput={lat['throughput_sentences_per_sec']} sents/s")
    for bucket, m in lat.get('by_length', {}).items():
        print(f"    {bucket}: n={m['n']} mean={m['mean_ms']}ms "
              f"p95={m['p95_ms']}ms")
    print("Concurrency stress test:")
    print(f"  {'conc':>4} {'reqs':>6} {'wall(s)':>8} {'req/s':>8} "
          f"{'mean(ms)':>9} {'p95(ms)':>8} {'err%':>6}")
    for lv in result['concurrency']['levels']:
        print(f"  {lv['concurrency']:>4} {lv['total_requests']:>6} "
              f"{lv['wall_time_sec']:>8} {lv['throughput_req_per_sec']:>8} "
              f"{lv['mean_ms']:>9} {lv['p95_ms']:>8} {lv['error_rate_pct']:>6}")
    print(f"  peak throughput: {result['concurrency']['peak_req_per_sec']} req/s")
    if result.get('peak_memory_mb'):
        print(f"  peak RSS: {result['peak_memory_mb']} MB")
    print(f"Note: {result['note']}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Throughput & concurrency benchmark')
    parser.add_argument('--live', action='store_true', help='Use real models (slow)')
    parser.add_argument('--concurrency', default='1,2,4,8,16',
                        help='Comma-separated concurrency levels')
    parser.add_argument('--iters', type=int, default=8,
                        help='Requests per worker per level')
    args = parser.parse_args()

    levels = tuple(int(x) for x in args.concurrency.split(','))

    if args.live:
        from app.services.model_registry import ModelRegistry
        registry = ModelRegistry.get_instance()

        def live_fn(text, source_lang='english', target_lang='hindi'):
            return registry.translate(
                text=text, source_lang=source_lang, target_lang=target_lang,
                model='nllb-200-distilled-600M', use_tm=False,
            ).to_dict()

        result = ThroughputBenchmark(translate_fn=live_fn).run(
            levels=levels, iters_per_worker=args.iters,
        )
    else:
        result = run_throughput_benchmark(levels=levels)
    _print_result(result)
