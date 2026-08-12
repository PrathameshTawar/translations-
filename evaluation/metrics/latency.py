"""
Latency and system performance benchmarks.

Measures:
    - Inference time per sentence (mean, median, P95, P99)
    - Throughput (sentences per second, tokens per second)
    - Peak memory usage during inference
    - Model loading time

Robustness:
    Durations are measured with ``time.monotonic()``, which does not advance
    while the machine is suspended (sleep/hibernate). Wall-clock timing
    (``time.time()``) would absorb an overnight suspend into a single sample
    and corrupt every percentile. As a second line of defence, samples that
    still exceed ``SUSPEND_GUARD_MS`` are treated as environmental artifacts
    (system suspend, clock jumps, debugger attach): they are excluded from the
    statistics and reported separately as ``suspended_*`` fields.
"""

import time
import logging
import os
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

#: Any single inference longer than this is a suspend/clock artifact, not a
#: real translation. NLLB-600M on CPU takes ~1-2 s/sentence; even a 30-word
#: sentence with beam 4 stays well under 30 s. 5 minutes is a generous bound.
SUSPEND_GUARD_MS = float(os.environ.get('LATENCY_SUSPEND_GUARD_MS', '300000'))


def _percentile(sorted_vals: List[float], p: float) -> float:
    """
    Linear-interpolation percentile of an already-sorted list (the default
    numpy/Excel method).

    The previous implementation used ``sorted_vals[int(n * p / 100)]``,
    which with small samples (n < 20) truncates straight to the maximum —
    i.e. "P95" of 9 samples was literally the slowest request, not the 95th
    percentile. With interpolation, P95 of N samples sits at rank
    ``0.95 * (N - 1)`` instead.
    """
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    rank = p / 100.0 * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


@dataclass
class LatencyResult:
    """Structured result from latency benchmarking."""
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    std_ms: float = 0.0
    throughput_sentences_per_sec: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    total_time_ms: float = 0.0
    num_sentences: int = 0
    peak_memory_mb: Optional[float] = None
    model_load_time_ms: Optional[float] = None
    warmup_time_ms: Optional[float] = None
    # Samples excluded because they exceeded the suspend guard.
    suspended_samples: int = 0
    suspended_total_ms: float = 0.0
    suspended_max_ms: float = 0.0


class LatencyBenchmark:
    """System performance benchmarking for translation models."""

    def __init__(self):
        self._psutil_available = False
        self._check_deps()

    def _check_deps(self):
        try:
            import psutil  # noqa: F401
            self._psutil_available = True
        except ImportError:
            logger.info("psutil not installed. Memory benchmarking will be unavailable.")

    @property
    def available(self) -> bool:
        # Latency benchmarking always works; memory tracking is optional.
        return True

    @property
    def memory_available(self) -> bool:
        return self._psutil_available

    def benchmark(
        self,
        translate_fn: Callable,
        sentences: List[str],
        source_lang: str = 'english',
        target_lang: str = 'hindi',
        warmup_runs: int = 2,
        track_memory: bool = True,
        tokenize_fn: Optional[Callable] = None,
    ) -> Dict:
        """
        Benchmark translation latency and throughput.

        Args:
            translate_fn: Function that takes (text, source_lang, target_lang) and returns dict.
            sentences: List of input sentences to translate.
            source_lang: Source language code.
            target_lang: Target language code.
            warmup_runs: Number of warmup sentences before benchmarking.
            track_memory: Whether to track peak memory usage.
            tokenize_fn: Optional function to count tokens in a sentence.

        Returns:
            dict with latency statistics.
        """
        if not sentences:
            return {'error': 'No sentences provided', 'latencies_ms': []}

        # Track total tokens
        if tokenize_fn is None:
            token_count = sum(len(s.split()) for s in sentences)
        else:
            token_count = sum(tokenize_fn(s) for s in sentences)

        # Track peak memory
        peak_memory = None
        if track_memory and self._psutil_available:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            peak_memory = process.memory_info().rss / (1024 * 1024)  # MB

        # Warmup runs (first inference often includes model loading overhead)
        warmup_times = []
        if warmup_runs > 0 and len(sentences) > warmup_runs:
            warmup_sents = sentences[:warmup_runs]
            for sent in warmup_sents:
                try:
                    t0 = time.monotonic()
                    translate_fn(sent, source_lang=source_lang, target_lang=target_lang)
                    elapsed_ms = (time.monotonic() - t0) * 1000
                except Exception as e:
                    logger.debug(f"Warmup translation failed: {e}")
                    continue
                # Same suspend guard as the benchmark samples: a sleep during
                # warmup must not inflate the reported model-load/warmup time.
                if elapsed_ms > SUSPEND_GUARD_MS:
                    logger.warning(
                        f'Warmup sample took {elapsed_ms:.0f} ms (> guard '
                        f'{SUSPEND_GUARD_MS:.0f} ms); treating as suspend artifact'
                    )
                    continue
                warmup_times.append(elapsed_ms)

        # Benchmark runs
        latencies_ms = []
        suspended_ms = []
        benchmark_sentences = sentences[warmup_runs:] if warmup_runs > 0 and len(sentences) > warmup_runs else sentences

        for i, sent in enumerate(benchmark_sentences):
            try:
                t0 = time.monotonic()
                translate_fn(sent, source_lang=source_lang, target_lang=target_lang)
                elapsed_ms = (time.monotonic() - t0) * 1000
            except Exception as e:
                logger.debug(f"Benchmark translation {i} failed: {e}")
                latencies_ms.append(0.0)  # mark as failed
                continue

            if elapsed_ms > SUSPEND_GUARD_MS:
                # Machine suspended or the clock jumped mid-inference. The
                # sample measures environmental downtime, not translation
                # speed — exclude it from every statistic so one sleep does
                # not fail the whole scorecard.
                logger.warning(
                    f'Latency sample {i} took {elapsed_ms:.0f} ms (> guard '
                    f'{SUSPEND_GUARD_MS:.0f} ms); treating as suspend artifact'
                )
                suspended_ms.append(elapsed_ms)
                continue

            latencies_ms.append(elapsed_ms)

        if suspended_ms:
            logger.warning(
                f'Excluded {len(suspended_ms)} suspend-artifact sample(s) '
                f'(total {sum(suspended_ms):.0f} ms) from latency statistics'
            )

        # Track peak memory after inference
        if track_memory and self._psutil_available:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            peak_memory = process.memory_info().rss / (1024 * 1024)  # MB

        if not latencies_ms:
            return {
                'error': 'No successful benchmark runs',
                'latencies_ms': [],
                'suspended_samples': len(suspended_ms),
                'suspended_total_ms': round(sum(suspended_ms), 2),
                'suspended_max_ms': round(max(suspended_ms), 2) if suspended_ms else 0.0,
            }

        # Compute statistics
        sorted_lat = sorted(latencies_ms)
        n = len(sorted_lat)
        mean = sum(sorted_lat) / n
        median = sorted_lat[n // 2] if n % 2 == 1 else (sorted_lat[n // 2 - 1] + sorted_lat[n // 2]) / 2
        p95 = _percentile(sorted_lat, 95)
        p99 = _percentile(sorted_lat, 99)

        # Filter out failed (0.0) for min/max
        valid_lat = [l for l in sorted_lat if l > 0]
        fmin = valid_lat[0] if valid_lat else 0.0
        fmax = valid_lat[-1] if valid_lat else 0.0

        # Throughput
        total_time_sec = sum(latencies_ms) / 1000
        throughput_sents = n / total_time_sec if total_time_sec > 0 else 0
        # Only count non-zero latencies for throughput
        successful_sentences = len(valid_lat)
        successful_time = sum(valid_lat) / 1000
        throughput_tokens = token_count / successful_time if successful_time > 0 else 0

        # Standard deviation
        variance = sum((l - mean) ** 2 for l in valid_lat) / max(len(valid_lat), 1)
        std = variance ** 0.5

        result = LatencyResult(
            mean_ms=round(mean, 2),
            median_ms=round(median, 2),
            p95_ms=round(p95, 2),
            p99_ms=round(p99, 2),
            min_ms=round(fmin, 2),
            max_ms=round(fmax, 2),
            std_ms=round(std, 2),
            throughput_sentences_per_sec=round(throughput_sents, 2),
            throughput_tokens_per_sec=round(throughput_tokens, 2),
            total_time_ms=round(sum(latencies_ms), 2),
            num_sentences=n,
            peak_memory_mb=round(peak_memory, 2) if peak_memory else None,
            model_load_time_ms=round(sum(warmup_times), 2) if warmup_times else None,
            warmup_time_ms=round(sum(warmup_times), 2) if warmup_times else None,
            suspended_samples=len(suspended_ms),
            suspended_total_ms=round(sum(suspended_ms), 2),
            suspended_max_ms=round(max(suspended_ms), 2) if suspended_ms else 0.0,
        )

        return asdict(result)

    def get_system_info(self) -> Dict:
        """Get system information for context."""
        info = {
            'platform': 'unknown',
            'cpu_count': 0,
            'ram_gb': 0,
            'gpu_available': False,
            'gpu_name': None,
        }

        import platform
        info['platform'] = platform.platform()

        try:
            import multiprocessing
            info['cpu_count'] = multiprocessing.cpu_count()
        except Exception:
            pass

        if self._psutil_available:
            import psutil
            info['ram_gb'] = round(psutil.virtual_memory().total / (1024 ** 3), 1)

        try:
            import torch
            info['gpu_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info['gpu_name'] = torch.cuda.get_device_name(0)
                info['gpu_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)
        except Exception:
            pass

        return info
