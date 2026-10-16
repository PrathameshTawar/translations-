"""
Base translator interface.

All translation model wrappers must implement this ABC
so they can be used interchangeably through ModelRegistry.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def resolve_max_new_tokens(
    input_len: int,
    env_prefix: str = 'NLLB',
    default_max: int = 256,
    default_min: int = 16,
    default_mult: float = 2.0,
) -> int:
    """
    Resolve how many tokens a seq2seq model may generate for an input.

    Generation on CPU is the dominant latency cost for NLLB/M2M100/
    IndicTrans2: it scales with the number of *output* tokens, so an
    unbounded ``max_length`` (e.g. 512) lets pathological runs without an
    early EOS blow past every latency target. The dynamic cap fixes that
    without touching translation quality for typical sentences, whose
    translations rarely exceed ``default_mult * input_len`` tokens.

    Precedence:
      1. ``{env_prefix}_MAX_NEW_TOKENS``  — fixed cap (>= 1), disables auto
      2. auto:  clamp(round(input_len * mult), min, max) with:
           ``{env_prefix}_MAX_NEW_TOKENS_MULT``  (default_mult)
           ``{env_prefix}_MAX_NEW_TOKENS_MIN``   (default_min)
           ``{env_prefix}_MAX_NEW_TOKENS_MAX``   (default_max)

    Args:
        input_len: number of source tokens (batch of one).
        env_prefix: env var namespace, e.g. 'NLLB', 'M2M100'.
        default_max: absolute ceiling for the auto mode.
        default_min: floor so tiny inputs are never starved.
        default_mult: output/input length ratio used by the auto mode.

    Returns:
        int: max_new_tokens to pass to ``model.generate()``.
    """
    fixed = os.getenv(f'{env_prefix}_MAX_NEW_TOKENS')
    if fixed:
        try:
            v = int(fixed)
            if v >= 1:
                return v
            logger.warning(f'{env_prefix}_MAX_NEW_TOKENS={fixed!r} must be >= 1; '
                           f'falling back to auto mode')
        except ValueError:
            logger.warning(f'{env_prefix}_MAX_NEW_TOKENS={fixed!r} is not an integer; '
                           f'falling back to auto mode')
    try:
        mult = float(os.getenv(f'{env_prefix}_MAX_NEW_TOKENS_MULT', str(default_mult)))
        lo = int(os.getenv(f'{env_prefix}_MAX_NEW_TOKENS_MIN', str(default_min)))
        hi = int(os.getenv(f'{env_prefix}_MAX_NEW_TOKENS_MAX', str(default_max)))
    except ValueError:
        logger.warning(f'Invalid {env_prefix}_MAX_NEW_TOKENS_* env value; '
                       f'using defaults mult={default_mult} min={default_min} max={default_max}')
        mult, lo, hi = default_mult, default_min, default_max
    return int(min(hi, max(lo, round(input_len * mult))))


def configure_torch_threads(env_var: str = 'TORCH_NUM_THREADS') -> None:
    """
    Apply torch intra-op thread count from an env var (no-op when unset).

    Lets operators tune CPU inference parallelism per host without touching
    code: e.g. ``TORCH_NUM_THREADS=8``. When unset, torch keeps its default
    (all physical cores).
    """
    raw = os.getenv(env_var)
    if not raw:
        return
    try:
        import torch
        n = int(raw)
        if n >= 1:
            torch.set_num_threads(n)
            logger.info(f'torch intra-op threads set to {n} ({env_var})')
    except (ValueError, ImportError) as e:
        logger.warning(f'Could not apply {env_var}={raw!r}: {e}')


@dataclass
class TranslationResult:
    """Standardised translation output across all models."""
    original: str
    translated: str
    source_lang: str
    target_lang: str
    model: str = 'unknown'
    context_used: bool = False
    processing_time_ms: float = 0.0
    src_code: Optional[str] = None
    tgt_code: Optional[str] = None
    error: Optional[str] = None
    # Translation Memory fields
    tm_hit: bool = False
    tm_similarity: float = 0.0
    tm_hit_count: int = 0
    # Per-stage latency breakdown (ms), filled by wrappers/registry.
    # Keys: tm_lookup_ms, tm_store_ms, tokenize_ms, generate_ms, decode_ms
    timings: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'original': self.original,
            'translated': self.translated,
            'source_lang': self.source_lang,
            'target_lang': self.target_lang,
            'context_used': self.context_used,
            'processing_time_ms': self.processing_time_ms,
            'tm_hit': self.tm_hit,
            'tm_similarity': self.tm_similarity,
            'tm_hit_count': self.tm_hit_count,
            'timings': self.timings,
            # Failures must stay visible in reports: without this, a wrapper
            # that catches an internal error and returns translated='' would
            # look identical to a genuine empty translation. Consumers (e.g.
            # BenchmarkRunner) check this field to record error entries.
            'error': self.error,
            'metadata': {
                'model': self.model,
                'src_code': self.src_code,
                'tgt_code': self.tgt_code,
            },
        }


class BaseTranslator(ABC):
    """Abstract base class for all translation model wrappers."""

    @abstractmethod
    def translate(
        self,
        text: str,
        source_lang: str = 'english',
        target_lang: str = 'hindi',
        context: Optional[str] = None,
        **kwargs,
    ) -> TranslationResult:
        """
        Translate text from source_lang to target_lang.

        Args:
            text: The text to translate.
            source_lang: Source language name (e.g., 'english', 'hindi').
            target_lang: Target language name.
            context: Optional context from RAG retrieval.

        Returns:
            TranslationResult with the translated text and metadata.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable name for this model (e.g., 'nllb-200-distilled-600M')."""
        ...

    @property
    @abstractmethod
    def available(self) -> bool:
        """Whether this model loaded successfully and is ready for inference."""
        ...

    @property
    @abstractmethod
    def supported_languages(self) -> Dict[str, str]:
        """
        Mapping of language names to model-specific codes.
        e.g. {'hindi': 'hin_Deva', 'english': 'eng_Latn'}
        """
        ...

    @property
    def max_length(self) -> int:
        """Maximum input length in tokens."""
        return 512

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, available={self.available})"
