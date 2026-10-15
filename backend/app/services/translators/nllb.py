"""
NLLB-200 translation model wrapper.

Uses facebook/nllb-200-distilled-600M via AutoModelForSeq2SeqLM.
Supports 200+ languages including 7 Indic languages.
"""

import os
import re
import time
import logging
from typing import Dict, Optional


from .base import (
    BaseTranslator,
    TranslationResult,
    configure_torch_threads,
    resolve_max_new_tokens,
)

logger = logging.getLogger(__name__)

# Echoed glossary pair: '<words> = <words>' (e.g. 'बैंक = बैंक'). The right
# side is captured non-greedily so consecutive echoed pairs are handled.
_ECHO_PAIR_RE = re.compile(r'^(.+?)\s*=\s*(.+?)(?=\s|$)', re.S)


def _strip_context_echo(translation: str, context_lines) -> str:
    """
    Remove glossary pairs the model echoed at the start of the output.

    When a terminology context ("term = translation") is prepended to the
    input, NLLB tends to translate those lines too and emit them at the start
    of the output as '<X> = <Y>' pairs joined by spaces (e.g. 'बैंक = बैंक
    शिक्षा = शिक्षा <real translation>'). The old split('\n\n') strip relied
    on the model preserving the separator, which it does not.

    Each echoed pair's right side is the glossary translation of some term, so
    a pair is only stripped when its right side matches one of the known
    glossary translations (exactly, or as a prefix/suffix — the model may
    inflect). A truncated multi-word match is completed before continuing.
    """
    glossary = []
    for line in context_lines:
        if '=' in line:
            trans = line.split('=', 1)[1].strip()
            if trans:
                glossary.append(trans)
    if not glossary:
        return translation.strip()

    def matches_glossary(text):
        return any(text == g or g.startswith(text) or text.startswith(g)
                   for g in glossary)

    rest = translation.strip()
    for _ in range(len(glossary) * 2 + 2):
        m = _ECHO_PAIR_RE.match(rest)
        if not m:
            break
        right = m.group(2).strip()
        if not right or not matches_glossary(right):
            break  # not an echoed pair -> real translation begins
        rest = rest[m.end():].strip()
        # If the pair's right side was a partial multi-word translation
        # (e.g. 'मशीन' of 'मशीन लर्निंग'), finish consuming its remainder.
        for g in glossary:
            if g.startswith(right) and len(g) > len(right):
                suffix = g[len(right):].strip()
                if suffix and rest.startswith(suffix):
                    rest = rest[len(suffix):].strip()
                    break
    return rest

# Default beam size used by generate() when neither the request, the
# constructor, nor the NLLB_NUM_BEAMS env var specifies one.
#
# beam=2 is the latency/quality sweet spot on CPU: live profiling shows
# P95 ~2.1s (vs ~3.9s at beam=4) with equal-or-better BLEU, because beam 4
# tends to over-generate past the natural EOS (hurting precision and
# latency). Quality-sensitive workloads can raise it via NLLB_NUM_BEAMS.
DEFAULT_NUM_BEAMS = 2


class NLLBTranslator(BaseTranslator):
    """
    NLLB-200 (No Language Left Behind) translation model.

    Model: facebook/nllb-200-distilled-600M (600M parameters)
    Languages: 200+ including Hindi, Bengali, Marathi, Gujarati,
               Tamil, Telugu, Kannada
    """

    MODEL_NAME = "facebook/nllb-200-distilled-600M"

    # FLORES-200 language codes
    LANGUAGE_CODES = {
        'english': 'eng_Latn',
        'hindi': 'hin_Deva',
        'marathi': 'mar_Deva',
        'bengali': 'ben_Beng',
        'gujarati': 'guj_Gujr',
        'tamil': 'tam_Taml',
        'telugu': 'tel_Telu',
        'kannada': 'kan_Knda',
    }

    def __init__(self, device: Optional[int] = None, num_beams: Optional[int] = None):
        self._model = None
        self._tokenizer = None
        self._device = device
        self._num_beams = self._resolve_num_beams(num_beams)
        self._loaded = False
        self._load_model()

    @staticmethod
    def _resolve_num_beams(num_beams: Optional[int]) -> int:
        """
        Resolve the default beam size for generate().

        Precedence: explicit constructor arg > NLLB_NUM_BEAMS env var > 2.
        Invalid values (non-int, < 1) warn and fall back to the default
        rather than crashing model startup.
        """
        candidates = [num_beams]
        if num_beams is None:
            env = os.getenv('NLLB_NUM_BEAMS')
            if env is not None:
                try:
                    candidates.append(int(env))
                except ValueError:
                    candidates.append(None)
                    logger.warning(f'NLLB_NUM_BEAMS={env!r} is not an integer; '
                                   f'using default {DEFAULT_NUM_BEAMS}')
        for value in candidates:
            if value is not None and value >= 1:
                return value
        if num_beams is not None:
            logger.warning(f'num_beams={num_beams!r} must be >= 1; '
                           f'using default {DEFAULT_NUM_BEAMS}')
        return DEFAULT_NUM_BEAMS

    @property
    def num_beams(self) -> int:
        """Configured beam size used when a request does not override it."""
        return self._num_beams

    def _load_model(self):
        """Load model and tokenizer from HuggingFace cache."""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            import torch  # lazy: the package stays importable without torch
            configure_torch_threads()
            device = self._device if self._device is not None else (
                0 if torch.cuda.is_available() else -1
            )

            logger.info(f"Loading {self.MODEL_NAME}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_NAME)

            if device >= 0:
                self._model = self._model.to(f'cuda:{device}')
                self._device = device
            else:
                self._model = self._model.to('cpu')
                self._device = -1

            self._model.eval()
            self._loaded = True
            logger.info(f"{self.MODEL_NAME} loaded (device={'cpu' if device < 0 else 'cuda'})")
        except Exception as e:
            logger.error(f"Failed to load {self.MODEL_NAME}: {e}")
            self._loaded = False

    @property
    def model_name(self) -> str:
        return self.MODEL_NAME

    @property
    def available(self) -> bool:
        return self._loaded and self._model is not None and self._tokenizer is not None

    @property
    def supported_languages(self) -> Dict[str, str]:
        return dict(self.LANGUAGE_CODES)

    def translate(
        self,
        text: str,
        source_lang: str = 'english',
        target_lang: str = 'hindi',
        context: Optional[str] = None,
        **kwargs,
    ) -> TranslationResult:
        if not self.available:
            return TranslationResult(
                original=text,
                translated='[NLLB model not loaded]',
                source_lang=source_lang,
                target_lang=target_lang,
                model=self.model_name,
                error='Model not loaded',
            )

        src_code = self.LANGUAGE_CODES.get(source_lang.lower())
        tgt_code = self.LANGUAGE_CODES.get(target_lang.lower())

        if not src_code or not tgt_code:
            raise ValueError(f"Unsupported language pair: {source_lang} -> {target_lang}")

        start = time.time()
        try:
            input_text = text
            if context:
                input_text = f"{context}\n\n{text}"

            # --- Stage: tokenization ------------------------------------
            t_tok = time.time()
            self._tokenizer.src_lang = src_code
            inputs = self._tokenizer(
                input_text, return_tensors='pt', truncation=True, max_length=self.max_length
            )
            tokenize_ms = (time.time() - t_tok) * 1000

            if self._device >= 0:
                inputs = {k: v.to(f'cuda:{self._device}') for k, v in inputs.items()}

            forced_bos_token_id = self._tokenizer.convert_tokens_to_ids(tgt_code)

            # --- Stage: generation (the dominant cost on CPU) ----------
            # Beam size: per-request kwarg > wrapper-configured default.
            beams = kwargs.get('num_beams')
            num_beams = beams if beams is not None else self._num_beams
            # Bound the output length: without a cap, generation may run all
            # the way to max_length=512 when the model delays its EOS, which
            # dominates CPU latency. The cap is dynamic (2x the input tokens,
            # configurable via NLLB_MAX_NEW_TOKENS* env vars) so normal
            # translations are unaffected while pathological runs are bounded.
            input_len = inputs['input_ids'].shape[1]
            max_new = resolve_max_new_tokens(input_len, 'NLLB')
            t_gen = time.time()
            import torch  # lazy: the package stays importable without torch
            with torch.inference_mode():
                generated = self._model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_new_tokens=max_new,
                    num_beams=num_beams,
                    early_stopping=True,
                )
            generate_ms = (time.time() - t_gen) * 1000

            # --- Stage: decoding ---------------------------------------
            t_dec = time.time()
            translation = self._tokenizer.decode(generated[0], skip_special_tokens=True)
            decode_ms = (time.time() - t_dec) * 1000

            if context:
                # NLLB echoes the glossary lines ('bank = बैंक' -> 'बैंक = बैंक')
                # at the start of the output instead of preserving the '\n\n'
                # separator, so strip those echoed 'X = Y' pairs.
                translation = _strip_context_echo(
                    translation, context.split('\n'))

            elapsed = (time.time() - start) * 1000

            return TranslationResult(
                original=text,
                translated=translation.strip(),
                source_lang=source_lang,
                target_lang=target_lang,
                model=self.model_name,
                context_used=bool(context),
                processing_time_ms=round(elapsed, 2),
                src_code=src_code,
                tgt_code=tgt_code,
                timings={
                    'tokenize_ms': round(tokenize_ms, 2),
                    'generate_ms': round(generate_ms, 2),
                    'decode_ms': round(decode_ms, 2),
                    'input_tokens': input_len,
                    'max_new_tokens': max_new,
                },
            )
        except Exception as e:
            logger.error(f"NLLB translation error: {e}")
            elapsed = (time.time() - start) * 1000
            return TranslationResult(
                original=text,
                translated='',
                source_lang=source_lang,
                target_lang=target_lang,
                model=self.model_name,
                error=str(e),
                processing_time_ms=round(elapsed, 2),
            )

    @property
    def max_length(self) -> int:
        return 512
