"""
M2M100 translation model wrapper.

Uses facebook/m2m100_418M via M2M100ForConditionalGeneration.
Supports direct translation between 100 languages without
using English as a pivot.
"""

import time
import logging
from typing import Dict, Optional

import torch

from .base import (
    BaseTranslator,
    TranslationResult,
    configure_torch_threads,
    resolve_max_new_tokens,
)

logger = logging.getLogger(__name__)


class M2M100Translator(BaseTranslator):
    """
    M2M100 (Many-to-Many) translation model from Meta.

    Model: facebook/m2m100_418M (418M parameters for faster inference)
    Alternative: facebook/m2m100_1.2B (1.2B parameters, better quality)
    Languages: 100 languages including 6 Indic languages (Hindi, Marathi,
    Bengali, Gujarati, Tamil, Kannada). Telugu is NOT part of M2M100's
    100-language set, so it is deliberately absent from LANGUAGE_CODES —
    asking for it raises an explicit "Unsupported language pair" error
    instead of silently producing an empty translation.
    """

    MODEL_NAME = "facebook/m2m100_418M"

    # M2M100 language codes (ISO 639-1 based). NOTE: telugu is intentionally
    # missing — 'te' is not in the m2m100 tokenizer's 100-language vocabulary
    # (verified: lang_code_to_id has no 'te'), so requesting telugu raises a
    # clear ValueError rather than a silent KeyError->empty output.
    LANGUAGE_CODES = {
        'english': 'en',
        'hindi': 'hi',
        'marathi': 'mr',
        'bengali': 'bn',
        'gujarati': 'gu',
        'tamil': 'ta',
        'kannada': 'kn',
    }

    def __init__(self, device: Optional[int] = None, model_name: Optional[str] = None):
        self._model_name = model_name or self.MODEL_NAME
        self._model = None
        self._tokenizer = None
        self._device = device
        self._loaded = False
        self._load_model()

    def _load_model(self):
        """Load M2M100 model and tokenizer."""
        try:
            from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

            configure_torch_threads()
            device = self._device if self._device is not None else (
                0 if torch.cuda.is_available() else -1
            )

            logger.info(f"Loading {self._model_name}...")
            self._tokenizer = M2M100Tokenizer.from_pretrained(self._model_name)
            self._model = M2M100ForConditionalGeneration.from_pretrained(self._model_name)

            if device >= 0:
                self._model = self._model.to(f'cuda:{device}')
                self._device = device
            else:
                self._model = self._model.to('cpu')
                self._device = -1

            self._model.eval()
            self._loaded = True
            logger.info(f"{self._model_name} loaded (device={'cpu' if device < 0 else 'cuda'})")
        except Exception as e:
            logger.error(f"Failed to load {self._model_name}: {e}")
            self._loaded = False

    @property
    def model_name(self) -> str:
        return self._model_name

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
                translated='[M2M100 model not loaded]',
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

            # Set source language on tokenizer
            self._tokenizer.src_lang = src_code
            inputs = self._tokenizer(
                input_text, return_tensors='pt', truncation=True, max_length=self.max_length
            )

            if self._device >= 0:
                inputs = {k: v.to(f'cuda:{self._device}') for k, v in inputs.items()}

            # Force target language via language ID token
            forced_bos_token_id = self._tokenizer.get_lang_id(tgt_code)

            input_len = inputs['input_ids'].shape[1]
            max_new = resolve_max_new_tokens(input_len, 'M2M100', default_max=192)
            with torch.no_grad():
                generated = self._model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_new_tokens=max_new,
                    num_beams=kwargs.get('num_beams', 2),
                    early_stopping=True,
                )

            translation = self._tokenizer.decode(generated[0], skip_special_tokens=True)

            if context:
                translation = translation.split('\n\n')[-1]

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
            )
        except Exception as e:
            logger.error(f"M2M100 translation error: {e}")
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
