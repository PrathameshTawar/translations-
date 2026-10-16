"""
IndicTrans2 translation model wrapper.

Uses AI4Bharat's IndicTrans2 models for high-quality translation
between English and 22 Indic languages.

Models:
  - ai4bharat/indictrans2-en-indic-1B (English -> Indic)
  - ai4bharat/indictrans2-indic-en-1B (Indic -> English)
"""

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


class IndicTrans2Translator(BaseTranslator):
    """
    IndicTrans2 — AI4Bharat's translation model fine-tuned for Indic languages.

    Uses separate models for en->indic and indic->en directions.
    Supports 22 scheduled Indian languages.
    """

    # Model names for each direction
    MODEL_EN_INDIC = "ai4bharat/indictrans2-en-indic-1B"
    MODEL_INDIC_EN = "ai4bharat/indictrans2-indic-en-1B"

    # IndicTrans2 uses FLORES-200 language codes (with Indic script variants)
    LANGUAGE_CODES = {
        'english': 'eng_Latn',
        'hindi': 'hin_Deva',
        'marathi': 'mar_Deva',
        'bengali': 'ben_Beng',
        'gujarati': 'guj_Gujr',
        'tamil': 'tam_Taml',
        'telugu': 'tel_Telu',
        'kannada': 'kan_Knda',
        # Additional Indic languages supported by this model
        'assamese': 'asm_Beng',
        'oriya': 'ory_Orya',
        'punjabi': 'pan_Guru',
        'malayalam': 'mal_Mlym',
        'sanskrit': 'san_Deva',
        'sindhi': 'snd_Arab',
        'urdu': 'urd_Arab',
        'nepali': 'npi_Deva',
        'maithili': 'mai_Deva',
        'dogri': 'doi_Deva',
        'konkani': 'gom_Deva',
        'bodo': 'brx_Deva',
        'kashmiri': 'kas_Arab',
        'manipuri': 'mni_Beng',
        'santali': 'sat_Olck',
    }

    def __init__(self, device: Optional[int] = None):
        self._en_indic_model = None
        self._indic_en_model = None
        self._tokenizer = None
        self._device = device
        self._loaded = False
        self._load_model()

    def _load_model(self):
        """Load IndicTrans2 models based on desired direction."""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            import torch  # lazy: the package stays importable without torch
            configure_torch_threads()
            device = self._device if self._device is not None else (
                0 if torch.cuda.is_available() else -1
            )
            self._device = device if device >= 0 else -1

            # Load shared tokenizer (same for both directions)
            logger.info(f"Loading IndicTrans2 tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_EN_INDIC,
                trust_remote_code=True,
            )

            # Load en->indic model
            logger.info(f"Loading {self.MODEL_EN_INDIC}...")
            self._en_indic_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.MODEL_EN_INDIC,
                trust_remote_code=True,
            )

            if self._device >= 0:
                self._en_indic_model = self._en_indic_model.to(f'cuda:{self._device}')
            else:
                self._en_indic_model = self._en_indic_model.to('cpu')

            self._en_indic_model.eval()

            # Load indic->en model
            logger.info(f"Loading {self.MODEL_INDIC_EN}...")
            self._indic_en_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.MODEL_INDIC_EN,
                trust_remote_code=True,
            )

            if self._device >= 0:
                self._indic_en_model = self._indic_en_model.to(f'cuda:{self._device}')
            else:
                self._indic_en_model = self._indic_en_model.to('cpu')

            self._indic_en_model.eval()

            self._loaded = True
            logger.info(f"IndicTrans2 models loaded (device={'cpu' if self._device < 0 else 'cuda'})")
        except Exception as e:
            logger.error(f"Failed to load IndicTrans2 models: {e}")
            self._loaded = False

    @property
    def model_name(self) -> str:
        """Return the appropriate model name based on direction (if known)."""
        return "indictrans2"

    @property
    def available(self) -> bool:
        return (self._loaded
                and self._tokenizer is not None
                and self._en_indic_model is not None
                and self._indic_en_model is not None)

    @property
    def supported_languages(self) -> Dict[str, str]:
        return dict(self.LANGUAGE_CODES)

    @property
    def max_length(self) -> int:
        return 256  # IndicTrans2 recommends 256 max tokens

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
                translated='[IndicTrans2 model not loaded]',
                source_lang=source_lang,
                target_lang=target_lang,
                model=self.model_name,
                error='Model not loaded',
            )

        src_code = self.LANGUAGE_CODES.get(source_lang.lower())
        tgt_code = self.LANGUAGE_CODES.get(target_lang.lower())

        if not src_code or not tgt_code:
            raise ValueError(f"Unsupported language pair: {source_lang} -> {target_lang}")

        # Choose model based on direction
        is_en_to_indic = source_lang.lower() == 'english' and target_lang.lower() != 'english'
        is_indic_to_en = target_lang.lower() == 'english' and source_lang.lower() != 'english'

        if not is_en_to_indic and not is_indic_to_en:
            # indic-to-indic not directly supported by these model variants
            # Fall back to pivot through English
            logger.warning("Indic->Indic not directly supported, pivoting through English")
            # Translate source -> English first
            result_indic_en = self._translate_direction(
                text, source_lang, 'english',
                self._indic_en_model, src_code, 'eng_Latn',
                context, **kwargs,
            )
            if result_indic_en.translated and result_indic_en.translated != text:
                # Then English -> target
                result_en_indic = self._translate_direction(
                    result_indic_en.translated, 'english', target_lang,
                    self._en_indic_model, 'eng_Latn', tgt_code,
                    context, **kwargs,
                )
                return result_en_indic
            return result_indic_en

        if is_en_to_indic:
            return self._translate_direction(
                text, source_lang, target_lang,
                self._en_indic_model, src_code, tgt_code,
                context, **kwargs,
            )
        else:  # indic_to_en
            return self._translate_direction(
                text, source_lang, target_lang,
                self._indic_en_model, src_code, tgt_code,
                context, **kwargs,
            )

    def _translate_direction(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        model,
        src_code: str,
        tgt_code: str,
        context: Optional[str] = None,
        **kwargs,
    ) -> TranslationResult:
        """Translate in a single direction using the provided model."""
        start = time.time()
        try:
            input_text = text
            if context:
                input_text = f"{context}\n\n{text}"

            # IndicTrans2 uses a specific prefix format
            # Format: <source_lang> <target_lang> <text>
            prefixed_text = f"<{src_code}> <{tgt_code}> {input_text}"

            inputs = self._tokenizer(
                prefixed_text,
                return_tensors='pt',
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )

            if self._device >= 0:
                inputs = {k: v.to(f'cuda:{self._device}') for k, v in inputs.items()}

            input_len = inputs['input_ids'].shape[1]
            max_new = resolve_max_new_tokens(
                input_len, 'INDICTRANS2', default_max=192, default_mult=1.5)
            import torch  # lazy: the package stays importable without torch
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
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
                model="indictrans2",
                context_used=bool(context),
                processing_time_ms=round(elapsed, 2),
                src_code=src_code,
                tgt_code=tgt_code,
            )
        except Exception as e:
            logger.error(f"IndicTrans2 translation error: {e}")
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
