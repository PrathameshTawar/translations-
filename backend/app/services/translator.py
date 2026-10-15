"""
Translation Engine — unified interface for all translation models.

Now uses ModelRegistry internally to support multiple backends
(NLLB-200, M2M100, IndicTrans2) while maintaining backward compatibility
with existing code that imports TranslationEngine.
"""

import logging
from typing import Dict, Optional

from .model_registry import ModelRegistry
from .translators import BaseTranslator, TranslationResult

logger = logging.getLogger(__name__)


class TranslationEngine:
    """
    Translation Engine with multi-model support.

    Wraps ModelRegistry for backward compatibility.
    New code should use ModelRegistry directly.
    """

    # Supported languages (union of all model language sets)
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

    def __init__(self):
        self._registry = ModelRegistry.get_instance()
        # Track the default model for backward compatibility
        self._default_model = self._registry.get_best_model('english', 'hindi')

    @property
    def translator(self):
        """Backward compatibility: return the default model or first available."""
        if self._default_model and self._default_model.available:
            return self._default_model
        # Try to find any available model
        for name in self._registry.model_names:
            model = self._registry.get_model(name)
            if model and model.available:
                return model
        return None

    @property
    def model(self):
        """Alias for translater."""
        return self.translator

    def translate(
        self,
        text: str,
        source_lang: str = "english",
        target_lang: str = "hindi",
        context: str = None,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """
        Translate text using the best available model or a specified one.
        Checks Translation Memory first (fast path).

        Args:
            text: Text to translate.
            source_lang: Source language.
            target_lang: Target language.
            context: Optional RAG context.
            model_name: Specific model to use. If None, auto-selects best.
            **kwargs: Extra generation knobs forwarded to the model wrapper
                      (e.g. num_beams).

        Returns:
            dict with keys: original, translated, source_lang, target_lang,
                           context_used, metadata, tm_hit, tm_similarity
        """
        try:
            result = self._registry.translate(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                model=model_name,
                context=context,
                **kwargs,
            )

            # Convert to legacy dict format for backward compatibility
            return {
                'original': result.original,
                'translated': result.translated,
                'source_lang': result.source_lang,
                'target_lang': result.target_lang,
                'context_used': result.context_used,
                'metadata': {
                    'model': result.model,
                    'src_code': result.src_code,
                    'tgt_code': result.tgt_code,
                    'processing_time_ms': result.processing_time_ms,
                    'tm_hit': result.tm_hit,
                    'tm_similarity': result.tm_similarity,
                    'tm_hit_count': result.tm_hit_count,
                    'timings': result.timings,
                    'error': result.error,
                },
            }
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            raise

    def get_available_models(self) -> Dict[str, bool]:
        """Get all available models and their status."""
        return self._registry.available_models

    def compare_models(
        self,
        text: str,
        source_lang: str = 'english',
        target_lang: str = 'hindi',
    ) -> Dict:
        """Translate with all available models for comparison."""
        return self._registry.compare_models(text, source_lang, target_lang)
