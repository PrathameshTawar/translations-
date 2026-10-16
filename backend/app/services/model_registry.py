"""
Model Registry — manages multiple translation model wrappers.

Provides a unified interface for:
  - Registering and discovering available models (lazy-loading)
  - Translating with a specific model (or auto-selecting the best)
  - Translation Memory integration (cache before translate, store after)
  - Listing supported languages per model
  - Comparing models side by side
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

from .translators import BaseTranslator, TranslationResult
from .translators import NLLBTranslator, M2M100Translator, IndicTrans2Translator

logger = logging.getLogger(__name__)

# Default model priority order (best quality first)
DEFAULT_MODEL_PRIORITY = [
    'indictrans2',
    'nllb-200-distilled-600M',
    'm2m100_418M',
]


class ModelRegistry:
    """
    Singleton registry for all available translation models.
    Models are loaded lazily — only when first requested.
    """

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._models: Dict[str, BaseTranslator] = {}
        self._loading_attempted: set = set()
        self._model_classes = {
            'nllb-200-distilled-600M': NLLBTranslator,
            'm2m100_418M': M2M100Translator,
            'indictrans2': IndicTrans2Translator,
        }
        logger.info(f"ModelRegistry initialized with {len(self._model_classes)} registered models (lazy-load)")

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _try_load(self, name: str) -> Optional[BaseTranslator]:
        """Attempt to load a specific model (caches result)."""
        if name in self._models:
            return self._models[name]
        if name in self._loading_attempted:
            return None  # Already tried and failed

        self._loading_attempted.add(name)
        cls = self._model_classes.get(name)
        if cls is None:
            logger.warning(f"Unknown model: {name}")
            return None

        try:
            logger.info(f"Loading model: {name}...")
            model = cls()
            if model.available:
                self._models[name] = model
                logger.info(f"  [OK] {name} loaded successfully")
            else:
                logger.warning(f"  [FAIL] {name} loaded but reports unavailable")
            return model
        except Exception as e:
            logger.warning(f"  [FAIL] {name} could not be loaded: {e}")
            return None

    def _load_all_available(self):
        """Try to load all registered models (called on first discovery request)."""
        for name in self._model_classes:
            if name not in self._models and name not in self._loading_attempted:
                self._try_load(name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available_models(self) -> Dict[str, bool]:
        """Get all registered models with their availability status.
        Triggers lazy-loading of all models."""
        self._load_all_available()
        return {name: name in self._models for name in self._model_classes}

    @property
    def model_names(self) -> List[str]:
        """List all registered model names (without loading them)."""
        return list(self._model_classes.keys())

    def get_model(self, name: str) -> Optional[BaseTranslator]:
        """Get a specific model by name (lazy-loads if needed)."""
        return self._try_load(name)

    def get_best_model(self, source_lang: str, target_lang: str) -> Optional[BaseTranslator]:
        """
        Auto-select the best available model for a language pair.
        Tries models in priority order and lazy-loads as needed.
        """
        is_indic = (
            source_lang.lower() != 'english'
            or target_lang.lower() != 'english'
        )

        for name in DEFAULT_MODEL_PRIORITY:
            model = self._try_load(name)
            if model and model.available:
                codes = model.supported_languages
                if source_lang.lower() in codes and target_lang.lower() in codes:
                    return model

        # Fallback: try loading any model
        for name in self._model_classes:
            model = self._try_load(name)
            if model and model.available:
                return model

        return None

    def translate(
        self,
        text: str,
        source_lang: str = 'english',
        target_lang: str = 'hindi',
        model: Optional[str] = None,
        context: Optional[str] = None,
        use_tm: bool = True,
        **kwargs,
    ) -> TranslationResult:
        """
        Translate text using the specified model (or auto-select best).
        Checks Translation Memory first for cached results.

        Args:
            text: Text to translate.
            source_lang: Source language name.
            target_lang: Target language name.
            model: Specific model name. If None, auto-selects best.
            context: Optional RAG context for terminology guidance.
            use_tm: Whether to check Translation Memory first (default: True).

        Returns:
            TranslationResult with translated text and metadata.
        """
        tm_lookup_ms = 0.0
        # Check Translation Memory first (fast path)
        if use_tm and not context:  # Skip TM when context is provided (RAG may change meaning)
            try:
                from .translation_memory import TranslationMemory
                t_tm = time.time()
                tm = TranslationMemory.get_instance()
                tm_entry = tm.lookup(text, source_lang, target_lang)
                tm_lookup_ms = (time.time() - t_tm) * 1000
                if tm_entry is not None:
                    logger.debug(f"TM hit for: {text[:50]}...")
                    result = TranslationResult(
                        original=text,
                        translated=tm_entry.translated_text,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        model=tm_entry.model,
                        processing_time_ms=1.0,  # TM lookup is nearly instant
                        src_code=None,
                        tgt_code=None,
                        timings={'tm_lookup_ms': round(tm_lookup_ms, 2)},
                    )
                    result.tm_hit = True
                    result.tm_similarity = tm_entry.similarity_score
                    result.tm_hit_count = tm_entry.hit_count
                    return result
            except Exception as e:
                tm_lookup_ms = 0.0
                logger.debug(f"TM lookup failed (non-fatal): {e}")

        # No TM hit — translate via model
        start = time.time()

        if model:
            translator = self.get_model(model)
            if not translator:
                raise ValueError(
                    f"Model '{model}' not found or failed to load. "
                    f"Available: {self.model_names}"
                )
        else:
            translator = self.get_best_model(source_lang, target_lang)
            if not translator:
                raise RuntimeError("No translation models are available")

        result = translator.translate(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            context=context,
            **kwargs,
        )

        elapsed = (time.time() - start) * 1000
        result.processing_time_ms = round(elapsed, 2)

        # Store in Translation Memory for future lookups
        if use_tm and result.translated and not result.error:
            try:
                from .translation_memory import TranslationMemory
                t_tm_store = time.time()
                tm = TranslationMemory.get_instance()
                tm.store(
                    text=text,
                    translated_text=result.translated,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    model=result.model,
                )
                result.timings['tm_store_ms'] = round((time.time() - t_tm_store) * 1000, 2)
            except Exception as e:
                result.timings['tm_store_ms'] = 0.0
                logger.debug(f"TM store failed (non-fatal): {e}")
        else:
            result.timings.setdefault('tm_store_ms', 0.0)
        result.timings.setdefault('tm_lookup_ms', round(tm_lookup_ms, 2))

        result.tm_hit = False
        return result

    def get_supported_languages(self) -> Dict[str, Dict[str, str]]:
        """Get supported languages from all loaded models, merged."""
        merged = {}
        for name, model in self._models.items():
            if model.available:
                for lang_name, code in model.supported_languages.items():
                    if lang_name not in merged:
                        merged[lang_name] = {}
                    merged[lang_name][name] = code

        # If no models loaded yet, try loading NLLB (lightest) first
        if not self._models:
            nllb = self._try_load('nllb-200-distilled-600M')
            if nllb and nllb.available:
                for lang_name, code in nllb.supported_languages.items():
                    if lang_name not in merged:
                        merged[lang_name] = {}
                    merged[lang_name]['nllb-200-distilled-600M'] = code

        return merged

    def compare_models(
        self,
        text: str,
        source_lang: str = 'english',
        target_lang: str = 'hindi',
    ) -> Dict:
        """
        Translate the same text with all available models for comparison.
        Only loads models that aren't already loaded.
        """
        results = {}
        for name in self._model_classes:
            model = self._try_load(name)
            if model and model.available:
                try:
                    result = model.translate(
                        text=text,
                        source_lang=source_lang,
                        target_lang=target_lang,
                    )
                    results[name] = result.to_dict()
                except Exception as e:
                    results[name] = {'error': str(e)}
        return results

    def __repr__(self) -> str:
        loaded = list(self._models.keys())
        return f"ModelRegistry(loaded={loaded}, registered={list(self._model_classes.keys())})"
