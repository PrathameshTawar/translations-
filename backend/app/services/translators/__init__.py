"""
Translation model wrappers.

Each wrapper implements a common BaseTranslator interface
so they can be used interchangeably through the ModelRegistry.
"""

from .base import BaseTranslator, TranslationResult
from .nllb import NLLBTranslator
from .m2m100 import M2M100Translator
from .indictrans2 import IndicTrans2Translator

__all__ = [
    'BaseTranslator',
    'TranslationResult',
    'NLLBTranslator',
    'M2M100Translator',
    'IndicTrans2Translator',
]
