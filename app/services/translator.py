from typing import Dict, List
from transformers import pipeline, AutoTokenizer
import logging
import torch

logger = logging.getLogger(__name__)

class TranslationEngine:
    """NLLB-200 Translation Engine with RAG integration"""
    
    LANGUAGE_CODES = {
        'hindi': 'hin_Deva',
        'marathi': 'mar_Deva',
        'bengali': 'ben_Beng',
        'gujarati': 'guj_Gujr',
        'tamil': 'tam_Taml',
        'telugu': 'tel_Telu',
        'kannada': 'kan_Knda',
        'english': 'eng_Latn'
    }
    
    def __init__(self):
        """Initialize NLLB-200 model"""
        device = 0 if torch.cuda.is_available() else -1
        try:
            self.translator = pipeline(
                "translation",
                model="facebook/nllb-200-distilled-600M",
                device=device
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "facebook/nllb-200-distilled-600M"
            )
            logger.info("Translation model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load translation model: {e}")
            self.translator = None
    
    def translate(
        self,
        text: str,
        source_lang: str = "english",
        target_lang: str = "hindi",
        context: str = None
    ) -> Dict:
        """
        Translate text with optional context
        """
        try:
            src_code = self.LANGUAGE_CODES.get(source_lang.lower())
            tgt_code = self.LANGUAGE_CODES.get(target_lang.lower())
            
            if not src_code or not tgt_code:
                raise ValueError(f"Unsupported language pair: {source_lang} -> {target_lang}")
            
            input_text = text
            if context:
                input_text = f"{context}\n\n{text}"
            
            if not self.translator:
                return {
                    'original': text,
                    'translated': "Translation engine not initialized.",
                    'source_lang': source_lang,
                    'target_lang': target_lang,
                    'context_used': bool(context)
                }

            result = self.translator(
                input_text,
                src_lang=src_code,
                tgt_lang=tgt_code,
                max_length=512
            )
            
            translation = result[0]['translation_text']
            
            if context:
                translation = translation.split('\n\n')[-1]
            
            return {
                'original': text,
                'translated': translation,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'context_used': bool(context),
                'metadata': {
                    'model': 'nllb-200-distilled-600M',
                    'src_code': src_code,
                    'tgt_code': tgt_code
                }
            }
        
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            raise
