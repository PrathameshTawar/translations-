import { useState, useCallback } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:8000/api/v1';

export function useTranslation() {
  const [translationResult, setTranslationResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const translate = useCallback(async (params) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post(`${API_BASE_URL}/translate/ask`, {
        query: params.text,
        source_lang: params.source_lang,
        target_lang: params.target_lang,
        include_context: params.include_context
      });
      
      setTranslationResult({
        translated_text: response.data.answer,
        quality_metrics: {
          overall_score: response.data.confidence,
          confidence: response.data.confidence > 0.8 ? 'high' : 'medium'
        },
        context_used: response.data.sources && response.data.sources.length > 0,
        retrieved_terms: response.data.sources
      });
    } catch (err) {
      setError(err.message || 'Failed to translate');
    } finally {
      setLoading(false);
    }
  }, []);

  return { translate, translationResult, loading, error };
}
