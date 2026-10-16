import { useState, useCallback } from 'react';
import axios from 'axios';

// Relative by default: the Vite dev server and nginx both proxy /api to the
// backend. Override with VITE_API_BASE_URL for other setups.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';

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
