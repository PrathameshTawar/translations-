import React, { useState, useEffect } from 'react';
import { useTranslation } from '../hooks/useTranslation';
import LanguageSelector from './LanguageSelector';
import TextInput from './TextInput';
import OutputDisplay from './OutputDisplay';
import QualityBadge from './QualityBadge';
import TerminologyList from './TerminologyList';

export default function TranslatorCard() {
  const [sourceText, setSourceText] = useState('');
  const [sourceLang, setSourceLang] = useState('english');
  const [targetLang, setTargetLang] = useState('hindi');
  const [includeContext, setIncludeContext] = useState(true);
  
  const { 
    translate, 
    translationResult, 
    loading, 
    error 
  } = useTranslation();

  const handleTranslate = async () => {
    if (!sourceText.trim()) return;
    
    await translate({
      text: sourceText,
      source_lang: sourceLang,
      target_lang: targetLang,
      include_context: includeContext
    });
  };

  useEffect(() => {
    const debounceTimer = setTimeout(() => {
      if (sourceText) handleTranslate();
    }, 1000);
    
    return () => clearTimeout(debounceTimer);
  }, [sourceText, sourceLang, targetLang, includeContext]);

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 max-w-6xl mx-auto w-full">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Source Column */}
        <div className="space-y-4 flex flex-col">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold text-gray-800">Source</h3>
            <LanguageSelector
              value={sourceLang}
              onChange={setSourceLang}
            />
          </div>
          <TextInput
            value={sourceText}
            onChange={setSourceText}
            placeholder="Enter text to translate..."
            maxLength={5000}
          />
          <div className="flex items-center">
            <input
              type="checkbox"
              id="context"
              checked={includeContext}
              onChange={(e) => setIncludeContext(e.target.checked)}
              className="mr-2"
            />
            <label htmlFor="context" className="text-sm text-gray-600 cursor-pointer">
              Use context-aware translation (RAG)
            </label>
          </div>
        </div>

        {/* Target Column */}
        <div className="space-y-4 flex flex-col">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold text-gray-800">Translation</h3>
            <LanguageSelector
              value={targetLang}
              onChange={setTargetLang}
            />
          </div>
          
          {loading ? (
            <div className="bg-gray-100 rounded-md p-4 h-40 flex items-center justify-center border">
              <div className="animate-spin h-8 w-8 border-4 border-blue-500 border-t-transparent rounded-full"></div>
            </div>
          ) : error ? (
            <div className="bg-red-50 rounded-md p-4 h-40 text-red-700 border border-red-200">
              {error}
            </div>
          ) : translationResult ? (
            <div className="space-y-3">
              <OutputDisplay text={translationResult.translated_text} />
              
              <QualityBadge 
                score={translationResult.quality_metrics.overall_score}
                confidence={translationResult.quality_metrics.confidence}
              />
              
              {translationResult.context_used && (
                <TerminologyList 
                  terms={translationResult.retrieved_terms}
                />
              )}
            </div>
          ) : (
            <div className="bg-gray-100 rounded-md p-4 h-40 text-gray-400 flex items-center justify-center border italic">
              Translation will appear here
            </div>
          )}
        </div>
      </div>

      {/* Action Buttons */}
      <div className="mt-6 flex gap-4">
        <button
          onClick={handleTranslate}
          disabled={!sourceText || loading}
          className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded disabled:opacity-50 transition-colors"
        >
          {loading ? 'Translating...' : 'Translate'}
        </button>
        <button
          onClick={() => {
            setSourceText('');
            setTargetLang(sourceLang);
            setSourceLang(targetLang);
          }}
          className="bg-gray-200 hover:bg-gray-300 text-gray-800 font-semibold py-2 px-4 rounded transition-colors"
        >
          Swap
        </button>
        <button
          onClick={() => {
            navigator.clipboard.writeText(translationResult?.translated_text || '');
          }}
          disabled={!translationResult}
          className="bg-gray-200 hover:bg-gray-300 text-gray-800 font-semibold py-2 px-4 rounded disabled:opacity-50 transition-colors"
        >
          Copy
        </button>
      </div>
    </div>
  );
}
