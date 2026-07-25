from flask import Blueprint, request, jsonify
from app.services.rag_service import IndicRAGPipeline
from app.services.translator import TranslationEngine

translate_bp = Blueprint('translate', __name__)

pipeline = None
translator_engine = None

# We can lazy load the pipeline or initialize it when the blueprint is registered
def get_pipeline():
    global pipeline
    if not pipeline:
        print("Initializing IndicRAGPipeline...")
        pipeline = IndicRAGPipeline()
    return pipeline

def get_translator():
    global translator_engine
    if not translator_engine:
        print("Initializing TranslationEngine...")
        translator_engine = TranslationEngine()
    return translator_engine

@translate_bp.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query field'}), 400
            
        query = data.get('query')
        source_lang = data.get('source_lang', 'english')
        target_lang = data.get('target_lang', 'hindi')
        include_context = data.get('include_context', True)
        
        context_text = None
        retrieved_terms = []
        detected_lang = source_lang
        
        if include_context:
            p = get_pipeline()
            rag_result = p.retrieve_and_answer(query, lang=source_lang, top_k=5, rerank_k=3)
            # Adapt legacy RAG pipeline response to context
            context_text = "\n".join(rag_result.get("sources", []))
            retrieved_terms = [{"term": "RAG Source", "context": s} for s in rag_result.get("sources", [])]
            detected_lang = rag_result.get("detected_lang", source_lang)
            
        t = get_translator()
        translation_result = t.translate(
            text=query,
            source_lang=source_lang,
            target_lang=target_lang,
            context=context_text
        )
        
        confidence = 0.85
            
        return jsonify({
            'answer': translation_result['translated'],
            'original': translation_result['original'],
            'confidence': confidence,
            'sources': retrieved_terms,
            'detected_lang': detected_lang
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
