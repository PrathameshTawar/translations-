import time
import logging
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity, verify_jwt_in_request
from app.extensions import db
from app.models import Translation
from app.services.translator import TranslationEngine
from app.services.quality import QualityScorer

logger = logging.getLogger(__name__)

from app.extensions import limiter
from app.utils.auth import config_limit

translate_bp = Blueprint('translate', __name__)

_translator_engine = None
_quality_scorer = None


def get_translator():
    global _translator_engine
    if _translator_engine is None:
        _translator_engine = TranslationEngine()
    return _translator_engine


_language_detector = None


def get_quality_scorer():
    global _quality_scorer
    if _quality_scorer is None:
        _quality_scorer = QualityScorer()
    return _quality_scorer


def get_language_detector():
    """Lazy singleton for the script-based language detector."""
    global _language_detector
    if _language_detector is None:
        from app.services.language_detection import LanguageDetector
        _language_detector = LanguageDetector()
    return _language_detector


SUPPORTED_LANGUAGES = [
    {'code': 'english',  'name': 'English',    'native': 'English'},
    {'code': 'hindi',    'name': 'Hindi',       'native': 'हिंदी'},
    {'code': 'marathi',  'name': 'Marathi',     'native': 'मराठी'},
    {'code': 'bengali',  'name': 'Bengali',     'native': 'বাংলা'},
    {'code': 'gujarati', 'name': 'Gujarati',    'native': 'ગુજરાતી'},
    {'code': 'tamil',    'name': 'Tamil',       'native': 'தமிழ்'},
    {'code': 'telugu',   'name': 'Telugu',      'native': 'తెలుగు'},
    {'code': 'kannada',  'name': 'Kannada',     'native': 'ಕನ್ನಡ'},
]

VALID_LANG_CODES = {l['code'] for l in SUPPORTED_LANGUAGES}


def _get_optional_user_id():
    """Return user_id if a valid JWT is present, else None (no error)."""
    try:
        verify_jwt_in_request(optional=True)
        return get_jwt_identity()
    except Exception:
        return None


@translate_bp.route('/languages', methods=['GET'])
def get_languages():
    """Return list of supported languages."""
    return jsonify({'languages': SUPPORTED_LANGUAGES}), 200


@translate_bp.route('/detect', methods=['POST'])
@limiter.limit(config_limit('RATELIMIT_TRANSLATE'))
def detect_language():
    """
    Detect the language of a text using Unicode script analysis.

    Body: { text }
    Returns: { detected_language, script, confidence, scores, note }
    """
    data = request.get_json() or {}
    # Top-level arrays/scalars must 400, not crash on .get() -> 500
    if not isinstance(data, dict):
        return jsonify({'error': 'Request body must be a JSON object'}), 400
    text = data.get('text')
    # Non-string payloads (numbers, lists, dicts) must 400, not 500.
    if not isinstance(text, str) or not text.strip():
        return jsonify({'error': 'text is required'}), 400
    text = text.strip()
    if len(text) > 5000:
        return jsonify({'error': 'text exceeds 5000 character limit'}), 400
    return jsonify(get_language_detector().detect_with_scores(text)), 200


@translate_bp.route('/models', methods=['GET'])
def get_models():
    """List available translation models."""
    from app.services.model_registry import ModelRegistry
    registry = ModelRegistry.get_instance()
    return jsonify({
        'models': registry.available_models,
        'default': 'auto',
    })


@translate_bp.route('/text', methods=['POST'])
@limiter.limit(config_limit('RATELIMIT_TRANSLATE'))
def translate_text():
    """
    Main translation endpoint.
    Body: { text, source_lang, target_lang, include_context?, model?,
            use_reranker? }
           model: 'auto' | 'nllb-200-distilled-600M' | 'm2m100_418M' | 'indictrans2'
    Returns: { translation_id, translated_text, quality_metrics, retrieved_terms,
               context_used, reranked, reranker_available, model_used }
    """
    data = request.get_json() or {}
    return _handle_translation(data)


@translate_bp.route('/ask', methods=['POST'])
@limiter.limit(config_limit('RATELIMIT_TRANSLATE'))
def ask():
    """Legacy endpoint — delegates to /text for backward compatibility."""
    data = request.get_json() or {}
    # Remap legacy field names: /text expects 'text'
    if data.get('query') and not data.get('text'):
        data = dict(data)
        data['text'] = data.pop('query')
    return _handle_translation(data)


def _handle_translation(data):
    """Shared implementation for /text and /ask."""
    start_ms = time.time() * 1000
    t_parse = time.time()
    if not data:
        return jsonify({'error': 'No JSON body provided'}), 400
    # Top-level arrays/scalars must 400, not crash on .get() -> 500
    if not isinstance(data, dict):
        return jsonify({'error': 'Request body must be a JSON object'}), 400

    text = data.get('text') or data.get('query') or ''
    # Non-string lang fields (numbers, lists, dicts) must 400, not 500:
    # (data.get('source_lang') or 'english').strip() would AttributeError.
    _src = data.get('source_lang')
    _tgt = data.get('target_lang')
    if _src is not None and not isinstance(_src, str):
        return jsonify({'error': 'source_lang must be a string'}), 400
    if _tgt is not None and not isinstance(_tgt, str):
        return jsonify({'error': 'target_lang must be a string'}), 400
    source_lang = (_src or 'english').strip().lower()
    target_lang = (_tgt or 'hindi').strip().lower()
    include_context = data.get('include_context', True)
    model_name = data.get('model')  # None means auto-select
    # Optional generation knob — forwarded to the model wrapper's generate().
    # Only forwarded when the client explicitly sets it, so wrappers without
    # **kwargs support (e.g. test stubs) keep working with the model default.
    num_beams = data.get('num_beams')
    if (num_beams is not None and (not isinstance(num_beams, int)
            or isinstance(num_beams, bool) or not 1 <= num_beams <= 10)):
        return jsonify({'error': 'num_beams must be an integer between 1 and 10'}), 400
    # Optional RAG knob: number of terminology terms to retrieve. Only used
    # when include_context is true; default stays 2.
    top_k = data.get('top_k')
    if (top_k is not None and (not isinstance(top_k, int)
            or isinstance(top_k, bool) or not 1 <= top_k <= 10)):
        return jsonify({'error': 'top_k must be an integer between 1 and 10'}), 400
    # Optional RAG knob: rescore retrieved terms with the cross-encoder
    # reranker. Only meaningful when include_context is true; the pipeline
    # falls back to dense order when the model is unavailable.
    use_reranker = data.get('use_reranker', False)
    if not isinstance(use_reranker, bool):
        return jsonify({'error': 'use_reranker must be a boolean'}), 400

    # Non-string payloads (numbers, lists, dicts) must 400, not 500.
    if not isinstance(text, str) or not text.strip():
        return jsonify({'error': 'text is required'}), 400
    text = text.strip()
    if len(text) > 5000:
        return jsonify({'error': 'text exceeds 5000 character limit'}), 400
    if source_lang not in VALID_LANG_CODES:
        return jsonify({'error': f'Unsupported source_lang: {source_lang}'}), 400
    if target_lang not in VALID_LANG_CODES:
        return jsonify({'error': f'Unsupported target_lang: {target_lang}'}), 400
    if source_lang == target_lang:
        return jsonify({'error': 'source_lang and target_lang must be different'}), 400

    parse_ms = (time.time() - t_parse) * 1000

    # Lazy-load RAG pipeline (optional)
    context_text = None
    retrieved_terms = []
    reranked = False
    reranker_available = False
    t_rag = time.time()
    if include_context:
        try:
            from app.services.rag_service import IndicRAGPipeline
            rag = IndicRAGPipeline.get_instance()
            # Terminology glossaries are selected by the *target* language
            # (each {target}_terms.json maps English terms -> target words)
            rag_result = rag.retrieve_context(
                text, source_lang=source_lang, target_lang=target_lang,
                top_k=top_k if top_k is not None else 2,
                use_reranker=use_reranker,
            )
            context_text = rag_result.get('context')
            retrieved_terms = rag_result.get('terms', [])
            reranked = bool(rag_result.get('reranked'))
            reranker_available = bool(rag_result.get('reranker_available'))
        except Exception as rag_err:
            # RAG is optional — continue without it, but log the failure
            logger.warning(f"RAG retrieval failed (continuing without context): {rag_err}")
    rag_ms = (time.time() - t_rag) * 1000

    # Translate using selected model
    translator = get_translator()
    t_translate = time.time()
    try:
        gen_kwargs = {'num_beams': num_beams} if num_beams is not None else {}
        result = translator.translate(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            context=context_text,
            model_name=model_name,
            **gen_kwargs,
        )
    except Exception as e:
        logger.exception(f"Translation failed for lang pair {source_lang}->{target_lang}")
        return jsonify({'error': 'Translation failed. Please try again later.'}), 500
    translate_ms = (time.time() - t_translate) * 1000

    translated_text = result.get('translated', '')
    model_used = result.get('metadata', {}).get('model', 'unknown')

    # A wrapper returns an error result (model not loaded / inference failure)
    # instead of raising. Surface it to the client instead of returning a 200.
    if not translated_text.strip():
        error_msg = result.get('metadata', {}).get('error') or 'Model produced an empty translation'
        logger.error(f"Translation produced no output ({model_used}): {error_msg}")
        return jsonify({'error': f'Translation failed: {error_msg}'}), 502

    processing_time = int(time.time() * 1000 - start_ms)

    # Quality scoring
    scorer = get_quality_scorer()
    t_quality = time.time()
    quality = scorer.score(
        source_text=text,
        translated_text=translated_text,
        retrieved_terms=retrieved_terms,
        target_lang=target_lang,
    )
    quality_ms = (time.time() - t_quality) * 1000

    # Optionally persist to DB (user may be anonymous)
    t_persist = time.time()
    user_id = _get_optional_user_id()
    t_record = Translation(
        user_id=user_id,
        source_text=text,
        target_text=translated_text,
        source_lang=source_lang,
        target_lang=target_lang,
        quality_score=quality['overall_score'],
        confidence=quality['confidence'],
        hallucination_detected=quality['hallucination_detected'],
        domain='general',
        character_count=len(text),
        processing_time_ms=processing_time,
        retrieved_terms_count=len(retrieved_terms),
        context_used=bool(context_text),
    )
    db.session.add(t_record)
    db.session.commit()
    persist_ms = (time.time() - t_persist) * 1000

    # Check if result came from Translation Memory
    tm_hit = result.get('metadata', {}).get('tm_hit', False)
    tm_similarity = result.get('metadata', {}).get('tm_similarity', 0.0)

    t_serialize = time.time()
    response = {
        'translation_id': t_record.translation_id,
        'translated_text': translated_text,
        'original_text': text,
        'source_lang': source_lang,
        'target_lang': target_lang,
        'quality_metrics': quality,
        'retrieved_terms': retrieved_terms,
        'context_used': bool(context_text),
        'reranked': reranked,
        'reranker_available': reranker_available,
        'model_used': model_used,
        'processing_time_ms': processing_time,
        'tm_hit': tm_hit,
        'tm_similarity': tm_similarity,
        'timings': {
            'validation_ms': round(parse_ms, 2),
            'rag_ms': round(rag_ms, 2),
            'translate_ms': round(translate_ms, 2),
            'quality_ms': round(quality_ms, 2),
            'persist_ms': round(persist_ms, 2),
            'total_ms': round(time.time() * 1000 - start_ms, 2),
            'model': (result.get('metadata') or {}).get('timings', {}),
        },
    }
    serialize_ms = (time.time() - t_serialize) * 1000
    response['timings']['serialize_ms'] = round(serialize_ms, 2)
    return jsonify(response), 200


@translate_bp.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback for a translation."""
    data = request.get_json() or {}
    if not isinstance(data, dict):
        return jsonify({'error': 'Request body must be a JSON object'}), 400
    translation_id = data.get('translation_id')
    rating = data.get('rating')

    if not translation_id or not rating:
        return jsonify({'error': 'translation_id and rating are required'}), 400
    if not isinstance(rating, int) or not (1 <= rating <= 5):
        return jsonify({'error': 'rating must be integer 1-5'}), 400

    t = Translation.query.filter_by(translation_id=translation_id).first()
    if not t:
        return jsonify({'error': 'Translation not found'}), 404

    from app.models import Feedback
    issues = data.get('issues')
    if issues is not None and not isinstance(issues, list):
        return jsonify({'error': 'issues must be a list of strings'}), 400
    issues_str = ','.join(issues)[:255] if issues else None

    fb = Feedback(
        translation_id=translation_id,
        rating=rating,
        comment=data.get('comment'),
        issues=issues_str,
    )
    db.session.add(fb)
    db.session.commit()
    return jsonify({'message': 'Feedback submitted'}), 201
