"""
Generative QA routes — grounded question answering (Groq Llama-3.1-8B).

Enabled when GROQ_API_KEY is set; otherwise the endpoint returns a clear
503 with instructions. See app/services/generative_qa.py for the pipeline.
"""

import logging

from flask import Blueprint, jsonify, request

from app.services.generative_qa import GenerativeQA, MAX_QUESTION_CHARS, is_available

logger = logging.getLogger(__name__)

qa_bp = Blueprint('qa', __name__)

_qa_service = None

SUPPORTED_LANGS = {
    'english', 'hindi', 'marathi', 'bengali', 'gujarati',
    'tamil', 'telugu', 'kannada',
}


def get_qa_service():
    global _qa_service
    if _qa_service is None:
        _qa_service = GenerativeQA()
    return _qa_service


@qa_bp.route('/ask', methods=['POST'])
def ask():
    """Grounded QA over the terminology RAG context.

    Body: { question, lang?, top_k?, use_reranker? }
    Returns: { answer, confidence, low_confidence, language, sources,
               model, timings } or a 503 when GROQ_API_KEY is missing.
    """
    data = request.get_json() or {}
    if not isinstance(data, dict):
        return jsonify({'error': 'Request body must be a JSON object'}), 400

    question = data.get('question')
    if not isinstance(question, str) or not question.strip():
        return jsonify({'error': 'question is required'}), 400
    question = question.strip()
    if len(question) > MAX_QUESTION_CHARS:
        return jsonify({
            'error': f'question exceeds {MAX_QUESTION_CHARS} character limit',
        }), 400

    lang = data.get('lang')
    if lang is not None and (not isinstance(lang, str)
                             or lang.lower() not in SUPPORTED_LANGS):
        return jsonify({'error': f'Unsupported lang: {lang}'}), 400
    lang = lang.lower() if lang else None

    top_k = data.get('top_k')
    if top_k is not None and (not isinstance(top_k, int)
                              or isinstance(top_k, bool) or not 1 <= top_k <= 10):
        return jsonify({'error': 'top_k must be an integer between 1 and 10'}), 400

    use_reranker = data.get('use_reranker', False)
    if not isinstance(use_reranker, bool):
        return jsonify({'error': 'use_reranker must be a boolean'}), 400

    if not is_available():
        return jsonify({
            'error': 'Generative QA is not configured. Set the GROQ_API_KEY '
                     'environment variable to enable Llama-3.1-8B answering.',
        }), 503

    try:
        result = get_qa_service().ask(
            question=question,
            lang=lang,
            top_k=top_k if top_k is not None else 5,
            use_reranker=use_reranker,
        )
    except Exception as e:
        logger.error(f'Generative QA failed: {e}')
        return jsonify({
            'error': 'Generative QA failed. Please try again later.',
        }), 502

    return jsonify(result), 200


@qa_bp.route('/status', methods=['GET'])
def status():
    """Report whether generative QA is configured."""
    return jsonify({
        'enabled': is_available(),
        'model': get_qa_service().model,
    }), 200
