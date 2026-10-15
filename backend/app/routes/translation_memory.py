"""
Translation Memory API routes.

Provides endpoints for querying TM stats, clearing the cache,
and manually adding/removing entries.
"""

import logging
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required

logger = logging.getLogger(__name__)

tm_bp = Blueprint('translation_memory', __name__)

VALID_SORTS = ('recent', 'popular')


def _parse_limit() -> int:
    """Parse and clamp the ?limit= param (default 20, max 100)."""
    try:
        limit = int(request.args.get('limit', 20))
    except (TypeError, ValueError):
        limit = 20
    return max(1, min(limit, 100))


@tm_bp.route('/stats', methods=['GET'])
def get_tm_stats():
    """Get Translation Memory statistics."""
    from app.services.translation_memory import TranslationMemory
    tm = TranslationMemory.get_instance()
    return jsonify(tm.get_stats())


@tm_bp.route('/entries', methods=['GET'])
def get_tm_entries():
    """Get recent or most-accessed TM entries."""
    from app.services.translation_memory import TranslationMemory
    tm = TranslationMemory.get_instance()
    sort_by = request.args.get('sort', 'recent')  # 'recent' or 'popular'
    if sort_by not in VALID_SORTS:
        return jsonify({'error': f"sort must be one of: {', '.join(VALID_SORTS)}"}), 400

    limit = _parse_limit()
    if sort_by == 'popular':
        entries = tm.get_most_accessed(limit)
    else:
        entries = tm.get_recent(limit)
    return jsonify({'entries': entries, 'total': tm.total_entries, 'sort_by': sort_by})


@tm_bp.route('/clear', methods=['POST'])
@jwt_required()
def clear_tm():
    """Clear all Translation Memory entries (admin only)."""
    from app.services.translation_memory import TranslationMemory
    tm = TranslationMemory.get_instance()
    count = tm.clear()
    logger.info(f"TM cleared by user via API: {count} entries")
    return jsonify({'message': f'Cleared {count} entries', 'cleared': count})


@tm_bp.route('/save', methods=['POST'])
@jwt_required()
def save_tm():
    """Persist Translation Memory to disk (admin only)."""
    from app.services.translation_memory import TranslationMemory
    tm = TranslationMemory.get_instance()
    success = tm.save()
    if success:
        return jsonify({'message': 'TM saved to disk', 'entries': len(tm._entries)})
    return jsonify({'error': 'Failed to save TM'}), 500
