from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.extensions import db
from app.models import Translation

history_bp = Blueprint('history', __name__)


@history_bp.route('', methods=['GET'])
@jwt_required()
def get_history():
    """Get paginated translation history for the authenticated user."""
    user_id = get_jwt_identity()
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)

    if not page or page < 1:
        return jsonify({'error': 'page must be a positive integer'}), 400
    if not per_page or per_page < 1 or per_page > 100:
        return jsonify({'error': 'per_page must be between 1 and 100'}), 400

    query = Translation.query.filter_by(user_id=user_id).order_by(Translation.created_at.desc())
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    return jsonify({
        'translations': [t.to_dict() for t in pagination.items],
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page,
        'per_page': per_page,
    }), 200


@history_bp.route('/<translation_id>', methods=['GET'])
@jwt_required()
def get_translation(translation_id):
    """Get a single translation by ID."""
    user_id = get_jwt_identity()
    t = Translation.query.filter_by(translation_id=translation_id, user_id=user_id).first()
    if not t:
        return jsonify({'error': 'Translation not found'}), 404
    return jsonify(t.to_dict()), 200


@history_bp.route('/<translation_id>', methods=['DELETE'])
@jwt_required()
def delete_translation(translation_id):
    """Delete a translation from history."""
    user_id = get_jwt_identity()
    t = Translation.query.filter_by(translation_id=translation_id, user_id=user_id).first()
    if not t:
        return jsonify({'error': 'Translation not found'}), 404
    db.session.delete(t)
    db.session.commit()
    return jsonify({'message': 'Deleted successfully'}), 200
