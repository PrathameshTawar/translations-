from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from sqlalchemy.exc import IntegrityError
from app.extensions import db
from app.models import User
from app.utils.validators import validate_register_request, validate_login_request

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new user."""
    data = request.get_json() or {}
    if not isinstance(data, dict):
        return jsonify({'error': 'Request body must be a JSON object'}), 400

    valid, error = validate_register_request(data)
    if not valid:
        return jsonify({'error': error}), 400

    name = data.get('name').strip()
    email = data.get('email').strip().lower()
    password = data.get('password')

    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 409

    user = User(name=name, email=email)
    user.set_password(password)
    db.session.add(user)
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        return jsonify({'error': 'Email already registered'}), 409

    token = create_access_token(identity=user.id)
    return jsonify({
        'message': 'Registration successful',
        'user': user.to_dict(),
        'access_token': token,
    }), 201


@auth_bp.route('/login', methods=['POST'])
def login():
    """Login with email + password, returns JWT."""
    data = request.get_json() or {}
    if not isinstance(data, dict):
        return jsonify({'error': 'Request body must be a JSON object'}), 400

    valid, error = validate_login_request(data)
    if not valid:
        return jsonify({'error': error}), 400

    email = data.get('email').strip().lower()
    password = data.get('password')

    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return jsonify({'error': 'Invalid email or password'}), 401
    if not user.is_active:
        return jsonify({'error': 'Account is deactivated'}), 403

    token = create_access_token(identity=user.id)
    return jsonify({
        'message': 'Login successful',
        'user': user.to_dict(),
        'access_token': token,
    }), 200


@auth_bp.route('/me', methods=['GET'])
@jwt_required()
def me():
    """Get current authenticated user info."""
    user_id = get_jwt_identity()
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    return jsonify({'user': user.to_dict()}), 200
