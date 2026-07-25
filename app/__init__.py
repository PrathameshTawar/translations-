from flask import Flask
from flask_cors import CORS
from app.config import config_by_name
from app.extensions import db, cache, jwt

def create_app(config_name='development'):
    app = Flask(__name__)
    app.config.from_object(config_by_name[config_name])
    
    # Initialize extensions
    db.init_app(app)
    cache.init_app(app)
    jwt.init_app(app)
    
    # CORS setup
    CORS(app, resources={
        r"/api/*": {
            "origins": ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173", "https://yourdomain.com"],
            "methods": ["GET", "POST", "PUT", "DELETE"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True
        }
    })
    
    # Register blueprints (to be created)
    from app.routes.translate import translate_bp
    app.register_blueprint(translate_bp, url_prefix='/api/v1/translate')
    
    # Error handlers
    register_error_handlers(app)
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    return app

def register_error_handlers(app):
    @app.errorhandler(400)
    def bad_request(e):
        return {'error': 'Bad request'}, 400
    
    @app.errorhandler(401)
    def unauthorized(e):
        return {'error': 'Unauthorized'}, 401
    
    @app.errorhandler(429)
    def rate_limit_exceeded(e):
        return {'error': 'Rate limit exceeded'}, 429
    
    @app.errorhandler(500)
    def internal_error(e):
        return {'error': 'Internal server error'}, 500
