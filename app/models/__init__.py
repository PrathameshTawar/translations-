from app.extensions import db
from datetime import datetime
from uuid import uuid4

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid4()))
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    translations = db.relationship('Translation', backref='user', lazy=True)

class Translation(db.Model):
    __tablename__ = 'translations'
    id = db.Column(db.Integer, primary_key=True)
    translation_id = db.Column(db.String(36), unique=True, default=lambda: str(uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'))
    
    source_text = db.Column(db.Text)
    target_text = db.Column(db.Text)
    
    source_lang = db.Column(db.String(50))
    target_lang = db.Column(db.String(50))
    
    quality_score = db.Column(db.Float)
    hallucination_detected = db.Column(db.Boolean, default=False)
    
    domain = db.Column(db.String(50), default='general')
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_rating = db.Column(db.Integer)  # 1-5 stars

class Feedback(db.Model):
    __tablename__ = 'feedbacks'
    id = db.Column(db.Integer, primary_key=True)
    translation_id = db.Column(db.String(36), db.ForeignKey('translations.translation_id'))
    rating = db.Column(db.Integer)
    comment = db.Column(db.Text)
    issues = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
