from app.extensions import db
from datetime import datetime, timezone
from uuid import uuid4
from werkzeug.security import generate_password_hash, check_password_hash


def utcnow() -> datetime:
    """Naive UTC now (DB columns are naive to stay SQLite-compatible)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid4()))
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=utcnow)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)

    translations = db.relationship('Translation', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class Translation(db.Model):
    __tablename__ = 'translations'

    id = db.Column(db.Integer, primary_key=True)
    translation_id = db.Column(db.String(36), unique=True, default=lambda: str(uuid4()), index=True)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=True)

    source_text = db.Column(db.Text, nullable=False)
    target_text = db.Column(db.Text)

    source_lang = db.Column(db.String(50))
    target_lang = db.Column(db.String(50))

    quality_score = db.Column(db.Float, default=0.0)
    confidence = db.Column(db.String(20), default='medium')  # high / medium / low
    hallucination_detected = db.Column(db.Boolean, default=False)

    domain = db.Column(db.String(50), default='general')
    character_count = db.Column(db.Integer, default=0)
    processing_time_ms = db.Column(db.Integer, default=0)
    retrieved_terms_count = db.Column(db.Integer, default=0)
    context_used = db.Column(db.Boolean, default=False)

    created_at = db.Column(db.DateTime, default=utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)

    def to_dict(self):
        return {
            'translation_id': self.translation_id,
            'source_text': self.source_text,
            'target_text': self.target_text,
            'source_lang': self.source_lang,
            'target_lang': self.target_lang,
            'quality_metrics': {
                'overall_score': self.quality_score or 0.0,
                'confidence': self.confidence or 'medium',
                'hallucination_detected': self.hallucination_detected,
            },
            'domain': self.domain,
            'character_count': self.character_count,
            'processing_time_ms': self.processing_time_ms,
            'retrieved_terms_count': self.retrieved_terms_count,
            'context_used': self.context_used,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class Feedback(db.Model):
    __tablename__ = 'feedbacks'

    id = db.Column(db.Integer, primary_key=True)
    translation_id = db.Column(db.String(36), db.ForeignKey('translations.translation_id'))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=True)
    rating = db.Column(db.Integer)  # 1-5
    comment = db.Column(db.Text)
    issues = db.Column(db.String(255))  # comma-separated: terminology,grammar,tone,accuracy
    created_at = db.Column(db.DateTime, default=utcnow)
