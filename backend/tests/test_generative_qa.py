"""Tests for the generative QA service and /api/v1/qa/ask route.

Pure-logic + mocked-upstream tests — no real Groq call, no encoder/LLM
downloads. The RAG pipeline and the Groq HTTP call are both replaced with
deterministic fakes.
"""

import pytest


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv('DATABASE_URL', 'sqlite://')
    monkeypatch.setenv('SECRET_KEY', 'test-secret')
    monkeypatch.setenv('JWT_SECRET_KEY', 'test-jwt')
    from app import create_app
    app = create_app('development')
    app.config['TESTING'] = True
    return app.test_client()


class _FakeRAG:
    """Deterministic retrieve_context: canned terminology context."""

    def __init__(self, context='bank = बैंक education = शिक्षा loans offers'):
        self.context = context

    def retrieve_context(self, query, source_lang='english', target_lang=None,
                         top_k=5, use_reranker=False):
        return {
            'context': self.context,
            'terms': [
                {'term': 'bank', 'translation': 'बैंक', 'context': ''},
                {'term': 'education', 'translation': 'शिक्षा', 'context': ''},
            ],
            'reranked': False,
            'reranker_available': False,
        }


class TestConsistencyScore:
    def test_grounded_answer_scores_high(self):
        from app.services.generative_qa import consistency_score
        score = consistency_score(
            'The bank offers education loans.',
            'bank = बैंक education = शिक्षा loans',
        )
        assert 0.5 <= score <= 1.0

    def test_invented_answer_scores_low(self):
        from app.services.generative_qa import consistency_score
        score = consistency_score(
            'The stock market crashed in 2008.',
            'bank = बैंक education = शिक्षा',
        )
        assert 0.0 <= score < 0.5

    def test_empty_answer_scores_zero(self):
        from app.services.generative_qa import consistency_score
        assert consistency_score('', 'any context') == 0.0
        assert consistency_score('   ', 'any context') == 0.0

    def test_repetitive_answer_penalized(self):
        from app.services.generative_qa import consistency_score
        # Content words ARE in the context, but the stutter is a
        # hallucination signature and must drag the score down.
        score = consistency_score(
            'bank bank bank bank bank bank bank bank bank bank',
            'bank = बैंक',
        )
        assert score < 0.5


class TestQaRoute:
    def test_ask_unconfigured_returns_503(self, client, monkeypatch):
        monkeypatch.delenv('GROQ_API_KEY', raising=False)
        r = client.post('/api/v1/qa/ask', json={'question': 'Hello'})
        assert r.status_code == 503
        assert 'GROQ_API_KEY' in r.get_json()['error']

    def test_ask_missing_question_400(self, client, monkeypatch):
        monkeypatch.setenv('GROQ_API_KEY', 'test-key')
        assert client.post('/api/v1/qa/ask', json={}).status_code == 400
        assert client.post('/api/v1/qa/ask',
                           json={'question': '   '}).status_code == 400
        assert client.post('/api/v1/qa/ask',
                           json={'question': 42}).status_code == 400

    def test_ask_bad_lang_and_topk_400(self, client, monkeypatch):
        monkeypatch.setenv('GROQ_API_KEY', 'test-key')
        assert client.post('/api/v1/qa/ask',
                           json={'question': 'x', 'lang': 'klingon'}).status_code == 400
        assert client.post('/api/v1/qa/ask',
                           json={'question': 'x', 'top_k': 99}).status_code == 400

    def test_ask_success_with_mocks(self, client, monkeypatch):
        monkeypatch.setenv('GROQ_API_KEY', 'test-key')
        import app.services.generative_qa as gqa
        monkeypatch.setattr(
            gqa, '_call_groq',
            lambda prompt, api_key, model=None, timeout=30:
                'The bank offers education loans.')
        from app.services import rag_service
        monkeypatch.setattr(
            rag_service.IndicRAGPipeline, 'get_instance',
            staticmethod(lambda: _FakeRAG()))

        r = client.post('/api/v1/qa/ask',
                        json={'question': 'What does the bank offer?',
                              'lang': 'hindi', 'top_k': 3})
        assert r.status_code == 200
        body = r.get_json()
        assert body['answer'] == 'The bank offers education loans.'
        assert body['model'] == gqa.DEFAULT_MODEL
        assert body['language'] == 'hindi'
        assert body['context_used'] is True
        assert body['confidence'] >= 0.5
        assert body['low_confidence'] is False
        assert len(body['sources']) == 2
        assert 'generate_ms' in body['timings']

    def test_ask_auto_detect_language(self, client, monkeypatch):
        monkeypatch.setenv('GROQ_API_KEY', 'test-key')
        import app.services.generative_qa as gqa
        monkeypatch.setattr(
            gqa, '_call_groq',
            lambda prompt, api_key, model=None, timeout=30:
                'The context does not contain this information.')

        class _FakeDetector:
            def detect_with_scores(self, text):
                return {'detected_language': 'hindi', 'confidence': 0.99}

        from app.services import rag_service, language_detection
        monkeypatch.setattr(language_detection, 'LanguageDetector', _FakeDetector)
        monkeypatch.setattr(
            rag_service.IndicRAGPipeline, 'get_instance',
            staticmethod(lambda: _FakeRAG()))

        r = client.post('/api/v1/qa/ask', json={'question': 'प्रश्न क्या है?'})
        assert r.status_code == 200
        body = r.get_json()
        assert body['language'] == 'hindi'
        # The "not in context" answer is honest -> flagged low confidence
        assert body['low_confidence'] is True

    def test_ask_upstream_failure_502(self, client, monkeypatch):
        monkeypatch.setenv('GROQ_API_KEY', 'test-key')
        import app.services.generative_qa as gqa

        def boom(prompt, api_key, model=None, timeout=30):
            raise RuntimeError('upstream down')

        monkeypatch.setattr(gqa, '_call_groq', boom)
        from app.services import rag_service
        monkeypatch.setattr(
            rag_service.IndicRAGPipeline, 'get_instance',
            staticmethod(lambda: _FakeRAG()))
        r = client.post('/api/v1/qa/ask',
                        json={'question': 'x', 'lang': 'hindi'})
        assert r.status_code == 502

    def test_status_endpoint(self, client, monkeypatch):
        monkeypatch.setenv('GROQ_API_KEY', 'test-key')
        r = client.get('/api/v1/qa/status')
        assert r.status_code == 200
        assert r.get_json()['enabled'] is True
        monkeypatch.delenv('GROQ_API_KEY')
        r = client.get('/api/v1/qa/status')
        assert r.get_json()['enabled'] is False
