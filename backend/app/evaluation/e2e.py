"""
End-to-end pipeline test — full request lifecycle.

Walks the entire service the way a real client would, through the Flask
test client (in-process, no server required):

    1. register a user              POST /api/v1/auth/register   -> 201 + JWT
    2. login as that user           POST /api/v1/auth/login      -> 200 + JWT
    3. fetch current profile        GET  /api/v1/auth/me         -> 200
    4. translate with RAG context   POST /api/v1/translate/text  -> 200 + id
    5. submit feedback              POST /api/v1/translate/feedback -> 201
    6. list history                 GET  /api/v1/history         -> 200
    7. fetch the translation detail GET  /api/v1/history/<id>    -> 200
    8. legacy /ask flow             POST /api/v1/translate/ask   -> 200
    9. language detection           POST /api/v1/translate/detect -> 200
   10. supported languages          GET  /api/v1/translate/languages -> 200

Each stage is scored independently (passed/failed + measured latency) so a
regression anywhere in the pipeline is visible, not hidden in an aggregate.

The RAG encoder download is suppressed (keyword-fallback retrieval still
runs); a real encoder download is a model-loading concern measured by the
retrieval/throughput benchmarks, not a pipeline-contract test.

Usage (from backend/):
    python -m app.evaluation.e2e [--live]
"""

import json
import logging
import tempfile
import time

logger = logging.getLogger(__name__)


class EndToEndTest:
    """Full lifecycle test against the Flask routes."""

    def __init__(self, use_mock=True, config_name='development'):
        self.use_mock = use_mock
        self.config_name = config_name
        self.app = None
        self.client = None
        self._orig_get_translator = None

    # -- lifecycle ---------------------------------------------------------

    def _create_app(self):
        import os as _os
        from app import create_app
        from app.extensions import db

        # Rate limiting has its own tests; these suites check behaviour.
        self.app = create_app(self.config_name,
                              overrides={'RATELIMIT_ENABLED': False})
        fd, self._db_path = tempfile.mkstemp(suffix='.db')
        _os.close(fd)
        uri = 'sqlite:///' + self._db_path.replace('\\', '/')
        self.app.config['SQLALCHEMY_DATABASE_URI'] = uri
        with self.app.app_context():
            db.drop_all()
            db.create_all()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    def _install_stubs(self):
        """Echo translator + offline RAG encoder (mock mode)."""
        from app.evaluation.functional import EchoTranslator
        import app.routes.translate as translate_mod
        import app.services.rag_service as rag_mod

        self._orig_get_translator = translate_mod.get_translator
        translate_mod.get_translator = lambda: EchoTranslator()
        self._orig_encoder_loader = rag_mod.IndicRAGPipeline._try_load_encoder
        rag_mod.IndicRAGPipeline._try_load_encoder = lambda self: None

    def _restore(self):
        import os
        import app.routes.translate as translate_mod
        import app.services.rag_service as rag_mod
        if self._orig_get_translator is not None:
            translate_mod.get_translator = self._orig_get_translator
        if getattr(self, '_orig_encoder_loader', None) is not None:
            rag_mod.IndicRAGPipeline._try_load_encoder = self._orig_encoder_loader
        if getattr(self, '_db_path', None) and os.path.exists(self._db_path):
            try:
                os.remove(self._db_path)
            except OSError:
                pass

    # -- stage helpers ------------------------------------------------------

    def _stage(self, name, fn):
        """Run a stage and record pass/fail + latency."""
        t0 = time.time()
        try:
            ok, status, detail = fn()
            elapsed = round((time.time() - t0) * 1000, 2)
        except Exception as e:
            ok, status, detail = False, 'exception', str(e)
            elapsed = round((time.time() - t0) * 1000, 2)
        return {'name': name, 'passed': bool(ok), 'status': status,
                'detail': detail, 'latency_ms': elapsed}

    @staticmethod
    def _post(client, path, body, token=None):
        headers = {'Content-Type': 'application/json'}
        if token:
            headers['Authorization'] = f'Bearer {token}'
        return client.post(path, data=json.dumps(body), headers=headers)

    # -- main ---------------------------------------------------------------

    def run(self, live=None) -> dict:
        """
        Run the full lifecycle.

        Returns:
            dict: {total, passed, failed, pass_rate, mode, stages, note}
        """
        use_mock = (not live) if live is not None else self.use_mock

        if self.app is None:
            self._create_app()
            if use_mock:
                self._install_stubs()

        stages = []
        self._token = None
        self._translation_id = None

        def register():
            r = self._post(self.client, '/api/v1/auth/register',
                           {'name': 'E2E User', 'email': 'e2e@example.com',
                            'password': 'e2estrongpass1'})
            data = r.get_json() or {}
            if r.status_code == 201 and data.get('access_token'):
                self._token = data['access_token']
                return True, r.status_code, 'registered + token issued'
            return False, r.status_code, data.get('error') or 'registration failed'

        def login():
            r = self._post(self.client, '/api/v1/auth/login',
                           {'email': 'e2e@example.com',
                            'password': 'e2estrongpass1'})
            data = r.get_json() or {}
            if r.status_code == 200 and data.get('access_token'):
                self._token = data['access_token']
                return True, r.status_code, 'logged in + token issued'
            return False, r.status_code, data.get('error') or 'login failed'

        def me():
            r = self.client.get('/api/v1/auth/me', headers={
                'Authorization': f"Bearer {self._token}",
            })
            return (r.status_code == 200 and (r.get_json() or {}).get('user') is not None,
                    r.status_code, 'profile fetched')

        def translate():
            # Default include_context=True exercises the RAG path; the JWT is
            # attached so the translation lands in the user's history.
            r = self._post(self.client, '/api/v1/translate/text',
                           {'text': 'The hospital is near the railway station.',
                            'source_lang': 'english', 'target_lang': 'hindi'},
                           token=self._token)
            data = r.get_json() or {}
            if r.status_code == 200 and data.get('translated_text'):
                self._translation_id = data['translation_id']
                detail = (f"translation_id={data['translation_id']} "
                          f"context_used={data.get('context_used')} "
                          f"model={data.get('model_used')}")
                return True, r.status_code, detail
            return False, r.status_code, data.get('error') or 'translation failed'

        def feedback():
            r = self._post(self.client, '/api/v1/translate/feedback', {
                'translation_id': self._translation_id,
                'rating': 5, 'comment': 'e2e check',
            })
            if r.status_code == 201:
                return True, r.status_code, 'feedback accepted'
            return False, r.status_code, (r.get_json() or {}).get('error')

        def history():
            r = self.client.get('/api/v1/history', headers={
                'Authorization': f"Bearer {self._token}",
            })
            data = r.get_json() or {}
            items = data.get('translations') or data.get('items') or []
            ok = r.status_code == 200 and any(
                (i.get('translation_id') or i.get('id')) == self._translation_id
                for i in items)
            return ok, r.status_code, f'{len(items)} history entries'

        def history_detail():
            r = self.client.get(
                f"/api/v1/history/{self._translation_id}",
                headers={'Authorization': f"Bearer {self._token}"},
            )
            data = r.get_json() or {}
            ok = r.status_code == 200 and bool(
                data.get('translated_text') or data.get('source_text'))
            return ok, r.status_code, 'translation detail fetched'

        def ask():
            r = self._post(self.client, '/api/v1/translate/ask',
                           {'query': 'How do I open a bank account?',
                            'source_lang': 'english', 'target_lang': 'hindi',
                            'include_context': False})
            data = r.get_json() or {}
            return (bool(data.get('translated_text')),
                    r.status_code, 'legacy /ask worked')

        def detect():
            r = self._post(self.client, '/api/v1/translate/detect',
                           {'text': 'भारत एक विविध देश है।'})
            return (r.get_json() or {}).get('detected_language') == 'hindi', \
                r.status_code, 'language detected'

        def languages():
            r = self.client.get('/api/v1/translate/languages')
            data = r.get_json() or {}
            return (bool(data.get('languages')), r.status_code,
                    f"{len(data.get('languages') or [])} languages")

        for name, fn in [('register', register), ('login', login), ('me', me),
                         ('translate_with_rag', translate), ('feedback', feedback),
                         ('history_list', history), ('history_detail', history_detail),
                         ('ask_legacy', ask), ('detect', detect),
                         ('languages', languages)]:
            stages.append(self._stage(name, fn))

        try:
            passed = sum(1 for s in stages if s['passed'])
            return {
                'total': len(stages),
                'passed': passed,
                'failed': len(stages) - passed,
                'pass_rate': round(passed / max(len(stages), 1) * 100, 2),
                'mode': 'mock' if use_mock else 'live',
                'stages': stages,
                'note': (
                    'Full lifecycle: auth -> translate (with RAG context) -> '
                    'feedback -> history. Mock mode uses the echo translator; '
                    'live mode uses the real ModelRegistry.'
                ),
            }
        finally:
            if use_mock:
                self._restore()


def run_e2e_test(use_mock=True) -> dict:
    """Module-level convenience entry point."""
    return EndToEndTest(use_mock=use_mock).run()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='End-to-end pipeline test')
    parser.add_argument('--live', action='store_true', help='Use real models (slow)')
    args = parser.parse_args()

    result = run_e2e_test(use_mock=not args.live)
    print(f"\nEnd-to-end test ({result['mode']} mode): "
          f"{result['passed']}/{result['total']} stages passed "
          f"({result['pass_rate']}%)")
    for stage in result['stages']:
        status = 'PASS' if stage['passed'] else 'FAIL'
        print(f"  [{status}] {stage['name']:<22} "
              f"(status={stage['status']}, {stage['latency_ms']}ms)")
        if not stage['passed'] and stage.get('detail'):
            print(f"         -> {stage['detail']}")
    print(f"Note: {result['note']}")
