"""
Functional / API contract test suite for the translation service.

Runs the plan's "Benchmark #1 — Functional Testing" matrix against the
live Flask routes using an in-process test client (no server required):

    Empty input            Unsupported language      Unicode text
    Valid EN->MR/MR->HI    Very long text            Emoji
    Numbers preserved      URLs preserved            Punctuation
    Multiple sentences     Malformed JSON            No body
    Wrong method           Model crash (500)         Empty model output (502)

Two modes:
  * mock  (default) — a deterministic stub translator is injected, so the
    whole matrix runs in seconds without loading any ML model.
  * live  — the real ModelRegistry is used; only the "valid translation"
    cases run (slow, needs models installed).

Usage (from translator-backend/):
    python -m app.evaluation.functional [--live]
"""

import json
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stub translators (deterministic, model-free)
# ---------------------------------------------------------------------------


class EchoTranslator:
    """Returns the input unchanged so preservation checks are meaningful."""

    def translate(self, text, source_lang, target_lang, context=None, model_name=None):
        return {
            'original': text,
            'translated': text,
            'source_lang': source_lang,
            'target_lang': target_lang,
            'context_used': context,
            'metadata': {
                'model': 'mock-echo',
                'src_code': None,
                'tgt_code': None,
                'processing_time_ms': 1.0,
                'tm_hit': False,
                'tm_similarity': 0.0,
                'tm_hit_count': 0,
                'error': None,
            },
        }


class FailTranslator:
    """Raises — the route must turn this into a graceful 500."""

    def translate(self, text, source_lang, target_lang, context=None, model_name=None):
        raise RuntimeError('mock model failure')


class EmptyTranslator:
    """Returns an empty translation with error metadata — route must 502."""

    def translate(self, text, source_lang, target_lang, context=None, model_name=None):
        return {
            'original': text,
            'translated': '',
            'source_lang': source_lang,
            'target_lang': target_lang,
            'context_used': None,
            'metadata': {
                'model': 'mock-empty',
                'src_code': None,
                'tgt_code': None,
                'processing_time_ms': 1.0,
                'tm_hit': False,
                'tm_similarity': 0.0,
                'tm_hit_count': 0,
                'error': 'mock model produced no output',
            },
        }


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

def _has(json_body, needle):
    return needle in (json_body.get('translated_text') or '')


# Each case: (name, method, path, body, expect_status, check_fn, modes, stub)
# body is either a dict (JSON), a raw string (malformed JSON) or None (no body).
PATH = '/api/v1/translate/text'

FUNCTIONAL_CASES = [
    # ---- Validation / error handling -------------------------------------
    dict(name='empty input rejected', method='POST', path=PATH,
         body={'text': '', 'source_lang': 'english', 'target_lang': 'hindi'},
         expect=400, modes=['mock', 'live'],
         check=lambda r: r.get_json().get('error') is not None),
    dict(name='whitespace-only input rejected', method='POST', path=PATH,
         body={'text': '   ', 'source_lang': 'english', 'target_lang': 'hindi'},
         expect=400, modes=['mock', 'live'],
         check=lambda r: r.get_json().get('error') is not None),
    dict(name='missing text field rejected', method='POST', path=PATH,
         body={'source_lang': 'english', 'target_lang': 'hindi'},
         expect=400, modes=['mock', 'live'],
         check=lambda r: r.get_json().get('error') is not None),
    dict(name='unsupported source language rejected', method='POST', path=PATH,
         body={'text': 'Hello', 'source_lang': 'klingon', 'target_lang': 'hindi'},
         expect=400, modes=['mock', 'live'],
         check=lambda r: r.get_json().get('error') is not None),
    dict(name='unsupported target language rejected', method='POST', path=PATH,
         body={'text': 'Hello', 'source_lang': 'english', 'target_lang': 'elvish'},
         expect=400, modes=['mock', 'live'],
         check=lambda r: r.get_json().get('error') is not None),
    dict(name='same source/target rejected', method='POST', path=PATH,
         body={'text': 'Hello', 'source_lang': 'hindi', 'target_lang': 'hindi'},
         expect=400, modes=['mock', 'live'],
         check=lambda r: r.get_json().get('error') is not None),
    dict(name='over-long text rejected (5001 chars)', method='POST', path=PATH,
         body={'text': 'a' * 5001, 'source_lang': 'english', 'target_lang': 'hindi'},
         expect=400, modes=['mock', 'live'],
         check=lambda r: r.get_json().get('error') is not None),
    dict(name='no JSON body rejected', method='POST', path=PATH,
         body=None, expect=400, modes=['mock', 'live'],
         check=lambda r: r.get_json().get('error') is not None),
    dict(name='malformed JSON rejected', method='POST', path=PATH,
         body='{"text": "Hello", "source_lang": "english"', expect=400,
         modes=['mock', 'live'],
         check=lambda r: r.get_json().get('error') is not None),
    dict(name='wrong method returns 405', method='GET', path=PATH,
         body=None, expect=405, modes=['mock', 'live'], check=None),

    # ---- Happy paths (mock: echo; live: real translation) ----------------
    dict(name='valid EN->MR returns translation', method='POST', path=PATH,
         body={'text': 'India is a diverse country.', 'source_lang': 'english',
               'target_lang': 'marathi', 'include_context': False},
         expect=200, modes=['mock', 'live'],
         check=lambda r: bool((r.get_json() or {}).get('translated_text'))),
    dict(name='valid MR->HI returns translation', method='POST', path=PATH,
         body={'text': 'नमस्कार, मी विद्यार्थी आहे.', 'source_lang': 'marathi',
               'target_lang': 'hindi', 'include_context': False},
         expect=200, modes=['mock', 'live'],
         check=lambda r: bool((r.get_json() or {}).get('translated_text'))),
    dict(name='valid HI->BN returns translation', method='POST', path=PATH,
         body={'text': 'मैं कल दिल्ली जा रहा हूँ।', 'source_lang': 'hindi',
               'target_lang': 'bengali', 'include_context': False},
         expect=200, modes=['mock', 'live'],
         check=lambda r: bool((r.get_json() or {}).get('translated_text'))),

    # ---- Content robustness (mock only — deterministic echo) -------------
    dict(name='unicode Indic text handled', method='POST', path=PATH,
         body={'text': 'भारत एक विविध देश है।', 'source_lang': 'hindi',
               'target_lang': 'english', 'include_context': False},
         expect=200, modes=['mock'], check=lambda r: bool(r.get_json().get('translated_text'))),
    dict(name='emoji does not crash', method='POST', path=PATH,
         body={'text': 'Hello world 🌸😊', 'source_lang': 'english',
               'target_lang': 'hindi', 'include_context': False},
         expect=200, modes=['mock'], check=lambda r: bool(r.get_json().get('translated_text'))),
    dict(name='numbers preserved', method='POST', path=PATH,
         body={'text': 'The total is 12,450 rupees with 8% tax in 2026.',
               'source_lang': 'english', 'target_lang': 'hindi', 'include_context': False},
         expect=200, modes=['mock'], check=lambda r: all(
             t in r.get_json().get('translated_text', '') for t in ('12,450', '8%', '2026'))),
    dict(name='URLs preserved', method='POST', path=PATH,
         body={'text': 'Visit https://example.com/india?lang=hi for details.',
               'source_lang': 'english', 'target_lang': 'hindi', 'include_context': False},
         expect=200, modes=['mock'], check=lambda r: 'https://example.com/india?lang=hi'
             in r.get_json().get('translated_text', '')),
    dict(name='punctuation preserved', method='POST', path=PATH,
         body={'text': 'Hello, world! How are you? (I am fine.)',
               'source_lang': 'english', 'target_lang': 'hindi', 'include_context': False},
         expect=200, modes=['mock'], check=lambda r: all(
             c in r.get_json().get('translated_text', '') for c in (',', '!', '?', '(', '.)'))),
    dict(name='multiple sentences handled', method='POST', path=PATH,
         body={'text': 'First sentence. Second sentence! Third sentence?',
               'source_lang': 'english', 'target_lang': 'hindi', 'include_context': False},
         expect=200, modes=['mock'], check=lambda r: bool(r.get_json().get('translated_text'))),

    # ---- Language detection endpoint ------------------------------------
    dict(name='language detection endpoint works', method='POST',
         path='/api/v1/translate/detect',
         body={'text': 'भारत एक विविध देश है।'}, expect=200,
         modes=['mock', 'live'],
         check=lambda r: r.get_json().get('detected_language') == 'hindi'),
    dict(name='language detection rejects empty text', method='POST',
         path='/api/v1/translate/detect',
         body={'text': '   '}, expect=400, modes=['mock', 'live'],
         check=lambda r: r.get_json().get('error') is not None),
    dict(name='language detection rejects non-string text', method='POST',
         path='/api/v1/translate/detect',
         body={'text': 12345}, expect=400, modes=['mock', 'live'],
         check=lambda r: r.get_json().get('error') is not None),
    dict(name='detect returns unknown for no-script text', method='POST',
         path='/api/v1/translate/detect',
         body={'text': '###!!! 1234'}, expect=200, modes=['mock', 'live'],
         check=lambda r: r.get_json().get('detected_language') == 'unknown'),
    dict(name='translate rejects non-string text', method='POST', path=PATH,
         body={'text': 12345, 'source_lang': 'english', 'target_lang': 'hindi'},
         expect=400, modes=['mock', 'live'],
         check=lambda r: r.get_json().get('error') is not None),

    # ---- Failure semantics (mock only) -----------------------------------
    dict(name='model crash returns graceful 500', method='POST', path=PATH,
         body={'text': 'Hello', 'source_lang': 'english', 'target_lang': 'hindi',
               'include_context': False},
         expect=500, modes=['mock'], stub='fail', check=None),
    dict(name='empty model output returns 502', method='POST', path=PATH,
         body={'text': 'Hello', 'source_lang': 'english', 'target_lang': 'hindi',
               'include_context': False},
         expect=502, modes=['mock'], stub='empty', check=None),
]


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------

class FunctionalTestSuite:
    """Runs the functional test matrix against the Flask routes."""

    def __init__(self, use_mock=True, config_name='development'):
        self.use_mock = use_mock
        self.config_name = config_name
        self.app = None
        self.client = None
        self._orig_get_translator = None

    # -- lifecycle ---------------------------------------------------------

    def _create_app(self):
        import os as _os
        import tempfile
        from app import create_app
        from app.extensions import db

        self.app = create_app(self.config_name)

        # Isolate DB writes into a throwaway sqlite file so the functional
        # matrix never pollutes the development database (history UI, etc.)
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
        """Replace the route's translator getter with deterministic stubs."""
        import app.routes.translate as translate_mod
        self._orig_get_translator = translate_mod.get_translator

        def make_stub(stub_cls):
            def get_stub():
                return stub_cls()
            return get_stub

        self._stubs = {
            'echo': make_stub(EchoTranslator),
            'fail': make_stub(FailTranslator),
            'empty': make_stub(EmptyTranslator),
        }
        translate_mod.get_translator = self._stubs['echo']

    def _restore(self):
        import os
        import app.routes.translate as translate_mod
        if self._orig_get_translator is not None:
            translate_mod.get_translator = self._orig_get_translator
        if getattr(self, '_db_path', None) and os.path.exists(self._db_path):
            try:
                os.remove(self._db_path)
            except OSError:
                pass

    # -- execution ---------------------------------------------------------

    def run(self, cases=None, live=None) -> dict:
        """
        Run the functional test matrix.

        Args:
            cases: Optional subset of FUNCTIONAL_CASES (default: all).
            live: Force live mode (True) or mock mode (False). Defaults to
                  the suite's use_mock setting.

        Returns:
            dict: {total, passed, failed, pass_rate, cases: [...]}
        """
        use_mock = (not live) if live is not None else self.use_mock

        if self.app is None:
            self._create_app()
            if use_mock:
                self._install_stubs()

        cases = cases or FUNCTIONAL_CASES
        results = []

        try:
            for case in cases:
                # Respect each case's mode whitelist
                allowed = case.get('modes', ['mock', 'live'])
                if use_mock and 'mock' not in allowed:
                    continue
                if not use_mock and 'live' not in allowed:
                    continue

                # Per-case stub (for failure-semantics cases)
                if use_mock:
                    stub = case.get('stub', 'echo')
                    import app.routes.translate as translate_mod
                    translate_mod.get_translator = self._stubs.get(stub, self._stubs['echo'])

                result = self._run_case(case)
                results.append(result)

            passed = sum(1 for r in results if r['passed'])
            return {
                'total': len(results),
                'passed': passed,
                'failed': len(results) - passed,
                'pass_rate': round(passed / max(len(results), 1) * 100, 2),
                'mode': 'mock' if use_mock else 'live',
                'cases': results,
            }
        finally:
            if use_mock:
                self._restore()

    def _run_case(self, case) -> dict:
        body = case.get('body')
        if body is None:
            data = None
            content_type = 'application/json'
        elif isinstance(body, str):
            data = body
            content_type = 'application/json'
        else:
            data = json.dumps(body)
            content_type = 'application/json'

        try:
            resp = self.client.open(
                case['path'],
                method=case['method'],
                data=data,
                content_type=content_type,
            )
        except Exception as e:
            return {
                'name': case['name'], 'passed': False,
                'status': 'exception', 'expected': case['expect'],
                'detail': f'request raised: {e}',
            }

        status = resp.status_code
        check = case.get('check')
        ok_status = status == case['expect']
        ok_check = True
        detail = None
        if ok_status and check is not None:
            try:
                ok_check = bool(check(resp))
                if not ok_check:
                    detail = (resp.get_json() or {}).get('error') or 'check failed'
            except Exception as e:
                ok_check = False
                detail = f'check raised: {e}'

        passed = ok_status and ok_check
        if not passed and detail is None:
            detail = (resp.get_json() or {}).get('error') or f'status {status}'

        return {
            'name': case['name'],
            'passed': passed,
            'status': status,
            'expected': case['expect'],
            'detail': detail,
        }


def run_functional_tests(use_mock=True) -> dict:
    """Module-level convenience entry point."""
    suite = FunctionalTestSuite(use_mock=use_mock)
    return suite.run()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run functional/API contract tests')
    parser.add_argument('--live', action='store_true', help='Use real models (slow)')
    args = parser.parse_args()

    result = run_functional_tests(use_mock=not args.live)
    print(f"\nFunctional tests ({result['mode']} mode): "
          f"{result['passed']}/{result['total']} passed "
          f"({result['pass_rate']}%)")
    for case in result['cases']:
        status = 'PASS' if case['passed'] else 'FAIL'
        print(f"  [{status}] {case['name']} (status={case['status']}, "
              f"expected={case['expected']})")
        if not case['passed'] and case.get('detail'):
            print(f"         -> {case['detail']}")
