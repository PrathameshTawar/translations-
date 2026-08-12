"""Core unit tests for the translator backend.

Pure-logic tests only — no model loading — so they run fast and reliably in CI.
Run with: venv\\Scripts\\python -m pytest tests -q
"""

import os
import sys
import threading
from unittest.mock import MagicMock

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# Input validators
# ---------------------------------------------------------------------------

class TestValidators:
    def test_valid_translation_request(self):
        from app.utils.validators import validate_translation_request
        ok, err = validate_translation_request({
            'text': 'Hello world', 'source_lang': 'english', 'target_lang': 'hindi'
        })
        assert ok is True
        assert err is None

    def test_missing_text(self):
        from app.utils.validators import validate_translation_request
        ok, err = validate_translation_request({'source_lang': 'english', 'target_lang': 'hindi'})
        assert ok is False
        assert 'text' in err

    def test_same_language_rejected(self):
        from app.utils.validators import validate_translation_request
        ok, err = validate_translation_request({
            'text': 'x', 'source_lang': 'hindi', 'target_lang': 'hindi'
        })
        assert ok is False
        assert 'differ' in err

    def test_unsupported_language_rejected(self):
        from app.utils.validators import validate_translation_request
        ok, err = validate_translation_request({
            'text': 'x', 'source_lang': 'klingon', 'target_lang': 'hindi'
        })
        assert ok is False
        assert 'klingon' in err

    def test_overlong_text_rejected(self):
        from app.utils.validators import validate_translation_request
        ok, err = validate_translation_request({
            'text': 'a' * 5001, 'source_lang': 'english', 'target_lang': 'hindi'
        })
        assert ok is False
        assert '5000' in err

    def test_register_requires_valid_email(self):
        from app.utils.validators import validate_register_request
        ok, err = validate_register_request({'name': 'Abc', 'email': 'not-an-email', 'password': 'secret123'})
        assert ok is False
        assert 'email' in err


# ---------------------------------------------------------------------------
# Quality scorer
# ---------------------------------------------------------------------------

class TestQualityScorer:
    def test_empty_input_is_failure_not_hallucination(self):
        from app.services.quality import QualityScorer
        result = QualityScorer().score('', '')
        assert result['overall_score'] == 0.0
        assert result['hallucination_detected'] is False
        assert result['confidence'] == 'low'

    def test_identical_text_scores_low(self):
        # Identical source/translation triggers the hallucination heuristic
        from app.services.quality import QualityScorer
        result = QualityScorer().score('hello world', 'hello world')
        assert result['hallucination_detected'] is True
        assert result['overall_score'] < 0.75

    def test_good_translation_scores_high(self):
        from app.services.quality import QualityScorer
        result = QualityScorer().score('good morning everyone', 'शुभ प्रभात सभी को')
        assert result['overall_score'] >= 0.5

    def test_wrong_language_penalized_when_target_known(self):
        # German output for an English->Hindi job: script mismatch must sink it.
        from app.services.quality import QualityScorer
        result = QualityScorer().score(
            'The weather is beautiful today.',
            'El tiempo es agradable hoy.',
            target_lang='hindi',
        )
        assert result['script_consistency'] == 0.0
        assert result['overall_score'] < 0.75

    def test_truncated_output_penalized(self):
        from app.services.quality import QualityScorer
        result = QualityScorer().score(
            'The bank offers a low interest rate on education loans.',
            'बैंक शिक्षा ऋण…',
            target_lang='hindi',
        )
        assert result['sentence_end'] < 1.0
        assert result['overall_score'] < 0.75

    def test_script_check_neutral_without_target_lang(self):
        # Backward compatibility: no target_lang -> script contributes 1.0.
        from app.services.quality import QualityScorer
        result = QualityScorer().score('good morning everyone', 'शुभ प्रभात सभी को')
        assert result['script_consistency'] == 1.0
        assert result['overall_score'] >= 0.5


# ---------------------------------------------------------------------------
# Quality scorer — Devanagari word-order sanity rules
# ---------------------------------------------------------------------------

class TestQualityScorerWordOrder:
    """Scrambled vs natural Hindi/Marathi output must be separable.

    The word-order rules flag postposition/auxiliary misplacement — the
    signature of word-salad that kept its sentence-final mark — while never
    firing on natural sentences (including those that open with temporal or
    locative adverbs, and compound postpositions like के लिए).
    """

    def test_scrambled_output_flagged_and_capped(self):
        # 'सकते को' — auxiliary verb directly before a postposition.
        from app.services.quality import QualityScorer
        result = QualityScorer().score(
            'Can you help me understand this document?',
            'क्या सकते को में मदद मुझे इस आप दस्तावेज़ कर समझने हैं?',
        )
        assert result['word_order_suspicious'] is True
        # Word-order violations are capped below the 0.80 'good' boundary.
        assert result['overall_score'] <= 0.70
        assert result['confidence'] != 'high'

    def test_adjacent_postpositions_flagged(self):
        # 'को में' — two postpositions back-to-back is never grammatical.
        from app.services.quality import QualityScorer
        result = QualityScorer().score('hello world', 'क्या सकते को में मदद।')
        assert result['word_order_suspicious'] is True
        assert result['overall_score'] <= 0.70

    def test_postposition_opening_sentence_flagged(self):
        # A sentence can never open with a postposition (में…).
        from app.services.quality import QualityScorer
        result = QualityScorer().score('hello world', 'में की यह है, रात आज देखना।')
        assert result['word_order_suspicious'] is True

    def test_compound_postpositions_not_flagged(self):
        # Valid compounds — including danda-attached tokens like 'लिए।' (the
        # punctuation regression) — must NOT trip the adjacency rule.
        from app.services.quality import QualityScorer
        for trans in (
            'अच्छे स्वास्थ्य के लिए नियमित व्यायाम महत्वपूर्ण हैं।',
            'वह मेरे साथ आया।',
            'काम के अनुसार हुआ।',
            'इन लोगों में से एक जीतेगा।',
        ):
            result = QualityScorer().score('hello world', trans)
            assert result['word_order_suspicious'] is False, trans

    def test_natural_sentences_never_flagged(self):
        # Temporal/locative adverbs legitimately open sentences and combine
        # with case markers — these must not read as scrambled.
        from app.services.quality import QualityScorer
        for trans in (
            'पहले मैं गया था।',
            'बाहर जाओ।',
            'हम बाद में मिलेंगे।',
            'यह पहले से तैयार है।',
            'किताब के बारे में बात करो।',
            'ऊपर से नीचे तक देखो।',
            'सब कुछ ठीक है।',
        ):
            result = QualityScorer().score('hello world', trans)
            assert result['word_order_suspicious'] is False, trans

    def test_cap_only_applies_to_suspicious_output(self):
        # A natural sentence keeps its full score; its scrambled twin is capped.
        from app.services.quality import QualityScorer
        natural = QualityScorer().score('hello world', 'सब कुछ ठीक है।')
        scrambled = QualityScorer().score('hello world', 'में की यह है ठीक।')
        assert natural['word_order_suspicious'] is False
        assert scrambled['word_order_suspicious'] is True
        assert natural['overall_score'] > scrambled['overall_score']

    def test_non_devanagari_and_empty_never_flagged(self):
        # The rules are Devanagari-specific: Latin text and empty output are
        # never flagged, and the empty-result shape still carries the key.
        from app.services.quality import QualityScorer
        latin = QualityScorer().score('hello world', 'The weather is nice today.')
        assert latin['word_order_suspicious'] is False
        empty = QualityScorer().score('hello world', '')
        assert empty['word_order_suspicious'] is False


# ---------------------------------------------------------------------------
# Cross-encoder reranker (reordering logic + graceful fallback)
# ---------------------------------------------------------------------------

class _FakeCrossEncoder:
    """Deterministic fake: scores each (query, candidate) pair by candidate
    text length, so longer candidates always outrank shorter ones."""

    def predict(self, pairs, batch_size=None):
        return [float(len(pair[1])) for pair in pairs]


class TestCrossEncoderReranker:
    def test_rerank_reorders_by_score(self, monkeypatch):
        from app.services.reranker import (
            CrossEncoderReranker, reset_reranker_singleton,
        )
        reset_reranker_singleton()
        monkeypatch.setattr('app.services.reranker._load_reranker_model',
                            lambda: _FakeCrossEncoder())
        out = CrossEncoderReranker().rerank('query', [
            {'term': 'a'},
            {'term': 'ccc'},
            {'term': 'bb'},
        ])
        assert out['available'] is True
        assert [c['term'] for c in out['reranked']] == ['ccc', 'bb', 'a']
        assert out['scores'] == [3.0, 2.0, 1.0]

    def test_top_k_truncation(self, monkeypatch):
        from app.services.reranker import (
            CrossEncoderReranker, reset_reranker_singleton,
        )
        reset_reranker_singleton()
        monkeypatch.setattr('app.services.reranker._load_reranker_model',
                            lambda: _FakeCrossEncoder())
        out = CrossEncoderReranker().rerank('q', [
            {'term': 'a'}, {'term': 'bb'}, {'term': 'ccc'},
        ], top_k=2)
        assert [c['term'] for c in out['reranked']] == ['ccc', 'bb']

    def test_unavailable_returns_input_order(self, monkeypatch):
        from app.services.reranker import (
            CrossEncoderReranker, reset_reranker_singleton,
        )
        reset_reranker_singleton()
        monkeypatch.setattr('app.services.reranker._load_reranker_model',
                            lambda: None)
        candidates = [{'term': 'x'}, {'term': 'y'}]
        out = CrossEncoderReranker().rerank('q', candidates)
        assert out['available'] is False
        assert out['reranked'] == candidates  # unchanged order
        assert out['scores'] == []

    def test_scoring_failure_degrades_gracefully(self, monkeypatch):
        from app.services.reranker import (
            CrossEncoderReranker, reset_reranker_singleton,
        )
        reset_reranker_singleton()

        class _Boom:
            def predict(self, pairs, batch_size=None):
                raise RuntimeError('boom')

        monkeypatch.setattr('app.services.reranker._load_reranker_model',
                            lambda: _Boom())
        out = CrossEncoderReranker().rerank('q', [
            {'term': 'x'}, {'term': 'y'},
        ])
        assert out['available'] is False
        assert [c['term'] for c in out['reranked']] == ['x', 'y']

    def test_empty_candidates_short_circuit(self, monkeypatch):
        from app.services.reranker import (
            CrossEncoderReranker, reset_reranker_singleton,
        )
        reset_reranker_singleton()
        monkeypatch.setattr('app.services.reranker._load_reranker_model',
                            lambda: _FakeCrossEncoder())
        out = CrossEncoderReranker().rerank('q', [])
        assert out['reranked'] == []
        assert out['available'] is False

    def test_string_candidates_accepted(self, monkeypatch):
        from app.services.reranker import (
            CrossEncoderReranker, reset_reranker_singleton,
        )
        reset_reranker_singleton()
        monkeypatch.setattr('app.services.reranker._load_reranker_model',
                            lambda: _FakeCrossEncoder())
        out = CrossEncoderReranker().rerank('q', ['z', 'mno'])
        assert out['reranked'] == ['mno', 'z']

    def test_concurrent_first_use_loads_model_once(self, monkeypatch):
        # The lazy singleton must survive concurrent first-use: the lock's
        # double-checked guard guarantees a single CrossEncoder construction
        # even when threads race the same probe (e.g. the concurrent
        # translate path with use_reranker=True).
        import sys
        import types
        from app.services import reranker as reranker_mod
        from app.services.reranker import reset_reranker_singleton

        reset_reranker_singleton()

        constructed = []

        class _CountingFake:
            def __init__(self, *a, **k):
                constructed.append(threading.get_ident())

            def predict(self, pairs, batch_size=None):
                return [float(len(p[1])) for p in pairs]

        # The real _load_reranker_model() does `from sentence_transformers
        # import CrossEncoder` internally, so patch sys.modules to keep the
        # test hermetic (no package import, no model download).
        fake_st = types.ModuleType('sentence_transformers')
        fake_st.CrossEncoder = _CountingFake
        monkeypatch.setitem(sys.modules, 'sentence_transformers', fake_st)

        barrier = threading.Barrier(4)
        results = []
        errors = []

        def worker():
            try:
                barrier.wait()
                results.append(reranker_mod._load_reranker_model())
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        assert len(results) == 4
        assert all(m is not None for m in results)
        # Exactly one model construction despite four racing threads.
        assert len(constructed) == 1
        reset_reranker_singleton()


# ---------------------------------------------------------------------------
# RAG pipeline — reranker wiring (use_reranker flag)
# ---------------------------------------------------------------------------

class TestRagRerankerWiring:
    def _make_rag(self):
        from app.services.rag_service import IndicRAGPipeline
        rag = IndicRAGPipeline.__new__(IndicRAGPipeline)
        rag.encoder = None
        rag.indices = {}
        rag.mappings = {}
        rag.terminology_db = {}
        rag._reranker = None
        rag._load_terminology_db()
        return rag

    def test_use_reranker_reorders_terms(self, monkeypatch):
        # Keyword path returns ['health', 'hospital'] (insertion order for
        # the tied query 'health hospital'); the fake cross-encoder (score by
        # term length) must move 'hospital' to the front.
        from app.services.reranker import reset_reranker_singleton
        reset_reranker_singleton()
        monkeypatch.setattr('app.services.reranker._load_reranker_model',
                            lambda: _FakeCrossEncoder())
        rag = self._make_rag()

        res_off = rag.retrieve_context('health hospital', target_lang='hindi',
                                       top_k=2)
        assert res_off['reranked'] is False
        assert [t['term'] for t in res_off['terms']] == ['health', 'hospital']

        res_on = rag.retrieve_context('health hospital', target_lang='hindi',
                                      top_k=2, use_reranker=True)
        assert res_on['reranked'] is True
        assert res_on['reranker_available'] is True
        assert [t['term'] for t in res_on['terms']] == ['hospital', 'health']
        assert 'hospital' in res_on['context']

    def test_unavailable_reranker_falls_back(self, monkeypatch):
        from app.services.reranker import reset_reranker_singleton
        reset_reranker_singleton()
        monkeypatch.setattr('app.services.reranker._load_reranker_model',
                            lambda: None)
        rag = self._make_rag()
        res = rag.retrieve_context('health hospital', target_lang='hindi',
                                   top_k=2, use_reranker=True)
        assert res['reranked'] is False
        assert res['reranker_available'] is False
        assert [t['term'] for t in res['terms']] == ['health', 'hospital']


# ---------------------------------------------------------------------------
# COMETkiwi reference-free QE (normalization + graceful fallback)
# ---------------------------------------------------------------------------

class TestCometkiwiQE:
    def test_raw_to_score_maps_unit_range(self):
        from app.services.quality import cometkiwi_raw_to_score
        assert cometkiwi_raw_to_score(-1.0) == 0.0
        assert cometkiwi_raw_to_score(0.0) == 0.5
        assert cometkiwi_raw_to_score(1.0) == 1.0

    def test_raw_to_score_clamps_out_of_range(self):
        from app.services.quality import cometkiwi_raw_to_score
        assert cometkiwi_raw_to_score(-3.0) == 0.0
        assert cometkiwi_raw_to_score(5.0) == 1.0
        assert 0.0 <= cometkiwi_raw_to_score(0.3) <= 1.0

    def test_scorer_unavailable_degrades_gracefully(self, monkeypatch):
        # Simulate the model being unloadable (no internet / no weights).
        from app.services.quality import COMETkiwiQualityScorer
        monkeypatch.setattr('app.services.quality._load_cometkiwi_model', lambda: None)
        scorer = COMETkiwiQualityScorer()
        assert scorer.available is False
        result = scorer.score('hello world', 'नमस्ते दुनिया')
        assert result['error']
        assert result['results'] == []

    def test_qe_benchmark_both_falls_back_to_heuristic(self, monkeypatch):
        # When COMETkiwi is unavailable, 'both' still reports the heuristic
        # numbers head-to-head with cometkiwi marked unavailable.
        from app.evaluation.quality_estimation import QualityEstimationBenchmark
        monkeypatch.setattr('app.services.quality._load_cometkiwi_model', lambda: None)
        result = QualityEstimationBenchmark(scorer='both').run()
        assert result['scorer'] == 'heuristic'
        assert result['num_samples'] > 0
        assert result['f1'] > 0.5
        assert result['scorers']['heuristic']['available'] is True
        assert result['scorers']['cometkiwi']['available'] is False
        assert result['winner'] == 'heuristic'

    def test_qe_benchmark_invalid_scorer_raises(self):
        import pytest
        from app.evaluation.quality_estimation import QualityEstimationBenchmark
        with pytest.raises(ValueError):
            QualityEstimationBenchmark(scorer='bogus')

    def test_cometkiwi_scorer_primary_reports_f1(self, monkeypatch):
        # scorer='cometkiwi' must degrade to an error result (not crash) when
        # the model is unavailable, so run_benchmark --qe-model cometkiwi
        # fails gracefully offline.
        from app.evaluation.quality_estimation import QualityEstimationBenchmark
        monkeypatch.setattr('app.services.quality._load_cometkiwi_model', lambda: None)
        result = QualityEstimationBenchmark(scorer='cometkiwi').run()
        assert result.get('error')
        assert result['scorers']['cometkiwi']['available'] is False


# ---------------------------------------------------------------------------
# Evaluator summary (metric keys + value scales the frontend depends on)
# ---------------------------------------------------------------------------

class TestEvaluatorSummary:
    @pytest.fixture
    def evaluator(self):
        from app.evaluation.evaluator import Evaluator
        # Only the 'quality' scorer is dependency-free and always available;
        # summary computation is logic-only so a minimal evaluator works.
        return Evaluator(metrics=['quality'])

    def _results(self):
        return {
            'metrics': {
                'bleu': {'bleu': 32.5},
                'chrf': {'chrf_plus': 48.2},
                'bertscore': {'f1': 0.87},
                'quality': {'quality_score': 0.91},
            },
        }

    def test_compute_summary_uses_expected_keys(self, evaluator):
        summary = evaluator._compute_summary(self._results())
        assert summary['bleu'] == 32.5        # 0-100 scale
        assert summary['chrf_plus'] == 48.2   # 0-100 scale
        assert summary['bertscore_f1'] == 0.87  # 0-1 scale
        assert summary['quality_score'] == 0.91 # 0-1 scale

    def test_compute_summary_missing_metric_is_none(self, evaluator):
        summary = evaluator._compute_summary({'metrics': {}})
        assert summary == {
            'bleu': None, 'chrf_plus': None,
            'bertscore_f1': None, 'comet_score': None,
            'quality_score': None,
        }


# ---------------------------------------------------------------------------
# Evaluate route summary builder (reads per-model summaries)
# ---------------------------------------------------------------------------

class TestRouteSummary:
    def test_build_summary_merges_per_model(self):
        from app.routes.evaluate import _build_summary
        results = {
            'hindi': {
                'models': {
                    'm1': {
                        'summary': {'bleu': 30.0, 'chrf_plus': None, 'bertscore_f1': 0.8},
                        'latency': {'mean_ms': 120.5},
                    },
                    'm2': {
                        'summary': {'bleu': 25.0, 'chrf_plus': 40.0, 'bertscore_f1': 0.7},
                    },
                }
            },
        }
        summary = _build_summary(results)
        assert 'en→hindi (m1)' in summary
        assert summary['en→hindi (m1)']['bleu'] == 30.0
        assert summary['en→hindi (m1)']['bertscore_f1'] == 0.8
        # None values are dropped
        assert 'chrf_plus' not in summary['en→hindi (m1)']
        # latency is attached when present
        assert summary['en→hindi (m1)']['latency_ms'] == 120.5
        assert 'en→hindi (m2)' in summary


# ---------------------------------------------------------------------------
# Translation Memory service (regression: entry/embedding row alignment)
# ---------------------------------------------------------------------------

class TestTranslationMemory:
    def _make_tm(self, encoder=None, encoder_available=False):
        from app.services.translation_memory import TranslationMemory
        tm = TranslationMemory.__new__(TranslationMemory)
        tm._encoder = encoder
        tm._encoder_available = encoder_available
        tm._encoder_failed = not encoder_available
        tm._entries = []
        tm._embeddings = None
        tm._embed_rows = []
        tm._similarity_threshold = 0.92
        tm._stats = {'total_entries': 0, 'hits': 0, 'misses': 0, 'size_bytes': 0}
        tm._lock = threading.Lock()
        return tm

    def test_store_without_encoder_keeps_exact_match_path(self):
        tm = self._make_tm()
        tm.store('hello', 'नमस्ते', 'english', 'hindi', 'test-model')
        assert tm._stats['total_entries'] == 1
        assert tm._embed_rows == [None]

    def test_store_with_failing_encoder_does_not_drift(self):
        encoder = MagicMock()
        encoder.encode.side_effect = RuntimeError('boom')
        tm = self._make_tm(encoder=encoder, encoder_available=True)
        tm.store('hello', 'नमस्ते', 'english', 'hindi', 'test-model')
        assert tm._stats['total_entries'] == 1
        assert tm._embed_rows == [None]

    def test_store_with_working_encoder_tracks_embedding_rows(self):
        import numpy as np
        encoder = MagicMock()
        encoder.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        tm = self._make_tm(encoder=encoder, encoder_available=True)
        tm.store('hello', 'नमस्ते', 'english', 'hindi', 'test-model')
        assert tm._embed_rows == [0]
        assert tm._embeddings is not None


# ---------------------------------------------------------------------------
# Latency percentile (P95 must be a percentile, not the max of a small sample)
# ---------------------------------------------------------------------------

class TestLatencyPercentile:
    def test_p95_small_sample_is_not_the_max(self):
        from app.evaluation.metrics.latency import _percentile
        # 9 samples: the old int(9 * 0.95) index = 8 -> the maximum (1000).
        vals = sorted([100.0, 200.0, 300.0, 400.0, 500.0,
                       600.0, 700.0, 800.0, 1000.0])
        p95 = _percentile(vals, 95)
        assert p95 < 1000.0, f'P95 of 9 samples must not be the max, got {p95}'
        assert 800.0 <= p95 < 1000.0

    def test_p95_matches_numpy_default_interpolation(self):
        import numpy as np
        from app.evaluation.metrics.latency import _percentile
        rng = np.random.default_rng(7)
        vals = sorted(rng.uniform(100, 1000, 25).tolist())
        assert _percentile(vals, 95) == float(np.percentile(vals, 95))

    def test_empty_and_single(self):
        from app.evaluation.metrics.latency import _percentile
        assert _percentile([], 95) == 0.0
        assert _percentile([42.0], 95) == 42.0


# ---------------------------------------------------------------------------
# Latency suspend guard (one system sleep must not corrupt the scorecard)
# ---------------------------------------------------------------------------

class TestLatencySuspendGuard:
    """A machine that suspends mid-benchmark must not poison the statistics.

    ``time.monotonic()`` does not advance during suspend, so a single sleep
    no longer inflates one sample by hours. The guard is the second line of
    defence: samples exceeding the guard (clock jumps, debugger attach) are
    excluded from every statistic and reported separately.
    """

    def test_suspend_artifact_excluded_from_stats(self, monkeypatch):
        from app.evaluation.metrics import latency as lat_mod

        monkeypatch.setattr(lat_mod, 'SUSPEND_GUARD_MS', 100.0)
        bench = lat_mod.LatencyBenchmark()

        # Four calls; the third absorbs a 1000 s "suspend" (monotonic jumps
        # forward), the rest are normal 10/20/15 ms inferences.
        seq = [0.0, 0.010, 0.010, 0.030, 0.030, 1000.030, 1000.030, 1000.045]
        it = iter(seq)
        monkeypatch.setattr(lat_mod.time, 'monotonic', lambda: next(it))

        called = []

        def fake_fn(text, **kw):
            called.append(text)
            return {'translated': 'x'}

        res = bench.benchmark(fake_fn, ['a', 'b', 'c', 'd'],
                              warmup_runs=0, track_memory=False)

        # All four inputs are still invoked.
        assert called == ['a', 'b', 'c', 'd']
        # The suspended sample is counted and reported separately...
        assert res['suspended_samples'] == 1
        assert res['suspended_max_ms'] == 1_000_000.0
        assert res['suspended_total_ms'] == 1_000_000.0
        # ...but excluded from the actual statistics.
        assert res['num_sentences'] == 3
        assert res['mean_ms'] == pytest.approx((10 + 20 + 15) / 3)
        assert res['max_ms'] == 20.0
        # Linear-interpolation P95 of [10, 15, 20]: rank 0.95*2 = 1.9.
        assert res['p95_ms'] == pytest.approx(19.5)

    def test_clean_run_reports_zero_suspended(self, monkeypatch):
        from app.evaluation.metrics import latency as lat_mod

        monkeypatch.setattr(lat_mod, 'SUSPEND_GUARD_MS', 100.0)
        bench = lat_mod.LatencyBenchmark()

        seq = [0.0, 0.010, 0.010, 0.025]
        it = iter(seq)
        monkeypatch.setattr(lat_mod.time, 'monotonic', lambda: next(it))

        res = bench.benchmark(lambda t, **kw: {'translated': 'x'},
                              ['a', 'b'], warmup_runs=0, track_memory=False)
        assert res['suspended_samples'] == 0
        assert res['suspended_total_ms'] == 0.0
        assert res['num_sentences'] == 2

    def test_warmup_uses_monotonic_clock(self, monkeypatch):
        from app.evaluation.metrics import latency as lat_mod

        monkeypatch.setattr(lat_mod, 'SUSPEND_GUARD_MS', 100.0)
        bench = lat_mod.LatencyBenchmark()

        # One warmup call (10 ms) + two benchmark calls (20, 30 ms).
        seq = [0.0, 0.010, 0.010, 0.030, 0.030, 0.060]
        it = iter(seq)
        monkeypatch.setattr(lat_mod.time, 'monotonic', lambda: next(it))

        res = bench.benchmark(lambda t, **kw: {'translated': 'x'},
                              ['a', 'b', 'c'], warmup_runs=1, track_memory=False)
        assert res['warmup_time_ms'] == pytest.approx(10.0)
        assert res['num_sentences'] == 2
        assert res['mean_ms'] == pytest.approx((20 + 30) / 2)

    def test_warmup_suspend_artifact_excluded(self, monkeypatch):
        from app.evaluation.metrics import latency as lat_mod

        monkeypatch.setattr(lat_mod, 'SUSPEND_GUARD_MS', 100.0)
        bench = lat_mod.LatencyBenchmark()

        # Warmup call absorbs a 1000 s suspend -> excluded from warmup_time.
        # Benchmark calls are normal (20, 30 ms).
        seq = [0.0, 1000.0, 1000.0, 1000.020, 1000.020, 1000.050]
        it = iter(seq)
        monkeypatch.setattr(lat_mod.time, 'monotonic', lambda: next(it))

        res = bench.benchmark(lambda t, **kw: {'translated': 'x'},
                              ['a', 'b', 'c'], warmup_runs=1, track_memory=False)
        assert res['warmup_time_ms'] is None  # suspend warmup dropped
        assert res['num_sentences'] == 2
        assert res['mean_ms'] == pytest.approx((20 + 30) / 2)

    def test_all_samples_suspended_reports_error_with_counts(self, monkeypatch):
        import pytest as _pytest
        from app.evaluation.metrics import latency as lat_mod

        monkeypatch.setattr(lat_mod, 'SUSPEND_GUARD_MS', 100.0)
        bench = lat_mod.LatencyBenchmark()

        # Both benchmark samples exceed the guard; warmup is disabled.
        seq = [0.0, 200.0, 200.0, 500.0]
        it = iter(seq)
        monkeypatch.setattr(lat_mod.time, 'monotonic', lambda: next(it))

        res = bench.benchmark(lambda t, **kw: {'translated': 'x'},
                              ['a', 'b'], warmup_runs=0, track_memory=False)
        assert res['error']
        assert res['latencies_ms'] == []
        assert res['suspended_samples'] == 2
        # Samples: 200 s and 300 s -> 500 s total, 300 s max.
        assert res['suspended_total_ms'] == _pytest.approx(500_000.0)
        assert res['suspended_max_ms'] == _pytest.approx(300_000.0)


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# RAG terminology glossary selection (target-language keyed)
# ---------------------------------------------------------------------------

class TestRagGlossarySelection:
    def _make_rag(self):
        from app.services.rag_service import IndicRAGPipeline
        rag = IndicRAGPipeline.__new__(IndicRAGPipeline)
        rag.encoder = None
        rag.indices = {}
        rag.mappings = {}
        rag.terminology_db = {}
        rag._load_terminology_db()
        return rag

    def test_glossary_selected_by_target_lang(self):
        # English -> Hindi must hit the Hindi glossary
        rag = self._make_rag()
        res = rag.retrieve_context(
            'The hospital is near the station.',
            source_lang='english', target_lang='hindi',
        )
        terms = [t['term'] for t in res['terms']]
        assert 'hospital' in terms
        assert res['context'] is not None
        assert 'अस्पताल' in res['context']

    def test_glossary_differs_per_target_lang(self):
        rag = self._make_rag()
        res_hi = rag.retrieve_context('hospital', target_lang='hindi')
        res_te = rag.retrieve_context('hospital', target_lang='telugu')
        hi_term = next(t for t in res_hi['terms'] if t['term'] == 'hospital')
        te_term = next(t for t in res_te['terms'] if t['term'] == 'hospital')
        assert hi_term['translation'] != te_term['translation']
        assert hi_term['translation'] == 'अस्पताल'

    def test_no_target_lang_falls_back_to_source(self):
        rag = self._make_rag()
        res = rag.retrieve_context('hospital', source_lang='hindi')
        assert any(t['term'] == 'hospital' for t in res['terms'])

    def test_english_target_has_no_glossary(self):
        # No english_terms.json exists, so Indic -> English yields no context
        rag = self._make_rag()
        res = rag.retrieve_context('hospital', source_lang='hindi', target_lang='english')
        assert res['terms'] == []
        assert res['context'] is None

    def test_unknown_language_returns_empty(self):
        rag = self._make_rag()
        res = rag.retrieve_context('hospital', target_lang='klingon')
        assert res['terms'] == []
        assert res['context'] is None


class TestConfig:
    def test_unknown_config_raises(self):
        from app.config import get_config
        with pytest.raises(RuntimeError):
            get_config('does-not-exist')

    def test_development_config_is_valid(self):
        from app.config import get_config
        cfg = get_config('development')
        assert cfg.DEBUG is True
        assert cfg.SQLALCHEMY_DATABASE_URI
        assert cfg.NLLB_MODEL


# ---------------------------------------------------------------------------
# m2m100 telugu support (regression: silent empty translations)
# ---------------------------------------------------------------------------

class _NoLoadM2M100:
    """
    M2M100Translator with model loading skipped (pure-logic tests).

    Mirrors the real wrapper's translate() path without importing torch or
    downloading weights: only the supported-language / error semantics are
    under test here.
    """

    def __init__(self):
        from app.services.translators.m2m100 import M2M100Translator
        import types
        obj = object.__new__(M2M100Translator)
        obj._model = object()
        obj._tokenizer = object()
        obj._device = -1
        obj._loaded = True
        obj._model_name = M2M100Translator.MODEL_NAME
        obj.translate = types.MethodType(M2M100Translator.translate, obj)
        self._translator = obj

    def translate(self, text, source_lang='english', target_lang='hindi'):
        return self._translator.translate(
            text=text, source_lang=source_lang, target_lang=target_lang,
        )


class TestM2M100TeluguSupport:
    """
    m2m100_418M cannot translate telugu — 'te' is not one of its 100
    languages (verified against lang_code_to_id). Requesting it must fail
    loudly, never silently return an empty translation (regression: the
    benchmark recorded all-empty telugu hypotheses with no error records).
    """

    def test_telugu_not_in_supported_languages(self):
        from app.services.translators.m2m100 import M2M100Translator
        assert 'telugu' not in M2M100Translator.LANGUAGE_CODES

    def test_other_indic_languages_still_supported(self):
        from app.services.translators.m2m100 import M2M100Translator
        for lang in ('hindi', 'marathi', 'bengali', 'gujarati', 'tamil', 'kannada'):
            assert lang in M2M100Translator.LANGUAGE_CODES, lang

    def test_telugu_translate_raises_unsupported_pair(self):
        t = _NoLoadM2M100()
        with pytest.raises(ValueError) as exc:
            t.translate('The weather is nice.', target_lang='telugu')
        assert 'telugu' in str(exc.value)
        assert 'Unsupported language pair' in str(exc.value)


class TestTranslationResultErrorPropagation:
    """TranslationResult.to_dict() must carry the error field — this is what
    lets BenchmarkRunner record wrapper-caught failures instead of silently
    treating them as empty translations."""

    def test_to_dict_includes_error(self):
        from app.services.translators.base import TranslationResult
        r = TranslationResult(original='x', translated='', source_lang='english',
                              target_lang='telugu', error='boom')
        assert r.to_dict()['error'] == 'boom'

    def test_to_dict_error_none_on_success(self):
        from app.services.translators.base import TranslationResult
        r = TranslationResult(original='x', translated='नमस्ते', source_lang='english',
                              target_lang='hindi')
        assert r.to_dict()['error'] is None


class TestBenchmarkRecordsReturnedErrors:
    """BenchmarkRunner.run() must record errors that wrappers return in the
    result dict (not only errors raised as exceptions)."""

    def _run(self, bad_fn, good_fn):
        from app.evaluation.benchmark import BenchmarkRunner
        return BenchmarkRunner(metrics=['bleu']).run(
            model_fns={'bad': bad_fn, 'good': good_fn},
            sources=['The weather is nice.'],
            references={'telugu': ['వాతావరణం బాగుంది']},
            source_lang='english', target_lang='telugu',
            run_latency=False,
        )

    def test_returned_error_recorded_with_index(self):
        def bad_fn(text, source_lang='english', target_lang='telugu'):
            return {'translated': '',
                    'error': 'Unsupported language pair: english -> telugu'}

        def good_fn(text, source_lang='english', target_lang='telugu'):
            return {'translated': 'వాతావరణం బాగుంది', 'error': None}

        out = self._run(bad_fn, good_fn)
        m = out['models']['bad']
        assert m['translations'][0]['hypothesis'] == ''
        assert len(m['errors']) == 1
        assert m['errors'][0]['index'] == 0
        assert 'Unsupported language pair' in m['errors'][0]['error']
        assert out['models']['good']['errors'] == []

    def test_no_error_field_means_no_error_record(self):
        def fn(text, source_lang='english', target_lang='telugu'):
            return {'translated': 'వాతావరణం బాగుంది'}

        out = self._run(fn, fn)
        assert out['models']['bad']['errors'] == []


# ---------------------------------------------------------------------------
# Evaluator memory lifecycle (COMET/BERTScore unload between matrix languages)
# ---------------------------------------------------------------------------

class TestCOMETUnload:
    """COMETScorer.unload() must release the loaded model so the benchmark
    matrix can free ~1.4GB between languages."""

    def test_unload_drops_model_reference(self):
        from app.evaluation.metrics.comet import COMETScorer
        s = COMETScorer.__new__(COMETScorer)
        s._available = True
        s._model = object()
        s._model_name = 'fake-model'
        s._device = -1
        s.unload()
        assert s._model is None

    def test_unload_noop_when_not_loaded(self):
        from app.evaluation.metrics.comet import COMETScorer
        s = COMETScorer.__new__(COMETScorer)
        s._available = True
        s._model = None
        s._model_name = 'fake-model'
        s._device = -1
        s.unload()  # must not raise
        assert s._model is None

    def test_score_reloads_after_unload(self, monkeypatch):
        # score() must lazily reload after unload() rather than erroring.
        from app.evaluation.metrics.comet import COMETScorer
        s = COMETScorer.__new__(COMETScorer)
        s._available = True
        s._model = None
        s._model_name = 'fake-model'
        s._device = -1
        loads = []

        class _FakeModel:
            def predict(self, data, batch_size=None, gpus=None):
                return {'scores': [0.9, 0.8]}

        def _fake_load():
            loads.append(1)
            s._model = _FakeModel()
            return True

        monkeypatch.setattr(s, '_load_model', _fake_load)
        result = s.score(['hi', 'hello'], [['hi'], ['hello']], sources=['hi', 'hello'])
        assert loads == [1]
        assert result['num_sentences'] == 2


class TestBERTScoreUnload:
    """BERTScoreScorer.unload() is best-effort (gc.collect) but must be a
    safe no-op on an idle scorer."""

    def test_unload_noop(self):
        from app.evaluation.metrics.bertscore import BERTScoreScorer
        s = BERTScoreScorer.__new__(BERTScoreScorer)
        s._available = True
        s.unload()  # must not raise


class TestEvaluatorUnloadsNeuralModels:
    """Evaluator must release COMET/BERTScore models right after each scoring
    call so a --metrics all matrix never holds both neural models plus the MT
    models at once (the original OOM on this 16GB machine)."""

    def test_evaluate_unloads_neural_metrics_after_scoring(self):
        from app.evaluation.evaluator import Evaluator

        calls = []

        class _FakeNeural:
            available = True

            def __init__(self, name):
                self._name = name

            def score(self, hypotheses, references, **kwargs):
                return {'fake': 0.9, 'comet_score': 0.9, 'f1': 0.9}

            def unload(self):
                calls.append(self._name)

        ev = Evaluator(metrics=['bleu', 'comet', 'bertscore'])
        ev._scorers['comet'] = _FakeNeural('comet')
        ev._scorers['bertscore'] = _FakeNeural('bertscore')

        out = ev.evaluate(
            hypotheses=['నమస్తే'], references=[['నమస్తే']],
            sources=['hello'], model_name='m', lang_pair='en->te',
        )
        assert calls == ['comet', 'bertscore']  # unloaded in scoring order
        assert 'comet' in out['metrics'] and 'bertscore' in out['metrics']

    def test_unload_neural_metrics_targets_only_neural(self):
        from app.evaluation.evaluator import Evaluator

        calls = []

        class _FakeNeural:
            available = True

            def score(self, h, r, **kw):
                return {'f1': 0.9}

            def unload(self):
                calls.append('neural')

        ev = Evaluator(metrics=['bleu', 'comet', 'bertscore', 'quality'])
        ev._scorers['comet'] = _FakeNeural()
        ev._scorers['bertscore'] = _FakeNeural()
        ev.unload_neural_metrics()
        assert calls == ['neural', 'neural']

    def test_unload_neural_metrics_tolerates_scorers_without_unload(self):
        from app.evaluation.evaluator import Evaluator
        ev = Evaluator(metrics=['bleu', 'quality'])
        ev.unload_neural_metrics()  # no-op; bleu/quality have no unload()

    def test_unload_disabled_keeps_models_resident(self):
        from app.evaluation.evaluator import Evaluator

        calls = []

        class _FakeNeural:
            available = True

            def score(self, hypotheses, references, **kwargs):
                return {'comet_score': 0.9, 'f1': 0.9}

            def unload(self):
                calls.append('unloaded')

        ev = Evaluator(metrics=['comet', 'bertscore'],
                       unload_neural_after_each=False)
        ev._scorers['comet'] = _FakeNeural()
        ev._scorers['bertscore'] = _FakeNeural()
        ev.evaluate(hypotheses=['hi'], references=[['hi']], sources=['hi'])
        assert calls == []  # opt-out respected


class TestBenchmarkRunnerUnloads:
    """BenchmarkRunner must release neural metrics at the end of run() and
    forward the unload flag to the Evaluator."""

    def test_run_calls_unload_neural_metrics(self):
        from app.evaluation.benchmark import BenchmarkRunner
        runner = BenchmarkRunner(metrics=['bleu'])
        calls = []
        orig = runner.evaluator.unload_neural_metrics
        runner.evaluator.unload_neural_metrics = lambda: calls.append(1) or orig()

        def fn(text, source_lang='english', target_lang='hindi'):
            return {'translated': 'नमस्ते', 'error': None}

        runner.run(
            model_fns={'m': fn},
            sources=['Hello'],
            references={'hindi': ['नमस्ते']},
            source_lang='english', target_lang='hindi',
            run_latency=False,
        )
        assert calls == [1]

    def test_constructor_forwards_unload_flag(self):
        from app.evaluation.benchmark import BenchmarkRunner
        assert BenchmarkRunner(metrics=['bleu']).evaluator._unload_neural_after_each is True
        assert (BenchmarkRunner(metrics=['bleu'],
                                unload_neural_after_each=False)
                .evaluator._unload_neural_after_each is False)


class TestBenchmarkModelSupportFilter:
    """Quality-matrix model functions must exclude models whose
    supported_languages don't cover the source->target pair (e.g. m2m100 for
    telugu), so unsupported pairs are skipped instead of run-and-failed."""

    def test_unsupported_pairs_excluded(self):
        from scripts.run_benchmark import make_live_model_fns

        class _Stub:
            available = True

            def __init__(self, langs):
                self._langs = langs

            @property
            def supported_languages(self):
                return self._langs

        class _Registry:
            def __init__(self):
                self._models = {
                    'nllb': _Stub({'english', 'telugu', 'hindi'}),
                    'm2m100': _Stub({'english', 'hindi', 'tamil', 'kannada',
                                     'marathi', 'bengali', 'gujarati'}),
                }

            def get_model(self, name):
                return self._models.get(name)

        reg = _Registry()

        # en -> telugu: only nllb qualifies
        fns = make_live_model_fns('telugu', reg, model_names=['nllb', 'm2m100'])
        assert list(fns) == ['nllb']

        # telugu -> en (code-mixed): only nllb qualifies
        fns_cm = make_live_model_fns('english', reg, model_names=['nllb', 'm2m100'],
                                     source_lang='telugu')
        assert list(fns_cm) == ['nllb']

        # en -> hindi: both qualify
        fns_hi = make_live_model_fns('hindi', reg, model_names=['nllb', 'm2m100'])
        assert set(fns_hi) == {'nllb', 'm2m100'}


# ---------------------------------------------------------------------------
# Periodic checkpointing (partial results survive a crashed run)
# ---------------------------------------------------------------------------

class TestBenchmarkCheckpoint:
    """save_checkpoint/load_checkpoint must round-trip partial results so a
    run killed mid-matrix (e.g. OOM) can resume instead of losing everything.
    """

    def _args(self, tmp_path):
        import argparse
        return argparse.Namespace(output_dir=str(tmp_path), resume=False)

    def test_save_and_load_round_trip(self, tmp_path):
        from scripts.run_benchmark import (
            save_checkpoint, load_checkpoint, checkpoint_path,
        )
        args = self._args(tmp_path)
        results = {'metadata': {'mode': 'live'},
                   'translation': {'hindi': {'num_sentences': 10}}}
        completed = {'functional', 'security', 'translation.hindi'}
        save_checkpoint(results, completed, args, 'test')

        assert checkpoint_path(args).endswith('benchmark_checkpoint.json')
        assert not os.path.exists(f'{checkpoint_path(args)}.tmp')  # atomic

        loaded, loaded_completed = load_checkpoint(args)
        assert loaded == results
        assert loaded_completed == completed

    def test_load_missing_checkpoint_returns_none(self, tmp_path):
        from scripts.run_benchmark import load_checkpoint
        loaded, completed = load_checkpoint(self._args(tmp_path))
        assert loaded is None
        assert completed == set()

    def test_load_corrupt_checkpoint_returns_none(self, tmp_path):
        from scripts.run_benchmark import load_checkpoint
        path = tmp_path / 'benchmark_checkpoint.json'
        path.write_text('{ not valid json', encoding='utf-8')
        loaded, completed = load_checkpoint(self._args(tmp_path))
        assert loaded is None
        assert completed == set()

    def test_save_creates_output_dir(self, tmp_path):
        import argparse
        from scripts.run_benchmark import save_checkpoint, checkpoint_path
        args = argparse.Namespace(
            output_dir=str(tmp_path / 'nested' / 'reports'))
        save_checkpoint({'a': 1}, {'x'}, args, 'deep-dir')
        assert os.path.exists(checkpoint_path(args))

    def test_has_per_language_completion(self):
        from scripts.run_benchmark import has_per_language_completion
        assert has_per_language_completion({'translation.hindi', 'security'},
                                           'translation') is True
        assert has_per_language_completion({'translation', 'security'},
                                           'translation') is False
        assert has_per_language_completion(set(), 'translation') is False
        assert has_per_language_completion({'flores_translation.telugu'},
                                           'translation') is False

    def test_load_rejects_unknown_checkpoint_version(self, tmp_path):
        import json
        from scripts.run_benchmark import load_checkpoint
        path = tmp_path / 'benchmark_checkpoint.json'
        path.write_text(json.dumps({'checkpoint_version': 99, 'results': {}}),
                        encoding='utf-8')
        loaded, completed = load_checkpoint(self._args(tmp_path))
        assert loaded is None
        assert completed == set()

    def test_load_warns_on_flag_mismatch(self, tmp_path, monkeypatch, caplog):
        import json
        import logging
        import scripts.run_benchmark as rb
        path = tmp_path / 'benchmark_checkpoint.json'
        path.write_text(json.dumps({
            'checkpoint_version': 1,
            'command': '--mock --metrics all',
            'completed': ['functional'],
            'results': {'functional': {'pass_rate': 100.0}},
        }), encoding='utf-8')
        monkeypatch.setattr(rb, '_normalized_command', lambda: '--mock --metrics bleu')
        with caplog.at_level(logging.WARNING, logger='run_benchmark'):
            loaded, completed = rb.load_checkpoint(self._args(tmp_path))
        assert loaded is not None
        assert 'differ from the checkpoint run' in caplog.text
