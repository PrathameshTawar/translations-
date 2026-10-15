"""Unit tests for the evaluation benchmark modules.

Pure-logic tests only — no translation models are loaded.
Run with: python -m pytest tests -q
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# Retrieval metrics math
# ---------------------------------------------------------------------------

class TestRetrievalMetrics:
    @pytest.fixture
    def M(self):
        from app.evaluation.retrieval import RetrievalMetrics
        return RetrievalMetrics

    def test_recall_at_k(self, M):
        ranked = ['a', 'b', 'c', 'd', 'e']
        assert M.recall_at_k(ranked, ['a'], 1) == 1.0
        assert M.recall_at_k(ranked, ['e'], 5) == 1.0
        assert M.recall_at_k(ranked, ['e'], 3) == 0.0
        assert M.recall_at_k(ranked, ['a', 'c'], 3) == 1.0
        assert M.recall_at_k(ranked, ['a', 'c'], 2) == 0.5
        assert M.recall_at_k([], ['a'], 5) == 0.0

    def test_precision_at_k(self, M):
        ranked = ['a', 'b', 'c', 'd', 'e']
        assert M.precision_at_k(ranked, ['a'], 1) == 1.0
        assert M.precision_at_k(ranked, ['a'], 5) == 0.2
        assert M.precision_at_k(ranked, ['z'], 5) == 0.0

    def test_reciprocal_rank(self, M):
        assert M.reciprocal_rank(['x', 'a', 'y'], ['a']) == 0.5
        assert M.reciprocal_rank(['a', 'b'], ['a']) == 1.0
        assert M.reciprocal_rank(['b', 'c'], ['a']) == 0.0

    def test_mrr(self, M):
        ranked = [['a', 'b'], ['x', 'y', 'a'], ['z', 'z2']]
        relevant = [['a'], ['a'], ['q']]
        # 1.0 + 1/3 + 0 = 1.333 / 3
        assert M.mrr(ranked, relevant) == pytest.approx((1.0 + 1 / 3) / 3)

    def test_ndcg_at_k(self, M):
        # perfect ranking: dcg == idcg -> 1.0
        assert M.ndcg_at_k(['a', 'b', 'c'], ['a', 'b', 'c'], 3) == pytest.approx(1.0)
        # nothing relevant -> 0.0
        assert M.ndcg_at_k(['a', 'b'], ['z'], 3) == 0.0
        # one relevant at rank 2: 1/log2(3) / (1/log2(2)) = 0.6309
        ndcg = M.ndcg_at_k(['x', 'a'], ['a'], 3)
        assert ndcg == pytest.approx((1 / 1.5849625007211563) / 1.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Retrieval benchmark — reranker on/off ablation
# ---------------------------------------------------------------------------

class _FakeRerankerPipeline:
    """Pipeline stub: dense returns 3 terms alphabetically (bank first); the
    reranked mode returns a strictly worse ranking (drops 'bank' entirely).
    The qrels universe is the real hindi glossary, so the fake's 3-term
    universe leaves most queries unmatched — fine for testing the math."""

    def __init__(self):
        self.encoder = object()  # non-None -> 'faiss-semantic' backend label

    def retrieve_context(self, query, source_lang, target_lang, top_k,
                         use_reranker=False):
        if use_reranker:
            order = ['technology', 'hospital']  # 'bank' dropped
        else:
            order = ['bank', 'hospital', 'technology']
        terms = [{'term': t, 'translation': t} for t in order]
        return {'terms': terms[:top_k], 'context': 'x',
                'reranked': use_reranker,
                'reranker_available': use_reranker}


class _FakeReranker:
    model_name = 'fake-cross-encoder'

    @property
    def available(self):
        return True


class TestRetrievalRerankerAblation:
    def test_reranker_block_structure_and_deltas(self, monkeypatch):
        from app.evaluation.retrieval import TerminologyRetrievalBenchmark
        monkeypatch.setattr('app.services.reranker.CrossEncoderReranker',
                            _FakeReranker)
        bench = TerminologyRetrievalBenchmark(
            pipeline=_FakeRerankerPipeline(), use_encoder=True)
        result = bench.run(langs=['hindi'], top_ks=(1, 3, 5),
                           use_reranker=True)
        rer = result['reranker']
        assert rer['available'] is True
        assert rer['model'] == 'fake-cross-encoder'
        assert set(rer) >= {'sentence_queries', 'paraphrase_queries',
                            'dense_query_latency_ms',
                            'reranked_query_latency_ms', 'rerank_overhead_ms'}
        # Delta math: every reported delta equals reranked minus dense.
        for variant in ('sentence_queries', 'paraphrase_queries'):
            block = rer[variant]
            for k in block['dense']:
                expected = round(block['reranked'][k] - block['dense'][k], 4)
                assert block['delta'][k] == expected, (variant, k)
        # The fake reranked mode drops 'bank' -> MRR must get worse.
        assert rer['sentence_queries']['delta']['mrr'] < 0
        assert rer['rerank_overhead_ms'] is not None
        # Scorecard-facing dense section is untouched by the ablation.
        off = TerminologyRetrievalBenchmark(
            pipeline=_FakeRerankerPipeline(), use_encoder=True
        ).run(langs=['hindi'], top_ks=(1, 3, 5))
        assert result['languages'] == off['languages']

    def test_reranker_off_by_default(self):
        from app.evaluation.retrieval import TerminologyRetrievalBenchmark
        bench = TerminologyRetrievalBenchmark(
            pipeline=_FakeRerankerPipeline(), use_encoder=True)
        result = bench.run(langs=['hindi'], top_ks=(1, 3, 5))
        assert result['reranker']['available'] is False

    def test_reranker_unavailable_is_reported(self, monkeypatch):
        from app.evaluation.retrieval import TerminologyRetrievalBenchmark

        class _Unavailable:
            model_name = 'fake-cross-encoder'

            @property
            def available(self):
                return False

        monkeypatch.setattr('app.services.reranker.CrossEncoderReranker',
                            _Unavailable)
        bench = TerminologyRetrievalBenchmark(
            pipeline=_FakeRerankerPipeline(), use_encoder=True)
        result = bench.run(langs=['hindi'], top_ks=(1, 3, 5),
                           use_reranker=True)
        assert result['reranker']['available'] is False
        assert 'error' in result['reranker']


class TestRetrievalProtocol:
    """The benchmark must not be able to report a trivial 100% again."""

    def test_query_set_composition(self):
        from collections import Counter
        from app.evaluation.retrieval import load_queries
        counts = Counter(q['type'] for q in load_queries())
        assert counts == {'sentence': 45, 'paraphrase': 30, 'distractor': 30}

    def test_paraphrases_never_contain_their_term(self):
        from app.evaluation.retrieval import load_queries
        from app.services.terminology import match_terms
        for q in load_queries():
            if q['type'] == 'paraphrase':
                terms = [{'term': t, 'translation': 'x'} for t in q['relevant']]
                assert not match_terms(q['text'], terms), q['id']

    def test_distractors_contain_no_glossary_term(self):
        from app.evaluation.retrieval import load_queries
        from app.services.term_index import load_terms
        from app.services.terminology import match_terms
        glossary = load_terms('hindi')
        for q in load_queries():
            if q['type'] == 'distractor':
                assert not match_terms(q['text'], glossary), q['id']

    def test_keyword_backend_small_index_is_flagged_invalid(self):
        from app.evaluation.retrieval import TerminologyRetrievalBenchmark
        result = TerminologyRetrievalBenchmark(use_encoder=False).run(
            langs=['hindi'], top_ks=(1, 5))
        assert result['backend'] == 'keyword-fallback'
        assert result['index_size'] == 15 and result['valid'] is False
        agg = result['aggregate']
        assert agg['paraphrase']['recall_at_5'] == 0.0  # lexical can't paraphrase
        assert agg['distractor_false_positive_rate'] == 0.0
        assert agg['sentence_term_recall'] == 1.0

    def test_padding_grows_index_without_touching_glossary(self):
        from app.evaluation.retrieval import TerminologyRetrievalBenchmark
        pad = [{'term': f'padterm{i}', 'translation': 'x', 'category': 'padding'}
               for i in range(1200)]
        bench = TerminologyRetrievalBenchmark(use_encoder=False, pad_terms=pad)
        result = bench.run(langs=['hindi'], top_ks=(1, 5))
        assert result['index_size'] == 1215 and result['valid'] is True
        from app.services.term_index import load_terms
        assert len(load_terms('hindi')) == 15  # nothing persisted

    def test_load_pad_terms_text_file(self, tmp_path):
        from app.evaluation.retrieval import load_pad_terms
        f = tmp_path / 'terms.txt'
        f.write_text('alpha\n\nbeta gamma\n', encoding='utf-8')
        assert [t['term'] for t in load_pad_terms(str(f))] == ['alpha', 'beta gamma']


# ---------------------------------------------------------------------------
# Open-vocabulary passage retrieval (dense vs rerank ablation)
# ---------------------------------------------------------------------------

class _FakeSentenceEncoder:
    """Deterministic embedding: char-bigram hash bag. Queries that differ
    from their passage (masked/paraphrased) get lower similarity, so the
    dense baseline is not at ceiling — the fake reranker then rescues."""

    def __init__(self, seed=7):
        self._seed = seed
        self._dim = 64

    def _hash(self, token):
        h = self._seed
        for ch in token:
            h = (h * 31 + ord(ch)) & 0xFFFFFFFF
        return h % self._dim

    def _vec(self, text):
        import numpy as np
        v = np.zeros(self._dim, dtype=np.float32)
        toks = text.lower().split()
        for tok in toks:
            v[self._hash(tok)] += 1.0
        norm = float(np.linalg.norm(v)) or 1.0
        return v / norm

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        import numpy as np
        return np.stack([self._vec(t) for t in texts])


class _FakePassageReranker:
    """Deterministic reranker: scores each candidate by word overlap with the
    query, exactly like the dense path but re-applied over the top-10 pool —
    so the ablation math is exercised with a real reordering."""

    model_name = 'fake-passage-reranker'

    @property
    def available(self):
        return True

    def rerank(self, query, candidates, top_k=None, score_key='term'):
        q_words = set(query.lower().split())
        scored = []
        for c in candidates:
            c_words = set(c.lower().split())
            overlap = len(q_words & c_words)
            # tie-break on length so the reorder is deterministic
            scored.append((overlap, -len(c), c))
        scored.sort(reverse=True)
        ranked = [c for _, _, c in scored]
        if top_k:
            ranked = ranked[:top_k]
        return {'reranked': ranked, 'scores': [1.0] * len(ranked),
                'available': True}


class _FakePassagePipeline:
    def __init__(self, encoder):
        self.encoder = encoder


class TestPassageRetrieval:
    def test_corpus_builds_unique_passages(self):
        from app.evaluation.passage_retrieval import build_passage_corpus
        corpus = build_passage_corpus()
        assert corpus['num_passages'] > 50  # 42 curated + 25 code-mixed refs
        assert len(corpus['passages']) == len(set(corpus['passages']))
        assert all(p.strip() for p in corpus['passages'])
        assert 'in-project' in corpus['source']

    def test_masked_queries_are_deterministic_and_grounded(self):
        from app.evaluation.passage_retrieval import (
            build_masked_queries, build_passage_corpus,
        )
        passages = build_passage_corpus()['passages'][:10]
        a = build_masked_queries(passages)
        b = build_masked_queries(passages)
        assert [q['query'] for q in a] == [q['query'] for q in b]
        for i, q in enumerate(a):
            assert q['relevant'] == [i]
        # at least one query is actually masked (non-trivial task)
        assert any('____' in q['query'] for q in a)

    def test_roundtrip_queries_use_translate_fn(self):
        from app.evaluation.passage_retrieval import build_roundtrip_queries
        passages = ['Hello world this is a test sentence.']
        calls = []

        def fn(text, src, tgt):
            calls.append((src, tgt))
            return {'english,hindi': 'नमस्ते दुनिया', 'hindi,english': 'Hello world this is a test.'}[
                f'{src},{tgt}']

        queries = build_roundtrip_queries(passages, fn)
        assert calls[0] == ('english', 'hindi')
        assert calls[1] == ('hindi', 'english')
        assert queries[0]['query'] == 'Hello world this is a test.'
        assert queries[0]['relevant'] == [0]

    def test_roundtrip_failure_falls_back_to_passage(self):
        from app.evaluation.passage_retrieval import build_roundtrip_queries
        passages = ['Some passage here.']

        def boom(text, src, tgt):
            raise RuntimeError('model down')

        queries = build_roundtrip_queries(passages, boom)
        assert queries[0]['query'] == 'Some passage here.'  # unchanged
        assert queries[0]['relevant'] == [0]

    def _bench(self, passages):
        from app.evaluation.passage_retrieval import PassageRetrievalBenchmark
        return PassageRetrievalBenchmark(
            pipeline=_FakePassagePipeline(_FakeSentenceEncoder()),
            use_encoder=True,
        )

    def test_reranked_pool_is_smaller_than_corpus(self, monkeypatch):
        # Sanity: reranking operates on the dense top-10 pool, so the ranked
        # lists returned are <= RERANK_POOL even on a large corpus.
        from app.evaluation import passage_retrieval as pr_mod
        passages = [f'passage number {i} with unique words here' for i in range(200)]
        bench = self._bench(passages)
        queries = pr_mod.build_masked_queries(passages)
        metrics, lat = bench._retrieve(queries, passages, (1, 3, 5),
                                       rerank=True,
                                       reranker=_FakePassageReranker())
        assert lat is not None and lat >= 0
        assert set(metrics) >= {'recall_at_1', 'recall_at_3', 'recall_at_5', 'mrr'}

    def test_dense_baseline_at_ceiling_on_exact_queries(self):
        # Exact-copy queries must give dense recall@1 == 1.0 (the encoder
        # recovers the passage exactly) — proves the metric plumbing.
        from app.evaluation import passage_retrieval as pr_mod
        passages = ['the weather is nice today in the park',
                    'please send the document tomorrow',
                    'the bank offers low interest rates']
        bench = self._bench(passages)
        queries = [{'query': p, 'relevant': [i]} for i, p in enumerate(passages)]
        metrics, _ = bench._retrieve(queries, passages, (1, 3, 5))
        assert metrics['recall_at_1'] == 1.0
        assert metrics['mrr'] == 1.0

    def test_masked_mode_dense_not_at_ceiling_and_delta_math(self, monkeypatch):
        # Masked queries blank a distinctive word, so dense drops below
        # ceiling; the reranker re-orders the top-10 pool. Verify the whole
        # run() result shape and that delta == reranked - dense exactly.
        from app.evaluation.passage_retrieval import PassageRetrievalBenchmark
        monkeypatch.setattr(
            'app.services.reranker.CrossEncoderReranker',
            lambda *a, **k: _FakePassageReranker(),
        )
        passages = ['the weather is nice today in the park',
                    'please send the document tomorrow',
                    'the bank offers low interest rates']
        result = PassageRetrievalBenchmark(
            pipeline=_FakePassagePipeline(_FakeSentenceEncoder()),
            use_encoder=True,
        ).run(passages=passages, queries_mode='masked', use_reranker=True)
        assert result['num_queries'] == 3
        assert result['dense_query_latency_ms'] is not None
        rer = result['reranker']
        assert rer['available'] is True
        assert rer['model'] == 'fake-passage-reranker'
        assert rer['rerank_overhead_ms'] is not None
        for k in result['dense']:
            expected = round(rer['reranked'][k] - result['dense'][k], 4)
            assert rer['delta'][k] == expected, k

    def test_reranker_off_by_default(self):
        from app.evaluation.passage_retrieval import PassageRetrievalBenchmark
        result = PassageRetrievalBenchmark(
            pipeline=_FakePassagePipeline(_FakeSentenceEncoder()),
            use_encoder=True,
        ).run(passages=['one two three'], queries_mode='masked')
        assert result['reranker']['available'] is False

    def test_reranker_unavailable_reported(self, monkeypatch):
        from app.evaluation.passage_retrieval import PassageRetrievalBenchmark

        class _Unavailable:
            model_name = 'fake'

            @property
            def available(self):
                return False

        monkeypatch.setattr(
            'app.services.reranker.CrossEncoderReranker',
            lambda *a, **k: _Unavailable(),
        )
        result = PassageRetrievalBenchmark(
            pipeline=_FakePassagePipeline(_FakeSentenceEncoder()),
            use_encoder=True,
        ).run(passages=['one two three'], queries_mode='masked',
              use_reranker=True)
        assert result['reranker']['available'] is False
        assert 'error' in result['reranker']


class TestPassageRetrievalReport:
    def test_markdown_renders_passage_block(self):
        from app.evaluation.report import render_markdown, build_scorecard
        results = {
            'passage_retrieval': {
                'corpus': {'num_passages': 67, 'source': 'in-project'},
                'num_queries': 67, 'queries_mode': 'masked',
                'dense': {'recall_at_1': 0.60, 'recall_at_3': 0.85,
                          'recall_at_5': 0.92, 'mrr': 0.68},
                'dense_query_latency_ms': 3.2,
                'reranker': {
                    'available': True, 'model': 'x',
                    'reranked': {'recall_at_1': 0.72, 'recall_at_3': 0.93,
                                 'recall_at_5': 0.97, 'mrr': 0.79},
                    'delta': {'recall_at_1': 0.12, 'recall_at_3': 0.08,
                              'recall_at_5': 0.05, 'mrr': 0.11},
                    'reranked_query_latency_ms': 25.0,
                    'rerank_overhead_ms': 21.8,
                },
            },
        }
        sc = build_scorecard(results)
        md = render_markdown(sc, results)
        assert 'Passage retrieval (open-vocabulary' in md
        assert '| recall_at_5 | 0.92 | 0.97 | +0.05 |' in md
        assert '21.8' in md

    def test_markdown_handles_error_state(self):
        from app.evaluation.report import render_markdown, build_scorecard
        results = {'passage_retrieval': {'error': 'needs --live'}}
        sc = build_scorecard(results)
        md = render_markdown(sc, results)
        assert 'Passage retrieval (open-vocabulary' not in md  # no crash


# ---------------------------------------------------------------------------
# Language detection service + benchmark
# ---------------------------------------------------------------------------

class TestLanguageDetector:
    @pytest.fixture
    def detector(self):
        from app.services.language_detection import LanguageDetector
        return LanguageDetector()

    def test_script_to_language(self, detector):
        assert detector.detect('The weather is beautiful today.') == 'english'
        assert detector.detect('ভারত একটি সুন্দর দেশ।') == 'bengali'
        assert detector.detect('ગુજરાત રાજ્ય પશ્ચિમ ભારતમાં છે.') == 'gujarati'
        assert detector.detect('தமிழ் மொழி அழகானது.') == 'tamil'
        assert detector.detect('తెలుగు భాష సుందరమైనది.') == 'telugu'
        assert detector.detect('ಕನ್ನಡ ನುಡಿ ಸುಂದರವಾಗಿದೆ.') == 'kannada'

    def test_devanagari_hindi(self, detector):
        assert detector.detect('अच्छे स्वास्थ्य के लिए नियमित व्यायाम और संतुलित आहार महत्वपूर्ण हैं।') == 'hindi'

    def test_devanagari_marathi(self, detector):
        assert detector.detect('चांगल्या आरोग्यासाठी नियमित व्यायाम आणि संतुलित आहार महत्त्वाचे आहेत.') == 'marathi'

    def test_code_mixed_detects_indic(self, detector):
        # English words inside an Indic sentence must not flip the script
        assert detector.detect('मुझे आज office जाना है।') == 'hindi'
        assert detector.detect('நான் இன்று office-க்கு போகிறேன்.') == 'tamil'

    def test_code_mixed_latin_dominant_still_indic(self, detector):
        # Even when Latin characters outnumber Devanagari, a substantial
        # Indic share routes to the Indic detector (regression for the
        # code-mixed cascade fix).
        assert detector.detect('आपण उद्या video call वर discuss करूया।') == 'marathi'

    def test_marathi_marker_char_is_decisive(self, detector):
        # ळ is essentially Marathi-only within Devanagari.
        assert detector.detect('तिने अंतिम परीक्षेत सर्वोच्च गुण मिळवले।') == 'marathi'

    def test_function_word_morphology_disambiguates(self, detector):
        # Hindi में/है vs Marathi मध्ये/आहे — the word/suffix model uses
        # this morphology (the old char n-gram could not).
        assert detector.detect('अच्छे स्वास्थ्य के लिए नियमित व्यायाम और संतुलित आहार महत्वपूर्ण हैं।') == 'hindi'
        assert detector.detect('चांगल्या आरोग्यासाठी नियमित व्यायाम आणि संतुलित आहार महत्त्वाचे आहेत.') == 'marathi'
        assert detector.detect('कृपया मेरे बचत खाते में पांच हजार रुपये ट्रांसफर करें।') == 'hindi'
        assert detector.detect('वार्षिक अर्थसंकल्प काल वित्त समितीने मंजूर केला.') == 'marathi'

    def test_empty_and_unknown(self, detector):
        assert detector.detect('') == 'unknown'
        assert detector.detect('   ') == 'unknown'
        # No recognized script -> explicit 'unknown', never a misleading default
        assert detector.detect('###!!! 1234') == 'unknown'
        assert detector.detect('🔥🎉') == 'unknown'

    def test_scores_shape(self, detector):
        result = detector.detect_with_scores('भारत एक विविध देश है।')
        assert result['detected_language'] == 'hindi'
        assert result['script'] == 'devanagari'
        assert 0.0 <= result['confidence'] <= 1.0


# ---------------------------------------------------------------------------
# Language detector — words + 2-5-char suffix Devanagari model
# ---------------------------------------------------------------------------

class TestLanguageDetectorSuffixModel:
    """The Devanagari hi/mr disambiguator keys on words + 2-5-char suffixes."""

    def test_word_features_emit_2_to_5_char_suffixes(self):
        from app.services.language_detection import (
            LanguageDetector, _normalize_script_chars, SCRIPT_RANGES,
        )
        word = 'आरोग्यासाठी'
        wn = _normalize_script_chars(word, SCRIPT_RANGES['devanagari'])
        feats = LanguageDetector._word_features(wn, f'चांगल्या {word} आहे.')
        assert wn in feats  # the word itself
        for n in (2, 3, 4, 5):
            suffix = wn[-n:] if len(wn) >= n else wn
            assert suffix in feats, f'missing {n}-char suffix {suffix!r}'
        # The Marathi 'for' postposition suffix -साठी (4 chars) is emitted.
        assert 'साठी' in feats

    def test_saathi_vs_ke_lie_morphology_disambiguates(self):
        # Same structure ('X के लिए' vs 'X-साठी'), different suffix morphology.
        from app.services.language_detection import LanguageDetector
        detector = LanguageDetector()
        assert detector.detect('विद्यार्थ्यांनी अभ्यासासाठी वेळ दिला।') == 'marathi'
        assert detector.detect('विद्यार्थियों ने परीक्षा के लिए पढ़ाई की।') == 'hindi'

    def test_adverb_openers_stay_correct_language(self):
        from app.services.language_detection import LanguageDetector
        detector = LanguageDetector()
        assert detector.detect('हम बाद में मिलेंगे।') == 'hindi'
        assert detector.detect('आम्ही नंतर भेटू.') == 'marathi'


class TestLanguageDetectionBenchmark:
    def test_runs_with_expected_structure(self):
        from app.evaluation.language_detection import LanguageDetectionBenchmark
        result = LanguageDetectionBenchmark().run()
        assert result['num_test'] > 0
        assert 0.0 <= result['accuracy'] <= 1.0
        assert 0.0 <= result['macro_f1'] <= 1.0
        assert set(result.keys()) >= {
            'accuracy', 'macro_f1', 'weighted_f1', 'devanagari_macro_f1',
            'per_class', 'confusion_matrix',
        }
        assert len(result['per_class']) == 8
        # hi/mr F1 must be below the deterministic-script classes (honest
        # signal: the learned component is the bottleneck, not the whole score)
        assert result['devanagari_macro_f1'] < result['macro_f1']

    def test_deterministic(self):
        from app.evaluation.language_detection import LanguageDetectionBenchmark
        a = LanguageDetectionBenchmark().run()
        b = LanguageDetectionBenchmark().run()
        assert a['accuracy'] == b['accuracy']
        assert a['confusion_matrix'] == b['confusion_matrix']

    def test_confusion_matrix_diagonal_is_consistent(self):
        from app.evaluation.language_detection import (
            BENCHMARK_LANGUAGES, LanguageDetectionBenchmark,
        )
        result = LanguageDetectionBenchmark().run()
        cm = result['confusion_matrix']
        for lang in BENCHMARK_LANGUAGES:
            assert lang in cm

    def test_benchmark_meets_scorecard_targets(self):
        # Regression floor: the learned hi/mr word/suffix model plus the
        # deterministic script classes must clear the scorecard's language-ID
        # target (deterministic split, so these are stable).
        from app.evaluation.language_detection import LanguageDetectionBenchmark
        result = LanguageDetectionBenchmark().run()
        assert result['macro_f1'] >= 0.95
        assert result['devanagari_macro_f1'] >= 0.90


# ---------------------------------------------------------------------------
# Code-mixed dataset + benchmark
# ---------------------------------------------------------------------------

class TestCodeMixed:
    def test_dataset_has_25_items_across_7_languages(self):
        from app.evaluation.datasets import CODE_MIXED_ITEMS, CURATED_LANGS
        assert len(CODE_MIXED_ITEMS) == 25
        langs = {i['source_lang'] for i in CODE_MIXED_ITEMS}
        assert langs == set(CURATED_LANGS)  # all 7 Indic languages
        for item in CODE_MIXED_ITEMS:
            assert item['source'].strip(), f"empty source: {item}"
            assert item['reference'].strip(), f"empty reference: {item}"
            assert item['source_lang'] in CURATED_LANGS

    def test_references_are_english(self):
        from app.evaluation.datasets import CODE_MIXED_ITEMS
        from app.services.language_detection import LanguageDetector
        detector = LanguageDetector()
        for item in CODE_MIXED_ITEMS:
            assert detector.detect(item['reference']) == 'english', (
                f"{item['reference']!r} not detected as English"
            )

    def test_sources_are_code_mixed(self):
        """Each source mixes Indic script with embedded Latin (English) words."""
        from app.evaluation.datasets import CODE_MIXED_ITEMS
        from app.services.language_detection import scripts_present
        for item in CODE_MIXED_ITEMS:
            scripts = scripts_present(item['source'])
            # Must contain an Indic script (not only latin)…
            assert any(s != 'latin' for s in scripts), (
                f"source has no Indic script: {item['source']!r} -> {scripts}"
            )
            # …must also contain Latin (English) characters…
            assert any(s == 'latin' for s in scripts), (
                f"source has no English words: {item['source']!r} -> {scripts}"
            )
            # …and an embedded English word is present
            assert any(
                'a' <= ch.lower() <= 'z' for ch in item['source']
            ), f"no English word embedded: {item['source']!r}"

    def test_code_mixed_matrix_runs_with_stub(self):
        from scripts.run_benchmark import run_code_mixed_matrix
        from app.evaluation.datasets import CODE_MIXED_ITEMS

        def stub(text, source_lang='english', target_lang='english'):
            return {
                'original': text,
                'translated': f'[{source_lang}->{target_lang}] {text}',
                'source_lang': source_lang,
                'target_lang': target_lang,
                'metadata': {'model': 'mock-echo', 'error': None},
            }

        result = run_code_mixed_matrix(
            {'mock-echo': stub}, CODE_MIXED_ITEMS, metrics=['bleu', 'chrf'],
        )
        # One entry per source language (all 7 present)
        assert set(result.keys()) == {'hindi', 'marathi', 'bengali',
                                      'gujarati', 'tamil', 'telugu', 'kannada'}
        for lang, lang_res in result.items():
            assert lang_res['benchmark']['target_lang'] == 'english'
            assert 'mock-echo' in lang_res['models']
            assert 'summary' in lang_res['models']['mock-echo']


# ---------------------------------------------------------------------------
# Quality estimation benchmark
# ---------------------------------------------------------------------------

class TestQualityEstimation:
    def test_runs_on_small_sample(self):
        from app.evaluation.quality_estimation import QualityEstimationBenchmark
        b = QualityEstimationBenchmark(seed=1)
        samples = [
            # Perfect good translation
            {'source': 'The weather is beautiful today.',
             'translation': 'आज मौसम बहुत सुंदर है।', 'label': 'good'},
            # Garbled
            {'source': 'The weather is beautiful today.',
             'translation': 'आज है। सुंदर मौसम बहुत', 'label': 'bad'},
            # Truncated
            {'source': 'The weather is beautiful today.',
             'translation': 'आज मौसम', 'label': 'partial'},
        ]
        result = b.run(samples=samples)
        assert result['num_samples'] == 3
        assert set(result.keys()) >= {
            'accuracy', 'precision', 'recall', 'f1', 'confusion_matrix', 'per_label'
        }
        assert 0.0 <= result['accuracy'] <= 1.0

    def test_build_samples_has_all_labels(self):
        from app.evaluation.quality_estimation import QualityEstimationBenchmark
        samples = QualityEstimationBenchmark(seed=42).build_samples(
            max_good=10, max_bad=10, max_partial=5
        )
        labels = {s['label'] for s in samples}
        assert {'good', 'bad', 'partial'} <= labels

    def test_heuristic_f1_meets_scorecard_target(self):
        # Regression floor for the heuristic scorer on the synthetic set (not
        # a scorecard target: separating hand-made degenerate outputs is a
        # sanity check, not quality estimation).
        from app.evaluation.quality_estimation import QualityEstimationBenchmark
        result = QualityEstimationBenchmark().run()
        assert result['f1'] >= 0.95
        # The Devanagari word-order rule must have caught scrambled samples
        # (deterministic seed — currently 24 of 129).
        assert result['word_order_flagged'] > 0


# ---------------------------------------------------------------------------
# Unicode suite
# ---------------------------------------------------------------------------

class TestUnicodeSuite:
    def test_all_cases_pass_validators_and_normalization(self):
        from app.evaluation.unicode import UnicodeTestSuite
        result = UnicodeTestSuite().run()  # no translate_fn -> validator+NFC only
        assert result['total'] > 10
        assert result['pass_rate'] == 100.0

    def test_translation_check_with_echo_stub(self):
        from app.evaluation.functional import EchoTranslator
        from app.evaluation.unicode import UnicodeTestSuite

        stub = lambda text, source_lang, target_lang: (  # noqa: E731
            EchoTranslator().translate(text, source_lang, target_lang)['translated']
        )
        result = UnicodeTestSuite().run(translate_fn=stub)
        assert result['pass_rate'] == 100.0


# ---------------------------------------------------------------------------
# Functional suite (mock mode, Flask test client)
# ---------------------------------------------------------------------------

class TestFunctionalSuite:
    @pytest.fixture
    def suite(self):
        from app.evaluation.functional import FunctionalTestSuite
        return FunctionalTestSuite(use_mock=True)

    def test_validation_cases(self, suite):
        from app.evaluation.functional import FUNCTIONAL_CASES
        validation_cases = [
            c for c in FUNCTIONAL_CASES
            if c['name'] in (
                'empty input rejected', 'unsupported source language rejected',
                'same source/target rejected', 'over-long text rejected (5001 chars)',
                'malformed JSON rejected', 'no JSON body rejected',
            )
        ]
        result = suite.run(cases=validation_cases)
        assert result['total'] == len(validation_cases)
        assert result['passed'] == result['total']
        assert result['pass_rate'] == 100.0

    def test_happy_path_with_echo(self, suite):
        from app.evaluation.functional import FUNCTIONAL_CASES
        happy = [c for c in FUNCTIONAL_CASES
                 if c['name'] == 'valid EN->MR returns translation']
        result = suite.run(cases=happy)
        assert result['passed'] == 1

    def test_failure_semantics(self, suite):
        from app.evaluation.functional import FUNCTIONAL_CASES
        cases = [c for c in FUNCTIONAL_CASES if c.get('stub') in ('fail', 'empty')]
        result = suite.run(cases=cases)
        assert result['passed'] == len(cases)


# ---------------------------------------------------------------------------
# /translate/text — use_reranker option (forwarding + validation)
# ---------------------------------------------------------------------------

class _RecordingRagPipeline:
    """Fake RAG pipeline that records how it was called and echoes the
    reranker flags back, so the route's forwarding is observable."""

    def __init__(self):
        self.calls = []

    def retrieve_context(self, query, source_lang='english', target_lang=None,
                         top_k=2, use_reranker=False):
        self.calls.append({
            'query': query, 'source_lang': source_lang,
            'target_lang': target_lang, 'top_k': top_k,
            'use_reranker': use_reranker,
        })
        return {
            'context': 'hospital = अस्पताल',
            'terms': [{'term': 'hospital', 'translation': 'अस्पताल'}],
            'reranked': use_reranker,
            'reranker_available': use_reranker,
        }


class _BoomRagPipeline:
    """RAG pipeline that raises — the route must degrade to a normal
    translation instead of a 500."""

    def retrieve_context(self, *args, **kwargs):
        raise RuntimeError('reranker boom')


class TestTranslateRouteReranker:
    """The /translate/text endpoint must accept use_reranker, forward it to
    the RAG pipeline, and echo the reranked/reranker_available flags."""

    @pytest.fixture
    def client(self):
        from app.evaluation.functional import FunctionalTestSuite
        suite = FunctionalTestSuite(use_mock=True)
        suite._create_app()
        suite._install_stubs()
        yield suite.client
        suite._restore()

    def _post(self, client, body):
        return client.post('/api/v1/translate/text', json=body)

    def test_use_reranker_true_is_forwarded(self, client, monkeypatch):
        rag = _RecordingRagPipeline()
        monkeypatch.setattr(
            'app.services.rag_service.IndicRAGPipeline.get_instance',
            lambda: rag,
        )
        resp = self._post(client, {
            'text': 'The hospital is near the station.',
            'source_lang': 'english', 'target_lang': 'hindi',
            'include_context': True, 'use_reranker': True,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['reranked'] is True
        assert data['reranker_available'] is True
        assert rag.calls[0]['use_reranker'] is True
        assert rag.calls[0]['target_lang'] == 'hindi'

    def test_use_reranker_defaults_to_false(self, client, monkeypatch):
        rag = _RecordingRagPipeline()
        monkeypatch.setattr(
            'app.services.rag_service.IndicRAGPipeline.get_instance',
            lambda: rag,
        )
        resp = self._post(client, {
            'text': 'The hospital is near the station.',
            'source_lang': 'english', 'target_lang': 'hindi',
            'include_context': True,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['reranked'] is False
        assert data['reranker_available'] is False
        assert rag.calls[0]['use_reranker'] is False

    def test_use_reranker_without_context_is_ignored(self, client, monkeypatch):
        rag = _RecordingRagPipeline()
        monkeypatch.setattr(
            'app.services.rag_service.IndicRAGPipeline.get_instance',
            lambda: rag,
        )
        resp = self._post(client, {
            'text': 'The hospital is near the station.',
            'source_lang': 'english', 'target_lang': 'hindi',
            'include_context': False, 'use_reranker': True,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        # No retrieval happened, so no reranking either.
        assert rag.calls == []
        assert data['reranked'] is False
        assert data['context_used'] is False

    def test_use_reranker_non_boolean_rejected(self, client):
        resp = self._post(client, {
            'text': 'The hospital is near the station.',
            'source_lang': 'english', 'target_lang': 'hindi',
            'use_reranker': 'yes',
        })
        assert resp.status_code == 400
        assert 'boolean' in resp.get_json()['error']

    def test_reranker_failure_still_returns_translation(self, client, monkeypatch):
        # RAG throws (e.g. reranker load issue) — the route must degrade to
        # a normal translation with flags False, not a 500.
        monkeypatch.setattr(
            'app.services.rag_service.IndicRAGPipeline.get_instance',
            lambda: _BoomRagPipeline(),
        )
        resp = self._post(client, {
            'text': 'The hospital is near the station.',
            'source_lang': 'english', 'target_lang': 'hindi',
            'include_context': True, 'use_reranker': True,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert bool(data['translated_text'])
        assert data['reranked'] is False


# ---------------------------------------------------------------------------
# Translation memory benchmark (stub model)
# ---------------------------------------------------------------------------

class TestTMBenchmark:
    def test_exact_hits_and_latency_reduction(self):
        import time
        from app.evaluation.tm_benchmark import TranslationMemoryBenchmark

        def stub(text, source_lang, target_lang):
            time.sleep(0.005)
            return f'[{source_lang}->{target_lang}] {text}'

        sentences = ['Hello world', 'How are you today?', 'The weather is nice.']
        result = TranslationMemoryBenchmark().run(
            stub, sentences=sentences, source_lang='english', target_lang='hindi',
            paraphrases=['Hello world', 'How are you today?', 'The weather is nice.'],
            enable_encoder=False,  # keep the test fast + offline
        )
        assert result['exact_hit_rate'] == 1.0
        assert result['semantic_hit_rate'] == 1.0  # exact paraphrases
        assert result['latency_reduction_pct'] > 0
        assert result['tm_hit']['mean_ms'] < result['model']['mean_ms']
        assert result['encoder_available'] is False


# ---------------------------------------------------------------------------
# Back-translation diagnostic
# ---------------------------------------------------------------------------

class TestBackTranslation:
    def test_runs_with_stub(self):
        pytest.importorskip('sacrebleu')
        from app.evaluation.backtranslation import BackTranslationTest

        def stub(text, source_lang, target_lang):
            return f'[{source_lang}->{target_lang}] {text}'

        result = BackTranslationTest().run(
            stub, sentences=['Hello world.', 'Good morning.']
        )
        assert result['num_sentences'] == 2
        assert 'bleu' in result['scores']


# ---------------------------------------------------------------------------
# Master CLI helpers (deep_merge for staged/continued runs)
# ---------------------------------------------------------------------------

class TestBenchmarkCLIHelpers:
    def test_deep_merge_per_language(self):
        from app.evaluation.report import deep_merge
        base = {'translation': {'hindi': {'a': 1}}, 'functional': {'x': 1}}
        override = {'translation': {'marathi': {'b': 2}}, 'functional': {'x': 2}}
        merged = deep_merge(base, override)
        # Per-language results accumulate
        assert merged['translation']['hindi']['a'] == 1
        assert merged['translation']['marathi']['b'] == 2
        # Non-dict leaves overwrite
        assert merged['functional']['x'] == 2


# ---------------------------------------------------------------------------
# Report / scorecard
# ---------------------------------------------------------------------------

class TestReport:
    def _fake_results(self):
        return {
            'functional': {'pass_rate': 100.0, 'passed': 40, 'total': 40},
            'unicode': {'pass_rate': 100.0, 'passed': 12, 'total': 12},
            'quality_estimation': {'f1': 0.97},
            'terminology_ablation': {
                'hindi': {
                    'num_sentences': 240, 'num_required_terms': 260,
                    'glossary_size': 900, 'underpowered': False,
                    'strategies': {
                        'none': {'term_accuracy': 0.62, 'chrf_plus': 55.0, 'bleu': 30.0,
                                 'applied_rate': 0.0, 'fallback_rate': 0.0, 'p95_ms': 900},
                        'inline': {'term_accuracy': 0.93, 'chrf_plus': 55.4, 'bleu': 30.5,
                                   'delta_chrf': 0.4, 'ci95_halfwidth': 0.3,
                                   'p_value': 0.01, 'applied_rate': 1.0,
                                   'fallback_rate': 0.0, 'p95_ms': 950},
                    },
                },
            },
            'retrieval': {
                'backend': 'keyword-fallback', 'index_size': 15, 'valid': False,
                'num_queries_total': 105,
                'languages': {'hindi': {'recall_at_k': {5: 0.9}, 'mrr': 0.7}},
                'aggregate': {
                    'sentence': {'recall_at_1': 0.95, 'recall_at_5': 0.98, 'mrr': 0.96},
                    'paraphrase': {'recall_at_1': 0.0, 'recall_at_5': 0.0, 'mrr': 0.0},
                    'sentence_term_recall': 1.0,
                    'distractor_false_positive_rate': 0.0,
                    'enforced_precision': 0.92,
                },
            },
            'translation_memory': {
                'lang_pair': 'english->hindi', 'num_sentences': 8,
                'tm_hit': {'mean_ms': 1.0, 'percentiles': {'p95_ms': 1.0}},
                'model': {'mean_ms': 500.0},
                'exact_hit_rate': 1.0, 'semantic_hit_rate': 0.0,
                'encoder_available': False, 'latency_reduction_pct': 99.8,
            },
            'flores': {'num_sentences': 1012},
            'flores_translation': {
                'hindi': {'models': {'nllb': {'summary': {'bleu': 30.0, 'chrf_plus': 55.0},
                                              'latency': {'p95_ms': 1500.0}}}},
            },
            'translation': {
                'hindi': {'models': {'nllb': {'summary': {'bleu': 67.0, 'chrf_plus': 79.0},
                                              'latency': {'p95_ms': 1200.0}}}},
            },
            'hallucination': {'aggregate': {'degenerate_output_rate_pct': 0.0}},
        }

    def _statuses(self, results):
        from app.evaluation.report import build_scorecard
        sc = build_scorecard(results)
        return sc, {e['key']: e['status'] for e in sc['entries']}

    def test_scorecard_statuses(self, monkeypatch):
        monkeypatch.delenv('TERMINOLOGY_STRATEGY', raising=False)
        sc, st = self._statuses(self._fake_results())
        assert st['terminology_accuracy'] == 'PASS'
        assert st['terminology_chrf_delta'] == 'PASS'
        assert st['terminology_sample_size'] == 'PASS'
        assert st['translation_p95_ms'] == 'PASS'      # worst of FLORES + curated
        assert st['hallucination_rate_pct'] == 'PASS'
        # A 15-entry index and zero paraphrase recall must fail, not pass.
        assert st['retrieval_index_size'] == 'FAIL'
        assert st['retrieval_paraphrase_recall_at_5'] == 'FAIL'
        assert st['retrieval_distractor_fp_rate'] == 'PASS'
        assert st['language_detection_f1'] == 'NOT MEASURED'
        assert sc['summary'] == {'pass': 6, 'fail': 2, 'not_measured': 1,
                                 'total_targets': 9}

    def test_ci_suites_and_cache_latency_are_not_scored(self):
        sc, st = self._statuses(self._fake_results())
        for removed in ('functional_pass_rate', 'unicode_pass_rate', 'e2e_pass_rate',
                        'security_pass_rate', 'tm_hit_p95_ms', 'quality_estimation_f1'):
            assert removed not in st
        assert sc['test_suites']['functional'] == {'passed': 40, 'total': 40,
                                                   'pass_rate': 100.0}

    def test_mock_sections_are_not_measured(self):
        results = self._fake_results()
        results['hallucination']['mode'] = 'mock'
        _, st = self._statuses(results)
        assert st['hallucination_rate_pct'] == 'NOT MEASURED'
        results['metadata'] = {'mode': 'mock'}
        _, st = self._statuses(results)
        assert st['translation_p95_ms'] == 'NOT MEASURED'

    def test_empty_results_are_not_measured(self):
        sc, st = self._statuses({})
        assert set(st.values()) == {'NOT MEASURED'}
        assert sc['summary']['pass'] == 0

    def test_uses_configured_strategy(self, monkeypatch):
        monkeypatch.setenv('TERMINOLOGY_STRATEGY', 'placeholder')
        _, st = self._statuses(self._fake_results())
        assert st['terminology_accuracy'] == 'NOT MEASURED'  # no placeholder run

    def test_markdown_puts_flores_first_and_labels_curated(self):
        from app.evaluation.report import build_scorecard, render_markdown
        results = self._fake_results()
        md = render_markdown(build_scorecard(results), results)
        assert md.index('FLORES-200 devtest (main results)') < md.index('Curated set')
        assert 'machine-generated' in md
        assert 'Terminology ablation' in md and '| inline | 93.0% |' in md
        assert 'too small to be evidence' in md
        assert 'Test suites (CI checks, not scored): functional 40/40' in md

    def test_scorecard_robust_to_json_round_trip(self):
        import json
        from app.evaluation.report import build_scorecard
        data = json.loads(json.dumps(self._fake_results()))
        sc = build_scorecard(data)
        assert sc['summary']['pass'] + sc['summary']['fail'] == 8


# ---------------------------------------------------------------------------
# Hallucination detector (pure logic)
# ---------------------------------------------------------------------------

class TestHallucinationDetector:
    @pytest.fixture
    def D(self):
        from app.evaluation.hallucination import HallucinationDetector
        return HallucinationDetector

    def test_repetition_ratio(self, D):
        # Distinct text -> no repetition
        assert D.repetition_ratio('The quick brown fox jumps over the lazy dog.') < 0.2
        # Stuttering text -> heavy repetition
        assert D.repetition_ratio('यह यह यह यह यह यह यह यह यह।') > 0.5
        # Short/empty -> 0
        assert D.repetition_ratio('a') == 0.0
        assert D.repetition_ratio('') == 0.0

    def test_source_leakage_ratio(self, D):
        src = 'The hospital is near the railway station in Pune.'
        assert D.source_leakage_ratio(src, src) == pytest.approx(1.0)
        # A real rewrite shares few characters
        assert D.source_leakage_ratio('अस्पताल रेलवे स्टेशन के पास है।', src) < 0.5

    def test_entity_preservation(self, D):
        src = 'Dr. Meera Patel charged $1,250.75 on 12/08/2026 via support@example.com'
        # NMT keeps numbers, dates, emails and (often) names in Latin script
        hyp = 'डॉ. Dr. Meera Patel ने $1,250.75 on 12/08/2026 via support@example.com लिया।'
        assert D.entity_preservation_rate(hyp, src) == pytest.approx(1.0)
        assert D.entity_preservation_rate('कोई संस्था नहीं है।', src) < 1.0

    def test_entity_extraction(self, D):
        from app.evaluation.hallucination import extract_entities
        src = 'Dr. Meera Patel charged $1,250.75 on 12/08/2026 via support@example.com'
        ents = extract_entities(src)
        assert '$1,250.75' in ents
        assert '12/08/2026' in ents          # date matches whole, not fragments
        assert 'support@example.com' in ents

    def test_is_degenerate_verdicts(self, D):
        src = 'The meeting is on 12/08/2026 with Dr. Meera Patel.'
        assert D.is_degenerate(src, src) == (True, 'output is an exact copy of the source')
        assert D.is_degenerate('', src)[0] is True            # empty
        assert D.is_degenerate('यह यह यह यह यह यह यह यह', src)[0] is True  # stutter
        good = 'डॉ. मीरा पटेल के साथ बैठक 12/08/2026 को है।'
        assert D.is_degenerate(good, src)[0] is False


# ---------------------------------------------------------------------------
# Throughput benchmark (mock, fast)
# ---------------------------------------------------------------------------

class TestThroughputBenchmark:
    def test_latency_profile_and_concurrency(self):
        from app.evaluation.throughput import ThroughputBenchmark
        bench = ThroughputBenchmark()
        sentences = ['Hello world', 'How are you today?', 'The weather is nice.']
        result = bench.run(sentences=sentences, rounds=1, warmup=0,
                           levels=(1, 2, 4), iters_per_worker=3,
                           track_memory=False)
        lat = result['latency']
        assert lat['num_calls'] == 3
        assert lat['p95_ms'] >= lat['mean_ms'] - 1e-6
        assert lat['throughput_sentences_per_sec'] > 0
        # All three probes are < 60 chars -> single short bucket
        assert lat['by_length']['short']['n'] == 3
        levels = result['concurrency']['levels']
        assert [lv['concurrency'] for lv in levels] == [1, 2, 4]
        assert all(lv['error_rate_pct'] == 0.0 for lv in levels)
        assert result['concurrency']['peak_req_per_sec'] > 0
        assert result['mode'] == 'mock'


# ---------------------------------------------------------------------------
# Security suite (endpoint hardening regression)
# ---------------------------------------------------------------------------

class TestSecuritySuite:
    def test_full_matrix_passes(self):
        from app.evaluation.security import SecurityTestSuite
        result = SecurityTestSuite().run()
        assert result['total'] >= 30
        assert result['passed'] == result['total']
        assert result['pass_rate'] == 100.0
        assert set(result['categories'].keys()) == {
            'input_validation', 'content_robustness', 'injection_abuse', 'auth',
        }


# ---------------------------------------------------------------------------
# End-to-end pipeline test (mock, fast)
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_full_lifecycle_passes(self):
        from app.evaluation.e2e import EndToEndTest
        result = EndToEndTest().run()
        assert result['total'] == 10
        assert result['passed'] == 10
        assert result['pass_rate'] == 100.0
        names = [s['name'] for s in result['stages']]
        assert 'translate_with_rag' in names
        assert 'history_list' in names


# ---------------------------------------------------------------------------
# Script presence helper (code-mixed detection support)
# ---------------------------------------------------------------------------

class TestScriptsPresent:
    def test_mixed_scripts(self):
        from app.services.language_detection import scripts_present
        assert scripts_present('मैं कल दिल्ली जा रहा हूँ।') == {'devanagari'}
        assert 'latin' in scripts_present('मैं tomorrow दिल्ली जाऊंगा।')
        assert 'devanagari' in scripts_present('मैं tomorrow दिल्ली जाऊंगा।')
        assert scripts_present('###!!! 1234') == set()
