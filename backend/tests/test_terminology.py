"""Terminology enforcement strategies (app/services/terminology.py)."""

import pytest

from app.services import terminology as T
from app.services.translators.base import TranslationResult

GLOSSARY = [
    {'term': 'bank', 'translation': 'बैंक'},
    {'term': 'machine learning', 'translation': 'मशीन लर्निंग'},
    {'term': 'machine', 'translation': 'मशीन'},
    {'term': 'education', 'translation': 'शिक्षा'},
]


class FakeModel:
    """Records calls; output is produced by a per-test function."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        out = self.fn(text, kwargs)
        if isinstance(out, TranslationResult):
            return out
        return TranslationResult(original=text, translated=out,
                                 source_lang='english', target_lang='hindi')


class TestMatchTerms:
    def test_only_terms_present_in_source(self):
        found = T.match_terms('The bank approved it.', GLOSSARY)
        assert [m.term for m in found] == ['bank']

    def test_longest_term_wins(self):
        found = T.match_terms('Machine learning is useful.', GLOSSARY)
        assert [m.term for m in found] == ['machine learning']

    def test_case_plural_and_word_boundaries(self):
        found = T.match_terms('Two Banks; no embankment.', GLOSSARY)
        assert [m.surface for m in found] == ['Banks']

    def test_multiple_matches_sorted_by_position(self):
        found = T.match_terms('Education loans from the bank', GLOSSARY)
        assert [m.term for m in found] == ['education', 'bank']

    def test_ignores_incomplete_entries(self):
        assert T.match_terms('bank', [{'term': 'bank'}]) == []


class TestTransforms:
    def test_inline_source(self):
        m = T.match_terms('I went to the bank today', GLOSSARY)
        assert T.inline_source('I went to the bank today', m) == 'I went to the बैंक today'

    def test_mask_and_unmask_roundtrip(self):
        text = 'The bank funds education.'
        masked, mapping = T.mask_source(text, T.match_terms(text, GLOSSARY))
        assert 'bank' not in masked and 'education' not in masked
        model_out = ' '.join(mapping)  # model kept both placeholders
        restored, ok = T.unmask(model_out, mapping)
        assert ok and 'बैंक' in restored and 'शिक्षा' in restored

    def test_unmask_tolerates_case_and_spacing(self):
        restored, ok = T.unmask('zxq a से', {'ZXQA': 'बैंक'})
        assert ok and restored == 'बैंक से'

    def test_unmask_reports_lost_placeholder(self):
        _, ok = T.unmask('कुछ नहीं', {'ZXQA': 'बैंक'})
        assert ok is False

    def test_placeholder_tokens_are_unique_and_letter_only(self):
        tokens = [T.placeholder_token(i) for i in range(60)]
        assert len(set(tokens)) == 60
        assert all(t.isalpha() for t in tokens)

    def test_term_accuracy_allows_suffixes(self):
        assert T.term_accuracy('வங்கியில் கடன்', ['வங்கி', 'கல்வி']) == (1, 2)


class TestTranslateWithTerms:
    def test_no_match_is_plain_translation_with_tm(self):
        model = FakeModel(lambda text, kw: 'नमस्ते')
        result, info = T.translate_with_terms(model, 'Hello', GLOSSARY, 'inline')
        assert result.translated == 'नमस्ते'
        assert info.applied is False and info.matched_terms == []
        assert 'use_tm' not in model.calls[0][1]

    def test_inline_rewrites_source_and_bypasses_tm(self):
        model = FakeModel(lambda text, kw: 'मैं बैंक गया')
        result, info = T.translate_with_terms(model, 'I went to the bank', GLOSSARY, 'inline')
        sent, kwargs = model.calls[0]
        assert sent == 'I went to the बैंक' and kwargs['use_tm'] is False
        assert result.original == 'I went to the bank'
        assert info.applied and info.to_dict()['term_accuracy'] == 1.0

    def test_prompt_passes_glossary_context(self):
        model = FakeModel(lambda text, kw: 'मैं बैंक गया')
        _, info = T.translate_with_terms(model, 'I went to the bank', GLOSSARY, 'prompt')
        assert model.calls[0][1]['context'] == 'bank = बैंक'
        assert info.applied

    def test_placeholder_restores_terms(self):
        model = FakeModel(lambda text, kw: text.replace('I went to the', 'मैं गया'))
        result, info = T.translate_with_terms(model, 'I went to the bank', GLOSSARY,
                                              'placeholder')
        assert result.translated == 'मैं गया बैंक'
        assert info.applied and info.fallback is None

    def test_placeholder_falls_back_when_lost(self):
        model = FakeModel(lambda text, kw: 'मैं गया')  # placeholder dropped
        result, info = T.translate_with_terms(model, 'I went to the bank', GLOSSARY,
                                              'placeholder')
        assert len(model.calls) == 2 and model.calls[1][0] == 'I went to the bank'
        assert info.applied is False and info.fallback

    def test_constrained_passes_force_words_and_beams(self):
        def fn(text, kw):
            r = TranslationResult(original=text, translated='बैंक', source_lang='english',
                                  target_lang='hindi')
            r.timings['constraints_applied'] = True
            return r
        model = FakeModel(fn)
        _, info = T.translate_with_terms(model, 'the bank', GLOSSARY, 'constrained',
                                         num_beams=1)
        kwargs = model.calls[0][1]
        assert kwargs['force_words'] == ['बैंक'] and kwargs['num_beams'] == 2
        assert info.applied

    def test_constrained_reports_unsupported_wrapper(self):
        model = FakeModel(lambda text, kw: 'बैंक')
        _, info = T.translate_with_terms(model, 'the bank', GLOSSARY, 'constrained')
        assert info.applied is False and 'does not support' in info.fallback

    def test_none_strategy_ignores_terms(self):
        model = FakeModel(lambda text, kw: 'x')
        _, info = T.translate_with_terms(model, 'the bank', GLOSSARY, 'none')
        assert model.calls[0][0] == 'the bank' and info.matched_terms == []

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            T.translate_with_terms(FakeModel(lambda t, k: 'x'), 'bank', GLOSSARY, 'magic')

    def test_default_strategy_from_env(self, monkeypatch):
        monkeypatch.setenv('TERMINOLOGY_STRATEGY', 'placeholder')
        assert T.default_strategy() == 'placeholder'
        monkeypatch.setenv('TERMINOLOGY_STRATEGY', 'bogus')
        assert T.default_strategy() == T.DEFAULT_STRATEGY


class TestRouteValidation:
    def test_invalid_strategy_is_400(self, monkeypatch):
        monkeypatch.setenv('DATABASE_URL', 'sqlite://')
        from app import create_app
        app = create_app('development', overrides={'TESTING': True,
                                                   'RATELIMIT_ENABLED': False})
        r = app.test_client().post('/api/v1/translate/text', json={
            'text': 'Hello', 'source_lang': 'english', 'target_lang': 'hindi',
            'terminology_strategy': 'magic'})
        assert r.status_code == 400
        assert 'terminology_strategy' in r.get_json()['error']
