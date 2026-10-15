"""Terminology ablation protocol (no models: a fake translate function)."""

from app.evaluation import terminology_ablation as A
from app.services.translators.base import TranslationResult

GLOSSARY = [{'term': 'bank', 'translation': 'बैंक'},
            {'term': 'hospital', 'translation': 'अस्पताल'}]


def fake_translate(text, **kwargs):
    """'Model' that copies target-script words and otherwise outputs a
    fixed phrase, so inline beats none on term accuracy."""
    words = [w for w in text.split() if not w.isascii()]
    out = ' '.join(words + ['गया']) if words else 'मैं गया'
    return TranslationResult(original=text, translated=out,
                             source_lang='english', target_lang='hindi')


class TestSelection:
    def test_requires_term_in_source_and_reference(self):
        subset = A.select_term_bearing(
            ['I went to the bank', 'I went to the bank', 'Nothing here'],
            ['मैं बैंक गया', 'मैं गया', 'कुछ नहीं'],
            GLOSSARY)
        assert [r['index'] for r in subset] == [0]
        assert subset[0]['required'] == ['बैंक']

    def test_skips_empty_pairs(self):
        assert A.select_term_bearing(['bank', ''], ['', 'बैंक'], GLOSSARY) == []


class TestRunAblation:
    def _data(self, n):
        src = ['I went to the bank'] * n
        ref = ['मैं बैंक गया'] * n
        return src, ref

    def test_inline_vs_none(self):
        src, ref = self._data(30)
        res = A.run_ablation(fake_translate, src, ref, GLOSSARY,
                             strategies=['none', 'inline'], min_sentences=20,
                             n_bootstrap=50)
        none, inline = res['strategies']['none'], res['strategies']['inline']
        assert res['num_sentences'] == 30 and res['underpowered'] is False
        assert none['term_accuracy'] == 0.0
        assert inline['term_accuracy'] == 1.0 and inline['applied_rate'] == 1.0
        assert inline['delta_chrf'] > 0 and 'p_value' in inline
        assert 'delta_chrf' not in none

    def test_baseline_always_included_and_underpowered_flag(self):
        src, ref = self._data(5)
        res = A.run_ablation(fake_translate, src, ref, GLOSSARY,
                             strategies=['inline'], n_bootstrap=20)
        assert set(res['strategies']) == {'none', 'inline'}
        assert res['underpowered'] is True

    def test_no_term_bearing_sentences(self):
        res = A.run_ablation(fake_translate, ['hello'], ['नमस्ते'], GLOSSARY)
        assert res['num_sentences'] == 0 and 'error' in res

    def test_translation_errors_are_counted_not_raised(self):
        def broken(text, **kwargs):
            raise RuntimeError('model crashed')
        src, ref = self._data(3)
        res = A.run_ablation(broken, src, ref, GLOSSARY, strategies=['none'],
                             n_bootstrap=10)
        assert res['strategies']['none']['errors'] == 3

    def test_markdown(self):
        src, ref = self._data(10)
        res = A.run_ablation(fake_translate, src, ref, GLOSSARY,
                             strategies=['none', 'inline'], n_bootstrap=20)
        md = A.render_markdown({'hindi': res})
        assert '| inline | 100.0% |' in md and 'underpowered' in md
