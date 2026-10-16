"""
Back-translation diagnostic (supplementary test only).

English -> Indic -> English, then compare the back-translated English
against the original with BLEU (and BERTScore when available).

IMPORTANT (as the plan warns): back-translation is NOT a primary quality
metric — a system can translate incorrectly and still produce a plausible
back-translation. Use it only as an additional diagnostic that can expose
gross information loss.

Usage (from backend/):
    python -m app.evaluation.backtranslation --mock        # stub (fast)
    python -m app.evaluation.backtranslation               # live model
"""

import logging

logger = logging.getLogger(__name__)

DEFAULT_SENTENCES = [
    "India is one of the most diverse countries in the world.",
    "The patient should drink enough water every day.",
    "Students should submit their assignments before Friday.",
    "The interest rate has increased this year.",
    "Artificial intelligence is transforming software development.",
    "The agreement will remain valid for one year.",
    "Could you please send me the document tomorrow?",
]


class BackTranslationTest:
    """Round-trip translation diagnostic with corpus-level similarity scores."""

    def __init__(self):
        self._bleu_available = False
        self._bertscore_available = False
        self._check_deps()

    def _check_deps(self):
        try:
            import sacrebleu  # noqa: F401
            self._bleu_available = True
        except ImportError:
            logger.warning('sacrebleu not installed — BLEU will be unavailable')
        try:
            import bert_score  # noqa: F401
            self._bertscore_available = True
        except ImportError:
            logger.warning('bert-score not installed — BERTScore will be unavailable')

    @property
    def available(self):
        return self._bleu_available

    def run(self, model_fn, sentences=None, source_lang='english',
            target_lang='hindi', use_bertscore=False) -> dict:
        """
        Run the back-translation diagnostic.

        Args:
            model_fn: callable (text, source_lang, target_lang) -> str.
            sentences: English source sentences (default: DEFAULT_SENTENCES).
            source_lang / target_lang: Language pair.
            use_bertscore: Also compute BERTScore. Off by default because it
                      downloads a ~570MB multilingual model on first use.

        Returns:
            dict with per-sentence round trips and corpus similarity scores.
        """
        sentences = sentences or DEFAULT_SENTENCES

        round_trips = []
        for src in sentences:
            try:
                fwd = model_fn(src, source_lang=source_lang, target_lang=target_lang)
                back = model_fn(fwd, source_lang=target_lang, target_lang=source_lang)
                round_trips.append({
                    'original': src,
                    'forward': fwd,
                    'back_translated': back,
                    'error': None,
                })
            except Exception as e:
                round_trips.append({
                    'original': src, 'forward': '', 'back_translated': '',
                    'error': str(e),
                })

        originals = [r['original'] for r in round_trips if not r['error']]
        back = [r['back_translated'] for r in round_trips if not r['error']]

        result = {
            'lang_pair': f'{source_lang}->{target_lang}->{source_lang}',
            'num_sentences': len(round_trips),
            'num_successful': len(originals),
            'round_trips': round_trips,
            'scores': {},
        }

        if originals and self._bleu_available:
            try:
                import sacrebleu
                bleu = sacrebleu.corpus_bleu(back, [[o] for o in originals], force=True)
                result['scores']['bleu'] = round(bleu.score, 2)
                result['scores']['sentence_bleu_mean'] = round(sum(
                    sacrebleu.sentence_bleu(b, [o]).score
                    for b, o in zip(back, originals)
                ) / len(originals), 2)
            except Exception as e:
                result['scores']['bleu_error'] = str(e)

        if use_bertscore and originals and self._bertscore_available:
            try:
                import bert_score
                _, _, f1 = bert_score.score(
                    back, originals, lang='en',
                    model_type='microsoft/mdeberta-v3-base', verbose=False,
                )
                result['scores']['bertscore_f1'] = round(f1.mean().item(), 4)
            except Exception as e:
                result['scores']['bertscore_error'] = str(e)
        elif use_bertscore and not self._bertscore_available:
            result['scores']['bertscore_error'] = 'bert-score not installed'

        if not result['scores']:
            result['scores']['note'] = (
                'No metric libraries available (pip install sacrebleu bert-score)'
            )

        return result


def run_backtranslation_test(model_fn=None, **kwargs) -> dict:
    """Module-level entry point; uses the real model when model_fn is None."""
    if model_fn is None:
        from app.services.model_registry import ModelRegistry

        registry = ModelRegistry.get_instance()

        def live_model_fn(text, source_lang, target_lang):
            result = registry.translate(
                text=text, source_lang=source_lang, target_lang=target_lang,
                use_tm=False,
            )
            return result.translated

        model_fn = live_model_fn

    return BackTranslationTest().run(model_fn, **kwargs)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Back-translation diagnostic')
    parser.add_argument('--mock', action='store_true',
                        help='Use a fast echo stub instead of the real model')
    parser.add_argument('--target', default='hindi')
    args = parser.parse_args()

    if args.mock:
        def stub(text, source_lang, target_lang):
            return f'[{source_lang}->{target_lang}] {text}'
        model_fn = stub
    else:
        model_fn = None

    result = run_backtranslation_test(model_fn, target_lang=args.target)
    print("\nBack-Translation Diagnostic")
    print("-" * 40)
    print(f"Pair: {result['lang_pair']}")
    print(f"Sentences: {result['num_sentences']} "
          f"(successful: {result['num_successful']})")
    print(f"Scores: {result['scores']}")
    if not args.mock:
        print("\nRound trips:")
        for r in result['round_trips']:
            print(f"  EN : {r['original']}")
            print(f"  {args.target.upper()}: {r['forward']}")
            print(f"  EN : {r['back_translated']}\n")
