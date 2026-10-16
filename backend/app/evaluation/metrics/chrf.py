"""
chrF++ metric — character n-gram F-score with word n-grams.
Uses sacrebleu for standardized computation.

chrF++ extends chrF by also considering word n-grams,
making it more sensitive to morphological variations —
especially important for agglutinative Indic languages.
"""

import logging
from typing import List, Dict

from .utils import std

logger = logging.getLogger(__name__)


class ChrFScorer:
    """chrF++ score computation."""

    def __init__(self):
        self._available = False
        self._check_deps()

    def _check_deps(self):
        try:
            import sacrebleu  # noqa: F401
            self._available = True
        except ImportError:
            logger.warning("sacrebleu not installed. chrF++ scoring will be unavailable.")

    @property
    def available(self) -> bool:
        return self._available

    def score(
        self,
        hypotheses: List[str],
        references: List[List[str]],
        nword_order: int = 2,
        ncorder: int = 6,
        beta: float = 2.0,
    ) -> Dict:
        """
        Compute chrF++ scores.

        Args:
            hypotheses: List of translated sentences.
            references: List of reference translations (list of list of strings).
            nword_order: Word n-gram order (default 2, as in original chrF++ paper).
            ncorder: Character n-gram order (default 6, as in original chrF++ paper).
            beta: Parameter determining importance of recall vs precision (beta > 1 favours recall).

        Returns:
            dict with keys: chrf_plus, chrf_plus_std, character_precision, character_recall,
                            word_precision, word_recall, num_sentences
        """
        if not self._available:
            return {'error': 'sacrebleu not installed', 'chrf_plus': 0.0}

        import sacrebleu

        try:
            # Extract single reference per sentence
            refs = []
            for ref_list in references:
                if isinstance(ref_list, list) and ref_list:
                    refs.append(ref_list[0])
                else:
                    refs.append('')

            if not hypotheses or not refs:
                return {'error': 'Empty hypotheses or references', 'chrf_plus': 0.0}

            # Corpus-level chrF++
            # chrF++ uses char_order=6 and word_order=2 as per the original paper
            corpus_chrf = sacrebleu.corpus_chrf(
                hypotheses,
                [refs],
                char_order=ncorder,       # Character n-gram order (default 6)
                word_order=nword_order,   # Word n-gram order (default 2)
                beta=beta,
                remove_whitespace=True,
            )

            # Sentence-level chrF++ for variance analysis
            sent_scores = []
            for hyp, ref_list in zip(hypotheses, references):
                try:
                    if isinstance(ref_list, list) and ref_list:
                        sent = sacrebleu.sentence_chrf(
                            hyp,
                            ref_list,
                            char_order=ncorder,
                            word_order=nword_order,
                            beta=beta,
                            remove_whitespace=True,
                        )
                        sent_scores.append(sent.score)
                    else:
                        sent_scores.append(0.0)
                except Exception:
                    sent_scores.append(0.0)

            result = {
                'chrf_plus': corpus_chrf.score,
                'chrf_plus_std': std(sent_scores),
                'char_precision': corpus_chrf.char_prec if hasattr(corpus_chrf, 'char_prec') else 0.0,
                'char_recall': corpus_chrf.char_recall if hasattr(corpus_chrf, 'char_recall') else 0.0,
                'word_precision': corpus_chrf.word_prec if hasattr(corpus_chrf, 'word_prec') else 0.0,
                'word_recall': corpus_chrf.word_recall if hasattr(corpus_chrf, 'word_recall') else 0.0,
                'sentence_scores_mean': sum(sent_scores) / max(len(sent_scores), 1),
                'sentence_scores_std': std(sent_scores),
                'num_sentences': len(hypotheses),
            }
            return result

        except Exception as e:
            logger.error(f"chrF++ scoring failed: {e}")
            return {'error': str(e), 'chrf_plus': 0.0}

    def __repr__(self):
        return f"ChrFScorer(available={self._available})"
