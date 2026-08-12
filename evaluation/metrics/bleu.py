"""
BLEU (Bilingual Evaluation Understudy) metric.
Uses sacrebleu for standardized, reproducible scoring.
"""

import logging
from typing import List, Dict, Optional

from .utils import std

logger = logging.getLogger(__name__)


class BLEUScorer:
    """BLEU score computation with multiple variant support."""

    def __init__(self):
        self._available = False
        self._check_deps()

    def _check_deps(self):
        try:
            import sacrebleu  # noqa: F401
            self._available = True
        except ImportError:
            logger.warning("sacrebleu not installed. BLEU scoring will be unavailable.")

    @property
    def available(self) -> bool:
        return self._available

    def score(
        self,
        hypotheses: List[str],
        references: List[List[str]],
        smooth_method: str = 'exp',
        smooth_value: Optional[float] = None,
    ) -> Dict:
        """
        Compute BLEU scores.

        Args:
            hypotheses: List of translated sentences.
            references: List of reference translations (each is a list of reference strings,
                       usually 1 reference is provided).
            smooth_method: Smoothing method ('exp', 'floor', 'add-k', 'none').
            smooth_value: Smoothing value (for 'floor' and 'add-k' methods).

        Returns:
            dict with keys: bleu, bleu_1, bleu_2, bleu_3, bleu_4, precisions, brevity_penalty,
                            reference_length, hypothesis_length, score
        """
        if not self._available:
            return {'error': 'sacrebleu not installed', 'bleu': 0.0}

        import sacrebleu

        # sacrebleu expects list of hypotheses and list of list of references
        # where references are flattened per sentence
        try:
            # Convert to format sacrebleu expects
            # hypotheses: List[str]
            # references: List[str] for single reference, or List[List[str]] for multiple

            # Handle single reference: [[ref1, ref2, ...]] -> list of single refs per hyp
            refs_for_bleu = []
            for ref_list in references:
                if isinstance(ref_list, list):
                    refs_for_bleu.append(ref_list[0] if ref_list else '')
                else:
                    refs_for_bleu.append(ref_list)

            if not hypotheses or not refs_for_bleu:
                return {'error': 'Empty hypotheses or references', 'bleu': 0.0}

            # Calculate overall corpus BLEU
            corpus_bleu = sacrebleu.corpus_bleu(
                hypotheses,
                [refs_for_bleu],
                smooth_method=smooth_method,
                smooth_value=smooth_value,
                force=True,
            )

            # Calculate sentence-level BLEU for variance analysis
            sentence_bleus = []
            for hyp, ref_list in zip(hypotheses, references):
                try:
                    if isinstance(ref_list, list) and ref_list:
                        sent_bleu = sacrebleu.sentence_bleu(
                            hyp,
                            ref_list,
                            smooth_method=smooth_method,
                        )
                        sentence_bleus.append(sent_bleu.score)
                    else:
                        sentence_bleus.append(0.0)
                except Exception:
                    sentence_bleus.append(0.0)

            result = {
                'bleu': corpus_bleu.score,
                'bleu_1': corpus_bleu.precisions[0] if hasattr(corpus_bleu, 'precisions') else 0.0,
                'bleu_2': corpus_bleu.precisions[1] if hasattr(corpus_bleu, 'precisions') and len(corpus_bleu.precisions) > 1 else 0.0,
                'bleu_3': corpus_bleu.precisions[2] if hasattr(corpus_bleu, 'precisions') and len(corpus_bleu.precisions) > 2 else 0.0,
                'bleu_4': corpus_bleu.precisions[3] if hasattr(corpus_bleu, 'precisions') and len(corpus_bleu.precisions) > 3 else 0.0,
                'precisions': corpus_bleu.precisions if hasattr(corpus_bleu, 'precisions') else [],
                'brevity_penalty': corpus_bleu.bp if hasattr(corpus_bleu, 'bp') else 1.0,
                'reference_length': corpus_bleu.ref_len if hasattr(corpus_bleu, 'ref_len') else 0,
                'hypothesis_length': corpus_bleu.hyp_len if hasattr(corpus_bleu, 'hyp_len') else 0,
                'sentence_bleu_mean': sum(sentence_bleus) / max(len(sentence_bleus), 1),
                'sentence_bleu_std': std(sentence_bleus),
                'num_sentences': len(hypotheses),
            }
            return result

        except Exception as e:
            logger.error(f"BLEU scoring failed: {e}")
            return {'error': str(e), 'bleu': 0.0}

    def score_pair(self, hypothesis: str, reference: str) -> Dict:
        """Score a single translation pair."""
        return self.score([hypothesis], [[reference]])

    def __repr__(self):
        return f"BLEUScorer(available={self._available})"
