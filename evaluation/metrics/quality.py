"""
Wrapper around the existing QualityScorer for use in the evaluation pipeline.

This integrates our lightweight built-in quality scoring into the
evaluation framework, allowing direct comparison between our heuristic
quality scores and standard NLP metrics.
"""

import logging
from typing import List, Dict

from .utils import std

logger = logging.getLogger(__name__)


class QualityScorerWrapper:
    """Wrapper for the built-in QualityScorer to match evaluation API."""

    def __init__(self):
        self._scorer = None
        self._load_scorer()

    def _load_scorer(self):
        try:
            from app.services.quality import QualityScorer
            self._scorer = QualityScorer()
        except Exception as e:
            logger.warning(f"Could not load QualityScorer: {e}")

    @property
    def available(self) -> bool:
        return self._scorer is not None

    def score(
        self,
        hypotheses: List[str],
        references: List[List[str]],
        **kwargs,
    ) -> Dict:
        """
        Score translations using the built-in QualityScorer.

        Note: QualityScorer doesn't use references
        (it's reference-free), so this wrapper returns
        the heuristic scores alongside standard metrics.

        Args:
            hypotheses: List of translated sentences.
            references: List of reference translations (unused, for API compatibility).

        Returns:
            dict with mean quality score, confidence distribution, etc.
        """
        if not self._scorer:
            return {'error': 'QualityScorer not available', 'quality_score': 0.0}

        if not hypotheses:
            return {'error': 'No hypotheses provided', 'quality_score': 0.0}

        # We don't have source sentences here, so score with source as hyp
        # This is mainly for structural consistency — the real quality
        # scoring happens inline during translation.
        scores = []
        confidences = {'high': 0, 'medium': 0, 'low': 0}
        hallucinations = 0

        for i, hyp in enumerate(hypotheses):
            try:
                # Use the first reference as source if available, else the hypothesis itself
                source = ''
                if references and i < len(references) and references[i]:
                    source = references[i][0] if isinstance(references[i], list) else references[i]
                else:
                    source = hyp

                # Pass the target language (parsed from lang_pair like
                # 'english->hindi') so the script-consistency check applies,
                # consistent with the translate route.
                target_lang = None
                pair = kwargs.get('lang_pair') or ''
                if '->' in pair:
                    target_lang = pair.split('->', 1)[1].strip()
                result = self._scorer.score(
                    source_text=source,
                    translated_text=hyp,
                    retrieved_terms=kwargs.get('retrieved_terms', []),
                    target_lang=target_lang,
                )
                scores.append(result['overall_score'])
                confidences[result['confidence']] = confidences.get(result['confidence'], 0) + 1
                if result['hallucination_detected']:
                    hallucinations += 1
            except Exception as e:
                logger.debug(f"Quality scoring failed for sentence {i}: {e}")
                scores.append(0.0)

        if not scores:
            return {'quality_score': 0.0, 'error': 'No valid scores'}

        return {
            'quality_score': sum(scores) / len(scores),                'quality_score_std': std(scores),
            'confidence_distribution': confidences,
            'hallucination_rate': hallucinations / max(len(scores), 1),
            'num_sentences': len(scores),
        }

    def __repr__(self):
        return f"QualityScorerWrapper(available={self.available})"
