"""
BERTScore — semantic similarity using contextual embeddings.

Uses multilingual BERT models to compute precision, recall, and F1
between hypothesis and reference tokens. This captures paraphrases
and semantically correct but lexically different translations.
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class BERTScoreScorer:
    """BERTScore computation using bert-score library."""

    # Best multilingual models for Indic languages
    DEFAULT_MODELS = [
        'xlm-roberta-large',       # Best multilingual, but large (~1.4GB)
        'microsoft/mdeberta-v3-base',  # Good multilingual, smaller (~500MB)
    ]

    def __init__(self, model_type: Optional[str] = None, verbose: bool = False):
        """
        Args:
            model_type: BERTScore model to use. If None, tries the best available.
            verbose: Whether to show BERTScore loading logs.
        """
        self._available = False
        self._model_type = model_type
        self._verbose = verbose
        self._check_deps()

    def _check_deps(self):
        try:
            import bert_score  # noqa: F401
            self._available = True
        except ImportError:
            logger.warning("bert-score not installed. BERTScore will be unavailable.")

    @property
    def available(self) -> bool:
        return self._available

    def score(
        self,
        hypotheses: List[str],
        references: List[List[str]],
        model_type: Optional[str] = None,
        lang: str = 'en',
        verbose: Optional[bool] = None,
    ) -> Dict:
        """
        Compute BERTScore for translation hypotheses against references.

        Args:
            hypotheses: List of translated sentences.
            references: List of reference translations (list of list of strings).
                       Uses the first reference from each entry.
            model_type: BERT model to use. Defaults to xlm-roberta-large for best quality.
            lang: Two-letter language code for IDF weighting.
            verbose: Whether to show loading/progress logs.

        Returns:
            dict with keys: precision, recall, f1, precision_std, recall_std, f1_std,
                            model_type, num_sentences
        """
        if not self._available:
            return {'error': 'bert-score not installed', 'f1': 0.0}

        import bert_score

        model = model_type or self._model_type or self.DEFAULT_MODELS[0]
        verb = verbose if verbose is not None else self._verbose

        try:
            # Extract single reference per sentence
            refs = []
            for ref_list in references:
                if isinstance(ref_list, list) and ref_list:
                    refs.append(ref_list[0])
                else:
                    refs.append('')

            if not hypotheses or not refs:
                return {'error': 'Empty hypotheses or references', 'f1': 0.0}

            # Filter empty pairs
            valid_pairs = [(h, r) for h, r in zip(hypotheses, refs) if h.strip() and r.strip()]
            if not valid_pairs:
                return {'error': 'No valid hypothesis-reference pairs', 'f1': 0.0}

            valid_hyp, valid_ref = zip(*valid_pairs)

            # Compute BERTScore
            P, R, F1 = bert_score.score(
                list(valid_hyp),
                list(valid_ref),
                model_type=model,
                lang=lang,
                verbose=verb,
                batch_size=32,
                device=None,  # auto-detect
            )

            result = {
                'precision': P.mean().item(),
                'recall': R.mean().item(),
                'f1': F1.mean().item(),
                'precision_std': P.std().item(),
                'recall_std': R.std().item(),
                'f1_std': F1.std().item(),
                'model_type': model,
                'num_sentences': len(valid_hyp),
            }
            return result

        except Exception as e:
            logger.error(f"BERTScore scoring failed: {e}")
            return {'error': str(e), 'f1': 0.0}

    def unload(self):
        """
        Release BERTScore model weights from memory (best-effort).

        bert-score 0.3.x keeps the transformer model on a per-call BERTScorer
        instance that becomes garbage once score() returns, but torch's
        reference cycles can keep it resident; release_torch_memory() runs
        gc.collect() to reclaim it. The benchmark matrix calls this between
        languages so peak memory stays bounded on machines that also hold
        NLLB/m2m100 + COMET.
        """
        from .utils import release_torch_memory
        release_torch_memory()
        logger.info("BERTScore model weights released")

    def __repr__(self):
        return f"BERTScoreScorer(available={self._available}, model={self._model_type})"
