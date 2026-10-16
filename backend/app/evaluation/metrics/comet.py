"""
COMET — neural quality estimation using cross-lingual embeddings.

COMET (Unbabel) provides reference-based and reference-free quality
estimation that correlates well with human judgment.

This module makes COMET fully optional — if `unbabel-comet` is not
installed, it gracefully returns a placeholder score.
"""

import logging
from typing import List, Dict, Optional

from .utils import std

logger = logging.getLogger(__name__)


class COMETScorer:
    """COMET quality estimation (optional dependency)."""

    # Recommended models for Indic languages
    MODELS = [
        'Unbabel/wmt22-comet-da',        # SOTA reference-based (default)
        'Unbabel/wmt22-cometkiwi-da',     # Reference-free (QE)
    ]

    def __init__(self, model_name: Optional[str] = None, device: Optional[int] = None):
        """
        Args:
            model_name: COMET model to load. Defaults to wmt22-comet-da.
            device: Device ID (-1 for CPU, None for auto-detect).
        """
        self._available = False
        self._model = None
        self._model_name = model_name or self.MODELS[0]
        self._device = device
        self._check_deps()

    def _check_deps(self):
        try:
            from comet import download_model, load_from_checkpoint  # noqa: F401
            self._available = True
        except ImportError:
            logger.info("unbabel-comet not installed. COMET scoring will be unavailable.")
            logger.info("Install with: pip install unbabel-comet")

    @property
    def available(self) -> bool:
        return self._available

    def _load_model(self):
        """Lazy-load the COMET model (downloads on first use, then cached)."""
        if self._model is not None:
            return True
        if not self._available:
            return False

        try:
            from comet import download_model, load_from_checkpoint

            logger.info(f"Loading COMET model: {self._model_name}...")
            model_path = download_model(self._model_name)
            self._model = load_from_checkpoint(model_path)
            logger.info(f"COMET model loaded: {self._model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load COMET model: {e}")
            return False

    def unload(self):
        """
        Release the loaded COMET model from memory.

        COMET keeps its (~1.4GB) model resident on the scorer once loaded, so
        the benchmark matrix unloads it between languages to keep peak memory
        low on machines that also hold NLLB/m2m100 + BERTScore. score() lazily
        reloads on the next call, so results are unaffected.
        """
        if self._model is not None:
            self._model = None
            from .utils import release_torch_memory
            release_torch_memory()
            logger.info(f"COMET model unloaded: {self._model_name}")

    def score(
        self,
        hypotheses: List[str],
        references: List[List[str]],
        sources: Optional[List[str]] = None,
        batch_size: int = 8,
        gpus: Optional[int] = None,
    ) -> Dict:
        """
        Compute COMET quality scores.

        For reference-based COMET, you need hypothesis, reference, and source.
        Returns mean score in [0, 1] range.

        Args:
            hypotheses: List of translated sentences.
            references: List of reference translations (list of list of strings).
            sources: Optional source sentences (required for reference-free COMET).
            batch_size: Batch size for inference.
            gpus: Number of GPUs to use (None = auto).

        Returns:
            dict with keys: comet_score, comet_std, model, num_sentences,
                            or {'error': ...} if unavailable.
        """
        if not self._available:
            return {'error': 'unbabel-comet not installed', 'comet_score': 0.0}

        if not self._load_model():
            return {'error': 'COMET model could not be loaded', 'comet_score': 0.0}

        try:
            # Extract single reference per sentence
            refs = []
            for ref_list in references:
                if isinstance(ref_list, list) and ref_list:
                    refs.append(ref_list[0])
                else:
                    refs.append('')

            if not hypotheses or not refs:
                return {'error': 'Empty hypotheses or references', 'comet_score': 0.0}

            # Prepare data in COMET format
            data = []
            for i, (hyp, ref) in enumerate(zip(hypotheses, refs)):
                entry = {
                    'src': sources[i] if sources else '',
                    'mt': hyp,
                    'ref': ref,
                }
                data.append(entry)

            # Score
            model_output = self._model.predict(
                data,
                batch_size=batch_size,
                gpus=gpus or -1,
            )

            scores = model_output['scores'] if isinstance(model_output, dict) else list(model_output)

            result = {
                'comet_score': sum(scores) / max(len(scores), 1),
                'comet_std': std(scores),
                'comet_scores': [round(s, 4) for s in scores],
                'model': self._model_name,
                'num_sentences': len(scores),
            }
            return result

        except Exception as e:
            logger.error(f"COMET scoring failed: {e}")
            return {'error': str(e), 'comet_score': 0.0}

    def __repr__(self):
        return f"COMETScorer(available={self._available}, model={self._model_name})"
