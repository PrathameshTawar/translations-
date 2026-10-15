"""
Quality scoring for translations.
Measures: length ratio, term overlap, script consistency (when the target
language is known), sentence-end punctuation, word-order sanity (Devanagari
postposition/auxiliary placement), hallucination heuristic, and overall
confidence. All scores are in [0, 1].
"""

import logging

# Expected dominant Unicode script per supported language. Used to catch
# wrong-language output (e.g. German/Spanish text for an English->Hindi job).
TARGET_SCRIPT = {
    'english': 'latin',
    'hindi': 'devanagari',
    'marathi': 'devanagari',
    'bengali': 'bengali',
    'gujarati': 'gujarati',
    'tamil': 'tamil',
    'telugu': 'telugu',
    'kannada': 'kannada',
}

# Sentence-final punctuation marks that signal a complete, well-formed sentence.
_SENTENCE_ENDERS = '।.!?'
# Truncated outputs (…) are partial, not complete.
_ELLIPSIS = '…'

# ---------------------------------------------------------------------------
# Devanagari word-order sanity rules
#
# Hindi and Marathi are strictly head-final: postpositions (case markers like
# में/को/से) attach AFTER a noun phrase, and auxiliary verbs cluster at the
# clause end. Word-scrambled output (a common degenerate mode — and the exact
# construction used for the 'bad' QE benchmark samples) violates these
# constraints: it can place a postposition as the very first token, stack two
# postpositions back-to-back (को में), or put a postposition immediately after
# an auxiliary verb (सकते को). A human reader catches these instantly; these
# rules encode that same prior at zero model cost.
# ---------------------------------------------------------------------------

# Case markers / postpositions. Restricted to *unambiguous* case markers that
# can never open a sentence and never follow another postposition or an
# auxiliary verb. Temporal/locative adverbs (बाद, पहले, ऊपर, नीचे, बाहर,
# अंदर, आगे, पीछे, पास, बीच) and nouns (बारे, वजह, जगह) are deliberately
# excluded: they legitimately open sentences ("पहले मैं गया था।") and combine
# freely ("बाद में", "पहले से", "के बारे में") without being ungrammatical.
_DEVA_POSTPOSITIONS = frozenset(
    'का की के को में से पर तक लिए साथ द्वारा बिना ने हेतु मार्फत जरिए '
    'मुताबिक अनुसार खिलाफ बजाय सहित'.split()
)

# Auxiliary / light verbs. Never valid sentence-openers and never directly
# followed by a postposition.
_DEVA_AUXILIARIES = frozenset(
    'है हैं हूँ हो था थी थे थीं गया गई गए गयी किया किए कीं '
    'रहा रही रहे रहीं जाता जाती जाते सकता सकती सकते सका सकी सके सकें '
    'चाहिए होगा होगी होंगे रहेगा रहेगी रहेंगे करेंगे होता होती होते '
    'हुआ हुई हुए जाऊं जाएगा जाएगी होऊं होगे चाहूंगा चाहूंगी'.split()
)

# Valid compound postpositions (post + post) — must not be flagged. Kept as
# explicit two-token strings (a single '.split()' would break them apart) and
# limited to pairs whose BOTH tokens are still in _DEVA_POSTPOSITIONS.
_DEVA_COMPOUND_POSTPOSITIONS = frozenset((
    'के लिए', 'के साथ', 'के जरिए', 'के द्वारा', 'के अनुसार', 'के हेतु',
    'के बजाय', 'के खिलाफ', 'के मुताबिक',
    'की बजाय', 'में से',
))

# Word-order violations cap the overall score at this value (below the 0.80
# 'good' threshold, above the 'low' band) — see _word_order_violation.
_WORD_ORDER_CAP = 0.70


class QualityScorer:
    """Lightweight quality scorer that works without any ML model."""

    def score(
        self,
        source_text: str,
        translated_text: str,
        retrieved_terms: list = None,
        target_lang: str = None,
    ) -> dict:
        """
        Returns a quality dict:
        {
            overall_score: float,
            confidence: str,          # high / medium / low
            length_ratio: float,
            term_overlap: float,
            script_consistency: float,
            sentence_end: float,
            hallucination_detected: bool,
        }

        Args:
            source_text: The original text.
            translated_text: The model output.
            retrieved_terms: Optional RAG terminology terms (for term overlap).
            target_lang: Optional target language name ('hindi', ...). When
                         provided, the output's dominant script is checked
                         against the target's script; wrong-language output
                         is penalised heavily.
        """
        if not translated_text or not source_text:
            return self._empty()

        length_score = self._length_ratio(source_text, translated_text)
        term_overlap = self._term_overlap(
            source_text, translated_text, retrieved_terms or [])
        script_score = self._script_consistency(translated_text, target_lang)
        end_score = self._sentence_end_score(translated_text)
        hallucination = self._detect_hallucination(source_text, translated_text)
        word_order_suspicious = self._word_order_violation(translated_text)

        # Weighted overall score. Script + sentence-end catch the failure
        # modes the old length/term/hallucination trio could not see
        # (wrong language, truncated output, word-salad). The word-order
        # check catches word-salad that still ends in the sentence-final
        # mark (scrambled output that randomly keeps '।' last).
        overall = (
            0.30 * length_score
            + 0.20 * term_overlap
            + 0.20 * script_score
            + 0.15 * end_score
            + 0.15 * (0.0 if hallucination else 1.0)
        )
        # Degenerate outputs are never rated 'high' regardless of the other
        # signals (e.g. an echo can be length- and script-consistent).
        if hallucination:
            overall = min(overall, 0.45)
        # Word-order violations (scrambled/misordered output) are capped too,
        # so they never read as high-quality even with a preserved final mark.
        if word_order_suspicious:
            overall = min(overall, _WORD_ORDER_CAP)
        overall = round(min(max(overall, 0.0), 1.0), 4)

        if overall >= 0.75:
            confidence = 'high'
        elif overall >= 0.50:
            confidence = 'medium'
        else:
            confidence = 'low'

        return {
            'overall_score': overall,
            'confidence': confidence,
            'length_ratio': round(length_score, 4),
            'term_overlap': round(term_overlap, 4),
            'script_consistency': round(script_score, 4),
            'sentence_end': round(end_score, 4),
            'hallucination_detected': hallucination,
            'word_order_suspicious': word_order_suspicious,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _empty(self) -> dict:
        # An empty translation means translation failed, NOT that the model
        # hallucinated. Flag it as a failure so callers can detect it.
        return {
            'overall_score': 0.0,
            'confidence': 'low',
            'length_ratio': 0.0,
            'term_overlap': 0.0,
            'script_consistency': 0.0,
            'sentence_end': 0.0,
            'hallucination_detected': False,
            'word_order_suspicious': False,
        }

    def _length_ratio(self, source: str, target: str) -> float:
        """
        Word-count ratio score between source and translation.

        Ratios in [0.5, 1.0] are credited proportionally (heavy truncation is
        penalised hard: 0.4 ratio scores only 0.4), ratios in (1.0, 1.6] fall
        off gently, and gross inflation (>1.6) degrades steadily.
        """
        src_words = max(len(source.split()), 1)
        tgt_words = max(len(target.split()), 1)
        ratio = tgt_words / src_words

        if ratio < 0.5:
            return ratio / 0.5 * 0.5          # 0 at 0, 0.5 at 0.5
        if ratio <= 1.0:
            return 0.5 + (ratio - 0.5) * 1.0  # 0.5 at 0.5, 1.0 at 1.0
        if ratio <= 1.6:
            return 1.0 - (ratio - 1.0) * 0.4  # 1.0 at 1.0, 0.76 at 1.6
        return max(0.0, 0.76 - (ratio - 1.6) * 0.3)  # 0.76 at 1.6, ~0.34 at 3.0

    def _script_consistency(self, translated_text: str, target_lang: str) -> float:
        """
        1.0 if the output's dominant script matches the target language's
        script; 0.0 otherwise. Unknown target -> 1.0 (no penalty).
        """
        if not target_lang:
            return 1.0
        expected = TARGET_SCRIPT.get(target_lang.strip().lower())
        if not expected:
            return 1.0
        try:
            from app.services.language_detection import detect_script
        except ImportError:
            return 1.0
        actual = detect_script(translated_text)
        if actual is None:
            return 0.0
        return 1.0 if actual == expected else 0.0

    def _sentence_end_score(self, translated_text: str) -> float:
        """
        1.0 for a proper sentence-ending mark (। . ! ?), 0.3 for a truncated
        ellipsis, 0.0 otherwise. Word-salad (scrambled) output almost never
        ends in the sentence-final mark, so this separates it from natural
        translations at no model cost.
        """
        text = translated_text.strip()
        if not text:
            return 0.0
        if text[-1] in _SENTENCE_ENDERS:
            return 1.0
        if text[-1] == _ELLIPSIS:
            return 0.3
        return 0.0

    def _word_order_violation(self, translated_text: str) -> bool:
        """
        True when the Devanagari word order is ungrammatical: a postposition
        or auxiliary verb opens the sentence, two postpositions sit adjacent
        (outside the compound whitelist), or a postposition directly follows
        an auxiliary verb. These patterns are diagnostic of scrambled/word-salad
        output — natural Hindi/Marathi almost never produce them.

        Only evaluated on Devanagari text (the rules are script-specific);
        Latin/other scripts return False.
        """
        try:
            from app.services.language_detection import (
                _normalize_script_chars, detect_script, SCRIPT_RANGES,
            )
        except ImportError:
            return False
        text = (translated_text or '').strip()
        if not text:
            return False
        if detect_script(text) != 'devanagari':
            return False

        norm = _normalize_script_chars(text, SCRIPT_RANGES['devanagari'])
        if not norm:
            return False

        words = []
        for tok in text.split():
            wn = _normalize_script_chars(tok, SCRIPT_RANGES['devanagari'])
            # Drop Devanagari sentence marks (। / ॥ live inside the script
            # block) so token comparison is punctuation-clean.
            wn = wn.rstrip('।॥')
            if wn:
                words.append(wn)
        if not words:
            return False

        # (1) A postposition or auxiliary verb can never open a sentence.
        if words[0] in _DEVA_POSTPOSITIONS or words[0] in _DEVA_AUXILIARIES:
            return True

        # (2) Adjacent postpositions / postposition after an auxiliary verb.
        for i in range(len(words) - 1):
            pair = f'{words[i]} {words[i + 1]}'
            if pair in _DEVA_COMPOUND_POSTPOSITIONS:
                continue
            a_post = words[i] in _DEVA_POSTPOSITIONS
            b_post = words[i + 1] in _DEVA_POSTPOSITIONS
            if a_post and b_post:
                return True
            if words[i] in _DEVA_AUXILIARIES and b_post:
                return True
        return False

    def _term_overlap(self, source: str, translated: str, retrieved_terms: list) -> float:
        """
        Checks how many retrieved term translations appear in the translated output.
        If no retrieved terms, returns 0.7 (neutral/unknown).
        """
        if not retrieved_terms:
            return 0.70

        hits = 0
        translated_lower = translated.lower()
        for term in retrieved_terms:
            translation_val = (term.get('translation') or term.get('context') or '').lower()
            if translation_val and translation_val in translated_lower:
                hits += 1

        return hits / len(retrieved_terms)

    def _detect_hallucination(self, source: str, translated: str) -> bool:
        """
        Simple heuristic: if translation is way longer than source (>4x words)
        or is identical to source (no translation occurred), flag it.
        """
        src_words = len(source.split())
        tgt_words = len(translated.split())

        if tgt_words > src_words * 4:
            return True
        if translated.strip() == source.strip():
            return True
        return False


# ---------------------------------------------------------------------------
# Neural QE (COMETkiwi) — reference-free quality estimation
# ---------------------------------------------------------------------------

# Default COMETkiwi reference-free model. WMT22 da-variant is a strong QE
# model for Indic languages and the same one already referenced by the
# evaluation framework's COMETScorer.
COMETKIWI_MODEL_NAME = 'Unbabel/wmt22-cometkiwi-da'

# Module-level lazy singleton so repeated scoring (benchmarks, batch calls)
# loads the ~2.7GB model exactly once per process.
_COMETKIWI_MODEL = None
_COMETKIWI_AVAILABLE = None  # None = not probed yet, False = probed & missing


def cometkiwi_raw_to_score(raw: float) -> float:
    """
    Map a raw COMETkiwi score to [0, 1].

    WMT22 da-models are trained on z-scored human labels so raw scores sit
    roughly in [-1, 1]: 0 is the 'average quality' boundary. Mapping
    (raw + 1) / 2 puts that boundary at 0.5 and clamps the extremes.
    """
    return round(min(max((raw + 1.0) / 2.0, 0.0), 1.0), 4)


def _load_cometkiwi_model():
    """Lazily download + load the COMETkiwi model (module singleton)."""
    global _COMETKIWI_MODEL, _COMETKIWI_AVAILABLE
    if _COMETKIWI_MODEL is not None:
        return _COMETKIWI_MODEL
    if _COMETKIWI_AVAILABLE is False:
        return None
    try:
        from comet import download_model, load_from_checkpoint  # noqa: F401
        model_path = download_model(COMETKIWI_MODEL_NAME)
        _COMETKIWI_MODEL = load_from_checkpoint(model_path)
        _COMETKIWI_AVAILABLE = True
    except Exception as exc:  # pragma: no cover - env dependent
        _COMETKIWI_AVAILABLE = False
        logging.getLogger(__name__).error(
            f'COMETkiwi model could not be loaded: {exc}')
        return None
    return _COMETKIWI_MODEL


def reset_cometkiwi_singleton():
    """Drop the cached model (used by tests)."""
    global _COMETKIWI_MODEL, _COMETKIWI_AVAILABLE
    _COMETKIWI_MODEL = None
    _COMETKIWI_AVAILABLE = None


class COMETkiwiQualityScorer:
    """
    Reference-free neural QE using Unbabel's COMETkiwi.

    Scores (source, translation) pairs without any reference translation.
    Same result shape as QualityScorer.score so evaluation code can swap
    engines transparently. Falls back gracefully when `unbabel-comet` is
    not installed or the model cannot be downloaded.
    """

    # COMETkiwi is a fine-grained model: translations scoring above the
    # neutral boundary (raw 0) are 'acceptable'. Exposed so callers can
    # recalibrate without touching the class.
    DEFAULT_THRESHOLD = 0.5

    def __init__(self, model_name: str = COMETKIWI_MODEL_NAME,
                 batch_size: int = 16):
        self._model_name = model_name
        self.batch_size = batch_size

    @property
    def available(self) -> bool:
        return _load_cometkiwi_model() is not None

    def score(self, source_text: str, translated_text: str,
              retrieved_terms: list = None, target_lang: str = None) -> dict:
        """Score one (source, translation) pair; same dict shape as QualityScorer."""
        result = self.score_batch([source_text], [translated_text])
        if result.get('error') or not result.get('results'):
            return result  # graceful degradation — caller sees the error
        return result['results'][0]

    def score_batch(self, sources: list, translations: list) -> dict:
        """
        Score N pairs in one model call (much faster than per-item predict).

        Returns:
            dict with 'results' (per-item dicts matching QualityScorer.shape)
            and 'model' name; or {'error': ..., 'results': []} on failure.
        """
        pairs = [(s, t) for s, t in zip(sources, translations)]
        if not pairs:
            return {'error': 'no pairs', 'results': [], 'model': self._model_name}

        model = _load_cometkiwi_model()
        if model is None:
            return {'error': 'unbabel-comet not available', 'results': [],
                    'model': self._model_name}

        try:
            data = [{'src': src or '', 'mt': mt or ''} for src, mt in pairs]
            output = model.predict(data, batch_size=self.batch_size, gpus=0)
            raw_scores = output['scores'] if isinstance(output, dict) else list(output)

            results = []
            for src, mt, raw in zip(sources, translations, raw_scores):
                score_01 = cometkiwi_raw_to_score(raw)
                results.append({
                    'overall_score': score_01,
                    'confidence': self._confidence(score_01),
                    'raw_comet_score': round(float(raw), 4),
                    'hallucination_detected': score_01 < self.DEFAULT_THRESHOLD,
                })
            return {'results': results, 'model': self._model_name,
                    'mean_score': round(
                        sum(r['overall_score'] for r in results) / len(results), 4)}
        except Exception as exc:
            logging.getLogger(__name__).error(f'COMETkiwi scoring failed: {exc}')
            return {'error': str(exc), 'results': [], 'model': self._model_name}

    @staticmethod
    def _confidence(score_01: float) -> str:
        if score_01 >= 0.75:
            return 'high'
        if score_01 >= 0.50:
            return 'medium'
        return 'low'
