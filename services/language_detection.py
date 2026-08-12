"""
Lightweight language detection based on Unicode scripts.

No heavy ML dependencies (no fasttext / langid). Detection works in two
stages:

1. SCRIPT DETECTION — count characters per Unicode block (Devanagari,
   Bengali, Gujarati, Tamil, Telugu, Kannada, Latin). The dominant script
   maps directly to a language for every script except Devanagari, which is
   shared by Hindi and Marathi. When Latin dominates but an Indic script has
   a substantial share (code-mixed Indic + English), the Indic script wins —
   so ``मुझे आज office जाना है।`` is detected as Hindi, not English.

2. DEVANAGARI DISAMBIGUATION (Hindi vs Marathi) — a trainable model over
   **words + 2-5-char word suffixes** (multinomial Naive Bayes, Laplace
   smoothed) plus a small set of language-marker characters (Marathi: ळ ऱ ॅ;
   Hindi: ऋ औ) injected as pseudo-features. Words and suffixes capture the
   function-word and inflectional morphology that separates the two
   languages (Hindi में/है/और vs Marathi मध्ये/आहे/आणि), which the old
   character-unigram/bigram model could not learn from small training data.
   Suffixes are kept out to 4-5 characters so verb and nominal endings
   (हूँ/हैं vs आहे/आहेत, -साठी vs -के लिए) contribute their full shape.
   A character n-gram (unigram+bigram) model is retained as a defensive
   fallback: it fires only for degenerate input where no whitespace token
   yields Devanagari characters, and guarantees identical behaviour to the
   historical detector on such input. It deliberately does NOT blend into
   the word verdict — experiments showed the char signal added noise to
   the word model's decisions.

The models are trainable so the benchmark can train on a split of the data
and report honest generalization numbers (see
app/evaluation/language_detection.py).

Limitations (documented for the paper):
- Only the 8 supported languages (english + 7 Indic). Other Latin-script
  languages are reported as 'english'.
- Romanized (Latin-script) Hindi/Marathi input is NOT detected as Indic —
  that requires a transliteration stage (out of scope).
- Very short / heavily code-mixed text can mislead script counts.

Usage:
    detector = LanguageDetector()
    detector.detect('भारत एक विविध देश है।')      # 'hindi'
    detector.detect_with_scores('...')             # full result dict
"""

import logging
import math
import unicodedata
from collections import Counter
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unicode script ranges (inclusive)
# ---------------------------------------------------------------------------

SCRIPT_RANGES = {
    'devanagari': [(0x0900, 0x097F), (0xA8E0, 0xA8FF), (0x1CD0, 0x1CFF)],
    'bengali': [(0x0980, 0x09FF)],
    'gujarati': [(0x0A80, 0x0AFF)],
    'tamil': [(0x0B80, 0x0BFF)],
    'telugu': [(0x0C00, 0x0C7F)],
    'kannada': [(0x0C80, 0x0CFF)],
    'latin': [(0x0041, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF)],
}

# Script -> language (Devanagari resolved separately)
SCRIPT_TO_LANGUAGE = {
    'bengali': 'bengali',
    'gujarati': 'gujarati',
    'tamil': 'tamil',
    'telugu': 'telugu',
    'kannada': 'kannada',
    'latin': 'english',
}

# Devanagari-family languages the model can separate
DEVANAGARI_LANGS = ('hindi', 'marathi')

SUPPORTED_LANGUAGES = (
    'english', 'hindi', 'marathi', 'bengali',
    'gujarati', 'tamil', 'telugu', 'kannada',
)

# Code-mixed cascade: when Latin is the dominant script, an Indic script is
# preferred if it reaches at least this share of the Latin count AND this
# many characters. Keeps English (with an occasional Indic character) as
# English while routing Indic-language sentences with English words to the
# Indic detector.
CODE_MIXED_INDIC_RATIO = 0.3
CODE_MIXED_MIN_INDIC_CHARS = 3

# Language-marker characters inside Devanagari. ळ (Llla), ऱ (Rra) and ॅ
# (candra E) are essentially Marathi-only; ऋ and ौ are (in this corpus)
# Hindi-leaning. These are injected as pseudo-features so a single
# occurrence is strong evidence — the same prior a human uses.
MARATHI_MARKER_CHARS = frozenset('ळऱॅ')
HINDI_MARKER_CHARS = frozenset('ऋौ')


def _in_ranges(char: str, ranges) -> bool:
    code = ord(char)
    return any(lo <= code <= hi for lo, hi in ranges)


def detect_script(text: str) -> Optional[str]:
    """Return the dominant script of the text (or None if no script chars).

    Latin-dominant text that still contains a substantial Indic script
    (code-mixed) is reported as the Indic script, so ``video call वर``
    counts as Devanagari, not English.
    """
    counts: Counter = Counter()
    for ch in text:
        for script, ranges in SCRIPT_RANGES.items():
            if _in_ranges(ch, ranges):
                counts[script] += 1
                break
    if not counts:
        return None
    top, top_count = counts.most_common(1)[0]
    if top == 'latin':
        # Among the Indic scripts that clear the code-mixed threshold, prefer
        # the one with the largest share (dict order would bias Devanagari).
        candidates = {
            script: n for script, n in counts.items()
            if script != 'latin'
            and n >= CODE_MIXED_MIN_INDIC_CHARS
            and n >= CODE_MIXED_INDIC_RATIO * top_count
        }
        if candidates:
            return max(candidates, key=candidates.get)
    return top


def scripts_present(text: str) -> set:
    """
    Return the set of scripts that occur anywhere in the text.

    Unlike detect_script (which returns only the *dominant* script), this
    reports every script that appears — essential for code-mixed text where
    English words are embedded inside an Indic sentence and may outnumber
    the Indic characters.
    """
    present = set()
    for ch in text:
        for script, ranges in SCRIPT_RANGES.items():
            if _in_ranges(ch, ranges):
                present.add(script)
                break
    return present


def _normalize_script_chars(text: str, script_ranges) -> str:
    """NFC-normalize and keep only characters of the given script."""
    out = []
    for ch in unicodedata.normalize('NFC', text):
        if _in_ranges(ch, script_ranges):
            out.append(ch)
    return ''.join(out)


class LanguageDetector:
    """
    Script-based language detector with a trainable Devanagari model.

    The Hindi/Marathi model is a multinomial Naive Bayes over words +
    2-5-char word suffixes (Laplace smoothed) plus language-marker
    characters, with a character unigram+bigram fallback. By default it
    trains on the full curated reference dataset; pass `train_texts` to
    train on a custom split (used by the benchmark to avoid circular
    evaluation).
    """

    def __init__(self, train_texts: Optional[Dict[str, List[str]]] = None):
        self._char_model: Optional[Dict[str, Dict[str, float]]] = None
        self._word_model: Optional[Dict[str, Dict[str, float]]] = None
        self._train(train_texts)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _default_train_texts(self) -> Dict[str, List[str]]:
        from app.evaluation.datasets import CURATED_ITEMS
        texts = {lang: [] for lang in DEVANAGARI_LANGS}
        for item in CURATED_ITEMS:
            for lang in DEVANAGARI_LANGS:
                ref = item[3].get(lang, '')
                if ref:
                    texts[lang].append(ref)
        return texts

    @staticmethod
    def _word_features(norm_text: str, raw_text: str) -> List[str]:
        """
        Words + 2-5-char suffixes + language-marker pseudo-features.

        `norm_text` is the Devanagari-only NFC-normalized form; `raw_text`
        is the original string (split on whitespace so words stay separate —
        the normalized form has its spaces stripped).
        """
        feats: List[str] = []
        for tok in raw_text.split():
            wn = _normalize_script_chars(tok, SCRIPT_RANGES['devanagari'])
            if not wn:
                continue
            feats.append(wn)
            # NOTE: for a word shorter than the suffix length the suffix
            # equals the word — these duplicated features intentionally
            # upweight short function words (में, है, की) and, for 4-5 char
            # suffixes, longer inflectional endings (हूँ/हैं, आहे/आहेत), which
            # are the strongest hi/mr discriminators.
            feats.append(wn[-5:] if len(wn) >= 5 else wn)
            feats.append(wn[-4:] if len(wn) >= 4 else wn)
            feats.append(wn[-3:] if len(wn) >= 3 else wn)
            feats.append(wn[-2:] if len(wn) >= 2 else wn)
        for c in MARATHI_MARKER_CHARS:
            if c in norm_text:
                feats.append(f'MRCH:{c}')
        for c in HINDI_MARKER_CHARS:
            if c in norm_text:
                feats.append(f'HICH:{c}')
        return feats

    def _train(self, train_texts: Optional[Dict[str, List[str]]]):
        train_texts = train_texts or self._default_train_texts()
        self._char_model = {}
        self._word_model = {}
        for lang in DEVANAGARI_LANGS:
            texts = train_texts.get(lang) or []
            char_counts: Counter = Counter()
            word_counts: Counter = Counter()
            for text in texts:
                norm = _normalize_script_chars(text, SCRIPT_RANGES['devanagari'])
                char_counts.update(norm)                     # unigrams
                char_counts.update(norm[i:i + 2]             # bigrams
                                   for i in range(len(norm) - 1))
                word_counts.update(self._word_features(norm, text))
            self._char_model[lang] = {
                'counts': char_counts,
                'total': sum(char_counts.values()),
                'vocab': len(char_counts),
            }
            self._word_model[lang] = {
                'counts': word_counts,
                'total': sum(word_counts.values()),
                'vocab': len(word_counts),
            }
        logger.info('LanguageDetector Devanagari models trained for '
                    f'{list(self._word_model.keys())} '
                    '(words + 2-5-char suffixes, char fallback)')

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def supported_languages(self):
        return list(SUPPORTED_LANGUAGES)

    def detect(self, text: str) -> str:
        """Detect the language of `text`; returns a language name or 'unknown'."""
        result = self.detect_with_scores(text)
        return result['detected_language']

    def detect_with_scores(self, text: str) -> Dict:
        """
        Full detection result.

        Returns:
            dict with detected_language, script, confidence (0-1), and
            per-language scores (higher = more likely).
        """
        if not text or not text.strip():
            return self._unknown_result(text, 'empty text')

        script = detect_script(text)
        if script is None:
            return self._unknown_result(text, 'no script characters found')

        language = SCRIPT_TO_LANGUAGE.get(script)
        if language is not None:
            return {
                'detected_language': language,
                'script': script,
                'confidence': 1.0,
                'scores': {language: 1.0},
                'note': 'single-script mapping',
            }

        # Devanagari -> Hindi vs Marathi via the word/suffix model
        if script == 'devanagari':
            scores = self._score_devanagari(text)
            best = max(scores, key=scores.get)
            return {
                'detected_language': best,
                'script': script,
                'confidence': round(scores[best], 4),
                'scores': scores,
                'note': 'devanagari disambiguated by word/suffix model',
            }

        return self._unknown_result(text, f'unsupported script: {script}')

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _laplace_logp(counts: Counter, total: int, vocab: int, feat: str,
                      alpha: float = 1.0) -> float:
        return math.log((counts.get(feat, 0) + alpha) / (total + alpha * vocab))

    @staticmethod
    def _softmax(log_likelihoods: Dict[str, float]) -> Dict[str, float]:
        max_ll = max(log_likelihoods.values())
        exps = {lang: math.exp(ll - max_ll)
                for lang, ll in log_likelihoods.items()}
        total = sum(exps.values())
        return {lang: round(prob / total, 4) for lang, prob in exps.items()}

    def _score_devanagari(self, text: str) -> Dict[str, float]:
        """Return softmax probabilities over the Devanagari languages."""
        norm = _normalize_script_chars(text, SCRIPT_RANGES['devanagari'])
        if not norm:
            return {lang: 0.0 for lang in DEVANAGARI_LANGS}

        # Primary: words + suffixes (morphology is what separates hi/mr).
        # norm non-empty implies at least one Devanagari-bearing token, so
        # word_feats is always non-empty here; the char model below is a
        # defensive holdover for degenerate input.
        word_feats = self._word_features(norm, text)
        if word_feats:
            log_likelihoods = {}
            for lang, model in (self._word_model or {}).items():
                counts, total, vocab = model['counts'], model['total'], model['vocab']
                logp = sum(self._laplace_logp(counts, total, vocab, f)
                           for f in word_feats)
                log_likelihoods[lang] = logp / max(len(word_feats), 1)
            return self._softmax(log_likelihoods)

        # Fallback: character n-grams (text with no known words/suffixes).
        ngrams = list(norm) + [norm[i:i + 2] for i in range(len(norm) - 1)]
        log_likelihoods = {}
        for lang, model in (self._char_model or {}).items():
            counts, total, vocab = model['counts'], model['total'], model['vocab']
            logp = sum(self._laplace_logp(counts, total, vocab, ng)
                       for ng in ngrams)
            log_likelihoods[lang] = logp / max(len(ngrams), 1)
        return self._softmax(log_likelihoods)

    @staticmethod
    def _unknown_result(text: str, reason: str) -> Dict:
        return {
            'detected_language': 'unknown',
            'script': None,
            'confidence': 0.0,
            'scores': {},
            'note': reason,
        }

    def __repr__(self):
        return (f'LanguageDetector(scripts={list(SCRIPT_RANGES.keys())}, '
                f'devanagari_model={list((self._word_model or {}).keys())})')
