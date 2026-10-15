"""
Terminology enforcement for seq2seq MT (NLLB / M2M100 / IndicTrans2).

The original "RAG" step prepended glossary lines (``bank = बैंक``) to the
model input. NLLB was never trained on that input format, so it translated
the glossary lines too and a regex had to strip the echo afterwards. This
module replaces that with techniques that fit an encoder-decoder MT model:

  none         plain translation (the ablation baseline)
  prompt       legacy glossary prefix + echo stripping; kept only so the
               ablation can compare against it
  inline       code-switched source: the English term is replaced by its
               glossary translation before translation ("I went to the
               बैंक"). Multilingual models tend to copy target-script tokens
               and can still inflect around them (Song et al., NAACL 2019,
               "Code-Switching for Enhancing NMT with Pre-Specified
               Translation"). No extra latency.
  placeholder  mask each term with an opaque token, translate, put the
               glossary translation back. Falls back to plain translation
               when the model drops or mangles a placeholder.
  constrained  lexically constrained beam search (HF ``force_words_ids``):
               the glossary translation is guaranteed to appear, at the cost
               of beam search and no inflection. Only the NLLB wrapper
               implements it.

Only glossary entries whose English term actually occurs in the source are
enforced. Retrieval proposes candidates; this module verifies them against
the source text, so a semantically similar but absent term is never injected.

Which strategy is best is an empirical question: run
``scripts/run_terminology_ablation.py`` and set TERMINOLOGY_STRATEGY to the
winner.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

STRATEGIES = ('none', 'prompt', 'inline', 'placeholder', 'constrained')
DEFAULT_STRATEGY = 'inline'


def default_strategy() -> str:
    value = (os.getenv('TERMINOLOGY_STRATEGY') or DEFAULT_STRATEGY).strip().lower()
    if value not in STRATEGIES:
        logger.warning(f'TERMINOLOGY_STRATEGY={value!r} is invalid; using {DEFAULT_STRATEGY}')
        return DEFAULT_STRATEGY
    return value


# ---------------------------------------------------------------------------
# Matching glossary terms in the source
# ---------------------------------------------------------------------------

@dataclass
class TermMatch:
    term: str           # glossary source term (English)
    translation: str    # glossary target term
    start: int          # span in the source text
    end: int
    surface: str        # text as it appears in the source


def _term_pattern(term: str) -> re.Pattern:
    words = [re.escape(w) for w in term.split()]
    # Whole-word, case-insensitive, flexible whitespace, optional English
    # plural on the last word ("bank" matches "banks").
    body = r'\s+'.join(words)
    return re.compile(rf'(?<!\w){body}(?:e?s)?(?!\w)', re.IGNORECASE)


def match_terms(text: str, terms: List[Dict]) -> List[TermMatch]:
    """
    Find glossary terms that occur in ``text``.

    Longer terms win over shorter overlapping ones ("machine learning" over
    "machine"); each glossary term is matched at most once per occurrence.
    Returns matches sorted by position.
    """
    seen = set()
    candidates = []
    for t in terms or []:
        term = (t.get('term') or '').strip()
        translation = (t.get('translation') or '').strip()
        if not term or not translation or term.lower() in seen:
            continue
        seen.add(term.lower())
        candidates.append((term, translation))
    candidates.sort(key=lambda c: -len(c[0]))

    taken: List[Tuple[int, int]] = []
    matches: List[TermMatch] = []
    for term, translation in candidates:
        for m in _term_pattern(term).finditer(text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            matches.append(TermMatch(term, translation, m.start(), m.end(), m.group(0)))
    matches.sort(key=lambda m: m.start)
    return matches


def _replace_spans(text: str, matches: List[TermMatch], replacements: List[str]) -> str:
    out, pos = [], 0
    for m, rep in zip(matches, replacements):
        out.append(text[pos:m.start])
        out.append(rep)
        pos = m.end
    out.append(text[pos:])
    return ''.join(out)


# ---------------------------------------------------------------------------
# Strategy helpers (pure functions, unit tested)
# ---------------------------------------------------------------------------

def build_prompt_context(matches: List[TermMatch]) -> str:
    """Legacy 'term = translation' prefix (the old RAG input format)."""
    return '\n'.join(f'{m.term} = {m.translation}' for m in matches)


def inline_source(text: str, matches: List[TermMatch]) -> str:
    """Code-switched source: each matched term replaced by its translation."""
    return _replace_spans(text, matches, [m.translation for m in matches])


_PLACEHOLDER_LETTERS = 'ABCDEFGHJKLMNPQRSTUVWXYZ'


def placeholder_token(i: int) -> str:
    # Letters only: digits may be localised to native numerals by the model.
    return 'ZXQ' + _PLACEHOLDER_LETTERS[i % len(_PLACEHOLDER_LETTERS)] * (1 + i // len(_PLACEHOLDER_LETTERS))


def mask_source(text: str, matches: List[TermMatch]) -> Tuple[str, Dict[str, str]]:
    tokens = [placeholder_token(i) for i in range(len(matches))]
    return _replace_spans(text, matches, tokens), {
        tok: m.translation for tok, m in zip(tokens, matches)}


def unmask(translation: str, mapping: Dict[str, str]) -> Tuple[str, bool]:
    """
    Restore placeholders. Returns (text, ok); ``ok`` is False when any
    placeholder is missing or duplicated, in which case the caller should
    fall back to a plain translation.
    """
    out, ok = translation, True
    for tok, target in mapping.items():
        pattern = re.compile(r'\s*'.join(map(re.escape, tok)), re.IGNORECASE)
        found = pattern.findall(out)
        if len(found) != 1:
            ok = False
        out = pattern.sub(target, out)
    return out, ok


def term_accuracy(hypothesis: str, targets: List[str]) -> Tuple[int, int]:
    """
    Terminology usage: how many required target terms appear in the output.

    Substring match, so agglutinated suffixes still count ("வங்கியில்"
    contains "வங்கி"); a changed stem does not. This is the exact-match term
    usage rate of Dinu et al. (ACL 2019).
    """
    hyp = hypothesis or ''
    hits = sum(1 for t in targets if t and t in hyp)
    return hits, len(targets)


# ---------------------------------------------------------------------------
# Applying a strategy around any translate function
# ---------------------------------------------------------------------------

@dataclass
class TerminologyInfo:
    strategy: str
    matched_terms: List[Dict] = field(default_factory=list)
    applied: bool = False          # the strategy actually changed decoding
    fallback: Optional[str] = None  # why we fell back to plain translation
    terms_found: int = 0
    terms_required: int = 0

    def to_dict(self) -> Dict:
        return {
            'strategy': self.strategy,
            'matched_terms': self.matched_terms,
            'applied': self.applied,
            'fallback': self.fallback,
            'term_accuracy': (round(self.terms_found / self.terms_required, 4)
                              if self.terms_required else None),
            'terms_found': self.terms_found,
            'terms_required': self.terms_required,
        }


def translate_with_terms(translate_fn: Callable, text: str, terms: List[Dict],
                         strategy: Optional[str] = None, **kwargs):
    """
    Run ``translate_fn`` with terminology enforced by ``strategy``.

    Args:
        translate_fn: ``fn(text, context=None, use_tm=True, **kwargs)`` that
            returns an object with ``.translated``/``.error``/``.timings``
            (a ``TranslationResult``).
        text: source text.
        terms: candidate glossary entries ({term, translation, ...}),
            typically the RAG retrieval output plus lexical glossary matches.
        strategy: one of STRATEGIES (default: TERMINOLOGY_STRATEGY env).

    Returns:
        (result, TerminologyInfo)
    """
    strategy = (strategy or default_strategy()).lower()
    if strategy not in STRATEGIES:
        raise ValueError(f'Unknown terminology strategy {strategy!r}; '
                         f'valid: {", ".join(STRATEGIES)}')

    matches = match_terms(text, terms) if strategy != 'none' else []
    info = TerminologyInfo(
        strategy=strategy,
        matched_terms=[{'term': m.term, 'translation': m.translation,
                        'surface': m.surface} for m in matches],
    )

    if not matches:
        result = translate_fn(text, **kwargs)
        return result, info

    # A terminology-constrained output must not be cached as the plain
    # translation of the source, and the modified inputs are not real source
    # sentences, so bypass the translation memory.
    kwargs = dict(kwargs, use_tm=False)

    if strategy == 'prompt':
        result = translate_fn(text, context=build_prompt_context(matches), **kwargs)
        info.applied = True
    elif strategy == 'inline':
        result = translate_fn(inline_source(text, matches), **kwargs)
        result.original = text
        info.applied = True
    elif strategy == 'placeholder':
        masked, mapping = mask_source(text, matches)
        result = translate_fn(masked, **kwargs)
        restored, ok = unmask(result.translated or '', mapping)
        if ok and not result.error:
            result.translated = restored
            info.applied = True
        else:
            info.fallback = 'placeholder lost or duplicated by the model'
            result = translate_fn(text, **kwargs)
        result.original = text
    elif strategy == 'constrained':
        beams = kwargs.pop('num_beams', None)
        result = translate_fn(text, force_words=[m.translation for m in matches],
                              num_beams=max(beams or 4, 2), **kwargs)
        if (result.timings or {}).get('constraints_applied'):
            info.applied = True
        else:
            info.fallback = ((result.timings or {}).get('constraints_error')
                             or 'model wrapper does not support constrained decoding')

    info.terms_found, info.terms_required = term_accuracy(
        result.translated, [m.translation for m in matches])
    return result, info
