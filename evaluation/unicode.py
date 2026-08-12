"""
Unicode & script-handling test suite.

Because the entire project revolves around Indic scripts, this is mandatory.
Each case checks, against a Unicode-sensitive input:

  1. The input passes the request validator (correct language codes).
  2. NFC normalization is idempotent (normalize(NFC, x) is stable).
  3. NFD-decomposed input normalizes to the same NFC form (no information
     loss from combining marks being split).
  4. If a translate_fn is provided, translation succeeds without crashing
     and returns non-empty output.

Covers: combining marks, matras, ZWJ/ZWNJ, emoji, mixed scripts, Indic
numerals, currency symbols, punctuation and Latin diacritics.

Usage (from translator-backend/):
    python -m app.evaluation.unicode
"""

import logging
import unicodedata

logger = logging.getLogger(__name__)

# Script-name shortcuts for the report
SCRIPT_NAMES = {
    'Devanagari': 'Devanagari (hi/mr)',
    'Bengali': 'Bengali (bn)',
    'Gujarati': 'Gujarati (gu)',
    'Tamil': 'Tamil (ta)',
    'Telugu': 'Telugu (te)',
    'Kannada': 'Kannada (kn)',
}

# Each case: (name, script, text) — validated against hi/mr target.
UNICODE_CASES = [
    ('devanagari with matras', 'Devanagari', 'विद्यालय में अध्यापक पढ़ाते हैं।'),
    ('devanagari conjuncts', 'Devanagari', 'ज्ञान और क्षमता दोनों आवश्यक हैं।'),
    ('devanagari nukta', 'Devanagari', 'फ़िल्म और क़ानून पर चर्चा हुई।'),
    ('bengali script', 'Bengali', 'বাংলাদেশের রাজধানী ঢাকা শহর।'),
    ('bengali matras', 'Bengali', 'ভালোবাসা এবং শিক্ষা জীবনে গুরুত্বপূর্ণ।'),
    ('gujarati script', 'Gujarati', 'ગુજરાત રાજ્ય પશ્ચિમ ભારતમાં આવેલું છે.'),
    ('tamil script', 'Tamil', 'தமிழ்நாட்டின் தலைநகரம் சென்னை.'),
    ('tamil grantha letters', 'Tamil', 'ஜனவரி மாதத்தில் பள்ளி திறக்கும்.'),
    ('telugu matras', 'Telugu', 'తెలుగు భాష ఆంధ్రప్రదేశ్‌లో మాట్లాడతారు.'),
    ('kannada script', 'Kannada', 'ಕನ್ನಡ ನಾಡು ಮತ್ತು ನುಡಿ ಸುಂದರವಾಗಿದೆ.'),
    ('zero-width joiner (ZWJ)', 'Devanagari', 'अभिनव\u200dप्रयोग सफल रहा।'),
    ('zero-width non-joiner (ZWNJ)', 'Devanagari', 'उच्चारण\u200c में सावधानी रखें।'),
    ('emoji mixed', 'Mixed', 'मुझे 🌸 और 🚀 बहुत पसंद हैं 😊'),
    ('mixed scripts', 'Mixed', 'English मिश्रण தமிழ் বাংলা ಕನ್ನಡ తెలుగు'),
    ('indic numerals', 'Devanagari', 'मूल्य १२३४५६७८९० रुपये है।'),
    ('latin diacritics', 'Latin', 'The café résumé mentioned naïve fiancés.'),
    ('currency symbols', 'Mixed', 'कीमत ₹100 + $50 + €30 है।'),
    ('punctuation heavy', 'Devanagari', 'क्या — तुमने… “यह” देखा?!'),
    ('nested quotes', 'Mixed', 'He said: “नमस्ते दुनिया” और फिर चला गया।'),
    # NFD-decomposed Devanagari (क ि स ा न ् + combining marks)
    ('nfd decomposition', 'Devanagari', '\u0915\u093f\u0938\u093e\u0928\u094d'),
]


class UnicodeTestSuite:
    """Runs Unicode robustness checks on Indic and mixed-script inputs."""

    def run(self, translate_fn=None, source_lang='hindi', target_lang='english') -> dict:
        """
        Run all Unicode cases.

        Args:
            translate_fn: Optional callable (text, source_lang, target_lang) -> str.
                          If provided, each case is also translated and must
                          succeed without crashing.
            source_lang / target_lang: Language pair used for the optional
                          translation check.

        Returns:
            dict: {total, passed, pass_rate, cases: [{name, script, passed, checks, detail}]}
        """
        from app.utils.validators import validate_translation_request

        results = []
        for name, script, text in UNICODE_CASES:
            checks = {}
            detail = None

            # 1. Validator must accept (text is well within length limits)
            ok, err = validate_translation_request({
                'text': text, 'source_lang': source_lang, 'target_lang': target_lang,
            })
            checks['validator_accepts'] = ok and err is None
            if not checks['validator_accepts']:
                detail = f'validator rejected: {err}'

            # 2. NFC idempotence
            nfc = unicodedata.normalize('NFC', text)
            checks['nfc_idempotent'] = unicodedata.normalize('NFC', nfc) == nfc

            # 3. NFD -> NFC round-trip preserves content
            nfd = unicodedata.normalize('NFD', text)
            checks['nfd_to_nfc_stable'] = unicodedata.normalize('NFC', nfd) == nfc

            # 4. Optional translation must not crash and must return output
            if translate_fn is not None:
                try:
                    out = translate_fn(text, source_lang=source_lang, target_lang=target_lang)
                    checks['translation_ok'] = bool(out and out.strip())
                    if not checks['translation_ok']:
                        detail = 'translation returned empty output'
                except Exception as e:
                    checks['translation_ok'] = False
                    detail = f'translation raised: {e}'

            passed = all(checks.values())
            results.append({
                'name': name,
                'script': script,
                'passed': passed,
                'checks': checks,
                'detail': detail,
            })

        passed_count = sum(1 for r in results if r['passed'])
        return {
            'total': len(results),
            'passed': passed_count,
            'failed': len(results) - passed_count,
            'pass_rate': round(passed_count / max(len(results), 1) * 100, 2),
            'cases': results,
        }


def run_unicode_tests(translate_fn=None) -> dict:
    """Module-level convenience entry point."""
    return UnicodeTestSuite().run(translate_fn=translate_fn)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run Unicode/script robustness tests')
    parser.add_argument('--mock', action='store_true',
                        help='Run translation checks with an echo stub (fast)')
    args = parser.parse_args()

    from app.evaluation.functional import EchoTranslator

    stub = None
    if args.mock:
        stub = lambda text, source_lang, target_lang: (  # noqa: E731
            EchoTranslator().translate(text, source_lang, target_lang)['translated']
        )

    result = run_unicode_tests(translate_fn=stub)
    print(f"\nUnicode tests: {result['passed']}/{result['total']} passed "
          f"({result['pass_rate']}%)")
    for case in result['cases']:
        status = 'PASS' if case['passed'] else 'FAIL'
        print(f"  [{status}] {case['name']} ({case['script']})")
        if not case['passed']:
            failed = {k: v for k, v in case['checks'].items() if not v}
            print(f"         -> failed checks: {failed}; {case['detail'] or ''}")
