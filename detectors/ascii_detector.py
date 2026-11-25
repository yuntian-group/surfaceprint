import pickle
import re
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator, List, Sequence

import numpy as np
import tqdm

from detectors.base_detector import BaseDetector

# Match a fenced code block (```...``` or ~~~...~~~) including newlines
_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~")
# Match an inline `...` span (single line only)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_WORD_RE = re.compile(r"\S+")


def _strip_code(text: str) -> str:
    """Remove fenced code blocks and inline code spans.

    Used for Group A (prose-focused) features so that typography inside
    code does not pollute the prose typography measurement.
    """
    text = _CODE_FENCE_RE.sub("", text)
    text = _INLINE_CODE_RE.sub("", text)
    return text


def _safe_ratio(num: int, denom: int, default: float = 0.5) -> float:
    """Compute num / denom or default when denom == 0."""
    return num / denom if denom > 0 else default


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _line_count(text: str) -> int:
    return len(text.splitlines()) or (1 if text else 0)


def _per_kb(count: int, text: str) -> float:
    if not text:
        return 0.0
    return count / max(len(text.encode("utf-8")) / 1024.0, 1e-9)


def _extract_fenced_code(text: str) -> List[str]:
    blocks: List[str] = []
    for match in _CODE_FENCE_RE.finditer(text):
        block = match.group(0)
        lines = block.splitlines()
        if len(lines) >= 2:
            blocks.append("\n".join(lines[1:-1]))
    return blocks


_INDENTED_CODE_RE = re.compile(r"(?:^(?: {4}|\t).*(?:\n|$))+", flags=re.M)


def _extract_indented_code(text: str) -> List[str]:
    return [
        m.group(0) for m in _INDENTED_CODE_RE.finditer(_CODE_FENCE_RE.sub("", text))
    ]


def _code_blob(text: str) -> str:
    blocks = _extract_fenced_code(text) + _extract_indented_code(text)
    return "\n".join(blocks)


_HAS_LATEX_RE = re.compile(r"\\[a-zA-Z]+|\$[^$\n]+?\$|\\begin\{")


def _has_latex(text: str) -> bool:
    """Quick check for the presence of LaTeX markers in the text."""
    return bool(_HAS_LATEX_RE.search(text))


# ============================================================================
# Group A — Universal punctuation (22 features)
# ============================================================================


# Emoji / pictographic symbol codepoints used by the emoji-density features.
# Covers the main Unicode emoji blocks plus the older pictographic
# Miscellaneous Symbols and Dingbats blocks that LLM outputs frequently
# repurpose as decorative bullets (e.g. ✓ ✨ ★ ⚡ ⚠).
_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001faff"
    "\U0001f1e6-\U0001f1ff"
    "\u2600-\u26ff"
    "\u2700-\u27bf"
    "]"
)


# A1 ---------------------------------------------------------------------------
def f_apostrophe_curly_rate(text: str) -> float:
    """Share of curly (U+2019) among all apostrophes."""
    prose = _strip_code(text)
    curly = prose.count("\u2019")
    straight = prose.count("'")
    return _safe_ratio(curly, curly + straight)


# A2 ---------------------------------------------------------------------------
_SINGLE_CURLY_PAIR_RE = re.compile(r"\u2018[^\u2018\u2019\n]+\u2019")
_SINGLE_STRAIGHT_PAIR_RE = re.compile(r"(?<![\w'])'([^'\n]{1,})'(?![\w'])")


def f_single_quote_pair_unicode_rate(text: str) -> float:
    """Share of Unicode pairs (U+2018..U+2019) among single-quoted spans."""
    prose = _strip_code(text)
    curly = len(_SINGLE_CURLY_PAIR_RE.findall(prose))
    straight = len(_SINGLE_STRAIGHT_PAIR_RE.findall(prose))
    return _safe_ratio(curly, curly + straight)


# A3 ---------------------------------------------------------------------------
_DOUBLE_CURLY_PAIR_RE = re.compile(r"\u201C[^\u201C\u201D\n]+\u201D")
_DOUBLE_STRAIGHT_PAIR_RE = re.compile(r'"([^"\n]+)"')


def f_double_quote_pair_unicode_rate(text: str) -> float:
    """Share of Unicode pairs (U+201C..U+201D) among double-quoted spans."""
    prose = _strip_code(text)
    curly = len(_DOUBLE_CURLY_PAIR_RE.findall(prose))
    straight = len(_DOUBLE_STRAIGHT_PAIR_RE.findall(prose))
    return _safe_ratio(curly, curly + straight)


# A4 ---------------------------------------------------------------------------
def f_dash_emdash_rate(text: str) -> float:
    """Share of em dash (U+2014) among all dash forms."""
    prose = _strip_code(text)
    em = prose.count("\u2014")
    en = prose.count("\u2013")
    hyphen = prose.count("-")
    return _safe_ratio(em, em + en + hyphen)


# A5 ---------------------------------------------------------------------------
def f_dash_endash_rate(text: str) -> float:
    """Share of en dash (U+2013) among all dash forms."""
    prose = _strip_code(text)
    em = prose.count("\u2014")
    en = prose.count("\u2013")
    hyphen = prose.count("-")
    return _safe_ratio(en, em + en + hyphen)


# A6 ---------------------------------------------------------------------------
_ASCII_ELLIPSIS_RE = re.compile(r"(?<!\.)\.\.\.(?!\.)")


def f_ellipsis_unicode_rate(text: str) -> float:
    """Share of U+2026 among all ellipsis-positions."""
    prose = _strip_code(text)
    uni = prose.count("\u2026")
    ascii_three = len(_ASCII_ELLIPSIS_RE.findall(prose))
    return _safe_ratio(uni, uni + ascii_three)


# A7 ---------------------------------------------------------------------------
_INLINE_MATH_SPANS_RE = re.compile(r"(?<!\\)(?<!\$)\$([^$\n]+?)\$(?!\$)")


def f_minus_unicode_rate(text: str) -> float:
    """Within inline-math spans, share of U+2212 (real minus) over '-' / U+2212."""
    spans = _INLINE_MATH_SPANS_RE.findall(text)
    if not spans:
        return 0.5
    blob = "".join(spans)
    uni = blob.count("\u2212")
    hyphen = blob.count("-")
    return _safe_ratio(uni, uni + hyphen)


# A8 ---------------------------------------------------------------------------
def f_nbsp_rate_per_word(text: str) -> float:
    """Count of NBSP / NNBSP / thin-space per word."""
    n_words = _word_count(text)
    if n_words == 0:
        return 0.0
    count = text.count("\u00a0") + text.count("\u202f") + text.count("\u2009")
    return count / n_words


# A9 ---------------------------------------------------------------------------
_INVISIBLE_CHARS = "\u200b\u200c\u200d\u200e\u200f\u2060\ufeff"


def f_invisible_char_presence(text: str) -> float:
    """Binary: any zero-width / BOM / word-joiner present?"""
    return 1.0 if any(c in _INVISIBLE_CHARS for c in text) else 0.0


# A10 --------------------------------------------------------------------------
_FRENCH_PUNCT_RE = re.compile(r"[;:!?]")
_SPACE_BEFORE_PUNCT_RE = re.compile(r" [;:!?]")


def f_french_spacing_rate(text: str) -> float:
    """Share of space-before-[;:!?] among all such punctuation marks."""
    prose = _strip_code(text)
    total = len(_FRENCH_PUNCT_RE.findall(prose))
    spaced = len(_SPACE_BEFORE_PUNCT_RE.findall(prose))
    return _safe_ratio(spaced, total, default=0.0)


# A11 --------------------------------------------------------------------------
def f_comma_fullwidth_rate(text: str) -> float:
    """Share of fullwidth (CJK) comma among all commas."""
    full = text.count("\uff0c")
    half = text.count(",")
    return _safe_ratio(full, full + half, default=0.0)


# A12 --------------------------------------------------------------------------
_BULLET_RE = re.compile(r"^[ \t]*([*\-+•])\s+", flags=re.M)


def f_bullet_star_rate(text: str) -> float:
    """Share of '*' among line-leading bullet markers."""
    bullets = _BULLET_RE.findall(text)
    if not bullets:
        return 0.5
    return bullets.count("*") / len(bullets)


# A13 --------------------------------------------------------------------------
def f_quote_unicode_rate(text: str) -> float:
    """Share of curly quote codepoints among all straight/curly quote marks."""
    prose = _strip_code(text)
    curly = sum(prose.count(c) for c in "\u2018\u2019\u201c\u201d")
    straight = prose.count("'") + prose.count('"')
    return _safe_ratio(curly, curly + straight)


# A14 --------------------------------------------------------------------------
def f_dash_unicode_rate(text: str) -> float:
    """Share of Unicode dash forms among Unicode dashes plus ASCII hyphens."""
    prose = _strip_code(text)
    unicode_dash = sum(
        prose.count(c) for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
    )
    hyphen = prose.count("-")
    return _safe_ratio(unicode_dash, unicode_dash + hyphen)


# A15 --------------------------------------------------------------------------
_SPECIAL_SPACE_CHARS = (
    "\u00a0\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u200b\u202f\u2060"
)


def f_special_space_rate_per_word(text: str) -> float:
    """Count of NBSP, thin/hair/zero-width/word-joiner spaces per word."""
    n_words = _word_count(text)
    if n_words == 0:
        return 0.0
    return sum(text.count(c) for c in _SPECIAL_SPACE_CHARS) / n_words


# A16 --------------------------------------------------------------------------
_VOWEL_CURLY_APOSTROPHE_RE = re.compile(r"[AEIOUaeiou]\u2019")
_VOWEL_STRAIGHT_APOSTROPHE_RE = re.compile(r"[AEIOUaeiou]'")


def f_post_vowel_apostrophe_curly_rate(text: str) -> float:
    """Curly apostrophe share after vowels, a rough contraction context."""
    prose = _strip_code(text)
    curly = len(_VOWEL_CURLY_APOSTROPHE_RE.findall(prose))
    straight = len(_VOWEL_STRAIGHT_APOSTROPHE_RE.findall(prose))
    return _safe_ratio(curly, curly + straight)


# A17 --------------------------------------------------------------------------
_CONSONANT_CURLY_APOSTROPHE_RE = re.compile(r"[B-DF-HJ-NP-TV-Zb-df-hj-np-tv-z]\u2019")
_CONSONANT_STRAIGHT_APOSTROPHE_RE = re.compile(r"[B-DF-HJ-NP-TV-Zb-df-hj-np-tv-z]'")


def f_post_consonant_apostrophe_curly_rate(text: str) -> float:
    """Curly apostrophe share after consonants."""
    prose = _strip_code(text)
    curly = len(_CONSONANT_CURLY_APOSTROPHE_RE.findall(prose))
    straight = len(_CONSONANT_STRAIGHT_APOSTROPHE_RE.findall(prose))
    return _safe_ratio(curly, curly + straight)


# A18 --------------------------------------------------------------------------
_TRAILING_SPACE_RE = re.compile(r"[ \t]+$", flags=re.M)


def f_trailing_space_rate_per_line(text: str) -> float:
    """Share of lines ending with literal space or tab."""
    lines = _line_count(text)
    if lines == 0:
        return 0.0
    return len(_TRAILING_SPACE_RE.findall(text)) / lines


# A19 --------------------------------------------------------------------------
_PERIOD_DOUBLE_SPACE_RE = re.compile(r"\.(?:  +)(?=\S)")
_PERIOD_SPACE_RE = re.compile(r"\. +(?=\S)")


def f_double_space_after_period_rate(text: str) -> float:
    """Share of period-following spaces that use two or more spaces."""
    prose = _strip_code(text)
    total = len(_PERIOD_SPACE_RE.findall(prose))
    double = len(_PERIOD_DOUBLE_SPACE_RE.findall(prose))
    return _safe_ratio(double, total, default=0.0)


# A20 --------------------------------------------------------------------------
def f_blank_line_between_paragraphs_rate(text: str) -> float:
    """Blank-line separators per non-empty prose line."""
    nonempty = sum(1 for line in text.splitlines() if line.strip())
    if nonempty == 0:
        return 0.0
    return len(_BLANK_LINE_RE.findall(text)) / nonempty


# A21 --------------------------------------------------------------------------
def f_emoji_presence(text: str) -> float:
    """Binary: any emoji / pictographic decoration codepoint present?"""
    return 1.0 if _EMOJI_RE.search(text) else 0.0


# A22 --------------------------------------------------------------------------
def f_emoji_rate_per_word(text: str) -> float:
    """Count of emoji / pictographic decoration codepoints per word."""
    n_words = _word_count(text)
    if n_words == 0:
        return 0.0
    return len(_EMOJI_RE.findall(text)) / n_words


# ============================================================================
# Group C — Math and LaTeX surface choices
# Each fires only when _has_latex(text) is True; returns 0.5 otherwise.
# ============================================================================

# B1 ---------------------------------------------------------------------------
_TEX_QUOTE_DOUBLE_RE = re.compile(r"``[^`\n]{1,200}''")
_TEX_QUOTE_SINGLE_RE = re.compile(r"(?<!`)`(?!`)[^`\n]{1,200}'(?!')")
_LITERAL_QUOTE_PAIR_RE = re.compile(r'"[^"\n]{1,200}"|\u201C[^\u201D\n]{1,200}\u201D')


def f_tex_quote_convention_rate(text: str) -> float:
    """Share of TeX-convention ``...''/`...' pairs among all quoted spans."""
    if not _has_latex(text):
        return 0.5
    tex_style = len(_TEX_QUOTE_DOUBLE_RE.findall(text)) + len(
        _TEX_QUOTE_SINGLE_RE.findall(text)
    )
    literal = len(_LITERAL_QUOTE_PAIR_RE.findall(text))
    return _safe_ratio(tex_style, tex_style + literal)


# B2 ---------------------------------------------------------------------------
_EMDASH_LIGATURE_RE = re.compile(r"(?<=[A-Za-z0-9])---(?=[A-Za-z0-9])")


def f_emdash_ligature_rate(text: str) -> float:
    """Share of '---' TeX ligature among all em-dash positions."""
    if not _has_latex(text):
        return 0.5
    ligature = len(_EMDASH_LIGATURE_RE.findall(text))
    literal = text.count("\u2014")
    return _safe_ratio(ligature, ligature + literal)


# B3 ---------------------------------------------------------------------------
_ENDASH_LIGATURE_RE = re.compile(r"(?<=[A-Za-z0-9])--(?=[A-Za-z0-9])")


def f_endash_ligature_rate(text: str) -> float:
    """Share of '--' TeX ligature among all en-dash positions."""
    if not _has_latex(text):
        return 0.5
    ligature = len(_ENDASH_LIGATURE_RE.findall(text))
    literal = text.count("\u2013")
    return _safe_ratio(ligature, ligature + literal)


# B4 ---------------------------------------------------------------------------
_INLINE_DOLLAR_RE = _INLINE_MATH_SPANS_RE  # reuse
_INLINE_PAREN_RE = re.compile(r"\\\(([^)\n]+?)\\\)")


def f_inline_math_dollar_rate(text: str) -> float:
    """Share of $...$ among inline-math delimiters."""
    dollars = len(_INLINE_DOLLAR_RE.findall(text))
    parens = len(_INLINE_PAREN_RE.findall(text))
    if dollars + parens == 0:
        return 0.5
    return dollars / (dollars + parens)


# B5 ---------------------------------------------------------------------------
_DISPLAY_DOLLAR_RE = re.compile(r"\$\$([^$]+?)\$\$", flags=re.DOTALL)
_DISPLAY_BRACKET_RE = re.compile(r"\\\[([^\]]+?)\\\]", flags=re.DOTALL)
_DISPLAY_BEGIN_RE = re.compile(
    r"\\begin\{(?:equation|align|gather|displaymath|eqnarray|multline)\*?\}"
)


def f_display_math_dollar_rate(text: str) -> float:
    """Share of $$...$$ among display-math delimiters."""
    dollar = len(_DISPLAY_DOLLAR_RE.findall(text))
    bracket = len(_DISPLAY_BRACKET_RE.findall(text))
    begin = len(_DISPLAY_BEGIN_RE.findall(text))
    total = dollar + bracket + begin
    if total == 0:
        return 0.5
    return dollar / total


# B6 ---------------------------------------------------------------------------
_GREEK_LOWER_NAMES = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa "
    "lambda mu nu xi omicron pi rho sigma tau upsilon phi chi "
    "psi omega varepsilon vartheta varpi varphi varsigma varrho"
).split()
_GREEK_UPPER_NAMES = (
    "Alpha Beta Gamma Delta Epsilon Zeta Eta Theta Iota Kappa "
    "Lambda Mu Nu Xi Omicron Pi Rho Sigma Tau Upsilon Phi Chi Psi Omega"
).split()
_GREEK_LOWER_CHARS = "αβγδεζηθικλμνξοπρστυφχψω"
_GREEK_UPPER_CHARS = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
_GREEK_CMD_RE = re.compile(
    r"\\(?:" + "|".join(_GREEK_LOWER_NAMES + _GREEK_UPPER_NAMES) + r")\b"
)


def f_greek_command_rate(text: str) -> float:
    """Share of LaTeX Greek-letter commands vs literal Unicode Greek."""
    if not _has_latex(text):
        return 0.5
    commands = len(_GREEK_CMD_RE.findall(text))
    literals = sum(text.count(c) for c in _GREEK_LOWER_CHARS + _GREEK_UPPER_CHARS)
    return _safe_ratio(commands, commands + literals)


# B7 ---------------------------------------------------------------------------
_OP_COMMANDS = {
    "times": "\u00d7",
    "div": "\u00f7",
    "pm": "\u00b1",
    "mp": "\u2213",
    "le": "\u2264",
    "leq": "\u2264",
    "ge": "\u2265",
    "geq": "\u2265",
    "ne": "\u2260",
    "neq": "\u2260",
    "approx": "\u2248",
    "equiv": "\u2261",
    "to": "\u2192",
    "rightarrow": "\u2192",
    "implies": "\u21d2",
    "iff": "\u21d4",
    "leftrightarrow": "\u21d4",
    "forall": "\u2200",
    "exists": "\u2203",
    "in": "\u2208",
    "notin": "\u2209",
    "subset": "\u2282",
    "subseteq": "\u2286",
    "cup": "\u222a",
    "cap": "\u2229",
    "infty": "\u221e",
    "partial": "\u2202",
    "nabla": "\u2207",
    "sum": "\u2211",
    "prod": "\u220f",
    "int": "\u222b",
    "sqrt": "\u221a",
}
_OP_CMD_RE = re.compile(r"\\(?:" + "|".join(_OP_COMMANDS.keys()) + r")\b")


def f_math_op_command_rate(text: str) -> float:
    """Share of LaTeX math-operator commands vs literal Unicode equivalents."""
    if not _has_latex(text):
        return 0.5
    cmd_count = len(_OP_CMD_RE.findall(text))
    literal_count = sum(text.count(c) for c in _OP_COMMANDS.values())
    return _safe_ratio(cmd_count, cmd_count + literal_count)


# B8 ---------------------------------------------------------------------------
_SUB_BRACED_RE = re.compile(r"_\{[^}]{2,}\}")
_SUB_BARE_RE = re.compile(r"_[a-zA-Z0-9](?!\w)")


def f_subscript_brace_rate(text: str) -> float:
    """Share of braced subscripts x_{...} vs bare x_i."""
    if not _has_latex(text):
        return 0.5
    braced = len(_SUB_BRACED_RE.findall(text))
    bare = len(_SUB_BARE_RE.findall(text))
    return _safe_ratio(braced, braced + bare)


# B9 ---------------------------------------------------------------------------
_SUP_BRACED_RE = re.compile(r"\^\{[^}]{2,}\}")
_SUP_BARE_RE = re.compile(r"\^[a-zA-Z0-9](?!\w)")


def f_superscript_brace_rate(text: str) -> float:
    """Share of braced superscripts x^{...} vs bare x^2."""
    if not _has_latex(text):
        return 0.5
    braced = len(_SUP_BRACED_RE.findall(text))
    bare = len(_SUP_BARE_RE.findall(text))
    return _safe_ratio(braced, braced + bare)


# B10 --------------------------------------------------------------------------
_FRAC_CMD_RE = re.compile(r"\\frac\s*\{[^{}]+\}\s*\{[^{}]+\}")
_PAREN_SLASH_FRAC_RE = re.compile(r"\([^()\n]+\)\s*/\s*\([^()\n]+\)")
_SIMPLE_SLASH_FRAC_RE = re.compile(
    r"(?<!https:)\\?\b[A-Za-z0-9]+(?:\s*)/(?:\s*)[A-Za-z0-9]+\b"
)


def f_fraction_command_rate(text: str) -> float:
    """Share of fractions written as \\frac{a}{b}."""
    frac = len(_FRAC_CMD_RE.findall(text))
    paren = len(_PAREN_SLASH_FRAC_RE.findall(text))
    slash = len(_SIMPLE_SLASH_FRAC_RE.findall(text))
    return _safe_ratio(frac, frac + paren + slash)


# B11 --------------------------------------------------------------------------
def f_fraction_parenthesized_slash_rate(text: str) -> float:
    """Share of slash fractions written as (a)/(b)."""
    paren = len(_PAREN_SLASH_FRAC_RE.findall(text))
    slash = len(_SIMPLE_SLASH_FRAC_RE.findall(text))
    return _safe_ratio(paren, paren + slash)


# B12 --------------------------------------------------------------------------
_LEFT_RIGHT_RE = re.compile(r"\\left[\(\[\{.|]|\\right[\)\]\}.|]")
_MATH_BRACKET_RE = re.compile(r"[\(\)\[\]\{\}]")


def f_bracket_left_right_rate(text: str) -> float:
    """Share of bracket delimiters using \\left/\\right sizing."""
    if not _has_latex(text):
        return 0.5
    sized = len(_LEFT_RIGHT_RE.findall(text))
    plain = len(_MATH_BRACKET_RE.findall(" ".join(_INLINE_MATH_SPANS_RE.findall(text))))
    return _safe_ratio(sized, sized + plain)


# B13 --------------------------------------------------------------------------
def f_emphasis_textit_rate(text: str) -> float:
    """Share of LaTeX emphasis written as \\textit{} rather than \\emph{}."""
    textit = len(re.findall(r"\\textit\s*\{", text))
    emph = len(re.findall(r"\\emph\s*\{", text))
    return _safe_ratio(textit, textit + emph)


# B14 --------------------------------------------------------------------------
_CITE_RE = re.compile(r"\\cite\s*(?:\[[^\]]*\])?\{")
_CITEP_RE = re.compile(r"\\citep\s*(?:\[[^\]]*\])?\{")
_CITET_RE = re.compile(r"\\citet\s*(?:\[[^\]]*\])?\{")


def f_cite_plain_rate(text: str) -> float:
    """Share of citation commands using plain \\cite{}."""
    cite = len(_CITE_RE.findall(text))
    citep = len(_CITEP_RE.findall(text))
    citet = len(_CITET_RE.findall(text))
    return _safe_ratio(cite, cite + citep + citet)


# B15 --------------------------------------------------------------------------
def f_cite_textual_rate(text: str) -> float:
    """Share of natbib citations using textual \\citet{} form."""
    citep = len(_CITEP_RE.findall(text))
    citet = len(_CITET_RE.findall(text))
    return _safe_ratio(citet, citep + citet)


# B16 --------------------------------------------------------------------------
def f_display_math_bracket_rate(text: str) -> float:
    """Share of display math delimited with \\[...\\]."""
    dollar = len(_DISPLAY_DOLLAR_RE.findall(text))
    bracket = len(_DISPLAY_BRACKET_RE.findall(text))
    begin = len(_DISPLAY_BEGIN_RE.findall(text))
    return _safe_ratio(bracket, dollar + bracket + begin)


# B17 --------------------------------------------------------------------------
def f_display_math_environment_rate(text: str) -> float:
    """Share of display math using equation/align-like environments."""
    dollar = len(_DISPLAY_DOLLAR_RE.findall(text))
    bracket = len(_DISPLAY_BRACKET_RE.findall(text))
    begin = len(_DISPLAY_BEGIN_RE.findall(text))
    return _safe_ratio(begin, dollar + bracket + begin)


# B18 --------------------------------------------------------------------------
_MATH_SPACING_CMD_RE = re.compile(r"\\(?:[,;:!]|(?:quad|qquad)\b)")


def f_operator_spacing_density_per_equation(text: str) -> float:
    """Density of TeX spacing commands per inline/display equation marker."""
    equation_count = (
        len(_INLINE_DOLLAR_RE.findall(text))
        + len(_INLINE_PAREN_RE.findall(text))
        + len(_DISPLAY_DOLLAR_RE.findall(text))
        + len(_DISPLAY_BRACKET_RE.findall(text))
        + len(_DISPLAY_BEGIN_RE.findall(text))
    )
    if equation_count == 0:
        return 0.0
    return len(_MATH_SPACING_CMD_RE.findall(text)) / equation_count


# B19 --------------------------------------------------------------------------
def f_boxed_expression_rate(text: str) -> float:
    """Density of \\boxed{} uses per equation marker."""
    equation_count = max(
        len(_INLINE_DOLLAR_RE.findall(text))
        + len(_INLINE_PAREN_RE.findall(text))
        + len(_DISPLAY_DOLLAR_RE.findall(text))
        + len(_DISPLAY_BRACKET_RE.findall(text))
        + len(_DISPLAY_BEGIN_RE.findall(text)),
        1,
    )
    return len(re.findall(r"\\boxed\s*\{", text)) / equation_count


# B20 --------------------------------------------------------------------------
def f_therefore_implies_rate(text: str) -> float:
    """Density of \\therefore/\\implies-style proof connectors per equation."""
    equation_count = max(
        len(_INLINE_DOLLAR_RE.findall(text))
        + len(_INLINE_PAREN_RE.findall(text))
        + len(_DISPLAY_DOLLAR_RE.findall(text))
        + len(_DISPLAY_BRACKET_RE.findall(text))
        + len(_DISPLAY_BEGIN_RE.findall(text)),
        1,
    )
    connectors = len(re.findall(r"\\(?:therefore|implies)\b", text))
    return connectors / equation_count


# B21 --------------------------------------------------------------------------
def f_equation_alignment_ampersand_rate(text: str) -> float:
    """Share of display environments that contain alignment ampersands."""
    env_spans = re.findall(
        r"\\begin\{(?:align|aligned|eqnarray|gather|multline)\*?\}([\s\S]*?)\\end\{",
        text,
    )
    if not env_spans:
        return 0.5
    return sum(1 for span in env_spans if "&" in span) / len(env_spans)


# B22 --------------------------------------------------------------------------
def f_equation_linebreak_density(text: str) -> float:
    """Density of LaTeX equation line breaks (`\\\\`) per display environment."""
    env_count = len(_DISPLAY_BEGIN_RE.findall(text))
    if env_count == 0:
        return 0.0
    return len(re.findall(r"\\\\", text)) / env_count


# B23 --------------------------------------------------------------------------
_REF_RE = re.compile(r"\\ref\s*\{")
_AUTOREF_RE = re.compile(r"\\autoref\s*\{")
_EQREF_RE = re.compile(r"\\eqref\s*\{")


def f_reference_autoref_rate(text: str) -> float:
    """Share of references using \\autoref{}."""
    ref = len(_REF_RE.findall(text))
    autoref = len(_AUTOREF_RE.findall(text))
    eqref = len(_EQREF_RE.findall(text))
    return _safe_ratio(autoref, ref + autoref + eqref)


# B24 --------------------------------------------------------------------------
def f_reference_eqref_rate(text: str) -> float:
    """Share of references using \\eqref{}."""
    ref = len(_REF_RE.findall(text))
    autoref = len(_AUTOREF_RE.findall(text))
    eqref = len(_EQREF_RE.findall(text))
    return _safe_ratio(eqref, ref + autoref + eqref)


# ============================================================================
# Group B — Markdown structure
# ============================================================================

# C1 ---------------------------------------------------------------------------
_BOLD_STAR_RE = re.compile(r"\*\*[^*\n]{1,}\*\*")
_BOLD_UNDER_RE = re.compile(r"__[^_\n]{1,}__")


def f_bold_double_star_rate(text: str) -> float:
    """Share of `**bold**` among bold delimiters."""
    star = len(_BOLD_STAR_RE.findall(text))
    under = len(_BOLD_UNDER_RE.findall(text))
    if star + under == 0:
        return 0.5
    return star / (star + under)


# C2 ---------------------------------------------------------------------------
_ITALIC_STAR_RE = re.compile(r"(?<![*\w])\*([^*\n]{1,})\*(?![*\w])")
_ITALIC_UNDER_RE = re.compile(r"(?<![_\w])_([^_\n]{1,})_(?![_\w])")


def f_italic_star_rate(text: str) -> float:
    """Share of `*italic*` among italic delimiters (after masking bold)."""
    masked = _BOLD_STAR_RE.sub("", text)
    masked = _BOLD_UNDER_RE.sub("", masked)
    star = len(_ITALIC_STAR_RE.findall(masked))
    under = len(_ITALIC_UNDER_RE.findall(masked))
    if star + under == 0:
        return 0.5
    return star / (star + under)


# C3 ---------------------------------------------------------------------------
_ATX_HEADING_RE = re.compile(r"^#{1,6}\s+\S", flags=re.M)
_SETEXT_HEADING_RE = re.compile(r"^[^\n]{1,}\n[=\-]{3,}\s*$", flags=re.M)


def f_heading_atx_rate(text: str) -> float:
    """Share of ATX (`# H1`) among all Markdown headings."""
    atx = len(_ATX_HEADING_RE.findall(text))
    setext = len(_SETEXT_HEADING_RE.findall(text))
    if atx + setext == 0:
        return 0.5
    return atx / (atx + setext)


# C4 ---------------------------------------------------------------------------
_FENCE_LANG_TAG_RE = re.compile(r"```[a-zA-Z][a-zA-Z0-9+\-]*\s*\n")
_FENCE_BARE_RE = re.compile(r"```\s*\n")


def f_code_fence_lang_tag_rate(text: str) -> float:
    """Share of opening code fences that carry a language tag."""
    with_lang = len(_FENCE_LANG_TAG_RE.findall(text))
    bare = len(_FENCE_BARE_RE.findall(text))
    if with_lang + bare == 0:
        return 0.5
    return with_lang / (with_lang + bare)


# C5 ---------------------------------------------------------------------------
_BACKTICK_FENCE_RE = re.compile(r"^```", flags=re.M)
_TILDE_FENCE_RE = re.compile(r"^~~~", flags=re.M)


def f_code_fence_backtick_rate(text: str) -> float:
    """Share of backtick fences (```) over tilde fences (~~~)."""
    bt = len(_BACKTICK_FENCE_RE.findall(text))
    ti = len(_TILDE_FENCE_RE.findall(text))
    if bt + ti == 0:
        return 0.5
    return bt / (bt + ti)


# C6 ---------------------------------------------------------------------------
_INLINE_LINK_RE = re.compile(r"\[[^\]\n]+\]\([^)\s]+\)")
_AUTOLINK_RE = re.compile(r"<https?://[^\s>]+>")
_REF_LINK_RE = re.compile(r"\[[^\]\n]+\]\[\w+\]")


def f_link_inline_rate(text: str) -> float:
    """Share of `[text](url)` inline links among all Markdown link forms."""
    inline = len(_INLINE_LINK_RE.findall(text))
    auto = len(_AUTOLINK_RE.findall(text))
    ref = len(_REF_LINK_RE.findall(text))
    total = inline + auto + ref
    if total == 0:
        return 0.5
    return inline / total


# C7 ---------------------------------------------------------------------------
_NUM_DOT_RE = re.compile(r"^\s*\d+\.\s+", flags=re.M)
_NUM_PAREN_RE = re.compile(r"^\s*\d+\)\s+", flags=re.M)
_NUM_BOTH_RE = re.compile(r"^\s*\(\d+\)\s+", flags=re.M)


def f_numbered_dot_rate(text: str) -> float:
    """Share of `1.` form among numbered-list openers."""
    dot = len(_NUM_DOT_RE.findall(text))
    paren = len(_NUM_PAREN_RE.findall(text))
    both = len(_NUM_BOTH_RE.findall(text))
    total = dot + paren + both
    if total == 0:
        return 0.5
    return dot / total


# C8 ---------------------------------------------------------------------------
_DOUBLE_SPACE_BREAK_RE = re.compile(r"  +\n")
_BACKSLASH_BREAK_RE = re.compile(r"\\\\\n")
_BLANK_LINE_RE = re.compile(r"\n\s*\n")


def f_hardbreak_double_space_rate(text: str) -> float:
    """Share of `  \\n` double-space hard breaks vs `\\\\\\n` vs blank line."""
    ds = len(_DOUBLE_SPACE_BREAK_RE.findall(text))
    bs = len(_BACKSLASH_BREAK_RE.findall(text))
    blank = len(_BLANK_LINE_RE.findall(text))
    total = ds + bs + blank
    if total == 0:
        return 0.5
    return ds / total


# C9 ---------------------------------------------------------------------------
def f_hardbreak_backslash_rate(text: str) -> float:
    """Share of hard breaks written as backslash-newline."""
    ds = len(_DOUBLE_SPACE_BREAK_RE.findall(text))
    bs = len(_BACKSLASH_BREAK_RE.findall(text))
    blank = len(_BLANK_LINE_RE.findall(text))
    return _safe_ratio(bs, ds + bs + blank)


# C10 --------------------------------------------------------------------------
def f_hardbreak_blank_line_rate(text: str) -> float:
    """Share of Markdown paragraph breaks represented by blank lines."""
    ds = len(_DOUBLE_SPACE_BREAK_RE.findall(text))
    bs = len(_BACKSLASH_BREAK_RE.findall(text))
    blank = len(_BLANK_LINE_RE.findall(text))
    return _safe_ratio(blank, ds + bs + blank)


# C11 --------------------------------------------------------------------------
def f_bullet_dash_rate(text: str) -> float:
    """Share of '-' among line-leading bullet markers."""
    bullets = _BULLET_RE.findall(text)
    return _safe_ratio(bullets.count("-"), len(bullets))


# C12 --------------------------------------------------------------------------
def f_bullet_plus_rate(text: str) -> float:
    """Share of '+' among line-leading bullet markers."""
    bullets = _BULLET_RE.findall(text)
    return _safe_ratio(bullets.count("+"), len(bullets))


# C13 --------------------------------------------------------------------------
_UNICODE_BULLET_RE = re.compile(r"^[ \t]*([•◦▪])\s+", flags=re.M)


def f_bullet_unicode_rate(text: str) -> float:
    """Share of Unicode bullets among Markdown-like bullet markers."""
    ascii_bullets = _BULLET_RE.findall(text)
    unicode_bullets = _UNICODE_BULLET_RE.findall(text)
    return _safe_ratio(len(unicode_bullets), len(ascii_bullets) + len(unicode_bullets))


# C12 --------------------------------------------------------------------------
_NUM_ROMAN_RE = re.compile(r"^\s*[ivxlcdmIVXLCDM]+\.\s+", flags=re.M)


def f_numbered_roman_rate(text: str) -> float:
    """Share of roman-numeral numbered-list openers."""
    dot = len(_NUM_DOT_RE.findall(text))
    paren = len(_NUM_PAREN_RE.findall(text))
    both = len(_NUM_BOTH_RE.findall(text))
    roman = len(_NUM_ROMAN_RE.findall(text))
    return _safe_ratio(roman, dot + paren + both + roman)


# C13 --------------------------------------------------------------------------
_NUM_ALPHA_RE = re.compile(r"^\s*[A-Za-z]\.\s+", flags=re.M)


def f_numbered_alpha_rate(text: str) -> float:
    """Share of alpha-letter numbered-list openers."""
    dot = len(_NUM_DOT_RE.findall(text))
    paren = len(_NUM_PAREN_RE.findall(text))
    both = len(_NUM_BOTH_RE.findall(text))
    alpha = len(_NUM_ALPHA_RE.findall(text))
    return _safe_ratio(alpha, dot + paren + both + alpha)


# C14 --------------------------------------------------------------------------
_HEADING_LINE_RE = re.compile(r"^(#{1,6}\s+\S.*|[^\n]+\n[=\-]{3,}\s*)", flags=re.M)


def f_blank_line_before_heading_rate(text: str) -> float:
    """Share of Markdown headings preceded by a blank line or start of file."""
    starts = [m.start() for m in _HEADING_LINE_RE.finditer(text)]
    if not starts:
        return 0.5
    good = 0
    for start in starts:
        prefix = text[:start]
        if start == 0 or prefix.endswith("\n\n"):
            good += 1
    return good / len(starts)


# C15 --------------------------------------------------------------------------
def f_code_fence_tilde_rate(text: str) -> float:
    """Share of tilde fences (~~~) over all fenced code blocks."""
    bt = len(_BACKTICK_FENCE_RE.findall(text))
    ti = len(_TILDE_FENCE_RE.findall(text))
    return _safe_ratio(ti, bt + ti)


# C16 --------------------------------------------------------------------------
def f_indented_code_block_rate(text: str) -> float:
    """Share of code blocks written as four-space/tab indented blocks."""
    fenced = len(_CODE_FENCE_RE.findall(text))
    indented = len(_extract_indented_code(text))
    return _safe_ratio(indented, fenced + indented)


# C17 --------------------------------------------------------------------------
_INLINE_CODE_SINGLE_RE = re.compile(r"(?<!`)`[^`\n]+`(?!`)")
_INLINE_CODE_DOUBLE_RE = re.compile(r"``[^`\n][\s\S]*?[^`\n]``")


def f_inline_code_double_backtick_rate(text: str) -> float:
    """Share of inline code spans delimited with double backticks."""
    single = len(_INLINE_CODE_SINGLE_RE.findall(text))
    double = len(_INLINE_CODE_DOUBLE_RE.findall(text))
    return _safe_ratio(double, single + double)


# C18 --------------------------------------------------------------------------
_IMAGE_INLINE_RE = re.compile(r"!\[[^\]\n]*\]\([^)]+\)")
_IMAGE_REF_RE = re.compile(r"!\[[^\]\n]*\]\[[^\]\n]+\]")


def f_image_inline_rate(text: str) -> float:
    """Share of images using inline `![alt](url)` syntax."""
    inline = len(_IMAGE_INLINE_RE.findall(text))
    ref = len(_IMAGE_REF_RE.findall(text))
    return _safe_ratio(inline, inline + ref)


# C19 --------------------------------------------------------------------------
_BLOCKQUOTE_RE = re.compile(r"^(>+)\s?", flags=re.M)


def f_blockquote_nested_rate(text: str) -> float:
    """Share of blockquote lines with nesting depth greater than one."""
    markers = _BLOCKQUOTE_RE.findall(text)
    if not markers:
        return 0.5
    return sum(1 for marker in markers if len(marker) > 1) / len(markers)


# C20 --------------------------------------------------------------------------
def f_blank_line_after_blockquote_rate(text: str) -> float:
    """Share of blockquote blocks followed by a blank line or EOF."""
    blocks = list(re.finditer(r"(?:^>.*(?:\n|$))+", text, flags=re.M))
    if not blocks:
        return 0.5
    good = 0
    for block in blocks:
        end = block.end()
        if end >= len(text) or text[end : end + 1] == "\n":
            good += 1
    return good / len(blocks)


# C21 --------------------------------------------------------------------------
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$", flags=re.M)


def f_table_cell_padding_rate(text: str) -> float:
    """Share of Markdown table cells padded with spaces on both sides."""
    cells = []
    for row in _TABLE_ROW_RE.findall(text):
        cells.extend(row.strip().strip("|").split("|"))
    content_cells = [cell for cell in cells if cell.strip()]
    if not content_cells:
        return 0.5
    return sum(
        1 for cell in content_cells if cell.startswith(" ") and cell.endswith(" ")
    ) / len(content_cells)


# C22 --------------------------------------------------------------------------
_TABLE_ALIGN_CELL_RE = re.compile(r":-+:")
_TABLE_ALIGN_ANY_RE = re.compile(r":?-{3,}:?")


def f_table_center_align_rate(text: str) -> float:
    """Share of Markdown table alignment cells using centered `:---:` form."""
    any_align = len(_TABLE_ALIGN_ANY_RE.findall(text))
    center = len(_TABLE_ALIGN_CELL_RE.findall(text))
    return _safe_ratio(center, any_align)


# C23 --------------------------------------------------------------------------
_MD_ESCAPE_RE = re.compile(r"\\[*_`]")


def f_markdown_escape_density_per_kb(text: str) -> float:
    """Density of Markdown escapes for `*`, `_`, and backticks."""
    return _per_kb(len(_MD_ESCAPE_RE.findall(text)), text)


# ============================================================================
# Group D — Code surface
# ============================================================================

_PY_SINGLE_STRING_RE = re.compile(r"(?<![A-Za-z])(?:[rubfRUBF]{0,2})'([^'\\]|\\.)*'")
_PY_DOUBLE_STRING_RE = re.compile(r'(?<![A-Za-z])(?:[rubfRUBF]{0,2})"([^"\\]|\\.)*"')


def f_code_string_single_quote_rate(text: str) -> float:
    """Share of code string literals using single quotes."""
    code = _code_blob(text)
    single = len(_PY_SINGLE_STRING_RE.findall(code))
    double = len(_PY_DOUBLE_STRING_RE.findall(code))
    return _safe_ratio(single, single + double)


def f_code_indent_2space_rate(text: str) -> float:
    """Share of indented code lines using a two-space indent."""
    code = _code_blob(text)
    indents = re.findall(r"^( +|\t+)\S", code, flags=re.M)
    if not indents:
        return 0.5
    return sum(1 for indent in indents if indent == "  ") / len(indents)


def f_code_indent_4space_rate(text: str) -> float:
    """Share of indented code lines using a four-space indent."""
    code = _code_blob(text)
    indents = re.findall(r"^( +|\t+)\S", code, flags=re.M)
    if not indents:
        return 0.5
    return sum(1 for indent in indents if indent == "    ") / len(indents)


def f_code_indent_tab_rate(text: str) -> float:
    """Share of indented code lines using tab indentation."""
    code = _code_blob(text)
    indents = re.findall(r"^( +|\t+)\S", code, flags=re.M)
    if not indents:
        return 0.5
    return sum(1 for indent in indents if indent.startswith("\t")) / len(indents)


_HASH_COMMENT_RE = re.compile(r"^\s*#", flags=re.M)
_SLASH_COMMENT_RE = re.compile(r"^\s*//", flags=re.M)
_BLOCK_COMMENT_RE = re.compile(r"/\*[\s\S]*?\*/")
_SQL_COMMENT_RE = re.compile(r"^\s*--", flags=re.M)


def f_code_comment_hash_rate(text: str) -> float:
    """Share of comments using Python/shell `#` form."""
    code = _code_blob(text)
    counts = [
        len(_HASH_COMMENT_RE.findall(code)),
        len(_SLASH_COMMENT_RE.findall(code)),
        len(_BLOCK_COMMENT_RE.findall(code)),
        len(_SQL_COMMENT_RE.findall(code)),
    ]
    return _safe_ratio(counts[0], sum(counts))


def f_code_comment_slash_rate(text: str) -> float:
    """Share of comments using `//` form."""
    code = _code_blob(text)
    counts = [
        len(_HASH_COMMENT_RE.findall(code)),
        len(_SLASH_COMMENT_RE.findall(code)),
        len(_BLOCK_COMMENT_RE.findall(code)),
        len(_SQL_COMMENT_RE.findall(code)),
    ]
    return _safe_ratio(counts[1], sum(counts))


def f_code_comment_block_rate(text: str) -> float:
    """Share of comments using `/* ... */` form."""
    code = _code_blob(text)
    counts = [
        len(_HASH_COMMENT_RE.findall(code)),
        len(_SLASH_COMMENT_RE.findall(code)),
        len(_BLOCK_COMMENT_RE.findall(code)),
        len(_SQL_COMMENT_RE.findall(code)),
    ]
    return _safe_ratio(counts[2], sum(counts))


def f_code_comment_sql_dash_rate(text: str) -> float:
    """Share of comments using SQL-style `--` form."""
    code = _code_blob(text)
    counts = [
        len(_HASH_COMMENT_RE.findall(code)),
        len(_SLASH_COMMENT_RE.findall(code)),
        len(_BLOCK_COMMENT_RE.findall(code)),
        len(_SQL_COMMENT_RE.findall(code)),
    ]
    return _safe_ratio(counts[3], sum(counts))


def f_type_annotation_spaced_rate(text: str) -> float:
    """Share of type annotations written as `x: int` rather than `x:int`."""
    code = _code_blob(text)
    spaced = len(re.findall(r"\b[A-Za-z_]\w*\s*:\s+[A-Za-z_][\w.\[\], ]*", code))
    compact = len(re.findall(r"\b[A-Za-z_]\w*:[A-Za-z_][\w.\[\], ]*", code))
    return _safe_ratio(spaced, spaced + compact)


_FSTRING_DOUBLE_RE = re.compile(r'(?<![A-Za-z])f"([^"\\]|\\.)*"')
_FSTRING_SINGLE_RE = re.compile(r"(?<![A-Za-z])f'([^'\\]|\\.)*'")
_FORMAT_METHOD_RE = re.compile(r"\.format\s*\(")
_PERCENT_FORMAT_RE = re.compile(r"(['\"][^'\"]*%[sdifrx][^'\"]*['\"])\s*%")


def f_fstring_double_quote_rate(text: str) -> float:
    """Share of formatting expressions using double-quoted f-strings."""
    code = _code_blob(text)
    f_double = len(_FSTRING_DOUBLE_RE.findall(code))
    f_single = len(_FSTRING_SINGLE_RE.findall(code))
    fmt = len(_FORMAT_METHOD_RE.findall(code))
    pct = len(_PERCENT_FORMAT_RE.findall(code))
    return _safe_ratio(f_double, f_double + f_single + fmt + pct)


def f_fstring_single_quote_rate(text: str) -> float:
    """Share of formatting expressions using single-quoted f-strings."""
    code = _code_blob(text)
    f_double = len(_FSTRING_DOUBLE_RE.findall(code))
    f_single = len(_FSTRING_SINGLE_RE.findall(code))
    fmt = len(_FORMAT_METHOD_RE.findall(code))
    pct = len(_PERCENT_FORMAT_RE.findall(code))
    return _safe_ratio(f_single, f_double + f_single + fmt + pct)


def f_format_method_rate(text: str) -> float:
    """Share of string formatting using `.format()`."""
    code = _code_blob(text)
    f_double = len(_FSTRING_DOUBLE_RE.findall(code))
    f_single = len(_FSTRING_SINGLE_RE.findall(code))
    fmt = len(_FORMAT_METHOD_RE.findall(code))
    pct = len(_PERCENT_FORMAT_RE.findall(code))
    return _safe_ratio(fmt, f_double + f_single + fmt + pct)


def f_percent_format_rate(text: str) -> float:
    """Share of string formatting using `%` interpolation."""
    code = _code_blob(text)
    f_double = len(_FSTRING_DOUBLE_RE.findall(code))
    f_single = len(_FSTRING_SINGLE_RE.findall(code))
    fmt = len(_FORMAT_METHOD_RE.findall(code))
    pct = len(_PERCENT_FORMAT_RE.findall(code))
    return _safe_ratio(pct, f_double + f_single + fmt + pct)


_COLLECTION_LITERAL_RE = re.compile(r"[\[\{\(][^\]\}\)]*[\]\}\)]", flags=re.S)
_TRAILING_COMMA_RE = re.compile(r",\s*[\]\}\)]")


def f_code_trailing_comma_rate(text: str) -> float:
    """Share of collection/call literals containing a trailing comma."""
    code = _code_blob(text)
    literals = len(_COLLECTION_LITERAL_RE.findall(code))
    trailing = len(_TRAILING_COMMA_RE.findall(code))
    return _safe_ratio(trailing, literals, default=0.0)


_TRIPLE_DOUBLE_RE = re.compile(r'"""[\s\S]*?"""')
_TRIPLE_SINGLE_RE = re.compile(r"'''[\s\S]*?'''")
_DEF_CLASS_RE = re.compile(r"^\s*(?:def|class)\s+\w+", flags=re.M)


def f_docstring_presence_rate(text: str) -> float:
    """Docstring count per Python def/class block in code."""
    code = _code_blob(text)
    blocks = len(_DEF_CLASS_RE.findall(code))
    docs = len(_TRIPLE_DOUBLE_RE.findall(code)) + len(_TRIPLE_SINGLE_RE.findall(code))
    if blocks == 0:
        return 0.0
    return docs / blocks


def f_docstring_triple_double_rate(text: str) -> float:
    """Share of docstrings using triple-double quotes."""
    code = _code_blob(text)
    double = len(_TRIPLE_DOUBLE_RE.findall(code))
    single = len(_TRIPLE_SINGLE_RE.findall(code))
    return _safe_ratio(double, double + single)


def f_docstring_first_line_summary_rate(text: str) -> float:
    """Share of docstrings whose first content line is a short summary."""
    code = _code_blob(text)
    docs = _TRIPLE_DOUBLE_RE.findall(code) + _TRIPLE_SINGLE_RE.findall(code)
    if not docs:
        return 0.5
    good = 0
    for doc in docs:
        inner = doc[3:-3].strip().splitlines()
        if inner and len(inner[0].strip()) <= 88 and inner[0].strip().endswith("."):
            good += 1
    return good / len(docs)


def f_lambda_over_def_rate(text: str) -> float:
    """Share of callable definitions written as `lambda` rather than `def`."""
    code = _code_blob(text)
    lambdas = len(re.findall(r"\blambda\b", code))
    defs = len(re.findall(r"^\s*def\s+\w+", code, flags=re.M))
    return _safe_ratio(lambdas, lambdas + defs)


def f_listcomp_over_loop_rate(text: str) -> float:
    """Share of list-building opportunities using list comprehensions."""
    code = _code_blob(text)
    listcomp = len(re.findall(r"\[[^\]\n]+\bfor\b[^\]\n]+\]", code))
    loops = len(re.findall(r"^\s*for\s+\w+.*:\s*$", code, flags=re.M))
    return _safe_ratio(listcomp, listcomp + loops)


def f_import_blankline_group_rate(text: str) -> float:
    """Share of adjacent import blocks separated by blank lines."""
    code = _code_blob(text)
    import_lines = [
        m.start() for m in re.finditer(r"^\s*(?:import|from)\s+", code, flags=re.M)
    ]
    if len(import_lines) < 2:
        return 0.5
    separated = 0
    for prev, curr in zip(import_lines, import_lines[1:]):
        if "\n\n" in code[prev:curr]:
            separated += 1
    return separated / (len(import_lines) - 1)


# ============================================================================
# Group E — Whitespace and invisible characters
# ============================================================================


def f_crlf_newline_rate(text: str) -> float:
    """Share of newlines encoded as CRLF rather than LF."""
    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    return _safe_ratio(crlf, crlf + lf)


def f_tokenizer_fragmentation_proxy_per_kb(text: str) -> float:
    """A light CPT proxy: non-ASCII/markup punctuation fragments per KB."""
    fragments = sum(1 for c in text if ord(c) > 127)
    fragments += len(re.findall(r"[\\`*_{}\[\]<>|~^]", text))
    return _per_kb(fragments, text)


def f_leading_tab_rate(text: str) -> float:
    """Share of indented lines whose leading indentation contains tabs."""
    indents = re.findall(r"^([ \t]+)\S", text, flags=re.M)
    if not indents:
        return 0.5
    return sum(1 for indent in indents if "\t" in indent) / len(indents)


def f_indent_consistency_rate(text: str) -> float:
    """Dominant indentation style share among tabs, two spaces, and four spaces."""
    indents = re.findall(r"^([ \t]+)\S", text, flags=re.M)
    if not indents:
        return 0.5
    buckets = {
        "tab": sum(1 for indent in indents if indent.startswith("\t")),
        "two": sum(
            1
            for indent in indents
            if indent.startswith("  ") and not indent.startswith("    ")
        ),
        "four": sum(1 for indent in indents if indent.startswith("    ")),
        "other": 0,
    }
    buckets["other"] = len(indents) - buckets["tab"] - buckets["two"] - buckets["four"]
    return max(buckets.values()) / len(indents)


def f_final_newline_present(text: str) -> float:
    """Binary: text ends with a newline."""
    if not text:
        return 0.0
    return 1.0 if text.endswith("\n") else 0.0


def f_section_blank_line_rate(text: str) -> float:
    """Share of section/list boundaries separated by a blank line."""
    boundaries = len(
        re.findall(r"^(?:#{1,6}\s+|\s*[-*+]\s+|\s*\d+\.\s+)", text, flags=re.M)
    )
    if boundaries == 0:
        return 0.0
    return len(_BLANK_LINE_RE.findall(text)) / boundaries


# ============================================================================
# Public API
# ============================================================================

FEATURE_FUNCS = [
    # Group A — prose typography (22)
    f_apostrophe_curly_rate,
    f_single_quote_pair_unicode_rate,
    f_double_quote_pair_unicode_rate,
    f_dash_emdash_rate,
    f_dash_endash_rate,
    f_ellipsis_unicode_rate,
    f_minus_unicode_rate,
    f_nbsp_rate_per_word,
    f_invisible_char_presence,
    f_french_spacing_rate,
    f_comma_fullwidth_rate,
    f_bullet_star_rate,
    f_quote_unicode_rate,
    f_dash_unicode_rate,
    f_special_space_rate_per_word,
    f_post_vowel_apostrophe_curly_rate,
    f_post_consonant_apostrophe_curly_rate,
    f_trailing_space_rate_per_line,
    f_double_space_after_period_rate,
    f_blank_line_between_paragraphs_rate,
    f_emoji_presence,
    f_emoji_rate_per_word,
    # Group C — math and LaTeX (24)
    f_tex_quote_convention_rate,
    f_emdash_ligature_rate,
    f_endash_ligature_rate,
    f_inline_math_dollar_rate,
    f_display_math_dollar_rate,
    f_greek_command_rate,
    f_math_op_command_rate,
    f_subscript_brace_rate,
    f_superscript_brace_rate,
    f_fraction_command_rate,
    f_fraction_parenthesized_slash_rate,
    f_bracket_left_right_rate,
    f_emphasis_textit_rate,
    f_cite_plain_rate,
    f_cite_textual_rate,
    f_display_math_bracket_rate,
    f_display_math_environment_rate,
    f_operator_spacing_density_per_equation,
    f_boxed_expression_rate,
    f_therefore_implies_rate,
    f_equation_alignment_ampersand_rate,
    f_equation_linebreak_density,
    f_reference_autoref_rate,
    f_reference_eqref_rate,
    # Group B — Markdown structure (25)
    f_bold_double_star_rate,
    f_italic_star_rate,
    f_heading_atx_rate,
    f_code_fence_lang_tag_rate,
    f_code_fence_backtick_rate,
    f_link_inline_rate,
    f_numbered_dot_rate,
    f_hardbreak_double_space_rate,
    f_hardbreak_backslash_rate,
    f_hardbreak_blank_line_rate,
    f_bullet_dash_rate,
    f_bullet_plus_rate,
    f_bullet_unicode_rate,
    f_numbered_roman_rate,
    f_numbered_alpha_rate,
    f_blank_line_before_heading_rate,
    f_code_fence_tilde_rate,
    f_indented_code_block_rate,
    f_inline_code_double_backtick_rate,
    f_image_inline_rate,
    f_blockquote_nested_rate,
    f_blank_line_after_blockquote_rate,
    f_table_cell_padding_rate,
    f_table_center_align_rate,
    f_markdown_escape_density_per_kb,
    # Group D — code surface (20)
    f_code_string_single_quote_rate,
    f_code_indent_2space_rate,
    f_code_indent_4space_rate,
    f_code_indent_tab_rate,
    f_code_comment_hash_rate,
    f_code_comment_slash_rate,
    f_code_comment_block_rate,
    f_code_comment_sql_dash_rate,
    f_type_annotation_spaced_rate,
    f_fstring_double_quote_rate,
    f_fstring_single_quote_rate,
    f_format_method_rate,
    f_percent_format_rate,
    f_code_trailing_comma_rate,
    f_docstring_presence_rate,
    f_docstring_triple_double_rate,
    f_docstring_first_line_summary_rate,
    f_lambda_over_def_rate,
    f_listcomp_over_loop_rate,
    f_import_blankline_group_rate,
    # Group E — whitespace and invisible (6)
    f_crlf_newline_rate,
    f_tokenizer_fragmentation_proxy_per_kb,
    f_leading_tab_rate,
    f_indent_consistency_rate,
    f_final_newline_present,
    f_section_blank_line_rate,
]

FEATURE_NAMES: List[str] = [f.__name__ for f in FEATURE_FUNCS]
FEATURE_GROUPS: List[str] = (
    ["A"] * 22 + ["C"] * 24 + ["B"] * 25 + ["D"] * 20 + ["E"] * 6
)
FEATURE_DIM = len(FEATURE_FUNCS)
FEATURE_GROUP_ALIASES = {
    "prose": "A",
    "typography": "A",
    "math": "C",
    "latex": "C",
    "markdown": "B",
    "code": "D",
    "whitespace": "E",
    "invisible": "E",
}


def _normalize_feature_group(group: str) -> str:
    normalized = str(group).strip().lower()
    return FEATURE_GROUP_ALIASES.get(normalized, normalized.upper())


def _coerce_feature_groups(
    disabled_feature_groups: Sequence[str] | str | None,
) -> Sequence[str]:
    if disabled_feature_groups is None:
        return ()
    if isinstance(disabled_feature_groups, str):
        return (disabled_feature_groups,)
    return disabled_feature_groups


def get_feature_indices(
    disabled_feature_groups: Sequence[str] | str | None = None,
) -> List[int]:
    """Return feature indices after removing the requested feature groups."""
    disabled = {
        _normalize_feature_group(group)
        for group in _coerce_feature_groups(disabled_feature_groups)
    }
    unknown = disabled - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown feature groups: {sorted(unknown)}")
    return [i for i, group in enumerate(FEATURE_GROUPS) if group not in disabled]


def get_feature_names(
    disabled_feature_groups: Sequence[str] | str | None = None,
) -> List[str]:
    """Return feature names after removing the requested feature groups."""
    return [FEATURE_NAMES[i] for i in get_feature_indices(disabled_feature_groups)]


def _get_feature_funcs(
    disabled_feature_groups: Sequence[str] | str | None = None,
) -> List[Any]:
    return [FEATURE_FUNCS[i] for i in get_feature_indices(disabled_feature_groups)]


def extract_features(
    text: str,
    disabled_feature_groups: Sequence[str] | str | None = None,
) -> np.ndarray:
    """Extract the FEATURE_DIM-dimensional SurfacePrint feature vector from a string.

    Parameters
    ----------
    text : str
        A document in any source form (prose, Markdown, LaTeX, mixed).
    disabled_feature_groups : Sequence[str] or str, optional
        Feature groups to omit. Use "E" or "whitespace" to remove whitespace
        features.

    Returns
    -------
    np.ndarray of shape (n_enabled_features,), dtype=float32
        Feature vector in the order given by `FEATURE_NAMES`. Each entry
        is in [0, 1] (ratio) or non-negative (rate). Missing-feature
        defaults: 0.5 for ratios, 0.0 for rates.
    """
    feature_funcs = _get_feature_funcs(disabled_feature_groups)
    if not text:
        return np.full(len(feature_funcs), 0.5, dtype=np.float32)
    return np.array([f(text) for f in feature_funcs], dtype=np.float32)


def extract_feature_matrix(
    texts: Sequence[str],
    disabled_feature_groups: Sequence[str] | str | None = None,
) -> np.ndarray:
    """Extract a feature matrix with shape (n_texts, FEATURE_DIM)."""
    feature_funcs = _get_feature_funcs(disabled_feature_groups)
    X = np.empty((len(texts), len(feature_funcs)), dtype=np.float32)
    for i, text in enumerate(texts):
        X[i] = extract_features(text, disabled_feature_groups)
    return X


class ASCIIDetector(BaseDetector):
    """Classifier over the FEATURE_DIM SurfacePrint feature functions.

    Config keys:
        model_path: optional pickle path for a trained sklearn-compatible model.
        positive_class: class whose probability is returned by predict (default 1).
        disabled_feature_groups/disable_feature_groups: group labels or aliases
            to omit from extraction, e.g. ["E"] or ["whitespace"].
        disable_whitespace_features: shorthand for disabling group E.
        batch_workers/num_workers: process workers used by batch_predict and
            fit() for parallel feature extraction (default: 4).
        fit_batch_size: chunk size used when extracting features for fit()
            (default: 1024). Caps in-flight pickled-text memory.
        fit_show_progress: whether fit() shows a tqdm progress bar
            (default: True).
        max_depth/min_samples_leaf/random_state/class_weight: DecisionTreeClassifier
            training parameters used by fit() when no model is supplied.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.positive_class = self.config.get("positive_class", 1)
        self.model: Any | None = self.config.get("model")
        disabled_feature_groups = list(
            _coerce_feature_groups(
                self.config.get(
                    "disabled_feature_groups",
                    self.config.get("disable_feature_groups", ()),
                )
            )
        )
        if self.config.get("disable_whitespace_features", False):
            disabled_feature_groups.append("E")
        self.disabled_feature_groups = tuple(disabled_feature_groups)
        self.feature_names = get_feature_names(self.disabled_feature_groups)
        self.feature_dim = len(self.feature_names)
        self.max_workers = int(
            self.config.get("batch_workers", self.config.get("num_workers", 4))
        )
        if self.max_workers < 1:
            raise ValueError("batch_workers must be >= 1")
        model_path = self.config.get("model_path")
        if model_path is not None:
            self.load(model_path)

    @property
    def is_main_process(self) -> bool:
        return True

    def _new_model(self) -> Any:
        try:
            from sklearn.tree import DecisionTreeClassifier
        except ImportError as exc:
            raise RuntimeError("ASCIIDetector training requires scikit-learn.") from exc

        return DecisionTreeClassifier(
            max_depth=self.config.get("max_depth"),
            min_samples_leaf=int(self.config.get("min_samples_leaf", 1)),
            random_state=self.config.get("random_state", 0),
            class_weight=self.config.get("class_weight"),
        )

    def _iter_feature_chunks(
        self,
        texts: Sequence[str],
        batch_size: int,
        show_progress: bool = False,
    ) -> Iterator[np.ndarray]:
        """Yield feature matrices in input order with bounded in-flight memory.

        Uses a sliding window of at most ``2 * max_workers`` pending futures so
        that only that many text chunks are pickled into worker processes at
        once, regardless of the total dataset size. Falls back to in-process
        extraction (no fork/pickle overhead) when only one worker is requested.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        n = len(texts)
        if n == 0:
            return

        n_chunks = (n + batch_size - 1) // batch_size
        max_workers = max(1, min(self.max_workers, n_chunks))
        pbar = (
            tqdm.tqdm(total=n_chunks, desc="ASCIIDetector", unit="batch")
            if show_progress
            else None
        )

        try:
            if max_workers == 1:
                for start in range(0, n, batch_size):
                    yield extract_feature_matrix(
                        texts[start : start + batch_size],
                        self.disabled_feature_groups,
                    )
                    if pbar is not None:
                        pbar.update(1)
                return

            in_flight = 2 * max_workers
            starts = iter(range(0, n, batch_size))
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                pending: deque = deque()

                def _submit_next() -> bool:
                    try:
                        s = next(starts)
                    except StopIteration:
                        return False
                    pending.append(
                        executor.submit(
                            extract_feature_matrix,
                            texts[s : s + batch_size],
                            self.disabled_feature_groups,
                        )
                    )
                    return True

                for _ in range(in_flight):
                    if not _submit_next():
                        break

                while pending:
                    future = pending.popleft()
                    yield future.result()
                    if pbar is not None:
                        pbar.update(1)
                    _submit_next()
        finally:
            if pbar is not None:
                pbar.close()

    def fit(self, texts: Sequence[str], labels: Sequence[Any]) -> "ASCIIDetector":
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        model = self.model if self.model is not None else self._new_model()
        n = len(texts)
        X = np.empty((n, self.feature_dim), dtype=np.float32)
        if n > 0:
            batch_size = int(self.config.get("fit_batch_size", 1024))
            show_progress = bool(self.config.get("fit_show_progress", True))
            offset = 0
            for chunk_X in self._iter_feature_chunks(
                texts, batch_size, show_progress=show_progress
            ):
                end = offset + chunk_X.shape[0]
                X[offset:end] = chunk_X
                offset = end
        model.fit(X, np.asarray(labels))
        self.model = model
        return self

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save ASCIIDetector before fit/load.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str | Path) -> "ASCIIDetector":
        with Path(path).open("rb") as f:
            self.model = pickle.load(f)
        return self

    def _positive_class_index(self, n_classes: int) -> int:
        classes = list(getattr(self.model, "classes_", range(n_classes)))
        return (
            classes.index(self.positive_class) if self.positive_class in classes else -1
        )

    def _scores_from_features(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(
                "ASCIIDetector needs a trained model. Call fit() or set model_path."
            )
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            idx = self._positive_class_index(probs.shape[1])
            return probs[:, idx].astype(np.float64, copy=False)
        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)
        else:
            scores = self.model.predict(X)
        return np.asarray(scores, dtype=np.float64).reshape(-1)

    def predict(self, x: str) -> float:
        return float(
            self._scores_from_features(
                extract_features(x, self.disabled_feature_groups)[None, :]
            )[0]
        )

    def batch_predict(
        self,
        x: List[str],
        batch_size: int = 1024,
        show_progress: bool = False,
    ) -> List[float]:
        if self.model is None:
            raise RuntimeError(
                "ASCIIDetector needs a trained model. Call fit() or set model_path."
            )
        if not x:
            return []

        scores: List[float] = []
        for X in self._iter_feature_chunks(x, batch_size, show_progress=show_progress):
            scores.extend(self._scores_from_features(X).tolist())
        return scores
