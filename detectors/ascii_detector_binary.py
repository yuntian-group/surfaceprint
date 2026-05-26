"""Binary ParaPunct detector: one presence bit per feature in ascii_detector."""

from __future__ import annotations

import pickle
import re
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator, List, Sequence

import numpy as np
import tqdm

from detectors import ascii_detector as ad
from detectors.base_detector import BaseDetector

_ALIGN_ENV_RE = re.compile(
    r"\\begin\{(?:align|aligned|eqnarray|gather|multline)\*?\}([\s\S]*?)\\end\{"
)
_TYPE_ANNOTATION_SPACED_RE = re.compile(r"\b[A-Za-z_]\w*\s*:\s+[A-Za-z_][\w.\[\], ]*")
_LISTCOMP_RE = re.compile(r"\[[^\]\n]+\bfor\b[^\]\n]+\]")
_IMPORT_LINE_RE = re.compile(r"^\s*(?:import|from)\s+", flags=re.M)
_SECTION_BOUNDARY_RE = re.compile(r"^(?:#{1,6}\s+|\s*[-*+]\s+|\s*\d+\.\s+)", flags=re.M)
_THEREFORE_IMPLIES_RE = re.compile(r"\\(?:therefore|implies)\b")
_BOXED_RE = re.compile(r"\\boxed\s*\{")
_TEXTIT_RE = re.compile(r"\\textit\s*\{")
_LAMBDA_RE = re.compile(r"\blambda\b")
_MARKUP_FRAGMENT_RE = re.compile(r"[\\`*_{}\[\]<>|~^]")


def _bin(value: bool) -> float:
    return 1.0 if value else 0.0


def _has_padded_table_cell(text: str) -> bool:
    for row in ad._TABLE_ROW_RE.findall(text):
        for cell in row.strip().strip("|").split("|"):
            if cell.strip() and cell.startswith(" ") and cell.endswith(" "):
                return True
    return False


def _has_summary_docstring(code: str) -> bool:
    for doc in ad._TRIPLE_DOUBLE_RE.findall(code) + ad._TRIPLE_SINGLE_RE.findall(code):
        inner = doc[3:-3].strip().splitlines()
        if inner and len(inner[0].strip()) <= 88 and inner[0].strip().endswith("."):
            return True
    return False


def _has_separated_import_groups(code: str) -> bool:
    import_lines = [m.start() for m in _IMPORT_LINE_RE.finditer(code)]
    if len(import_lines) < 2:
        return False
    for prev, curr in zip(import_lines, import_lines[1:]):
        if "\n\n" in code[prev:curr]:
            return True
    return False


def _has_blank_line_before_heading(text: str) -> bool:
    starts = [m.start() for m in ad._HEADING_LINE_RE.finditer(text)]
    for start in starts:
        if start == 0 or text[:start].endswith("\n\n"):
            return True
    return False


def _has_blank_line_after_blockquote(text: str) -> bool:
    for block in re.finditer(r"(?:^>.*(?:\n|$))+", text, flags=re.M):
        end = block.end()
        if end >= len(text) or text[end : end + 1] == "\n":
            return True
    return False


def _has_nested_blockquote(text: str) -> bool:
    markers = ad._BLOCKQUOTE_RE.findall(text)
    return any(len(marker) > 1 for marker in markers)


def _has_aligned_env_ampersand(text: str) -> bool:
    return any("&" in span for span in _ALIGN_ENV_RE.findall(text))


# ============================================================================
# Group A — Universal punctuation (22 features)
# ============================================================================


def f_apostrophe_curly_rate(text: str) -> float:
    """Binary: any curly apostrophe (U+2019) in prose."""
    return _bin("\u2019" in ad._strip_code(text))


def f_single_quote_pair_unicode_rate(text: str) -> float:
    """Binary: any Unicode single-quoted span (U+2018..U+2019)."""
    return _bin(bool(ad._SINGLE_CURLY_PAIR_RE.search(ad._strip_code(text))))


def f_double_quote_pair_unicode_rate(text: str) -> float:
    """Binary: any Unicode double-quoted span (U+201C..U+201D)."""
    return _bin(bool(ad._DOUBLE_CURLY_PAIR_RE.search(ad._strip_code(text))))


def f_dash_emdash_rate(text: str) -> float:
    """Binary: any em dash (U+2014) in prose."""
    return _bin("\u2014" in ad._strip_code(text))


def f_dash_endash_rate(text: str) -> float:
    """Binary: any en dash (U+2013) in prose."""
    return _bin("\u2013" in ad._strip_code(text))


def f_ellipsis_unicode_rate(text: str) -> float:
    """Binary: any Unicode ellipsis (U+2026) in prose."""
    return _bin("\u2026" in ad._strip_code(text))


def f_minus_unicode_rate(text: str) -> float:
    """Binary: any Unicode minus (U+2212) in inline-math spans."""
    inline_math_blob = "".join(ad._INLINE_MATH_SPANS_RE.findall(text))
    return _bin("\u2212" in inline_math_blob)


def f_nbsp_rate_per_word(text: str) -> float:
    """Binary: any NBSP / NNBSP / thin-space codepoint."""
    return _bin(any(c in text for c in "\u00a0\u202f\u2009"))


def f_invisible_char_presence(text: str) -> float:
    """Binary: any zero-width / BOM / word-joiner present."""
    return _bin(any(c in ad._INVISIBLE_CHARS for c in text))


def f_french_spacing_rate(text: str) -> float:
    """Binary: any space-before-[;:!?] in prose."""
    return _bin(bool(ad._SPACE_BEFORE_PUNCT_RE.search(ad._strip_code(text))))


def f_comma_fullwidth_rate(text: str) -> float:
    """Binary: any fullwidth (CJK) comma."""
    return _bin("\uff0c" in text)


def f_bullet_star_rate(text: str) -> float:
    """Binary: any '*' line-leading bullet marker."""
    return _bin("*" in ad._BULLET_RE.findall(text))


def f_quote_unicode_rate(text: str) -> float:
    """Binary: any curly quote codepoint in prose."""
    prose = ad._strip_code(text)
    return _bin(any(c in prose for c in "\u2018\u2019\u201c\u201d"))


def f_dash_unicode_rate(text: str) -> float:
    """Binary: any Unicode dash form in prose."""
    prose = ad._strip_code(text)
    return _bin(any(c in prose for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"))


def f_special_space_rate_per_word(text: str) -> float:
    """Binary: any NBSP, thin/hair/zero-width/word-joiner space."""
    return _bin(any(c in text for c in ad._SPECIAL_SPACE_CHARS))


def f_post_vowel_apostrophe_curly_rate(text: str) -> float:
    """Binary: any curly apostrophe after a vowel."""
    return _bin(bool(ad._VOWEL_CURLY_APOSTROPHE_RE.search(ad._strip_code(text))))


def f_post_consonant_apostrophe_curly_rate(text: str) -> float:
    """Binary: any curly apostrophe after a consonant."""
    return _bin(bool(ad._CONSONANT_CURLY_APOSTROPHE_RE.search(ad._strip_code(text))))


def f_trailing_space_rate_per_line(text: str) -> float:
    """Binary: any line ending with literal space or tab."""
    return _bin(bool(ad._TRAILING_SPACE_RE.search(text)))


def f_double_space_after_period_rate(text: str) -> float:
    """Binary: any period followed by two or more spaces."""
    return _bin(bool(ad._PERIOD_DOUBLE_SPACE_RE.search(ad._strip_code(text))))


def f_blank_line_between_paragraphs_rate(text: str) -> float:
    """Binary: any blank-line paragraph separator."""
    return _bin(bool(ad._BLANK_LINE_RE.search(text)))


def f_emoji_presence(text: str) -> float:
    """Binary: any emoji / pictographic decoration codepoint present."""
    return _bin(bool(ad._EMOJI_RE.search(text)))


def f_emoji_rate_per_word(text: str) -> float:
    """Binary: any emoji / pictographic decoration codepoint present."""
    return _bin(bool(ad._EMOJI_RE.search(text)))


# ============================================================================
# Group C — Math and LaTeX surface choices (24 features)
# ============================================================================


def f_tex_quote_convention_rate(text: str) -> float:
    """Binary: any TeX-convention ``...'' or `...' quote pair when LaTeX present."""
    return _bin(
        ad._has_latex(text)
        and (
            bool(ad._TEX_QUOTE_DOUBLE_RE.search(text))
            or bool(ad._TEX_QUOTE_SINGLE_RE.search(text))
        )
    )


def f_emdash_ligature_rate(text: str) -> float:
    """Binary: any '---' TeX em-dash ligature."""
    return _bin(bool(ad._EMDASH_LIGATURE_RE.search(text)))


def f_endash_ligature_rate(text: str) -> float:
    """Binary: any '--' TeX en-dash ligature."""
    return _bin(bool(ad._ENDASH_LIGATURE_RE.search(text)))


def f_inline_math_dollar_rate(text: str) -> float:
    """Binary: any inline $...$ math span."""
    return _bin(bool(ad._INLINE_DOLLAR_RE.search(text)))


def f_display_math_dollar_rate(text: str) -> float:
    """Binary: any $$...$$ display-math span."""
    return _bin(bool(ad._DISPLAY_DOLLAR_RE.search(text)))


def f_greek_command_rate(text: str) -> float:
    """Binary: any LaTeX Greek-letter command when LaTeX present."""
    return _bin(ad._has_latex(text) and bool(ad._GREEK_CMD_RE.search(text)))


def f_math_op_command_rate(text: str) -> float:
    """Binary: any LaTeX math-operator command when LaTeX present."""
    return _bin(ad._has_latex(text) and bool(ad._OP_CMD_RE.search(text)))


def f_subscript_brace_rate(text: str) -> float:
    """Binary: any braced subscript x_{...} when LaTeX present."""
    return _bin(ad._has_latex(text) and bool(ad._SUB_BRACED_RE.search(text)))


def f_superscript_brace_rate(text: str) -> float:
    """Binary: any braced superscript x^{...} when LaTeX present."""
    return _bin(ad._has_latex(text) and bool(ad._SUP_BRACED_RE.search(text)))


def f_fraction_command_rate(text: str) -> float:
    """Binary: any \\frac{a}{b} fraction."""
    return _bin(bool(ad._FRAC_CMD_RE.search(text)))


def f_fraction_parenthesized_slash_rate(text: str) -> float:
    """Binary: any parenthesized slash fraction (a)/(b)."""
    return _bin(bool(ad._PAREN_SLASH_FRAC_RE.search(text)))


def f_bracket_left_right_rate(text: str) -> float:
    """Binary: any \\left/\\right bracket sizing when LaTeX present."""
    return _bin(ad._has_latex(text) and bool(ad._LEFT_RIGHT_RE.search(text)))


def f_emphasis_textit_rate(text: str) -> float:
    """Binary: any \\textit{} emphasis command."""
    return _bin(bool(_TEXTIT_RE.search(text)))


def f_cite_plain_rate(text: str) -> float:
    """Binary: any plain \\cite{} command."""
    return _bin(bool(ad._CITE_RE.search(text)))


def f_cite_textual_rate(text: str) -> float:
    """Binary: any textual \\citet{} citation command."""
    return _bin(bool(ad._CITET_RE.search(text)))


def f_display_math_bracket_rate(text: str) -> float:
    """Binary: any \\[...\\] display-math delimiter."""
    return _bin(bool(ad._DISPLAY_BRACKET_RE.search(text)))


def f_display_math_environment_rate(text: str) -> float:
    """Binary: any equation/align-like display-math environment."""
    return _bin(bool(ad._DISPLAY_BEGIN_RE.search(text)))


def f_operator_spacing_density_per_equation(text: str) -> float:
    """Binary: any TeX spacing command (\\,, \\;, \\quad, etc.)."""
    return _bin(bool(ad._MATH_SPACING_CMD_RE.search(text)))


def f_boxed_expression_rate(text: str) -> float:
    """Binary: any \\boxed{} expression."""
    return _bin(bool(_BOXED_RE.search(text)))


def f_therefore_implies_rate(text: str) -> float:
    """Binary: any \\therefore or \\implies proof connector."""
    return _bin(bool(_THEREFORE_IMPLIES_RE.search(text)))


def f_equation_alignment_ampersand_rate(text: str) -> float:
    """Binary: any display environment containing an alignment ampersand."""
    return _bin(_has_aligned_env_ampersand(text))


def f_equation_linebreak_density(text: str) -> float:
    """Binary: any LaTeX equation line break (\\\\)."""
    return _bin("\\\\" in text)


def f_reference_autoref_rate(text: str) -> float:
    """Binary: any \\autoref{} reference command."""
    return _bin(bool(ad._AUTOREF_RE.search(text)))


def f_reference_eqref_rate(text: str) -> float:
    """Binary: any \\eqref{} reference command."""
    return _bin(bool(ad._EQREF_RE.search(text)))


# ============================================================================
# Group B — Markdown structure (25 features)
# ============================================================================


def f_bold_double_star_rate(text: str) -> float:
    """Binary: any **bold** span."""
    return _bin(bool(ad._BOLD_STAR_RE.search(text)))


def f_italic_star_rate(text: str) -> float:
    """Binary: any *italic* span (after masking bold)."""
    masked = ad._BOLD_STAR_RE.sub("", text)
    masked = ad._BOLD_UNDER_RE.sub("", masked)
    return _bin(bool(ad._ITALIC_STAR_RE.search(masked)))


def f_heading_atx_rate(text: str) -> float:
    """Binary: any ATX (# H1) heading."""
    return _bin(bool(ad._ATX_HEADING_RE.search(text)))


def f_code_fence_lang_tag_rate(text: str) -> float:
    """Binary: any opening code fence with a language tag."""
    return _bin(bool(ad._FENCE_LANG_TAG_RE.search(text)))


def f_code_fence_backtick_rate(text: str) -> float:
    """Binary: any backtick code fence (```)."""
    return _bin(bool(ad._BACKTICK_FENCE_RE.search(text)))


def f_link_inline_rate(text: str) -> float:
    """Binary: any [text](url) inline link."""
    return _bin(bool(ad._INLINE_LINK_RE.search(text)))


def f_numbered_dot_rate(text: str) -> float:
    """Binary: any 1. numbered-list opener."""
    return _bin(bool(ad._NUM_DOT_RE.search(text)))


def f_hardbreak_double_space_rate(text: str) -> float:
    """Binary: any double-space hard break (  \\n)."""
    return _bin(bool(ad._DOUBLE_SPACE_BREAK_RE.search(text)))


def f_hardbreak_backslash_rate(text: str) -> float:
    """Binary: any backslash-newline hard break."""
    return _bin(bool(ad._BACKSLASH_BREAK_RE.search(text)))


def f_hardbreak_blank_line_rate(text: str) -> float:
    """Binary: any blank-line paragraph break."""
    return _bin(bool(ad._BLANK_LINE_RE.search(text)))


def f_bullet_dash_rate(text: str) -> float:
    """Binary: any '-' line-leading bullet marker."""
    return _bin("-" in ad._BULLET_RE.findall(text))


def f_bullet_plus_rate(text: str) -> float:
    """Binary: any '+' line-leading bullet marker."""
    return _bin("+" in ad._BULLET_RE.findall(text))


def f_bullet_unicode_rate(text: str) -> float:
    """Binary: any Unicode bullet marker."""
    return _bin(bool(ad._UNICODE_BULLET_RE.search(text)))


def f_numbered_roman_rate(text: str) -> float:
    """Binary: any roman-numeral numbered-list opener."""
    return _bin(bool(ad._NUM_ROMAN_RE.search(text)))


def f_numbered_alpha_rate(text: str) -> float:
    """Binary: any alpha-letter numbered-list opener."""
    return _bin(bool(ad._NUM_ALPHA_RE.search(text)))


def f_blank_line_before_heading_rate(text: str) -> float:
    """Binary: any heading preceded by a blank line or start of file."""
    return _bin(_has_blank_line_before_heading(text))


def f_code_fence_tilde_rate(text: str) -> float:
    """Binary: any tilde code fence (~~~)."""
    return _bin(bool(ad._TILDE_FENCE_RE.search(text)))


def f_indented_code_block_rate(text: str) -> float:
    """Binary: any four-space/tab indented code block."""
    return _bin(bool(ad._INDENTED_CODE_RE.search(ad._CODE_FENCE_RE.sub("", text))))


def f_inline_code_double_backtick_rate(text: str) -> float:
    """Binary: any inline code span delimited with double backticks."""
    return _bin(bool(ad._INLINE_CODE_DOUBLE_RE.search(text)))


def f_image_inline_rate(text: str) -> float:
    """Binary: any inline ![alt](url) image."""
    return _bin(bool(ad._IMAGE_INLINE_RE.search(text)))


def f_blockquote_nested_rate(text: str) -> float:
    """Binary: any nested blockquote line (depth > 1)."""
    return _bin(_has_nested_blockquote(text))


def f_blank_line_after_blockquote_rate(text: str) -> float:
    """Binary: any blockquote block followed by a blank line or EOF."""
    return _bin(_has_blank_line_after_blockquote(text))


def f_table_cell_padding_rate(text: str) -> float:
    """Binary: any Markdown table cell padded with spaces on both sides."""
    return _bin(_has_padded_table_cell(text))


def f_table_center_align_rate(text: str) -> float:
    """Binary: any centered Markdown table alignment cell (:---:)."""
    return _bin(bool(ad._TABLE_ALIGN_CELL_RE.search(text)))


def f_markdown_escape_density_per_kb(text: str) -> float:
    """Binary: any Markdown escape for *, _, or backticks."""
    return _bin(bool(ad._MD_ESCAPE_RE.search(text)))


# ============================================================================
# Group D — Code surface (20 features)
# ============================================================================


def f_code_string_single_quote_rate(text: str) -> float:
    """Binary: any single-quoted code string literal."""
    code = ad._code_blob(text)
    return _bin(bool(ad._PY_SINGLE_STRING_RE.search(code)))


def f_code_indent_2space_rate(text: str) -> float:
    """Binary: any code line indented with two spaces."""
    code = ad._code_blob(text)
    indents = re.findall(r"^( +|\t+)\S", code, flags=re.M)
    return _bin(any(indent == "  " for indent in indents))


def f_code_indent_4space_rate(text: str) -> float:
    """Binary: any code line indented with four spaces."""
    code = ad._code_blob(text)
    indents = re.findall(r"^( +|\t+)\S", code, flags=re.M)
    return _bin(any(indent == "    " for indent in indents))


def f_code_indent_tab_rate(text: str) -> float:
    """Binary: any tab-indented code line."""
    code = ad._code_blob(text)
    indents = re.findall(r"^( +|\t+)\S", code, flags=re.M)
    return _bin(any(indent.startswith("\t") for indent in indents))


def f_code_comment_hash_rate(text: str) -> float:
    """Binary: any Python/shell # comment in code."""
    code = ad._code_blob(text)
    return _bin(bool(ad._HASH_COMMENT_RE.search(code)))


def f_code_comment_slash_rate(text: str) -> float:
    """Binary: any // comment in code."""
    code = ad._code_blob(text)
    return _bin(bool(ad._SLASH_COMMENT_RE.search(code)))


def f_code_comment_block_rate(text: str) -> float:
    """Binary: any /* ... */ block comment in code."""
    code = ad._code_blob(text)
    return _bin(bool(ad._BLOCK_COMMENT_RE.search(code)))


def f_code_comment_sql_dash_rate(text: str) -> float:
    """Binary: any SQL-style -- comment in code."""
    code = ad._code_blob(text)
    return _bin(bool(ad._SQL_COMMENT_RE.search(code)))


def f_type_annotation_spaced_rate(text: str) -> float:
    """Binary: any spaced type annotation (x: int)."""
    code = ad._code_blob(text)
    return _bin(bool(_TYPE_ANNOTATION_SPACED_RE.search(code)))


def f_fstring_double_quote_rate(text: str) -> float:
    """Binary: any double-quoted f-string in code."""
    code = ad._code_blob(text)
    return _bin(bool(ad._FSTRING_DOUBLE_RE.search(code)))


def f_fstring_single_quote_rate(text: str) -> float:
    """Binary: any single-quoted f-string in code."""
    code = ad._code_blob(text)
    return _bin(bool(ad._FSTRING_SINGLE_RE.search(code)))


def f_format_method_rate(text: str) -> float:
    """Binary: any .format() call in code."""
    code = ad._code_blob(text)
    return _bin(bool(ad._FORMAT_METHOD_RE.search(code)))


def f_percent_format_rate(text: str) -> float:
    """Binary: any % string interpolation in code."""
    code = ad._code_blob(text)
    return _bin(bool(ad._PERCENT_FORMAT_RE.search(code)))


def f_code_trailing_comma_rate(text: str) -> float:
    """Binary: any trailing comma in a collection/call literal."""
    code = ad._code_blob(text)
    return _bin(bool(ad._TRAILING_COMMA_RE.search(code)))


def f_docstring_presence_rate(text: str) -> float:
    """Binary: any triple-quoted docstring in code."""
    code = ad._code_blob(text)
    return _bin(
        bool(ad._TRIPLE_DOUBLE_RE.search(code) or ad._TRIPLE_SINGLE_RE.search(code))
    )


def f_docstring_triple_double_rate(text: str) -> float:
    """Binary: any triple-double-quoted docstring in code."""
    code = ad._code_blob(text)
    return _bin(bool(ad._TRIPLE_DOUBLE_RE.search(code)))


def f_docstring_first_line_summary_rate(text: str) -> float:
    """Binary: any docstring whose first line is a short summary sentence."""
    return _bin(_has_summary_docstring(ad._code_blob(text)))


def f_lambda_over_def_rate(text: str) -> float:
    """Binary: any lambda expression in code."""
    code = ad._code_blob(text)
    return _bin(bool(_LAMBDA_RE.search(code)))


def f_listcomp_over_loop_rate(text: str) -> float:
    """Binary: any list comprehension in code."""
    code = ad._code_blob(text)
    return _bin(bool(_LISTCOMP_RE.search(code)))


def f_import_blankline_group_rate(text: str) -> float:
    """Binary: any adjacent import blocks separated by a blank line."""
    return _bin(_has_separated_import_groups(ad._code_blob(text)))


# ============================================================================
# Group E — Whitespace and invisible characters (6 features)
# ============================================================================


def f_crlf_newline_rate(text: str) -> float:
    """Binary: any CRLF newline encoding."""
    return _bin("\r\n" in text)


def f_tokenizer_fragmentation_proxy_per_kb(text: str) -> float:
    """Binary: any non-ASCII codepoint or markup punctuation fragment."""
    return _bin(
        any(ord(c) > 127 for c in text) or bool(_MARKUP_FRAGMENT_RE.search(text))
    )


def f_leading_tab_rate(text: str) -> float:
    """Binary: any indented line whose leading indentation contains tabs."""
    indents = re.findall(r"^([ \t]+)\S", text, flags=re.M)
    return _bin(any("\t" in indent for indent in indents))


def f_indent_consistency_rate(text: str) -> float:
    """Binary: any leading indentation on a line."""
    indents = re.findall(r"^([ \t]+)\S", text, flags=re.M)
    return _bin(bool(indents))


def f_final_newline_present(text: str) -> float:
    """Binary: text ends with a newline."""
    if not text:
        return 0.0
    return _bin(text.endswith("\n"))


def f_section_blank_line_rate(text: str) -> float:
    """Binary: blank lines present alongside section/list boundaries."""
    return _bin(
        bool(ad._BLANK_LINE_RE.search(text)) and bool(_SECTION_BOUNDARY_RE.search(text))
    )


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
    """Extract the binary ParaPunct feature vector from a string.

    Returns
    -------
    np.ndarray of shape (n_enabled_features,), dtype=float32
        Each entry is 0.0 or 1.0 indicating presence of the feature signal.
    """
    feature_funcs = _get_feature_funcs(disabled_feature_groups)
    if not text:
        return np.zeros(len(feature_funcs), dtype=np.float32)
    return np.array([f(text) for f in feature_funcs], dtype=np.float32)


def extract_feature_matrix(
    texts: Sequence[str],
    disabled_feature_groups: Sequence[str] | str | None = None,
) -> np.ndarray:
    """Extract a feature matrix with shape (n_texts, n_enabled_features)."""
    feature_funcs = _get_feature_funcs(disabled_feature_groups)
    X = np.empty((len(texts), len(feature_funcs)), dtype=np.float32)
    for i, text in enumerate(texts):
        X[i] = extract_features(text, disabled_feature_groups)
    return X


class ASCIIDetectorBinary(BaseDetector):
    """Decision-tree detector over binary ParaPunct features.

    Config keys match :class:`detectors.ascii_detector.ASCIIDetector`.
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
            raise RuntimeError(
                "ASCIIDetectorBinary training requires scikit-learn."
            ) from exc

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
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        n = len(texts)
        if n == 0:
            return

        n_chunks = (n + batch_size - 1) // batch_size
        max_workers = max(1, min(self.max_workers, n_chunks))
        pbar = (
            tqdm.tqdm(total=n_chunks, desc="ASCIIDetectorBinary", unit="batch")
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
                        start = next(starts)
                    except StopIteration:
                        return False
                    pending.append(
                        executor.submit(
                            extract_feature_matrix,
                            texts[start : start + batch_size],
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

    def fit(self, texts: Sequence[str], labels: Sequence[Any]) -> "ASCIIDetectorBinary":
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
            raise RuntimeError("Cannot save ASCIIDetectorBinary before fit/load.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str | Path) -> "ASCIIDetectorBinary":
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
                "ASCIIDetectorBinary needs a trained model. Call fit() or set model_path."
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
                "ASCIIDetectorBinary needs a trained model. Call fit() or set model_path."
            )
        if not x:
            return []

        scores: List[float] = []
        for X in self._iter_feature_chunks(x, batch_size, show_progress=show_progress):
            scores.extend(self._scores_from_features(X).tolist())
        return scores
