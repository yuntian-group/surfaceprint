"""Binary ASCII/style detector using 45 discovery features (Experiments 2a/2b).

Each feature is a binary presence check: "does this text contain X?"
Groups: char (15), word (19), pattern (11).
"""

from __future__ import annotations

import pickle
import re
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator, List, Sequence

import numpy as np
import tqdm

from detectors.base_detector import BaseDetector

# --- Character features (Experiment 2a) ------------------------------------
CHAR_FEATURES_LLM: tuple[tuple[str, str], ...] = (
    ("`", "grave_accent"),
    ("\n", "newline"),
    ("'", "apostrophe"),
    ("!", "exclamation"),
    ("-", "hyphen_minus"),
    ("\u2013", "en_dash"),
    ("\u2014", "em_dash"),
)

CHAR_FEATURES_HUMAN: tuple[tuple[str, str], ...] = (
    ("\u2018", "left_single_curly_quote"),
    ("\u2019", "right_single_curly_quote"),
    ("\u201c", "left_double_curly_quote"),
    ("\u201d", "right_double_curly_quote"),
    ("\t", "tab"),
    ("~", "tilde"),
    ("\u2026", "ellipsis_char"),
    ("?", "question_mark"),
)

CHAR_FEATURES: tuple[tuple[str, str], ...] = CHAR_FEATURES_LLM + CHAR_FEATURES_HUMAN
_CHAR_LOOKUP: tuple[str, ...] = tuple(char for char, _ in CHAR_FEATURES)

# --- Word features (Experiment 2b) -------------------------------------------
LLM_WORDS: tuple[str, ...] = (
    "here's",
    "apologize",
    "couldn't",
    "hello!",
    "certainly!",
    "sure!",
    "apologies",
    "confusion.",
    "respectful",
    "wicked",
)

HUMAN_WORDS: tuple[str, ...] = (
    "doesn't",
    "isn't",
    "don't",
    "write",
    "rewrite",
    "basically",
    "reply",
    "i'll",
    "lots",
)

WORD_FEATURES: tuple[tuple[str, str], ...] = tuple(
    (word, f"word_{word.strip('!.,').replace(' ', '_')}") for word in LLM_WORDS
) + tuple(
    (
        word,
        f"word_{word.strip('!.,').replace(' ', '_').replace(chr(8217), 'q')}",
    )
    for word in HUMAN_WORDS
)
_WORD_LOOKUP: tuple[str, ...] = tuple(word for word, _ in WORD_FEATURES)

# --- Formatting pattern features (Experiment 2b) -----------------------------
_GREETING_RE = re.compile(
    r"^(Sure|Certainly|Of course|Absolutely|Great question|I'd be happy to|"
    r"I can help|Here's|Here is|That's a great|Thank you for)",
    re.IGNORECASE,
)
_NUMBERED_LIST_RE = re.compile(r"^\d+\.\s+\S", re.MULTILINE)
_BULLET_LIST_RE = re.compile(r"^[-*]\s+\S", re.MULTILINE)
_CODE_BLOCK_RE = re.compile(r"```")
_MARKDOWN_HEADER_RE = re.compile(r"^#{1,4}\s+\S", re.MULTILINE)
_MARKDOWN_BOLD_RE = re.compile(r"\*\*[^*]+\*\*")
_COLON_HEADER_RE = re.compile(r"^[A-Z][^.!?\n]{2,40}:\s*$", re.MULTILINE)
_BOLD_LIST_ITEM_RE = re.compile(
    r"^[-*]\s+\*\*[^*]+\*\*[:\s\u2014-]|^\d+\.\s+\*\*[^*]+\*\*[:\s\u2014-]",
    re.MULTILINE,
)
_MARKDOWN_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)[^*\n]+\*(?!\*)")

_STRUCTURAL_SYMBOLS = frozenset(
    "\u2713\u2714\u2705" "\u274c\u274e\u2718" "\u26a0" "\u2192\u2190\u21d2"
)

_PATTERN_NAMES: tuple[str, ...] = (
    "pat_starts_greeting",
    "pat_numbered_list",
    "pat_bullet_list",
    "pat_code_block",
    "pat_markdown_header",
    "pat_markdown_bold",
    "pat_colon_header",
    "pat_bold_list_item",
    "pat_markdown_italic",
    "pat_excessive_formatting",
    "pat_structural_symbol",
)

FEATURE_NAMES: List[str] = (
    [f"char_{name}" for _, name in CHAR_FEATURES]
    + [name for _, name in WORD_FEATURES]
    + list(_PATTERN_NAMES)
)
FEATURE_GROUPS: List[str] = (
    ["char"] * len(CHAR_FEATURES)
    + ["word"] * len(WORD_FEATURES)
    + ["pattern"] * len(_PATTERN_NAMES)
)
FEATURE_DIM = len(FEATURE_NAMES)
FEATURE_GROUP_ALIASES = {
    "character": "char",
    "characters": "char",
    "words": "word",
    "patterns": "pattern",
    "formatting": "pattern",
}


def _normalize_feature_group(group: str) -> str:
    normalized = str(group).strip().lower()
    return FEATURE_GROUP_ALIASES.get(normalized, normalized)


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


def _extract_all_features(text: str) -> np.ndarray:
    """Extract all FEATURE_DIM binary features in a single pass over *text*."""
    out = np.zeros(FEATURE_DIM, dtype=np.float32)
    if not text:
        return out

    idx = 0
    text_chars = set(text)
    for char in _CHAR_LOOKUP:
        out[idx] = char in text_chars
        idx += 1

    token_set = set(text.lower().split())
    for word in _WORD_LOOKUP:
        out[idx] = word in token_set
        idx += 1

    has_bullet = bool(_BULLET_LIST_RE.search(text))
    has_numbered = bool(_NUMBERED_LIST_RE.search(text))
    has_header = bool(_MARKDOWN_HEADER_RE.search(text))
    has_code = bool(_CODE_BLOCK_RE.search(text))
    out[idx] = float(bool(_GREETING_RE.match(text)))
    out[idx + 1] = float(has_numbered)
    out[idx + 2] = float(has_bullet)
    out[idx + 3] = float(has_code)
    out[idx + 4] = float(has_header)
    out[idx + 5] = float(bool(_MARKDOWN_BOLD_RE.search(text)))
    out[idx + 6] = float(bool(_COLON_HEADER_RE.search(text)))
    out[idx + 7] = float(bool(_BOLD_LIST_ITEM_RE.search(text)))
    out[idx + 8] = float(bool(_MARKDOWN_ITALIC_RE.search(text)))
    out[idx + 9] = float(
        sum(
            (
                "**" in text,
                has_bullet,
                has_header,
                has_code,
                has_numbered,
            )
        )
        >= 3
    )
    out[idx + 10] = float(bool(text_chars & _STRUCTURAL_SYMBOLS))

    return out


def extract_features(
    text: str,
    disabled_feature_groups: Sequence[str] | str | None = None,
) -> np.ndarray:
    """Extract the binary feature vector from a string."""
    indices = get_feature_indices(disabled_feature_groups)
    full = _extract_all_features(text)
    if len(indices) == FEATURE_DIM:
        return full
    return full[np.asarray(indices, dtype=np.intp)]


def extract_feature_matrix(
    texts: Sequence[str],
    disabled_feature_groups: Sequence[str] | str | None = None,
) -> np.ndarray:
    """Extract a feature matrix with shape (n_texts, n_enabled_features)."""
    indices = get_feature_indices(disabled_feature_groups)
    n_features = len(indices)
    X = np.empty((len(texts), n_features), dtype=np.float32)
    if n_features == FEATURE_DIM:
        for i, text in enumerate(texts):
            X[i] = _extract_all_features(text)
    else:
        index_array = np.asarray(indices, dtype=np.intp)
        for i, text in enumerate(texts):
            X[i] = _extract_all_features(text)[index_array]
    return X


class ASCIIDetectorBasic(BaseDetector):
    """Decision-tree detector over 45 binary discovery features.

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
                "ASCIIDetectorBasic training requires scikit-learn."
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
            tqdm.tqdm(total=n_chunks, desc="ASCIIDetectorBasic", unit="batch")
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

    def fit(self, texts: Sequence[str], labels: Sequence[Any]) -> "ASCIIDetectorBasic":
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
            raise RuntimeError("Cannot save ASCIIDetectorBasic before fit/load.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str | Path) -> "ASCIIDetectorBasic":
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
                "ASCIIDetectorBasic needs a trained model. Call fit() or set model_path."
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
                "ASCIIDetectorBasic needs a trained model. Call fit() or set model_path."
            )
        if not x:
            return []

        scores: List[float] = []
        for chunk_X in self._iter_feature_chunks(
            x, batch_size, show_progress=show_progress
        ):
            scores.extend(self._scores_from_features(chunk_X).tolist())
        return scores


__all__ = [
    "ASCIIDetectorBasic",
    "FEATURE_DIM",
    "FEATURE_GROUPS",
    "FEATURE_NAMES",
    "extract_feature_matrix",
    "extract_features",
    "get_feature_indices",
    "get_feature_names",
]
