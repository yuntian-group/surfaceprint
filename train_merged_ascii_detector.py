from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Type

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import Dataset, DatasetDict, load_from_disk

from detectors.base_detector import BaseDetector

ASCII_DETECTOR_CHOICES = ("full", "basic", "binary")


def resolve_ascii_detector_class(name: str) -> Type[BaseDetector]:
    if name == "full":
        from detectors.ascii_detector import ASCIIDetector

        return ASCIIDetector
    if name == "basic":
        from detectors.ascii_detector_basic import ASCIIDetectorBasic

        return ASCIIDetectorBasic
    if name == "binary":
        from detectors.ascii_detector_binary import ASCIIDetectorBinary

        return ASCIIDetectorBinary
    raise ValueError(
        f"Unknown ascii detector {name!r}; expected one of {ASCII_DETECTOR_CHOICES}."
    )


def build_ascii_detector(args: argparse.Namespace) -> BaseDetector:
    detector_cls = resolve_ascii_detector_class(args.ascii_detector)
    return detector_cls(make_detector_config(args))


def add_ascii_detector_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ascii-detector",
        choices=ASCII_DETECTOR_CHOICES,
        default="full",
        help=(
            "'full': ParaPunct/rate features (detectors.ascii_detector.ASCIIDetector). "
            "'basic': 45 binary discovery features "
            "(detectors.ascii_detector_basic.ASCIIDetectorBasic). "
            "'binary': 97 binary ParaPunct presence features "
            "(detectors.ascii_detector_binary.ASCIIDetectorBinary)."
        ),
    )


def load_merged_dataset(path: str | Path) -> DatasetDict:
    data = load_from_disk(str(path))
    if not isinstance(data, DatasetDict):
        raise ValueError(f"Expected DatasetDict at {path}")
    return data


def parse_filters(values: list[str]) -> list[tuple[str, str]]:
    filters: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected COLUMN=VALUE filter, got {value!r}")
        column, expected = value.split("=", 1)
        column = column.strip()
        if not column:
            raise ValueError(f"Expected non-empty filter column in {value!r}")
        filters.append((column, expected))
    return filters


def apply_filters(
    ds: Dataset,
    *,
    include_dataset: list[str],
    exclude_dataset: list[str],
    include_source: list[str],
    exclude_source: list[str],
    filters: list[tuple[str, str]],
    num_proc: int | None,
) -> Dataset:
    required_columns = {column for column, _ in filters}
    if include_dataset or exclude_dataset:
        required_columns.add("dataset")
    if include_source or exclude_source:
        required_columns.add("source")

    missing = [column for column in required_columns if column not in ds.column_names]
    if missing:
        raise ValueError(f"Split is missing required filter columns: {missing}")

    if not required_columns:
        return ds

    include_dataset_set = set(include_dataset)
    exclude_dataset_set = set(exclude_dataset)
    include_source_set = set(include_source)
    exclude_source_set = set(exclude_source)
    filter_num_proc = num_proc if num_proc and num_proc > 1 else None

    def keep(row: dict) -> bool:
        dataset_value = row.get("dataset")
        source_value = row.get("source")
        return (
            (not include_dataset_set or dataset_value in include_dataset_set)
            and dataset_value not in exclude_dataset_set
            and (not include_source_set or source_value in include_source_set)
            and source_value not in exclude_source_set
            and all(str(row[column]) == expected for column, expected in filters)
        )

    return ds.filter(keep, num_proc=filter_num_proc)


def build_labels(ds: Dataset, label_mode: str, target_source: str | None) -> list[int]:
    if "source" not in ds.column_names:
        raise ValueError("Merged dataset must contain a 'source' column for labels.")

    if label_mode == "machine":
        return [0 if source == "human" else 1 for source in ds["source"]]

    if label_mode == "source":
        if not target_source:
            raise ValueError("--target-source is required when --label-mode=source.")
        return [1 if source == target_source else 0 for source in ds["source"]]

    raise ValueError(f"Unknown label mode: {label_mode}")


def normalize_whitespace(text: str) -> str:
    """D-NORM text view: delete space and newline shortcuts."""
    return text.translate(str.maketrans("", "", " \t\r\n"))


def normalize_text(text: str, mode: str) -> str:
    if mode == "raw":
        return text.strip()
    if mode == "whitespace":
        return normalize_whitespace(text)
    raise ValueError(f"Unknown normalization mode: {mode}")


def split_texts(ds: Dataset, text_field: str, normalize_mode: str) -> list[str]:
    if text_field == "prompt_response":
        missing = [
            column for column in ("prompt", "response") if column not in ds.column_names
        ]
        if missing:
            raise ValueError(f"Split is missing required text columns: {missing}")
        return [
            normalize_text(f"{a}\n\n{b}", normalize_mode)
            for a, b in zip(ds["prompt"], ds["response"])
        ]

    if text_field not in ds.column_names:
        raise ValueError(f"Split is missing text field {text_field!r}")
    return [normalize_text(str(text), normalize_mode) for text in ds[text_field]]


def build_split(
    data: DatasetDict,
    split_name: str,
    args: argparse.Namespace,
    label: str,
) -> tuple[list[str], list[int]]:
    if split_name not in data:
        raise ValueError(f"Expected split {split_name!r}; got {list(data.keys())}")

    ds = apply_filters(
        data[split_name],
        include_dataset=args.include_dataset,
        exclude_dataset=args.exclude_dataset,
        include_source=args.include_source,
        exclude_source=args.exclude_source,
        filters=parse_filters(args.filter),
        num_proc=args.filter_num_proc,
    )
    texts = split_texts(ds, args.text_field, args.normalize_mode)
    labels = build_labels(ds, args.label_mode, args.target_source)
    if not texts:
        raise ValueError(f"No rows left in {split_name!r} split after filtering.")

    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2)
    print(
        f"Loaded {label} split {split_name!r}: n={len(texts)}, "
        f"negative={int(counts[0])}, positive={int(counts[1])}, "
        f"normalize_mode={args.normalize_mode}"
    )
    if counts[0] == 0 or counts[1] == 0:
        raise ValueError(
            "Training/evaluation needs both labels after filtering; got "
            f"negative={int(counts[0])}, positive={int(counts[1])}."
        )
    return texts, labels


def make_detector_config(args: argparse.Namespace) -> dict[str, Any]:
    base: dict[str, Any] = {
        "positive_class": 1,
        "disabled_feature_groups": args.disable_feature_group,
        "num_workers": args.num_workers,
        "fit_batch_size": args.fit_batch_size,
        "fit_show_progress": True,
    }
    if args.classifier == "tree":
        base.update(
            {
                "max_depth": args.max_depth,
                "min_samples_leaf": args.min_samples_leaf,
                "random_state": args.random_state,
                "class_weight": args.class_weight,
            }
        )
        return base

    if args.classifier == "forest":
        base["model"] = RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            class_weight=args.class_weight,
            n_jobs=args.n_jobs,
        )
        return base

    if args.classifier == "logreg":
        base["model"] = LogisticRegression(
            max_iter=args.max_iter,
            random_state=args.random_state,
            class_weight=args.class_weight,
            solver="lbfgs",
            C=args.logreg_c,
        )
        return base

    raise ValueError(f"Unknown classifier: {args.classifier!r}")


def train_metrics(
    detector: BaseDetector,
    texts: list[str],
    labels: list[int],
    batch_size: int = 4096,
) -> dict[str, float]:
    scores = np.asarray(
        detector.batch_predict(texts, batch_size=batch_size, show_progress=True),
        dtype=np.float64,
    )
    labels_arr = np.asarray(labels)
    preds = scores >= 0.5

    return {
        "accuracy": float(accuracy_score(labels_arr, preds)),
        "log_loss": float(log_loss(labels_arr, scores, labels=[0, 1])),
        "auroc": float(roc_auc_score(labels_arr, scores)),
        "auprc": float(average_precision_score(labels_arr, scores)),
        "f1": float(f1_score(labels_arr, preds, zero_division=0)),
    }


def train_from_merged_dataset(args: argparse.Namespace) -> None:
    data = load_merged_dataset(args.dataset_path)
    train_texts, train_labels = build_split(data, args.split, args, "train")

    detector = build_ascii_detector(args)
    print(
        f"Training ASCII binary detector ({args.ascii_detector}, {args.classifier}) "
        f"with {detector.feature_dim} features..."
    )
    detector.fit(train_texts, train_labels)
    metrics = train_metrics(
        detector, train_texts, train_labels, args.predict_batch_size
    )
    print(
        "Train metrics: "
        f"accuracy={metrics['accuracy']:.4f}, log_loss={metrics['log_loss']:.4f}, "
        f"auroc={metrics['auroc']:.4f}, auprc={metrics['auprc']:.4f}, "
        f"f1={metrics['f1']:.4f}"
    )
    n_train = len(train_texts)
    del train_texts, train_labels

    val_metrics: dict[str, float] | None = None
    if args.val_split:
        val_texts, val_labels = build_split(data, args.val_split, args, "validation")
        val_metrics = train_metrics(
            detector, val_texts, val_labels, args.predict_batch_size
        )
        print(
            "Validation metrics: "
            f"accuracy={val_metrics['accuracy']:.4f}, "
            f"log_loss={val_metrics['log_loss']:.4f}, "
            f"auroc={val_metrics['auroc']:.4f}, "
            f"auprc={val_metrics['auprc']:.4f}, "
            f"f1={val_metrics['f1']:.4f}"
        )
        del val_texts, val_labels

    detector.save(args.output)
    print(f"Saved ASCII detector to {args.output} (n_train={n_train})")

    if args.metrics_json:
        payload: dict[str, object] = {
            "ascii_detector": args.ascii_detector,
            "classifier": args.classifier,
            "target_source": args.target_source,
            "output": str(args.output),
            "n_train": n_train,
            "train": metrics,
            "validation": val_metrics,
            "dataset_path": str(args.dataset_path),
            "split": args.split,
            "val_split": args.val_split,
            "normalize_mode": args.normalize_mode,
            "label_mode": args.label_mode,
            "include_dataset": list(args.include_dataset),
            "include_source": list(args.include_source),
            "disable_feature_group": list(args.disable_feature_group),
        }
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote metrics to {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an ASCII detector from the merged multi-source dataset."
    )
    parser.add_argument("--dataset-path", default="dataset/merged_dataset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--text-field",
        default="response",
        help="Column to train on, or 'prompt_response' to join prompt and response.",
    )
    parser.add_argument(
        "--normalize-mode",
        choices=["raw", "whitespace"],
        default="raw",
        help=(
            "'raw' trains on original text. 'whitespace' normalizes line endings, "
            "trailing whitespace, repeated inline spaces, and 3+ newlines for D-NORM."
        ),
    )
    parser.add_argument(
        "--label-mode",
        choices=["machine", "source"],
        default="machine",
        help="'machine': source != human. 'source': source == --target-source.",
    )
    parser.add_argument(
        "--target-source",
        default=None,
        help="Positive source/model name for --label-mode=source.",
    )
    parser.add_argument(
        "--include-dataset",
        nargs="+",
        default=[],
        help="Keep only these dataset values, e.g. writingprompts reddit_eli5.",
    )
    parser.add_argument(
        "--exclude-dataset",
        nargs="+",
        default=[],
        help="Drop these dataset values.",
    )
    parser.add_argument(
        "--include-source",
        nargs="+",
        default=[],
        help="Keep only these source values, e.g. human facebook/opt-1.3b.",
    )
    parser.add_argument(
        "--exclude-source",
        nargs="+",
        default=[],
        help="Drop these source values.",
    )
    parser.add_argument(
        "--filter",
        nargs="+",
        action="extend",
        default=[],
        help="Extra exact-match filters as COLUMN=VALUE. Can be repeated.",
    )
    parser.add_argument("--filter-num-proc", type=int, default=None)
    add_ascii_detector_arg(parser)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--fit-batch-size", type=int, default=1024)
    parser.add_argument("--predict-batch-size", type=int, default=4096)
    parser.add_argument(
        "--classifier",
        choices=["tree", "forest", "logreg"],
        default="tree",
        help="Sklearn head on ASCII features: single tree, random forest, or logistic regression.",
    )
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--class-weight", default="balanced")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument(
        "--logreg-c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for --classifier=logreg.",
    )
    parser.add_argument(
        "--disable-feature-group",
        nargs="+",
        action="extend",
        default=[],
        help=(
            "Disable feature groups by label or alias. Can be repeated. "
            "For --ascii-detector=full or binary: E, whitespace, A, prose, B, "
            "markdown, C, math, D, code. For --ascii-detector=basic: char, word, "
            "pattern."
        ),
    )
    parser.add_argument(
        "--metrics-json",
        default=None,
        help="If set, write train/validation metrics and run metadata to this JSON path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_from_merged_dataset(parse_args())
