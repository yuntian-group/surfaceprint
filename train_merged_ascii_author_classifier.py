from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
    top_k_accuracy_score,
)

from detectors.base_detector import BaseDetector

# Sibling module: script directory is on sys.path when invoked as
# `python scripts/train_merged_ascii_author_classifier.py`.
import train_merged_ascii_detector as tmd  # noqa: E402


def json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _filtered_ds(
    data: DatasetDict, split_name: str, args: argparse.Namespace
) -> Dataset:
    if split_name not in data:
        raise ValueError(f"Expected split {split_name!r}; got {list(data.keys())}")
    return tmd.apply_filters(
        data[split_name],
        include_dataset=args.include_dataset,
        exclude_dataset=args.exclude_dataset,
        include_source=args.include_source,
        exclude_source=args.exclude_source,
        filters=tmd.parse_filters(args.filter),
        num_proc=args.filter_num_proc,
    )


def valid_sources_for_train(ds: Dataset, min_samples_per_class: int) -> frozenset[str]:
    if "source" not in ds.column_names:
        raise ValueError("Merged dataset must contain a 'source' column for labels.")
    counts = Counter(ds["source"])
    valid = frozenset(s for s, n in counts.items() if n >= min_samples_per_class)
    dropped = sorted(s for s, n in counts.items() if n < min_samples_per_class)
    if dropped:
        print(
            f"Dropping {len(dropped)} class(es) with fewer than "
            f"{min_samples_per_class} example(s): {dropped}"
        )
    if len(valid) < 2:
        raise ValueError(
            "Need at least two classes with >= min-samples-per-class after "
            f"filtering; got {len(valid)}."
        )
    return valid


def _keep_source_in(valid: frozenset[str]):
    def keep(row: dict) -> bool:
        return row.get("source") in valid

    return keep


def build_split(
    data: DatasetDict,
    split_name: str,
    args: argparse.Namespace,
    label: str,
    valid_sources: frozenset[str],
    *,
    allow_empty: bool = False,
) -> tuple[list[str], list[str]]:
    ds = _filtered_ds(data, split_name, args)
    filter_num_proc = (
        args.filter_num_proc
        if args.filter_num_proc and args.filter_num_proc > 1
        else None
    )
    ds = ds.filter(_keep_source_in(valid_sources), num_proc=filter_num_proc)
    texts = tmd.split_texts(ds, args.text_field, args.normalize_mode)
    if "source" not in ds.column_names:
        raise ValueError("Merged dataset must contain a 'source' column for labels.")
    labels = list(ds["source"])
    if not texts:
        if allow_empty:
            return [], []
        raise ValueError(f"No rows left in {split_name!r} split after filtering.")
    counts = Counter(labels)
    if len(counts) < 2:
        if allow_empty:
            print(
                f"Skipping {label} split {split_name!r}: need >=2 classes; "
                f"got {len(counts)} ({sorted(counts.keys())!r})."
            )
            return [], []
        raise ValueError(
            f"Training/evaluation needs at least two label values; got {len(counts)}."
        )
    print(
        f"Loaded {label} split {split_name!r}: n={len(texts)}, "
        f"n_classes={len(counts)}, normalize_mode={args.normalize_mode}"
    )
    for name, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {name!r}: {n}")
    return texts, labels


def stacked_predict_proba(
    detector: BaseDetector,
    texts: list[str],
    batch_size: int,
    show_progress: bool,
) -> np.ndarray:
    model = detector.model
    if model is None:
        raise RuntimeError("Detector has no fitted model.")
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Multiclass metrics require predict_proba support.")
    parts: list[np.ndarray] = []
    for X in detector._iter_feature_chunks(
        texts, batch_size, show_progress=show_progress
    ):
        parts.append(model.predict_proba(X))
    if not parts:
        return np.zeros((0, len(model.classes_)), dtype=np.float64)
    return np.vstack(parts)


def multiclass_metrics(
    detector: BaseDetector,
    texts: list[str],
    labels: list[str],
    batch_size: int,
) -> dict[str, Any]:
    model = detector.model
    if model is None:
        raise RuntimeError("Detector has no fitted model.")
    classes = np.asarray(model.classes_)
    y_true = np.asarray(labels)
    prob = stacked_predict_proba(detector, texts, batch_size, show_progress=True)
    class_index = np.argmax(prob, axis=1)
    y_pred = classes[class_index]
    labels_list = list(classes)

    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "log_loss": float(log_loss(y_true, prob, labels=labels_list)),
    }

    n_classes = len(classes)
    if n_classes > 3:
        out["top_3_accuracy"] = float(
            top_k_accuracy_score(y_true, prob, k=3, labels=labels_list)
        )
    if n_classes > 5:
        out["top_5_accuracy"] = float(
            top_k_accuracy_score(y_true, prob, k=5, labels=labels_list)
        )

    try:
        out["roc_auc_ovr_macro"] = float(
            roc_auc_score(
                y_true,
                prob,
                multi_class="ovr",
                average="macro",
                labels=labels_list,
            )
        )
    except ValueError as exc:
        out["roc_auc_ovr_macro"] = None
        out["roc_auc_ovr_macro_error"] = str(exc)

    cm = confusion_matrix(y_true, y_pred, labels=labels_list)
    out["confusion_matrix"] = {
        "labels": labels_list,
        "matrix": cm.tolist(),
    }
    rep = classification_report(
        y_true, y_pred, labels=labels_list, output_dict=True, zero_division=0
    )
    out["classification_report"] = json_sanitize(rep)
    return out


def print_confusion_matrix(prefix: str, metrics: dict[str, Any]) -> None:
    cm_info = metrics.get("confusion_matrix")
    if not cm_info:
        return
    labels: list[str] = list(cm_info["labels"])
    matrix = np.asarray(cm_info["matrix"], dtype=int)
    n = len(labels)
    print(f"\n{prefix} confusion matrix (rows=true, cols=predicted):")
    for i, name in enumerate(labels):
        print(f"  [{i}] {name}")
    col_width = max(6, *(len(str(int(x))) for x in matrix.flat), len(str(n - 1)))
    header = " " * 8 + "".join(f"{j:>{col_width}d}" for j in range(n))
    print(header)
    for i in range(n):
        row = "".join(f"{int(matrix[i, j]):>{col_width}d}" for j in range(n))
        print(f"  [{i}]{row}")


def print_metrics_line(prefix: str, metrics: dict[str, Any]) -> None:
    parts = [
        f"accuracy={metrics['accuracy']:.4f}",
        f"log_loss={metrics['log_loss']:.4f}",
        f"f1_macro={metrics['f1_macro']:.4f}",
        f"f1_weighted={metrics['f1_weighted']:.4f}",
    ]
    if "top_3_accuracy" in metrics:
        parts.append(f"top_3_accuracy={metrics['top_3_accuracy']:.4f}")
    if "top_5_accuracy" in metrics:
        parts.append(f"top_5_accuracy={metrics['top_5_accuracy']:.4f}")
    auc = metrics.get("roc_auc_ovr_macro")
    if auc is not None:
        parts.append(f"roc_auc_ovr_macro={auc:.4f}")
    print(f"{prefix}: " + ", ".join(parts))


def train_author_classifier(args: argparse.Namespace) -> None:
    data = tmd.load_merged_dataset(args.dataset_path)
    train_ds = _filtered_ds(data, args.split, args)
    valid_sources = valid_sources_for_train(train_ds, args.min_samples_per_class)

    train_texts, train_labels = build_split(
        data, args.split, args, "train", valid_sources
    )

    detector = tmd.build_ascii_detector(args)
    print(
        f"Training multiclass ASCII author classifier "
        f"({args.ascii_detector}, {args.classifier}) "
        f"with {detector.feature_dim} features, {len(set(train_labels))} classes..."
    )
    detector.fit(train_texts, train_labels)

    metrics = multiclass_metrics(
        detector, train_texts, train_labels, args.predict_batch_size
    )
    print_metrics_line("Train metrics", metrics)
    print_confusion_matrix("Train", metrics)
    n_train = len(train_texts)
    del train_texts, train_labels

    val_metrics: dict[str, Any] | None = None
    if args.val_split:
        if args.val_split not in data:
            raise ValueError(
                f"Val split {args.val_split!r} missing; keys={list(data.keys())}"
            )
        val_texts, val_labels = build_split(
            data,
            args.val_split,
            args,
            "validation",
            valid_sources,
            allow_empty=True,
        )
        if not val_texts:
            print(
                f"Skipping validation: no rows left in {args.val_split!r} "
                "after filters / train-derived class list."
            )
        else:
            val_metrics = multiclass_metrics(
                detector, val_texts, val_labels, args.predict_batch_size
            )
            print_metrics_line("Validation metrics", val_metrics)
            print_confusion_matrix("Validation", val_metrics)
        del val_texts, val_labels

    detector.save(args.output)
    print(f"Saved multiclass ASCII model to {args.output} (n_train={n_train})")

    if args.metrics_json:
        model = detector.model
        classes_payload = (
            list(getattr(model, "classes_", [])) if model is not None else []
        )
        payload: dict[str, Any] = {
            "task": "multiclass_source",
            "ascii_detector": args.ascii_detector,
            "classifier": args.classifier,
            "classes": classes_payload,
            "output": str(args.output),
            "n_train": n_train,
            "train": json_sanitize(metrics),
            "validation": json_sanitize(val_metrics) if val_metrics else None,
            "dataset_path": str(args.dataset_path),
            "split": args.split,
            "val_split": args.val_split,
            "normalize_mode": args.normalize_mode,
            "text_field": args.text_field,
            "min_samples_per_class": args.min_samples_per_class,
            "include_dataset": list(args.include_dataset),
            "exclude_dataset": list(args.exclude_dataset),
            "include_source": list(args.include_source),
            "exclude_source": list(args.exclude_source),
            "disable_feature_group": list(args.disable_feature_group),
        }
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote metrics to {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a multiclass ASCII-feature classifier: predict which "
            "'source' (model or human) produced the text."
        )
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
        help="Same as train_merged_ascii_detector.py.",
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
        help="Drop these source values (e.g. 'human' for models-only K-way).",
    )
    parser.add_argument(
        "--filter",
        nargs="+",
        action="extend",
        default=[],
        help="Extra exact-match filters as COLUMN=VALUE. Can be repeated.",
    )
    parser.add_argument("--filter-num-proc", type=int, default=None)
    tmd.add_ascii_detector_arg(parser)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--fit-batch-size", type=int, default=1024)
    parser.add_argument("--predict-batch-size", type=int, default=4096)
    parser.add_argument(
        "--min-samples-per-class",
        type=int,
        default=1,
        help="Drop any class with fewer than this many train examples after filters.",
    )
    parser.add_argument(
        "--classifier",
        choices=["tree", "forest", "logreg"],
        default="forest",
        help="Sklearn head on ASCII features: single tree, random forest, or multinomial logistic regression.",
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
    train_author_classifier(parse_args())
