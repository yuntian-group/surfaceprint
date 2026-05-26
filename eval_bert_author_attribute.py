from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModel, AutoTokenizer

import train_merged_ascii_detector as tmd
from eval_bert_common import predict_dataset
from train_bert_author_attribute import (
    DEFAULT_INSTRUCT,
    DEFAULT_MODEL,
    EmbeddingMulticlassClassifier,
    _filtered_ds,
    load_split,
    make_compute_metrics,
    make_data_collator,
    make_dataset,
    valid_sources_for_train,
)


def load_label_map(
    checkpoint: Path, args: argparse.Namespace
) -> tuple[list[str], dict[str, int]]:
    for path in (checkpoint / "label_map.json", checkpoint.parent / "label_map.json"):
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            classes = list(data["classes"])
            label_to_id = {str(k): int(v) for k, v in data["label_to_id"].items()}
            return classes, label_to_id

    data = tmd.load_merged_dataset(args.dataset_path)
    train_ds = _filtered_ds(data, args.split, args)
    valid_sources = valid_sources_for_train(train_ds, args.min_samples_per_class)
    classes = sorted(valid_sources)
    label_to_id = {name: i for i, name in enumerate(classes)}
    return classes, label_to_id


def load_checkpoint_model(
    checkpoint: Path,
    *,
    model_name: str,
    num_classes: int,
    dtype: torch.dtype,
) -> tuple[EmbeddingMulticlassClassifier, AutoTokenizer]:
    checkpoint = checkpoint.resolve()
    weights_path = checkpoint / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing weights file: {weights_path}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModel.from_pretrained(
        model_name,
        dtype=dtype,
        attn_implementation="flash_attention_3",
    )
    model = EmbeddingMulticlassClassifier(
        backbone, backbone.config.hidden_size, num_classes
    )
    model.classifier.to(dtype=dtype)
    state = load_file(weights_path)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint state mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    return model, tokenizer


def per_class_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
) -> dict[str, dict[str, float | int]]:
    preds = probs.argmax(axis=1)
    class_ids = list(range(len(classes)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=class_ids,
        zero_division=0,
    )

    results: dict[str, dict[str, float | int]] = {}
    for class_id, class_name in enumerate(classes):
        n = int(support[class_id])
        n_correct = int((preds[labels == class_id] == class_id).sum()) if n else 0
        row: dict[str, float | int] = {
            "n": n,
            "n_correct": n_correct,
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1": float(f1[class_id]),
        }
        if n:
            row["accuracy"] = float(n_correct / n)
        results[class_name] = row
    return results


def mean_metric(
    per_class: dict[str, dict[str, float | int]], key: str
) -> float | None:
    values = [
        float(row[key])
        for row in per_class.values()
        if key in row and int(row.get("n", 0)) > 0
    ]
    return sum(values) / len(values) if values else None


def evaluate(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")

    classes, label_to_id = load_label_map(checkpoint, args)
    num_classes = len(classes)
    print(f"Evaluating {num_classes}-way classifier: {classes}")

    data = tmd.load_merged_dataset(args.dataset_path)
    valid_sources = frozenset(classes)
    test_texts, test_labels = load_split(
        data,
        args.test_split,
        args,
        label="test",
        valid_sources=valid_sources,
        label_to_id=label_to_id,
        allow_empty=False,
    )
    print(f"Test split {args.test_split!r}: n={len(test_texts)}")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]
    instruct = args.instruct or None
    print("Loading checkpoint...")
    model, tokenizer = load_checkpoint_model(
        checkpoint,
        model_name=args.model_name,
        num_classes=num_classes,
        dtype=dtype,
    )
    print(f"Tokenizing {len(test_texts)} test examples...")
    test_dataset = make_dataset(
        test_texts, test_labels, tokenizer, args.max_length, instruct
    )
    collator = make_data_collator(tokenizer, args.max_length)

    logits, labels = predict_dataset(
        model,
        test_dataset,
        collator,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        desc="Evaluating test",
    )
    metrics_fn = make_compute_metrics(num_classes)
    test_metrics = metrics_fn((logits, labels))
    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
    per_class_test = per_class_metrics(probs, labels, classes)

    print(
        f"Test metrics ({args.test_split!r}, n={len(test_texts)}): "
        f"accuracy={test_metrics['accuracy']:.4f}, "
        f"log_loss={test_metrics['log_loss']:.4f}, "
        f"f1_macro={test_metrics['f1_macro']:.4f}"
    )
    if "top_3_accuracy" in test_metrics:
        print(f"  top_3_accuracy={test_metrics['top_3_accuracy']:.4f}")

    for class_name in classes:
        row = per_class_test[class_name]
        if int(row["n"]) == 0:
            continue
        print(
            f"  {class_name}: n={row['n']} acc={row['accuracy']:.4f} "
            f"prec={row['precision']:.4f} rec={row['recall']:.4f} f1={row['f1']:.4f}"
        )

    metrics_path = (
        Path(args.metrics_json)
        if args.metrics_json
        else checkpoint / "per_class_test_metrics.json"
    )
    payload = {
        "checkpoint": str(checkpoint),
        "task": "multiclass_source",
        "classes": classes,
        "label_to_id": label_to_id,
        "test": test_metrics,
        "per_class_test": per_class_test,
        "mean_per_class_f1": mean_metric(per_class_test, "f1"),
        "mean_per_class_accuracy": mean_metric(per_class_test, "accuracy"),
        "n_test": len(test_texts),
        "n_classes": num_classes,
        "dataset_path": str(args.dataset_path),
        "split": args.split,
        "test_split": args.test_split,
        "model_name": args.model_name,
        "normalize_mode": args.normalize_mode,
        "min_samples_per_class": args.min_samples_per_class,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved metrics to {metrics_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a bert_author_attribute checkpoint on the test split."
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset-path", default="dataset/merged_dataset")
    p.add_argument("--split", default="train")
    p.add_argument("--test-split", default="test")
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--text-field", default="response")
    p.add_argument("--normalize-mode", choices=["raw", "whitespace"], default="whitespace")
    p.add_argument("--include-dataset", nargs="+", default=[])
    p.add_argument("--exclude-dataset", nargs="+", default=[])
    p.add_argument("--include-source", nargs="+", default=[])
    p.add_argument("--exclude-source", nargs="+", default=[])
    p.add_argument("--filter", nargs="+", action="extend", default=[])
    p.add_argument("--filter-num-proc", type=int, default=None)
    p.add_argument("--min-samples-per-class", type=int, default=1)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--instruct", default=DEFAULT_INSTRUCT)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--metrics-json", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.instruct == "":
        args.instruct = None
    evaluate(args)
