from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

import train_merged_ascii_detector as tmd
from eval_bert_common import predict_dataset
from train_bert_detector_baseline import (
    DEFAULT_INSTRUCT,
    DEFAULT_MODEL,
    EmbeddingBinaryClassifier,
    compute_metrics,
    load_split_with_sources,
    machine_sources_in_split,
    make_data_collator,
    make_dataset,
    mean_per_model,
    per_model_metrics,
)


def load_checkpoint_model(
    checkpoint: Path,
    *,
    model_name: str,
    dtype: torch.dtype,
) -> tuple[EmbeddingBinaryClassifier, AutoTokenizer]:
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
    model = EmbeddingBinaryClassifier(backbone, backbone.config.hidden_size)
    model.classifier.to(dtype=dtype)
    state = load_file(weights_path)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint state mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    return model, tokenizer


def evaluate(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")

    data = tmd.load_merged_dataset(args.dataset_path)
    test_texts, test_labels, test_sources = load_split_with_sources(
        data, args.test_split, args, "test", subsample=False
    )
    test_machine_sources = machine_sources_in_split(test_sources, args.human_source)
    print(
        f"Test split {args.test_split!r}: n={len(test_texts)}, "
        f"n_models={len(test_machine_sources)}"
    )

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]
    print("Loading checkpoint...")
    model, tokenizer = load_checkpoint_model(
        checkpoint,
        model_name=args.model_name,
        dtype=dtype,
    )
    print(f"Tokenizing {len(test_texts)} test examples...")
    test_dataset = make_dataset(
        test_texts,
        test_labels,
        test_sources,
        tokenizer,
        args.max_length,
        args.instruct,
    )
    collator = make_data_collator(tokenizer, args.max_length)

    logits_np, labels_np = predict_dataset(
        model,
        test_dataset,
        collator,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        desc="Evaluating test",
    )
    sources = np.asarray(test_dataset["source"])
    if len(sources) != len(labels_np):
        raise ValueError(
            f"Test source count {len(sources)} != prediction count {len(labels_np)}."
        )

    probs = F.softmax(torch.tensor(logits_np), dim=-1).numpy()[:, 1]
    test_metrics = compute_metrics((logits_np, labels_np))
    per_model_test = per_model_metrics(
        probs,
        labels_np,
        sources,
        human_source=args.human_source,
        machine_sources=test_machine_sources,
    )
    if len(per_model_test) != len(test_machine_sources):
        missing = sorted(set(test_machine_sources) - set(per_model_test))
        raise ValueError(f"Missing per-model test metrics for: {missing}")
    n_test = len(labels_np)

    print(f"Scored all {n_test} test examples.")
    for src in test_machine_sources:
        row = per_model_test[src]
        print(
            f"  {src}: n={row['n']} human={row['n_human']} machine={row['n_machine']} "
            f"auroc={row['auroc']:.4f} auprc={row['auprc']:.4f} "
            f"f1={row['f1']:.4f}"
        )

    metrics_path = (
        Path(args.metrics_json)
        if args.metrics_json
        else checkpoint / "per_model_test_metrics.json"
    )
    payload = {
        "checkpoint": str(checkpoint),
        "test": test_metrics,
        "per_model_test": per_model_test,
        "mean_per_model_test_auroc": mean_per_model(per_model_test, "auroc"),
        "mean_per_model_test_auprc": mean_per_model(per_model_test, "auprc"),
        "n_test": n_test,
        "test_machine_sources": test_machine_sources,
        "human_source": args.human_source,
        "positive_class": "human",
        "dataset_path": str(args.dataset_path),
        "test_split": args.test_split,
        "model_name": args.model_name,
        "normalize_mode": args.normalize_mode,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved metrics to {metrics_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a bert_detector_baseline checkpoint on the test split."
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset-path", default="dataset/merged_dataset")
    p.add_argument("--test-split", default="test")
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--text-field", default="response")
    p.add_argument(
        "--normalize-mode", choices=["raw", "whitespace"], default="whitespace"
    )
    p.add_argument("--human-source", default="human")
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--include-dataset", nargs="+", default=[])
    p.add_argument("--exclude-dataset", nargs="+", default=[])
    p.add_argument("--include-source", nargs="+", default=[])
    p.add_argument("--exclude-source", nargs="+", default=[])
    p.add_argument("--filter", nargs="+", action="extend", default=[])
    p.add_argument("--filter-num-proc", type=int, default=None)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--instruct", default=DEFAULT_INSTRUCT)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--metrics-json", default=None)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
