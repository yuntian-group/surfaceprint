from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, log_loss, top_k_accuracy_score
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_merged_ascii_detector as tmd  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_INSTRUCT = "Given a text, classify which model or author produced it."


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask[:, -1].sum() == attention_mask.shape[0]:
        return hidden[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]


class EmbeddingMulticlassClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **_,
    ) -> SequenceClassifierOutput:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = last_token_pool(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(pooled)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )
        if hasattr(self.backbone, "config") and hasattr(
            self.backbone.config, "use_cache"
        ):
            self.backbone.config.use_cache = False


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


def _keep_source_in(valid: frozenset[str]):
    def keep(row: dict) -> bool:
        return row.get("source") in valid

    return keep


def valid_sources_for_train(ds: Dataset, min_samples_per_class: int) -> frozenset[str]:
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
            "Need at least two classes with >= min-samples-per-class after filtering."
        )
    return valid


def load_split(
    data: DatasetDict,
    split_name: str,
    args: argparse.Namespace,
    *,
    label: str,
    valid_sources: frozenset[str],
    label_to_id: dict[str, int],
    allow_empty: bool = False,
) -> tuple[list[str], list[int]]:
    if split_name not in data:
        if allow_empty:
            return [], []
        raise ValueError(f"Expected split {split_name!r}; got {list(data.keys())}")

    ds = _filtered_ds(data, split_name, args)
    filter_num_proc = (
        args.filter_num_proc
        if args.filter_num_proc and args.filter_num_proc > 1
        else None
    )
    ds = ds.filter(_keep_source_in(valid_sources), num_proc=filter_num_proc)
    texts = tmd.split_texts(ds, args.text_field, args.normalize_mode)
    if not texts:
        if allow_empty:
            return [], []
        raise ValueError(f"No rows left in {split_name!r} split after filtering.")

    labels = [label_to_id[s] for s in ds["source"]]
    counts = Counter(labels)
    if len(counts) < 2 and not allow_empty:
        raise ValueError(
            f"{label} split {split_name!r} needs >=2 classes; got {len(counts)}."
        )
    print(
        f"Loaded {label} split {split_name!r}: n={len(texts)}, "
        f"n_classes={len(counts)}, normalize_mode={args.normalize_mode}"
    )
    return texts, labels


def tokenize_texts(tokenizer, texts: list[str], max_length: int):
    return tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding=False,
    )


def make_dataset(
    texts: list[str],
    labels: list[int],
    tokenizer,
    max_length: int,
    instruct: str | None,
) -> Dataset:
    ds = Dataset.from_dict({"text": texts, "labels": labels})

    def tokenize(batch):
        batch_texts = batch["text"]
        if instruct:
            batch_texts = [
                f"Instruct: {instruct}\nQuery:{text}" for text in batch_texts
            ]
        return tokenize_texts(tokenizer, batch_texts, max_length)

    return ds.map(tokenize, batched=True, remove_columns=["text"])


def make_data_collator(tokenizer, max_length: int):
    base = DataCollatorWithPadding(tokenizer)

    def collate(features):
        for feature in features:
            if len(feature["input_ids"]) > max_length:
                feature["input_ids"] = feature["input_ids"][:max_length]
                feature["attention_mask"] = feature["attention_mask"][:max_length]
        return base(features)

    return collate


def make_compute_metrics(num_classes: int):
    class_labels = list(range(num_classes))

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
        preds = probs.argmax(axis=1)
        out = {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(
                f1_score(labels, preds, average="macro", zero_division=0)
            ),
            "log_loss": float(log_loss(labels, probs, labels=class_labels)),
        }
        if num_classes > 3:
            out["top_3_accuracy"] = float(
                top_k_accuracy_score(labels, probs, k=3, labels=class_labels)
            )
        return out

    return compute_metrics


def fsdp_training_args(args: argparse.Namespace) -> tuple[str, dict | None]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_fsdp = args.fsdp2 == "true" or (args.fsdp2 == "auto" and world_size > 1)
    if not use_fsdp:
        return "", None
    return "full_shard auto_wrap", {
        "fsdp_version": 2,
        "fsdp_transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"],
        "activation_checkpointing": True,
    }


def build_epoch_eval_dataset(
    val_dataset: Dataset | None,
    test_dataset: Dataset | None,
    *,
    val_split: str | None,
    test_split: str | None,
) -> tuple[dict[str, Dataset] | None, str | None]:
    parts: dict[str, Dataset] = {}
    if val_dataset is not None:
        parts["validation"] = val_dataset
    if test_dataset is not None and test_split != val_split:
        parts["test"] = test_dataset
    if not parts:
        return None, None
    best_metric = "eval_validation_loss" if "validation" in parts else "eval_test_loss"
    return parts, best_metric


@torch.no_grad()
def predict_metrics(
    trainer: Trainer, dataset: Dataset, num_classes: int, batch_size: int
) -> dict[str, float]:
    output = trainer.predict(dataset)
    metrics_fn = make_compute_metrics(num_classes)
    return metrics_fn((output.predictions, output.label_ids))


def train(args: argparse.Namespace) -> None:
    data = tmd.load_merged_dataset(args.dataset_path)
    train_ds = _filtered_ds(data, args.split, args)
    valid_sources = valid_sources_for_train(train_ds, args.min_samples_per_class)
    classes = sorted(valid_sources)
    label_to_id = {name: i for i, name in enumerate(classes)}
    num_classes = len(classes)
    print(f"Training {num_classes}-way classifier: {classes}")

    train_texts, train_labels = load_split(
        data,
        args.split,
        args,
        label="train",
        valid_sources=valid_sources,
        label_to_id=label_to_id,
    )

    val_texts: list[str] | None = None
    val_labels: list[int] | None = None
    if args.val_split:
        val_texts, val_labels = load_split(
            data,
            args.val_split,
            args,
            label="validation",
            valid_sources=valid_sources,
            label_to_id=label_to_id,
            allow_empty=True,
        )
        if not val_texts:
            print(f"No validation rows in {args.val_split!r}; training without eval.")
            val_texts, val_labels = None, None

    test_texts: list[str] | None = None
    test_labels: list[int] | None = None
    if args.test_split:
        test_texts, test_labels = load_split(
            data,
            args.test_split,
            args,
            label="test",
            valid_sources=valid_sources,
            label_to_id=label_to_id,
            allow_empty=True,
        )
        if not test_texts:
            print(f"No test rows in {args.test_split!r}; skipping test evaluation.")
            test_texts, test_labels = None, None

    instruct = args.instruct or None
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]
    fsdp, fsdp_config = fsdp_training_args(args)
    load_dtype = torch.float32 if fsdp else dtype

    backbone = AutoModel.from_pretrained(
        args.model_name,
        dtype=load_dtype,
        attn_implementation="flash_attention_3",
    )
    model = EmbeddingMulticlassClassifier(
        backbone, backbone.config.hidden_size, num_classes
    )
    model.classifier.to(dtype=load_dtype)

    train_dataset = make_dataset(
        train_texts, train_labels, tokenizer, args.max_length, instruct
    )
    val_dataset = (
        make_dataset(val_texts, val_labels, tokenizer, args.max_length, instruct)
        if val_texts is not None and val_labels is not None
        else None
    )
    test_dataset = (
        make_dataset(test_texts, test_labels, tokenizer, args.max_length, instruct)
        if test_texts is not None and test_labels is not None
        else None
    )

    eval_dataset, best_metric = build_epoch_eval_dataset(
        val_dataset,
        test_dataset,
        val_split=args.val_split,
        test_split=args.test_split,
    )
    has_eval = eval_dataset is not None
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=not fsdp,
        bf16=args.dtype == "bf16",
        fp16=args.dtype == "fp16",
        eval_strategy="epoch" if has_eval else "no",
        save_strategy="epoch" if has_eval else "no",
        save_total_limit=3,
        load_best_model_at_end=has_eval,
        metric_for_best_model=best_metric,
        greater_is_better=False,
        logging_steps=5,
        report_to="none",
        fsdp=fsdp,
        fsdp_config=fsdp_config,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=make_data_collator(tokenizer, args.max_length),
        compute_metrics=make_compute_metrics(num_classes) if has_eval else None,
        processing_class=tokenizer,
    )
    trainer.train()

    test_metrics: dict[str, float] | None = None
    if test_dataset is not None:
        test_metrics = predict_metrics(
            trainer, test_dataset, num_classes, args.eval_batch_size
        )
        print(
            f"Test metrics ({args.test_split!r}, n={len(test_texts)}): "
            f"accuracy={test_metrics['accuracy']:.4f}, "
            f"log_loss={test_metrics['log_loss']:.4f}, "
            f"f1_macro={test_metrics['f1_macro']:.4f}"
        )

    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    label_path = Path(args.output) / "label_map.json"
    label_path.write_text(
        json.dumps({"classes": classes, "label_to_id": label_to_id}, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.metrics_json:
        payload = {
            "task": "multiclass_source",
            "classes": classes,
            "test": test_metrics,
            "n_train": len(train_texts),
            "n_test": len(test_texts) if test_texts is not None else 0,
            "n_classes": num_classes,
            "dataset_path": str(args.dataset_path),
            "split": args.split,
            "val_split": args.val_split,
            "test_split": args.test_split,
            "model_name": args.model_name,
            "normalize_mode": args.normalize_mode,
            "min_samples_per_class": args.min_samples_per_class,
        }
        Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_json).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune Qwen3-Embedding as an n-way author/source classifier."
    )
    p.add_argument("--dataset-path", default="dataset/merged_dataset")
    p.add_argument("--split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--test-split", default="test")
    p.add_argument("--output", required=True)
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--text-field", default="response")
    p.add_argument("--normalize-mode", choices=["raw", "whitespace"], default="raw")
    p.add_argument("--include-dataset", nargs="+", default=[])
    p.add_argument("--exclude-dataset", nargs="+", default=[])
    p.add_argument("--include-source", nargs="+", default=[])
    p.add_argument("--exclude-source", nargs="+", default=[])
    p.add_argument("--filter", nargs="+", action="extend", default=[])
    p.add_argument("--filter-num-proc", type=int, default=None)
    p.add_argument(
        "--min-samples-per-class",
        type=int,
        default=1,
        help="Drop train classes with fewer than this many examples.",
    )
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--instruct", default=DEFAULT_INSTRUCT)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--fsdp2", default="auto")
    p.add_argument("--metrics-json", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.instruct == "":
        args.instruct = None
    if args.val_split == "":
        args.val_split = None
    if args.test_split == "":
        args.test_split = None
    train(args)
