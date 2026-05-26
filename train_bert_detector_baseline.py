from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
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
DEFAULT_INSTRUCT = (
    "Given a text, classify whether it was written by a human or a machine."
)


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask[:, -1].sum() == attention_mask.shape[0]:
        return hidden[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]


class EmbeddingBinaryClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(hidden_size, 2)

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
    sources: list[str],
    tokenizer,
    max_length: int,
    instruct: str | None,
):
    ds = Dataset.from_dict({"text": texts, "labels": labels, "source": sources})

    def tokenize(batch):
        texts = batch["text"]
        if instruct:
            texts = [f"Instruct: {instruct}\nQuery:{text}" for text in texts]
        encoded = tokenize_texts(tokenizer, texts, max_length)
        encoded["source"] = batch["source"]
        return encoded

    return ds.map(tokenize, batched=True, remove_columns=["text"])


def make_data_collator(tokenizer, max_length: int):
    base = DataCollatorWithPadding(tokenizer)

    def collate(features):
        for feature in features:
            if len(feature["input_ids"]) > max_length:
                feature["input_ids"] = feature["input_ids"][:max_length]
                feature["attention_mask"] = feature["attention_mask"][:max_length]
            feature.pop("source", None)
        return base(features)

    return collate


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    preds = probs >= 0.5
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "log_loss": float(log_loss(labels, probs, labels=[0, 1])),
        "auroc": float(roc_auc_score(labels, probs)),
        "auprc": float(average_precision_score(labels, probs)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


def subsample_non_human(
    ds: Dataset,
    *,
    human_source: str,
    non_human_sources: list[str],
    rng: np.random.Generator,
) -> Dataset:
    if "source" not in ds.column_names:
        raise ValueError("Merged dataset must contain a 'source' column.")

    sources = np.asarray(ds["source"])
    human_idx = np.flatnonzero(sources == human_source)
    n_human = int(human_idx.size)
    if n_human == 0:
        raise ValueError(f"No rows with source={human_source!r}.")

    n_models = len(non_human_sources)
    per_model = n_human // n_models
    if per_model == 0:
        raise ValueError(
            f"Need at least {n_models} human rows to subsample {n_models} non-human "
            f"sources; got {n_human}."
        )

    keep = human_idx.tolist()
    picked: dict[str, int] = {human_source: n_human}

    for src in non_human_sources:
        idx = np.flatnonzero(sources == src)
        n_avail = int(idx.size)
        if n_avail == 0:
            raise ValueError(f"No rows with source={src!r} after filtering.")
        n_take = min(per_model, n_avail)
        if n_avail > n_take:
            idx = rng.choice(idx, size=n_take, replace=False)
        keep.extend(int(i) for i in idx)
        picked[src] = n_take

    keep.sort()
    print(f"  per_model_cap={per_model}")
    for src, n in picked.items():
        print(f"  {src!r}: {n}")
    return ds.select(keep)


def human_positive_labels(sources: list[str], human_source: str) -> list[int]:
    return [1 if source == human_source else 0 for source in sources]


def load_split_with_sources(
    data,
    split_name: str,
    args: argparse.Namespace,
    label: str,
    *,
    subsample: bool = False,
) -> tuple[list[str], list[int], list[str]]:
    ds = tmd.apply_filters(
        data[split_name],
        include_dataset=args.include_dataset,
        exclude_dataset=args.exclude_dataset,
        include_source=args.include_source,
        exclude_source=args.exclude_source,
        filters=tmd.parse_filters(args.filter),
        num_proc=args.filter_num_proc,
    )
    non_human_sources = [s for s in args.include_source if s != args.human_source]
    if subsample:
        split_seed = {"train": 0, "test": 1, "validation": 2, "val": 2}.get(
            split_name, 3
        )
        rng = np.random.default_rng(args.random_state + split_seed)
        ds = subsample_non_human(
            ds,
            human_source=args.human_source,
            non_human_sources=non_human_sources,
            rng=rng,
        )

    texts = tmd.split_texts(ds, args.text_field, args.normalize_mode)
    sources = list(ds["source"])
    labels = human_positive_labels(sources, args.human_source)
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2)
    mode = "subsampled" if subsample else "full"
    print(
        f"Loaded {label} split {split_name!r} ({mode}): n={len(texts)}, "
        f"negative(non-human)={int(counts[0])}, positive(human)={int(counts[1])}, "
        f"normalize_mode={args.normalize_mode}"
    )
    return texts, labels, sources


def machine_sources_in_split(sources: list[str], human_source: str) -> list[str]:
    return sorted(set(sources) - {human_source})


def per_model_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    *,
    human_source: str,
    machine_sources: list[str],
) -> dict[str, dict[str, float | int]]:
    results: dict[str, dict[str, float | int]] = {}
    for src in machine_sources:
        mask = (sources == human_source) | (sources == src)
        if not mask.any():
            raise ValueError(
                f"No test rows for {src!r} vs {human_source!r} after filtering."
            )
        subset_labels = labels[mask]
        subset_scores = scores[mask]
        counts = np.bincount(subset_labels, minlength=2)
        if counts[0] == 0 or counts[1] == 0:
            raise ValueError(
                f"Per-model test metrics for {src!r} need human and machine rows; "
                f"got non-human={int(counts[0])}, human={int(counts[1])}."
            )
        preds = subset_scores >= 0.5
        results[src] = {
            "accuracy": float(accuracy_score(subset_labels, preds)),
            "log_loss": float(log_loss(subset_labels, subset_scores, labels=[0, 1])),
            "auroc": float(roc_auc_score(subset_labels, subset_scores)),
            "auprc": float(average_precision_score(subset_labels, subset_scores)),
            "f1": float(f1_score(subset_labels, preds, zero_division=0)),
            "n": int(mask.sum()),
            "n_human": int(counts[1]),
            "n_machine": int(counts[0]),
        }
    return results


@torch.no_grad()
def predict_test_metrics(
    trainer: Trainer,
    dataset: Dataset,
    *,
    human_source: str,
    machine_sources: list[str],
    batch_size: int,
) -> tuple[dict[str, float], dict[str, dict[str, float | int]], int]:
    output = trainer.predict(dataset)
    n = len(output.label_ids)
    if n != len(dataset):
        raise ValueError(
            f"Test prediction count {n} != dataset size {len(dataset)}; "
            "not all test examples were scored."
        )

    labels = np.asarray(output.label_ids)
    sources = np.asarray(dataset["source"])
    if len(sources) != n:
        raise ValueError(f"Test source count {len(sources)} != prediction count {n}.")

    probs = F.softmax(torch.tensor(output.predictions), dim=-1).numpy()[:, 1]
    aggregate = compute_metrics((output.predictions, output.label_ids))
    per_model = per_model_metrics(
        probs,
        labels,
        sources,
        human_source=human_source,
        machine_sources=machine_sources,
    )
    if len(per_model) != len(machine_sources):
        missing = sorted(set(machine_sources) - set(per_model))
        raise ValueError(f"Missing per-model test metrics for: {missing}")
    return aggregate, per_model, n


def mean_per_model(
    per_model: dict[str, dict[str, float | int]], key: str
) -> float | None:
    values = [float(row[key]) for row in per_model.values() if key in row]
    return sum(values) / len(values) if values else None


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


def train(args: argparse.Namespace) -> None:
    data = tmd.load_merged_dataset(args.dataset_path)
    train_texts, train_labels, train_sources = load_split_with_sources(
        data, args.split, args, "train", subsample=not args.no_subsample
    )
    val_texts, val_labels, val_sources = load_split_with_sources(
        data, args.val_split, args, "validation", subsample=False
    )
    test_texts, test_labels, test_sources = load_split_with_sources(
        data, args.test_split, args, "test", subsample=False
    )
    test_machine_sources = machine_sources_in_split(test_sources, args.human_source)
    print(
        f"Test split {args.test_split!r}: n={len(test_texts)}, "
        f"n_models={len(test_machine_sources)}"
    )

    instruct = args.instruct
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
    model = EmbeddingBinaryClassifier(backbone, backbone.config.hidden_size)
    model.classifier.to(dtype=load_dtype)

    train_dataset = make_dataset(
        train_texts, train_labels, train_sources, tokenizer, args.max_length, instruct
    )
    val_dataset = make_dataset(
        val_texts, val_labels, val_sources, tokenizer, args.max_length, instruct
    )
    test_dataset = make_dataset(
        test_texts, test_labels, test_sources, tokenizer, args.max_length, instruct
    )

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
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_validation_auroc",
        greater_is_better=True,
        logging_steps=5,
        report_to="none",
        fsdp=fsdp,
        fsdp_config=fsdp_config,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset={"validation": val_dataset},
        data_collator=make_data_collator(tokenizer, args.max_length),
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )
    trainer.train()

    test_metrics, per_model_test, n_test = predict_test_metrics(
        trainer,
        test_dataset,
        human_source=args.human_source,
        machine_sources=test_machine_sources,
        batch_size=args.eval_batch_size,
    )
    print(f"Scored all {n_test} test examples.")
    for src in test_machine_sources:
        row = per_model_test[src]
        print(
            f"  {src}: n={row['n']} human={row['n_human']} machine={row['n_machine']} "
            f"auroc={row['auroc']:.4f} auprc={row['auprc']:.4f} "
            f"f1={row['f1']:.4f}"
        )

    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    if args.metrics_json:
        payload = {
            "test": test_metrics,
            "per_model_test": per_model_test,
            "mean_per_model_test_auroc": mean_per_model(per_model_test, "auroc"),
            "mean_per_model_test_auprc": mean_per_model(per_model_test, "auprc"),
            "n_train": len(train_texts),
            "n_test": n_test,
            "test_machine_sources": test_machine_sources,
            "human_source": args.human_source,
            "positive_class": "human",
            "subsample_train": not args.no_subsample,
            "random_state": args.random_state,
            "dataset_path": str(args.dataset_path),
            "split": args.split,
            "val_split": args.val_split,
            "test_split": args.test_split,
            "model_name": args.model_name,
        }
        Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_json).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune Qwen3-Embedding as a binary classifier (HF Trainer + FSDP2)."
    )
    p.add_argument("--dataset-path", default="dataset/merged_dataset")
    p.add_argument("--split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--test-split", default="test")
    p.add_argument("--output", required=True)
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--text-field", default="response")
    p.add_argument("--normalize-mode", choices=["raw", "whitespace"], default="raw")
    p.add_argument("--human-source", default="human")
    p.add_argument(
        "--no-subsample",
        action="store_true",
        help="Use all non-human rows in train instead of per-model subsampling.",
    )
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--include-dataset", nargs="+", default=[])
    p.add_argument("--exclude-dataset", nargs="+", default=[])
    p.add_argument("--include-source", nargs="+", default=[])
    p.add_argument("--exclude-source", nargs="+", default=[])
    p.add_argument("--filter", nargs="+", action="extend", default=[])
    p.add_argument("--filter-num-proc", type=int, default=None)
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
    train(parse_args())
