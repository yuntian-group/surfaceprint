from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import train_merged_ascii_detector as tmd

DEFAULT_INSTRUCT_SOURCES: tuple[str, ...] = (
    "gpt-3.5-turbo",
    "gpt-4.1-mini",
    "gpt-5.4-mini",
    "claude-sonnet-4-6",
    "facebook/opt-iml-1.3b",
    "/model-weights/Llama-2-7b-chat-hf/",
    "meta-llama/Llama-3.1-8B-Instruct",
    "allenai/Olmo-3-7B-Instruct",
    "google/gemma-4-E4B-it",
    "/model-weights/Qwen3.5-9B",
)


def selected_sources(args: argparse.Namespace) -> list[str]:
    return [args.human_source, *args.instruct_source]


def subsample_human_vs_instruct(
    ds: Dataset,
    *,
    human_source: str,
    instruct_sources: list[str],
    rng: np.random.Generator,
) -> tuple[Dataset, dict[str, int]]:
    if "source" not in ds.column_names:
        raise ValueError("Merged dataset must contain a 'source' column.")

    sources = np.asarray(ds["source"])
    human_idx = np.flatnonzero(sources == human_source)
    n_human = int(human_idx.size)
    if n_human == 0:
        raise ValueError(f"No rows with source={human_source!r}.")

    n_models = len(instruct_sources)
    per_model = n_human // n_models
    if per_model == 0:
        raise ValueError(
            f"Need at least {n_models} human rows to subsample {n_models} instruct "
            f"sources; got {n_human}."
        )

    keep = human_idx.tolist()
    picked: dict[str, int] = {human_source: n_human}

    for src in instruct_sources:
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
    return ds.select(keep), picked


def source_counts(ds: Dataset, sources: list[str]) -> dict[str, int]:
    if "source" not in ds.column_names:
        raise ValueError("Merged dataset must contain a 'source' column.")

    arr = np.asarray(ds["source"])
    picked: dict[str, int] = {}
    for src in sources:
        n = int(np.sum(arr == src))
        if n == 0:
            raise ValueError(f"No rows with source={src!r} after filtering.")
        picked[src] = n
    return picked


def load_split(
    data: DatasetDict,
    split_name: str,
    args: argparse.Namespace,
    label: str,
    *,
    subsample: bool = True,
) -> tuple[list[str], list[int], list[str], dict[str, int]]:
    if split_name not in data:
        raise ValueError(f"Expected split {split_name!r}; got {list(data.keys())}")

    ds = tmd.apply_filters(
        data[split_name],
        include_dataset=args.include_dataset,
        exclude_dataset=args.exclude_dataset,
        include_source=selected_sources(args),
        exclude_source=args.exclude_source,
        filters=tmd.parse_filters(args.filter),
        num_proc=args.filter_num_proc,
    )
    instruct_sources = list(args.instruct_source)
    all_sources = [args.human_source, *instruct_sources]
    if subsample:
        split_seed = {"train": 0, "test": 1, "val": 2}.get(split_name, 3)
        rng = np.random.default_rng(args.random_state + split_seed)
        ds, picked = subsample_human_vs_instruct(
            ds,
            human_source=args.human_source,
            instruct_sources=instruct_sources,
            rng=rng,
        )
    else:
        picked = source_counts(ds, all_sources)

    texts = tmd.split_texts(ds, args.text_field, args.normalize_mode)
    labels = tmd.build_labels(ds, "machine", None)
    sources = list(ds["source"])
    if not texts:
        raise ValueError(f"No rows left in {split_name!r} split after loading.")

    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2)
    mode = "subsampled" if subsample else "full"
    print(
        f"Loaded {label} split {split_name!r} ({mode}): n={len(texts)}, "
        f"human={int(counts[0])}, machine={int(counts[1])}, "
        f"normalize_mode={args.normalize_mode}"
    )
    if subsample:
        print(f"  per_model_cap={picked[args.human_source] // len(instruct_sources)}")
    for src, n in picked.items():
        print(f"  {src!r}: {n}")
    if counts[0] == 0 or counts[1] == 0:
        raise ValueError(
            f"{label} split needs both labels after loading; got "
            f"human={int(counts[0])}, machine={int(counts[1])}."
        )
    return texts, labels, sources, picked


def per_model_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    *,
    human_source: str,
    instruct_sources: list[str],
) -> dict[str, dict[str, float | int]]:
    results: dict[str, dict[str, float | int]] = {}
    for src in instruct_sources:
        mask = (sources == human_source) | (sources == src)
        if not mask.any():
            raise ValueError(f"No test rows for {src!r} vs {human_source!r}.")
        subset_labels = labels[mask]
        subset_scores = scores[mask]
        counts = np.bincount(subset_labels, minlength=2)
        if counts[0] == 0 or counts[1] == 0:
            raise ValueError(
                f"Per-model eval for {src!r} needs both labels; got "
                f"human={int(counts[0])}, machine={int(counts[1])}."
            )
        results[src] = {
            "auroc": float(roc_auc_score(subset_labels, subset_scores)),
            "auprc": float(average_precision_score(subset_labels, subset_scores)),
            "n": int(mask.sum()),
            "n_human": int(counts[0]),
            "n_machine": int(counts[1]),
        }
    return results


def print_per_model_metrics_table(
    results: dict[str, dict[str, float | int]], *, split_name: str
) -> None:
    if not results:
        return
    print(f"\nPer-model metrics on {split_name!r} (human + one instruct source):")
    header = (
        f"{'source':<40} | {'n':>6} | {'n_machine':>9} | {'auroc':>8} | {'auprc':>8}"
    )
    print(header)
    print("-" * len(header))
    for src in sorted(results):
        row = results[src]
        print(
            f"{src:<40} | {int(row['n']):>6} | {int(row['n_machine']):>9} | "
            f"{float(row['auroc']):>8.4f} | {float(row['auprc']):>8.4f}"
        )
    aurocs = [float(row["auroc"]) for row in results.values()]
    auprcs = [float(row["auprc"]) for row in results.values()]
    print(
        f"mean_auroc={sum(aurocs) / len(aurocs):.4f}, "
        f"mean_auprc={sum(auprcs) / len(auprcs):.4f} ({len(results)} models)"
    )


def train_unified_detector(args: argparse.Namespace) -> None:
    data = tmd.load_merged_dataset(args.dataset_path)
    train_texts, train_labels, _, train_picked = load_split(
        data, args.split, args, "train"
    )

    detector = tmd.build_ascii_detector(args)
    print(
        f"Training unified human-vs-instruct ASCII detector "
        f"({args.ascii_detector}, {args.classifier}) "
        f"with {detector.feature_dim} features, "
        f"{len(args.instruct_source)} instruct sources..."
    )
    detector.fit(train_texts, train_labels)
    train_metrics = tmd.train_metrics(
        detector, train_texts, train_labels, args.predict_batch_size
    )
    print(
        "Train metrics: "
        f"accuracy={train_metrics['accuracy']:.4f}, "
        f"log_loss={train_metrics['log_loss']:.4f}, "
        f"auroc={train_metrics['auroc']:.4f}, "
        f"auprc={train_metrics['auprc']:.4f}, "
        f"f1={train_metrics['f1']:.4f}"
    )
    n_train = len(train_texts)
    del train_texts, train_labels

    test_metrics: dict[str, float] | None = None
    per_model: dict[str, dict[str, float | int]] = {}
    mean_auroc: float | None = None
    mean_auprc: float | None = None
    test_picked: dict[str, int] | None = None
    if args.test_split:
        test_texts, test_labels, test_sources, test_picked = load_split(
            data, args.test_split, args, "test", subsample=False
        )
        test_metrics = tmd.train_metrics(
            detector, test_texts, test_labels, args.predict_batch_size
        )
        print(
            "Test metrics: "
            f"accuracy={test_metrics['accuracy']:.4f}, "
            f"log_loss={test_metrics['log_loss']:.4f}, "
            f"auroc={test_metrics['auroc']:.4f}, "
            f"auprc={test_metrics['auprc']:.4f}, "
            f"f1={test_metrics['f1']:.4f}"
        )

        scores = np.asarray(
            detector.batch_predict(
                test_texts, batch_size=args.predict_batch_size, show_progress=False
            ),
            dtype=np.float64,
        )
        labels_arr = np.asarray(test_labels, dtype=int)
        sources_arr = np.asarray(test_sources)
        per_model = per_model_metrics(
            scores,
            labels_arr,
            sources_arr,
            human_source=args.human_source,
            instruct_sources=list(args.instruct_source),
        )
        print_per_model_metrics_table(per_model, split_name=args.test_split)
        if per_model:
            mean_auroc = sum(float(row["auroc"]) for row in per_model.values()) / len(
                per_model
            )
            mean_auprc = sum(float(row["auprc"]) for row in per_model.values()) / len(
                per_model
            )
        del test_texts, test_labels, test_sources, scores

    detector.save(args.output)
    print(f"Saved unified ASCII detector to {args.output} (n_train={n_train})")

    if args.metrics_json:
        payload: dict[str, object] = {
            "task": "unified_human_vs_instruct",
            "ascii_detector": args.ascii_detector,
            "classifier": args.classifier,
            "human_source": args.human_source,
            "instruct_sources": list(args.instruct_source),
            "output": str(args.output),
            "n_train": n_train,
            "train": train_metrics,
            "test": test_metrics,
            "test_split": args.test_split,
            "per_model_test": per_model,
            "mean_per_model_test_auroc": mean_auroc,
            "mean_per_model_test_auprc": mean_auprc,
            "train_source_counts": train_picked,
            "test_source_counts": test_picked,
            "dataset_path": str(args.dataset_path),
            "split": args.split,
            "normalize_mode": args.normalize_mode,
            "include_dataset": list(args.include_dataset),
            "disable_feature_group": list(args.disable_feature_group),
        }
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote metrics to {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one human-vs-machine ASCII detector on subsampled selected sources, "
            "then evaluate on the full test split with per-model AUROC/AUPRC."
        )
    )
    parser.add_argument("--dataset-path", default="dataset/merged_dataset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--human-source", default="human")
    parser.add_argument(
        "--instruct-source",
        nargs="+",
        default=list(DEFAULT_INSTRUCT_SOURCES),
        help="Instruction-tuned model ids in the merged dataset 'source' column.",
    )
    parser.add_argument(
        "--text-field",
        default="response",
        help="Column to train on, or 'prompt_response' to join prompt and response.",
    )
    parser.add_argument(
        "--normalize-mode",
        choices=["raw", "whitespace"],
        default="whitespace",
    )
    parser.add_argument(
        "--include-dataset",
        nargs="+",
        default=["writingprompts", "reddit_eli5", "stackmathqa", "stackprogram"],
    )
    parser.add_argument("--exclude-dataset", nargs="+", default=[])
    parser.add_argument("--exclude-source", nargs="+", default=[])
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
        "--classifier",
        choices=["tree", "forest", "logreg"],
        default="tree",
    )
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--class-weight", default="balanced")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--logreg-c", type=float, default=1.0)
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
    parser.add_argument("--metrics-json", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train_unified_detector(parse_args())
