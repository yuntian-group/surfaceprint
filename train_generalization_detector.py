from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detectors.ascii_detector import ASCIIDetector  # noqa: E402

import train_merged_ascii_detector as tmd  # noqa: E402


@dataclass(frozen=True)
class Plan:
    name: str
    train_sources: list[str]
    test_sources: list[str] | None = None


@dataclass
class SplitFeatures:
    name: str
    sources: np.ndarray
    labels: np.ndarray
    X: np.ndarray


def sanitize_name(value: str) -> str:
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in value)
    return safe.strip("_") or "plan"


TRAIN_KEYS = ("train_sources", "train_models", "train_source", "train")
TEST_KEYS = ("test_sources", "test_models", "test_source", "test")


def parse_sources(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part) for part in value]


def first_present(obj: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in obj:
            return obj[key]
    return None


def parse_plan_obj(obj: dict[str, Any], idx: int) -> Plan:
    train_value = first_present(obj, TRAIN_KEYS)
    test_value = first_present(obj, TEST_KEYS)
    name = str(obj.get("name") or f"plan_{idx}")
    return Plan(
        name=name,
        train_sources=parse_sources(train_value),
        test_sources=None if test_value is None else parse_sources(test_value),
    )


def plans_from_json_file(path: str) -> list[Plan]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = payload["plans"] if isinstance(payload, dict) else payload
    return [parse_plan_obj(item, idx) for idx, item in enumerate(items, start=1)]


def requested_plans(args: argparse.Namespace) -> list[Plan]:
    plans: list[Plan] = []

    if args.plans_json:
        plans.extend(plans_from_json_file(args.plans_json))

    for raw in args.plan:
        plans.append(parse_plan_obj(json.loads(raw), len(plans) + 1))

    if args.train_source:
        plans.append(
            Plan(
                name=args.plan_name,
                train_sources=list(args.train_source),
                test_sources=list(args.test_source) if args.test_source else None,
            )
        )

    return plans


def validate_sources(
    *,
    plan_name: str,
    sources: list[str],
    known_sources: set[str],
    split_name: str,
    role: str,
) -> None:
    missing = sorted(set(sources) - known_sources)
    if missing:
        raise ValueError(
            f"Plan {plan_name!r} has {role} sources missing from "
            f"{split_name!r}: {missing}"
        )


def resolve_plan(
    plan: Plan,
    *,
    human_source: str,
    train_sources: list[str],
    test_sources: list[str],
    train_split: str,
    test_split: str,
) -> Plan:
    train_machine = [s for s in plan.train_sources if s != human_source]
    test_machine = (
        [s for s in test_sources if s not in set(train_machine)]
        if plan.test_sources is None
        else [s for s in plan.test_sources if s != human_source]
    )

    validate_sources(
        plan_name=plan.name,
        sources=train_machine,
        known_sources=set(train_sources),
        split_name=train_split,
        role="train",
    )
    validate_sources(
        plan_name=plan.name,
        sources=test_machine,
        known_sources=set(test_sources),
        split_name=test_split,
        role="test",
    )
    if not train_machine:
        raise ValueError(f"Plan {plan.name!r} has no machine train sources.")
    if not test_machine:
        raise ValueError(f"Plan {plan.name!r} has no machine test sources.")
    return Plan(plan.name, train_machine, test_machine)


def ensure_unique_plan_names(plans: list[Plan]) -> None:
    counts = Counter(plan.name for plan in plans)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Plan names must be unique; duplicates: {duplicates}")


def load_plans(
    args: argparse.Namespace,
    train_machine_sources: list[str],
    test_machine_sources: list[str],
) -> list[Plan]:
    raw_plans = requested_plans(args)
    if not raw_plans:
        raise ValueError(
            "No plans provided. Use --plans-json, --plan, or --train-source."
        )

    resolved = [
        resolve_plan(
            plan,
            human_source=args.human_source,
            train_sources=train_machine_sources,
            test_sources=test_machine_sources,
            train_split=args.split,
            test_split=args.test_split,
        )
        for plan in raw_plans
    ]
    ensure_unique_plan_names(resolved)
    return resolved


def filtered_split(
    data: DatasetDict, split_name: str, args: argparse.Namespace
) -> Dataset:
    if split_name not in data:
        raise ValueError(f"Expected split {split_name!r}; got {list(data.keys())}")

    ds = data[split_name]
    filters = tmd.parse_filters(args.filter)
    if not filters:
        return ds

    filter_num_proc = (
        args.filter_num_proc
        if args.filter_num_proc and args.filter_num_proc > 1
        else None
    )

    def keep(row: dict) -> bool:
        return all(str(row[column]) == expected for column, expected in filters)

    return ds.filter(keep, num_proc=filter_num_proc)


def extract_all_features(texts: list[str], args: argparse.Namespace) -> np.ndarray:
    detector = ASCIIDetector(
        {
            "disabled_feature_groups": args.disable_feature_group,
            "num_workers": args.num_workers,
        }
    )
    chunks = list(
        detector._iter_feature_chunks(
            texts,
            batch_size=args.fit_batch_size,
            show_progress=True,
        )
    )
    if not chunks:
        return np.empty((0, detector.feature_dim), dtype=np.float32)
    return np.vstack(chunks)


def materialize_split(
    data: DatasetDict,
    split_name: str,
    args: argparse.Namespace,
) -> SplitFeatures:
    ds = filtered_split(data, split_name, args)
    if "source" not in ds.column_names:
        raise ValueError("Merged dataset must contain a 'source' column.")

    texts = tmd.split_texts(ds, args.text_field, args.normalize_mode)
    if not texts:
        raise ValueError(f"No rows left in {split_name!r} after filtering.")

    sources = np.asarray(ds["source"])
    labels = (sources != args.human_source).astype(np.int64)
    counts = Counter(sources.tolist())
    print(
        f"Loaded {split_name!r}: n={len(texts)}, "
        f"human={int((labels == 0).sum())}, machine={int((labels == 1).sum())}, "
        f"sources={len(counts)}, normalize_mode={args.normalize_mode}"
    )
    for source, n in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {source!r}: {n}")

    print(f"Extracting ASCII features for {split_name!r} once...")
    X = extract_all_features(texts, args)
    return SplitFeatures(split_name, sources, labels, X)


def machine_sources(split: SplitFeatures, human_source: str) -> list[str]:
    sources = sorted(set(split.sources.tolist()) - {human_source})
    if not sources:
        raise ValueError(f"No machine sources found in {split.name!r} after filtering.")
    return sources


def indices_for_sources(
    split: SplitFeatures, human_source: str, model_sources: list[str]
) -> np.ndarray:
    keep = (split.sources == human_source) | np.isin(split.sources, model_sources)
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        raise ValueError(f"No rows selected from {split.name!r}.")
    labels = split.labels[idx]
    counts = np.bincount(labels, minlength=2)
    if counts[0] == 0 or counts[1] == 0:
        raise ValueError(
            f"Selection from {split.name!r} needs human and machine rows; "
            f"got human={int(counts[0])}, machine={int(counts[1])}."
        )
    return idx


def make_model(args: argparse.Namespace) -> Any:
    if args.classifier == "tree":
        return DecisionTreeClassifier(
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
            class_weight=args.class_weight,
        )
    if args.classifier == "forest":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            class_weight=args.class_weight,
            n_jobs=args.n_jobs,
        )
    if args.classifier == "logreg":
        return LogisticRegression(
            max_iter=args.max_iter,
            random_state=args.random_state,
            class_weight=args.class_weight,
            solver="lbfgs",
            C=args.logreg_c,
        )
    raise ValueError(f"Unknown classifier: {args.classifier!r}")


def positive_scores(model: Any, X: np.ndarray) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("This script expects a classifier with predict_proba.")
    classes = list(getattr(model, "classes_", [0, 1]))
    positive_idx = classes.index(1) if 1 in classes else -1
    return model.predict_proba(X)[:, positive_idx].astype(np.float64, copy=False)


def binary_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    preds = scores >= 0.5
    counts = np.bincount(labels.astype(int), minlength=2)
    return {
        "n": int(labels.size),
        "human": int(counts[0]),
        "machine": int(counts[1]),
        "accuracy": float(accuracy_score(labels, preds)),
        "log_loss": float(log_loss(labels, scores, labels=[0, 1])),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


def per_source_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    *,
    human_source: str,
    model_sources: list[str],
) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for source in model_sources:
        mask = (sources == human_source) | (sources == source)
        if not mask.any():
            continue
        out[source] = binary_metrics(scores[mask], labels[mask])
    return out


def print_metrics(prefix: str, metrics: dict[str, float | int]) -> None:
    print(
        f"{prefix}: n={metrics['n']} human={metrics['human']} "
        f"machine={metrics['machine']} accuracy={metrics['accuracy']:.4f} "
        f"log_loss={metrics['log_loss']:.4f} auroc={metrics['auroc']:.4f} "
        f"auprc={metrics['auprc']:.4f} f1={metrics['f1']:.4f}"
    )


METRIC_KEYS = ("accuracy", "log_loss", "auroc", "auprc", "f1")


def summarize_metric_rows(rows: list[dict[str, float | int]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}

    summary: dict[str, Any] = {
        "n": int(sum(int(row["n"]) for row in rows)),
        "human": int(sum(int(row["human"]) for row in rows)),
        "machine": int(sum(int(row["machine"]) for row in rows)),
        "n_results": len(rows),
    }
    for key in METRIC_KEYS:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    per_source_rows: dict[str, list[dict[str, float | int]]] = {}
    for result in results:
        for source, metrics in result["per_test_source"].items():
            per_source_rows.setdefault(source, []).append(metrics)

    return {
        "n_plans": len(results),
        "train": summarize_metric_rows([result["train"] for result in results]),
        "test": summarize_metric_rows([result["test"] for result in results]),
        "per_test_source": {
            source: summarize_metric_rows(rows)
            for source, rows in sorted(per_source_rows.items())
        },
    }


def print_aggregate_summary(aggregate: dict[str, Any]) -> None:
    def metric_summary(scope: str, key: str) -> str:
        row = aggregate[scope][key]
        return (
            f"{key}_mean={row['mean']:.4f}, {key}_std={row['std']:.4f}, "
            f"{key}_min={row['min']:.4f}, {key}_max={row['max']:.4f}"
        )

    print(f"\nAggregate across {aggregate['n_plans']} plan(s):")
    for scope in ("train", "test"):
        print(
            f"  {scope}: {metric_summary(scope, 'auroc')}, "
            f"{metric_summary(scope, 'auprc')}"
        )

    per_source = aggregate["per_test_source"]
    if per_source:
        print("\nAggregate held-out source metrics:")
        header = (
            f"{'source':<40} | {'plans':>5} | {'auroc_mean':>10} | {'auprc_mean':>10}"
        )
        print(header)
        print("-" * len(header))
        for source, row in per_source.items():
            print(
                f"{source:<40} | {row['n_results']:>5} | "
                f"{row['auroc']['mean']:>10.4f} | {row['auprc']['mean']:>10.4f}"
            )


def print_plan_result(result: dict[str, Any]) -> None:
    print(
        f"\n[{result['name']}] train_models={len(result['train_sources'])} "
        f"test_models={len(result['test_sources'])}"
    )
    print_metrics("Train", result["train"])
    print_metrics("Test", result["test"])
    for source, metrics in sorted(result["per_test_source"].items()):
        print(
            f"  test_source={source!r} auroc={metrics['auroc']:.4f} "
            f"auprc={metrics['auprc']:.4f}"
        )
    if result["output"]:
        print(f"Saved model to {result['output']}")


def compact_result_for_json(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": result["name"],
        "train_sources": result["train_sources"],
        "test_sources": result["test_sources"],
        "train": result["train"],
        "test": result["test"],
        "per_test_source": result["per_test_source"],
    }


def save_model(model: Any, output_dir: str | None, plan_name: str) -> str | None:
    if not output_dir:
        return None
    path = Path(output_dir) / f"{sanitize_name(plan_name)}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    return str(path)


def run_plan(
    plan: Plan,
    train: SplitFeatures,
    test: SplitFeatures,
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_idx = indices_for_sources(train, args.human_source, plan.train_sources)
    test_idx = indices_for_sources(test, args.human_source, plan.test_sources or [])

    model = make_model(args)
    model.fit(train.X[train_idx], train.labels[train_idx])

    train_scores = positive_scores(model, train.X[train_idx])
    test_scores = positive_scores(model, test.X[test_idx])
    train_metrics = binary_metrics(train_scores, train.labels[train_idx])
    test_metrics = binary_metrics(test_scores, test.labels[test_idx])

    test_sources = test.sources[test_idx]
    per_source = per_source_metrics(
        test_scores,
        test.labels[test_idx],
        test_sources,
        human_source=args.human_source,
        model_sources=plan.test_sources or [],
    )

    output = save_model(model, args.output_dir, plan.name)

    return {
        "name": plan.name,
        "train_sources": plan.train_sources,
        "test_sources": plan.test_sources,
        "output": output,
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "train": train_metrics,
        "test": test_metrics,
        "per_test_source": per_source,
    }


def run_plans(
    plans: list[Plan],
    train: SplitFeatures,
    test: SplitFeatures,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if args.plan_workers <= 1 or len(plans) <= 1:
        results = []
        for plan in plans:
            result = run_plan(plan, train, test, args)
            print_plan_result(result)
            results.append(result)
        return results

    worker_count = min(args.plan_workers, len(plans))
    print(f"\nRunning {len(plans)} plans with {worker_count} parallel worker(s)...")
    results: list[dict[str, Any] | None] = [None] * len(plans)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_idx = {
            executor.submit(run_plan, plan, train, test, args): idx
            for idx, plan in enumerate(plans)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()
            results[idx] = result
            print(f"Finished plan {idx + 1}/{len(plans)}: {result['name']}")

    ordered_results = [result for result in results if result is not None]
    for result in ordered_results:
        print_plan_result(result)
    return ordered_results


def run(args: argparse.Namespace) -> None:
    data = tmd.load_merged_dataset(args.dataset_path)
    train = materialize_split(data, args.split, args)
    test = materialize_split(data, args.test_split, args)
    train_machine_sources = machine_sources(train, args.human_source)
    test_machine_sources = machine_sources(test, args.human_source)
    plans = load_plans(args, train_machine_sources, test_machine_sources)

    print("\nPlans:")
    for plan in plans:
        print(
            f"  {plan.name}: train={plan.train_sources}, " f"test={plan.test_sources}"
        )

    results = run_plans(plans, train, test, args)
    aggregate = aggregate_results(results)
    print_aggregate_summary(aggregate)

    if args.metrics_json:
        payload = {
            "aggregate": aggregate,
            "plans": [compact_result_for_json(result) for result in results],
        }
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote metrics to {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train ASCII detectors on selected model sources and test on held-out "
            "model sources to measure source generalization."
        )
    )
    parser.add_argument("--dataset-path", default="dataset/merged_dataset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--human-source", default="human")
    parser.add_argument(
        "--plans-json",
        default=None,
        help=(
            "JSON list of plans, or object with 'plans'. Each plan needs "
            "train_sources/train_models and optional test_sources/test_models."
        ),
    )
    parser.add_argument(
        "--plan",
        action="append",
        default=[],
        help='Inline JSON plan, e.g. \'{"name":"gpt_to_rest","train_sources":["gpt-4.1-mini"]}\'.',
    )
    parser.add_argument("--plan-name", default="cli_plan")
    parser.add_argument("--train-source", nargs="+", default=None)
    parser.add_argument("--test-source", nargs="+", default=None)
    parser.add_argument(
        "--text-field",
        default="response",
        help="Column to train on, or 'prompt_response' to join prompt and response.",
    )
    parser.add_argument(
        "--normalize-mode", choices=["raw", "whitespace"], default="raw"
    )
    parser.add_argument(
        "--filter",
        nargs="+",
        action="extend",
        default=[],
        help="Extra exact-match filters as COLUMN=VALUE. Can be repeated.",
    )
    parser.add_argument("--filter-num-proc", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--fit-batch-size", type=int, default=1024)
    parser.add_argument(
        "--plan-workers",
        type=int,
        default=1,
        help=(
            "Number of plans to train/evaluate in parallel after feature extraction. "
            "Use 1 for serial output."
        ),
    )
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
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--metrics-json", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
