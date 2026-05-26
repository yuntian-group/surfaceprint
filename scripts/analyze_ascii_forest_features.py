from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for path in (str(SCRIPT_DIR), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import analyze_ascii_features as aaf  # noqa: E402


def ensure_forest_model(model: Any) -> None:
    if not hasattr(model, "estimators_"):
        raise ValueError(
            f"Model {type(model).__module__}.{type(model).__name__} is not a "
            "tree ensemble (missing estimators_). Use "
            "scripts/analyze_ascii_features.py for a single decision tree."
        )
    if getattr(model, "feature_importances_", None) is None:
        raise ValueError("Forest model does not expose feature_importances_.")


def forest_summary(
    model: Any,
    model_path: Path,
    feature_names: list[str],
    disabled_feature_groups: list[str],
) -> dict[str, Any]:
    summary = aaf.model_summary(
        model, model_path, feature_names, disabled_feature_groups
    )
    estimators = list(model.estimators_)
    depths = [int(est.tree_.max_depth) for est in estimators]
    leaves = [int(est.get_n_leaves()) for est in estimators]
    nodes = [int(est.tree_.node_count) for est in estimators]
    summary.update(
        {
            "n_estimators": int(getattr(model, "n_estimators", len(estimators))),
            "max_features": getattr(model, "max_features", None),
            "bootstrap": getattr(model, "bootstrap", None),
            "oob_score": getattr(model, "oob_score", None),
            "tree_depth_min": int(min(depths)) if depths else None,
            "tree_depth_max": int(max(depths)) if depths else None,
            "tree_depth_mean": float(np.mean(depths)) if depths else None,
            "tree_leaves_min": int(min(leaves)) if leaves else None,
            "tree_leaves_max": int(max(leaves)) if leaves else None,
            "tree_leaves_mean": float(np.mean(leaves)) if leaves else None,
            "tree_nodes_mean": float(np.mean(nodes)) if nodes else None,
        }
    )
    return summary


def split_usage_rows(
    model: Any,
    feature_names: list[str],
    feature_groups: list[str],
) -> list[dict[str, Any]]:
    counts: dict[int, int] = defaultdict(int)
    thresholds: dict[int, list[float]] = defaultdict(list)
    total_splits = 0

    for estimator in model.estimators_:
        tree = estimator.tree_
        for node_id in range(tree.node_count):
            feature_idx = int(tree.feature[node_id])
            if feature_idx < 0:
                continue
            counts[feature_idx] += 1
            thresholds[feature_idx].append(float(tree.threshold[node_id]))
            total_splits += 1

    rows: list[dict[str, Any]] = []
    for feature_idx, count in counts.items():
        if feature_idx >= len(feature_names):
            feature_name = f"unknown_feature_{feature_idx}"
            group = "?"
        else:
            feature_name = feature_names[feature_idx]
            group = feature_groups[feature_idx]
        thresh = thresholds[feature_idx]
        rows.append(
            {
                "feature_index": feature_idx,
                "group": group,
                "group_label": aaf.GROUP_LABELS.get(group, "unknown"),
                "feature": feature_name,
                "split_count": int(count),
                "split_fraction": float(count / total_splits) if total_splits else 0.0,
                "mean_threshold": float(np.mean(thresh)),
                "median_threshold": float(np.median(thresh)),
            }
        )
    return sorted(rows, key=lambda row: row["split_count"], reverse=True)


def per_tree_rows(model: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tree_index, estimator in enumerate(model.estimators_):
        tree = estimator.tree_
        rows.append(
            {
                "tree_index": tree_index,
                "depth": int(tree.max_depth),
                "n_leaves": int(estimator.get_n_leaves()),
                "n_nodes": int(tree.node_count),
                "root_samples": int(tree.n_node_samples[0]),
            }
        )
    return rows


def print_forest_summary(summary: dict[str, Any]) -> None:
    aaf.print_summary(summary)
    print(
        "  forest: "
        f"n_estimators={summary.get('n_estimators')}, "
        f"max_features={summary.get('max_features')}, "
        f"bootstrap={summary.get('bootstrap')}, "
        f"oob_score={summary.get('oob_score')}"
    )
    print(
        "  tree depth (min/mean/max): "
        f"{summary.get('tree_depth_min')} / "
        f"{summary.get('tree_depth_mean', 0):.2f} / "
        f"{summary.get('tree_depth_max')}"
    )
    print(
        "  tree leaves (min/mean/max): "
        f"{summary.get('tree_leaves_min')} / "
        f"{summary.get('tree_leaves_mean', 0):.2f} / "
        f"{summary.get('tree_leaves_max')}"
    )


def print_split_usage(rows: list[dict[str, Any]], top: int | None) -> None:
    rows_to_print = rows if top is None else rows[:top]
    print("\nSplit usage across forest (internal nodes only)")
    for rank, row in enumerate(rows_to_print, start=1):
        print(
            f"  {rank:>2}. {row['feature']:<45} "
            f"group={row['group']} idx={row['feature_index']:02d} "
            f"splits={row['split_count']} "
            f"frac={row['split_fraction']:.4f} "
            f"threshold~={row['median_threshold']:.6f}"
        )


def print_per_tree_stats(rows: list[dict[str, Any]], top: int | None) -> None:
    rows_to_print = rows if top is None else rows[:top]
    print("\nPer-tree structure")
    for row in rows_to_print:
        print(
            f"  tree={row['tree_index']:>3} "
            f"depth={row['depth']} "
            f"leaves={row['n_leaves']} "
            f"nodes={row['n_nodes']} "
            f"root_samples={row['root_samples']}"
        )


def print_tree(model: Any, tree_index: int, feature_names: list[str]) -> None:
    try:
        from sklearn.tree import export_text
    except ImportError as exc:
        raise SystemExit(
            "Tree export requires scikit-learn. Install it with `pip install scikit-learn`."
        ) from exc

    estimators = list(model.estimators_)
    if tree_index < 0 or tree_index >= len(estimators):
        raise ValueError(
            f"tree_index must be in [0, {len(estimators) - 1}]; got {tree_index}."
        )
    print(f"\nDecision tree #{tree_index}")
    print(
        export_text(
            estimators[tree_index],
            feature_names=feature_names,
            decimals=6,
            spacing=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze feature importances and split usage for a trained random "
            "forest ASCII classifier."
        )
    )
    parser.add_argument(
        "--model",
        default="models/merged_ascii_author_classifiers/models_only_forest.pkl",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Only print the top N features / split-usage rows.",
    )
    parser.add_argument(
        "--show-zero",
        action="store_true",
        help="Include zero-importance features in the feature ranking.",
    )
    parser.add_argument(
        "--split-usage",
        action="store_true",
        help="Print how often each feature is used as a split across all trees.",
    )
    parser.add_argument(
        "--per-tree",
        action="store_true",
        help="Print depth/leaf/node counts for each tree in the forest.",
    )
    parser.add_argument(
        "--tree-index",
        type=int,
        default=None,
        help="Print sklearn text export for the tree at this index (0-based).",
    )
    parser.add_argument(
        "--disable-feature-group",
        nargs="+",
        action="extend",
        default=[],
        help=(
            "Feature groups disabled when the model was trained. "
            "Examples: E, whitespace, A, prose, B, markdown, C, math, D, code."
        ),
    )
    parser.add_argument(
        "--disable-whitespace-features",
        action="store_true",
        help="Analyze a model trained without whitespace/invisible features (group E).",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to write the full analysis as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    model = aaf.load_model(model_path)
    ensure_forest_model(model)

    disabled_feature_groups = list(args.disable_feature_group)
    if args.disable_whitespace_features:
        disabled_feature_groups.append("E")
    feature_names, feature_groups = aaf.selected_feature_metadata(
        disabled_feature_groups
    )

    all_feature_rows = aaf.feature_importance_rows(
        model,
        include_zero=True,
        feature_names=feature_names,
        feature_groups=feature_groups,
    )
    feature_rows = (
        all_feature_rows
        if args.show_zero
        else [row for row in all_feature_rows if row["importance"] > 0.0]
    )
    group_rows = aaf.group_importance_rows(all_feature_rows)
    split_rows = split_usage_rows(model, feature_names, feature_groups)
    tree_rows = per_tree_rows(model)
    summary = forest_summary(model, model_path, feature_names, disabled_feature_groups)

    print_forest_summary(summary)
    aaf.print_group_importances(group_rows)
    aaf.print_feature_importances(feature_rows, args.top)

    if args.split_usage:
        print_split_usage(split_rows, args.top)
    if args.per_tree:
        print_per_tree_stats(tree_rows, args.top)
    if args.tree_index is not None:
        print_tree(model, args.tree_index, feature_names)

    if args.json_output is not None:
        payload = {
            "summary": summary,
            "group_importances": group_rows,
            "feature_importances": feature_rows,
            "split_usage": split_rows,
            "per_tree": tree_rows,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=aaf.json_ready)
            f.write("\n")
        print(f"\nWrote JSON analysis to {args.json_output}")


if __name__ == "__main__":
    main()
