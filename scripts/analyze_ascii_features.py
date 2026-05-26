from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detectors import ascii_detector as full_ascii
from detectors import ascii_detector_basic as basic_ascii
from train_merged_ascii_detector import ASCII_DETECTOR_CHOICES


@dataclass(frozen=True)
class FeatureRegistry:
    name: str
    feature_names: list[str]
    feature_groups: list[str]
    group_labels: dict[str, str]
    get_feature_indices: Callable[[Sequence[str] | str | None], list[int]]
    disable_whitespace_group: str | None = None


REGISTRIES: dict[str, FeatureRegistry] = {
    "full": FeatureRegistry(
        name="full",
        feature_names=full_ascii.FEATURE_NAMES,
        feature_groups=full_ascii.FEATURE_GROUPS,
        group_labels={
            "A": "prose typography",
            "B": "Markdown structure",
            "C": "math and LaTeX",
            "D": "code surface",
            "E": "whitespace and invisible",
        },
        get_feature_indices=full_ascii.get_feature_indices,
        disable_whitespace_group="E",
    ),
    "basic": FeatureRegistry(
        name="basic",
        feature_names=basic_ascii.FEATURE_NAMES,
        feature_groups=basic_ascii.FEATURE_GROUPS,
        group_labels={
            "char": "character presence",
            "word": "word presence",
            "pattern": "formatting pattern",
        },
        get_feature_indices=basic_ascii.get_feature_indices,
        disable_whitespace_group=None,
    ),
}


def load_model(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def metrics_json_path(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.stem}.metrics.json")


def read_ascii_detector_from_metrics(model_path: Path) -> str | None:
    metrics_path = metrics_json_path(model_path)
    if not metrics_path.is_file():
        return None
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    value = payload.get("ascii_detector")
    if value in REGISTRIES:
        return str(value)
    return None


def infer_ascii_detector(model: Any, registry_names: Sequence[str]) -> str | None:
    n_features = getattr(model, "n_features_in_", None)
    if n_features is None:
        return None

    matches = [
        name
        for name in registry_names
        if int(n_features) == len(REGISTRIES[name].feature_names)
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def resolve_registry_name(
    ascii_detector: str,
    model: Any,
    model_path: Path,
) -> str:
    if ascii_detector != "auto":
        if ascii_detector not in REGISTRIES:
            raise ValueError(
                f"Unknown ascii detector {ascii_detector!r}; "
                f"expected one of {(*ASCII_DETECTOR_CHOICES, 'auto')}."
            )
        return ascii_detector

    detected = read_ascii_detector_from_metrics(model_path)
    if detected is None:
        detected = infer_ascii_detector(model, REGISTRIES)
    if detected is None:
        n_features = getattr(model, "n_features_in_", "?")
        raise ValueError(
            "Could not infer ascii detector automatically. Pass "
            f"--ascii-detector full or basic (model has {n_features} features)."
        )
    return detected


def resolve_registry(
    ascii_detector: str,
    model: Any,
    model_path: Path,
    disabled_feature_groups: list[str],
) -> FeatureRegistry:
    registry = REGISTRIES[resolve_registry_name(ascii_detector, model, model_path)]
    expected = len(registry.get_feature_indices(disabled_feature_groups))
    actual = getattr(model, "n_features_in_", None)
    if actual is not None and int(actual) != expected:
        raise ValueError(
            f"Feature count mismatch for --ascii-detector={registry.name}: "
            f"model has {int(actual)} features, registry expects {expected} "
            f"with disabled groups {disabled_feature_groups!r}."
        )
    return registry


def selected_feature_metadata(
    registry: FeatureRegistry,
    disabled_feature_groups: list[str],
) -> tuple[list[str], list[str]]:
    feature_indices = registry.get_feature_indices(disabled_feature_groups)
    return (
        [registry.feature_names[idx] for idx in feature_indices],
        [registry.feature_groups[idx] for idx in feature_indices],
    )


def feature_importance_rows(
    model: Any,
    *,
    include_zero: bool,
    feature_names: list[str],
    feature_groups: list[str],
    group_labels: dict[str, str],
) -> list[dict[str, Any]]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        raise ValueError("Model does not expose feature_importances_.")

    rows: list[dict[str, Any]] = []
    for idx, value in enumerate(np.asarray(importances, dtype=float)):
        if idx >= len(feature_names):
            name = f"unknown_feature_{idx}"
            group = "?"
        else:
            name = feature_names[idx]
            group = feature_groups[idx]
        if include_zero or value > 0.0:
            rows.append(
                {
                    "index": idx,
                    "group": group,
                    "group_label": group_labels.get(group, "unknown"),
                    "feature": name,
                    "importance": float(value),
                }
            )
    return sorted(rows, key=lambda row: row["importance"], reverse=True)


def group_importance_rows(
    feature_rows: list[dict[str, Any]],
    group_labels: dict[str, str],
) -> list[dict[str, Any]]:
    totals: dict[str, float] = defaultdict(float)
    for group in group_labels:
        totals[group] = 0.0
    for row in feature_rows:
        totals[str(row["group"])] += float(row["importance"])

    rows = [
        {
            "group": group,
            "group_label": group_labels.get(group, "unknown"),
            "importance": value,
        }
        for group, value in totals.items()
    ]
    return sorted(rows, key=lambda row: row["importance"], reverse=True)


def tree_split_rows(
    model: Any,
    feature_names: list[str],
    feature_groups: list[str],
    group_labels: dict[str, str],
) -> list[dict[str, Any]]:
    tree = getattr(model, "tree_", None)
    if tree is None:
        return []

    rows: list[dict[str, Any]] = []
    stack = [(0, 0)]
    while stack:
        node_id, depth = stack.pop()
        feature_idx = int(tree.feature[node_id])
        if feature_idx < 0:
            continue

        if feature_idx >= len(feature_names):
            feature_name = f"unknown_feature_{feature_idx}"
            group = "?"
        else:
            feature_name = feature_names[feature_idx]
            group = feature_groups[feature_idx]
        rows.append(
            {
                "node": node_id,
                "depth": depth,
                "feature_index": feature_idx,
                "group": group,
                "group_label": group_labels.get(group, "unknown"),
                "feature": feature_name,
                "threshold": float(tree.threshold[node_id]),
                "impurity": float(tree.impurity[node_id]),
                "samples": int(tree.n_node_samples[node_id]),
                "weighted_samples": float(tree.weighted_n_node_samples[node_id]),
                "left": int(tree.children_left[node_id]),
                "right": int(tree.children_right[node_id]),
            }
        )
        stack.append((int(tree.children_right[node_id]), depth + 1))
        stack.append((int(tree.children_left[node_id]), depth + 1))
    return rows


def model_summary(
    model: Any,
    model_path: Path,
    registry: FeatureRegistry,
    feature_names: list[str],
    disabled_feature_groups: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "model_path": str(model_path),
        "ascii_detector": registry.name,
        "model_type": f"{type(model).__module__}.{type(model).__name__}",
        "classes": getattr(model, "classes_", None),
        "n_features_in": getattr(model, "n_features_in_", None),
        "registered_features": len(feature_names),
        "all_registered_features": len(registry.feature_names),
        "disabled_feature_groups": disabled_feature_groups,
    }

    tree = getattr(model, "tree_", None)
    if tree is not None:
        summary.update(
            {
                "tree_depth": int(tree.max_depth),
                "node_count": int(tree.node_count),
                "leaf_count": int(model.get_n_leaves()),
                "root_samples": int(tree.n_node_samples[0]),
                "root_weighted_samples": float(tree.weighted_n_node_samples[0]),
            }
        )
    return summary


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def print_summary(summary: dict[str, Any]) -> None:
    print("Model summary")
    print(f"  path: {summary['model_path']}")
    print(f"  ascii_detector: {summary.get('ascii_detector')}")
    print(f"  type: {summary['model_type']}")
    print(f"  classes: {summary.get('classes')}")
    print(
        "  features: "
        f"{summary.get('n_features_in')} in model / "
        f"{summary['registered_features']} registered"
    )
    if "tree_depth" in summary:
        print(
            "  tree: "
            f"depth={summary['tree_depth']}, "
            f"nodes={summary['node_count']}, "
            f"leaves={summary['leaf_count']}, "
            f"root_samples={summary['root_samples']}"
        )


def print_group_importances(rows: list[dict[str, Any]]) -> None:
    print("\nGroup importances")
    for row in rows:
        print(f"  {row['group']} ({row['group_label']}): " f"{row['importance']:.6f}")


def print_feature_importances(rows: list[dict[str, Any]], top: int | None) -> None:
    rows_to_print = rows if top is None else rows[:top]
    print("\nFeature importances")
    for rank, row in enumerate(rows_to_print, start=1):
        print(
            f"  {rank:>2}. {row['feature']:<45} "
            f"group={row['group']} idx={row['index']:02d} "
            f"importance={row['importance']:.6f}"
        )


def print_split_nodes(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("\nSplit nodes: unavailable")
        return

    print("\nSplit nodes")
    for row in rows:
        indent = "  " * int(row["depth"])
        print(
            f"{indent}- node={row['node']} depth={row['depth']} "
            f"{row['feature']} <= {row['threshold']:.6f} "
            f"(group={row['group']}, samples={row['samples']}, "
            f"impurity={row['impurity']:.6f})"
        )


AUTO_TREE_MAX_DEPTH = 5


def should_print_tree(model: Any, *, no_tree: bool, force_tree: bool) -> bool:
    if no_tree:
        return False
    if force_tree:
        return getattr(model, "tree_", None) is not None
    tree = getattr(model, "tree_", None)
    if tree is None:
        return False
    return int(tree.max_depth) <= AUTO_TREE_MAX_DEPTH


def tree_class_names(model: Any) -> list[str] | None:
    """Map binary 0/1 labels to human/llm for tree export."""
    classes = getattr(model, "classes_", None)
    if classes is None or len(classes) != 2:
        return None
    try:
        class_values = [int(c) for c in classes]
    except (TypeError, ValueError):
        return None
    if set(class_values) != {0, 1}:
        return None
    label_for_value = {0: "human", 1: "llm"}
    return [label_for_value[value] for value in class_values]


def print_tree(model: Any, feature_names: list[str]) -> None:
    try:
        from sklearn.tree import export_text
    except ImportError as exc:
        raise SystemExit(
            "Tree export requires scikit-learn. Install it with `pip install scikit-learn`."
        ) from exc

    print("\nDecision tree")
    export_kwargs: dict[str, Any] = {
        "feature_names": feature_names,
        "decimals": 6,
        "spacing": 2,
    }
    class_names = tree_class_names(model)
    if class_names is not None:
        export_kwargs["class_names"] = class_names
    print(export_text(model, **export_kwargs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze feature importances for a trained ASCII detector."
    )
    parser.add_argument("--model", default="models/ascii_detector.pkl")
    parser.add_argument(
        "--ascii-detector",
        choices=[*ASCII_DETECTOR_CHOICES, "auto"],
        default="auto",
        help=(
            "Feature registry to use. 'auto' reads ascii_detector from a sibling "
            ".metrics.json file or infers from feature count."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Only print the top N features. By default, print all selected features.",
    )
    parser.add_argument(
        "--show-zero",
        action="store_true",
        help="Include zero-importance features in the feature ranking.",
    )
    parser.add_argument(
        "--splits",
        action="store_true",
        help="Print decision-tree split nodes with thresholds.",
    )
    parser.add_argument(
        "--tree",
        action="store_true",
        help="Force printing the decision tree even when deeper than 5.",
    )
    parser.add_argument(
        "--no-tree",
        action="store_true",
        help="Skip the decision-tree text export (shown by default for depth <= 5).",
    )
    parser.add_argument(
        "--disable-feature-group",
        nargs="+",
        action="extend",
        default=[],
        help=(
            "Feature groups disabled when the model was trained. "
            "For full: E, whitespace, A, prose, B, markdown, C, math, D, code. "
            "For basic: char, word, pattern."
        ),
    )
    parser.add_argument(
        "--disable-whitespace-features",
        action="store_true",
        help="Analyze a full model trained without whitespace/invisible features (group E).",
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
    model = load_model(model_path)
    disabled_feature_groups = list(args.disable_feature_group)

    if args.disable_whitespace_features:
        if args.ascii_detector == "basic":
            raise ValueError(
                "--disable-whitespace-features applies only to --ascii-detector=full."
            )
        if args.ascii_detector == "auto":
            detected = resolve_registry_name("auto", model, model_path)
            if detected == "basic":
                raise ValueError(
                    "--disable-whitespace-features applies only to full-feature models."
                )
        disabled_feature_groups.append("E")

    registry = resolve_registry(
        args.ascii_detector,
        model,
        model_path,
        disabled_feature_groups,
    )
    if args.disable_whitespace_features and registry.name != "full":
        raise ValueError(
            "--disable-whitespace-features applies only to full-feature models."
        )

    feature_names, feature_groups = selected_feature_metadata(
        registry, disabled_feature_groups
    )
    group_labels = registry.group_labels

    all_feature_rows = feature_importance_rows(
        model,
        include_zero=True,
        feature_names=feature_names,
        feature_groups=feature_groups,
        group_labels=group_labels,
    )
    feature_rows = (
        all_feature_rows
        if args.show_zero
        else [row for row in all_feature_rows if row["importance"] > 0.0]
    )
    group_rows = group_importance_rows(all_feature_rows, group_labels)
    split_rows = tree_split_rows(model, feature_names, feature_groups, group_labels)
    summary = model_summary(
        model, model_path, registry, feature_names, disabled_feature_groups
    )

    print_summary(summary)
    print_group_importances(group_rows)
    print_feature_importances(feature_rows, args.top)

    if args.splits:
        print_split_nodes(split_rows)
    if should_print_tree(model, no_tree=args.no_tree, force_tree=args.tree):
        print_tree(model, feature_names)

    if args.json_output is not None:
        payload = {
            "summary": summary,
            "group_importances": group_rows,
            "feature_importances": feature_rows,
            "split_nodes": split_rows,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=json_ready)
            f.write("\n")
        print(f"\nWrote JSON analysis to {args.json_output}")


if __name__ == "__main__":
    main()
