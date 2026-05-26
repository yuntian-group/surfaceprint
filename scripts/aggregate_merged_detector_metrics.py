#!/usr/bin/env python3
"""Merge per-model `*.metrics.json` files and print a summary table + optional JSON."""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path


def load_metrics_paths_from_list(list_file: Path) -> list[Path]:
    lines = list_file.read_text(encoding="utf-8").splitlines()
    paths: list[Path] = []
    for line in lines:
        p = Path(line.strip())
        if p.is_file():
            paths.append(p)
    return paths


def load_metrics_paths_from_glob(pattern: str) -> list[Path]:
    return sorted(Path(p) for p in glob(pattern, recursive=True) if Path(p).is_file())


def flatten_row(obj: dict) -> dict[str, object]:
    train = obj.get("train") or {}
    val = obj.get("validation")
    out: dict[str, object] = {
        "target_source": obj.get("target_source", ""),
        "n_train": obj.get("n_train", ""),
        "output": obj.get("output", ""),
    }
    for prefix, block in (("train", train), ("val", val if isinstance(val, dict) else {})):
        if prefix == "val" and not isinstance(val, dict):
            for key in ("accuracy", "log_loss", "auroc", "auprc", "f1"):
                out[f"{prefix}_{key}"] = None
            continue
        for key in ("accuracy", "log_loss", "auroc", "auprc", "f1"):
            out[f"{prefix}_{key}"] = block.get(key)
    return out


def _fmt_float(x: object, decimals: int = 4) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{decimals}f}"
    except (TypeError, ValueError):
        return str(x)


def _fmt_int(x: object) -> str:
    if x is None:
        return "—"
    try:
        return str(int(x))
    except (TypeError, ValueError):
        return str(x)


def render_table(rows: list[dict[str, object]]) -> str:
    """Fixed-width ASCII table (no third-party deps)."""
    if not rows:
        return "(no metrics rows)\n"

    headers = [
        "target_source",
        "n_train",
        "tr_acc",
        "tr_auroc",
        "tr_auprc",
        "tr_f1",
        "val_acc",
        "val_auroc",
        "val_auprc",
        "val_f1",
    ]

    def cell(row: dict[str, object], h: str) -> str:
        if h == "target_source":
            return str(row.get("target_source", ""))
        if h == "n_train":
            return _fmt_int(row.get("n_train"))
        if h.startswith("tr_"):
            key = "train_" + h[3:].replace("acc", "accuracy")
            return _fmt_float(row.get(key))
        if h.startswith("val_"):
            key = "val_" + h[4:].replace("acc", "accuracy")
            return _fmt_float(row.get(key))
        return str(row.get(h, ""))

    table_rows: list[list[str]] = []
    for r in rows:
        table_rows.append([cell(r, h) for h in headers])

    col_widths: list[int] = [len(h) for h in headers]
    for tr in table_rows:
        for i, text in enumerate(tr):
            col_widths[i] = max(col_widths[i], len(text))

    sep = "-+-".join("-" * w for w in col_widths)
    def fmt_line(cells: list[str]) -> str:
        return " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells))

    lines = [fmt_line(headers), sep]
    for tr in table_rows:
        lines.append(fmt_line(tr))
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Load merged ASCII detector per-model metrics JSON files, "
            "write one summary JSON, and print a table."
        )
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--list-file",
        type=Path,
        help="Newline-separated paths to *.metrics.json (written during training).",
    )
    src.add_argument(
        "--glob",
        dest="metrics_glob",
        metavar="PATTERN",
        help='Glob for metrics files, e.g. "models/merged_ascii_detectors/*.metrics.json".',
    )
    p.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path for merged summary JSON (n_models + results list).",
    )
    p.add_argument(
        "--sort-by",
        choices=["target_source", "val_auroc", "train_auroc"],
        default="target_source",
        help="Column to sort the printed table by (default: target_source).",
    )
    p.add_argument(
        "--no-table",
        action="store_true",
        help="Only write JSON; do not print the table.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_file is not None:
        paths = load_metrics_paths_from_list(args.list_file)
    else:
        paths = load_metrics_paths_from_glob(args.metrics_glob)

    raw_results: list[dict] = []
    for path in paths:
        raw_results.append(json.loads(path.read_text(encoding="utf-8")))

    summary = {"n_models": len(raw_results), "results": raw_results}
    out_path: Path = args.output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote aggregate summary to {out_path}")

    flat = [flatten_row(r) for r in raw_results]

    def sort_key(row: dict[str, object]) -> tuple:
        if args.sort_by == "target_source":
            return (str(row.get("target_source", "")),)
        if args.sort_by == "val_auroc":
            v = row.get("val_auroc")
            missing = v is None
            score = 0.0 if v is None else -float(v)
            return (missing, score, str(row.get("target_source", "")))
        if args.sort_by == "train_auroc":
            v = row.get("train_auroc")
            missing = v is None
            score = 0.0 if v is None else -float(v)
            return (missing, score, str(row.get("target_source", "")))
        return (str(row.get("target_source", "")),)

    flat_sorted = sorted(flat, key=sort_key)

    if not args.no_table:
        print()
        print(render_table(flat_sorted))


if __name__ == "__main__":
    main()
