"""Score UTF-8 documents with a trained SurfacePrint detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_merged_ascii_detector import normalize_text, resolve_ascii_detector_class


def load_detector(args: argparse.Namespace):
    metadata_path = args.metadata or args.model.with_suffix(".metrics.json")
    metadata = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    elif args.metadata:
        raise ValueError(f"Metadata file does not exist: {metadata_path}")

    mode = args.normalize_mode or metadata.get("normalize_mode")
    if mode not in ("raw", "whitespace"):
        raise ValueError(
            "Supply the training .metrics.json with --metadata, or set "
            "--normalize-mode to the value used for training."
        )
    if args.normalize_mode and metadata.get("normalize_mode", mode) != mode:
        raise ValueError("--normalize-mode differs from the training metadata.")
    detector_name = args.ascii_detector or metadata.get("ascii_detector", "full")
    if args.ascii_detector and metadata.get("ascii_detector", detector_name) != detector_name:
        raise ValueError("--ascii-detector differs from the training metadata.")
    disabled = args.disable_feature_group
    if disabled is None:
        disabled = metadata.get("disable_feature_group", [])
    elif metadata and disabled != metadata.get("disable_feature_group", []):
        raise ValueError("--disable-feature-group differs from the training metadata.")

    detector = resolve_ascii_detector_class(detector_name)({
        "model_path": str(args.model),
        "num_workers": args.num_workers,
        "disabled_feature_groups": disabled,
    })
    if list(detector.model.classes_) != [0, 1]:
        raise ValueError("Use a binary detector trained with human=0 and AI=1.")
    if detector.model.n_features_in_ != detector.feature_dim:
        raise ValueError("Feature configuration differs from the saved model.")
    return detector, mode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--metadata", type=Path, help="Training metrics JSON; defaults to MODEL.metrics.json.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--file", nargs="+", type=Path, help="UTF-8 files, one document per file.")
    inputs.add_argument("--text", help="One document supplied as a command-line argument.")
    parser.add_argument("--normalize-mode", choices=("raw", "whitespace"))
    parser.add_argument("--ascii-detector", choices=("full", "basic", "binary"))
    parser.add_argument("--disable-feature-group", nargs="+")
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    try:
        detector, mode = load_detector(args)
        if args.file:
            documents = [(str(path), path.read_bytes().decode("utf-8")) for path in args.file]
        else:
            documents = [(None, args.text)]
        scores = detector.batch_predict([normalize_text(text, mode) for _, text in documents])
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for (name, _), score in zip(documents, scores):
        print(json.dumps({"file": name, "ai_score": score}, ensure_ascii=False))


if __name__ == "__main__":
    main()
