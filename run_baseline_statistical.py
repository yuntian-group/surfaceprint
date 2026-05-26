from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_merged_ascii_detector as tmd  # noqa: E402

from detectors.binoculars import BinocularsDetector  # noqa: E402
from detectors.dna_detectllm import DNADetectLLMDetector  # noqa: E402
from detectors.fastdetect_gpt import FastDetectGPT  # noqa: E402
from detectors.lastde import LastdeDetector, LastdePPDetector  # noqa: E402
from detectors.simple_detector import (  # noqa: E402
    EntropyDetector,
    LRRDetector,
    LikelihoodDetector,
    LogRankDetector,
)

DETECTORS: dict[str, type] = {
    "likelihood": LikelihoodDetector,
    "logrank": LogRankDetector,
    "entropy": EntropyDetector,
    "lrr": LRRDetector,
    "fast_detect_gpt": FastDetectGPT,
    "binoculars": BinocularsDetector,
    "dna_detectllm": DNADetectLLMDetector,
    "lastde": LastdeDetector,
    "lastde_pp": LastdePPDetector,
}


def sanitize_model_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def effective_reference_model(scoring_model: str, reference_model: str | None) -> str:
    return reference_model if reference_model is not None else scoring_model


def build_result_stem(
    criterion: str, scoring_model: str, reference_model: str | None
) -> str:
    if criterion in {"binoculars", "dna_detectllm"}:
        if reference_model is None:
            raise ValueError(f"{criterion} results require reference_model (observer).")
        return (
            f"{criterion}.observer-{sanitize_model_name(reference_model)}"
            f".performer-{sanitize_model_name(scoring_model)}"
        )
    stem = f"{criterion}.scorer-{sanitize_model_name(scoring_model)}"
    if criterion in {"fast_detect_gpt", "lastde_pp"}:
        ref = effective_reference_model(scoring_model, reference_model)
        stem += f".ref-{sanitize_model_name(ref)}"
    return stem


def build_result_path(
    path: str,
    criterion: str,
    scoring_model: str,
    reference_model: str | None,
    *,
    suffix: str,
) -> Path:
    p = Path(path)
    stem = build_result_stem(criterion, scoring_model, reference_model)
    if p.suffix in {".json", ".npz"} or p.name.endswith(".metrics.json"):
        return p.parent / f"{stem}{suffix}"
    return p / f"{stem}{suffix}"


def is_main_rank() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def build_test_split(
    args: argparse.Namespace,
) -> tuple[list[str], list[int]]:
    data = tmd.load_merged_dataset(args.dataset_path)
    return tmd.build_split(data, args.split, args, "test")


def build_detector_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.criterion in {"binoculars", "dna_detectllm"}:
        if args.reference_model is None:
            raise ValueError(
                f"--reference-model (observer LM) is required for "
                f"--criterion={args.criterion}."
            )
        return {
            "model_name": args.reference_model,
            "performer_model": args.model_name,
            "device": args.device,
            "dtype": args.dtype,
            "max_length": args.max_length,
            "trust_remote_code": args.trust_remote_code,
        }

    config: dict[str, Any] = {
        "model_name": args.model_name,
        "device": args.device,
        "dtype": args.dtype,
        "max_length": args.max_length,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.criterion == "fast_detect_gpt":
        config["analytic"] = True
        if args.reference_model is not None:
            config["ref_model"] = args.reference_model
    elif args.criterion == "lastde_pp":
        if args.reference_model is not None:
            config["ref_model"] = args.reference_model
    return config


def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(scores)
    n_dropped = int((~finite).sum())
    s = scores[finite]
    y = labels[finite]
    auroc = float(roc_auc_score(y, s))
    return {
        "n": int(len(scores)),
        "n_scored": int(len(s)),
        "n_dropped": n_dropped,
        "positive": int((y == 1).sum()),
        "negative": int((y == 0).sum()),
        "auroc": auroc,
        "auroc_best_direction": max(auroc, 1.0 - auroc),
        "auprc": float(average_precision_score(y, s)),
        "auprc_inverted": float(average_precision_score(y, -s)),
        "score_mean_positive": (
            float(s[y == 1].mean()) if (y == 1).any() else float("nan")
        ),
        "score_mean_negative": (
            float(s[y == 0].mean()) if (y == 0).any() else float("nan")
        ),
    }


def print_metrics(criterion: str, model_name: str, metrics: dict[str, float]) -> None:
    print(
        f"[{criterion}] model={model_name} n={metrics['n']} "
        f"pos={metrics['positive']} neg={metrics['negative']} "
        f"auroc={metrics['auroc']:.4f} "
        f"(best_direction={metrics['auroc_best_direction']:.4f}) "
        f"auprc={metrics['auprc']:.4f} "
        f"score_mean[pos]={metrics['score_mean_positive']:.4f} "
        f"score_mean[neg]={metrics['score_mean_negative']:.4f}"
    )


def _metrics_output_path(args: argparse.Namespace) -> Path | None:
    if not args.metrics_json:
        return None
    return build_result_path(
        args.metrics_json,
        args.criterion,
        args.model_name,
        args.reference_model,
        suffix=".metrics.json",
    )


def _skip_if_metrics_exist(args: argparse.Namespace) -> bool:
    if args.force_rerun:
        return False
    metrics_path = _metrics_output_path(args)
    if metrics_path is None or not metrics_path.is_file():
        return False
    if not is_main_rank():
        return True
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    print(f"Skipping {args.criterion}: metrics already exist at {metrics_path}")
    existing = payload.get("metrics")
    if isinstance(existing, dict) and existing:
        print_metrics(args.criterion, args.model_name, existing)
    return True


def run(args: argparse.Namespace) -> None:
    if args.criterion not in DETECTORS:
        raise ValueError(
            f"Unknown criterion {args.criterion!r}; choose from {sorted(DETECTORS)}"
        )

    args.model_name = (
        args.scoring_model if args.scoring_model is not None else args.model
    )
    if _skip_if_metrics_exist(args):
        return
    if is_main_rank():
        msg = f"Scoring model: {args.model_name} (criterion={args.criterion})"
        if (
            args.criterion in {"fast_detect_gpt", "lastde_pp"}
            and args.reference_model is not None
        ):
            msg += f", reference model: {args.reference_model}"
        if args.criterion in {"binoculars", "dna_detectllm"}:
            msg += f", observer model: {args.reference_model}"
        print(msg)

    texts, labels = build_test_split(args)
    detector_cls = DETECTORS[args.criterion]
    detector = detector_cls(build_detector_config(args))

    scores = detector.batch_predict(
        texts,
        batch_size=args.predict_batch_size,
        show_progress=is_main_rank(),
    )

    if not is_main_rank():
        return

    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    metrics = compute_metrics(scores_arr, labels_arr)
    print_metrics(args.criterion, args.model_name, metrics)

    ref_model: str | None
    if args.criterion in {"fast_detect_gpt", "lastde_pp"}:
        ref_model = effective_reference_model(args.model_name, args.reference_model)
    elif args.criterion in {"binoculars", "dna_detectllm"}:
        ref_model = args.reference_model
    else:
        ref_model = None

    if args.metrics_json:
        payload: dict[str, Any] = {
            "criterion": args.criterion,
            "model": args.model,
            "scoring_model": args.model_name,
            "reference_model": ref_model,
            "dataset_path": str(args.dataset_path),
            "split": args.split,
            "normalize_mode": args.normalize_mode,
            "text_field": args.text_field,
            "label_mode": args.label_mode,
            "target_source": args.target_source,
            "include_dataset": list(args.include_dataset),
            "exclude_dataset": list(args.exclude_dataset),
            "include_source": list(args.include_source),
            "exclude_source": list(args.exclude_source),
            "metrics": metrics,
        }
        metrics_path = _metrics_output_path(args)
        assert metrics_path is not None
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote metrics to {metrics_path}")

    if args.scores_npz:
        scores_path = build_result_path(
            args.scores_npz,
            args.criterion,
            args.model_name,
            args.reference_model,
            suffix=".npz",
        )
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            scores_path,
            scores=scores_arr,
            labels=labels_arr,
        )
        print(f"Wrote scores to {scores_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run unsupervised statistical baselines (likelihood / logrank / "
            "entropy / lrr / fast_detect_gpt / binoculars / dna_detectllm / "
            "lastde / lastde_pp) "
            "on the test split "
            "of the merged dataset."
        )
    )
    parser.add_argument(
        "--criterion",
        choices=sorted(DETECTORS),
        default="likelihood",
        help="Which simple_detector score to evaluate.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Target source / model whose generations to detect (filtering and "
            "labeling)."
        ),
    )
    parser.add_argument(
        "--scoring-model",
        default=None,
        help=(
            "HF model id/path used for likelihood scoring. Defaults to --model "
            "when omitted (e.g. pass the base model for instruct-tuned targets)."
        ),
    )
    parser.add_argument(
        "--reference-model",
        default=None,
        help=(
            "Optional reference model for --criterion=fast_detect_gpt or "
            "lastde_pp (ref LM). Required observer LM for --criterion=binoculars "
            "or dna_detectllm. "
            "When omitted for fast_detect_gpt / lastde_pp, the scoring model is "
            "used as the reference."
        ),
    )

    parser.add_argument("--dataset-path", default="dataset/merged_dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--text-field",
        default="response",
        help="Column to score, or 'prompt_response' to join prompt and response.",
    )
    parser.add_argument(
        "--normalize-mode",
        choices=["raw", "whitespace"],
        default="raw",
    )
    parser.add_argument(
        "--label-mode",
        choices=["machine", "source"],
        default="machine",
        help="'machine': positive=non-human. 'source': positive=--target-source.",
    )
    parser.add_argument(
        "--target-source",
        default=None,
        help="Positive source/model for --label-mode=source.",
    )
    parser.add_argument("--include-dataset", nargs="+", default=[])
    parser.add_argument("--exclude-dataset", nargs="+", default=[])
    parser.add_argument("--include-source", nargs="+", default=[])
    parser.add_argument("--exclude-source", nargs="+", default=[])
    parser.add_argument(
        "--filter",
        nargs="+",
        action="extend",
        default=[],
        help="Extra exact-match filters as COLUMN=VALUE. Can be repeated.",
    )
    parser.add_argument("--filter-num-proc", type=int, default=None)

    parser.add_argument("--predict-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--device",
        default=_default_device(),
        help="Single-process device. Ignored under torchrun (FSDP2 auto-engages).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="auto, bf16/bfloat16, fp16/float16, fp32/float32.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code to HF loaders.",
    )

    parser.add_argument("--metrics-json", default=None)
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Recompute even when the output metrics JSON already exists.",
    )
    parser.add_argument(
        "--scores-npz",
        default=None,
        help="Optional path to save raw scores + labels as .npz.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    run(args if args is not None else parse_args())


if __name__ == "__main__":
    main()
