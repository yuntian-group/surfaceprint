import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from huggingface_hub import snapshot_download
except (
    ImportError
):  # pragma: no cover - transformers normally depends on huggingface_hub.
    snapshot_download = None

try:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        MixedPrecisionPolicy,
        fully_shard,
    )
except ImportError:  # pragma: no cover - depends on installed torch version.
    CPUOffloadPolicy = MixedPrecisionPolicy = fully_shard = init_device_mesh = None

from detectors.base_detector import BaseDetector


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _config_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _use_fsdp2(config: Dict[str, Any]) -> bool:
    value = config.get("fsdp2", config.get("use_fsdp2", "auto"))
    if isinstance(value, str) and value.strip().lower() == "auto":
        return _env_int("WORLD_SIZE", 1) > 1
    return _config_bool(value)


def _parse_dtype(value: Any) -> Any:
    if value is None or isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "auto":
            return "auto"
        aliases = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "half": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
            "float": torch.float32,
        }
        if lowered in aliases:
            return aliases[lowered]
    raise ValueError(f"Unsupported dtype: {value!r}")


def _from_pretrained(model_name: str, model_kwargs: Dict[str, Any]) -> torch.nn.Module:
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except TypeError:
        if "dtype" not in model_kwargs:
            raise
        legacy_kwargs = dict(model_kwargs)
        legacy_kwargs["torch_dtype"] = legacy_kwargs.pop("dtype")
        return AutoModelForCausalLM.from_pretrained(model_name, **legacy_kwargs)


def _is_local_model_path(model_name: str) -> bool:
    path = Path(model_name)
    return path.exists() or path.is_absolute()


def _snapshot_download_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("cache_dir", "revision", "token")
    kwargs = {key: config[key] for key in keys if key in config}
    kwargs["allow_patterns"] = config.get(
        "download_allow_patterns",
        [
            "*.safetensors",
            "*.bin",
            "*.safetensors.index.json",
            "*.bin.index.json",
        ],
    )
    return kwargs


def _hf_metadata_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("cache_dir", "revision", "token", "trust_remote_code")
    return {key: config[key] for key in keys if key in config}


def _module_by_path(module: torch.nn.Module, path: str) -> Optional[torch.nn.Module]:
    current = module
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def _unique_modules(modules: Iterable[torch.nn.Module]) -> List[torch.nn.Module]:
    seen: set[int] = set()
    out: List[torch.nn.Module] = []
    for module in modules:
        ident = id(module)
        if ident not in seen:
            seen.add(ident)
            out.append(module)
    return out


def _transformer_blocks(model: torch.nn.Module) -> List[torch.nn.Module]:
    common_paths = (
        "model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "backbone.layers",
    )
    for path in common_paths:
        module = _module_by_path(model, path)
        if isinstance(module, (torch.nn.ModuleList, list, tuple)) and module:
            return list(module)

    no_split = set(getattr(model, "_no_split_modules", []) or [])
    config = getattr(model, "config", None)
    no_split.update(getattr(config, "_no_split_modules", []) or [])
    if not no_split:
        return []
    return _unique_modules(
        module
        for module in model.modules()
        if module is not model and module.__class__.__name__ in no_split
    )


def _make_parameters_contiguous(model: torch.nn.Module) -> int:
    changed = 0
    with torch.no_grad():
        for param in model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
                changed += 1
    return changed


class CausalLMDetectorBase(BaseDetector):
    def __init__(self, config: Dict[str, Any]) -> None:
        model_name = config["model_name"]
        self._fsdp2 = _use_fsdp2(config)
        self._rank = _env_int("RANK", 0)
        self._local_rank = _env_int("LOCAL_RANK", 0)
        self._world_size = _env_int("WORLD_SIZE", 1)
        if self._fsdp2:
            if not torch.cuda.is_available():
                raise RuntimeError("FSDP2 requires CUDA.")
            torch.cuda.set_device(self._local_rank)
            self._device = torch.device("cuda", self._local_rank)
        else:
            self._device = torch.device(config.get("device", "cpu"))
        self._setup_distributed(config)

        tokenizer_kwargs = self._tokenizer_kwargs(config, model_name)
        self._model = self._load_and_prepare_model(model_name, config)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._max_length = int(
            config.get(
                "max_length",
                getattr(self._model.config, "max_position_embeddings", 1024),
            )
        )
        self._dummy_text = self._tokenizer.eos_token or self._tokenizer.pad_token or "."

    @property
    def is_main_process(self) -> bool:
        return self._rank == 0

    def _setup_distributed(self, config: Dict[str, Any]) -> None:
        if not self._fsdp2:
            return
        if (
            fully_shard is None
            or MixedPrecisionPolicy is None
            or init_device_mesh is None
        ):
            raise RuntimeError("FSDP2 is not available in this PyTorch installation.")
        if self._world_size < 2:
            raise RuntimeError(
                "FSDP2 needs torchrun/torch.distributed with WORLD_SIZE > 1."
            )
        if dist.is_initialized():
            return
        backend = str(config.get("dist_backend", "nccl"))
        dist.init_process_group(backend=backend)

    def _download_model_once(self, model_name: str, config: Dict[str, Any]) -> None:
        if _is_local_model_path(model_name):
            return
        if not _config_bool(config.get("download_model_once"), default=True):
            return
        if snapshot_download is None:
            raise RuntimeError(
                "download_model_once requires huggingface_hub.snapshot_download."
            )

        if self.is_main_process:
            metadata_kwargs = _hf_metadata_kwargs(config)
            AutoConfig.from_pretrained(model_name, **metadata_kwargs)
            AutoTokenizer.from_pretrained(model_name, **metadata_kwargs)
            snapshot_download(model_name, **_snapshot_download_kwargs(config))
        dist.barrier()

    def _model_kwargs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        model_kwargs = dict(config.get("model_kwargs", {}))
        default_dtype = "auto" if self._fsdp2 else None
        dtype = _parse_dtype(
            config.get("dtype", config.get("torch_dtype", default_dtype))
        )
        if dtype is not None:
            model_kwargs.setdefault("dtype", dtype)
        if self._fsdp2:
            model_kwargs.setdefault("low_cpu_mem_usage", True)
        return model_kwargs

    def _tokenizer_kwargs(
        self, config: Dict[str, Any], model_name: str
    ) -> Dict[str, Any]:
        tokenizer_kwargs = dict(config.get("tokenizer_kwargs", {}))
        if self._fsdp2 and not _is_local_model_path(model_name):
            tokenizer_kwargs["local_files_only"] = True
        return tokenizer_kwargs

    def _load_and_prepare_model(
        self, model_name: str, config: Dict[str, Any]
    ) -> torch.nn.Module:
        model_kwargs = self._model_kwargs(config)
        if self._fsdp2:
            self._download_model_once(model_name, config)
            if not _is_local_model_path(model_name):
                model_kwargs["local_files_only"] = True
        return self._prepare_model(_from_pretrained(model_name, model_kwargs), config)

    def _fsdp_kwargs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "mesh": init_device_mesh("cuda", (self._world_size,)),
        }
        mp_dtype = _parse_dtype(
            config.get("fsdp_mp_dtype", config.get("mixed_precision_dtype"))
        )
        if isinstance(mp_dtype, torch.dtype):
            kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=mp_dtype,
                reduce_dtype=mp_dtype,
            )
        if _config_bool(config.get("cpu_offload"), default=False):
            kwargs["offload_policy"] = CPUOffloadPolicy()
        return kwargs

    def _prepare_model(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
    ) -> torch.nn.Module:
        if not self._fsdp2:
            model = model.to(self._device)
            model.eval()
            return model

        if _config_bool(config.get("fsdp_make_params_contiguous"), default=True):
            _make_parameters_contiguous(model)
        fsdp_kwargs = self._fsdp_kwargs(config)
        blocks = _transformer_blocks(model)
        if not blocks:
            raise RuntimeError(
                "Could not find transformer blocks to shard with FSDP2. "
                "Set a supported architecture or add its block path to "
                "_transformer_blocks()."
            )
        for block in blocks:
            fully_shard(block, **fsdp_kwargs)
        fully_shard(model, **fsdp_kwargs)
        model = model.to(self._device)
        model.eval()
        return model

    def _forward_logits(
        self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        batch = {k: v.to(self._device) for k, v in batch.items() if torch.is_tensor(v)}
        return model(**batch).logits[:, :-1]

    def _score_batch_texts(self, texts: List[str]) -> List[float]:
        raise NotImplementedError

    def predict(self, x: str) -> float:
        return self._score_batch_texts([x])[0]

    def batch_predict(
        self,
        x: List[str],
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> List[float]:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self._fsdp2:
            return self._distributed_batch_predict(x, batch_size, show_progress)

        n = len(x)
        if n == 0:
            return []
        chunks = range(0, n, batch_size)
        iterator: Iterable[int] = chunks
        if show_progress:
            iterator = tqdm.tqdm(
                chunks,
                desc=type(self).__name__,
                total=(n + batch_size - 1) // batch_size,
                unit="batch",
            )

        results: List[float] = []
        for start in iterator:
            results.extend(self._score_batch_texts(x[start : start + batch_size]))
        return results

    def _distributed_batch_predict(
        self,
        x: List[str],
        batch_size: int,
        show_progress: bool,
    ) -> List[float]:
        indexed_texts: List[Tuple[int, str]] = list(enumerate(x))[
            self._rank :: self._world_size
        ]
        local_batches = (
            math.ceil(len(indexed_texts) / batch_size) if indexed_texts else 0
        )
        count_device = (
            self._device if self._device.type == "cuda" else torch.device("cpu")
        )
        max_batches_tensor = torch.tensor(local_batches, device=count_device)
        dist.all_reduce(max_batches_tensor, op=dist.ReduceOp.MAX)
        max_batches = int(max_batches_tensor.item())

        local_results: List[Tuple[int, float]] = []
        iterator: Iterable[int] = range(max_batches)
        if show_progress and self.is_main_process:
            iterator = tqdm.tqdm(
                iterator,
                desc=type(self).__name__,
                total=max_batches,
                unit="batch",
            )

        for step in iterator:
            start = step * batch_size
            batch = indexed_texts[start : start + batch_size]
            if not batch:
                self._score_batch_texts([self._dummy_text])
                continue
            indices, texts = zip(*batch)
            scores = self._score_batch_texts(list(texts))
            local_results.extend(zip(indices, scores))

        gathered: List[List[Tuple[int, float]]] = [
            list() for _ in range(self._world_size)
        ]
        dist.all_gather_object(gathered, local_results)
        if not self.is_main_process:
            return []

        by_index = {
            index: score for rank_results in gathered for index, score in rank_results
        }
        return [by_index[i] for i in range(len(x))]


__all__ = [
    "CausalLMDetectorBase",
]
