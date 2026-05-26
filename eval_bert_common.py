from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.inference_mode()
def predict_dataset(
    model: torch.nn.Module,
    dataset,
    collator,
    *,
    batch_size: int,
    num_workers: int,
    desc: str = "Evaluating",
) -> tuple[np.ndarray, np.ndarray]:
    device = resolve_device()
    model.to(device)
    n_examples = len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    n_batches = len(loader)
    print(
        f"{desc}: {n_examples} examples, "
        f"batch_size={batch_size}, batches={n_batches}, device={device}"
    )

    logits_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    seen = 0
    pbar = tqdm(loader, total=n_batches, desc=desc, unit="batch", dynamic_ncols=True)
    for batch in pbar:
        labels = batch.pop("labels")
        batch = {
            key: value.to(device, non_blocking=True) for key, value in batch.items()
        }
        logits = model(**batch).logits
        batch_size_actual = len(labels)
        logits_chunks.append(logits.detach().cpu())
        label_chunks.append(labels.detach().cpu())
        seen += batch_size_actual
        pbar.set_postfix(examples=f"{seen}/{n_examples}", refresh=False)
    pbar.close()

    logits = torch.cat(logits_chunks, dim=0)
    labels = torch.cat(label_chunks, dim=0)
    if len(labels) != n_examples:
        raise ValueError(
            f"Prediction count {len(labels)} != dataset size {n_examples}."
        )
    print(f"{desc} complete: scored {len(labels)} examples.")
    return logits.float().numpy(), labels.numpy()
