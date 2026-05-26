from typing import Any, Callable, Dict, List

import torch

from detectors.causal_lm_base import CausalLMDetectorBase


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)


def _per_token_log_rank(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    chosen = logits.gather(dim=-1, index=labels.unsqueeze(-1))
    rank = (logits > chosen).sum(dim=-1) + 1
    return torch.log(rank.float())


def likelihood_score(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_p = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return _masked_mean(token_log_p, mask)


def log_rank_score(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    return _masked_mean(_per_token_log_rank(logits, labels), mask)


def entropy_score(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return _masked_mean(entropy, mask)


def lrr_score(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_p = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    log_rank = _per_token_log_rank(logits, labels)

    mask_f = mask.to(token_log_p.dtype)
    neg_log_p_sum = -(token_log_p * mask_f).sum(dim=-1)
    log_rank_sum = (log_rank * mask_f).sum(dim=-1).clamp(min=eps)
    return neg_log_p_sum / log_rank_sum


CriterionFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
CRITERIA: Dict[str, CriterionFn] = {
    "likelihood": likelihood_score,
    "logrank": log_rank_score,
    "entropy": entropy_score,
    "lrr": lrr_score,
}


class _LogitsDetector(CausalLMDetectorBase):

    criterion_name: str = ""

    def __init__(self, config: Dict[str, Any]) -> None:
        if self.criterion_name not in CRITERIA:
            raise ValueError(
                f"{type(self).__name__}.criterion_name must be one of "
                f"{sorted(CRITERIA)}; got {self.criterion_name!r}."
            )
        self._criterion: CriterionFn = CRITERIA[self.criterion_name]
        super().__init__(config)

    @torch.inference_mode()
    def _score_batch_texts(self, texts: List[str]) -> List[float]:
        batch = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_token_type_ids=False,
        )
        input_ids = batch["input_ids"].to(self._device)
        attention_mask = batch["attention_mask"].to(self._device)

        logits = self._forward_logits(self._model, batch).float()
        labels = input_ids[:, 1:]
        mask = (attention_mask[:, :-1] & attention_mask[:, 1:]).bool()

        scores = self._criterion(logits, labels, mask)
        return scores.detach().cpu().tolist()


class LikelihoodDetector(_LogitsDetector):
    criterion_name = "likelihood"


class LogRankDetector(_LogitsDetector):
    criterion_name = "logrank"


class EntropyDetector(_LogitsDetector):
    criterion_name = "entropy"


class LRRDetector(_LogitsDetector):
    criterion_name = "lrr"


__all__ = [
    "CRITERIA",
    "EntropyDetector",
    "LRRDetector",
    "LikelihoodDetector",
    "LogRankDetector",
    "entropy_score",
    "likelihood_score",
    "log_rank_score",
    "lrr_score",
]
