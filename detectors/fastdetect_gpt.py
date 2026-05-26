from typing import Any, Dict, List, Optional

import torch

from detectors.causal_lm_base import CausalLMDetectorBase


def _align_vocab(
    logits_ref: torch.Tensor, logits_score: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits_ref.size(-1) == logits_score.size(-1):
        return logits_ref, logits_score
    vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
    return logits_ref[..., :vocab_size], logits_score[..., :vocab_size]


def _get_samples(
    logits: torch.Tensor, labels: torch.Tensor, nsamples: int
) -> torch.Tensor:
    assert logits.shape[0] == 1 and labels.shape[0] == 1
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    return distrib.sample([nsamples]).permute(1, 2, 0)


def _get_likelihood(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    assert logits.shape[0] == 1 and labels.shape[0] == 1
    if labels.ndim == logits.ndim - 1:
        labels = labels.unsqueeze(-1)
    lprobs = torch.log_softmax(logits, dim=-1)
    return lprobs.gather(dim=-1, index=labels).mean(dim=1)


def get_sampling_discrepancy(
    logits_ref: torch.Tensor,
    logits_score: torch.Tensor,
    labels: torch.Tensor,
    nsamples: int = 10_000,
) -> float:
    logits_ref, logits_score = _align_vocab(logits_ref, logits_score)
    samples = _get_samples(logits_ref, labels, nsamples)
    log_likelihood_x = _get_likelihood(logits_score, labels)
    log_likelihood_x_tilde = _get_likelihood(logits_score, samples)
    mu_tilde = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1)
    discrepancy = (log_likelihood_x.squeeze(-1) - mu_tilde) / sigma_tilde
    return float(discrepancy.item())


def get_sampling_discrepancy_analytic(
    logits_ref: torch.Tensor, logits_score: torch.Tensor, labels: torch.Tensor
) -> float:
    logits_ref, logits_score = _align_vocab(logits_ref, logits_score)
    if labels.ndim == logits_score.ndim - 1:
        labels = labels.unsqueeze(-1)
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(
        mean_ref
    )
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(
        dim=-1
    ).sqrt()
    return float(discrepancy.mean().item())


class FastDetectGPT(CausalLMDetectorBase):

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        ref = config.get("ref_model")
        self._ref_model: Optional[torch.nn.Module] = None
        if ref is not None:
            if isinstance(ref, str):
                self._ref_model = self._load_and_prepare_model(ref, config)
            elif isinstance(ref, torch.nn.Module):
                self._ref_model = self._prepare_model(ref, config)
            else:
                raise TypeError(
                    f"ref_model must be a model name or nn.Module, got {type(ref)!r}"
                )
        self._analytic: bool = bool(config.get("analytic", True))
        self._nsamples: int = int(config.get("nsamples", 10_000))

    def _criterion(
        self, logits_ref: torch.Tensor, logits_score: torch.Tensor, labels: torch.Tensor
    ) -> float:
        if self._analytic:
            return get_sampling_discrepancy_analytic(logits_ref, logits_score, labels)
        return get_sampling_discrepancy(
            logits_ref, logits_score, labels, self._nsamples
        )

    def _score_batch_texts(self, texts: List[str]) -> List[float]:
        tokenized = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_token_type_ids=False,
        )
        labels = tokenized["input_ids"][:, 1:].to(self._device)
        attn = tokenized["attention_mask"].to(self._device)
        with torch.inference_mode():
            logits_score = self._forward_logits(self._model, tokenized)
            if self._ref_model is None:
                logits_ref = logits_score
            else:
                logits_ref = self._forward_logits(self._ref_model, tokenized)
        return self._score_tokenized_batch(logits_score, logits_ref, labels, attn)

    def _score_tokenized_batch(
        self,
        logits_score: torch.Tensor,
        logits_ref: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> List[float]:
        """Per-row scores; logits_* shape [B, T-1, V], labels [B, T-1], mask [B, T]."""
        seq_lens = attention_mask.sum(dim=1).long()
        out: List[float] = []
        for b in range(logits_score.shape[0]):
            L = int(seq_lens[b].item())
            if L < 2:
                out.append(float("nan"))
                continue
            n = L - 1
            lr_s = logits_score[b : b + 1, :n, :]
            lr_r = logits_ref[b : b + 1, :n, :]
            lab = labels[b : b + 1, :n]
            out.append(self._criterion(lr_r, lr_s, lab))
        return out
