"""Lastde and Lastde++ — token probability sequence mining detectors.

Reference:
    Xu et al., "Training-free LLM-generated Text Detection by Mining
    Token Probability Sequences", ICLR 2025.
    https://github.com/TrustMedia-zju/Lastde_Detector

**Lastde** (sample-based) computes Multiscale Distribution Entropy (MDE)
of the per-token log-probability sequence.  MDE captures the regularity
of the probability landscape: machine text tends to have more regular
(lower-entropy) patterns, yielding a lower MDE.  The final score is
``mean(log_prob) / MDE``.

**Lastde++** (distribution-based) extends the idea with sampling from
the reference model's conditional distribution (like Fast-DetectGPT),
computing the Lastde score for both observed and sampled tokens and
reporting the normalised discrepancy ``(lastde_x - mu) / sigma``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from detectors.causal_lm_base import CausalLMDetectorBase
from detectors.fastdetect_gpt import _align_vocab, _get_samples


def _token_log_likelihood(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-token log-probs; shape ``[1, T, S]`` (``S=1`` for observed tokens)."""
    assert logits.shape[0] == 1 and labels.shape[0] == 1
    if labels.ndim == logits.ndim - 1:
        labels = labels.unsqueeze(-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=labels)


def _batched_histogram_de(values: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Distribution entropy from cosine-similarity columns; ``values`` is ``[N, S]``."""
    if values.numel() == 0:
        return torch.zeros(values.shape[1], device=values.device, dtype=values.dtype)

    n_steps, n_cols = values.shape
    scaled = (
        ((values.clamp(-1.0 + 1e-6, 1.0 - 1e-6) + 1.0) * 0.5 * n_bins)
        .long()
        .clamp(0, n_bins - 1)
    )
    col_offsets = torch.arange(n_cols, device=values.device) * n_bins
    flat_idx = (scaled + col_offsets).reshape(-1)
    hist_flat = torch.zeros(
        n_cols * n_bins,
        device=values.device,
        dtype=values.dtype,
    )
    hist_flat.scatter_add_(
        0,
        flat_idx,
        torch.ones(n_steps * n_cols, device=values.device, dtype=values.dtype),
    )
    probs = hist_flat.view(n_cols, n_bins)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-30)
    log_probs = torch.log(probs + 1e-30)
    norm = -1.0 / math.log(n_bins)
    return norm * (probs * log_probs).nansum(dim=-1)


def _compute_de(
    log_likelihood: torch.Tensor,
    embed_size: int,
    n_bins: int,
) -> torch.Tensor:
    """Diversity entropy per sample column; ``log_likelihood`` is ``[1, T, S]``."""
    seq_len = log_likelihood.shape[1]
    n_samples = log_likelihood.shape[2]
    if seq_len < embed_size + 2:
        return torch.zeros(n_samples, device=log_likelihood.device, dtype=log_likelihood.dtype)

    orbits = log_likelihood.unfold(1, embed_size, 1)
    cos_sim = F.cosine_similarity(orbits[:, :-1], orbits[:, 1:], dim=-1)[0]
    return _batched_histogram_de(cos_sim, n_bins)


def _compute_mde(
    log_likelihood: torch.Tensor,
    embed_size: int,
    n_bins: int,
    tau_prime: int,
) -> torch.Tensor:
    """Multiscale DE: std of DE across time scales ``1..tau_prime``; shape ``[S]``."""
    n_samples = log_likelihood.shape[2]
    de_values: List[torch.Tensor] = []
    for tau in range(1, tau_prime + 1):
        if tau > 1:
            windows = log_likelihood.unfold(1, tau, 1)
            seq = windows.mean(dim=-1)
        else:
            seq = log_likelihood
        if seq.shape[1] < embed_size + 2:
            break
        de_values.append(_compute_de(seq, embed_size, n_bins))
    if len(de_values) < 2:
        return torch.zeros(n_samples, device=log_likelihood.device, dtype=log_likelihood.dtype)
    return torch.stack(de_values, dim=0).std(dim=0)


def _lastde_scores(
    log_likelihood: torch.Tensor,
    embed_size: int,
    n_bins: int,
    tau_prime: int,
) -> torch.Tensor:
    """Lastde = mean(log_prob) / MDE for each sample column; shape ``[S]``."""
    mean_lp = log_likelihood.mean(dim=1).squeeze(0)
    mde = _compute_mde(log_likelihood, embed_size, n_bins, tau_prime).clamp(min=1e-10)
    return mean_lp / mde


def get_lastde_pp_discrepancy(
    logits_ref: torch.Tensor,
    logits_score: torch.Tensor,
    labels: torch.Tensor,
    *,
    embed_size: int = 4,
    epsilon_scale: float = 8.0,
    tau_prime: int = 15,
    nsamples: int = 100,
) -> float:
    """Normalised Lastde++ discrepancy for a single sequence (batch size 1)."""
    logits_ref, logits_score = _align_vocab(logits_ref, logits_score)
    samples = _get_samples(logits_ref, labels, nsamples)
    ll_x = _token_log_likelihood(logits_score, labels)
    ll_tilde = _token_log_likelihood(logits_score, samples)

    n_bins = max(2, int(epsilon_scale * ll_x.shape[1]))
    lastde_x = _lastde_scores(ll_x, embed_size, n_bins, tau_prime)
    lastde_tilde = _lastde_scores(ll_tilde, embed_size, n_bins, tau_prime)

    mu = lastde_tilde.mean()
    sigma = lastde_tilde.std()
    if float(sigma) < 1e-10:
        return 0.0
    return float((lastde_x[0] - mu) / sigma)


class LastdeDetector(CausalLMDetectorBase):
    """Lastde detector — MDE of token log-probability sequence (sample-based).

    Config keys:
        model_name, device, dtype, max_length, …: Same as
            :class:`CausalLMDetectorBase`.
        embed_size: Orbit embedding dimension (default ``3``).
        epsilon_scale: ``n_bins = epsilon_scale * T`` (default ``10``).
        tau_prime: Maximum time scale (default ``5``).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._embed_size = int(config.get("embed_size", 3))
        self._epsilon_scale = float(config.get("epsilon_scale", 10.0))
        self._tau_prime = int(config.get("tau_prime", 5))
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

        seq_lens = attention_mask.sum(dim=1).long()
        out: List[float] = []
        for b in range(logits.shape[0]):
            seq_len = int(seq_lens[b].item())
            if seq_len < 2:
                out.append(float("nan"))
                continue
            n_tokens = seq_len - 1
            row_logits = logits[b : b + 1, :n_tokens, :]
            labels = input_ids[b : b + 1, 1 : seq_len]
            ll = _token_log_likelihood(row_logits, labels)
            n_bins = max(2, int(self._epsilon_scale * n_tokens))
            score = _lastde_scores(ll, self._embed_size, n_bins, self._tau_prime)
            out.append(float(score[0].item()))
        return out


class LastdePPDetector(CausalLMDetectorBase):
    """Lastde++ detector — distribution-based Lastde with reference sampling.

    Mirrors the official ``lastde_doubleplus.py`` / Fast-DetectGPT setup:
    alternative tokens are drawn from the reference LM, log-probs are
    evaluated on the scoring LM, and the Lastde score discrepancy is
    normalised against sampled alternatives.

    Config keys:
        model_name: Scoring LM name or path.
        ref_model: Optional reference LM (defaults to scoring model).
        embed_size: Orbit dimension (default ``4``).
        epsilon_scale: ``n_bins = epsilon_scale * T`` (default ``8``).
        tau_prime: Max time scale (default ``15``).
        nsamples: Sampled alternatives per position (default ``100``).
        device, dtype, max_length, …: Same as :class:`CausalLMDetectorBase`.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._embed_size = int(config.get("embed_size", 4))
        self._epsilon_scale = float(config.get("epsilon_scale", 8.0))
        self._tau_prime = int(config.get("tau_prime", 15))
        self._nsamples = int(config.get("nsamples", 100))
        super().__init__(config)
        ref = config.get("ref_model")
        self._ref_model: torch.nn.Module | None = None
        if ref is not None:
            if isinstance(ref, str):
                self._ref_model = self._load_and_prepare_model(ref, config)
            elif isinstance(ref, torch.nn.Module):
                self._ref_model = self._prepare_model(ref, config)
            else:
                raise TypeError(
                    f"ref_model must be a model name or nn.Module, got {type(ref)!r}"
                )

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
        labels = batch["input_ids"][:, 1:].to(self._device)
        attn = batch["attention_mask"].to(self._device)
        logits_score = self._forward_logits(self._model, batch).float()
        if self._ref_model is None:
            logits_ref = logits_score
        else:
            logits_ref = self._forward_logits(self._ref_model, batch).float()
        return self._score_tokenized_batch(logits_score, logits_ref, labels, attn)

    def _score_tokenized_batch(
        self,
        logits_score: torch.Tensor,
        logits_ref: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> List[float]:
        seq_lens = attention_mask.sum(dim=1).long()
        out: List[float] = []
        for b in range(logits_score.shape[0]):
            seq_len = int(seq_lens[b].item())
            if seq_len < 2:
                out.append(float("nan"))
                continue
            n_tokens = seq_len - 1
            lr_s = logits_score[b : b + 1, :n_tokens, :]
            lr_r = logits_ref[b : b + 1, :n_tokens, :]
            lab = labels[b : b + 1, :n_tokens]
            out.append(
                get_lastde_pp_discrepancy(
                    lr_r,
                    lr_s,
                    lab,
                    embed_size=self._embed_size,
                    epsilon_scale=self._epsilon_scale,
                    tau_prime=self._tau_prime,
                    nsamples=self._nsamples,
                )
            )
        return out


__all__ = [
    "LastdeDetector",
    "LastdePPDetector",
    "get_lastde_pp_discrepancy",
    "_compute_de",
    "_compute_mde",
    "_lastde_scores",
]
