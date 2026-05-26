"""Binoculars detector — PPL / cross-perplexity ratio from two LMs.

Reference:
    Hans et al., "Spotting LLMs With Binoculars: Zero-Shot
    Detection of Machine-Generated Text", ICML 2024.

The detector uses an *observer* and a *performer* model.  The Binoculars
score is:

    B(s) = PPL_performer(s) / X-PPL_{observer, performer}(s)

where PPL is the standard perplexity of the *performer* and X-PPL is
the cross-perplexity: the cross-entropy between the observer's
softmax distribution and the performer's log-softmax, averaged over
token positions.

Low Binoculars score → likely AI.  We negate the log-ratio so that
*higher* score → more likely AI, matching the other statistical detectors
in this repo.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from detectors.causal_lm_base import CausalLMDetectorBase
from detectors.fastdetect_gpt import _align_vocab


def _batched_performer_log_ppl(
    performer_logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean token CE (log-PPL) of the performer; shape [B]."""
    shifted_logits = performer_logits[:, :-1, :].contiguous()
    shifted_labels = input_ids[:, 1:].contiguous()
    shifted_mask = attention_mask[:, 1:].contiguous().to(shifted_logits.dtype)
    ce = F.cross_entropy(
        shifted_logits.transpose(1, 2),
        shifted_labels,
        reduction="none",
    )
    return (ce * shifted_mask).sum(dim=-1) / shifted_mask.sum(dim=-1).clamp(min=1.0)


def _batched_cross_perplexity(
    observer_logits: torch.Tensor,
    performer_logits: torch.Tensor,
    input_ids: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """Cross-perplexity; shape [B]. Uses full-sequence logits (official Binoculars)."""
    observer_logits, performer_logits = _align_vocab(observer_logits, performer_logits)
    batch_size, total_tokens, vocab_size = observer_logits.shape
    observer_probs = F.softmax(observer_logits, dim=-1).reshape(-1, vocab_size)
    performer_scores = performer_logits.reshape(-1, vocab_size)
    ce = F.cross_entropy(performer_scores, observer_probs, reduction="none")
    ce = ce.view(batch_size, total_tokens)
    padding_mask = (input_ids != pad_token_id).to(ce.dtype)
    return (ce * padding_mask).sum(dim=-1) / padding_mask.sum(dim=-1).clamp(min=1.0)


class BinocularsDetector(CausalLMDetectorBase):
    """Binoculars detector (PPL / cross-perplexity).

    Config keys:
        model_name: Observer LM name or path.
        performer_model: Performer LM name, path, or ``nn.Module``.
        device, dtype, max_length, …: Same as :class:`CausalLMDetectorBase`.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        performer = config.get("performer_model")
        if performer is None:
            raise ValueError("BinocularsDetector requires config['performer_model'].")
        super().__init__(config)
        if isinstance(performer, str):
            self._performer_model = self._load_and_prepare_model(performer, config)
        elif isinstance(performer, torch.nn.Module):
            self._performer_model = self._prepare_model(performer, config)
        else:
            raise TypeError(
                f"performer_model must be a model name or nn.Module, got {type(performer)!r}"
            )

    def _forward_full_logits(
        self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        batch = {
            key: value.to(self._device)
            for key, value in batch.items()
            if torch.is_tensor(value)
        }
        return model(**batch).logits

    @torch.inference_mode()
    def _score_batch_texts(self, texts: List[str]) -> List[float]:
        tokenized = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_token_type_ids=False,
        )
        input_ids = tokenized["input_ids"].to(self._device)
        attention_mask = tokenized["attention_mask"].to(self._device)

        observer_logits = self._forward_full_logits(self._model, tokenized)
        performer_logits = self._forward_full_logits(self._performer_model, tokenized)

        log_ppl = _batched_performer_log_ppl(
            performer_logits, input_ids, attention_mask
        )
        x_ppl = _batched_cross_perplexity(
            observer_logits,
            performer_logits,
            input_ids,
            pad_token_id=self._tokenizer.pad_token_id,
        )

        # Low Binoculars score → likely AI; negate for higher = more AI.
        binoculars_raw = log_ppl / x_ppl.clamp(min=1e-8)
        scores = -binoculars_raw

        seq_lens = attention_mask.sum(dim=1)
        scores = torch.where(seq_lens >= 2, scores, torch.full_like(scores, float("nan")))
        return scores.detach().cpu().tolist()


__all__ = [
    "BinocularsDetector",
    "_batched_cross_perplexity",
    "_batched_performer_log_ppl",
]
