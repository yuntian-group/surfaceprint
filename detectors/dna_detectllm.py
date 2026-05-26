"""DNA-DetectLLM — mutation-repair paradigm for zero-shot detection.

Reference:
    Zhu et al., "DNA-DetectLLM: Unveiling AI-Generated Text via a
    DNA-Inspired Mutation-Repair Paradigm", NeurIPS 2025.
    https://github.com/Xiaoweizhu57/DNA-DetectLLM

The method constructs an "ideal AI sequence" by greedily selecting the most
probable token at each position under a performer model, then measures repair
effort via the sum of standard and ideal-sequence perplexities, normalized by
cross-perplexity between an observer and performer model:

    score = (ppl_std + ppl_max) / (2 * x_ppl)

Low score → AI-generated; high score → human-written.  We negate so that
*higher* score → more likely AI, matching the other statistical detectors
in this repo.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from detectors.binoculars import _batched_cross_perplexity
from detectors.causal_lm_base import CausalLMDetectorBase


def _batched_sum_perplexity(
    performer_logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Performer ppl_std + ppl_max; shape ``[B]`` (official ``sum_perplexity``)."""
    shifted_logits = performer_logits[:, :-1, :].contiguous()
    shifted_labels = input_ids[:, 1:].contiguous()
    shifted_mask = attention_mask[:, 1:].contiguous().to(shifted_logits.dtype)
    labels_max = shifted_logits.argmax(dim=-1)

    ce_std = F.cross_entropy(
        shifted_logits.transpose(1, 2),
        shifted_labels,
        reduction="none",
    )
    ce_max = F.cross_entropy(
        shifted_logits.transpose(1, 2),
        labels_max,
        reduction="none",
    )
    attn_sum = shifted_mask.sum(dim=-1).clamp(min=1.0)
    ppl_std = (ce_std * shifted_mask).sum(dim=-1) / attn_sum
    ppl_max = (ce_max * shifted_mask).sum(dim=-1) / attn_sum
    return ppl_std + ppl_max


class DNADetectLLMDetector(CausalLMDetectorBase):
    """DNA-DetectLLM detector (sum-perplexity / cross-perplexity ratio).

    Config keys:
        model_name: Observer (base) LM name or path.
        performer_model: Performer (instruction-tuned) LM name, path, or
            ``nn.Module``.
        device, dtype, max_length, …: Same as :class:`CausalLMDetectorBase`.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        performer = config.get("performer_model")
        if performer is None:
            raise ValueError("DNADetectLLMDetector requires config['performer_model'].")
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

        ppl = _batched_sum_perplexity(performer_logits, input_ids, attention_mask)
        x_ppl = _batched_cross_perplexity(
            observer_logits,
            performer_logits,
            input_ids,
            pad_token_id=self._tokenizer.pad_token_id,
        )

        # Low DNA-DetectLLM score → likely AI; negate for higher = more AI.
        dna_raw = ppl / (2.0 * x_ppl.clamp(min=1e-8))
        scores = -dna_raw

        seq_lens = attention_mask.sum(dim=1)
        scores = torch.where(
            seq_lens >= 2, scores, torch.full_like(scores, float("nan"))
        )
        return scores.detach().cpu().tolist()


__all__ = [
    "DNADetectLLMDetector",
    "_batched_sum_perplexity",
]
