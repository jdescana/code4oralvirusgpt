from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from oral_virus_gpt.fusion.alternatives import (
    ConcatenationFusion,
    StackedTransformerFusion,
    WeightedAverageFusion,
)
from oral_virus_gpt.fusion.gated_fusion import GatedResidual, SigmoidGate
from oral_virus_gpt.fusion.missing_modality import NullEmbedding, mask_modality
from oral_virus_gpt.fusion.semantic_xattn import SemanticCrossAttention
from oral_virus_gpt.fusion.token_xattn import (
    KVProjection,
    RadiographCrossAttention,
    TokenCrossAttention,
)


@dataclass(slots=True)
class HGCFConfig:
    hidden_dim: int = 4096
    num_heads: int = 16
    num_concept_slots: int = 16
    num_classes: int = 120
    use_token: bool = True
    use_semantic: bool = True
    use_gating: bool = True
    use_uncertainty_gating: bool = True
    style: str = "hgcf"
    match_param_budget: bool = False
    gating_init: str = "zero"
    residual_alpha_init: float = 0.0


@dataclass(slots=True)
class HGCFOutput:
    fused: Tensor
    logits: Tensor
    gates: Tensor | None
    semantic_photo: Tensor | None
    semantic_radio: Tensor | None
    semantic_text: Tensor | None


class HGCF(nn.Module):
    def __init__(self, config: HGCFConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_dim
        if config.style == "hgcf":
            self._init_hgcf(d)
        elif config.style == "concat":
            self.alt: nn.Module = ConcatenationFusion(d)
        elif config.style == "weighted":
            self.alt = WeightedAverageFusion(d)
        elif config.style == "stacked_xformer":
            num_layers = 2 if not config.match_param_budget else 4
            self.alt = StackedTransformerFusion(
                d, num_layers=num_layers, num_heads=config.num_heads
            )
        else:
            raise ValueError(f"unknown fusion style {config.style!r}")
        self.classifier = nn.Linear(d, config.num_classes)
        self.null_photo = NullEmbedding(config.num_concept_slots, d)
        self.null_radio = NullEmbedding(config.num_concept_slots, d)
        self.null_text = NullEmbedding(config.num_concept_slots, d)
        self.gated_residual = GatedResidual(d, d, alpha_init=config.residual_alpha_init)

    def _init_hgcf(self, d: int) -> None:
        kv = KVProjection(d)
        if self.config.use_token:
            self.photo_xattn: nn.Module = TokenCrossAttention(d, self.config.num_heads, kv_proj=kv)
            self.radio_xattn: nn.Module = RadiographCrossAttention(
                d, self.config.num_heads, kv_proj=kv
            )
        else:
            self.photo_xattn = nn.Identity()
            self.radio_xattn = nn.Identity()
        if self.config.use_semantic:
            self.semantic_photo: nn.Module = SemanticCrossAttention(
                d, self.config.num_concept_slots, self.config.num_heads
            )
            self.semantic_radio: nn.Module = SemanticCrossAttention(
                d, self.config.num_concept_slots, self.config.num_heads
            )
            self.semantic_text: nn.Module = SemanticCrossAttention(
                d, self.config.num_concept_slots, self.config.num_heads
            )
        else:
            self.semantic_photo = nn.Identity()
            self.semantic_radio = nn.Identity()
            self.semantic_text = nn.Identity()
        if self.config.use_gating:
            self.gate: nn.Module = SigmoidGate(
                self.config.num_concept_slots, num_modalities=3, init=self.config.gating_init
            )
        else:
            self.gate = nn.Identity()

    def forward(
        self,
        photo_tokens: Tensor,
        radio_tokens: Tensor,
        text_tokens: Tensor,
        photo_present: Tensor | None = None,
        radio_present: Tensor | None = None,
    ) -> HGCFOutput:
        if self.config.style != "hgcf":
            return self._forward_alt(photo_tokens, radio_tokens, text_tokens)
        b = photo_tokens.shape[0]
        device = photo_tokens.device
        if photo_present is None:
            photo_present = torch.ones(b, device=device)
        if radio_present is None:
            radio_present = torch.ones(b, device=device)
        if self.config.use_token:
            photo_aligned = self.photo_xattn(photo_tokens, text_tokens)
            radio_aligned = self.radio_xattn(radio_tokens, text_tokens)
        else:
            photo_aligned = photo_tokens
            radio_aligned = radio_tokens
        if self.config.use_semantic:
            sem_photo = self.semantic_photo(photo_aligned)
            sem_radio = self.semantic_radio(radio_aligned)
            sem_text = self.semantic_text(text_tokens)
        else:
            sem_photo = photo_aligned[:, : self.config.num_concept_slots]
            sem_radio = radio_aligned[:, : self.config.num_concept_slots]
            sem_text = text_tokens[:, : self.config.num_concept_slots]
        sem_photo, _ = mask_modality(sem_photo, photo_present, self.null_photo)
        sem_radio, _ = mask_modality(sem_radio, radio_present, self.null_radio)
        if self.config.use_gating:
            fused, gates = self.gate([sem_photo, sem_radio, sem_text])
        else:
            fused = (sem_photo + sem_radio + sem_text) / 3.0
            gates = None
        pooled = fused.mean(dim=1)
        logits = self.classifier(pooled)
        return HGCFOutput(
            fused=fused,
            logits=logits,
            gates=gates,
            semantic_photo=sem_photo,
            semantic_radio=sem_radio,
            semantic_text=sem_text,
        )

    def _forward_alt(self, photo: Tensor, radio: Tensor, text: Tensor) -> HGCFOutput:
        fused = self.alt([photo, radio, text])
        pooled = fused.mean(dim=1)
        logits = self.classifier(pooled)
        return HGCFOutput(
            fused=fused,
            logits=logits,
            gates=None,
            semantic_photo=None,
            semantic_radio=None,
            semantic_text=None,
        )

    def gated_residual_into(self, hidden_states: Tensor, fused: Tensor) -> Tensor:
        return self.gated_residual(hidden_states, fused)
