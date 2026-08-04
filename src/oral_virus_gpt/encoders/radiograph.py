from __future__ import annotations

from torch import Tensor, nn

from oral_virus_gpt.encoders.photo import PhotoEncoder, freeze_module


class RadiographAdapter(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 64,
        kernel_size: int = 3,
        use_batchnorm: bool = True,
        residual: bool = True,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        padding = kernel_size // 2
        self.residual = residual
        self.conv1 = nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=not use_batchnorm,
        )
        self.norm = nn.BatchNorm2d(mid_channels) if use_batchnorm else nn.Identity()
        if activation == "gelu":
            self.activation: nn.Module = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"unknown activation {activation!r}")
        self.conv2 = nn.Conv2d(
            mid_channels, in_channels, kernel_size=kernel_size, padding=padding, bias=True
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError("expected 4D input [B, C, H, W]")
        h = self.conv1(x)
        h = self.norm(h)
        h = self.activation(h)
        h = self.conv2(h)
        return x + h if self.residual else h


class RadiographEncoder(nn.Module):
    def __init__(
        self,
        adapter: RadiographAdapter,
        photo_encoder: PhotoEncoder,
        share_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.photo_encoder = (
            photo_encoder if share_backbone else _clone_frozen_encoder(photo_encoder)
        )
        if not share_backbone:
            freeze_module(self.photo_encoder.vision_model)

    def forward(self, pixel_values: Tensor) -> Tensor:
        adapted = self.adapter(pixel_values)
        return self.photo_encoder(adapted)


def _clone_frozen_encoder(encoder: PhotoEncoder) -> PhotoEncoder:
    cloned = PhotoEncoder(
        vision_model=encoder.vision_model,
        hidden_dim=encoder.hidden_dim,
        output_dim=encoder.output_dim,
        freeze=True,
    )
    return cloned
