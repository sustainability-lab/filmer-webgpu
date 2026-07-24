"""Exact FiLMeR v1.0 Variant B architecture used by the published checkpoint.

Transcribed from the read-only GMD submission package supplied by Balbir
Prasad on 2026-07-24. The checkpoint loads with zero missing and zero
unexpected keys.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


def conv1x1(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class ProjectionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [16, 64, 128, 256]
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(dims, dims[1:]):
            layers.extend(
                [
                    nn.Linear(input_dim, output_dim),
                    nn.LayerNorm(output_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                ]
            )
        layers.extend([nn.Linear(256, 256), nn.GELU()])
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class FiLMLayer2D(nn.Module):
    def __init__(self, feature_channels: int):
        super().__init__()
        self.gamma_fc = nn.Linear(256, feature_channels)
        self.beta_fc = nn.Linear(256, feature_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_fc(cond).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_fc(cond).unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class AttentionGate(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int, inner_channels: int):
        super().__init__()
        self.W_g = nn.Sequential(
            conv1x1(gate_channels, inner_channels),
            nn.BatchNorm2d(inner_channels),
        )
        self.W_x = nn.Sequential(
            conv1x1(skip_channels, inner_channels),
            nn.BatchNorm2d(inner_channels),
        )
        self.psi = nn.Sequential(
            conv1x1(inner_channels, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        gate = F.interpolate(
            gate, size=skip.shape[2:], mode="bilinear", align_corners=False
        )
        attention = self.psi(self.relu(self.W_g(gate) + self.W_x(skip)))
        return skip * attention


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.use_film = False

    def forward(
        self, x: torch.Tensor, _condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int
    ):
        super().__init__()
        self.use_attention = True
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2
        )
        self.attention = AttentionGate(
            in_channels, skip_channels, out_channels
        )
        self.use_film = True
        self.film = FiLMLayer2D(in_channels + skip_channels)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)
        self.use_residual = in_channels == out_channels
        if self.use_residual:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        identity = x if self.use_residual else None
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
        skip = self.attention(x, skip)
        x = torch.cat([x, skip], dim=1)
        x = self.film(x, condition)
        x = self.conv(x)
        if self.use_residual and identity is not None:
            identity = F.interpolate(
                identity, size=x.shape[2:], mode="bilinear", align_corners=False
            )
            x = x + self.residual_conv(identity)
        return x


class StateHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 5, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class PrecipHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.occ = nn.Conv2d(32, 1, 1)
        self.inten = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Softplus())

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(x)
        return self.occ(features), self.inten(features)


class ModalityEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.surf_conv = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.upper_conv = nn.Sequential(
            nn.Conv2d(24, 32, 3, padding=1, groups=4),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, surface: torch.Tensor, upper: torch.Tensor
    ) -> torch.Tensor:
        return self.fuse(
            torch.cat([self.surf_conv(surface), self.upper_conv(upper)], dim=1)
        )


class FiLMeRVariantB(nn.Module):
    """Variant B, fixed 70x127x137 input and 6x99x99 output."""

    def __init__(self):
        super().__init__()
        self.ablation_variant = "B"
        self.static_enc = nn.Sequential(
            nn.Conv2d(30, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.gfs_enc = ModalityEncoder()
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.proj_mlp = ProjectionMLP()
        self.encoders = nn.ModuleList(
            [
                EncoderBlock(64, 64),
                EncoderBlock(64, 128),
                EncoderBlock(128, 256),
                EncoderBlock(256, 512),
            ]
        )
        self.bottleneck = DoubleConv(512, 1024)
        self.decoders = nn.ModuleList(
            [
                DecoderBlock(1024, 512, 512),
                DecoderBlock(512, 256, 256),
                DecoderBlock(256, 128, 128),
                DecoderBlock(128, 64, 64),
            ]
        )
        self.state_head = StateHead()
        self.precip_head = PrecipHead()

    def forward(
        self, input_data: torch.Tensor, projection: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gfs = input_data[:, :40]
        static = input_data[:, 40:70]
        static_features = self.static_enc(static)

        surface_tm1, upper_tm1 = gfs[:, 0:8], gfs[:, 8:20]
        surface_t, upper_t = gfs[:, 20:28], gfs[:, 28:40]
        surface = torch.cat([surface_tm1, surface_t], dim=1)
        upper = torch.cat(
            [
                upper_tm1[:, 0:3],
                upper_t[:, 0:3],
                upper_tm1[:, 3:6],
                upper_t[:, 3:6],
                upper_tm1[:, 6:9],
                upper_t[:, 6:9],
                upper_tm1[:, 9:12],
                upper_t[:, 9:12],
            ],
            dim=1,
        )
        x = self.fusion(
            torch.cat([self.gfs_enc(surface, upper), static_features], dim=1)
        )
        condition = self.proj_mlp(projection)

        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            x, skip = encoder(x, condition)
            skips.append(skip)
        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip, condition)
        if x.shape[2:] != (99, 99):
            x = F.interpolate(
                x, size=(99, 99), mode="bilinear", align_corners=False
            )
        occurrence, intensity = self.precip_head(x)
        return self.state_head(x), occurrence, intensity
