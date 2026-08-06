"""CFP-only Task 1 segmentation models."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.fuse = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(
            inputs,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.fuse(torch.cat((self.reduce(inputs), skip), dim=1))


class Task1UNet(nn.Module):
    """Shared encoder/decoder with artery, vessel-union, and vein logits."""

    output_channels = ("artery", "vessel", "vein")

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        if base_channels < 8 or base_channels % 8:
            raise ValueError("base_channels must be a multiple of 8 and at least 8")
        widths = [base_channels * multiplier for multiplier in (1, 2, 4, 8, 16)]
        self.encoder1 = ConvBlock(3, widths[0])
        self.encoder2 = ConvBlock(widths[0], widths[1])
        self.encoder3 = ConvBlock(widths[1], widths[2])
        self.encoder4 = ConvBlock(widths[2], widths[3])
        self.bottleneck = ConvBlock(widths[3], widths[4])
        self.pool = nn.MaxPool2d(2)
        self.decoder4 = UpBlock(widths[4], widths[3], widths[3])
        self.decoder3 = UpBlock(widths[3], widths[2], widths[2])
        self.decoder2 = UpBlock(widths[2], widths[1], widths[1])
        self.decoder1 = UpBlock(widths[1], widths[0], widths[0])
        self.head = nn.Conv2d(widths[0], 3, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        encoder4 = self.encoder4(self.pool(encoder3))
        bottleneck = self.bottleneck(self.pool(encoder4))
        decoded = self.decoder4(bottleneck, encoder4)
        decoded = self.decoder3(decoded, encoder3)
        decoded = self.decoder2(decoded, encoder2)
        decoded = self.decoder1(decoded, encoder1)
        return self.head(decoded)


class ConvNeXtTinyTask1UNet(nn.Module):
    """ImageNet-pretrained ConvNeXt-Tiny encoder with a compact U-Net decoder."""

    output_channels = ("artery", "vessel", "vein")
    weight_name = "ConvNeXt_Tiny_Weights.IMAGENET1K_V1"
    weight_url = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.url

    def __init__(self, *, pretrained: bool = False) -> None:
        super().__init__()
        weights = (
            ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        )
        backbone = convnext_tiny(weights=weights)
        self.encoder = backbone.features
        self.decoder3 = UpBlock(768, 384, 256)
        self.decoder2 = UpBlock(256, 192, 128)
        self.decoder1 = UpBlock(128, 96, 64)
        self.refine_half = ConvBlock(64, 32)
        self.refine_full = ConvBlock(32, 16)
        self.head = nn.Conv2d(16, 3, 1)
        self.register_buffer(
            "input_mean",
            torch.tensor((0.485, 0.456, 0.406))[None, :, None, None],
            persistent=False,
        )
        self.register_buffer(
            "input_std",
            torch.tensor((0.229, 0.224, 0.225))[None, :, None, None],
            persistent=False,
        )

    def set_encoder_trainable(self, trainable: bool) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(trainable)
        self.encoder.train(trainable)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output_size = inputs.shape[-2:]
        # Task1PatchDataset uses (RGB - 0.5) / 0.25. Convert it to the
        # normalization expected by the torchvision ImageNet weights.
        raw_rgb = inputs * 0.25 + 0.5
        encoded = (raw_rgb - self.input_mean) / self.input_std
        stage1 = self.encoder[1](self.encoder[0](encoded))
        stage2 = self.encoder[3](self.encoder[2](stage1))
        stage3 = self.encoder[5](self.encoder[4](stage2))
        stage4 = self.encoder[7](self.encoder[6](stage3))
        decoded = self.decoder3(stage4, stage3)
        decoded = self.decoder2(decoded, stage2)
        decoded = self.decoder1(decoded, stage1)
        decoded = F.interpolate(
            decoded,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )
        decoded = self.refine_half(decoded)
        decoded = F.interpolate(
            decoded,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        return self.head(self.refine_full(decoded))


class ProbabilityRefinementUNet(nn.Module):
    """Compact U-Net that refines artery/vein probabilities recurrently."""

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        if base_channels < 8 or base_channels % 8:
            raise ValueError("base_channels must be a multiple of 8 and at least 8")
        widths = [base_channels * multiplier for multiplier in (1, 2, 4, 8, 16)]
        self.encoder1 = ConvBlock(3, widths[0])
        self.encoder2 = ConvBlock(widths[0], widths[1])
        self.encoder3 = ConvBlock(widths[1], widths[2])
        self.encoder4 = ConvBlock(widths[2], widths[3])
        self.bottleneck = ConvBlock(widths[3], widths[4])
        self.pool = nn.MaxPool2d(2)
        self.decoder4 = UpBlock(widths[4], widths[3], widths[3])
        self.decoder3 = UpBlock(widths[3], widths[2], widths[2])
        self.decoder2 = UpBlock(widths[2], widths[1], widths[1])
        self.decoder1 = UpBlock(widths[1], widths[0], widths[0])
        self.head = nn.Conv2d(widths[0], 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        encoder4 = self.encoder4(self.pool(encoder3))
        bottleneck = self.bottleneck(self.pool(encoder4))
        decoded = self.decoder4(bottleneck, encoder4)
        decoded = self.decoder3(decoded, encoder3)
        decoded = self.decoder2(decoded, encoder2)
        decoded = self.decoder1(decoded, encoder1)
        return self.head(decoded)


class ConvNeXtTinyRecursiveTask1UNet(nn.Module):
    """Pretrained coarse segmentation plus shared recurrent A/V refinement.

    The vessel-union logit is anchored to the coarse CFP prediction. A shared
    refinement U-Net repeatedly updates artery and vein logits from the current
    three-channel probabilities. This independently implements the recurrent
    two-stage hypothesis without importing challenge-specific code or weights.
    """

    output_channels = ("artery", "vessel", "vein")
    weight_name = ConvNeXtTinyTask1UNet.weight_name
    weight_url = ConvNeXtTinyTask1UNet.weight_url

    def __init__(
        self,
        *,
        pretrained: bool = False,
        refinement_base_channels: int = 32,
        refinement_iterations: int = 3,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if refinement_iterations < 1:
            raise ValueError("refinement_iterations must be positive")
        self.coarse = ConvNeXtTinyTask1UNet(pretrained=pretrained)
        self.refiner = ProbabilityRefinementUNet(refinement_base_channels)
        self.refinement_iterations = refinement_iterations
        self.activation_checkpointing = activation_checkpointing

    @property
    def encoder(self) -> nn.Module:
        """Expose the coarse encoder for separate learning-rate control."""

        return self.coarse.encoder

    def set_encoder_trainable(self, trainable: bool) -> None:
        self.coarse.set_encoder_trainable(trainable)

    def forward_stages(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        coarse_logits = self.coarse(inputs)
        stages = [coarse_logits]
        vessel_logits = coarse_logits[:, 1:2]
        current_logits = coarse_logits
        state = torch.sigmoid(current_logits)
        for _ in range(self.refinement_iterations):
            if (
                self.activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                artery_vein_delta = checkpoint(
                    self.refiner,
                    state,
                    use_reentrant=False,
                )
            else:
                artery_vein_delta = self.refiner(state)
            current_artery_vein = torch.cat(
                (current_logits[:, 0:1], current_logits[:, 2:3]),
                dim=1,
            )
            artery_vein_logits = current_artery_vein + artery_vein_delta
            refined_logits = torch.cat(
                (
                    artery_vein_logits[:, 0:1],
                    vessel_logits,
                    artery_vein_logits[:, 1:2],
                ),
                dim=1,
            )
            stages.append(refined_logits)
            current_logits = refined_logits
            state = torch.sigmoid(refined_logits)
        return stages

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_stages(inputs)[-1]


def build_task1_model(
    name: str,
    *,
    base_channels: int = 16,
    pretrained: bool = False,
    refinement_base_channels: int = 32,
    refinement_iterations: int = 3,
    activation_checkpointing: bool = False,
) -> nn.Module:
    if name == "unet":
        if pretrained:
            raise ValueError("The compact U-Net has no external pretrained weights")
        return Task1UNet(base_channels=base_channels)
    if name == "convnext_tiny":
        return ConvNeXtTinyTask1UNet(pretrained=pretrained)
    if name == "convnext_tiny_recursive":
        return ConvNeXtTinyRecursiveTask1UNet(
            pretrained=pretrained,
            refinement_base_channels=refinement_base_channels,
            refinement_iterations=refinement_iterations,
            activation_checkpointing=activation_checkpointing,
        )
    raise ValueError(f"Unknown Task 1 model: {name}")
