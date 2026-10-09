from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import math


try:
    import torch
    from torch import nn
    from torch.nn import functional as F

    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR: Exception | None = None
except (ImportError, OSError) as exc:  # pragma: no cover - depends on host runtime
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = exc


@dataclass(frozen=True)
class MethodSpec:
    name: str
    learned: bool
    generative: bool
    uses_image: bool
    uses_laser: bool
    uses_uncertainty_condition: bool
    uses_spatial_mobility: bool
    uses_spatial_gate: bool
    terminal_dc: bool
    alias_of: str | None = None


METHODS: dict[str, MethodSpec] = {
    "coarse": MethodSpec("coarse", False, False, False, False, False, False, False, False),
    "gp-footprint": MethodSpec(
        "gp-footprint", False, False, False, True, True, False, False, False
    ),
    "dem-sr-dc": MethodSpec(
        "dem-sr-dc", True, False, False, True, False, False, False, True
    ),
    "det-image": MethodSpec(
        "det-image", True, False, True, False, False, False, False, False
    ),
    "det-fusion": MethodSpec(
        "det-fusion", True, False, True, True, True, False, False, True
    ),
    "Iso-SB": MethodSpec("Iso-SB", True, True, True, True, False, False, False, True),
    "Iso-SB-cond": MethodSpec(
        "Iso-SB-cond", True, True, True, True, True, False, False, True
    ),
    "DDPM-DC": MethodSpec("DDPM-DC", True, True, True, True, True, False, False, True),
    "OT-SB-full": MethodSpec(
        "OT-SB-full", True, True, True, True, True, True, True, True
    ),
}

ABLATION_IDENTITIES: dict[str, MethodSpec] = {
    "F00": MethodSpec("F00", True, True, True, True, True, False, False, True),
    "F10": MethodSpec("F10", True, True, True, True, True, True, False, True),
    "F01": MethodSpec("F01", True, True, True, True, True, False, True, True),
    "F11": MethodSpec(
        "F11", True, True, True, True, True, True, True, True, alias_of="OT-SB-full"
    ),
    "spatial-shift-M": MethodSpec(
        "spatial-shift-M", True, True, True, True, True, True, True, True
    ),
}


def method_spec(name: str) -> MethodSpec:
    if name in METHODS:
        return METHODS[name]
    if name in ABLATION_IDENTITIES:
        return ABLATION_IDENTITIES[name]
    raise ValueError(f"unknown Paper2 method: {name}")


def canonical_model_identity(name: str) -> str:
    spec = method_spec(name)
    return spec.alias_of or spec.name


def formal_model_identities() -> tuple[str, ...]:
    main = [name for name, spec in METHODS.items() if spec.learned]
    ablations = ["F00", "F10", "F01", "spatial-shift-M"]
    identities = tuple(main + ablations)
    if len(identities) != 11:
        raise AssertionError("formal protocol must contain exactly 11 learned identities")
    return identities


@dataclass(frozen=True)
class UNetConfig:
    patch_size: int = 256
    sample_channels: int = 1
    condition_channels: int = 6
    output_channels: int = 1
    base_channels: int = 96
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 4)
    residual_blocks_per_level: int = 2
    attention_max_resolution: int = 32
    time_embedding_channels: int = 384

    def validate(self, *, formal: bool = False) -> None:
        if self.patch_size <= 0 or self.patch_size % (
            2 ** (len(self.channel_multipliers) - 1)
        ):
            raise ValueError("patch_size must be divisible by all U-Net downsampling levels")
        if self.base_channels <= 0 or self.base_channels % 8:
            raise ValueError("base_channels must be positive and divisible by GroupNorm groups")
        if self.residual_blocks_per_level <= 0:
            raise ValueError("residual_blocks_per_level must be positive")
        if not self.channel_multipliers or any(value <= 0 for value in self.channel_multipliers):
            raise ValueError("channel multipliers must be nonempty and positive")
        if self.attention_max_resolution <= 0 or self.time_embedding_channels <= 0:
            raise ValueError("attention/time-embedding resolutions must be positive")
        if formal:
            if self.patch_size != 256:
                raise ValueError("formal model requires 256x256 patches")
            if self.residual_blocks_per_level != 2:
                raise ValueError("formal model requires exactly two ResBlocks per level")
            if tuple(self.channel_multipliers) != (1, 2, 4, 4):
                raise ValueError("formal model requires channel multipliers [1,2,4,4]")
            if self.base_channels != 96:
                raise ValueError("formal model requires base_channels=96")
            if (self.sample_channels, self.condition_channels, self.output_channels) != (1, 6, 1):
                raise ValueError(
                    "formal model requires 1 sample, 6 condition, and 1 output channel"
                )
            if self.attention_max_resolution != 32:
                raise ValueError(
                    "formal model requires attention at resolutions no greater than 32"
                )
            if self.time_embedding_channels != 384:
                raise ValueError("formal model requires a 384-channel time embedding")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(f"PyTorch is unavailable: {TORCH_IMPORT_ERROR}")


if TORCH_AVAILABLE:

    class TimeEmbedding(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.channels = channels
            self.projection = nn.Sequential(
                nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, channels)
            )

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            half = self.channels // 2
            frequency = torch.exp(
                -math.log(10000.0)
                * torch.arange(half, device=t.device, dtype=torch.float32)
                / max(half - 1, 1)
            )
            phase = t.float().reshape(-1, 1) * frequency.reshape(1, -1)
            embedding = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)
            if embedding.shape[1] < self.channels:
                embedding = F.pad(embedding, (0, self.channels - embedding.shape[1]))
            return self.projection(embedding)


    class ResidualBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, time_channels: int) -> None:
            super().__init__()
            self.norm1 = nn.GroupNorm(8, in_channels)
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.time = nn.Linear(time_channels, 2 * out_channels)
            self.norm2 = nn.GroupNorm(8, out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.skip = (
                nn.Identity()
                if in_channels == out_channels
                else nn.Conv2d(in_channels, out_channels, 1)
            )

        def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            h = self.conv1(F.silu(self.norm1(x)))
            scale, shift = self.time(F.silu(time)).chunk(2, dim=1)
            h = self.norm2(h) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
            h = self.conv2(F.silu(h))
            return h + self.skip(x)


    class SpatialAttention(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.norm = nn.GroupNorm(8, channels)
            self.qkv = nn.Conv2d(channels, 3 * channels, 1)
            self.output = nn.Conv2d(channels, channels, 1)
            self.scale = (channels // 8) ** -0.5
            self.heads = 8

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch, channels, height, width = x.shape
            qkv = self.qkv(self.norm(x)).reshape(
                batch, 3, self.heads, channels // self.heads, height * width
            )
            q, k, v = qkv.unbind(dim=1)
            weights = torch.einsum("bhcn,bhcm->bhnm", q * self.scale, k).softmax(dim=-1)
            attended = torch.einsum("bhnm,bhcm->bhcn", weights, v).reshape(
                batch, channels, height, width
            )
            return x + self.output(attended)


    class BlockStack(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            time_channels: int,
            blocks: int,
            attention: bool,
        ) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            current = in_channels
            for _ in range(blocks):
                layers.append(ResidualBlock(current, out_channels, time_channels))
                current = out_channels
            self.blocks = nn.ModuleList(layers)
            self.attention = SpatialAttention(out_channels) if attention else nn.Identity()

        def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            for block in self.blocks:
                x = block(x, time)
            return self.attention(x)


    class ConditionedUNet(nn.Module):
        """Shared formal U-Net used by all learned Paper2 reconstruction identities.

        The six condition planes have a fixed order: Z0, image, image mask, GP-fused
        Zf, posterior omega, and rasterized laser observations. Methods that do not
        use a plane receive zeros, keeping architecture and parameter count matched.
        """

        condition_order = ("z0", "image", "image_mask", "zf", "omega", "laser")

        def __init__(self, config: UNetConfig | None = None) -> None:
            super().__init__()
            self.config = config or UNetConfig()
            self.config.validate()
            widths = [self.config.base_channels * value for value in self.config.channel_multipliers]
            total_input = self.config.sample_channels + self.config.condition_channels
            self.time = TimeEmbedding(self.config.time_embedding_channels)
            self.input = nn.Conv2d(total_input, widths[0], 3, padding=1)
            encoders: list[nn.Module] = []
            downsamples: list[nn.Module] = []
            for index, width in enumerate(widths):
                resolution = self.config.patch_size // (2**index)
                encoders.append(
                    BlockStack(
                        width,
                        width,
                        self.config.time_embedding_channels,
                        self.config.residual_blocks_per_level,
                        attention=resolution <= self.config.attention_max_resolution,
                    )
                )
                if index < len(widths) - 1:
                    downsamples.append(nn.Conv2d(width, widths[index + 1], 3, stride=2, padding=1))
            self.encoders = nn.ModuleList(encoders)
            self.downsamples = nn.ModuleList(downsamples)
            self.middle = BlockStack(
                widths[-1], widths[-1], self.config.time_embedding_channels, 2, True
            )
            decoders: list[nn.Module] = []
            upsamples: list[nn.Module] = []
            for index in range(len(widths) - 2, -1, -1):
                upsamples.append(
                    nn.ConvTranspose2d(widths[index + 1], widths[index], 4, stride=2, padding=1)
                )
                resolution = self.config.patch_size // (2**index)
                decoders.append(
                    BlockStack(
                        2 * widths[index],
                        widths[index],
                        self.config.time_embedding_channels,
                        self.config.residual_blocks_per_level,
                        attention=resolution <= self.config.attention_max_resolution,
                    )
                )
            self.upsamples = nn.ModuleList(upsamples)
            self.decoders = nn.ModuleList(decoders)
            self.output = nn.Sequential(
                nn.GroupNorm(8, widths[0]),
                nn.SiLU(),
                nn.Conv2d(widths[0], self.config.output_channels, 3, padding=1),
            )
            # Every learned identity predicts a correction to an existing DEM.
            # Start from the exact identity mapping so an arbitrary output-head
            # bias cannot create a large absolute-elevation offset before training.
            nn.init.zeros_(self.output[-1].weight)
            nn.init.zeros_(self.output[-1].bias)

        def forward(
            self, sample: torch.Tensor, condition: torch.Tensor, time: torch.Tensor
        ) -> torch.Tensor:
            if condition.shape[1] != self.config.condition_channels:
                raise ValueError(
                    f"expected {self.config.condition_channels} condition channels, "
                    f"got {condition.shape[1]}"
                )
            embedding = self.time(time)
            x = self.input(torch.cat([sample, condition], dim=1))
            skips: list[torch.Tensor] = []
            for index, encoder in enumerate(self.encoders):
                x = encoder(x, embedding)
                skips.append(x)
                if index < len(self.downsamples):
                    x = self.downsamples[index](x)
            x = self.middle(x, embedding)
            for offset, (up, decoder) in enumerate(zip(self.upsamples, self.decoders)):
                x = up(x)
                skip = skips[-2 - offset]
                x = decoder(torch.cat([x, skip], dim=1), embedding)
            return self.output(x)


else:

    class ConditionedUNet:  # pragma: no cover - error path is tested through factory
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


def build_model(config: UNetConfig | None = None) -> Any:
    _require_torch()
    return ConditionedUNet(config)
