"""Small CPU-safe grid dynamics model definitions."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


class GridAutoencoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        latent_dim: int = 32,
        base_channels: int = 16,
        input_shape: tuple[int, int, int] | None = None,
    ):
        super().__init__()
        self.input_channels = int(input_channels)
        self.latent_dim = int(latent_dim)
        self.base_channels = int(base_channels)
        self.input_shape = tuple(int(v) for v in (input_shape or (self.input_channels, 32, 32)))
        if len(self.input_shape) != 3:
            raise ValueError("input_shape must be (channels, height, width).")
        if self.input_shape[0] != self.input_channels:
            raise ValueError("input_shape channel count must match input_channels.")
        height, width = self.input_shape[1], self.input_shape[2]
        if height % 4 or width % 4:
            raise ValueError("GridAutoencoder input height and width must be divisible by 4.")
        hidden_channels = self.base_channels * 2
        self.encoded_shape = (hidden_channels, height // 4, width // 4)
        encoded_features = hidden_channels * self.encoded_shape[1] * self.encoded_shape[2]
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(self.input_channels, self.base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(self.base_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.encoder_fc = nn.Linear(encoded_features, self.latent_dim)
        self.decoder_fc = nn.Linear(self.latent_dim, encoded_features)
        self.decoder_cnn = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(hidden_channels, self.base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(self.base_channels, self.input_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_cnn(x)
        return self.encoder_fc(h.reshape(h.shape[0], -1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z).reshape(z.shape[0], *self.encoded_shape)
        return self.decoder_cnn(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


class LatentGRUPredictor(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.gru = nn.GRU(input_size=self.latent_dim, hidden_size=self.hidden_dim, num_layers=int(num_layers), batch_first=True)
        self.head = nn.Linear(self.hidden_dim, self.latent_dim)

    def forward(self, z_window: torch.Tensor) -> torch.Tensor:
        output, _hidden = self.gru(z_window)
        return self.head(output[:, -1, :])


class LatentTransformerPredictor(nn.Module):
    """Small temporal Transformer over standardized latent-code windows."""

    def __init__(
        self,
        latent_dim: int = 32,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        max_window_frames: int = 64,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.max_window_frames = int(max_window_frames)
        self.input = nn.Linear(self.latent_dim, self.model_dim)
        self.position = nn.Parameter(torch.zeros(1, self.max_window_frames, self.model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.num_heads,
            dim_feedforward=self.model_dim * 4,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.num_layers)
        self.norm = nn.LayerNorm(self.model_dim)
        self.head = nn.Linear(self.model_dim, self.latent_dim)

    def forward(self, z_window: torch.Tensor) -> torch.Tensor:
        if z_window.shape[1] > self.max_window_frames:
            raise ValueError(f"Window has {z_window.shape[1]} frames, max_window_frames={self.max_window_frames}.")
        x = self.input(z_window) + self.position[:, : z_window.shape[1], :]
        encoded = self.encoder(x)
        return self.head(self.norm(encoded[:, -1, :]))


class ConvGRUCell(nn.Module):
    """Convolutional GRU cell for small grid sequences."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        padding = int(kernel_size) // 2
        gate_channels = self.input_channels + self.hidden_channels
        self.gates = nn.Conv2d(gate_channels, self.hidden_channels * 2, kernel_size=int(kernel_size), padding=padding)
        self.candidate = nn.Conv2d(gate_channels, self.hidden_channels, kernel_size=int(kernel_size), padding=padding)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        if hidden is None:
            hidden = x.new_zeros((x.shape[0], self.hidden_channels, x.shape[2], x.shape[3]))
        reset_gate, update_gate = torch.chunk(torch.sigmoid(self.gates(torch.cat([x, hidden], dim=1))), 2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat([x, reset_gate * hidden], dim=1)))
        return (1.0 - update_gate) * hidden + update_gate * candidate


class PixelConvGRUResidual(nn.Module):
    """Pixel-space residual predictor that preserves 2-D spatial structure."""

    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: int = 32,
        num_layers: int = 1,
        kernel_size: int = 3,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.num_layers = int(num_layers)
        self.kernel_size = int(kernel_size)
        self.residual_scale = float(residual_scale)
        cells = []
        for layer_index in range(self.num_layers):
            in_channels = self.input_channels if layer_index == 0 else self.hidden_channels
            cells.append(ConvGRUCell(in_channels, self.hidden_channels, kernel_size=self.kernel_size))
        self.cells = nn.ModuleList(cells)
        self.head = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_channels, self.input_channels, kernel_size=3, padding=1),
        )

    def forward(self, x_window: torch.Tensor, *, return_residual: bool = False):
        if x_window.ndim != 5:
            raise ValueError("x_window must have shape (batch, time, channels, height, width).")
        states: list[torch.Tensor | None] = [None] * len(self.cells)
        for t in range(x_window.shape[1]):
            x = x_window[:, t]
            for layer_index, cell in enumerate(self.cells):
                states[layer_index] = cell(x, states[layer_index])
                x = states[layer_index]
        residual = self.residual_scale * torch.tanh(self.head(states[-1]))
        pred = torch.clamp(x_window[:, -1] + residual, 0.0, 1.0)
        if return_residual:
            return pred, residual
        return pred



class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell for small grid sequences."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        padding = int(kernel_size) // 2
        self.gates = nn.Conv2d(self.input_channels + self.hidden_channels, self.hidden_channels * 4, kernel_size=int(kernel_size), padding=padding)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            h = x.new_zeros((x.shape[0], self.hidden_channels, x.shape[2], x.shape[3]))
            c = x.new_zeros((x.shape[0], self.hidden_channels, x.shape[2], x.shape[3]))
        else:
            h, c = state
        i, f, o, g = torch.chunk(self.gates(torch.cat([x, h], dim=1)), 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class PixelConvLSTMResidual(nn.Module):
    """Pixel-space residual predictor using stacked ConvLSTM cells."""

    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: int = 32,
        num_layers: int = 1,
        kernel_size: int = 3,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.num_layers = int(num_layers)
        self.kernel_size = int(kernel_size)
        self.residual_scale = float(residual_scale)
        cells = []
        for layer_index in range(self.num_layers):
            in_channels = self.input_channels if layer_index == 0 else self.hidden_channels
            cells.append(ConvLSTMCell(in_channels, self.hidden_channels, kernel_size=self.kernel_size))
        self.cells = nn.ModuleList(cells)
        self.head = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_channels, self.input_channels, kernel_size=3, padding=1),
        )

    def forward(self, x_window: torch.Tensor, *, return_residual: bool = False):
        if x_window.ndim != 5:
            raise ValueError("x_window must have shape (batch, time, channels, height, width).")
        states: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.cells)
        x = x_window[:, 0]
        for t in range(x_window.shape[1]):
            x = x_window[:, t]
            for layer_index, cell in enumerate(self.cells):
                states[layer_index] = cell(x, states[layer_index])
                x = states[layer_index][0]
        residual = self.residual_scale * torch.tanh(self.head(x))
        pred = torch.clamp(x_window[:, -1] + residual, 0.0, 1.0)
        if return_residual:
            return pred, residual
        return pred


class TemporalCNNResidual(nn.Module):
    """Temporal-stack CNN residual predictor over the full input window."""

    def __init__(
        self,
        input_channels: int = 1,
        window_frames: int = 8,
        hidden_channels: int = 32,
        residual_scale: float = 0.1,
        dilation: int = 2,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.input_channels = int(input_channels)
        self.window_frames = int(window_frames)
        self.hidden_channels = int(hidden_channels)
        self.residual_scale = float(residual_scale)
        self.num_blocks = max(1, int(num_blocks))
        in_channels = self.input_channels * self.window_frames
        base_dilation = max(1, int(dilation))
        dilation_cycle = (1, base_dilation, max(1, base_dilation * 2))
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        ]
        for block_index in range(self.num_blocks):
            block_dilation = dilation_cycle[block_index % len(dilation_cycle)]
            layers.extend(
                [
                    nn.Conv2d(
                        self.hidden_channels,
                        self.hidden_channels,
                        kernel_size=3,
                        padding=block_dilation,
                        dilation=block_dilation,
                    ),
                    nn.ReLU(),
                ]
            )
        layers.append(nn.Conv2d(self.hidden_channels, self.input_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_window: torch.Tensor, *, return_residual: bool = False):
        if x_window.ndim != 5:
            raise ValueError("x_window must have shape (batch, time, channels, height, width).")
        if x_window.shape[1] != self.window_frames:
            raise ValueError(f"Expected {self.window_frames} input frames, got {x_window.shape[1]}.")
        b, t, c, h, w = x_window.shape
        x = x_window.reshape(b, t * c, h, w)
        residual = self.residual_scale * torch.tanh(self.net(x))
        pred = torch.clamp(x_window[:, -1] + residual, 0.0, 1.0)
        if return_residual:
            return pred, residual
        return pred


def _activation_layer(name: str) -> nn.Module:
    value = str(name or "relu").strip().lower()
    if value == "relu":
        return nn.ReLU(inplace=True)
    if value == "gelu":
        return nn.GELU()
    if value in {"silu", "swish"}:
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


def _normalization_layer(name: str, channels: int) -> nn.Module:
    value = str(name or "none").strip().lower()
    if value in {"none", "identity", ""}:
        return nn.Identity()
    if value == "batch":
        return nn.BatchNorm2d(int(channels))
    if value == "instance":
        return nn.InstanceNorm2d(int(channels), affine=True)
    if value == "group":
        for groups in (8, 4, 2, 1):
            if int(channels) % groups == 0:
                return nn.GroupNorm(groups, int(channels))
    raise ValueError(f"Unsupported normalization: {name}")


class ConvBlock2d(nn.Module):
    """Configurable convolution block used by scalable pixel predictors."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        normalization: str = "none",
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        kernel = int(kernel_size)
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        dilation = max(1, int(dilation))
        padding = dilation * (kernel // 2)
        layers: list[nn.Module] = [
            nn.Conv2d(
                int(in_channels),
                int(out_channels),
                kernel_size=kernel,
                stride=int(stride),
                padding=padding,
                dilation=dilation,
                bias=str(normalization or "none").lower() in {"none", "identity", ""},
            ),
            _normalization_layer(normalization, int(out_channels)),
            _activation_layer(activation),
        ]
        if float(dropout) > 0:
            layers.append(nn.Dropout2d(float(dropout)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _int_list(value: Any, *, fallback: tuple[int, ...]) -> list[int]:
    if value is None:
        return [int(v) for v in fallback]
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        out = [int(v) for v in value]
        if out:
            return out
    raise ValueError("Expected a non-empty integer or integer list.")


def _stage_counts(value: Any, *, stages: int, fallback: int = 1) -> list[int]:
    if value is None:
        return [int(fallback)] * int(stages)
    if isinstance(value, int):
        return [max(1, int(value))] * int(stages)
    counts = [max(1, int(v)) for v in value]
    if not counts:
        return [int(fallback)] * int(stages)
    if len(counts) < stages:
        counts.extend([counts[-1]] * (stages - len(counts)))
    return counts[:stages]


def _dilation_at(pattern: list[int], index: int) -> int:
    if not pattern:
        return 1
    return max(1, int(pattern[int(index) % len(pattern)]))


class ScalableTemporalCNNResidual(nn.Module):
    """Configurable temporal-stack CNN residual predictor.

    The model treats the input window as a channel stack, predicts a bounded
    residual in pixel space, and adds it to the last observed frame.
    """

    def __init__(
        self,
        input_channels: int = 1,
        window_frames: int = 8,
        architecture_spec: Mapping[str, Any] | None = None,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.input_channels = int(input_channels)
        self.window_frames = int(window_frames)
        self.residual_scale = float(residual_scale)
        spec = dict(architecture_spec or {})
        self.architecture_spec = self._normalize_spec(spec)
        topology = str(self.architecture_spec["topology"])
        in_channels = self.input_channels * self.window_frames
        if topology == "stack":
            self.encoder = nn.ModuleList()
            self.bottleneck = nn.Identity()
            self.decoder = nn.ModuleList()
            self.stack = self._build_stack(in_channels)
            head_channels = int(self.architecture_spec["stack_channels"][-1])
        elif topology == "encoder_decoder":
            self.stack = nn.Identity()
            self.encoder, self.bottleneck, self.decoder, head_channels = self._build_encoder_decoder(in_channels)
        else:
            raise ValueError(f"Unsupported scalable temporal CNN topology: {topology}")
        self.head = nn.Conv2d(head_channels, self.input_channels, kernel_size=3, padding=1)

    @staticmethod
    def _normalize_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
        topology = str(spec.get("topology", "encoder_decoder")).strip().lower()
        if topology not in {"stack", "encoder_decoder"}:
            raise ValueError(f"Unsupported topology: {topology}")
        kernel_size = int(spec.get("kernel_size", 3))
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        dilations = _int_list(spec.get("dilations"), fallback=(1,))
        normalized = {
            "architecture_id": str(spec.get("architecture_id") or topology),
            "topology": topology,
            "kernel_size": kernel_size,
            "dilations": dilations,
            "normalization": str(spec.get("normalization", "group")).strip().lower(),
            "activation": str(spec.get("activation", "silu")).strip().lower(),
            "dropout": float(spec.get("dropout", 0.0)),
            "skip_connections": bool(spec.get("skip_connections", topology == "encoder_decoder")),
        }
        if topology == "stack":
            stack_channels = _int_list(spec.get("stack_channels") or spec.get("channels"), fallback=(32, 32, 32))
            normalized["stack_channels"] = stack_channels
            normalized["stack_blocks"] = _stage_counts(spec.get("stack_blocks") or spec.get("blocks"), stages=len(stack_channels), fallback=1)
            normalized["encoder_channels"] = []
            normalized["encoder_blocks"] = []
            normalized["decoder_channels"] = []
            normalized["decoder_blocks"] = []
            normalized["bottleneck_channels"] = stack_channels[-1]
            normalized["bottleneck_blocks"] = 0
        else:
            encoder_channels = _int_list(spec.get("encoder_channels"), fallback=(16, 32, 64))
            if len(encoder_channels) < 1:
                raise ValueError("encoder_channels must contain at least one stage.")
            decoder_default = tuple(reversed(encoder_channels[:-1]))
            decoder_channels = _int_list(spec.get("decoder_channels"), fallback=decoder_default or (encoder_channels[0],))
            normalized["encoder_channels"] = encoder_channels
            normalized["encoder_blocks"] = _stage_counts(spec.get("encoder_blocks"), stages=len(encoder_channels), fallback=1)
            normalized["decoder_channels"] = decoder_channels
            normalized["decoder_blocks"] = _stage_counts(spec.get("decoder_blocks"), stages=len(decoder_channels), fallback=1)
            normalized["bottleneck_channels"] = int(spec.get("bottleneck_channels", encoder_channels[-1]))
            normalized["bottleneck_blocks"] = max(0, int(spec.get("bottleneck_blocks", 1)))
            normalized["stack_channels"] = []
            normalized["stack_blocks"] = []
        return normalized

    def _block(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        dilation: int = 1,
    ) -> ConvBlock2d:
        return ConvBlock2d(
            in_channels,
            out_channels,
            kernel_size=int(self.architecture_spec["kernel_size"]),
            stride=stride,
            dilation=dilation,
            normalization=str(self.architecture_spec["normalization"]),
            activation=str(self.architecture_spec["activation"]),
            dropout=float(self.architecture_spec["dropout"]),
        )

    def _build_stack(self, in_channels: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        current = int(in_channels)
        block_index = 0
        for stage_channels, stage_blocks in zip(self.architecture_spec["stack_channels"], self.architecture_spec["stack_blocks"]):
            for _ in range(int(stage_blocks)):
                layers.append(self._block(current, int(stage_channels), dilation=_dilation_at(self.architecture_spec["dilations"], block_index)))
                current = int(stage_channels)
                block_index += 1
        return nn.Sequential(*layers)

    def _build_encoder_decoder(self, in_channels: int):
        encoder = nn.ModuleList()
        current = int(in_channels)
        block_index = 0
        for stage_index, (stage_channels, stage_blocks) in enumerate(zip(self.architecture_spec["encoder_channels"], self.architecture_spec["encoder_blocks"])):
            layers: list[nn.Module] = []
            for block_in_stage in range(int(stage_blocks)):
                stride = 2 if stage_index > 0 and block_in_stage == 0 else 1
                layers.append(self._block(current, int(stage_channels), stride=stride, dilation=_dilation_at(self.architecture_spec["dilations"], block_index)))
                current = int(stage_channels)
                block_index += 1
            encoder.append(nn.Sequential(*layers))

        bottleneck_layers: list[nn.Module] = []
        bottleneck_channels = int(self.architecture_spec["bottleneck_channels"])
        for _ in range(int(self.architecture_spec["bottleneck_blocks"])):
            bottleneck_layers.append(self._block(current, bottleneck_channels, dilation=_dilation_at(self.architecture_spec["dilations"], block_index)))
            current = bottleneck_channels
            block_index += 1
        bottleneck = nn.Sequential(*bottleneck_layers) if bottleneck_layers else nn.Identity()

        decoder = nn.ModuleList()
        encoder_channels = list(self.architecture_spec["encoder_channels"])
        skip_enabled = bool(self.architecture_spec["skip_connections"])
        for stage_index, (stage_channels, stage_blocks) in enumerate(zip(self.architecture_spec["decoder_channels"], self.architecture_spec["decoder_blocks"])):
            skip_channels = 0
            skip_index = len(encoder_channels) - 2 - stage_index
            if skip_enabled and skip_index >= 0:
                skip_channels = int(encoder_channels[skip_index])
            layers = []
            stage_in = current + skip_channels
            for _ in range(int(stage_blocks)):
                layers.append(self._block(stage_in, int(stage_channels), dilation=_dilation_at(self.architecture_spec["dilations"], block_index)))
                stage_in = int(stage_channels)
                block_index += 1
            current = int(stage_channels)
            decoder.append(nn.Sequential(*layers))
        return encoder, bottleneck, decoder, current

    def forward(self, x_window: torch.Tensor, *, return_residual: bool = False):
        if x_window.ndim != 5:
            raise ValueError("x_window must have shape (batch, time, channels, height, width).")
        if x_window.shape[1] != self.window_frames:
            raise ValueError(f"Expected {self.window_frames} input frames, got {x_window.shape[1]}.")
        b, t, c, h, w = x_window.shape
        x = x_window.reshape(b, t * c, h, w)
        if self.architecture_spec["topology"] == "stack":
            x = self.stack(x)
        else:
            skips: list[torch.Tensor] = []
            for stage in self.encoder:
                x = stage(x)
                skips.append(x)
            x = self.bottleneck(x)
            for stage_index, stage in enumerate(self.decoder):
                skip_index = len(skips) - 2 - stage_index
                use_skip = bool(self.architecture_spec["skip_connections"]) and skip_index >= 0
                target_size = skips[skip_index].shape[-2:] if use_skip else None
                x = F.interpolate(x, size=target_size, scale_factor=None if target_size is not None else 2, mode="bilinear", align_corners=False)
                if use_skip:
                    skip = skips[skip_index]
                    if x.shape[-2:] != skip.shape[-2:]:
                        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                    x = torch.cat([x, skip], dim=1)
                x = stage(x)
            if x.shape[-2:] != (h, w):
                x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        residual = self.residual_scale * torch.tanh(self.head(x))
        pred = torch.clamp(x_window[:, -1] + residual, 0.0, 1.0)
        if return_residual:
            return pred, residual
        return pred



class UNetConvGRUResidual(nn.Module):
    """Multi-scale U-Net residual predictor with a ConvGRU bottleneck."""

    def __init__(
        self,
        input_channels: int = 1,
        base_channels: int = 16,
        hidden_channels: int = 32,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.input_channels = int(input_channels)
        self.base_channels = int(base_channels)
        self.hidden_channels = int(hidden_channels)
        self.residual_scale = float(residual_scale)
        self.enc1 = nn.Sequential(
            nn.Conv2d(self.input_channels, self.base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.base_channels, self.base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(self.base_channels, self.base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.base_channels * 2, self.base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(self.base_channels * 2, self.hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.temporal = ConvGRUCell(self.hidden_channels, self.hidden_channels)
        self.dec2 = nn.Sequential(
            nn.Conv2d(self.hidden_channels + self.base_channels * 2, self.base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.base_channels * 2, self.base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(self.base_channels * 2 + self.base_channels, self.base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.base_channels, self.base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Conv2d(self.base_channels, self.input_channels, kernel_size=3, padding=1)

    def forward(self, x_window: torch.Tensor, *, return_residual: bool = False):
        if x_window.ndim != 5:
            raise ValueError("x_window must have shape (batch, time, channels, height, width).")
        state = None
        skip1 = None
        skip2 = None
        for t in range(x_window.shape[1]):
            e1 = self.enc1(x_window[:, t])
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            state = self.temporal(e3, state)
            skip1 = e1
            skip2 = e2
        assert state is not None and skip1 is not None and skip2 is not None
        x = nn.functional.interpolate(state, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([x, skip2], dim=1))
        x = nn.functional.interpolate(x, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat([x, skip1], dim=1))
        residual = self.residual_scale * torch.tanh(self.head(x))
        pred = torch.clamp(x_window[:, -1] + residual, 0.0, 1.0)
        if return_residual:
            return pred, residual
        return pred
