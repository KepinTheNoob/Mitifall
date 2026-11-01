from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .data import InputLayout

RAD_TO_DEG = 180.0 / math.pi

class InputPrep(nn.Module):

    def __init__(self, layout: InputLayout, center=None, scale=None):
        super().__init__()
        self.layout = layout
        n = len(layout.model_channels)
        center = torch.zeros(n) if center is None else torch.as_tensor(center, dtype=torch.float32)
        scale = torch.ones(n) if scale is None else torch.as_tensor(scale, dtype=torch.float32)
        self.register_buffer("center", center.reshape(1, n, 1).clone())
        self.register_buffer("scale", scale.reshape(1, n, 1).clone())

    @staticmethod
    def _norm(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:

        return torch.sqrt(torch.clamp_min(x * x + y * y + z * z, 1e-12))

    def derive(self, raw: torch.Tensor) -> torch.Tensor:
        ax, ay, az = raw[:, 0:1], raw[:, 1:2], raw[:, 2:3]
        acc_mag = self._norm(ax, ay, az)
        tilt = torch.acos(torch.clamp(az / acc_mag, -1.0 + 1e-6, 1.0 - 1e-6)) * RAD_TO_DEG
        parts = [ax, ay, az, acc_mag, tilt]
        nxt = 3
        if self.layout.use_gyro:
            gx, gy, gz = raw[:, 3:4], raw[:, 4:5], raw[:, 5:6]
            parts += [gx, gy, gz, self._norm(gx, gy, gz)]
            nxt = 6
        if self.layout.use_baro:
            parts.append(raw[:, nxt:nxt + 1])
        return torch.cat(parts, dim=1)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return (self.derive(raw) - self.center) / self.scale

class TemporalBlock(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.conv1 = nn.Conv1d(cin, cout, kernel, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, kernel, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop(F.relu(self.bn1(self.conv1(x))))
        y = self.drop(F.relu(self.bn2(self.conv2(y))))
        return F.relu(y + self.skip(x))

class LiteSelfAttention(nn.Module):

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads, self.dk = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.dk), dim=-1)
        y = (self.drop(att) @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(y)

class AttentionPool(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(h), dim=1)
        return (w * h).sum(dim=1)

def _head(dim: int, dropout: float) -> nn.Module:
    return nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, 1))

class TCNAttention(nn.Module):
    def __init__(self, in_ch: int, cfg: ModelConfig):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, cfg.channels, cfg.kernel, stride=2, padding=cfg.kernel // 2),
            nn.BatchNorm1d(cfg.channels), nn.ReLU())
        self.tcn = nn.Sequential(*[
            TemporalBlock(cfg.channels, cfg.channels, cfg.kernel, 2 ** level, cfg.dropout)
            for level in range(cfg.levels)])
        self.attn = LiteSelfAttention(cfg.channels, cfg.heads, cfg.dropout)
        self.norm = nn.LayerNorm(cfg.channels)
        self.pool = AttentionPool(cfg.channels)
        self.head = _head(cfg.channels, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tcn(self.stem(x)).transpose(1, 2)
        h = self.norm(h + self.attn(h))
        return self.head(self.pool(h)).squeeze(-1)

class CNNBiLSTM(nn.Module):
    def __init__(self, in_ch: int, cfg: ModelConfig):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, cfg.channels, 5, padding=2), nn.BatchNorm1d(cfg.channels), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(cfg.channels, 2 * cfg.channels, 5, padding=2), nn.BatchNorm1d(2 * cfg.channels),
            nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(cfg.dropout),
        )
        self.lstm = nn.LSTM(2 * cfg.channels, cfg.lstm_hidden, batch_first=True, bidirectional=True)
        self.pool = AttentionPool(2 * cfg.lstm_hidden)
        self.head = _head(2 * cfg.lstm_hidden, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(self.conv(x).transpose(1, 2))
        return self.head(self.pool(h)).squeeze(-1)

class FallNet(nn.Module):

    def __init__(self, layout: InputLayout, cfg: ModelConfig, center=None, scale=None):
        super().__init__()
        self.layout, self.cfg = layout, cfg
        self.prep = InputPrep(layout, center, scale)
        n_in = len(layout.model_channels)
        if cfg.arch == "tcn":
            self.backbone = TCNAttention(n_in, cfg)
        elif cfg.arch == "cnn_bilstm":
            self.backbone = CNNBiLSTM(n_in, cfg)
        else:
            raise ValueError(f"unknown arch {cfg.arch!r}")

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.prep(raw))

    def probability(self, raw: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(raw))

class FallEnsemble(nn.Module):

    def __init__(self, members: list[FallNet]):
        super().__init__()
        if not members:
            raise ValueError("an ensemble needs at least one member")
        self.members = nn.ModuleList(members)
        self.layout, self.cfg = members[0].layout, members[0].cfg

    def probability(self, raw: torch.Tensor) -> torch.Tensor:
        return torch.stack([m.probability(raw) for m in self.members]).mean(dim=0)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(self.probability(raw), 1e-7, 1.0 - 1e-7)
        return torch.log(p) - torch.log1p(-p)

class ProbabilityModel(nn.Module):

    def __init__(self, net: FallNet | FallEnsemble):
        super().__init__()
        self.net = net

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return self.net.probability(raw)

class BinaryFocalLoss(nn.Module):

    def __init__(self, alpha: float, gamma: float):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def estimate_macs(model: FallNet, n_raw: int, T: int) -> int:
    macs = 0

    def conv_hook(m, _inp, out):
        nonlocal macs
        macs += out.shape[-1] * m.out_channels * (m.in_channels // m.groups) * m.kernel_size[0]

    def linear_hook(m, inp, _out):
        nonlocal macs
        tokens = inp[0].numel() // inp[0].shape[-1]
        macs += tokens * m.in_features * m.out_features

    def attn_hook(m, inp, _out):
        nonlocal macs
        _, t, d = inp[0].shape
        macs += 2 * t * t * d

    def lstm_hook(m, inp, _out):
        nonlocal macs
        t = inp[0].shape[1]
        dirs = 2 if m.bidirectional else 1
        macs += dirs * t * 4 * m.hidden_size * (m.input_size + m.hidden_size)

    hooks = []
    for mod in model.modules():
        if isinstance(mod, nn.Conv1d):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(linear_hook))
        elif isinstance(mod, LiteSelfAttention):
            hooks.append(mod.register_forward_hook(attn_hook))
        elif isinstance(mod, nn.LSTM):
            hooks.append(mod.register_forward_hook(lstm_hook))
    was_training = model.training
    model.eval()
    model(torch.zeros(1, n_raw, T))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return int(macs)
