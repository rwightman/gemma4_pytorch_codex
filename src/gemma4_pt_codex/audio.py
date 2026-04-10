from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AudioConfig
from .layers import ClippedLinear, RMSNorm, gelu_tanh, init_linear_module


def _make_linear(in_features: int, out_features: int, *, bias: bool = False, std: float = 1e-2) -> nn.Module:
    return ClippedLinear(in_features, out_features, bias=bias, init_std=std)


def _pad_time_dim(x: torch.Tensor, left: int, right: int, value: float = 0.0) -> torch.Tensor:
    if left == 0 and right == 0:
        return x
    parts = []
    if left > 0:
        left_shape = list(x.shape)
        left_shape[1] = left
        parts.append(x.new_full(left_shape, value))
    parts.append(x)
    if right > 0:
        right_shape = list(x.shape)
        right_shape[1] = right
        parts.append(x.new_full(right_shape, value))
    return torch.cat(parts, dim=1)


class Gemma4AudioSubsampler(nn.Module):
    def __init__(self, config: AudioConfig) -> None:
        super().__init__()
        c0, c1 = config.subsampling_channels
        subsampled_bins = ((config.num_mel_bins + 1) // 2 + 1) // 2
        self.conv0 = nn.Conv2d(1, c0, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1 = nn.Conv2d(c0, c1, kernel_size=3, stride=2, padding=1, bias=False)
        init_linear_module(self.conv0, config.init_std)
        init_linear_module(self.conv1, config.init_std)
        self.norm0 = nn.LayerNorm(c0)
        self.norm1 = nn.LayerNorm(c1)
        self.output_proj = _make_linear(c1 * subsampled_bins, config.hidden_size, bias=False, std=config.init_std)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = features.unsqueeze(1)
        hidden = self.conv0(hidden).permute(0, 2, 3, 1)
        mask = mask[:, ::2][:, : hidden.shape[1]]
        hidden = F.relu(self.norm0(hidden))

        hidden = hidden.permute(0, 3, 1, 2)
        hidden = self.conv1(hidden).permute(0, 2, 3, 1)
        mask = mask[:, ::2][:, : hidden.shape[1]]
        hidden = F.relu(self.norm1(hidden))

        hidden = hidden.flatten(-2)
        hidden = self.output_proj(hidden)
        return hidden, mask


class Gemma4AudioFeedForward(nn.Module):
    def __init__(self, config: AudioConfig, residual_weight: float = 0.5) -> None:
        super().__init__()
        self.residual_weight = residual_weight
        self.gradient_clipping = config.gradient_clipping
        self.pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn1 = _make_linear(config.hidden_size, config.hidden_size * 4, bias=False, std=config.init_std)
        self.ffn2 = _make_linear(config.hidden_size * 4, config.hidden_size, bias=False, std=config.init_std)
        self.post_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.clamp(-self.gradient_clipping, self.gradient_clipping)
        x = self.pre_norm(x)
        x = F.silu(self.ffn1(x))
        x = self.ffn2(x)
        x = x.clamp(-self.gradient_clipping, self.gradient_clipping)
        x = self.post_norm(x)
        return residual + x * self.residual_weight


class Gemma4AudioLightConv(nn.Module):
    def __init__(self, config: AudioConfig) -> None:
        super().__init__()
        self.pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.linear_start = _make_linear(config.hidden_size, 2 * config.hidden_size, bias=False, std=config.init_std)
        self.depthwise = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.conv_kernel_size,
            groups=config.hidden_size,
            bias=False,
        )
        init_linear_module(self.depthwise, config.init_std)
        self.conv_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.linear_end = _make_linear(config.hidden_size, config.hidden_size, bias=False, std=config.init_std)
        self.gradient_clipping = config.gradient_clipping

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pre_norm(x)
        x = F.glu(self.linear_start(x), dim=-1)
        x = x.transpose(1, 2)
        x = F.pad(x, (self.depthwise.kernel_size[0] - 1, 0))
        x = self.depthwise(x).transpose(1, 2)
        x = x.clamp(-self.gradient_clipping, self.gradient_clipping)
        x = F.silu(self.conv_norm(x))
        x = self.linear_end(x)
        return residual + x


def _extract_block_context(
        x: torch.Tensor,
        block_size: int,
        left_context: int,
        right_context: int,
        padding_value: float = 0.0,
) -> torch.Tensor:
    if left_context < 0 or right_context < 0:
        raise ValueError("Context sizes must be non-negative.")
    x = _pad_time_dim(x, left_context, right_context + block_size - 1, padding_value)
    frame_length = block_size + left_context + right_context
    num_frames = (x.shape[1] - frame_length) // block_size + 1
    start = torch.arange(num_frames, device=x.device) * block_size
    indices = start[:, None] + torch.arange(frame_length, device=x.device)
    return x[:, indices]


def _convert_to_block(x: torch.Tensor, block_size: int, padding_value: float = 0.0) -> torch.Tensor:
    batch, seq_len = x.shape[:2]
    num_blocks = (seq_len + block_size - 1) // block_size
    pad = num_blocks * block_size - seq_len
    if pad > 0:
        x = _pad_time_dim(x, 0, pad, padding_value)
    return x.view(batch, num_blocks, block_size, *x.shape[2:])


def _causal_valid_mask(config: AudioConfig, device: torch.device) -> torch.Tensor:
    chunk = config.chunk_size
    left = max(0, config.left_context - 1)
    right = config.right_context
    context = chunk + left + right
    rows = torch.arange(chunk, device=device)[:, None]
    cols = torch.arange(context, device=device)[None, :]
    lower = cols >= rows
    upper = cols <= rows + left + right
    return lower & upper


class Gemma4AudioRelativePosition(nn.Module):
    def __init__(self, config: AudioConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads
        self.left_context = config.left_context
        self.right_context = config.right_context
        self.pos_proj = init_linear_module(
            nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=False),
            config.init_std,
        )
        nn.init.xavier_uniform_(self.pos_proj.weight)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        batch, num_blocks, block_size, num_heads, head_dim = queries.shape
        context = keys.shape[2]
        left = max(0, self.left_context - 1)
        right = self.right_context

        q = queries.permute(0, 3, 1, 2, 4)
        k = keys.permute(0, 3, 1, 4, 2)
        term_ac = torch.matmul(q, k)

        positions = torch.arange(left, -right - 1, -1, device=queries.device).view(1, -1)
        num_timescales = self.hidden_size // 2
        log_increment = math.log(10_000.0 / 1.0) / max(num_timescales - 1, 1)
        inv_timescales = torch.exp(
            torch.arange(num_timescales, device=queries.device, dtype=torch.float32) * -log_increment
        )
        scaled_time = positions[:, :, None].float() * inv_timescales[None, None, :]
        sin_emb = torch.cat([scaled_time.sin(), scaled_time.cos()], dim=-1)
        pos_length = sin_emb.shape[1]
        sin_emb = self.pos_proj(sin_emb.to(dtype=self.pos_proj.weight.dtype)).float()
        sin_emb = sin_emb.view(1, pos_length, num_heads, head_dim).squeeze(0)
        sin_emb = sin_emb.permute(1, 2, 0)

        term_bd = torch.matmul(
            q.unsqueeze(-2),
            sin_emb.unsqueeze(0).unsqueeze(2).unsqueeze(3),
        ).squeeze(-2)
        if context + 1 > term_bd.shape[-1]:
            term_bd = F.pad(term_bd, (0, (context + 1) - term_bd.shape[-1]))
        term_bd = term_bd.reshape(batch, num_heads, num_blocks, block_size * (context + 1))
        term_bd = term_bd[:, :, :, : block_size * context]
        term_bd = term_bd.view(batch, num_heads, num_blocks, block_size, context)
        return term_ac + term_bd


class Gemma4AudioLocalAttention(nn.Module):
    def __init__(self, config: AudioConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.q_proj = _make_linear(config.hidden_size, config.hidden_size, bias=False, std=config.init_std)
        self.k_proj = _make_linear(config.hidden_size, config.hidden_size, bias=False, std=config.init_std)
        self.v_proj = _make_linear(config.hidden_size, config.hidden_size, bias=False, std=config.init_std)
        self.relative_position = Gemma4AudioRelativePosition(config)
        self.per_dim_scale = nn.Parameter(torch.ones(self.head_dim))
        self.softcap = 50.0
        self.query_scale = (self.head_dim**-0.5) / math.log(2.0)
        self.key_scale = math.log1p(math.e) / math.log(2.0)
        self.register_buffer(
            "causal_valid_mask",
            _causal_valid_mask(config, device=torch.device("cpu")),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        query = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).float()
        key = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).float()
        value = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).float()

        query = query * (self.query_scale * F.softplus(self.per_dim_scale).view(1, 1, 1, -1))
        key = key * self.key_scale

        key_blocks = _extract_block_context(
            key,
            self.config.chunk_size,
            max(0, self.config.left_context - 1),
            self.config.right_context,
        )
        query_blocks = _convert_to_block(query, self.config.chunk_size)
        logits = self.relative_position(query_blocks, key_blocks)
        logits = self.softcap * torch.tanh(logits / self.softcap)

        valid_mask = _extract_block_context(
            mask.float().unsqueeze(-1),
            self.config.chunk_size,
            max(0, self.config.left_context - 1),
            self.config.right_context,
            padding_value=0.0,
        ).squeeze(-1) > 0
        valid_mask = valid_mask[:, None, :, None, :]
        causal_valid_mask = self.causal_valid_mask.to(device=x.device)[None, None, None, :, :]
        logits = logits.masked_fill(~(valid_mask & causal_valid_mask), -1e9)
        probs = F.softmax(logits, dim=-1).float()

        value_blocks = _extract_block_context(
            value,
            self.config.chunk_size,
            max(0, self.config.left_context - 1),
            self.config.right_context,
        )
        probs = probs.permute(0, 2, 1, 3, 4)
        value_blocks = value_blocks.permute(0, 1, 3, 2, 4)
        context = torch.matmul(probs, value_blocks).permute(0, 1, 3, 2, 4)
        context = context.reshape(batch, -1, self.num_heads, self.head_dim)
        return context[:, :seq_len]


class Gemma4AudioAttentionBlock(nn.Module):
    def __init__(self, config: AudioConfig) -> None:
        super().__init__()
        self.gradient_clipping = config.gradient_clipping
        self.pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Gemma4AudioLocalAttention(config)
        self.post = _make_linear(config.hidden_size, config.hidden_size, bias=False, std=config.init_std)
        self.post_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.clamp(-self.gradient_clipping, self.gradient_clipping)
        x = self.pre_norm(x)
        x = self.attn(x, mask).reshape(x.shape[0], x.shape[1], -1)
        x = x.to(dtype=self.post.weight.dtype)
        x = self.post(x)
        x = x.clamp(-self.gradient_clipping, self.gradient_clipping)
        x = self.post_norm(x)
        return residual + x


class Gemma4AudioConformerLayer(nn.Module):
    def __init__(self, config: AudioConfig) -> None:
        super().__init__()
        self.ffn_start = Gemma4AudioFeedForward(config, residual_weight=0.5)
        self.attn = Gemma4AudioAttentionBlock(config)
        self.lightconv = Gemma4AudioLightConv(config)
        self.ffn_end = Gemma4AudioFeedForward(config, residual_weight=0.5)
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_clipping = config.gradient_clipping

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.ffn_start(x)
        x = self.attn(x, mask)
        x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        x = self.lightconv(x)
        x = self.ffn_end(x)
        x = x.clamp(-self.gradient_clipping, self.gradient_clipping)
        return self.final_norm(x)


class Gemma4AudioEncoder(nn.Module):
    def __init__(self, config: AudioConfig) -> None:
        super().__init__()
        self.config = config
        self.subsampler = Gemma4AudioSubsampler(config)
        self.layers = nn.ModuleList([Gemma4AudioConformerLayer(config) for _ in range(config.num_layers)])
        self.output_proj = init_linear_module(
            nn.Linear(config.hidden_size, config.output_size, bias=True),
            config.init_std,
        )

    def forward(self, features: torch.Tensor, feature_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, mask = self.subsampler(features, feature_mask)
        for layer in self.layers:
            hidden = layer(hidden, mask)

        if self.config.reduction_factor > 1:
            hidden = hidden[:, :: self.config.reduction_factor]
            mask = mask[:, :: self.config.reduction_factor]

        hidden = self.output_proj(hidden)
        pad_mask = ~mask
        hidden = hidden.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return hidden, pad_mask


class Gemma4AudioTower(nn.Module):
    def __init__(self, config: AudioConfig, text_hidden_size: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.encoder = Gemma4AudioEncoder(config)
        self.to_text = None
        self.to_text_norm = None
        if text_hidden_size is not None:
            self.to_text = init_linear_module(
                nn.Linear(config.output_size, text_hidden_size, bias=False),
                config.init_std,
            )
            norm_dim = config.output_size if config.projection_norm_before_text else text_hidden_size
            self.to_text_norm = RMSNorm(norm_dim, eps=config.rms_norm_eps, with_scale=False)

    def forward(self, features: torch.Tensor, feature_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(features, feature_mask)

    def encode_to_text(self, features: torch.Tensor, feature_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flatten_groups = False
        if features.ndim == 4:
            batch, num_clips = features.shape[:2]
            features = features.view(batch * num_clips, *features.shape[2:])
            feature_mask = feature_mask.view(batch * num_clips, feature_mask.shape[-1])
            flatten_groups = True
        else:
            batch = features.shape[0]
            num_clips = 1

        tokens, pad_mask = self.encoder(features, feature_mask)
        if self.to_text is not None:
            if self.config.projection_norm_before_text:
                tokens = self.to_text_norm(tokens)
                tokens = self.to_text(tokens)
            else:
                tokens = self.to_text(tokens)
                tokens = self.to_text_norm(tokens)

        valid_mask = ~pad_mask
        if flatten_groups:
            tokens = tokens.view(batch, num_clips, tokens.shape[1], tokens.shape[2]).flatten(1, 2)
            valid_mask = valid_mask.view(batch, num_clips, valid_mask.shape[1]).flatten(1, 2)
        return tokens, valid_mask
