from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AudioConfig
from .layers import (
    ClippedLinear,
    RMSNorm,
    gelu_tanh,
)
from .module_utils import InitModule, InitContext, factory_kwargs, resolve_residual_init_std


def _make_linear(
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
) -> nn.Module:
    return ClippedLinear(in_features, out_features, bias=bias, device=device, dtype=dtype)


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


class Gemma4AudioSubsampler(InitModule):
    def __init__(
            self,
            config: AudioConfig,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.init_std = config.init_std
        c0, c1 = config.subsampling_channels
        subsampled_bins = ((config.num_mel_bins + 1) // 2 + 1) // 2
        dd = factory_kwargs(device, dtype)
        self.conv0 = nn.Conv2d(1, c0, kernel_size=3, stride=2, padding=1, bias=False, **dd)
        self.conv1 = nn.Conv2d(c0, c1, kernel_size=3, stride=2, padding=1, bias=False, **dd)
        self.norm0 = nn.LayerNorm(c0, **dd)
        self.norm1 = nn.LayerNorm(c1, **dd)
        self.output_proj = nn.Linear(c1 * subsampled_bins, config.hidden_size, bias=False, **dd)

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

    def _init_weights(self, ctx: InitContext) -> None:
        nn.init.normal_(self.conv0.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        nn.init.normal_(self.conv1.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.norm0.weight is not None:
            nn.init.ones_(self.norm0.weight)
        if self.norm0.bias is not None:
            nn.init.zeros_(self.norm0.bias)
        if self.norm1.weight is not None:
            nn.init.ones_(self.norm1.weight)
        if self.norm1.bias is not None:
            nn.init.zeros_(self.norm1.bias)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)


class Gemma4AudioFeedForward(InitModule):
    def __init__(
            self,
            config: AudioConfig,
            residual_weight: float = 0.5,
            residual_init_std: float | None = None,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.init_std = config.init_std
        self.residual_init_std = config.init_std if residual_init_std is None else residual_init_std
        self.residual_weight = residual_weight
        self.gradient_clipping = config.gradient_clipping
        dd = factory_kwargs(device, dtype)
        self.pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, **dd)
        self.ffn1 = _make_linear(config.hidden_size, config.hidden_size * 4, bias=False, **dd)
        self.ffn2 = _make_linear(config.hidden_size * 4, config.hidden_size, bias=False, **dd)
        self.post_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.clamp(-self.gradient_clipping, self.gradient_clipping)
        x = self.pre_norm(x)
        x = F.silu(self.ffn1(x))
        x = self.ffn2(x)
        x = x.clamp(-self.gradient_clipping, self.gradient_clipping)
        x = self.post_norm(x)
        return residual + x * self.residual_weight

    def _init_weights(self, ctx: InitContext) -> None:
        if self.pre_norm.weight is not None:
            nn.init.ones_(self.pre_norm.weight)
        nn.init.normal_(self.ffn1.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.ffn1.bias is not None:
            nn.init.zeros_(self.ffn1.bias)
        nn.init.normal_(self.ffn2.weight, mean=0.0, std=self.residual_init_std, generator=ctx.generator)
        if self.ffn2.bias is not None:
            nn.init.zeros_(self.ffn2.bias)
        if self.post_norm.weight is not None:
            nn.init.ones_(self.post_norm.weight)


class Gemma4AudioLightConv(InitModule):
    def __init__(
            self,
            config: AudioConfig,
            residual_init_std: float | None = None,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.init_std = config.init_std
        self.residual_init_std = config.init_std if residual_init_std is None else residual_init_std
        dd = factory_kwargs(device, dtype)
        self.pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, **dd)
        self.linear_start = _make_linear(config.hidden_size, 2 * config.hidden_size, bias=False, **dd)
        self.depthwise = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.conv_kernel_size,
            groups=config.hidden_size,
            bias=False,
            **dd,
        )
        self.conv_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, **dd)
        self.linear_end = _make_linear(config.hidden_size, config.hidden_size, bias=False, **dd)
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

    def _init_weights(self, ctx: InitContext) -> None:
        if self.pre_norm.weight is not None:
            nn.init.ones_(self.pre_norm.weight)
        nn.init.normal_(self.linear_start.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.linear_start.bias is not None:
            nn.init.zeros_(self.linear_start.bias)
        nn.init.normal_(self.depthwise.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.depthwise.bias is not None:
            nn.init.zeros_(self.depthwise.bias)
        if self.conv_norm.weight is not None:
            nn.init.ones_(self.conv_norm.weight)
        nn.init.normal_(self.linear_end.weight, mean=0.0, std=self.residual_init_std, generator=ctx.generator)
        if self.linear_end.bias is not None:
            nn.init.zeros_(self.linear_end.bias)


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


class Gemma4AudioRelativePosition(InitModule):
    def __init__(
            self,
            config: AudioConfig,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads
        self.left_context = config.left_context
        self.right_context = config.right_context
        self.pos_proj = nn.Linear(
            config.hidden_size,
            config.num_heads * self.head_dim,
            bias=False,
            **factory_kwargs(device, dtype),
        )

    def _init_weights(self, ctx: InitContext) -> None:
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


class Gemma4AudioLocalAttention(InitModule):
    def __init__(
            self,
            config: AudioConfig,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.init_std = config.init_std
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        dd = factory_kwargs(device, dtype)
        self.q_proj = _make_linear(config.hidden_size, config.hidden_size, bias=False, **dd)
        self.k_proj = _make_linear(config.hidden_size, config.hidden_size, bias=False, **dd)
        self.v_proj = _make_linear(config.hidden_size, config.hidden_size, bias=False, **dd)
        self.relative_position = Gemma4AudioRelativePosition(config, **dd)
        self.per_dim_scale = nn.Parameter(torch.ones(self.head_dim, **dd))
        self.softcap = 50.0
        self.query_scale = (self.head_dim**-0.5) / math.log(2.0)
        self.key_scale = math.log1p(math.e) / math.log(2.0)
        context = config.chunk_size + max(0, config.left_context - 1) + config.right_context
        self.register_buffer(
            "causal_valid_mask",
            torch.empty(config.chunk_size, context, dtype=torch.bool, device=device),
            persistent=False,
        )
        self._init_non_persistent_buffers()

    def _build_causal_valid_mask(self) -> torch.Tensor:
        return _causal_valid_mask(self.config, device=self.per_dim_scale.device)

    def _init_non_persistent_buffers(self) -> None:
        mask = self._build_causal_valid_mask()
        if self.causal_valid_mask.is_meta or mask.is_meta:
            self.causal_valid_mask = mask
        else:
            with torch.no_grad():
                self.causal_valid_mask.copy_(mask)

    def _init_weights(self, ctx: InitContext) -> None:
        self._init_non_persistent_buffers()
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.k_proj.bias is not None:
            nn.init.zeros_(self.k_proj.bias)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.v_proj.bias is not None:
            nn.init.zeros_(self.v_proj.bias)
        nn.init.ones_(self.per_dim_scale)

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


class Gemma4AudioAttentionBlock(InitModule):
    def __init__(
            self,
            config: AudioConfig,
            residual_init_std: float | None = None,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.init_std = config.init_std
        self.residual_init_std = config.init_std if residual_init_std is None else residual_init_std
        self.gradient_clipping = config.gradient_clipping
        dd = factory_kwargs(device, dtype)
        self.pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, **dd)
        self.attn = Gemma4AudioLocalAttention(config, **dd)
        self.post = _make_linear(config.hidden_size, config.hidden_size, bias=False, **dd)
        self.post_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, **dd)

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

    def _init_weights(self, ctx: InitContext) -> None:
        if self.pre_norm.weight is not None:
            nn.init.ones_(self.pre_norm.weight)
        nn.init.normal_(self.post.weight, mean=0.0, std=self.residual_init_std, generator=ctx.generator)
        if self.post.bias is not None:
            nn.init.zeros_(self.post.bias)
        if self.post_norm.weight is not None:
            nn.init.ones_(self.post_norm.weight)


class Gemma4AudioConformerLayer(InitModule):
    def __init__(
            self,
            config: AudioConfig,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        dd = factory_kwargs(device, dtype)
        residual_init_std = resolve_residual_init_std(
            config.init_std,
            config.residual_init_std,
            config.use_depth_scaled_residual_init,
            config.num_layers,
        )
        self.ffn_start = Gemma4AudioFeedForward(
            config,
            residual_weight=0.5,
            residual_init_std=residual_init_std,
            **dd,
        )
        self.attn = Gemma4AudioAttentionBlock(
            config,
            residual_init_std=residual_init_std,
            **dd,
        )
        self.lightconv = Gemma4AudioLightConv(
            config,
            residual_init_std=residual_init_std,
            **dd,
        )
        self.ffn_end = Gemma4AudioFeedForward(
            config,
            residual_weight=0.5,
            residual_init_std=residual_init_std,
            **dd,
        )
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, **dd)
        self.gradient_clipping = config.gradient_clipping

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.ffn_start(x)
        x = self.attn(x, mask)
        x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        x = self.lightconv(x)
        x = self.ffn_end(x)
        x = x.clamp(-self.gradient_clipping, self.gradient_clipping)
        return self.final_norm(x)

    def _init_weights(self, ctx: InitContext) -> None:
        if self.final_norm.weight is not None:
            nn.init.ones_(self.final_norm.weight)


class Gemma4AudioEncoder(InitModule):
    def __init__(
            self,
            config: AudioConfig,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.init_std = config.init_std
        dd = factory_kwargs(device, dtype)
        self.subsampler = Gemma4AudioSubsampler(config, **dd)
        self.layers = nn.ModuleList([
            Gemma4AudioConformerLayer(config, **dd) for _ in range(config.num_layers)
        ])
        self.output_proj = nn.Linear(
            config.hidden_size,
            config.output_size,
            bias=True,
            **dd,
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

    def _init_weights(self, ctx: InitContext) -> None:
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)


class Gemma4AudioTower(InitModule):
    def __init__(
            self,
            config: AudioConfig,
            text_hidden_size: int | None = None,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        dd = factory_kwargs(device, dtype)
        self.config = config
        self.init_std = config.init_std
        self.encoder = Gemma4AudioEncoder(config, **dd)
        self.to_text = None
        self.to_text_norm = None
        if text_hidden_size is not None:
            self.to_text = nn.Linear(
                config.output_size,
                text_hidden_size,
                bias=False,
                **dd,
            )
            norm_dim = config.output_size if config.projection_norm_before_text else text_hidden_size
            self.to_text_norm = RMSNorm(norm_dim, eps=config.rms_norm_eps, with_scale=False, **dd)

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

    def _init_weights(self, ctx: InitContext) -> None:
        if self.to_text is not None:
            nn.init.normal_(self.to_text.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
            if self.to_text.bias is not None:
                nn.init.zeros_(self.to_text.bias)
        if self.to_text_norm is not None and self.to_text_norm.weight is not None:
            nn.init.ones_(self.to_text_norm.weight)
