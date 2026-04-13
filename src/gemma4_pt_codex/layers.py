from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module_utils import factory_kwargs


K_MASK = -2.3819763e38


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


def safe_token_ids(token_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    valid = (token_ids >= 0) & (token_ids < vocab_size)
    return torch.where(valid, token_ids, token_ids.new_zeros(()))


def build_positions_from_mask(mask: torch.Tensor) -> torch.Tensor:
    positions = mask.to(torch.long).cumsum(dim=-1) - 1
    positions = positions.clamp_min_(0)
    return positions.masked_fill(~mask, 0)


def make_causal_mask(input_mask: torch.Tensor) -> torch.Tensor:
    seq_len = input_mask.shape[-1]
    causal = torch.tril(torch.ones(seq_len, seq_len, device=input_mask.device, dtype=torch.bool))
    return input_mask[:, None, :] & causal.unsqueeze(0)


def _make_block_mask_indices(bidirectional_mask: torch.Tensor) -> torch.Tensor:
    padded = F.pad(bidirectional_mask.to(torch.long), (1, 0), value=0)
    boundary = padded[..., 1:] > padded[..., :-1]
    numbered = boundary.to(torch.long).cumsum(dim=-1)
    return bidirectional_mask.to(torch.long) * numbered


def make_causal_bidirectional_mask(
        input_mask: torch.Tensor,
        bidirectional_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    attn_mask = make_causal_mask(input_mask)
    if bidirectional_mask is None:
        return attn_mask

    q_blocks = _make_block_mask_indices(bidirectional_mask)
    kv_blocks = q_blocks
    return attn_mask | (
        (kv_blocks[:, None, :] == q_blocks[..., None]) & (q_blocks[..., None] > 0)
    )


def create_sliding_mask(
        positions: torch.Tensor,
        sliding_window: int,
        cache_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    if cache_positions is None:
        cache_positions = positions
    cache_positions = cache_positions[:, None, :]
    positions = positions[:, :, None]
    return (cache_positions > positions - sliding_window) & (
        cache_positions < positions + sliding_window
    )


def repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch,
        num_kv_heads,
        repeats,
        seq_len,
        head_dim,
    )
    return hidden_states.reshape(batch, num_kv_heads * repeats, seq_len, head_dim)


def merge_flat_embeddings(
        text_embeddings: torch.Tensor,
        multimodal_embeddings: torch.Tensor,
        target_mask: torch.Tensor,
        multimodal_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    merged = text_embeddings.clone()
    if multimodal_mask is None:
        multimodal_mask = torch.ones(
            multimodal_embeddings.shape[:2],
            dtype=torch.bool,
            device=multimodal_embeddings.device,
        )

    for batch_idx in range(text_embeddings.shape[0]):
        target_positions = target_mask[batch_idx].nonzero(as_tuple=False).squeeze(-1)
        source_positions = multimodal_mask[batch_idx].nonzero(as_tuple=False).squeeze(-1)
        if target_positions.numel() != source_positions.numel():
            raise ValueError(
                "Mismatch between placeholder count and multimodal token count: "
                f"{target_positions.numel()} vs {source_positions.numel()}."
            )
        if target_positions.numel() == 0:
            continue
        merged[batch_idx, target_positions] = multimodal_embeddings[batch_idx, source_positions]
    return merged


class RMSNorm(nn.RMSNorm):
    def __init__(
            self,
            dim: int,
            eps: float = 1e-6,
            with_scale: bool = True,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            dim,
            eps=eps,
            elementwise_affine=with_scale,
            **factory_kwargs(device, dtype),
        )


class VisionRMSNorm(RMSNorm):
    def __init__(
            self,
            dim: int,
            eps: float = 1e-6,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(dim, eps=eps, with_scale=True, device=device, dtype=dtype)

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.zeros_(self.weight)

class ClippedLinear(nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = False,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            **factory_kwargs(device, dtype),
        )
        scalar_dd = factory_kwargs(device, dtype)
        self.register_buffer("input_min", torch.tensor(float("-inf"), **scalar_dd))
        self.register_buffer("input_max", torch.tensor(float("inf"), **scalar_dd))
        self.register_buffer("output_min", torch.tensor(float("-inf"), **scalar_dd))
        self.register_buffer("output_max", torch.tensor(float("inf"), **scalar_dd))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(self.input_min.item(), self.input_max.item())
        x = F.linear(x, self.weight, self.bias)
        return x.clamp(self.output_min.item(), self.output_max.item())


def _rope_half_rotation(
        first_half: torch.Tensor,
        second_half: torch.Tensor,
        positions: torch.Tensor,
        base_theta: float,
        scale_factor: float,
        head_dim: int,
        rotary_half: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rotary_half == 0:
        return first_half, second_half

    freq_exponents = (2.0 / head_dim) * torch.arange(
        rotary_half,
        device=first_half.device,
        dtype=torch.float32,
    )
    timescale = base_theta**freq_exponents
    sinusoid = positions.to(torch.float32).unsqueeze(-1) / timescale
    sinusoid = sinusoid / scale_factor
    sin = torch.sin(sinusoid).unsqueeze(-2)
    cos = torch.cos(sinusoid).unsqueeze(-2)

    first_rot = first_half[..., :rotary_half]
    second_rot = second_half[..., :rotary_half]
    first_pass = first_half[..., rotary_half:]
    second_pass = second_half[..., rotary_half:]

    rotated_first = first_rot * cos - second_rot * sin
    rotated_second = second_rot * cos + first_rot * sin

    return (
        torch.cat([rotated_first, first_pass], dim=-1),
        torch.cat([rotated_second, second_pass], dim=-1),
    )


def apply_text_rope(
        x: torch.Tensor,
        positions: torch.Tensor,
        *,
        base_theta: float,
        scale_factor: float = 1.0,
        rope_proportion: float | None = 1.0,
) -> torch.Tensor:
    head_dim = x.shape[-1]
    half_dim = head_dim // 2
    rotary_half = int(((1.0 if rope_proportion is None else rope_proportion) * head_dim) // 2)
    rotary_half = max(0, min(rotary_half, half_dim))
    first_half, second_half = x.split(half_dim, dim=-1)
    first_half, second_half = _rope_half_rotation(
        first_half,
        second_half,
        positions,
        base_theta,
        scale_factor,
        head_dim,
        rotary_half,
    )
    return torch.cat([first_half, second_half], dim=-1).to(dtype=x.dtype)


def apply_multidim_rope(
        x: torch.Tensor,
        positions_xy: torch.Tensor,
        *,
        base_theta: float,
        scale_factor: float = 1.0,
        rotary_fraction: float | None = None,
) -> torch.Tensor:
    ndim = positions_xy.shape[-1]
    num_input_channels = x.shape[-1]
    num_rotated_channels = (
        num_input_channels
        if rotary_fraction is None
        else int(round(num_input_channels * rotary_fraction))
    )
    per_dim_channels = 2 * (num_rotated_channels // (2 * ndim))
    if per_dim_channels <= 0:
        raise ValueError(
            f"Not enough channels for {ndim}D RoPE: channels={num_input_channels}, ndim={ndim}."
        )

    split_sizes = [per_dim_channels] * ndim
    remainder = num_input_channels - per_dim_channels * ndim
    if remainder > 0:
        split_sizes.append(remainder)

    parts = list(torch.split(x, split_sizes, dim=-1))
    out_parts: list[torch.Tensor] = []
    for dim_idx in range(ndim):
        out_parts.append(
            apply_text_rope(
                parts[dim_idx],
                positions_xy[..., dim_idx],
                base_theta=base_theta,
                scale_factor=scale_factor,
                rope_proportion=1.0,
            )
        )
    if remainder > 0:
        out_parts.append(parts[-1])
    return torch.cat(out_parts, dim=-1).to(dtype=x.dtype)
