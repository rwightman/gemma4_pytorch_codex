from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AttentionKind, TextConfig, create_kv_sharing_patterns
from .layers import (
    K_MASK,
    RMSNorm,
    ScaledEmbedding,
    apply_text_rope,
    create_sliding_mask,
    gelu_tanh,
    init_linear_module,
    repeat_kv,
    safe_token_ids,
)


@dataclass
class LayerKVCache:
    key: torch.Tensor
    value: torch.Tensor
    positions: torch.Tensor
    mask: torch.Tensor


@dataclass
class TextKVCache:
    layers: list[LayerKVCache | None]

    def _first_cache(self) -> LayerKVCache | None:
        for layer in self.layers:
            if layer is not None:
                return layer
        return None

    def valid_lengths(self) -> torch.Tensor:
        first = self._first_cache()
        if first is None:
            raise ValueError("Cannot query lengths from an empty KV cache.")
        return first.mask.long().sum(dim=-1)

    def key_positions(self) -> torch.Tensor:
        first = self._first_cache()
        if first is None:
            raise ValueError("Cannot query positions from an empty KV cache.")
        return first.positions

    def key_mask(self) -> torch.Tensor:
        first = self._first_cache()
        if first is None:
            raise ValueError("Cannot query masks from an empty KV cache.")
        return first.mask


def _make_linear(in_features: int, out_features: int, *, bias: bool, std: float) -> nn.Linear:
    return init_linear_module(nn.Linear(in_features, out_features, bias=bias), std)


def _append_to_cache(
        cache: LayerKVCache | None,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
) -> LayerKVCache:
    if cache is None:
        return LayerKVCache(key=key, value=value, positions=positions, mask=mask)
    return LayerKVCache(
        key=torch.cat([cache.key, key], dim=-2),
        value=torch.cat([cache.value, value], dim=-2),
        positions=torch.cat([cache.positions, positions], dim=-1),
        mask=torch.cat([cache.mask, mask], dim=-1),
    )


class Gemma4DenseMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, init_std: float) -> None:
        super().__init__()
        self.gate_proj = _make_linear(hidden_size, intermediate_size, bias=False, std=init_std)
        self.up_proj = _make_linear(hidden_size, intermediate_size, bias=False, std=init_std)
        self.down_proj = _make_linear(intermediate_size, hidden_size, bias=False, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(gelu_tanh(self.gate_proj(x)) * self.up_proj(x))


class Gemma4MoE(nn.Module):
    def __init__(self, hidden_size: int, expert_dim: int, num_experts: int, top_k: int, init_std: float) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_norm = RMSNorm(hidden_size, with_scale=False)
        self.router_scale = nn.Parameter(torch.ones(hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(num_experts))
        self.router = _make_linear(hidden_size, num_experts, bias=False, std=init_std)
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * expert_dim, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, expert_dim))
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=init_std)
        nn.init.normal_(self.down_proj, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router_input = self.router_norm(x)
        router_input = router_input * self.router_scale * (self.hidden_size**-0.5)
        router_logits = self.router(router_input).float()
        router_probs = F.softmax(router_logits, dim=-1)
        _, topk_index = torch.topk(router_logits, k=self.top_k, dim=-1)
        topk_weights = router_probs.gather(-1, topk_index)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights * self.per_expert_scale[topk_index]

        output = torch.zeros_like(x)
        active_experts = torch.unique(topk_index)
        for expert_idx in active_experts.tolist():
            match = topk_index == expert_idx
            token_idx, topk_pos = match.nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            states = x[token_idx]
            gate_up = F.linear(states, self.gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = gelu_tanh(gate) * up
            hidden = F.linear(hidden, self.down_proj[expert_idx])
            hidden = hidden * topk_weights[token_idx, topk_pos].unsqueeze(-1).to(hidden.dtype)
            output.index_add_(0, token_idx, hidden.to(output.dtype))

        return output


class Gemma4TextAttention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_full = self.layer_type == AttentionKind.FULL
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = (
            config.num_global_kv_heads
            if self.is_full and config.num_global_kv_heads is not None
            else config.num_kv_heads
        )
        self.key_dim = (
            config.global_head_dim
            if self.is_full and config.global_head_dim is not None
            else config.head_dim
        )
        self.k_eq_v = self.is_full and config.attention_k_eq_v_global
        self.rope_theta = config.global_rope_theta if self.is_full else config.local_rope_theta
        self.rope_scale = config.global_rope_scale if self.is_full else config.local_rope_scale
        self.rope_proportion = config.global_rope_proportion if self.is_full else config.local_rope_proportion
        self.sliding_window = config.sliding_window if self.layer_type == AttentionKind.SLIDING else None
        self.attn_logits_softcap = config.attn_logits_softcap
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        self.q_proj = _make_linear(
            config.hidden_size,
            self.num_heads * self.key_dim,
            bias=False,
            std=config.init_std,
        )
        self.k_proj = _make_linear(
            config.hidden_size,
            self.num_kv_heads * self.key_dim,
            bias=False,
            std=config.init_std,
        )
        self.v_proj = None
        if not self.k_eq_v:
            self.v_proj = _make_linear(
                config.hidden_size,
                self.num_kv_heads * self.key_dim,
                bias=False,
                std=config.init_std,
            )
        self.o_proj = _make_linear(self.num_heads * self.key_dim, config.hidden_size, bias=False, std=config.init_std)
        self.q_norm = RMSNorm(self.key_dim, eps=config.rms_norm_eps, with_scale=config.qk_norm_with_scale)
        self.k_norm = RMSNorm(self.key_dim, eps=config.rms_norm_eps, with_scale=config.qk_norm_with_scale)
        self.v_norm = RMSNorm(self.key_dim, eps=config.rms_norm_eps, with_scale=False)

    def forward(
            self,
            hidden_states: torch.Tensor,
            positions: torch.Tensor,
            attention_mask: torch.Tensor,
            query_mask: torch.Tensor,
            kv_cache: LayerKVCache | None = None,
            shared_kv: LayerKVCache | None = None,
    ) -> tuple[torch.Tensor, LayerKVCache]:
        batch_size, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.key_dim)
        query = self.q_norm(query)
        query = apply_text_rope(
            query,
            positions,
            base_theta=self.rope_theta,
            scale_factor=self.rope_scale,
            rope_proportion=self.rope_proportion,
        )
        query = query.transpose(1, 2)

        if shared_kv is None:
            key = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.key_dim)
            value = (
                key
                if self.k_eq_v
                else self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.key_dim)
            )
            key = self.k_norm(key)
            value = self.v_norm(value)
            key = apply_text_rope(
                key,
                positions,
                base_theta=self.rope_theta,
                scale_factor=self.rope_scale,
                rope_proportion=self.rope_proportion,
            )
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            layer_cache = _append_to_cache(kv_cache, key, value, positions, query_mask)
            key = layer_cache.key
            value = layer_cache.value
        else:
            layer_cache = shared_kv
            key = layer_cache.key
            value = layer_cache.value

        repeated_key = repeat_kv(key, self.num_key_value_groups)
        repeated_value = repeat_kv(value, self.num_key_value_groups)
        attn_logits = torch.matmul(query, repeated_key.transpose(-1, -2))
        if self.attn_logits_softcap is not None:
            attn_logits = torch.tanh(attn_logits / self.attn_logits_softcap) * self.attn_logits_softcap

        if self.sliding_window is not None:
            attention_mask = attention_mask & create_sliding_mask(
                positions,
                self.sliding_window,
                cache_positions=layer_cache.positions,
            )

        attn_logits = attn_logits.masked_fill(~attention_mask[:, None, :, :], K_MASK)
        attn_probs = F.softmax(attn_logits.float(), dim=-1).to(dtype=query.dtype)
        attn_output = torch.matmul(attn_probs, repeated_value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.key_dim)
        return self.o_proj(attn_output), layer_cache


class Gemma4TextBlock(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int, ffn_hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.enable_moe = config.enable_moe
        self.use_post_attn_norm = config.use_post_attn_norm
        self.use_post_ffn_norm = config.use_post_ffn_norm
        self.per_layer_input_dim = config.per_layer_input_dim

        self.attn = Gemma4TextAttention(config, layer_idx)
        self.pre_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attn_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if config.use_post_attn_norm else None
        )

        if self.enable_moe:
            self.pre_ffn2_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.mlp2 = Gemma4DenseMLP(config.hidden_size, ffn_hidden_size, config.init_std)
            self.post_ffn2_norm = (
                RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if config.use_post_ffn_norm else None
            )
            self.pre_ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.moe = Gemma4MoE(
                config.hidden_size,
                config.expert_dim,
                config.num_experts,
                config.top_k_experts,
                config.init_std,
            )
            self.post_ffn1_norm = (
                RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if config.use_post_ffn_norm else None
            )
        else:
            self.pre_ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.mlp = Gemma4DenseMLP(config.hidden_size, ffn_hidden_size, config.init_std)

        self.post_ffn_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if config.use_post_ffn_norm else None
        )
        self.layer_scalar = nn.Parameter(torch.ones(1))

        if self.per_layer_input_dim:
            self.per_layer_input_gate = _make_linear(
                config.hidden_size,
                config.per_layer_input_dim,
                bias=False,
                std=config.init_std,
            )
            self.per_layer_projection = _make_linear(
                config.per_layer_input_dim,
                config.hidden_size,
                bias=False,
                std=config.init_std,
            )
            self.post_per_layer_input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _forward_dense(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.pre_ffn_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.post_ffn_norm is not None:
            hidden_states = self.post_ffn_norm(hidden_states)
        return hidden_states

    def _forward_moe(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dense_branch = self.pre_ffn2_norm(hidden_states)
        dense_branch = self.mlp2(dense_branch)
        if self.post_ffn2_norm is not None:
            dense_branch = self.post_ffn2_norm(dense_branch)

        moe_branch = self.pre_ffn_norm(hidden_states)
        moe_branch = self.moe(moe_branch.view(-1, moe_branch.shape[-1])).view_as(moe_branch)
        if self.post_ffn1_norm is not None:
            moe_branch = self.post_ffn1_norm(moe_branch)

        hidden_states = dense_branch + moe_branch
        if self.post_ffn_norm is not None:
            hidden_states = self.post_ffn_norm(hidden_states)
        return hidden_states

    def forward(
            self,
            hidden_states: torch.Tensor,
            positions: torch.Tensor,
            attention_mask: torch.Tensor,
            query_mask: torch.Tensor,
            per_layer_input: torch.Tensor | None = None,
            kv_cache: LayerKVCache | None = None,
            shared_kv: LayerKVCache | None = None,
    ) -> tuple[torch.Tensor, LayerKVCache]:
        residual = hidden_states
        hidden_states = self.pre_attn_norm(hidden_states)
        hidden_states, kv_to_share = self.attn(
            hidden_states,
            positions,
            attention_mask,
            query_mask=query_mask,
            kv_cache=kv_cache,
            shared_kv=shared_kv,
        )
        if self.post_attn_norm is not None:
            hidden_states = self.post_attn_norm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        if self.enable_moe:
            hidden_states = self._forward_moe(hidden_states)
        else:
            hidden_states = self._forward_dense(hidden_states)
        hidden_states = residual + hidden_states

        if per_layer_input is not None:
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = gelu_tanh(hidden_states) * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = hidden_states * self.layer_scalar
        return hidden_states, kv_to_share


class Gemma4TextTower(nn.Module):
    def __init__(self, config: TextConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = ScaledEmbedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
            init_std=config.init_std,
        )
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.kv_sharing_patterns = create_kv_sharing_patterns(
            config.kv_sharing,
            config.num_layers,
            config.layer_types,
        )
        self.layers = nn.ModuleList(
            [
                Gemma4TextBlock(config, idx, self._ffn_hidden_size(idx))
                for idx in range(config.num_layers)
            ]
        )

        if config.per_layer_input_dim:
            self.per_layer_token_embedding = nn.Parameter(
                torch.empty(config.vocab_size, config.num_layers, config.per_layer_input_dim)
            )
            nn.init.normal_(self.per_layer_token_embedding, mean=0.0, std=config.init_std)
            self.per_layer_model_projection = _make_linear(
                config.hidden_size,
                config.num_layers * config.per_layer_input_dim,
                bias=False,
                std=config.init_std,
            )
            with torch.no_grad():
                self.per_layer_model_projection.weight.mul_(config.hidden_size**-0.5)
            self.per_layer_projection_norm = RMSNorm(config.per_layer_input_dim, eps=config.rms_norm_eps)

    def _ffn_hidden_size(self, layer_idx: int) -> int:
        if self.config.enable_moe:
            return self.config.moe_dense_hidden_size
        if (
            self.config.override_kv_shared_ffn_hidden is not None
            and self.kv_sharing_patterns[layer_idx] != layer_idx
        ):
            return self.config.override_kv_shared_ffn_hidden
        return self.config.intermediate_size

    @property
    def weight(self) -> torch.Tensor:
        return self.token_embedding.weight

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(safe_token_ids(input_ids, self.config.vocab_size))

    def project_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.token_embedding.weight)

    def _build_per_layer_inputs(
            self,
            input_ids: torch.Tensor,
            hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.config.per_layer_input_dim:
            return None

        safe_ids = safe_token_ids(input_ids, self.config.vocab_size)
        model_side = self.per_layer_model_projection(hidden_states)
        model_side = model_side.view(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.config.num_layers,
            self.config.per_layer_input_dim,
        )
        model_side = self.per_layer_projection_norm(model_side)

        token_side = self.per_layer_token_embedding[safe_ids]
        token_side = token_side * math.sqrt(self.config.per_layer_input_dim)
        return (model_side + token_side) * (2.0**-0.5)

    def forward(
            self,
            input_ids: torch.Tensor,
            *,
            inputs_embeds: torch.Tensor | None = None,
            position_ids: torch.Tensor,
            query_mask: torch.Tensor | None = None,
            full_attention_mask: torch.Tensor,
            sliding_attention_mask: torch.Tensor | None = None,
            kv_cache: TextKVCache | None = None,
            return_kv_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, TextKVCache]:
        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if query_mask is None:
            query_mask = input_ids != self.config.pad_token_id
        per_layer_inputs = self._build_per_layer_inputs(input_ids, hidden_states)
        new_cache_layers: list[LayerKVCache | None] = [None] * self.config.num_layers
        current_kv: dict[int, LayerKVCache] = {}

        for layer_idx, layer in enumerate(self.layers):
            source_layer = self.kv_sharing_patterns[layer_idx]
            kv_to_reuse = current_kv.get(source_layer) if source_layer != layer_idx else None
            attention_mask = (
                sliding_attention_mask
                if sliding_attention_mask is not None and self.config.layer_types[layer_idx] == AttentionKind.SLIDING
                else full_attention_mask
            )
            per_layer_input = None if per_layer_inputs is None else per_layer_inputs[:, :, layer_idx, :]
            hidden_states, kv_to_share = layer(
                hidden_states,
                position_ids,
                attention_mask,
                query_mask,
                per_layer_input=per_layer_input,
                kv_cache=None if kv_cache is None else kv_cache.layers[layer_idx],
                shared_kv=kv_to_reuse,
            )
            current_kv[layer_idx] = kv_to_share
            new_cache_layers[layer_idx] = kv_to_share

        hidden_states = self.final_norm(hidden_states)
        if return_kv_cache:
            return hidden_states, TextKVCache(layers=new_cache_layers)
        return hidden_states
