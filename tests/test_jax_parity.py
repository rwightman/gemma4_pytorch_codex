from __future__ import annotations

import sys
import types
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
flax = pytest.importorskip("flax")
import flax.linen as nn
from flax.traverse_util import flatten_dict

from gemma4_pytorch_codex.config import AttentionKind, Gemma4Config, TextConfig, VisionConfig
from gemma4_pytorch_codex.layers import make_causal_mask
from gemma4_pytorch_codex.model import Gemma4Model
from gemma4_pytorch_codex.text import Gemma4TextTower
from gemma4_pytorch_codex.vision import Gemma4VisionEncoder


ROOT = Path(__file__).resolve().parents[2]


def make_parity_text_config() -> TextConfig:
    return TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_layers=4,
        num_heads=4,
        head_dim=8,
        num_kv_heads=2,
        global_head_dim=8,
        layer_types=(
            AttentionKind.SLIDING,
            AttentionKind.SLIDING,
            AttentionKind.SLIDING,
            AttentionKind.FULL,
        ),
        sliding_window=8,
        per_layer_input_dim=8,
        kv_sharing=None,
        final_logit_softcap=30.0,
    )


def make_parity_vision_config() -> VisionConfig:
    return VisionConfig(
        hidden_size=24,
        intermediate_size=48,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        head_dim=6,
        patch_size=2,
        position_embedding_size=8,
        output_length=4,
        pooling_kernel_size=1,
        use_clipped_linears=False,
    )


def _install_original_gemma_stubs() -> None:
    if "kauldron" not in sys.modules:
        class _TypeStub:
            def __getitem__(self, item):
                return object

        def _typechecked(fn=None, **kwargs):
            if fn is None:
                return lambda real_fn: real_fn
            return fn

        class _Identity(nn.Module):
            @nn.compact
            def __call__(self, x):
                return x

        kauldron = types.ModuleType("kauldron")
        kd = types.ModuleType("kauldron.kd")
        kd.nn = types.SimpleNamespace(Identity=_Identity)
        ktyping = types.ModuleType("kauldron.ktyping")
        for name in ("Array", "Bool", "Float", "Int", "UInt8"):
            setattr(ktyping, name, _TypeStub())
        ktyping.typechecked = _typechecked
        kauldron.kd = kd
        kauldron.ktyping = ktyping
        sys.modules["kauldron"] = kauldron
        sys.modules["kauldron.kd"] = kd
        sys.modules["kauldron.ktyping"] = ktyping

    if "einops" not in sys.modules:
        einops = types.ModuleType("einops")

        def rearrange(x, pattern, **axes):
            if pattern == "... (h p) (w q) c -> ... (h w) (p q c)":
                p = axes["p"]
                q = axes["q"]
                *batch, hp, wq, channels = x.shape
                h = hp // p
                w = wq // q
                perm = [*range(len(batch)), len(batch), len(batch) + 2, len(batch) + 1, len(batch) + 3, len(batch) + 4]
                x = x.reshape(*batch, h, p, w, q, channels)
                return x.transpose(perm).reshape(*batch, h * w, p * q * channels)
            if pattern == "y x c -> (y x) c":
                y, x_dim, channels = x.shape
                return x.reshape(y * x_dim, channels)
            raise NotImplementedError(pattern)

        einops.rearrange = rearrange
        sys.modules["einops"] = einops


@lru_cache(maxsize=1)
def _load_original_gemma_modules():
    _install_original_gemma_stubs()
    gemma_root = ROOT / "gemma"
    if str(gemma_root) not in sys.path:
        sys.path.insert(0, str(gemma_root))

    from gemma.gm.nn.gemma4 import _layers as jax_base_layers
    from gemma.gm.nn.gemma4 import _modules as jax_text_modules
    from gemma.gm.nn.gemma4.vision import _encoder as jax_vision_encoder

    return jax_base_layers, jax_text_modules, jax_vision_encoder


jax_base_layers, jax_text_modules, jax_vision_encoder = _load_original_gemma_modules()


class JaxTinyTextTower(nn.Module):
    @nn.compact
    def __call__(self, tokens, positions, full_mask, sliding_mask=None, cache=None):
        embedder = jax_text_modules.Embedder(
            vocab_size=64,
            embed_dim=32,
            num_layers=4,
            per_layer_input_dim=8,
        )
        hidden_states = embedder.encode(tokens)
        per_layer_inputs = embedder.encode_per_layer_input(hidden_states, tokens)
        new_cache = {}
        layer_types = (
            jax_text_modules.AttentionType.LOCAL_SLIDING,
            jax_text_modules.AttentionType.LOCAL_SLIDING,
            jax_text_modules.AttentionType.LOCAL_SLIDING,
            jax_text_modules.AttentionType.GLOBAL,
        )

        for layer_idx, attn_type in enumerate(layer_types):
            block = jax_text_modules.Block(
                name=f"layer_{layer_idx}",
                num_heads=4,
                num_kv_heads=2,
                embed_dim=32,
                head_dim=8,
                hidden_dim=64,
                sliding_window_size=8,
                use_post_attn_norm=True,
                use_post_ffw_norm=True,
                attn_type=attn_type,
                qk_norm_with_scale=True,
                num_global_kv_heads=None,
                global_key_size=8,
                k_eq_v_global=False,
                global_rope_proportion=0.25,
                local_rope_proportion=1.0,
                rope_base_frequency=10_000
                if attn_type == jax_text_modules.AttentionType.LOCAL_SLIDING
                else 1_000_000,
                rope_scale_factor=1.0,
                per_layer_input_dim=8,
                enable_moe=False,
            )
            layer_cache = None if cache is None else cache[f"layer_{layer_idx}"]
            attn_mask = (
                sliding_mask
                if attn_type == jax_text_modules.AttentionType.LOCAL_SLIDING and sliding_mask is not None
                else full_mask
            )
            layer_cache, hidden_states = block(
                hidden_states,
                positions,
                layer_cache,
                attn_mask,
                per_layer_inputs[..., layer_idx, :],
                None,
            )
            new_cache[f"layer_{layer_idx}"] = layer_cache

        hidden_states = jax_base_layers.RMSNorm(name="final_norm")(hidden_states)
        return hidden_states, new_cache


def _flat_params(params) -> dict[str, object]:
    return {"/".join(path): value for path, value in flatten_dict(params).items()}


def _torch_from_jax(value) -> torch.Tensor:
    return torch.from_numpy(np.array(value, copy=True))


def _copy_text_params(flat_params: dict[str, object], tower: Gemma4TextTower) -> None:
    with torch.no_grad():
        tower.token_embedding.weight.copy_(_torch_from_jax(flat_params["Embedder_0/input_embedding"]))
        tower.per_layer_token_embedding.copy_(_torch_from_jax(flat_params["Embedder_0/per_layer_embeddings"]))

        projection = _torch_from_jax(flat_params["Embedder_0/per_layer_model_projection/w"])
        projection = projection.permute(1, 2, 0).reshape(4 * 8, 32)
        projection = projection * (32**-0.5)
        tower.per_layer_model_projection.weight.copy_(projection)
        tower.per_layer_projection_norm.weight.copy_(
            _torch_from_jax(flat_params["Embedder_0/per_layer_projection_norm/scale"])
        )
        tower.final_norm.weight.copy_(_torch_from_jax(flat_params["final_norm/scale"]))

        for layer_idx, layer in enumerate(tower.layers):
            prefix = f"layer_{layer_idx}"
            layer.layer_scalar.copy_(_torch_from_jax(flat_params[f"{prefix}/skip_scale"]))
            layer.pre_attn_norm.weight.copy_(_torch_from_jax(flat_params[f"{prefix}/pre_attention_norm/scale"]))
            layer.attn.q_proj.weight.copy_(
                _torch_from_jax(flat_params[f"{prefix}/attn/q_einsum/w"]).permute(0, 2, 1).reshape(4 * 8, 32)
            )

            kv = _torch_from_jax(flat_params[f"{prefix}/attn/kv_einsum/w"])
            layer.attn.k_proj.weight.copy_(kv[0].permute(0, 2, 1).reshape(2 * 8, 32))
            layer.attn.v_proj.weight.copy_(kv[1].permute(0, 2, 1).reshape(2 * 8, 32))
            layer.attn.o_proj.weight.copy_(
                _torch_from_jax(flat_params[f"{prefix}/attn/attn_vec_einsum/w"])
                .permute(2, 0, 1)
                .reshape(32, 4 * 8)
            )
            layer.attn.q_norm.weight.copy_(_torch_from_jax(flat_params[f"{prefix}/attn/query_norm/scale"]))
            layer.attn.k_norm.weight.copy_(_torch_from_jax(flat_params[f"{prefix}/attn/key_norm/scale"]))
            layer.post_attn_norm.weight.copy_(_torch_from_jax(flat_params[f"{prefix}/post_attention_norm/scale"]))
            layer.pre_ffn_norm.weight.copy_(_torch_from_jax(flat_params[f"{prefix}/pre_ffw_norm/scale"]))

            gating = _torch_from_jax(flat_params[f"{prefix}/mlp/gating_einsum"])
            layer.mlp.gate_proj.weight.copy_(gating[0])
            layer.mlp.up_proj.weight.copy_(gating[1])
            layer.mlp.down_proj.weight.copy_(_torch_from_jax(flat_params[f"{prefix}/mlp/linear"]).transpose(0, 1))
            layer.post_ffn_norm.weight.copy_(_torch_from_jax(flat_params[f"{prefix}/post_ffw_norm/scale"]))

            layer.per_layer_input_gate.weight.copy_(
                _torch_from_jax(flat_params[f"{prefix}/per_layer_input_gate/w"]).transpose(0, 1)
            )
            layer.per_layer_projection.weight.copy_(
                _torch_from_jax(flat_params[f"{prefix}/per_layer_projection/w"]).transpose(0, 1)
            )
            layer.post_per_layer_input_norm.weight.copy_(
                _torch_from_jax(flat_params[f"{prefix}/post_per_layer_input_norm/scale"])
            )


def _copy_vision_params(flat_params: dict[str, object], encoder: Gemma4VisionEncoder) -> None:
    with torch.no_grad():
        encoder.patch_embed.position_table.copy_(_torch_from_jax(flat_params["entry/pos_emb"]))
        encoder.patch_embed.input_proj.weight.copy_(_torch_from_jax(flat_params["entry/input_projection/w"]).transpose(0, 1))

        query = _torch_from_jax(flat_params["transformer/stacked_layers/block/attn/q_einsum/w"])
        kv = _torch_from_jax(flat_params["transformer/stacked_layers/block/attn/kv_einsum/w"])
        output = _torch_from_jax(flat_params["transformer/stacked_layers/block/attn/attn_vec_einsum/w"])
        query_norm = _torch_from_jax(flat_params["transformer/stacked_layers/block/attn/query_norm/scale"])
        key_norm = _torch_from_jax(flat_params["transformer/stacked_layers/block/attn/key_norm/scale"])
        pre_attn = _torch_from_jax(flat_params["transformer/stacked_layers/block/pre_attention_norm/scale"])
        post_attn = _torch_from_jax(flat_params["transformer/stacked_layers/block/post_attention_norm/scale"])
        pre_ffn = _torch_from_jax(flat_params["transformer/stacked_layers/block/pre_ffw_norm/scale"])
        post_ffn = _torch_from_jax(flat_params["transformer/stacked_layers/block/post_ffw_norm/scale"])
        gating = _torch_from_jax(flat_params["transformer/stacked_layers/block/mlp/gating_einsum/w"])
        down = _torch_from_jax(flat_params["transformer/stacked_layers/block/mlp/linear/w"])

        for layer_idx, layer in enumerate(encoder.layers):
            layer.input_norm.weight.copy_(pre_attn[layer_idx])
            layer.post_attn_norm.weight.copy_(post_attn[layer_idx])
            layer.pre_ffn_norm.weight.copy_(pre_ffn[layer_idx])
            layer.post_ffn_norm.weight.copy_(post_ffn[layer_idx])

            layer.attn.q_proj.weight.copy_(query[layer_idx].permute(0, 2, 1).reshape(4 * 6, 24))
            layer.attn.k_proj.weight.copy_(kv[layer_idx, 0].permute(0, 2, 1).reshape(4 * 6, 24))
            layer.attn.v_proj.weight.copy_(kv[layer_idx, 1].permute(0, 2, 1).reshape(4 * 6, 24))
            layer.attn.o_proj.weight.copy_(output[layer_idx].permute(2, 0, 1).reshape(24, 4 * 6))
            layer.attn.q_norm.weight.copy_(query_norm[layer_idx])
            layer.attn.k_norm.weight.copy_(key_norm[layer_idx])

            layer.mlp.gate_proj.weight.copy_(gating[layer_idx, 0])
            layer.mlp.up_proj.weight.copy_(gating[layer_idx, 1])
            layer.mlp.down_proj.weight.copy_(down[layer_idx].transpose(0, 1))


def _init_jax_text_cache(cache_length: int) -> dict[str, object]:
    cache = {}
    for layer_idx in range(4):
        cache[f"layer_{layer_idx}"] = jax_text_modules.Attention.init_cache(
            cache_length,
            2,
            8,
            1,
            jnp.float32,
        )
    return cache


def _slot_causal_mask(query_positions, cache_length: int):
    slots = jnp.arange(cache_length, dtype=jnp.int32)[None, None, :]
    return slots <= query_positions[:, :, None]


def test_text_tower_matches_original_jax_full_forward() -> None:
    config = make_parity_text_config()
    jax_model = JaxTinyTextTower()
    tokens = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32)
    positions = jnp.arange(tokens.shape[1], dtype=jnp.int32)[None, :]
    full_mask = jnp.tril(jnp.ones((1, 8, 8), dtype=bool))

    variables = jax_model.init(jax.random.PRNGKey(0), tokens, positions, full_mask, full_mask)
    flat_params = _flat_params(variables["params"])

    torch_model = Gemma4TextTower(config)
    _copy_text_params(flat_params, torch_model)
    torch_model.eval()

    with torch.no_grad():
        torch_tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
        torch_positions = torch.arange(8, dtype=torch.long).unsqueeze(0)
        torch_mask = make_causal_mask(torch.ones_like(torch_tokens, dtype=torch.bool))
        torch_hidden = torch_model(
            torch_tokens,
            position_ids=torch_positions,
            full_attention_mask=torch_mask,
            sliding_attention_mask=torch_mask,
        )
        torch_logits = torch.tanh(torch_model.project_logits(torch_hidden) / 30.0) * 30.0

    jax_hidden, _ = jax_model.apply(variables, tokens, positions, full_mask, full_mask)
    jax_logits = jnp.tanh(jnp.dot(jax_hidden, variables["params"]["Embedder_0"]["input_embedding"].T) / 30.0) * 30.0

    torch.testing.assert_close(torch_hidden, _torch_from_jax(jax_hidden), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(torch_logits, _torch_from_jax(jax_logits), atol=1e-6, rtol=1e-5)


def test_text_kv_cache_matches_original_jax() -> None:
    config = make_parity_text_config()
    jax_model = JaxTinyTextTower()
    full_tokens = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32)
    full_positions = jnp.arange(full_tokens.shape[1], dtype=jnp.int32)[None, :]
    full_mask = jnp.tril(jnp.ones((1, 8, 8), dtype=bool))

    variables = jax_model.init(jax.random.PRNGKey(0), full_tokens, full_positions, full_mask, full_mask)
    flat_params = _flat_params(variables["params"])

    model = Gemma4Model(Gemma4Config(text=config))
    _copy_text_params(flat_params, model.text)
    model.eval()

    prefill_tokens = full_tokens[:, :5]
    prefill_positions = full_positions[:, :5]
    prefill_mask = _slot_causal_mask(prefill_positions, 8)
    jax_cache = _init_jax_text_cache(8)
    _, jax_cache = jax_model.apply(
        variables,
        prefill_tokens,
        prefill_positions,
        prefill_mask,
        prefill_mask,
        cache=jax_cache,
    )

    step_token = full_tokens[:, 5:6]
    step_position = full_positions[:, 5:6]
    step_mask = _slot_causal_mask(step_position, 8)
    jax_hidden, _ = jax_model.apply(
        variables,
        step_token,
        step_position,
        step_mask,
        step_mask,
        cache=jax_cache,
    )
    jax_logits = jnp.tanh(jnp.dot(jax_hidden, variables["params"]["Embedder_0"]["input_embedding"].T) / 30.0) * 30.0

    with torch.no_grad():
        prefill_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        prefill_out = model(
            prefill_ids,
            attention_mask=torch.ones_like(prefill_ids, dtype=torch.bool),
            return_hidden_states=True,
            return_kv_cache=True,
        )
        step_ids = torch.tensor([[6]], dtype=torch.long)
        step_out = model(
            step_ids,
            attention_mask=torch.ones_like(step_ids, dtype=torch.bool),
            kv_cache=prefill_out.kv_cache,
            return_hidden_states=True,
            return_kv_cache=True,
        )

    assert step_out.hidden_states is not None
    torch.testing.assert_close(step_out.hidden_states, _torch_from_jax(jax_hidden), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(step_out.logits, _torch_from_jax(jax_logits), atol=1e-6, rtol=1e-5)


def test_vision_encoder_matches_original_jax() -> None:
    config = make_parity_vision_config()
    jax_model = jax_vision_encoder.VisionEncoder(
        d_model=24,
        ffw_hidden=48,
        num_layers=2,
        num_heads=4,
        patch_size=2,
        output_length=4,
        pos_emb_shape_yx=(8, 2),
        pooling_kernel_size=1,
        use_clipped_linears=False,
        standardize_embeddings=False,
    )
    patches = jax.random.uniform(jax.random.PRNGKey(3), (1, 4, 12), dtype=jnp.float32)
    positions_xy = jnp.array([[[0, 0], [1, 0], [0, 1], [1, 1]]], dtype=jnp.int32)

    variables = jax_model.init(jax.random.PRNGKey(1), patches, positions_xy)
    flat_params = _flat_params(variables["params"])

    torch_model = Gemma4VisionEncoder(config)
    _copy_vision_params(flat_params, torch_model)
    torch_model.eval()

    with torch.no_grad():
        torch_outputs = torch_model(_torch_from_jax(patches), _torch_from_jax(positions_xy))

    (jax_tokens, jax_mask), = jax_model.apply(variables, patches, positions_xy)
    (torch_tokens, torch_mask), = torch_outputs

    torch.testing.assert_close(torch_tokens, _torch_from_jax(jax_tokens), atol=1e-6, rtol=1e-5)
    assert torch.equal(torch_mask, _torch_from_jax(jax_mask))
