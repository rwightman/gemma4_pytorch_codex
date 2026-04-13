from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file as save_safetensors

from gemma4_pt_codex import AudioConfig, Gemma4Config, Gemma4Model, TextConfig, VisionConfig
from gemma4_pt_codex.config import AttentionKind
from gemma4_pt_codex.convert import (
    convert_hf_checkpoint,
    convert_hf_state_dict_to_native,
    convert_jax_tree_to_native,
    native_config_from_hf_dict,
)


def make_dense_multimodal_config(*, hf_audio_order: bool) -> Gemma4Config:
    return Gemma4Config(
        text=TextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_layers=2,
            num_heads=4,
            head_dim=4,
            num_kv_heads=2,
            num_global_kv_heads=2,
            global_head_dim=4,
            layer_types=(
                AttentionKind.SLIDING,
                AttentionKind.FULL,
            ),
            per_layer_input_dim=4,
            sliding_window=8,
            pad_token_id=0,
            image_token_id=258_880,
            audio_token_id=258_881,
            final_logit_softcap=30.0,
        ),
        vision=VisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            head_dim=4,
            patch_size=2,
            position_embedding_size=8,
            output_length=4,
            pooling_kernel_size=1,
            use_clipped_linears=True,
        ),
        audio=AudioConfig(
            num_layers=2,
            hidden_size=16,
            output_size=16,
            num_heads=4,
            left_context=3,
            right_context=0,
            chunk_size=4,
            conv_kernel_size=3,
            subsampling_channels=(4, 4),
            num_mel_bins=8,
            projection_norm_before_text=hf_audio_order,
        ),
    )


def make_moe_text_config() -> Gemma4Config:
    return Gemma4Config(
        text=TextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=12,
            num_layers=2,
            num_heads=4,
            head_dim=4,
            num_kv_heads=2,
            num_global_kv_heads=2,
            global_head_dim=4,
            layer_types=(
                AttentionKind.SLIDING,
                AttentionKind.FULL,
            ),
            sliding_window=8,
            per_layer_input_dim=0,
            enable_moe=True,
            num_experts=4,
            expert_dim=8,
            top_k_experts=2,
            moe_dense_hidden_size=12,
            final_logit_softcap=30.0,
            image_token_id=258_880,
            audio_token_id=258_881,
        )
    )


def make_hf_config_dict(config: Gemma4Config, *, text_only: bool = False) -> dict[str, object]:
    text = config.text
    text_config = {
        "model_type": "gemma4_text",
        "vocab_size": text.vocab_size,
        "hidden_size": text.hidden_size,
        "intermediate_size": text.intermediate_size,
        "num_hidden_layers": text.num_layers,
        "num_attention_heads": text.num_heads,
        "num_key_value_heads": text.num_kv_heads,
        "head_dim": text.head_dim,
        "rms_norm_eps": text.rms_norm_eps,
        "pad_token_id": text.pad_token_id,
        "sliding_window": text.sliding_window,
        "layer_types": [
            "full_attention" if layer_type == AttentionKind.FULL else "sliding_attention"
            for layer_type in text.layer_types
        ],
        "final_logit_softcapping": text.final_logit_softcap,
        "use_bidirectional_attention": text.use_bidirectional_attention,
        "hidden_size_per_layer_input": text.per_layer_input_dim,
        "num_global_key_value_heads": text.num_global_kv_heads,
        "global_head_dim": text.global_head_dim,
        "attention_k_eq_v": text.attention_k_eq_v_global,
        "num_kv_shared_layers": (
            0
            if text.kv_sharing is None
            else int(round(text.kv_sharing.frac_shared_layers * text.num_layers))
        ),
        "use_double_wide_mlp": text.override_kv_shared_ffn_hidden is not None,
        "enable_moe_block": text.enable_moe,
        "num_experts": text.num_experts or None,
        "top_k_experts": text.top_k_experts or None,
        "moe_intermediate_size": text.expert_dim or None,
        "rope_parameters": {
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": float(text.local_rope_theta),
            },
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": float(text.global_rope_proportion or 0.25),
                "rope_theta": float(text.global_rope_theta),
            },
        },
    }
    if text_only:
        return text_config

    assert config.vision is not None
    assert config.audio is not None
    vision = config.vision
    audio = config.audio
    return {
        "model_type": "gemma4",
        "text_config": text_config,
        "vision_config": {
            "model_type": "gemma4_vision",
            "hidden_size": vision.hidden_size,
            "intermediate_size": vision.intermediate_size,
            "num_hidden_layers": vision.num_layers,
            "num_attention_heads": vision.num_heads,
            "num_key_value_heads": vision.num_kv_heads,
            "head_dim": vision.head_dim,
            "patch_size": vision.patch_size,
            "position_embedding_size": vision.position_embedding_size,
            "pooling_kernel_size": vision.pooling_kernel_size,
            "default_output_length": vision.output_length,
            "use_clipped_linears": vision.use_clipped_linears,
            "standardize": vision.standardize_embeddings,
            "rms_norm_eps": vision.rms_norm_eps,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": vision.rope_theta,
            },
        },
        "audio_config": {
            "model_type": "gemma4_audio",
            "hidden_size": audio.hidden_size,
            "num_hidden_layers": audio.num_layers,
            "num_attention_heads": audio.num_heads,
            "subsampling_conv_channels": list(audio.subsampling_channels),
            "conv_kernel_size": audio.conv_kernel_size,
            "attention_chunk_size": audio.chunk_size,
            "attention_context_left": audio.left_context,
            "attention_context_right": audio.right_context,
            "num_mel_bins": audio.num_mel_bins,
            "rms_norm_eps": audio.rms_norm_eps,
            "gradient_clipping": audio.gradient_clipping,
            "output_proj_dims": audio.output_size,
        },
        "vision_soft_tokens_per_image": vision.output_length,
        "image_token_id": text.image_token_id,
        "audio_token_id": text.audio_token_id,
    }


def native_to_hf_dense_state_dict(model: Gemma4Model) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    config = model.config
    text = config.text
    hf_state: dict[str, torch.Tensor] = {}
    text_prefix = "model.language_model"

    hf_state[f"{text_prefix}.embed_tokens.weight"] = state["text.token_embedding.weight"].clone()
    hf_state[f"{text_prefix}.embed_tokens_per_layer.weight"] = state["text.per_layer_token_embedding"].reshape(
        text.vocab_size,
        text.num_layers * text.per_layer_input_dim,
    ).clone()
    hf_state[f"{text_prefix}.per_layer_model_projection.weight"] = (
        state["text.per_layer_model_projection.weight"] / (text.hidden_size**-0.5)
    ).clone()
    hf_state[f"{text_prefix}.per_layer_projection_norm.weight"] = state["text.per_layer_projection_norm.weight"].clone()
    hf_state[f"{text_prefix}.norm.weight"] = state["text.final_norm.weight"].clone()

    for layer_idx in range(text.num_layers):
        native_prefix = f"text.layers.{layer_idx}."
        hf_prefix = f"{text_prefix}.layers.{layer_idx}."
        hf_state[f"{hf_prefix}self_attn.q_proj.weight"] = state[f"{native_prefix}attn.q_proj.weight"].clone()
        hf_state[f"{hf_prefix}self_attn.k_proj.weight"] = state[f"{native_prefix}attn.k_proj.weight"].clone()
        if f"{native_prefix}attn.v_proj.weight" in state:
            hf_state[f"{hf_prefix}self_attn.v_proj.weight"] = state[f"{native_prefix}attn.v_proj.weight"].clone()
        hf_state[f"{hf_prefix}self_attn.o_proj.weight"] = state[f"{native_prefix}attn.o_proj.weight"].clone()
        hf_state[f"{hf_prefix}self_attn.q_norm.weight"] = state[f"{native_prefix}attn.q_norm.weight"].clone()
        hf_state[f"{hf_prefix}self_attn.k_norm.weight"] = state[f"{native_prefix}attn.k_norm.weight"].clone()
        hf_state[f"{hf_prefix}input_layernorm.weight"] = state[f"{native_prefix}pre_attn_norm.weight"].clone()
        hf_state[f"{hf_prefix}post_attention_layernorm.weight"] = state[f"{native_prefix}post_attn_norm.weight"].clone()
        hf_state[f"{hf_prefix}pre_feedforward_layernorm.weight"] = state[f"{native_prefix}pre_ffn_norm.weight"].clone()
        hf_state[f"{hf_prefix}post_feedforward_layernorm.weight"] = state[
            f"{native_prefix}post_ffn_norm.weight"
        ].clone()
        hf_state[f"{hf_prefix}mlp.gate_proj.weight"] = state[f"{native_prefix}mlp.gate_proj.weight"].clone()
        hf_state[f"{hf_prefix}mlp.up_proj.weight"] = state[f"{native_prefix}mlp.up_proj.weight"].clone()
        hf_state[f"{hf_prefix}mlp.down_proj.weight"] = state[f"{native_prefix}mlp.down_proj.weight"].clone()
        hf_state[f"{hf_prefix}layer_scalar"] = state[f"{native_prefix}layer_scalar"].clone()
        hf_state[f"{hf_prefix}per_layer_input_gate.weight"] = state[
            f"{native_prefix}per_layer_input_gate.weight"
        ].clone()
        hf_state[f"{hf_prefix}per_layer_projection.weight"] = state[
            f"{native_prefix}per_layer_projection.weight"
        ].clone()
        hf_state[f"{hf_prefix}post_per_layer_input_norm.weight"] = state[
            f"{native_prefix}post_per_layer_input_norm.weight"
        ].clone()

    assert config.vision is not None
    for layer_idx in range(config.vision.num_layers):
        native_prefix = f"vision.encoder.layers.{layer_idx}."
        hf_prefix = f"model.vision_tower.encoder.layers.{layer_idx}."
        vision_map = {
            "input_layernorm.weight": "input_norm.weight",
            "post_attention_layernorm.weight": "post_attn_norm.weight",
            "pre_feedforward_layernorm.weight": "pre_ffn_norm.weight",
            "post_feedforward_layernorm.weight": "post_ffn_norm.weight",
            "self_attn.q_norm.weight": "attn.q_norm.weight",
            "self_attn.k_norm.weight": "attn.k_norm.weight",
            "self_attn.q_proj.linear.weight": "attn.q_proj.weight",
            "self_attn.k_proj.linear.weight": "attn.k_proj.weight",
            "self_attn.v_proj.linear.weight": "attn.v_proj.weight",
            "self_attn.o_proj.linear.weight": "attn.o_proj.weight",
            "self_attn.q_proj.input_min": "attn.q_proj.input_min",
            "self_attn.q_proj.input_max": "attn.q_proj.input_max",
            "self_attn.q_proj.output_min": "attn.q_proj.output_min",
            "self_attn.q_proj.output_max": "attn.q_proj.output_max",
            "self_attn.k_proj.input_min": "attn.k_proj.input_min",
            "self_attn.k_proj.input_max": "attn.k_proj.input_max",
            "self_attn.k_proj.output_min": "attn.k_proj.output_min",
            "self_attn.k_proj.output_max": "attn.k_proj.output_max",
            "self_attn.v_proj.input_min": "attn.v_proj.input_min",
            "self_attn.v_proj.input_max": "attn.v_proj.input_max",
            "self_attn.v_proj.output_min": "attn.v_proj.output_min",
            "self_attn.v_proj.output_max": "attn.v_proj.output_max",
            "self_attn.o_proj.input_min": "attn.o_proj.input_min",
            "self_attn.o_proj.input_max": "attn.o_proj.input_max",
            "self_attn.o_proj.output_min": "attn.o_proj.output_min",
            "self_attn.o_proj.output_max": "attn.o_proj.output_max",
            "mlp.gate_proj.linear.weight": "mlp.gate_proj.weight",
            "mlp.up_proj.linear.weight": "mlp.up_proj.weight",
            "mlp.down_proj.linear.weight": "mlp.down_proj.weight",
            "mlp.gate_proj.input_min": "mlp.gate_proj.input_min",
            "mlp.gate_proj.input_max": "mlp.gate_proj.input_max",
            "mlp.gate_proj.output_min": "mlp.gate_proj.output_min",
            "mlp.gate_proj.output_max": "mlp.gate_proj.output_max",
            "mlp.up_proj.input_min": "mlp.up_proj.input_min",
            "mlp.up_proj.input_max": "mlp.up_proj.input_max",
            "mlp.up_proj.output_min": "mlp.up_proj.output_min",
            "mlp.up_proj.output_max": "mlp.up_proj.output_max",
            "mlp.down_proj.input_min": "mlp.down_proj.input_min",
            "mlp.down_proj.input_max": "mlp.down_proj.input_max",
            "mlp.down_proj.output_min": "mlp.down_proj.output_min",
            "mlp.down_proj.output_max": "mlp.down_proj.output_max",
        }
        for hf_suffix, native_suffix in vision_map.items():
            hf_state[f"{hf_prefix}{hf_suffix}"] = state[f"{native_prefix}{native_suffix}"].clone()

    hf_state["model.vision_tower.patch_embedder.input_proj.weight"] = state[
        "vision.encoder.patch_embed.input_proj.weight"
    ].clone()
    hf_state["model.vision_tower.patch_embedder.position_embedding_table"] = state[
        "vision.encoder.patch_embed.position_table"
    ].transpose(0, 1).clone()
    hf_state["model.embed_vision.embedding_projection.weight"] = state["vision.to_text.weight"].clone()

    assert config.audio is not None
    hf_state["model.audio_tower.subsample_conv_projection.layer0.conv.weight"] = state[
        "audio.encoder.subsampler.conv0.weight"
    ].clone()
    hf_state["model.audio_tower.subsample_conv_projection.layer1.conv.weight"] = state[
        "audio.encoder.subsampler.conv1.weight"
    ].clone()
    hf_state["model.audio_tower.subsample_conv_projection.layer0.norm.weight"] = state[
        "audio.encoder.subsampler.norm0.weight"
    ].clone()
    hf_state["model.audio_tower.subsample_conv_projection.layer1.norm.weight"] = state[
        "audio.encoder.subsampler.norm1.weight"
    ].clone()
    hf_state["model.audio_tower.subsample_conv_projection.input_proj_linear.weight"] = state[
        "audio.encoder.subsampler.output_proj.weight"
    ].clone()
    hf_state["model.audio_tower.output_proj.weight"] = state["audio.encoder.output_proj.weight"].clone()
    hf_state["model.audio_tower.output_proj.bias"] = state["audio.encoder.output_proj.bias"].clone()
    hf_state["model.embed_audio.embedding_projection.weight"] = state["audio.to_text.weight"].clone()

    for layer_idx in range(config.audio.num_layers):
        native_prefix = f"audio.encoder.layers.{layer_idx}."
        hf_prefix = f"model.audio_tower.layers.{layer_idx}."
        audio_map = {
            "feed_forward1.pre_layer_norm.weight": "ffn_start.pre_norm.weight",
            "feed_forward1.ffw_layer_1.linear.weight": "ffn_start.ffn1.weight",
            "feed_forward1.ffw_layer_1.input_min": "ffn_start.ffn1.input_min",
            "feed_forward1.ffw_layer_1.input_max": "ffn_start.ffn1.input_max",
            "feed_forward1.ffw_layer_1.output_min": "ffn_start.ffn1.output_min",
            "feed_forward1.ffw_layer_1.output_max": "ffn_start.ffn1.output_max",
            "feed_forward1.ffw_layer_2.linear.weight": "ffn_start.ffn2.weight",
            "feed_forward1.ffw_layer_2.input_min": "ffn_start.ffn2.input_min",
            "feed_forward1.ffw_layer_2.input_max": "ffn_start.ffn2.input_max",
            "feed_forward1.ffw_layer_2.output_min": "ffn_start.ffn2.output_min",
            "feed_forward1.ffw_layer_2.output_max": "ffn_start.ffn2.output_max",
            "feed_forward1.post_layer_norm.weight": "ffn_start.post_norm.weight",
            "self_attn.q_proj.linear.weight": "attn.attn.q_proj.weight",
            "self_attn.q_proj.input_min": "attn.attn.q_proj.input_min",
            "self_attn.q_proj.input_max": "attn.attn.q_proj.input_max",
            "self_attn.q_proj.output_min": "attn.attn.q_proj.output_min",
            "self_attn.q_proj.output_max": "attn.attn.q_proj.output_max",
            "self_attn.k_proj.linear.weight": "attn.attn.k_proj.weight",
            "self_attn.k_proj.input_min": "attn.attn.k_proj.input_min",
            "self_attn.k_proj.input_max": "attn.attn.k_proj.input_max",
            "self_attn.k_proj.output_min": "attn.attn.k_proj.output_min",
            "self_attn.k_proj.output_max": "attn.attn.k_proj.output_max",
            "self_attn.v_proj.linear.weight": "attn.attn.v_proj.weight",
            "self_attn.v_proj.input_min": "attn.attn.v_proj.input_min",
            "self_attn.v_proj.input_max": "attn.attn.v_proj.input_max",
            "self_attn.v_proj.output_min": "attn.attn.v_proj.output_min",
            "self_attn.v_proj.output_max": "attn.attn.v_proj.output_max",
            "self_attn.per_dim_scale": "attn.attn.per_dim_scale",
            "self_attn.relative_k_proj.weight": "attn.attn.relative_position.pos_proj.weight",
            "self_attn.post.linear.weight": "attn.post.weight",
            "self_attn.post.input_min": "attn.post.input_min",
            "self_attn.post.input_max": "attn.post.input_max",
            "self_attn.post.output_min": "attn.post.output_min",
            "self_attn.post.output_max": "attn.post.output_max",
            "norm_pre_attn.weight": "attn.pre_norm.weight",
            "norm_post_attn.weight": "attn.post_norm.weight",
            "lconv1d.pre_layer_norm.weight": "lightconv.pre_norm.weight",
            "lconv1d.linear_start.linear.weight": "lightconv.linear_start.weight",
            "lconv1d.linear_start.input_min": "lightconv.linear_start.input_min",
            "lconv1d.linear_start.input_max": "lightconv.linear_start.input_max",
            "lconv1d.linear_start.output_min": "lightconv.linear_start.output_min",
            "lconv1d.linear_start.output_max": "lightconv.linear_start.output_max",
            "lconv1d.depthwise_conv1d.weight": "lightconv.depthwise.weight",
            "lconv1d.conv_norm.weight": "lightconv.conv_norm.weight",
            "lconv1d.linear_end.linear.weight": "lightconv.linear_end.weight",
            "lconv1d.linear_end.input_min": "lightconv.linear_end.input_min",
            "lconv1d.linear_end.input_max": "lightconv.linear_end.input_max",
            "lconv1d.linear_end.output_min": "lightconv.linear_end.output_min",
            "lconv1d.linear_end.output_max": "lightconv.linear_end.output_max",
            "feed_forward2.pre_layer_norm.weight": "ffn_end.pre_norm.weight",
            "feed_forward2.ffw_layer_1.linear.weight": "ffn_end.ffn1.weight",
            "feed_forward2.ffw_layer_1.input_min": "ffn_end.ffn1.input_min",
            "feed_forward2.ffw_layer_1.input_max": "ffn_end.ffn1.input_max",
            "feed_forward2.ffw_layer_1.output_min": "ffn_end.ffn1.output_min",
            "feed_forward2.ffw_layer_1.output_max": "ffn_end.ffn1.output_max",
            "feed_forward2.ffw_layer_2.linear.weight": "ffn_end.ffn2.weight",
            "feed_forward2.ffw_layer_2.input_min": "ffn_end.ffn2.input_min",
            "feed_forward2.ffw_layer_2.input_max": "ffn_end.ffn2.input_max",
            "feed_forward2.ffw_layer_2.output_min": "ffn_end.ffn2.output_min",
            "feed_forward2.ffw_layer_2.output_max": "ffn_end.ffn2.output_max",
            "feed_forward2.post_layer_norm.weight": "ffn_end.post_norm.weight",
            "norm_out.weight": "final_norm.weight",
        }
        for hf_suffix, native_suffix in audio_map.items():
            hf_state[f"{hf_prefix}{hf_suffix}"] = state[f"{native_prefix}{native_suffix}"].clone()

    return hf_state


def native_to_hf_moe_state_dict(model: Gemma4Model) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    config = model.config.text
    hf_state: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": state["text.token_embedding.weight"].clone(),
        "model.norm.weight": state["text.final_norm.weight"].clone(),
    }

    for layer_idx in range(config.num_layers):
        native_prefix = f"text.layers.{layer_idx}."
        hf_prefix = f"model.layers.{layer_idx}."
        hf_state[f"{hf_prefix}self_attn.q_proj.weight"] = state[f"{native_prefix}attn.q_proj.weight"].clone()
        hf_state[f"{hf_prefix}self_attn.k_proj.weight"] = state[f"{native_prefix}attn.k_proj.weight"].clone()
        if f"{native_prefix}attn.v_proj.weight" in state:
            hf_state[f"{hf_prefix}self_attn.v_proj.weight"] = state[f"{native_prefix}attn.v_proj.weight"].clone()
        hf_state[f"{hf_prefix}self_attn.o_proj.weight"] = state[f"{native_prefix}attn.o_proj.weight"].clone()
        hf_state[f"{hf_prefix}self_attn.q_norm.weight"] = state[f"{native_prefix}attn.q_norm.weight"].clone()
        hf_state[f"{hf_prefix}self_attn.k_norm.weight"] = state[f"{native_prefix}attn.k_norm.weight"].clone()
        hf_state[f"{hf_prefix}input_layernorm.weight"] = state[f"{native_prefix}pre_attn_norm.weight"].clone()
        hf_state[f"{hf_prefix}post_attention_layernorm.weight"] = state[f"{native_prefix}post_attn_norm.weight"].clone()
        hf_state[f"{hf_prefix}pre_feedforward_layernorm.weight"] = state[f"{native_prefix}pre_ffn2_norm.weight"].clone()
        hf_state[f"{hf_prefix}post_feedforward_layernorm.weight"] = state[
            f"{native_prefix}post_ffn_norm.weight"
        ].clone()
        hf_state[f"{hf_prefix}pre_feedforward_layernorm_2.weight"] = state[
            f"{native_prefix}pre_ffn_norm.weight"
        ].clone()
        hf_state[f"{hf_prefix}post_feedforward_layernorm_1.weight"] = state[
            f"{native_prefix}post_ffn2_norm.weight"
        ].clone()
        hf_state[f"{hf_prefix}post_feedforward_layernorm_2.weight"] = state[
            f"{native_prefix}post_ffn1_norm.weight"
        ].clone()
        hf_state[f"{hf_prefix}mlp.gate_proj.weight"] = state[f"{native_prefix}mlp2.gate_proj.weight"].clone()
        hf_state[f"{hf_prefix}mlp.up_proj.weight"] = state[f"{native_prefix}mlp2.up_proj.weight"].clone()
        hf_state[f"{hf_prefix}mlp.down_proj.weight"] = state[f"{native_prefix}mlp2.down_proj.weight"].clone()
        hf_state[f"{hf_prefix}router.proj.weight"] = state[f"{native_prefix}moe.router.weight"].clone()
        hf_state[f"{hf_prefix}router.scale"] = state[f"{native_prefix}moe.router_scale"].clone()
        hf_state[f"{hf_prefix}router.per_expert_scale"] = state[f"{native_prefix}moe.per_expert_scale"].clone()
        hf_state[f"{hf_prefix}experts.gate_up_proj"] = state[f"{native_prefix}moe.gate_up_proj"].clone()
        hf_state[f"{hf_prefix}experts.down_proj"] = state[f"{native_prefix}moe.down_proj"].clone()
        hf_state[f"{hf_prefix}layer_scalar"] = state[f"{native_prefix}layer_scalar"].clone()

    return hf_state


def native_to_jax_tree_dense(model: Gemma4Model) -> dict[str, object]:
    state = model.state_dict()
    config = model.config
    tree: dict[str, object] = {"params": {"transformer": {"embedder": {}}, "PatchInputVariablePoolingEncoder_0": {}}}

    params = tree["params"]
    assert isinstance(params, dict)
    transformer = params["transformer"]
    assert isinstance(transformer, dict)
    embedder = transformer["embedder"]
    assert isinstance(embedder, dict)

    embedder["input_embedding"] = state["text.token_embedding.weight"].numpy()
    embedder["per_layer_embeddings"] = state["text.per_layer_token_embedding"].numpy()
    embedder["per_layer_model_projection"] = {
        "w": (
            state["text.per_layer_model_projection.weight"] / (config.text.hidden_size**-0.5)
        )
        .view(
            config.text.num_layers,
            config.text.per_layer_input_dim,
            config.text.hidden_size,
        )
        .permute(2, 0, 1)
        .numpy()
    }
    embedder["per_layer_projection_norm"] = {"scale": state["text.per_layer_projection_norm.weight"].numpy()}
    embedder["mm_input_projection"] = {"w": state["vision.to_text.weight"].transpose(0, 1).numpy()}
    embedder["audio_input_projection"] = {"w": state["audio.to_text.weight"].transpose(0, 1).numpy()}
    transformer["final_norm"] = {"scale": state["text.final_norm.weight"].numpy()}

    for layer_idx in range(config.text.num_layers):
        native_prefix = f"text.layers.{layer_idx}."
        transformer[f"layer_{layer_idx}"] = {
            "skip_scale": state[f"{native_prefix}layer_scalar"].numpy(),
            "pre_attention_norm": {"scale": state[f"{native_prefix}pre_attn_norm.weight"].numpy()},
            "post_attention_norm": {"scale": state[f"{native_prefix}post_attn_norm.weight"].numpy()},
            "pre_ffw_norm": {"scale": state[f"{native_prefix}pre_ffn_norm.weight"].numpy()},
            "post_ffw_norm": {"scale": state[f"{native_prefix}post_ffn_norm.weight"].numpy()},
            "attn": {
                "q_einsum": {
                    "w": state[f"{native_prefix}attn.q_proj.weight"].transpose(0, 1).view(
                        config.text.hidden_size,
                        config.text.num_heads,
                        config.text.head_dim,
                    ).permute(1, 0, 2).numpy()
                },
                "kv_einsum": {
                    "w": torch.stack(
                        [
                            state[f"{native_prefix}attn.k_proj.weight"].transpose(0, 1).view(
                                config.text.hidden_size,
                                config.text.num_kv_heads,
                                config.text.head_dim,
                            ).permute(1, 0, 2),
                            state[f"{native_prefix}attn.v_proj.weight"].transpose(0, 1).view(
                                config.text.hidden_size,
                                config.text.num_kv_heads,
                                config.text.head_dim,
                            ).permute(1, 0, 2),
                        ],
                        dim=0,
                    ).numpy()
                },
                "attn_vec_einsum": {
                    "w": state[f"{native_prefix}attn.o_proj.weight"].view(
                        config.text.hidden_size,
                        config.text.num_heads,
                        config.text.head_dim,
                    ).permute(1, 2, 0).numpy()
                },
                "query_norm": {"scale": state[f"{native_prefix}attn.q_norm.weight"].numpy()},
                "key_norm": {"scale": state[f"{native_prefix}attn.k_norm.weight"].numpy()},
            },
            "mlp": {
                "gating_einsum": {
                    "w": torch.stack(
                        [
                            state[f"{native_prefix}mlp.gate_proj.weight"],
                            state[f"{native_prefix}mlp.up_proj.weight"],
                        ],
                        dim=0,
                    ).numpy()
                },
                "linear": {"w": state[f"{native_prefix}mlp.down_proj.weight"].transpose(0, 1).numpy()},
            },
            "per_layer_input_gate": {"w": state[f"{native_prefix}per_layer_input_gate.weight"].transpose(0, 1).numpy()},
            "per_layer_projection": {"w": state[f"{native_prefix}per_layer_projection.weight"].transpose(0, 1).numpy()},
            "post_per_layer_input_norm": {"scale": state[f"{native_prefix}post_per_layer_input_norm.weight"].numpy()},
        }

    vision_root = {
        "_model": {
            "vit": {
                "entry": {
                    "pos_emb": state["vision.encoder.patch_embed.position_table"].numpy(),
                    "input_projection": {
                        "w": state["vision.encoder.patch_embed.input_proj.weight"].transpose(0, 1).numpy(),
                    },
                },
                "transformer": {
                    "stacked_layers": {
                        "block": {
                            "pre_attention_norm": {
                                "scale": torch.stack(
                                    [
                                        state[f"vision.encoder.layers.{layer_idx}.input_norm.weight"]
                                        for layer_idx in range(config.vision.num_layers)
                                    ],
                                    dim=0,
                                ).numpy()
                            },
                            "post_attention_norm": {
                                "scale": torch.stack(
                                    [
                                        state[f"vision.encoder.layers.{layer_idx}.post_attn_norm.weight"]
                                        for layer_idx in range(config.vision.num_layers)
                                    ],
                                    dim=0,
                                ).numpy()
                            },
                            "pre_ffw_norm": {
                                "scale": torch.stack(
                                    [
                                        state[f"vision.encoder.layers.{layer_idx}.pre_ffn_norm.weight"]
                                        for layer_idx in range(config.vision.num_layers)
                                    ],
                                    dim=0,
                                ).numpy()
                            },
                            "post_ffw_norm": {
                                "scale": torch.stack(
                                    [
                                        state[f"vision.encoder.layers.{layer_idx}.post_ffn_norm.weight"]
                                        for layer_idx in range(config.vision.num_layers)
                                    ],
                                    dim=0,
                                ).numpy()
                            },
                            "attn": {
                                "q_einsum": {
                                    "w": torch.stack(
                                        [
                                            state[f"vision.encoder.layers.{layer_idx}.attn.q_proj.weight"]
                                            .transpose(0, 1)
                                            .view(
                                                config.vision.hidden_size,
                                                config.vision.num_heads,
                                                config.vision.head_dim,
                                            )
                                            .permute(1, 0, 2)
                                            for layer_idx in range(config.vision.num_layers)
                                        ],
                                        dim=0,
                                    ).numpy()
                                },
                                "kv_einsum": {
                                    "w": torch.stack(
                                        [
                                            torch.stack(
                                                [
                                                    state[f"vision.encoder.layers.{layer_idx}.attn.k_proj.weight"]
                                                    .transpose(0, 1)
                                                    .view(
                                                        config.vision.hidden_size,
                                                        config.vision.num_kv_heads,
                                                        config.vision.head_dim,
                                                    )
                                                    .permute(1, 0, 2),
                                                    state[f"vision.encoder.layers.{layer_idx}.attn.v_proj.weight"]
                                                    .transpose(0, 1)
                                                    .view(
                                                        config.vision.hidden_size,
                                                        config.vision.num_kv_heads,
                                                        config.vision.head_dim,
                                                    )
                                                    .permute(1, 0, 2),
                                                ],
                                                dim=0,
                                            )
                                            for layer_idx in range(config.vision.num_layers)
                                        ],
                                        dim=0,
                                    ).numpy()
                                },
                                "attn_vec_einsum": {
                                    "w": torch.stack(
                                        [
                                            state[f"vision.encoder.layers.{layer_idx}.attn.o_proj.weight"]
                                            .view(
                                                config.vision.hidden_size,
                                                config.vision.num_heads,
                                                config.vision.head_dim,
                                            )
                                            .permute(1, 2, 0)
                                            for layer_idx in range(config.vision.num_layers)
                                        ],
                                        dim=0,
                                    ).numpy()
                                },
                                "query_norm": {
                                    "scale": torch.stack(
                                        [
                                            state[f"vision.encoder.layers.{layer_idx}.attn.q_norm.weight"]
                                            for layer_idx in range(config.vision.num_layers)
                                        ],
                                        dim=0,
                                    ).numpy()
                                },
                                "key_norm": {
                                    "scale": torch.stack(
                                        [
                                            state[f"vision.encoder.layers.{layer_idx}.attn.k_norm.weight"]
                                            for layer_idx in range(config.vision.num_layers)
                                        ],
                                        dim=0,
                                    ).numpy()
                                },
                            },
                            "mlp": {
                                "gating_einsum": {
                                    "w": torch.stack(
                                        [
                                            torch.stack(
                                                [
                                                    state[f"vision.encoder.layers.{layer_idx}.mlp.gate_proj.weight"],
                                                    state[f"vision.encoder.layers.{layer_idx}.mlp.up_proj.weight"],
                                                ],
                                                dim=0,
                                            )
                                            for layer_idx in range(config.vision.num_layers)
                                        ],
                                        dim=0,
                                    ).numpy()
                                },
                                "linear": {
                                    "w": torch.stack(
                                        [
                                            state[f"vision.encoder.layers.{layer_idx}.mlp.down_proj.weight"]
                                            .transpose(0, 1)
                                            for layer_idx in range(config.vision.num_layers)
                                        ],
                                        dim=0,
                                    ).numpy()
                                },
                            },
                        }
                    }
                },
            }
        }
    }
    params["PatchInputVariablePoolingEncoder_0"] = vision_root

    audio_root = {
        "encoder": {
            "feature": {
                "subsampling_0": {
                    "kernel": state["audio.encoder.subsampler.conv0.weight"].permute(2, 3, 1, 0).numpy(),
                },
                "subsampling_1": {
                    "kernel": state["audio.encoder.subsampler.conv1.weight"].permute(2, 3, 1, 0).numpy(),
                },
                "norm_0": {"scale": state["audio.encoder.subsampler.norm0.weight"].numpy()},
                "norm_1": {"scale": state["audio.encoder.subsampler.norm1.weight"].numpy()},
                "input_proj": {
                    "kernel": state["audio.encoder.subsampler.output_proj.weight"]
                    .view(config.audio.hidden_size, -1, config.audio.subsampling_channels[1])
                    .permute(1, 2, 0)
                    .numpy(),
                },
            },
        }
    }
    for layer_idx in range(config.audio.num_layers):
        native_prefix = f"audio.encoder.layers.{layer_idx}."
        audio_root["encoder"][f"conformer/stacked_layers_{layer_idx}"] = {
            "fflayer_start": {
                "pre_layer_norm": {"scale": state[f"{native_prefix}ffn_start.pre_norm.weight"].numpy()},
                "ffn_layer1": {"kernel": state[f"{native_prefix}ffn_start.ffn1.weight"].transpose(0, 1).numpy()},
                "ffn_layer2": {"kernel": state[f"{native_prefix}ffn_start.ffn2.weight"].transpose(0, 1).numpy()},
                "post_layer_norm": {"scale": state[f"{native_prefix}ffn_start.post_norm.weight"].numpy()},
            },
            "trans_atten": {
                "pre_norm": {"scale": state[f"{native_prefix}attn.pre_norm.weight"].numpy()},
                "post_norm": {"scale": state[f"{native_prefix}attn.post_norm.weight"].numpy()},
                "post": {
                    "kernel": state[f"{native_prefix}attn.post.weight"]
                    .view(
                        config.audio.hidden_size,
                        config.audio.num_heads,
                        config.audio.hidden_size // config.audio.num_heads,
                    )
                    .permute(1, 2, 0)
                    .numpy(),
                },
                "self_atten": {
                    "query": {
                        "kernel": state[f"{native_prefix}attn.attn.q_proj.weight"].transpose(0, 1).numpy(),
                    },
                    "key": {
                        "kernel": state[f"{native_prefix}attn.attn.k_proj.weight"].transpose(0, 1).numpy(),
                    },
                    "value": {
                        "kernel": state[f"{native_prefix}attn.attn.v_proj.weight"].transpose(0, 1).numpy(),
                    },
                    "per_dim_scale": state[f"{native_prefix}attn.attn.per_dim_scale"].numpy(),
                    "relative_position_embedding": {
                        "pos_proj": {
                            "kernel": state[f"{native_prefix}attn.attn.relative_position.pos_proj.weight"]
                            .transpose(0, 1)
                            .view(
                                config.audio.hidden_size,
                                config.audio.num_heads,
                                config.audio.hidden_size // config.audio.num_heads,
                            )
                            .numpy(),
                        }
                    },
                },
            },
            "lconv": {
                "ln": {"scale": state[f"{native_prefix}lightconv.pre_norm.weight"].numpy()},
                "linear_start": {
                    "kernel": state[f"{native_prefix}lightconv.linear_start.weight"].transpose(0, 1).numpy(),
                },
                "depthwise_conv1d": {
                    "kernel": state[f"{native_prefix}lightconv.depthwise.weight"].permute(2, 1, 0).numpy(),
                },
                "conv_norm": {"scale": state[f"{native_prefix}lightconv.conv_norm.weight"].numpy()},
                "linear_end": {
                    "kernel": state[f"{native_prefix}lightconv.linear_end.weight"].transpose(0, 1).numpy(),
                },
            },
            "fflayer_end": {
                "pre_layer_norm": {"scale": state[f"{native_prefix}ffn_end.pre_norm.weight"].numpy()},
                "ffn_layer1": {"kernel": state[f"{native_prefix}ffn_end.ffn1.weight"].transpose(0, 1).numpy()},
                "ffn_layer2": {"kernel": state[f"{native_prefix}ffn_end.ffn2.weight"].transpose(0, 1).numpy()},
                "post_layer_norm": {"scale": state[f"{native_prefix}ffn_end.post_norm.weight"].numpy()},
            },
            "final_ln": {"scale": state[f"{native_prefix}final_norm.weight"].numpy()},
        }

    audio_root["encoder"]["output_projection"] = {
        "kernel": state["audio.encoder.output_proj.weight"].transpose(0, 1).numpy(),
        "bias": state["audio.encoder.output_proj.bias"].numpy(),
    }
    params["AudioEncoder"] = audio_root
    return tree


def write_sharded_hf_checkpoint(
        path: Path,
        config_dict: dict[str, object],
        state_dict: dict[str, torch.Tensor],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    keys = sorted(state_dict.keys())
    midpoint = len(keys) // 2
    shards = [
        ("model-00001-of-00002.safetensors", keys[:midpoint]),
        ("model-00002-of-00002.safetensors", keys[midpoint:]),
    ]
    weight_map: dict[str, str] = {}
    for shard_name, shard_keys in shards:
        shard_state = {key: state_dict[key].contiguous() for key in shard_keys}
        save_safetensors(shard_state, str(path / shard_name))
        for key in shard_keys:
            weight_map[key] = shard_name

    with (path / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, f, indent=2)


def assert_state_dicts_close(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    assert set(actual.keys()) == set(expected.keys())
    for key, expected_value in expected.items():
        torch.testing.assert_close(actual[key], expected_value, atol=1e-6, rtol=1e-6)


def test_hf_checkpoint_conversion_roundtrip(tmp_path: Path) -> None:
    config = make_dense_multimodal_config(hf_audio_order=True)
    model = Gemma4Model(config)
    hf_config = make_hf_config_dict(config)
    hf_state = native_to_hf_dense_state_dict(model)

    hf_dir = tmp_path / "hf"
    write_sharded_hf_checkpoint(hf_dir, hf_config, hf_state)

    output_dir = tmp_path / "native"
    converted_config = convert_hf_checkpoint(hf_dir, output_dir)
    restored = Gemma4Model.from_pretrained(output_dir)

    assert converted_config.audio is not None
    assert converted_config.audio.projection_norm_before_text
    assert converted_config.text.image_token_id == 258_880
    assert converted_config.text.audio_token_id == 258_881
    assert converted_config.text.image_placeholder_token_id == -2
    assert converted_config.text.audio_placeholder_token_id == -4
    assert_state_dicts_close(restored.state_dict(), model.state_dict())


def test_hf_moe_state_dict_conversion() -> None:
    config = make_moe_text_config()
    model = Gemma4Model(config)
    hf_config = make_hf_config_dict(config, text_only=True)
    native_config = native_config_from_hf_dict(hf_config)
    hf_state = native_to_hf_moe_state_dict(model)

    converted_state = convert_hf_state_dict_to_native(native_config, hf_state)
    restored = Gemma4Model(native_config)
    restored.load_state_dict(converted_state)

    assert_state_dicts_close(restored.state_dict(), model.state_dict())


def test_jax_tree_conversion_roundtrip() -> None:
    config = make_dense_multimodal_config(hf_audio_order=False)
    model = Gemma4Model(config)
    jax_tree = native_to_jax_tree_dense(model)

    converted_state = convert_jax_tree_to_native(config, jax_tree)
    restored = Gemma4Model(config)
    restored.load_state_dict(converted_state)

    assert_state_dicts_close(restored.state_dict(), model.state_dict())
