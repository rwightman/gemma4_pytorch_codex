from __future__ import annotations

import argparse
import ast
import json
import pickle
from collections.abc import Iterable, Mapping
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from .config import (
    AttentionKind,
    AudioConfig,
    Gemma4Config,
    KVSharingConfig,
    TextConfig,
    VisionConfig,
    gemma4_26b_a4b_config,
    gemma4_31b_config,
    gemma4_e2b_config,
    gemma4_e4b_config,
)
from .model import CONFIG_NAME, Gemma4Model, SAFE_WEIGHTS_NAME, TORCH_WEIGHTS_NAME
from .tokenizer import Gemma4Tokenizer


HF_SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
HF_SAFE_WEIGHTS_NAME = "model.safetensors"
HF_CONFIG_NAME = "config.json"

_TEXT_PREFIXES = (
    "model.language_model.",
    "model.",
)
_VISION_PREFIXES = (
    "model.vision_tower.",
    "vision_tower.",
)
_VISION_EMBED_PREFIXES = (
    "model.embed_vision.",
    "embed_vision.",
)
_AUDIO_PREFIXES = (
    "model.audio_tower.",
    "audio_tower.",
)
_AUDIO_EMBED_PREFIXES = (
    "model.embed_audio.",
    "embed_audio.",
)

_AUDIO_ENCODER_PARAMETER = "AudioEncoder/encoder"
_AUDIO_ENCODER_CONFORMER = f"{_AUDIO_ENCODER_PARAMETER}/conformer/stacked_layers"
_AUDIO_ENCODER_SSCP = f"{_AUDIO_ENCODER_PARAMETER}/feature"
_TRANSFORMER_PARAMETER = "transformer"
_TRANSFORMER_EMBEDDER = f"{_TRANSFORMER_PARAMETER}/embedder"
_TRANSFORMER_FINAL_NORM = "transformer/final_norm"
_TRANSFORMER_POST_TRAINING_PREFIX = "rlx_networks/policy_network/"
_VISION_ENCODER_PARAMETER = "PatchInputVariablePoolingEncoder_0"
_VISION_ENCODER_VIT_PARAMETER = f"{_VISION_ENCODER_PARAMETER}/_model/vit"
_VISION_ENCODER_ENTRY = f"{_VISION_ENCODER_VIT_PARAMETER}/entry"
_VISION_ENCODER_STANDARDIZE = f"{_VISION_ENCODER_PARAMETER}/standardize"
_VISION_ENCODER_TRANSFORMER = f"{_VISION_ENCODER_VIT_PARAMETER}/transformer/stacked_layers/block"

_CLIP_PARAM_NAMES = {
    "clip_input_min": "input_min",
    "clip_input_max": "input_max",
    "clip_output_min": "output_min",
    "clip_output_max": "output_max",
}
_TOKENIZER_CANDIDATES = (
    "tokenizer.model",
    "tokenizer.json",
    "gemma4_cleaned_262144.model",
    "spiece.model",
    "tokenizer.spm",
)
_IGNORED_HF_SOURCE_KEYS = {
    "lm_head.weight",
}


def load_hf_config(path: str | Path) -> dict[str, Any]:
    """Load a Hugging Face config JSON from a file or checkpoint directory."""
    path = Path(path)
    config_path = path if path.is_file() and path.name == HF_CONFIG_NAME else path / HF_CONFIG_NAME
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find {HF_CONFIG_NAME} at {config_path}.")

    try:
        with config_path.open(encoding="utf-8") as f:
            return json.load(f)
    except OSError as exc:
        raise OSError(f"Failed to read HF config from {config_path}.") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid HF config JSON in {config_path}.") from exc


def native_config_from_hf_dict(config_dict: Mapping[str, Any]) -> Gemma4Config:
    """Translate a Hugging Face Gemma4 config dict to the native config."""
    if "text_config" in config_dict:
        text_data = _as_dict(config_dict["text_config"], "text_config")
        vision_data = _optional_dict(config_dict.get("vision_config"), "vision_config")
        audio_data = _optional_dict(config_dict.get("audio_config"), "audio_config")
        image_token_id = int(config_dict.get("image_token_id", 258_880))
        audio_token_id = int(config_dict.get("audio_token_id", 258_881))
    else:
        text_data = _as_dict(config_dict, "config")
        vision_data = None
        audio_data = None
        image_token_id = 258_880
        audio_token_id = 258_881

    text_config = _native_text_config_from_hf(
        text_data,
        image_token_id=image_token_id,
        audio_token_id=audio_token_id,
    )
    vision_config = None if vision_data is None else _native_vision_config_from_hf(vision_data, config_dict)
    audio_config = None if audio_data is None else _native_audio_config_from_hf(audio_data)
    return Gemma4Config(text=text_config, vision=vision_config, audio=audio_config)


def native_config_from_hf_path(path: str | Path) -> Gemma4Config:
    """Load and translate a Hugging Face Gemma4 config."""
    return native_config_from_hf_dict(load_hf_config(path))


def load_hf_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a local HF state dict from safetensors or sharded safetensors."""
    path = Path(path)
    if path.is_file():
        if path.suffix != ".safetensors":
            raise ValueError(f"Unsupported HF weight file: {path}. Expected a `.safetensors` file.")
        return load_safetensors(str(path))

    index_path = path / HF_SAFE_WEIGHTS_INDEX_NAME
    if index_path.exists():
        try:
            with index_path.open(encoding="utf-8") as f:
                index_data = json.load(f)
        except OSError as exc:
            raise OSError(f"Failed to read HF index from {index_path}.") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid HF index JSON in {index_path}.") from exc

        weight_map = _as_dict(index_data.get("weight_map"), "weight_map")
        state_dict: dict[str, torch.Tensor] = {}
        for shard_name in sorted(set(str(name) for name in weight_map.values())):
            shard_path = path / shard_name
            shard_state = load_safetensors(str(shard_path))
            state_dict.update(shard_state)
        return state_dict

    direct_path = path / HF_SAFE_WEIGHTS_NAME
    if direct_path.exists():
        return load_safetensors(str(direct_path))

    safetensor_files = sorted(
        candidate
        for candidate in path.glob("*.safetensors")
        if candidate.name != SAFE_WEIGHTS_NAME
    )
    if len(safetensor_files) == 1:
        return load_safetensors(str(safetensor_files[0]))

    torch_path = path / TORCH_WEIGHTS_NAME
    if torch_path.exists():
        try:
            return torch.load(torch_path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, pickle.UnpicklingError) as exc:
            raise RuntimeError(f"Failed to load PyTorch weights from {torch_path}.") from exc

    raise FileNotFoundError(f"Could not find HF safetensors weights under {path}.")


def convert_hf_state_dict_to_native(
        config: Gemma4Config,
        state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert a HF Gemma4 state dict into the native package layout."""
    native_state: dict[str, torch.Tensor] = {}
    mapped_source_keys: set[str] = set()

    for key, value in state_dict.items():
        if key in _IGNORED_HF_SOURCE_KEYS:
            mapped_source_keys.add(key)
            continue

        if _convert_hf_text_entry(config.text, key, value, native_state):
            mapped_source_keys.add(key)
            continue
        if config.vision is not None and _convert_hf_vision_entry(config.vision, key, value, native_state):
            mapped_source_keys.add(key)
            continue
        if config.audio is not None and _convert_hf_audio_entry(config.audio, key, value, native_state):
            mapped_source_keys.add(key)
            continue

    unexpected_source_keys = sorted(set(state_dict.keys()) - mapped_source_keys)
    if unexpected_source_keys:
        sample = ", ".join(unexpected_source_keys[:8])
        raise ValueError(f"Unmapped HF state dict keys remain: {sample}.")

    return _finalize_native_state_dict(config, native_state)


def convert_hf_checkpoint(
        input_path: str | Path,
        output_dir: str | Path,
        *,
        tokenizer_source: str | Path | None = None,
) -> Gemma4Config:
    """Convert a local HF Gemma4 checkpoint into the native package format."""
    input_path = Path(input_path)
    config = native_config_from_hf_path(input_path)
    state_dict = load_hf_state_dict(input_path)
    native_state = convert_hf_state_dict_to_native(config, state_dict)
    _save_native_checkpoint(
        config,
        native_state,
        output_dir,
        tokenizer_source=tokenizer_source or input_path,
    )
    return config


def restore_orbax_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    """Restore an Orbax checkpoint tree onto CPU."""
    checkpoint_path = Path(checkpoint_path)
    try:
        import jax
        from jax.sharding import SingleDeviceSharding
        from orbax import checkpoint as obc
        from orbax.checkpoint import args as obc_args
        from orbax.checkpoint import type_handlers
    except ImportError as exc:
        raise ImportError(
            "Orbax conversion requires `jax` and `orbax-checkpoint` to be installed."
        ) from exc

    metadata_path = checkpoint_path / "_METADATA"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find Orbax metadata at {metadata_path}.")

    try:
        with metadata_path.open("rb") as f:
            metadata = json.loads(f.read())
    except OSError as exc:
        raise OSError(f"Failed to read Orbax metadata from {metadata_path}.") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Orbax metadata JSON in {metadata_path}.") from exc

    tree_metadata = _as_dict(metadata.get("tree_metadata"), "tree_metadata")
    target: dict[str, Any] = {}
    for key_str in tree_metadata:
        keys = ast.literal_eval(key_str)
        if not isinstance(keys, (tuple, list)):
            raise ValueError(f"Invalid Orbax tree path entry: {key_str!r}.")
        cursor = target
        for key in keys[:-1]:
            key = str(key)
            cursor = cursor.setdefault(key, {})
        cursor[str(keys[-1])] = np.zeros(1, dtype=np.float32)

    device = jax.devices("cpu")[0]
    sharding = SingleDeviceSharding(device)
    restore_args_tree = _tree_map(
        target,
        lambda _: type_handlers.ArrayRestoreArgs(sharding=sharding),
    )
    restore = obc_args.PyTreeRestore(item=target, restore_args=restore_args_tree)
    checkpointer = obc.PyTreeCheckpointer()
    return checkpointer.restore(str(checkpoint_path), args=restore)


def convert_jax_tree_to_native(
        config: Gemma4Config,
        checkpoint_tree: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Convert a restored Gemma4 JAX tree into the native package layout."""
    native_state: dict[str, torch.Tensor] = {}

    for raw_path, value in _flatten_tree(checkpoint_tree):
        normalized_path = _strip_to_params(raw_path)
        if not normalized_path:
            continue

        param = normalized_path[-1]
        path_parts = normalized_path[:-1]
        if not path_parts:
            continue

        path = "/".join(path_parts)
        if path.startswith(_TRANSFORMER_POST_TRAINING_PREFIX):
            path = path[len(_TRANSFORMER_POST_TRAINING_PREFIX):]

        if config.audio is not None and path.endswith("audio_input_projection"):
            native_state["audio.to_text.weight"] = _to_torch_tensor(value).transpose(0, 1).contiguous()
            continue
        if config.vision is not None and path.endswith("mm_input_projection"):
            native_state["vision.to_text.weight"] = _to_torch_tensor(value).transpose(0, 1).contiguous()
            continue
        if config.text is not None and path.startswith(_TRANSFORMER_PARAMETER):
            _convert_jax_text_entry(config.text, path, param, value, native_state)
            continue
        if config.vision is not None and path.startswith(_VISION_ENCODER_PARAMETER):
            _convert_jax_vision_entry(config.vision, path, param, value, native_state)
            continue
        if config.audio is not None and path.startswith(_AUDIO_ENCODER_PARAMETER):
            _convert_jax_audio_entry(config.audio, path, param, value, native_state)
            continue

    return _finalize_native_state_dict(config, native_state)


def convert_orbax_checkpoint(
        checkpoint_path: str | Path,
        output_dir: str | Path,
        *,
        variant: str | None = None,
        config: Gemma4Config | None = None,
        text_only: bool = False,
        tokenizer_source: str | Path | None = None,
) -> Gemma4Config:
    """Convert an Orbax Gemma4 checkpoint into the native package format."""
    if config is None:
        if variant is None:
            raise ValueError("Orbax conversion requires either `config` or `variant`.")
        config = resolve_variant_config(variant, text_only=text_only)

    checkpoint_tree = restore_orbax_checkpoint(checkpoint_path)
    native_state = convert_jax_tree_to_native(config, checkpoint_tree)
    _save_native_checkpoint(
        config,
        native_state,
        output_dir,
        tokenizer_source=tokenizer_source,
    )
    return config


def resolve_variant_config(variant: str, *, text_only: bool = False) -> Gemma4Config:
    """Resolve a Gemma4 preset name into a native config."""
    normalized = variant.strip().lower().replace("_", "-")
    variant_map = {
        "gemma-4-e2b": gemma4_e2b_config,
        "e2b": gemma4_e2b_config,
        "gemma4-e2b": gemma4_e2b_config,
        "gemma-4-e4b": gemma4_e4b_config,
        "e4b": gemma4_e4b_config,
        "gemma4-e4b": gemma4_e4b_config,
        "gemma-4-31b": gemma4_31b_config,
        "31b": gemma4_31b_config,
        "gemma4-31b": gemma4_31b_config,
        "gemma-4-26b-a4b": gemma4_26b_a4b_config,
        "26b-a4b": gemma4_26b_a4b_config,
        "gemma4-26b-a4b": gemma4_26b_a4b_config,
    }
    try:
        builder = variant_map[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported Gemma4 variant: {variant}.") from exc
    return builder(text_only=text_only)


def main(argv: list[str] | None = None) -> int:
    """Run the local weight conversion CLI."""
    parser = argparse.ArgumentParser(description="Convert Gemma4 checkpoints into the native package format.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    hf_parser = subparsers.add_parser("hf", help="Convert a local Hugging Face Gemma4 checkpoint.")
    hf_parser.add_argument("input", type=Path, help="HF checkpoint directory or .safetensors file.")
    hf_parser.add_argument("output", type=Path, help="Output directory for the native checkpoint.")
    hf_parser.add_argument(
        "--tokenizer-source",
        type=Path,
        default=None,
        help="Optional tokenizer file or directory to copy into the output.",
    )

    orbax_parser = subparsers.add_parser("orbax", help="Convert a local Orbax Gemma4 checkpoint.")
    orbax_parser.add_argument("input", type=Path, help="Orbax checkpoint directory.")
    orbax_parser.add_argument("output", type=Path, help="Output directory for the native checkpoint.")
    orbax_parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Preset variant name such as `gemma-4-e2b` or `gemma-4-31b`.",
    )
    orbax_parser.add_argument(
        "--text-only",
        action="store_true",
        help="Resolve the preset as a text-only config.",
    )
    orbax_parser.add_argument(
        "--tokenizer-source",
        type=Path,
        default=None,
        help="Optional tokenizer file or directory to copy into the output.",
    )

    args = parser.parse_args(argv)
    if args.command == "hf":
        convert_hf_checkpoint(args.input, args.output, tokenizer_source=args.tokenizer_source)
        return 0

    convert_orbax_checkpoint(
        args.input,
        args.output,
        variant=args.variant,
        text_only=args.text_only,
        tokenizer_source=args.tokenizer_source,
    )
    return 0


def _native_text_config_from_hf(
        text_data: Mapping[str, Any],
        *,
        image_token_id: int,
        audio_token_id: int,
) -> TextConfig:
    num_layers = int(text_data["num_hidden_layers"])
    num_kv_shared_layers = int(text_data.get("num_kv_shared_layers", 0) or 0)
    layer_types = tuple(_attention_kind_from_hf(value) for value in text_data.get("layer_types", ()))
    rope_parameters = _as_dict(text_data.get("rope_parameters", {}), "rope_parameters")
    sliding_rope = _as_dict(rope_parameters.get("sliding_attention", {}), "sliding_attention")
    full_rope = _as_dict(rope_parameters.get("full_attention", {}), "full_attention")
    enable_moe = bool(text_data.get("enable_moe_block", False))
    intermediate_size = int(text_data["intermediate_size"])

    kv_sharing = None
    if num_kv_shared_layers > 0:
        kv_sharing = KVSharingConfig(frac_shared_layers=num_kv_shared_layers / num_layers)

    override_kv_shared_ffn_hidden = None
    if bool(text_data.get("use_double_wide_mlp", False)) and num_kv_shared_layers > 0:
        override_kv_shared_ffn_hidden = intermediate_size * 2

    return TextConfig(
        vocab_size=int(text_data["vocab_size"]),
        hidden_size=int(text_data["hidden_size"]),
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        num_heads=int(text_data["num_attention_heads"]),
        head_dim=int(text_data["head_dim"]),
        num_kv_heads=int(text_data["num_key_value_heads"]),
        final_logit_softcap=_optional_float(text_data.get("final_logit_softcapping")),
        layer_types=layer_types,
        sliding_window=int(text_data.get("sliding_window", 512)),
        num_global_kv_heads=_optional_int(text_data.get("num_global_key_value_heads")),
        global_head_dim=_optional_int(text_data.get("global_head_dim")),
        attention_k_eq_v_global=bool(text_data.get("attention_k_eq_v", False)),
        global_rope_proportion=_optional_float(full_rope.get("partial_rotary_factor"), 0.25),
        local_rope_proportion=_optional_float(sliding_rope.get("partial_rotary_factor"), 1.0),
        local_rope_theta=int(sliding_rope.get("rope_theta", 10_000)),
        global_rope_theta=int(full_rope.get("rope_theta", 1_000_000)),
        per_layer_input_dim=int(text_data.get("hidden_size_per_layer_input", 0) or 0),
        kv_sharing=kv_sharing,
        override_kv_shared_ffn_hidden=override_kv_shared_ffn_hidden,
        use_bidirectional_attention=text_data.get("use_bidirectional_attention"),
        enable_moe=enable_moe,
        num_experts=int(text_data.get("num_experts", 0) or 0),
        expert_dim=int(text_data.get("moe_intermediate_size", 0) or 0),
        top_k_experts=int(text_data.get("top_k_experts", 0) or 0),
        moe_dense_hidden_size=intermediate_size if enable_moe else 0,
        pad_token_id=int(text_data.get("pad_token_id", 0) or 0),
        image_token_id=image_token_id,
        audio_token_id=audio_token_id,
        rms_norm_eps=float(text_data.get("rms_norm_eps", 1e-6)),
    )


def _native_vision_config_from_hf(
        vision_data: Mapping[str, Any],
        config_dict: Mapping[str, Any],
) -> VisionConfig:
    rope_parameters = _as_dict(vision_data.get("rope_parameters", {}), "rope_parameters")
    output_length = vision_data.get("default_output_length", config_dict.get("vision_soft_tokens_per_image", 280))
    return VisionConfig(
        hidden_size=int(vision_data["hidden_size"]),
        intermediate_size=int(vision_data["intermediate_size"]),
        num_layers=int(vision_data["num_hidden_layers"]),
        num_heads=int(vision_data["num_attention_heads"]),
        num_kv_heads=int(vision_data["num_key_value_heads"]),
        head_dim=int(vision_data["head_dim"]),
        patch_size=int(vision_data.get("patch_size", 16)),
        position_embedding_size=int(vision_data.get("position_embedding_size", 10_240)),
        output_length=int(output_length),
        pooling_kernel_size=int(vision_data.get("pooling_kernel_size", 3)),
        rope_theta=float(rope_parameters.get("rope_theta", 100.0)),
        use_clipped_linears=bool(vision_data.get("use_clipped_linears", False)),
        standardize_embeddings=bool(vision_data.get("standardize", False)),
        rms_norm_eps=float(vision_data.get("rms_norm_eps", 1e-6)),
        projection_norm_eps=float(vision_data.get("rms_norm_eps", 1e-6)),
    )


def _native_audio_config_from_hf(audio_data: Mapping[str, Any]) -> AudioConfig:
    subsampling_channels = audio_data.get("subsampling_conv_channels", (128, 32))
    if not isinstance(subsampling_channels, (list, tuple)) or len(subsampling_channels) != 2:
        raise ValueError("Expected `subsampling_conv_channels` to contain exactly two values.")

    return AudioConfig(
        num_layers=int(audio_data["num_hidden_layers"]),
        hidden_size=int(audio_data["hidden_size"]),
        output_size=int(audio_data.get("output_proj_dims", 1_536)),
        num_heads=int(audio_data["num_attention_heads"]),
        left_context=int(audio_data.get("attention_context_left", 13)),
        right_context=int(audio_data.get("attention_context_right", 0)),
        chunk_size=int(audio_data.get("attention_chunk_size", 12)),
        conv_kernel_size=int(audio_data.get("conv_kernel_size", 5)),
        gradient_clipping=float(audio_data.get("gradient_clipping", 1.0e10)),
        reduction_factor=int(audio_data.get("conf_reduction_factor", 1)),
        subsampling_channels=(int(subsampling_channels[0]), int(subsampling_channels[1])),
        num_mel_bins=int(audio_data.get("num_mel_bins", 128)),
        rms_norm_eps=float(audio_data.get("rms_norm_eps", 1e-6)),
        projection_norm_before_text=True,
    )


def _convert_hf_text_entry(
        config: TextConfig,
        key: str,
        value: torch.Tensor,
        native_state: dict[str, torch.Tensor],
) -> bool:
    text_key = _strip_prefixed_key(key, _TEXT_PREFIXES)
    if text_key is None:
        return False
    if text_key.startswith(("vision_tower.", "audio_tower.", "embed_vision.", "embed_audio.")):
        return False

    if text_key == "embed_tokens.weight":
        native_state["text.token_embedding.weight"] = value.contiguous()
        return True
    if text_key == "embed_tokens_per_layer.weight":
        native_state["text.per_layer_token_embedding"] = value.view(
            value.shape[0],
            config.num_layers,
            config.per_layer_input_dim,
        ).contiguous()
        return True
    if text_key == "per_layer_model_projection.weight":
        scale = value.new_tensor(config.hidden_size**-0.5)
        native_state["text.per_layer_model_projection.weight"] = (value * scale).contiguous()
        return True
    if text_key == "per_layer_projection_norm.weight":
        native_state["text.per_layer_projection_norm.weight"] = value.contiguous()
        return True
    if text_key == "norm.weight":
        native_state["text.final_norm.weight"] = value.contiguous()
        return True

    match = re.fullmatch(r"layers\.(\d+)\.(.+)", text_key)
    if match is None:
        return False

    layer_idx = int(match.group(1))
    suffix = match.group(2)
    native_prefix = f"text.layers.{layer_idx}."
    if suffix == "self_attn.q_proj.weight":
        native_state[f"{native_prefix}attn.q_proj.weight"] = value.contiguous()
        return True
    if suffix == "self_attn.k_proj.weight":
        native_state[f"{native_prefix}attn.k_proj.weight"] = value.contiguous()
        return True
    if suffix == "self_attn.v_proj.weight":
        native_state[f"{native_prefix}attn.v_proj.weight"] = value.contiguous()
        return True
    if suffix == "self_attn.o_proj.weight":
        native_state[f"{native_prefix}attn.o_proj.weight"] = value.contiguous()
        return True
    if suffix == "self_attn.q_norm.weight":
        native_state[f"{native_prefix}attn.q_norm.weight"] = value.contiguous()
        return True
    if suffix == "self_attn.k_norm.weight":
        native_state[f"{native_prefix}attn.k_norm.weight"] = value.contiguous()
        return True
    if suffix == "input_layernorm.weight":
        native_state[f"{native_prefix}pre_attn_norm.weight"] = value.contiguous()
        return True
    if suffix == "post_attention_layernorm.weight":
        native_state[f"{native_prefix}post_attn_norm.weight"] = value.contiguous()
        return True
    if suffix == "layer_scalar":
        native_state[f"{native_prefix}layer_scalar"] = value.contiguous()
        return True
    if suffix == "per_layer_input_gate.weight":
        native_state[f"{native_prefix}per_layer_input_gate.weight"] = value.contiguous()
        return True
    if suffix == "per_layer_projection.weight":
        native_state[f"{native_prefix}per_layer_projection.weight"] = value.contiguous()
        return True
    if suffix == "post_per_layer_input_norm.weight":
        native_state[f"{native_prefix}post_per_layer_input_norm.weight"] = value.contiguous()
        return True

    if config.enable_moe:
        if suffix == "pre_feedforward_layernorm.weight":
            native_state[f"{native_prefix}pre_ffn2_norm.weight"] = value.contiguous()
            return True
        if suffix == "post_feedforward_layernorm.weight":
            native_state[f"{native_prefix}post_ffn_norm.weight"] = value.contiguous()
            return True
        if suffix == "pre_feedforward_layernorm_2.weight":
            native_state[f"{native_prefix}pre_ffn_norm.weight"] = value.contiguous()
            return True
        if suffix == "post_feedforward_layernorm_1.weight":
            native_state[f"{native_prefix}post_ffn2_norm.weight"] = value.contiguous()
            return True
        if suffix == "post_feedforward_layernorm_2.weight":
            native_state[f"{native_prefix}post_ffn1_norm.weight"] = value.contiguous()
            return True
        if suffix == "mlp.gate_proj.weight":
            native_state[f"{native_prefix}mlp2.gate_proj.weight"] = value.contiguous()
            return True
        if suffix == "mlp.up_proj.weight":
            native_state[f"{native_prefix}mlp2.up_proj.weight"] = value.contiguous()
            return True
        if suffix == "mlp.down_proj.weight":
            native_state[f"{native_prefix}mlp2.down_proj.weight"] = value.contiguous()
            return True
        if suffix == "router.proj.weight":
            native_state[f"{native_prefix}moe.router.weight"] = value.contiguous()
            return True
        if suffix == "router.scale":
            native_state[f"{native_prefix}moe.router_scale"] = value.contiguous()
            return True
        if suffix == "router.per_expert_scale":
            native_state[f"{native_prefix}moe.per_expert_scale"] = value.contiguous()
            return True
        if suffix == "experts.gate_up_proj":
            native_state[f"{native_prefix}moe.gate_up_proj"] = value.contiguous()
            return True
        if suffix == "experts.down_proj":
            native_state[f"{native_prefix}moe.down_proj"] = value.contiguous()
            return True
        return False

    if suffix == "pre_feedforward_layernorm.weight":
        native_state[f"{native_prefix}pre_ffn_norm.weight"] = value.contiguous()
        return True
    if suffix == "post_feedforward_layernorm.weight":
        native_state[f"{native_prefix}post_ffn_norm.weight"] = value.contiguous()
        return True
    if suffix == "mlp.gate_proj.weight":
        native_state[f"{native_prefix}mlp.gate_proj.weight"] = value.contiguous()
        return True
    if suffix == "mlp.up_proj.weight":
        native_state[f"{native_prefix}mlp.up_proj.weight"] = value.contiguous()
        return True
    if suffix == "mlp.down_proj.weight":
        native_state[f"{native_prefix}mlp.down_proj.weight"] = value.contiguous()
        return True
    return False


def _convert_hf_vision_entry(
        config: VisionConfig,
        key: str,
        value: torch.Tensor,
        native_state: dict[str, torch.Tensor],
) -> bool:
    vision_key = _strip_prefixed_key(key, _VISION_PREFIXES)
    if vision_key is not None:
        if vision_key == "patch_embedder.input_proj.weight":
            native_state["vision.encoder.patch_embed.input_proj.weight"] = value.contiguous()
            return True
        if vision_key == "patch_embedder.position_embedding_table":
            native_state["vision.encoder.patch_embed.position_table"] = value.transpose(0, 1).contiguous()
            return True
        if vision_key == "std_bias":
            native_state["vision.encoder.standardize.bias"] = value.contiguous()
            return True
        if vision_key == "std_scale":
            native_state["vision.encoder.standardize.scale"] = value.contiguous()
            return True
        if vision_key == "pooler.scale":
            return True

        match = re.fullmatch(r"encoder\.layers\.(\d+)\.(.+)", vision_key)
        if match is None:
            return False

        layer_idx = int(match.group(1))
        suffix = match.group(2)
        native_prefix = f"vision.encoder.layers.{layer_idx}."
        rename_map = {
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
        native_suffix = rename_map.get(suffix)
        if native_suffix is None:
            return False
        native_state[f"{native_prefix}{native_suffix}"] = value.contiguous()
        return True

    vision_embed_key = _strip_prefixed_key(key, _VISION_EMBED_PREFIXES)
    if vision_embed_key == "embedding_projection.weight":
        native_state["vision.to_text.weight"] = value.contiguous()
        return True
    return False


def _convert_hf_audio_entry(
        config: AudioConfig,
        key: str,
        value: torch.Tensor,
        native_state: dict[str, torch.Tensor],
) -> bool:
    audio_key = _strip_prefixed_key(key, _AUDIO_PREFIXES)
    if audio_key is not None:
        rename_map = {
            "subsample_conv_projection.layer0.conv.weight": "audio.encoder.subsampler.conv0.weight",
            "subsample_conv_projection.layer1.conv.weight": "audio.encoder.subsampler.conv1.weight",
            "subsample_conv_projection.layer0.norm.weight": "audio.encoder.subsampler.norm0.weight",
            "subsample_conv_projection.layer1.norm.weight": "audio.encoder.subsampler.norm1.weight",
            "subsample_conv_projection.input_proj_linear.weight": "audio.encoder.subsampler.output_proj.weight",
            "subsample_conv_projection.input_proj_linear.input_min": "audio.encoder.subsampler.output_proj.input_min",
            "subsample_conv_projection.input_proj_linear.input_max": "audio.encoder.subsampler.output_proj.input_max",
            "subsample_conv_projection.input_proj_linear.output_min": "audio.encoder.subsampler.output_proj.output_min",
            "subsample_conv_projection.input_proj_linear.output_max": "audio.encoder.subsampler.output_proj.output_max",
            "output_proj.weight": "audio.encoder.output_proj.weight",
            "output_proj.bias": "audio.encoder.output_proj.bias",
        }
        direct_target = rename_map.get(audio_key)
        if direct_target is not None:
            native_state[direct_target] = value.contiguous()
            return True

        match = re.fullmatch(r"layers\.(\d+)\.(.+)", audio_key)
        if match is None:
            return False

        layer_idx = int(match.group(1))
        suffix = match.group(2)
        native_prefix = f"audio.encoder.layers.{layer_idx}."
        layer_map = {
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
        native_suffix = layer_map.get(suffix)
        if native_suffix is None:
            return False
        native_state[f"{native_prefix}{native_suffix}"] = value.contiguous()
        return True

    audio_embed_key = _strip_prefixed_key(key, _AUDIO_EMBED_PREFIXES)
    if audio_embed_key == "embedding_projection.weight":
        native_state["audio.to_text.weight"] = value.contiguous()
        return True
    return False


def _convert_jax_text_entry(
        config: TextConfig,
        path: str,
        param: str,
        value: Any,
        native_state: dict[str, torch.Tensor],
) -> None:
    if path == _TRANSFORMER_EMBEDDER:
        if param == "input_embedding":
            native_state["text.token_embedding.weight"] = _to_torch_tensor(value)
        elif param == "per_layer_embeddings":
            native_state["text.per_layer_token_embedding"] = _to_torch_tensor(value)
        return

    if path.startswith(_TRANSFORMER_EMBEDDER):
        if path.endswith("per_layer_model_projection") and param in {"w", "kernel"}:
            tensor = _to_torch_tensor(value)
            scale = tensor.new_tensor(config.hidden_size**-0.5)
            tensor = tensor.permute(1, 2, 0).reshape(
                config.num_layers * config.per_layer_input_dim,
                config.hidden_size,
            )
            native_state["text.per_layer_model_projection.weight"] = (tensor * scale).contiguous()
        elif path.endswith("per_layer_projection_norm") and param == "scale":
            native_state["text.per_layer_projection_norm.weight"] = _to_torch_tensor(value)
        return

    if path == _TRANSFORMER_FINAL_NORM and param == "scale":
        native_state["text.final_norm.weight"] = _to_torch_tensor(value)
        return

    if path.startswith(f"{_TRANSFORMER_PARAMETER}/layer_"):
        layer_name = path.split("/")[1]
        layer_idx = int(layer_name.removeprefix("layer_"))
        _convert_jax_text_layer_entry(config, layer_idx, path, param, value, native_state)
        return

    stacked_prefix = f"{_TRANSFORMER_PARAMETER}/stacked_layers/attention_type_"
    if path.startswith(stacked_prefix):
        attention_type_index = int(path[len(stacked_prefix)])
        pattern_length = _attention_pattern_length(config.layer_types)
        stacked = _to_torch_tensor(value)
        if stacked.ndim == 0:
            stacked = stacked.view(1)
        for group_idx, matrix in enumerate(stacked):
            layer_idx = pattern_length * group_idx + attention_type_index
            _convert_jax_text_layer_entry(config, layer_idx, path, param, matrix, native_state, stacked_path=True)


def _convert_jax_text_layer_entry(
        config: TextConfig,
        layer_idx: int,
        path: str,
        param: str,
        value: Any,
        native_state: dict[str, torch.Tensor],
        *,
        stacked_path: bool = False,
) -> None:
    del stacked_path
    prefix = f"text.layers.{layer_idx}."
    tensor = _to_torch_tensor(value)

    if param == "skip_scale" and path.endswith(f"layer_{layer_idx}"):
        native_state[f"{prefix}layer_scalar"] = tensor
        return
    if param == "skip_scale" and "/stacked_layers/" in path:
        native_state[f"{prefix}layer_scalar"] = tensor
        return
    if path.endswith("attn/q_einsum") and param in {"w", "kernel"}:
        native_state[f"{prefix}attn.q_proj.weight"] = _reshape_jax_q_proj(tensor)
        return
    if path.endswith("attn/k_einsum") and param in {"w", "kernel"}:
        native_state[f"{prefix}attn.k_proj.weight"] = _reshape_jax_q_proj(tensor)
        return
    if path.endswith("attn/kv_einsum") and param in {"w", "kernel"}:
        key, value_proj = _reshape_jax_kv_proj(tensor)
        native_state[f"{prefix}attn.k_proj.weight"] = key
        native_state[f"{prefix}attn.v_proj.weight"] = value_proj
        return
    if path.endswith("attn/attn_vec_einsum") and param in {"w", "kernel"}:
        native_state[f"{prefix}attn.o_proj.weight"] = _reshape_jax_o_proj(tensor)
        return
    if path.endswith("attn/query_norm") and param == "scale":
        native_state[f"{prefix}attn.q_norm.weight"] = tensor
        return
    if path.endswith("attn/key_norm") and param == "scale":
        native_state[f"{prefix}attn.k_norm.weight"] = tensor
        return
    if path.endswith("pre_attention_norm") and param == "scale":
        native_state[f"{prefix}pre_attn_norm.weight"] = tensor
        return
    if path.endswith("post_attention_norm") and param == "scale":
        native_state[f"{prefix}post_attn_norm.weight"] = tensor
        return
    if path.endswith("per_layer_input_gate") and param in {"w", "kernel"}:
        native_state[f"{prefix}per_layer_input_gate.weight"] = tensor.transpose(0, 1).contiguous()
        return
    if path.endswith("per_layer_projection") and param in {"w", "kernel"}:
        native_state[f"{prefix}per_layer_projection.weight"] = tensor.transpose(0, 1).contiguous()
        return
    if path.endswith("post_per_layer_input_norm") and param == "scale":
        native_state[f"{prefix}post_per_layer_input_norm.weight"] = tensor
        return

    if config.enable_moe:
        if path.endswith("pre_ffw2_norm") and param == "scale":
            native_state[f"{prefix}pre_ffn2_norm.weight"] = tensor
            return
        if path.endswith("post_ffw_norm") and param == "scale":
            native_state[f"{prefix}post_ffn_norm.weight"] = tensor
            return
        if path.endswith("pre_ffw_norm") and param == "scale":
            native_state[f"{prefix}pre_ffn_norm.weight"] = tensor
            return
        if path.endswith("post_ffw1_norm") and param == "scale":
            native_state[f"{prefix}post_ffn1_norm.weight"] = tensor
            return
        if path.endswith("post_ffw2_norm") and param == "scale":
            native_state[f"{prefix}post_ffn2_norm.weight"] = tensor
            return
        if path.endswith("mlp/router_logits") and param in {"w", "kernel"}:
            native_state[f"{prefix}moe.router.weight"] = tensor.transpose(0, 1).contiguous()
            return
        if path.endswith("mlp") and param == "router_scale":
            native_state[f"{prefix}moe.router_scale"] = tensor
            return
        if path.endswith("mlp") and param == "per_expert_scale":
            native_state[f"{prefix}moe.per_expert_scale"] = tensor
            return
        if path.endswith("mlp/gating_einsum") and param in {"gating_einsum", "w", "kernel"}:
            native_state[f"{prefix}moe.gate_up_proj"] = tensor.reshape(
                tensor.shape[0],
                tensor.shape[1] * tensor.shape[2],
                tensor.shape[3],
            ).contiguous()
            return
        if path.endswith("mlp/linear") and param in {"w", "kernel"}:
            native_state[f"{prefix}moe.down_proj"] = tensor.transpose(0, 2, 1).contiguous()
            return
        if path.endswith("mlp2/gating_einsum") and param in {"w", "kernel"}:
            gate, up = tensor[0], tensor[1]
            native_state[f"{prefix}mlp2.gate_proj.weight"] = gate.contiguous()
            native_state[f"{prefix}mlp2.up_proj.weight"] = up.contiguous()
            return
        if path.endswith("mlp2/linear") and param in {"w", "kernel"}:
            native_state[f"{prefix}mlp2.down_proj.weight"] = tensor.transpose(0, 1).contiguous()
            return
        return

    if path.endswith("pre_ffw_norm") and param == "scale":
        native_state[f"{prefix}pre_ffn_norm.weight"] = tensor
        return
    if path.endswith("post_ffw_norm") and param == "scale":
        native_state[f"{prefix}post_ffn_norm.weight"] = tensor
        return
    if path.endswith("mlp/gating_einsum") and param in {"w", "kernel"}:
        gate, up = tensor[0], tensor[1]
        native_state[f"{prefix}mlp.gate_proj.weight"] = gate.contiguous()
        native_state[f"{prefix}mlp.up_proj.weight"] = up.contiguous()
        return
    if path.endswith("mlp/linear") and param in {"w", "kernel"}:
        native_state[f"{prefix}mlp.down_proj.weight"] = tensor.transpose(0, 1).contiguous()


def _convert_jax_vision_entry(
        config: VisionConfig,
        path: str,
        param: str,
        value: Any,
        native_state: dict[str, torch.Tensor],
) -> None:
    tensor = _to_torch_tensor(value)
    if path == _VISION_ENCODER_ENTRY and param == "pos_emb":
        native_state["vision.encoder.patch_embed.position_table"] = tensor
        return
    if path == f"{_VISION_ENCODER_ENTRY}/input_projection" and param in {"w", "kernel"}:
        native_state["vision.encoder.patch_embed.input_proj.weight"] = tensor.transpose(0, 1).contiguous()
        return
    if path == _VISION_ENCODER_STANDARDIZE and param == "bias":
        native_state["vision.encoder.standardize.bias"] = tensor
        return
    if path == _VISION_ENCODER_STANDARDIZE and param == "scale":
        native_state["vision.encoder.standardize.scale"] = tensor
        return
    if not path.startswith(_VISION_ENCODER_TRANSFORMER):
        return

    stacked = tensor
    if stacked.ndim == 0:
        stacked = stacked.view(1)

    for layer_idx, matrix in enumerate(stacked):
        prefix = f"vision.encoder.layers.{layer_idx}."
        if path.endswith("pre_attention_norm") and param == "scale":
            native_state[f"{prefix}input_norm.weight"] = matrix
        elif path.endswith("post_attention_norm") and param == "scale":
            native_state[f"{prefix}post_attn_norm.weight"] = matrix
        elif path.endswith("pre_ffw_norm") and param == "scale":
            native_state[f"{prefix}pre_ffn_norm.weight"] = matrix
        elif path.endswith("post_ffw_norm") and param == "scale":
            native_state[f"{prefix}post_ffn_norm.weight"] = matrix
        elif path.endswith("attn/query_norm") and param == "scale":
            native_state[f"{prefix}attn.q_norm.weight"] = matrix
        elif path.endswith("attn/key_norm") and param == "scale":
            native_state[f"{prefix}attn.k_norm.weight"] = matrix
        elif path.endswith("attn/q_einsum") and param in {"w", "kernel"}:
            native_state[f"{prefix}attn.q_proj.weight"] = _reshape_jax_q_proj(matrix)
        elif path.endswith("attn/kv_einsum") and param in {"w", "kernel"}:
            key, value_proj = _reshape_jax_kv_proj(matrix)
            native_state[f"{prefix}attn.k_proj.weight"] = key
            native_state[f"{prefix}attn.v_proj.weight"] = value_proj
        elif path.endswith("attn/attn_vec_einsum") and param in {"w", "kernel"}:
            native_state[f"{prefix}attn.o_proj.weight"] = _reshape_jax_o_proj(matrix)
        elif path.endswith("mlp/gating_einsum") and param in {"w", "kernel"}:
            native_state[f"{prefix}mlp.gate_proj.weight"] = matrix[0].contiguous()
            native_state[f"{prefix}mlp.up_proj.weight"] = matrix[1].contiguous()
        elif path.endswith("mlp/linear") and param in {"w", "kernel"}:
            native_state[f"{prefix}mlp.down_proj.weight"] = matrix.transpose(0, 1).contiguous()
        else:
            clip_suffix = _vision_jax_clip_target(path, param)
            if clip_suffix is not None:
                if isinstance(clip_suffix, tuple):
                    for suffix_name in clip_suffix:
                        native_state[f"{prefix}{suffix_name}"] = matrix.contiguous()
                else:
                    native_state[f"{prefix}{clip_suffix}"] = matrix.contiguous()


def _convert_jax_audio_entry(
        config: AudioConfig,
        path: str,
        param: str,
        value: Any,
        native_state: dict[str, torch.Tensor],
) -> None:
    if path.endswith("output_projection"):
        tensor = _to_torch_tensor(value)
        if param in {"kernel", "w"}:
            native_state["audio.encoder.output_proj.weight"] = tensor.transpose(0, 1).contiguous()
        elif param == "bias":
            native_state["audio.encoder.output_proj.bias"] = tensor
        return

    if path.startswith(_AUDIO_ENCODER_SSCP):
        _convert_jax_audio_subsampler_entry(config, path, param, value, native_state)
        return

    match = re.match(rf"{re.escape(_AUDIO_ENCODER_PARAMETER)}/conformer/stacked_layers_(\d+)/(.*)", path)
    if match is not None:
        layer_idx = int(match.group(1))
        suffix = match.group(2)
        _convert_jax_audio_layer_entry(layer_idx, suffix, param, value, native_state)
        return

    if path.startswith(f"{_AUDIO_ENCODER_CONFORMER}/"):
        _convert_jax_audio_stacked_entry(path, param, value, native_state)


def _convert_jax_audio_subsampler_entry(
        config: AudioConfig,
        path: str,
        param: str,
        value: Any,
        native_state: dict[str, torch.Tensor],
) -> None:
    tensor = _to_torch_tensor(value)
    if path.endswith("subsampling_0") and param == "kernel":
        native_state["audio.encoder.subsampler.conv0.weight"] = tensor.permute(3, 2, 0, 1).contiguous()
        return
    if path.endswith("subsampling_1") and param == "kernel":
        native_state["audio.encoder.subsampler.conv1.weight"] = tensor.permute(3, 2, 0, 1).contiguous()
        return
    if path.endswith("norm_0") and param == "scale":
        native_state["audio.encoder.subsampler.norm0.weight"] = tensor
        return
    if path.endswith("norm_1") and param == "scale":
        native_state["audio.encoder.subsampler.norm1.weight"] = tensor
        return
    if path.endswith("input_proj") and param in {"kernel", "w"}:
        native_state["audio.encoder.subsampler.output_proj.weight"] = tensor.permute(2, 0, 1).reshape(
            config.hidden_size,
            -1,
        ).contiguous()


def _convert_jax_audio_stacked_entry(
        path: str,
        param: str,
        value: Any,
        native_state: dict[str, torch.Tensor],
) -> None:
    stacked = _to_torch_tensor(value)
    if stacked.ndim == 0:
        stacked = stacked.view(1)
    suffix = path[len(_AUDIO_ENCODER_CONFORMER):].lstrip("/")
    for layer_idx, matrix in enumerate(stacked):
        _convert_jax_audio_layer_entry(layer_idx, suffix, param, matrix, native_state)


def _convert_jax_audio_layer_entry(
        layer_idx: int,
        suffix: str,
        param: str,
        value: Any,
        native_state: dict[str, torch.Tensor],
) -> None:
    prefix = f"audio.encoder.layers.{layer_idx}."
    tensor = _to_torch_tensor(value)

    direct_map = {
        ("fflayer_start/pre_layer_norm", "scale"): "ffn_start.pre_norm.weight",
        ("fflayer_start/post_layer_norm", "scale"): "ffn_start.post_norm.weight",
        ("trans_atten/pre_norm", "scale"): "attn.pre_norm.weight",
        ("trans_atten/post_norm", "scale"): "attn.post_norm.weight",
        ("lconv/ln", "scale"): "lightconv.pre_norm.weight",
        ("lconv/conv_norm", "scale"): "lightconv.conv_norm.weight",
        ("fflayer_end/pre_layer_norm", "scale"): "ffn_end.pre_norm.weight",
        ("fflayer_end/post_layer_norm", "scale"): "ffn_end.post_norm.weight",
        ("final_ln", "scale"): "final_norm.weight",
        ("trans_atten/self_atten", "per_dim_scale"): "attn.attn.per_dim_scale",
    }
    target = direct_map.get((suffix, param))
    if target is not None:
        native_state[f"{prefix}{target}"] = tensor
        return

    if suffix == "fflayer_start/ffn_layer1" and param in {"kernel", "w"}:
        native_state[f"{prefix}ffn_start.ffn1.weight"] = tensor.transpose(0, 1).contiguous()
        return
    if suffix == "fflayer_start/ffn_layer2" and param in {"kernel", "w"}:
        native_state[f"{prefix}ffn_start.ffn2.weight"] = tensor.transpose(0, 1).contiguous()
        return
    if suffix == "fflayer_end/ffn_layer1" and param in {"kernel", "w"}:
        native_state[f"{prefix}ffn_end.ffn1.weight"] = tensor.transpose(0, 1).contiguous()
        return
    if suffix == "fflayer_end/ffn_layer2" and param in {"kernel", "w"}:
        native_state[f"{prefix}ffn_end.ffn2.weight"] = tensor.transpose(0, 1).contiguous()
        return
    if suffix == "lconv/linear_start" and param in {"kernel", "w"}:
        native_state[f"{prefix}lightconv.linear_start.weight"] = tensor.transpose(0, 1).contiguous()
        return
    if suffix == "lconv/linear_end" and param in {"kernel", "w"}:
        native_state[f"{prefix}lightconv.linear_end.weight"] = tensor.transpose(0, 1).contiguous()
        return
    if suffix == "lconv/depthwise_conv1d" and param == "kernel":
        native_state[f"{prefix}lightconv.depthwise.weight"] = tensor.permute(2, 1, 0).contiguous()
        return
    if suffix == "trans_atten/self_atten/query" and param in {"kernel", "w"}:
        native_state[f"{prefix}attn.attn.q_proj.weight"] = tensor.transpose(0, 1).contiguous()
        return
    if suffix == "trans_atten/self_atten/key" and param in {"kernel", "w"}:
        native_state[f"{prefix}attn.attn.k_proj.weight"] = tensor.transpose(0, 1).contiguous()
        return
    if suffix == "trans_atten/self_atten/value" and param in {"kernel", "w"}:
        native_state[f"{prefix}attn.attn.v_proj.weight"] = tensor.transpose(0, 1).contiguous()
        return
    if suffix == "trans_atten/self_atten/query_key_value_projection" and param in {"kernel", "w"}:
        qkv = tensor.transpose(1, 0, 2, 3)
        native_state[f"{prefix}attn.attn.q_proj.weight"] = qkv[0].reshape(-1, tensor.shape[0]).contiguous()
        native_state[f"{prefix}attn.attn.k_proj.weight"] = qkv[1].reshape(-1, tensor.shape[0]).contiguous()
        native_state[f"{prefix}attn.attn.v_proj.weight"] = qkv[2].reshape(-1, tensor.shape[0]).contiguous()
        return
    if suffix == "trans_atten/self_atten/relative_position_embedding/pos_proj" and param in {"kernel", "w"}:
        native_state[f"{prefix}attn.attn.relative_position.pos_proj.weight"] = tensor.reshape(
            tensor.shape[0],
            -1,
        ).transpose(0, 1).contiguous()
        return
    if suffix == "trans_atten/self_atten/pos_proj" and param in {"kernel", "w"}:
        native_state[f"{prefix}attn.attn.relative_position.pos_proj.weight"] = tensor.reshape(
            tensor.shape[0],
            -1,
        ).transpose(0, 1).contiguous()
        return
    if suffix == "trans_atten/post" and param in {"kernel", "w"}:
        native_state[f"{prefix}attn.post.weight"] = tensor.permute(2, 0, 1).reshape(
            tensor.shape[2],
            -1,
        ).contiguous()
        return

    clip_suffix = _audio_jax_clip_target(suffix, param)
    if clip_suffix is not None:
        native_state[f"{prefix}{clip_suffix}"] = tensor.contiguous()


def _reshape_jax_q_proj(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.permute(0, 2, 1).reshape(tensor.shape[0] * tensor.shape[2], tensor.shape[1]).contiguous()


def _reshape_jax_kv_proj(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    key = tensor[0].permute(0, 2, 1).reshape(tensor.shape[1] * tensor.shape[3], tensor.shape[2]).contiguous()
    value = tensor[1].permute(0, 2, 1).reshape(tensor.shape[1] * tensor.shape[3], tensor.shape[2]).contiguous()
    return key, value


def _reshape_jax_o_proj(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.permute(2, 0, 1).reshape(tensor.shape[2], tensor.shape[0] * tensor.shape[1]).contiguous()


def _vision_jax_clip_target(path: str, param: str) -> str | tuple[str, str] | None:
    suffix = _CLIP_PARAM_NAMES.get(param)
    if suffix is None:
        return None

    if path.endswith("attn/q_einsum"):
        return f"attn.q_proj.{suffix}"
    if path.endswith("attn/kv_einsum"):
        return (
            f"attn.k_proj.{suffix}",
            f"attn.v_proj.{suffix}",
        )
    if path.endswith("attn/attn_vec_einsum"):
        return f"attn.o_proj.{suffix}"
    if path.endswith("mlp/gating_einsum"):
        return f"mlp.gate_proj.{suffix}"
    if path.endswith("mlp/linear"):
        return f"mlp.down_proj.{suffix}"
    return None


def _audio_jax_clip_target(suffix: str, param: str) -> str | None:
    clip_suffix = _CLIP_PARAM_NAMES.get(param)
    if clip_suffix is None:
        return None

    mapping = {
        "fflayer_start/ffn_layer1": f"ffn_start.ffn1.{clip_suffix}",
        "fflayer_start/ffn_layer2": f"ffn_start.ffn2.{clip_suffix}",
        "trans_atten/self_atten/query": f"attn.attn.q_proj.{clip_suffix}",
        "trans_atten/self_atten/key": f"attn.attn.k_proj.{clip_suffix}",
        "trans_atten/self_atten/value": f"attn.attn.v_proj.{clip_suffix}",
        "trans_atten/post": f"attn.post.{clip_suffix}",
        "lconv/linear_start": f"lightconv.linear_start.{clip_suffix}",
        "lconv/linear_end": f"lightconv.linear_end.{clip_suffix}",
        "fflayer_end/ffn_layer1": f"ffn_end.ffn1.{clip_suffix}",
        "fflayer_end/ffn_layer2": f"ffn_end.ffn2.{clip_suffix}",
    }
    if suffix in mapping:
        return mapping[suffix]
    if suffix == "trans_atten/self_atten/query_key_value_projection":
        return f"attn.attn.q_proj.{clip_suffix}"
    return None


def _finalize_native_state_dict(
        config: Gemma4Config,
        native_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    expected_state = _expected_native_meta_state(config)
    unexpected_keys = sorted(set(native_state.keys()) - set(expected_state.keys()))
    if unexpected_keys:
        sample = ", ".join(unexpected_keys[:8])
        raise ValueError(f"Converted unexpected native keys: {sample}.")

    missing_keys = sorted(set(expected_state.keys()) - set(native_state.keys()))
    if missing_keys:
        for key in list(missing_keys):
            default_tensor = _default_missing_tensor(key, expected_state[key])
            if default_tensor is None:
                continue
            native_state[key] = default_tensor
        missing_keys = sorted(set(expected_state.keys()) - set(native_state.keys()))
    if missing_keys:
        sample = ", ".join(missing_keys[:8])
        raise ValueError(f"Converted checkpoint is missing native keys: {sample}.")

    return {key: value.detach().cpu().contiguous() for key, value in native_state.items()}


def _save_native_checkpoint(
        config: Gemma4Config,
        state_dict: Mapping[str, torch.Tensor],
        output_dir: str | Path,
        *,
        tokenizer_source: str | Path | None = None,
) -> None:
    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Failed to create output directory {output_dir}.") from exc

    try:
        with (output_dir / CONFIG_NAME).open("w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)
    except OSError as exc:
        raise OSError(f"Failed to write native config to {output_dir / CONFIG_NAME}.") from exc

    try:
        save_safetensors(dict(state_dict), str(output_dir / SAFE_WEIGHTS_NAME))
    except OSError as exc:
        raise OSError(f"Failed to write native weights to {output_dir / SAFE_WEIGHTS_NAME}.") from exc

    tokenizer_path = _find_tokenizer_source(tokenizer_source)
    if tokenizer_path is None:
        return

    tokenizer = Gemma4Tokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(output_dir)


def _find_tokenizer_source(path: str | Path | None) -> Path | None:
    if path is None:
        return None

    candidate = Path(path)
    if candidate.is_file():
        return candidate

    for name in _TOKENIZER_CANDIDATES:
        resolved = candidate / name
        if resolved.exists():
            return resolved

    model_files = sorted(candidate.glob("*.model"))
    if model_files:
        return model_files[0]
    json_files = sorted(candidate.glob("*.json"))
    for resolved in json_files:
        if resolved.name == "tokenizer.json":
            return resolved
    return None


def _expected_native_meta_state(config: Gemma4Config) -> dict[str, torch.Tensor]:
    with torch.device("meta"):
        model = Gemma4Model(config)
    return model.state_dict()


def _default_missing_tensor(key: str, spec: torch.Tensor) -> torch.Tensor | None:
    shape = tuple(spec.shape)
    if key.endswith(".bias") and ".subsampler.norm" in key:
        return torch.zeros(shape, dtype=torch.float32)
    if key.endswith("input_min") or key.endswith("output_min"):
        return _scalar_or_full(shape, -float("inf"))
    if key.endswith("input_max") or key.endswith("output_max"):
        return _scalar_or_full(shape, float("inf"))
    return None


def _scalar_or_full(shape: tuple[int, ...], value: float) -> torch.Tensor:
    if not shape:
        return torch.tensor(value, dtype=torch.float32)
    return torch.full(shape, value, dtype=torch.float32)


def _to_torch_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()

    if hasattr(value, "dtype") and str(value.dtype) == "bfloat16":
        array = np.asarray(value, dtype=np.float32)
        return torch.tensor(array, dtype=torch.bfloat16)

    array = np.asarray(value)
    if str(array.dtype) == "bfloat16":
        return torch.tensor(array.astype(np.float32), dtype=torch.bfloat16)
    return torch.from_numpy(np.array(array, copy=True))


def _strip_prefixed_key(key: str, prefixes: tuple[str, ...]) -> str | None:
    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix):]
    return None


def _attention_kind_from_hf(value: Any) -> AttentionKind:
    if value == "sliding_attention":
        return AttentionKind.SLIDING
    if value == "full_attention":
        return AttentionKind.FULL
    raise ValueError(f"Unsupported Gemma4 attention layer type: {value!r}.")


def _attention_pattern_length(layer_types: tuple[AttentionKind, ...]) -> int:
    for period in range(1, len(layer_types) + 1):
        if len(layer_types) % period != 0:
            continue
        pattern = layer_types[:period]
        if pattern * (len(layer_types) // period) == layer_types:
            return period
    return len(layer_types)


def _optional_dict(value: Any, name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return _as_dict(value, name)


def _as_dict(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected {name} to be a mapping, got {type(value).__name__}.")
    return dict(value)


def _optional_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    return int(value)


def _optional_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    return float(value)


def _strip_to_params(path: tuple[str, ...]) -> tuple[str, ...]:
    if "params" not in path:
        return path
    index = path.index("params")
    return path[index + 1:]


def _flatten_tree(tree: Any, prefix: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(tree, Mapping):
        for key, value in tree.items():
            yield from _flatten_tree(value, prefix + (str(key),))
        return
    yield prefix, tree


def _tree_map(tree: Any, fn) -> Any:
    if isinstance(tree, Mapping):
        return {key: _tree_map(value, fn) for key, value in tree.items()}
    return fn(tree)


__all__ = [
    "convert_hf_checkpoint",
    "convert_hf_state_dict_to_native",
    "convert_jax_tree_to_native",
    "convert_orbax_checkpoint",
    "load_hf_config",
    "load_hf_state_dict",
    "main",
    "native_config_from_hf_dict",
    "native_config_from_hf_path",
    "resolve_variant_config",
    "restore_orbax_checkpoint",
]
