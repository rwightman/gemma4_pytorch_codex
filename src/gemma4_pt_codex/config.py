from __future__ import annotations

import dataclasses
import enum
from dataclasses import dataclass, field
from typing import Literal, TypeAlias


JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)
AttentionImpl: TypeAlias = Literal["eager", "sdpa"]


class AttentionKind(str, enum.Enum):
    FULL = "full"
    SLIDING = "sliding"


@dataclass
class KVSharingConfig:
    frac_shared_layers: float = 0.0
    share_global: bool = True
    share_local: bool = True


def make_attention_layer_types(
        pattern: tuple[AttentionKind, ...],
        num_layers: int,
) -> tuple[AttentionKind, ...]:
    repeats, remainder = divmod(num_layers, len(pattern))
    return pattern * repeats + pattern[:remainder]


def create_kv_sharing_patterns(
        kv_sharing: KVSharingConfig | None,
        num_layers: int,
        layer_types: tuple[AttentionKind, ...],
) -> list[int]:
    if kv_sharing is None:
        return list(range(num_layers))

    num_unshared_layers = int(num_layers - kv_sharing.frac_shared_layers * num_layers)
    patterns: list[int] = []
    for idx in range(num_layers):
        if idx < num_unshared_layers:
            patterns.append(idx)
            continue

        if layer_types[idx] == AttentionKind.FULL and kv_sharing.share_global:
            patterns.append(num_unshared_layers - 1)
        elif layer_types[idx] == AttentionKind.SLIDING and kv_sharing.share_local:
            patterns.append(num_unshared_layers - 2)
        else:
            patterns.append(idx)
    return patterns


@dataclass
class TextConfig:
    vocab_size: int = 262_144
    hidden_size: int = 2_304
    intermediate_size: int = 9_216
    num_layers: int = 30
    num_heads: int = 8
    head_dim: int = 256
    num_kv_heads: int = 4
    final_logit_softcap: float | None = None
    use_post_attn_norm: bool = True
    use_post_ffn_norm: bool = True
    layer_types: tuple[AttentionKind, ...] = field(default_factory=tuple)
    attn_logits_softcap: float | None = None
    sliding_window: int | None = 512
    qk_norm_with_scale: bool = True
    num_global_kv_heads: int | None = None
    global_head_dim: int | None = 512
    attention_k_eq_v_global: bool = False
    global_rope_proportion: float | None = 0.25
    local_rope_proportion: float | None = 1.0
    local_rope_theta: int = 10_000
    global_rope_theta: int = 1_000_000
    local_rope_scale: float = 1.0
    global_rope_scale: float = 1.0
    per_layer_input_dim: int = 256
    kv_sharing: KVSharingConfig | None = None
    override_kv_shared_ffn_hidden: int | None = None
    use_bidirectional_attention: Literal["vision", "all"] | None = None
    enable_moe: bool = False
    num_experts: int = 0
    expert_dim: int = 0
    top_k_experts: int = 0
    moe_dense_hidden_size: int = 0
    pad_token_id: int = 0
    image_token_id: int = 258_880
    audio_token_id: int = 258_881
    image_placeholder_token_id: int = -2
    audio_placeholder_token_id: int = -4
    rms_norm_eps: float = 1e-6
    init_std: float = 1e-2
    residual_init_std: float | None = None
    use_depth_scaled_residual_init: bool = False
    attn_impl: AttentionImpl = "eager"

    def __post_init__(self) -> None:
        if not self.layer_types:
            self.layer_types = make_attention_layer_types(
                (AttentionKind.SLIDING,) * 5 + (AttentionKind.FULL,),
                self.num_layers,
            )
        elif len(self.layer_types) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} layer types, got {len(self.layer_types)}."
            )

        if self.layer_types[-1] != AttentionKind.FULL:
            layer_types = list(self.layer_types)
            layer_types[-1] = AttentionKind.FULL
            self.layer_types = tuple(layer_types)

        _validate_attn_impl(self.attn_impl)


@dataclass
class VisionConfig:
    hidden_size: int = 768
    intermediate_size: int = 3_072
    num_layers: int = 16
    num_heads: int = 12
    num_kv_heads: int = 12
    head_dim: int = 64
    patch_size: int = 16
    position_embedding_size: int = 10_240
    output_length: int | tuple[int, ...] = 280
    pooling_kernel_size: int = 3
    rope_theta: float = 100.0
    rope_scale: float = 1.0
    use_clipped_linears: bool = False
    standardize_embeddings: bool = False
    rms_norm_eps: float = 1e-6
    init_std: float = 1e-2
    position_init_std: float = 2e-2
    residual_init_std: float | None = None
    use_depth_scaled_residual_init: bool = False
    projection_norm_eps: float = 1e-6
    attn_impl: AttentionImpl = "eager"

    def __post_init__(self) -> None:
        _validate_attn_impl(self.attn_impl)

    @property
    def max_patches(self) -> int:
        output_length = self.output_length
        if isinstance(output_length, tuple):
            output_length = max(output_length)
        return int(output_length * self.pooling_kernel_size**2)

    @property
    def num_mm_tokens_per_image(self) -> int:
        output_length = self.output_length
        if isinstance(output_length, tuple):
            output_length = max(output_length)
        return int(output_length)

    @property
    def patch_dim(self) -> int:
        return self.patch_size * self.patch_size * 3


@dataclass
class AudioConfig:
    num_layers: int = 12
    hidden_size: int = 1_024
    output_size: int = 1_536
    num_heads: int = 8
    left_context: int = 13
    right_context: int = 0
    chunk_size: int = 12
    conv_kernel_size: int = 5
    gradient_clipping: float = 1.0e10
    reduction_factor: int = 1
    subsampling_channels: tuple[int, int] = (128, 32)
    num_mel_bins: int = 128
    sample_rate: int = 16_000
    win_length: int = 320
    hop_length: int = 160
    mel_floor: float = 1e-3
    rms_norm_eps: float = 1e-6
    init_std: float = 1e-2
    residual_init_std: float | None = None
    use_depth_scaled_residual_init: bool = False
    feature_init_std: float = 1e-2
    projection_norm_before_text: bool = False


@dataclass
class Gemma4Config:
    text: TextConfig
    vision: VisionConfig | None = None
    audio: AudioConfig | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        return _to_jsonable(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "Gemma4Config":
        text = TextConfig(
            **_restore_text_config(data["text"]),
        )
        vision_data = data.get("vision")
        vision = VisionConfig(**vision_data) if vision_data is not None else None
        audio_data = data.get("audio")
        audio = AudioConfig(**audio_data) if audio_data is not None else None
        return cls(text=text, vision=vision, audio=audio)


def _to_jsonable(value: object) -> JsonValue:
    if dataclasses.is_dataclass(value):
        return _to_jsonable(dataclasses.asdict(value))
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    return value


def _validate_attn_impl(attn_impl: str) -> None:
    if attn_impl not in {"eager", "sdpa"}:
        raise ValueError(f"Unsupported attention implementation: {attn_impl}.")


def _restore_text_config(data: JsonValue) -> dict[str, object]:
    if not isinstance(data, dict):
        raise TypeError(f"Expected a text config mapping, got {type(data).__name__}.")

    restored = dict(data)
    restored["layer_types"] = tuple(AttentionKind(v) for v in restored.get("layer_types", ()))
    kv_sharing = restored.get("kv_sharing")
    if kv_sharing is not None:
        if not isinstance(kv_sharing, dict):
            raise TypeError("Expected kv_sharing to be a mapping.")
        restored["kv_sharing"] = KVSharingConfig(**kv_sharing)
    return restored


def _small_vision_config(attn_impl: AttentionImpl = "eager") -> VisionConfig:
    return VisionConfig(
        hidden_size=768,
        intermediate_size=3_072,
        num_layers=16,
        num_heads=12,
        num_kv_heads=12,
        head_dim=64,
        output_length=280,
        pooling_kernel_size=3,
        use_clipped_linears=True,
        standardize_embeddings=False,
        attn_impl=attn_impl,
    )


def _large_vision_config(attn_impl: AttentionImpl = "eager") -> VisionConfig:
    return VisionConfig(
        hidden_size=1_152,
        intermediate_size=4_304,
        num_layers=27,
        num_heads=16,
        num_kv_heads=16,
        head_dim=72,
        output_length=280,
        pooling_kernel_size=3,
        use_clipped_linears=False,
        standardize_embeddings=True,
        attn_impl=attn_impl,
    )


def _default_audio_config() -> AudioConfig:
    return AudioConfig()


def gemma4_e2b_config(
        text_only: bool = False,
        *,
        attn_impl: AttentionImpl = "eager",
) -> Gemma4Config:
    """Build the Gemma 4 E2B preset."""
    text = TextConfig(
        hidden_size=1_536,
        intermediate_size=4 * 1_536,
        num_layers=35,
        num_heads=8,
        head_dim=256,
        num_kv_heads=1,
        layer_types=make_attention_layer_types(
            (AttentionKind.SLIDING,) * 4 + (AttentionKind.FULL,),
            35,
        ),
        sliding_window=512,
        per_layer_input_dim=256,
        kv_sharing=KVSharingConfig(frac_shared_layers=20.0 / 35.0),
        override_kv_shared_ffn_hidden=4 * 1_536 * 2,
        final_logit_softcap=30.0,
        attn_impl=attn_impl,
    )
    return Gemma4Config(
        text=text,
        vision=None if text_only else _small_vision_config(attn_impl=attn_impl),
        audio=None if text_only else _default_audio_config(),
    )


def gemma4_e4b_config(
        text_only: bool = False,
        *,
        attn_impl: AttentionImpl = "eager",
) -> Gemma4Config:
    """Build the Gemma 4 E4B preset."""
    text = TextConfig(
        hidden_size=2_560,
        intermediate_size=4 * 2_560,
        num_layers=42,
        num_heads=8,
        head_dim=256,
        num_kv_heads=2,
        layer_types=make_attention_layer_types(
            (AttentionKind.SLIDING,) * 5 + (AttentionKind.FULL,),
            42,
        ),
        sliding_window=512,
        per_layer_input_dim=256,
        kv_sharing=KVSharingConfig(frac_shared_layers=18.0 / 42.0),
        final_logit_softcap=30.0,
        attn_impl=attn_impl,
    )
    return Gemma4Config(
        text=text,
        vision=None if text_only else _small_vision_config(attn_impl=attn_impl),
        audio=None if text_only else _default_audio_config(),
    )


def gemma4_31b_config(
        text_only: bool = False,
        *,
        attn_impl: AttentionImpl = "eager",
) -> Gemma4Config:
    """Build the Gemma 4 31B preset."""
    text = TextConfig(
        hidden_size=5_376,
        intermediate_size=4 * 5_376,
        num_layers=60,
        num_heads=32,
        head_dim=256,
        num_kv_heads=16,
        num_global_kv_heads=4,
        global_head_dim=512,
        attention_k_eq_v_global=True,
        layer_types=make_attention_layer_types(
            (AttentionKind.SLIDING,) * 5 + (AttentionKind.FULL,),
            60,
        ),
        sliding_window=1_024,
        per_layer_input_dim=0,
        use_bidirectional_attention="vision",
        final_logit_softcap=30.0,
        attn_impl=attn_impl,
    )
    return Gemma4Config(
        text=text,
        vision=None if text_only else _large_vision_config(attn_impl=attn_impl),
        audio=None,
    )


def gemma4_26b_a4b_config(
        text_only: bool = False,
        *,
        attn_impl: AttentionImpl = "eager",
) -> Gemma4Config:
    """Build the Gemma 4 26B-A4B MoE preset."""
    text = TextConfig(
        hidden_size=2_816,
        intermediate_size=2_112,
        num_layers=30,
        num_heads=16,
        head_dim=256,
        num_kv_heads=8,
        num_global_kv_heads=2,
        global_head_dim=512,
        attention_k_eq_v_global=True,
        layer_types=make_attention_layer_types(
            (AttentionKind.SLIDING,) * 5 + (AttentionKind.FULL,),
            30,
        ),
        sliding_window=1_024,
        per_layer_input_dim=0,
        use_bidirectional_attention="vision",
        enable_moe=True,
        num_experts=128,
        expert_dim=704,
        top_k_experts=8,
        moe_dense_hidden_size=2_112,
        final_logit_softcap=30.0,
        attn_impl=attn_impl,
    )
    return Gemma4Config(
        text=text,
        vision=None if text_only else _large_vision_config(attn_impl=attn_impl),
        audio=None,
    )
