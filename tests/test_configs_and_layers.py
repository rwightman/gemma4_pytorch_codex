from __future__ import annotations

import torch

from gemma4_pt_codex import (
    Gemma4Config,
    Gemma4Model,
    TextConfig,
    gemma4_26b_a4b_config,
    gemma4_31b_config,
    gemma4_e2b_config,
    gemma4_e4b_config,
)
from gemma4_pt_codex.layers import (
    RMSNorm,
    VisionRMSNorm,
    build_positions_from_mask,
    create_sliding_mask,
    make_causal_bidirectional_mask,
    merge_flat_embeddings,
    repeat_kv,
    safe_token_ids,
)


def make_small_config() -> Gemma4Config:
    return Gemma4Config(
        text=TextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_layers=3,
            num_heads=4,
            head_dim=4,
            num_kv_heads=2,
        )
    )


def test_config_roundtrip_restores_nested_types() -> None:
    config = gemma4_e2b_config()
    restored = Gemma4Config.from_dict(config.to_dict())

    assert restored.text.hidden_size == config.text.hidden_size
    assert restored.text.layer_types == config.text.layer_types
    assert restored.text.kv_sharing is not None
    assert config.text.kv_sharing is not None
    assert restored.text.kv_sharing.frac_shared_layers == config.text.kv_sharing.frac_shared_layers
    assert restored.vision is not None
    assert config.vision is not None
    assert restored.vision.hidden_size == config.vision.hidden_size
    assert restored.audio is not None
    assert config.audio is not None
    assert restored.audio.hidden_size == config.audio.hidden_size


def test_main_presets_are_instantiable_on_meta() -> None:
    presets = {
        "e2b": gemma4_e2b_config(),
        "e4b": gemma4_e4b_config(),
        "31b": gemma4_31b_config(),
        "26b_a4b": gemma4_26b_a4b_config(),
    }

    with torch.device("meta"):
        models = {
            name: Gemma4Model(config)
            for name, config in presets.items()
        }

    assert models["e2b"].vision is not None
    assert models["e2b"].audio is not None
    assert models["e4b"].vision is not None
    assert models["e4b"].audio is not None
    assert models["31b"].vision is not None
    assert models["31b"].audio is None
    assert models["26b_a4b"].vision is not None
    assert models["26b_a4b"].audio is None
    assert models["26b_a4b"].text.config.enable_moe


def test_main_presets_support_text_only_variant() -> None:
    for build_config in (
        gemma4_e2b_config,
        gemma4_e4b_config,
        gemma4_31b_config,
        gemma4_26b_a4b_config,
    ):
        config = build_config(text_only=True)
        assert config.vision is None
        assert config.audio is None

        with torch.device("meta"):
            model = Gemma4Model(config)

        assert model.vision is None
        assert model.audio is None


def test_preset_builders_accept_attn_impl_override() -> None:
    config = gemma4_e2b_config(attn_impl="sdpa")
    assert config.text.attn_impl == "sdpa"
    assert config.vision is not None
    assert config.vision.attn_impl == "sdpa"


def test_safe_token_ids_clamps_invalid_entries() -> None:
    token_ids = torch.tensor([3, -1, 7, 99])
    safe_ids = safe_token_ids(token_ids, vocab_size=8)

    assert safe_ids.tolist() == [3, 0, 7, 0]


def test_build_positions_from_mask_respects_padding() -> None:
    mask = torch.tensor(
        [
            [True, True, False, True],
            [False, True, True, False],
        ]
    )
    positions = build_positions_from_mask(mask)

    expected = torch.tensor(
        [
            [0, 1, 0, 2],
            [0, 0, 1, 0],
        ]
    )
    assert torch.equal(positions, expected)


def test_make_causal_bidirectional_mask_unblocks_same_segment() -> None:
    input_mask = torch.tensor([[True, True, True, True]])
    bidirectional_mask = torch.tensor([[False, True, True, False]])
    mask = make_causal_bidirectional_mask(input_mask, bidirectional_mask=bidirectional_mask)

    expected = torch.tensor(
        [
            [
                [True, False, False, False],
                [True, True, True, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        ]
    )
    assert torch.equal(mask, expected)


def test_create_sliding_mask_limits_context() -> None:
    positions = torch.tensor([[0, 1, 2, 3]])
    mask = create_sliding_mask(positions, sliding_window=2)

    expected = torch.tensor(
        [
            [
                [True, True, False, False],
                [True, True, True, False],
                [False, True, True, True],
                [False, False, True, True],
            ]
        ]
    )
    assert torch.equal(mask, expected)


def test_repeat_kv_repeats_head_groups() -> None:
    hidden_states = torch.arange(2 * 1 * 3 * 2, dtype=torch.float32).view(2, 1, 3, 2)
    repeated = repeat_kv(hidden_states, repeats=4)

    assert repeated.shape == (2, 4, 3, 2)
    for head_idx in range(4):
        torch.testing.assert_close(repeated[:, head_idx], hidden_states[:, 0])


def test_merge_flat_embeddings_replaces_placeholder_slots() -> None:
    text_embeddings = torch.zeros(1, 5, 3)
    multimodal_embeddings = torch.tensor(
        [
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
            ]
        ]
    )
    target_mask = torch.tensor([[False, True, False, True, False]])
    multimodal_mask = torch.tensor([[True, False, True]])
    merged = merge_flat_embeddings(
        text_embeddings,
        multimodal_embeddings,
        target_mask,
        multimodal_mask,
    )

    expected = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                [3.0, 3.0, 3.0],
                [0.0, 0.0, 0.0],
            ]
        ]
    )
    torch.testing.assert_close(merged, expected)


def test_merge_flat_embeddings_raises_on_count_mismatch() -> None:
    text_embeddings = torch.zeros(1, 4, 2)
    multimodal_embeddings = torch.zeros(1, 1, 2)
    target_mask = torch.tensor([[False, True, True, False]])

    try:
        merge_flat_embeddings(text_embeddings, multimodal_embeddings, target_mask)
    except ValueError as exc:
        assert "Mismatch between placeholder count and multimodal token count" in str(exc)
    else:
        raise AssertionError("Expected placeholder/token count mismatch to raise ValueError.")


def test_rms_norm_without_scale_matches_manual_result() -> None:
    norm = RMSNorm(3, eps=1e-6, with_scale=False)
    x = torch.tensor([[1.0, 2.0, 3.0]])
    out = norm(x)

    manual = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(out, manual)


def test_vision_rms_norm_is_zero_init() -> None:
    norm = VisionRMSNorm(8)
    assert torch.equal(norm.weight, torch.zeros_like(norm.weight))
