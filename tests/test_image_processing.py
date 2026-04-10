from __future__ import annotations

import torch

from gemma4_pt_codex import (
    Gemma4Config,
    Gemma4ImageProcessor,
    Gemma4Model,
    TextConfig,
    VisionConfig,
    get_target_dimensions,
)
from gemma4_pt_codex.vision import avg_pool_by_positions


def make_tiny_text_config() -> TextConfig:
    return TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        head_dim=8,
        num_kv_heads=2,
        global_head_dim=8,
    )


def make_tiny_vision_config() -> VisionConfig:
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


def test_get_target_dimensions_matches_expected_patch_budget() -> None:
    target_height, target_width = get_target_dimensions(
        480,
        640,
        patch_size=16,
        max_patches=280 * 3**2,
        pooling_kernel_size=3,
    )

    assert (target_height, target_width) == (672, 912)


def test_image_processor_patchifies_and_pads_images() -> None:
    processor = Gemma4ImageProcessor.from_config(make_tiny_vision_config())
    image = torch.arange(2 * 6 * 3, dtype=torch.uint8).view(2, 6, 3)

    batch = processor.preprocess(image)

    assert batch.pixel_values.shape == (1, 4, 12)
    assert batch.image_position_ids.shape == (1, 4, 2)
    assert batch.num_soft_tokens_per_image.tolist() == [3]
    assert float(batch.pixel_values[0, :3].amin()) >= 0.0
    assert float(batch.pixel_values[0, :3].amax()) <= 1.0
    assert torch.equal(batch.image_position_ids[0, -1], torch.tensor([-1, -1]))
    assert torch.equal(batch.pixel_values[0, -1], torch.zeros(12))


def test_raw_image_and_explicit_patch_paths_match() -> None:
    vision_config = make_tiny_vision_config()
    model = Gemma4Model(Gemma4Config(text=make_tiny_text_config(), vision=vision_config))
    images = torch.rand(1, 4, 4, 3)

    image_batch = model.preprocess_images(images)
    raw_tokens, raw_mask = model.encode_images_to_text(images)
    patch_tokens, patch_mask = model.encode_images_to_text(
        image_batch.pixel_values,
        image_batch.image_position_ids,
    )

    torch.testing.assert_close(raw_tokens, patch_tokens, atol=1e-6, rtol=1e-6)
    assert torch.equal(raw_mask, patch_mask)


def test_grouped_raw_image_and_explicit_patch_paths_match() -> None:
    vision_config = make_tiny_vision_config()
    model = Gemma4Model(Gemma4Config(text=make_tiny_text_config(), vision=vision_config))
    images = torch.rand(1, 2, 4, 4, 3)

    image_batch = model.preprocess_images(images)
    raw_tokens, raw_mask = model.encode_images_to_text(images)
    patch_tokens, patch_mask = model.encode_images_to_text(
        image_batch.pixel_values,
        image_batch.image_position_ids,
    )

    torch.testing.assert_close(raw_tokens, patch_tokens, atol=1e-6, rtol=1e-6)
    assert torch.equal(raw_mask, patch_mask)


def test_bfloat16_vision_accepts_float32_images() -> None:
    vision_config = make_tiny_vision_config()
    model = Gemma4Model(Gemma4Config(text=make_tiny_text_config(), vision=vision_config)).bfloat16()
    images = torch.rand(1, 4, 4, 3, dtype=torch.float32)

    tokens, mask = model.encode_images_to_text(images)

    assert tokens.dtype == torch.bfloat16
    assert mask.dtype == torch.bool


def test_vision_with_padded_patches_stays_finite() -> None:
    vision_config = make_tiny_vision_config()
    model = Gemma4Model(Gemma4Config(text=make_tiny_text_config(), vision=vision_config))
    image = torch.arange(2 * 6 * 3, dtype=torch.uint8).view(2, 6, 3)

    tokens, mask = model.encode_images_to_text(image)

    assert torch.isfinite(tokens).all()
    assert mask.dtype == torch.bool


def test_vision_encoder_resolve_patch_inputs_matches_preprocess() -> None:
    vision_config = make_tiny_vision_config()
    model = Gemma4Model(Gemma4Config(text=make_tiny_text_config(), vision=vision_config))
    images = torch.rand(1, 2, 4, 4, 3)

    image_batch = model.preprocess_images(images)
    resolved_patches, resolved_positions = model.vision.encoder.resolve_patch_inputs(images)

    torch.testing.assert_close(resolved_patches, image_batch.pixel_values, atol=1e-6, rtol=1e-6)
    assert torch.equal(resolved_positions, image_batch.image_position_ids)


def test_avg_pool_by_positions_accepts_bfloat16_inputs() -> None:
    x = torch.randn(1, 4, 8, dtype=torch.bfloat16)
    positions = torch.tensor(
        [[[0, 0], [1, 0], [0, 1], [1, 1]]],
        dtype=torch.long,
    )

    pooled, mask = avg_pool_by_positions(x, positions, 1)

    assert pooled.dtype == torch.bfloat16
    assert mask.dtype == torch.bool
    assert pooled.shape == (1, 1, 8)
