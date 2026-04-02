from __future__ import annotations

from pathlib import Path

import sentencepiece as spm
import torch

from gemma4_pytorch_codex.config import (
    AttentionKind,
    AudioConfig,
    Gemma4Config,
    KVSharingConfig,
    TextConfig,
    VisionConfig,
)
from gemma4_pytorch_codex.model import Gemma4Model
from gemma4_pytorch_codex.text import Gemma4TextTower
from gemma4_pytorch_codex.tokenizer import Gemma4Tokenizer
from gemma4_pytorch_codex.vision import Gemma4VisionTower


def make_tiny_text_config() -> TextConfig:
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
        kv_sharing=KVSharingConfig(frac_shared_layers=0.25),
        final_logit_softcap=30.0,
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


def make_tiny_audio_config() -> AudioConfig:
    return AudioConfig(
        num_layers=2,
        hidden_size=16,
        output_size=24,
        num_heads=4,
        left_context=3,
        right_context=0,
        chunk_size=4,
        conv_kernel_size=3,
        subsampling_channels=(8, 4),
        num_mel_bins=8,
    )


def test_text_tower_forward_shapes() -> None:
    config = make_tiny_text_config()
    model = Gemma4TextTower(config)
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 0, 0],
            [7, 8, 9, 10, 11, 12, 13, 0],
        ]
    )
    attention_mask = input_ids != 0
    positions = attention_mask.long().cumsum(dim=-1).sub(1).clamp_min_(0)
    full_mask = attention_mask[:, None, :] & torch.tril(torch.ones(8, 8, dtype=torch.bool)).unsqueeze(0)
    hidden = model(
        input_ids,
        position_ids=positions,
        full_attention_mask=full_mask,
        sliding_attention_mask=full_mask,
    )
    assert hidden.shape == (2, 8, 32)
    logits = model.project_logits(hidden)
    assert logits.shape == (2, 8, 64)


def test_vision_tower_is_reusable() -> None:
    text_config = make_tiny_text_config()
    vision = Gemma4VisionTower(make_tiny_vision_config(), text_hidden_size=text_config.hidden_size)
    images = torch.rand(1, 2, 4, 4, 3)
    tokens, mask = vision.encode_to_text(images)
    assert tokens.shape == (1, 8, text_config.hidden_size)
    assert mask.shape == (1, 8)
    assert mask.all()


def test_audio_tower_shapes() -> None:
    audio = make_tiny_audio_config()
    model = Gemma4Model(Gemma4Config(text=make_tiny_text_config(), audio=audio))
    features = torch.randn(2, 20, audio.num_mel_bins)
    feature_mask = torch.ones(2, 20, dtype=torch.bool)
    tokens, mask = model.encode_audio_to_text(features, feature_mask)
    assert tokens.ndim == 3
    assert mask.shape[:2] == tokens.shape[:2]
    assert tokens.shape[-1] == model.config.text.hidden_size
    assert mask.any()


def test_multimodal_placeholder_merge() -> None:
    config = Gemma4Config(
        text=make_tiny_text_config(),
        vision=make_tiny_vision_config(),
    )
    model = Gemma4Model(config)
    input_ids = torch.tensor([[1, 2, -2, -2, -2, -2, 3, 0]])
    images = torch.rand(1, 4, 4, 3)
    vision_tokens, vision_mask = model.encode_images_to_text(images)
    out = model(
        input_ids,
        attention_mask=input_ids != 0,
        vision_tokens=vision_tokens,
        vision_token_mask=vision_mask,
        return_hidden_states=True,
    )
    assert out.logits.shape == (1, 8, config.text.vocab_size)
    assert out.hidden_states is not None
    assert out.hidden_states.shape == (1, 8, config.text.hidden_size)


def test_kv_cache_matches_full_forward() -> None:
    model = Gemma4Model(Gemma4Config(text=make_tiny_text_config()))
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    full_out = model(input_ids, attention_mask=torch.ones_like(input_ids, dtype=torch.bool))

    prefill_ids = input_ids[:, :5]
    prefill_out = model(
        prefill_ids,
        attention_mask=torch.ones_like(prefill_ids, dtype=torch.bool),
        return_kv_cache=True,
    )
    kv_cache = prefill_out.kv_cache
    assert kv_cache is not None

    for pos in range(5, input_ids.shape[1]):
        step_ids = input_ids[:, pos : pos + 1]
        step_out = model(
            step_ids,
            attention_mask=torch.ones_like(step_ids, dtype=torch.bool),
            kv_cache=kv_cache,
            return_kv_cache=True,
        )
        kv_cache = step_out.kv_cache
        assert kv_cache is not None
        torch.testing.assert_close(
            step_out.logits[:, -1],
            full_out.logits[:, pos],
            atol=1e-5,
            rtol=1e-5,
        )


def test_generate_with_cache() -> None:
    model = Gemma4Model(Gemma4Config(text=make_tiny_text_config()))
    prompt = torch.tensor([[1, 2, 3, 4]])
    generated = model.generate(prompt, max_new_tokens=3, do_sample=False)
    assert generated.shape == (1, 7)


def _train_tiny_sentencepiece_model(tmp_path: Path) -> Path:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(
        "\n".join(
            [
                "hello world",
                "the quick brown fox jumps over the lazy dog",
                "gemma four tokenizer test",
                "audio image tool response turn",
            ]
        )
    )
    model_prefix = tmp_path / "toy_tokenizer"
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=64,
        model_type="bpe",
        pad_id=0,
        eos_id=1,
        bos_id=2,
        unk_id=3,
        user_defined_symbols=[
            "<|turn>",
            "<turn|>",
            "<|image|>",
            "<|image>",
            "<image|>",
            "<|audio|>",
            "<|audio>",
            "<audio|>",
            "<|tool_response>",
            "<tool_response|>",
        ],
    )
    return model_prefix.with_suffix(".model")


def test_tokenizer_and_pretrained_roundtrip(tmp_path: Path) -> None:
    tokenizer_model = _train_tiny_sentencepiece_model(tmp_path)
    tokenizer = Gemma4Tokenizer(tokenizer_model)

    encoded = tokenizer("hello world", add_bos=True, return_tensors="pt")
    assert encoded["input_ids"].shape == (1, encoded["attention_mask"].shape[1])
    assert encoded["attention_mask"].dtype == torch.bool

    model = Gemma4Model(Gemma4Config(text=make_tiny_text_config()))
    model.eval()

    save_dir = tmp_path / "pretrained"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    reloaded_model = Gemma4Model.from_pretrained(save_dir)
    reloaded_model.eval()
    reloaded_tokenizer = Gemma4Tokenizer.from_pretrained(save_dir)

    with torch.no_grad():
        original = model(
            encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        ).logits
        restored = reloaded_model(
            encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        ).logits

    torch.testing.assert_close(original, restored)

    generated = reloaded_model.generate_text(
        reloaded_tokenizer,
        "hello world",
        max_new_tokens=2,
        do_sample=False,
    )
    assert isinstance(generated, str)
