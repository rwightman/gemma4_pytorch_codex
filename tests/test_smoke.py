from __future__ import annotations

from pathlib import Path

import sentencepiece as spm
import torch
from tokenizers import Tokenizer as FastTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from gemma4_pt_codex.config import (
    AttentionKind,
    AudioConfig,
    Gemma4Config,
    KVSharingConfig,
    TextConfig,
    VisionConfig,
)
from gemma4_pt_codex.model import Gemma4Model, Gemma4PreparedInputs
from gemma4_pt_codex.processing import Gemma4Processor
from gemma4_pt_codex.text import Gemma4TextTower
from gemma4_pt_codex.tokenizer import Gemma4Tokenizer
from gemma4_pt_codex.vision import Gemma4VisionTower


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


def make_tiny_text_config_for_impl(attn_impl: str) -> TextConfig:
    config = make_tiny_text_config()
    config.attn_impl = attn_impl
    return config


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


def make_tiny_vision_config_for_impl(attn_impl: str) -> VisionConfig:
    config = make_tiny_vision_config()
    config.attn_impl = attn_impl
    return config


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
    for attn_impl in ("eager", "sdpa"):
        config = make_tiny_text_config_for_impl(attn_impl)
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


def test_text_embedding_wraps_internal_negative_placeholder_ids() -> None:
    model = Gemma4TextTower(make_tiny_text_config())
    input_ids = torch.tensor([[1, -2, -4, 0]], dtype=torch.long)

    embedded = model.embed_tokens(input_ids)
    expected_ids = torch.tensor(
        [[1, model.config.vocab_size - 2, model.config.vocab_size - 4, 0]],
        dtype=torch.long,
    )
    expected = model.token_embedding(expected_ids)

    torch.testing.assert_close(embedded, expected, atol=1e-6, rtol=1e-6)


def test_text_tower_sdpa_matches_eager() -> None:
    eager = Gemma4TextTower(make_tiny_text_config_for_impl("eager"))
    sdpa = Gemma4TextTower(make_tiny_text_config_for_impl("sdpa"))
    sdpa.load_state_dict(eager.state_dict())

    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 0, 0],
            [7, 8, 9, 10, 11, 12, 13, 0],
        ]
    )
    attention_mask = input_ids != 0
    positions = attention_mask.long().cumsum(dim=-1).sub(1).clamp_min_(0)
    full_mask = attention_mask[:, None, :] & torch.tril(torch.ones(8, 8, dtype=torch.bool)).unsqueeze(0)

    eager_hidden = eager(
        input_ids,
        position_ids=positions,
        full_attention_mask=full_mask,
        sliding_attention_mask=full_mask,
    )
    sdpa_hidden = sdpa(
        input_ids,
        position_ids=positions,
        full_attention_mask=full_mask,
        sliding_attention_mask=full_mask,
    )

    torch.testing.assert_close(eager_hidden, sdpa_hidden, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        eager.project_logits(eager_hidden),
        sdpa.project_logits(sdpa_hidden),
        atol=1e-5,
        rtol=1e-5,
    )


def test_vision_tower_is_reusable() -> None:
    for attn_impl in ("eager", "sdpa"):
        text_config = make_tiny_text_config_for_impl(attn_impl)
        vision = Gemma4VisionTower(
            make_tiny_vision_config_for_impl(attn_impl),
            text_hidden_size=text_config.hidden_size,
        )
        images = torch.rand(1, 2, 4, 4, 3)
        tokens, mask = vision.encode_to_text(images)
        assert tokens.shape == (1, 8, text_config.hidden_size)
        assert mask.shape == (1, 8)
        assert mask.all()


def test_vision_sdpa_matches_eager() -> None:
    eager = Gemma4VisionTower(
        make_tiny_vision_config_for_impl("eager"),
        text_hidden_size=make_tiny_text_config().hidden_size,
    )
    sdpa = Gemma4VisionTower(
        make_tiny_vision_config_for_impl("sdpa"),
        text_hidden_size=make_tiny_text_config().hidden_size,
    )
    sdpa.load_state_dict(eager.state_dict())

    images = torch.rand(1, 2, 4, 4, 3)
    eager_tokens, eager_mask = eager.encode_to_text(images)
    sdpa_tokens, sdpa_mask = sdpa.encode_to_text(images)

    torch.testing.assert_close(eager_tokens, sdpa_tokens, atol=1e-5, rtol=1e-5)
    assert torch.equal(eager_mask, sdpa_mask)


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


def test_audio_tower_bfloat16_shapes() -> None:
    audio = make_tiny_audio_config()
    model = Gemma4Model(Gemma4Config(text=make_tiny_text_config(), audio=audio)).to(dtype=torch.bfloat16)
    features = torch.randn(2, 20, audio.num_mel_bins, dtype=torch.bfloat16)
    feature_mask = torch.ones(2, 20, dtype=torch.bool)
    tokens, mask = model.encode_audio_to_text(features, feature_mask)
    assert tokens.ndim == 3
    assert mask.shape[:2] == tokens.shape[:2]
    assert tokens.dtype == torch.bfloat16
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
    for attn_impl in ("eager", "sdpa"):
        model = Gemma4Model(Gemma4Config(text=make_tiny_text_config_for_impl(attn_impl)))
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
    for attn_impl in ("eager", "sdpa"):
        model = Gemma4Model(Gemma4Config(text=make_tiny_text_config_for_impl(attn_impl)))
        prompt = torch.tensor([[1, 2, 3, 4]])
        generated = model.generate(prompt, max_new_tokens=3, do_sample=False)
        assert generated.shape == (1, 7)


def test_generate_with_cache_bfloat16() -> None:
    for attn_impl in ("eager", "sdpa"):
        model = Gemma4Model(Gemma4Config(text=make_tiny_text_config_for_impl(attn_impl)))
        model = model.to(dtype=torch.bfloat16)
        prompt = torch.tensor([[1, 2, 3, 4]])
        generated = model.generate(prompt, max_new_tokens=3, do_sample=False)
        assert generated.shape == (1, 7)


def test_from_pretrained_attn_impl_override(tmp_path: Path) -> None:
    model = Gemma4Model(Gemma4Config(text=make_tiny_text_config()))
    save_dir = tmp_path / "override_attn_impl"
    model.save_pretrained(save_dir)

    reloaded = Gemma4Model.from_pretrained(save_dir, attn_impl="sdpa")
    assert reloaded.config.text.attn_impl == "sdpa"

    prompt = torch.tensor([[1, 2, 3, 4]])
    generated = reloaded.generate(prompt, max_new_tokens=2, do_sample=False)
    assert generated.shape == (1, 6)


def test_init_non_persistent_buffers_recurses_into_audio() -> None:
    model = Gemma4Model(
        Gemma4Config(
            text=make_tiny_text_config(),
            audio=make_tiny_audio_config(),
        )
    )
    audio_attn = model.audio.encoder.layers[0].attn.attn
    audio_attn.causal_valid_mask = torch.zeros_like(audio_attn.causal_valid_mask)

    model.init_non_persistent_buffers()

    assert audio_attn.causal_valid_mask.any()
    assert audio_attn.causal_valid_mask.shape == (
        model.config.audio.chunk_size,
        model.config.audio.chunk_size + max(0, model.config.audio.left_context - 1) + model.config.audio.right_context,
    )


def test_from_pretrained_rebuilds_audio_non_persistent_buffers(tmp_path: Path) -> None:
    model = Gemma4Model(
        Gemma4Config(
            text=make_tiny_text_config(),
            audio=make_tiny_audio_config(),
        )
    )
    save_dir = tmp_path / "audio_reload"
    model.save_pretrained(save_dir)

    reloaded = Gemma4Model.from_pretrained(save_dir, dtype=torch.bfloat16)
    audio_attn = reloaded.audio.encoder.layers[0].attn.attn

    assert audio_attn.causal_valid_mask.dtype == torch.bool
    assert not audio_attn.causal_valid_mask.is_meta
    assert reloaded.audio.to_text.weight.dtype == torch.bfloat16


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


def _build_tiny_tokenizer_json(tmp_path: Path) -> Path:
    vocab = {
        "<pad>": 0,
        "<eos>": 1,
        "<bos>": 2,
        "<unk>": 3,
        "<mask>": 4,
        "<|image|>": 5,
        "<|image>": 6,
        "<image|>": 7,
        "<|audio|>": 8,
        "<|audio>": 9,
        "<audio|>": 10,
        "hello": 11,
        "world": 12,
    }
    tokenizer = FastTokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    return tokenizer_path


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


def test_tokenizer_json_and_pretrained_roundtrip(tmp_path: Path) -> None:
    tokenizer_file = _build_tiny_tokenizer_json(tmp_path)
    tokenizer = Gemma4Tokenizer(tokenizer_file)

    encoded = tokenizer("hello world", add_bos=True, return_tensors="pt")
    assert encoded["input_ids"].shape == (1, encoded["attention_mask"].shape[1])
    assert encoded["attention_mask"].dtype == torch.bool
    assert tokenizer.bos_token_id == 2
    assert tokenizer.eos_token_id == 1

    save_dir = tmp_path / "fast_pretrained"
    tokenizer.save_pretrained(save_dir)

    reloaded_tokenizer = Gemma4Tokenizer.from_pretrained(save_dir)
    assert reloaded_tokenizer.vocab_size == tokenizer.vocab_size
    assert reloaded_tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True) == "hello world"

    model = Gemma4Model(Gemma4Config(text=make_tiny_text_config()))
    model.eval()
    generated = model.generate_text(
        reloaded_tokenizer,
        "hello world",
        max_new_tokens=2,
        do_sample=False,
    )
    assert isinstance(generated, str)


def test_processor_expands_visible_image_placeholder_tokens(tmp_path: Path) -> None:
    tokenizer = Gemma4Tokenizer(_build_tiny_tokenizer_json(tmp_path))
    processor = Gemma4Processor(
        tokenizer=tokenizer,
        text_config=make_tiny_text_config(),
    )

    expanded = processor.expand_image_placeholders(
        [11, 5, 12],
        [3],
        begin_image_token_id=6,
        end_image_token_id=7,
        double_newline_token_ids=[108],
    )

    assert expanded == [11, 108, 6, -2, -2, -2, 7, 108, 12]


def test_prepare_inputs_expands_images_into_internal_soft_tokens(tmp_path: Path) -> None:
    tokenizer = Gemma4Tokenizer(_train_tiny_sentencepiece_model(tmp_path))
    model = Gemma4Model(
        Gemma4Config(
            text=make_tiny_text_config(),
            vision=make_tiny_vision_config(),
        )
    )
    images = torch.rand(1, 4, 4, 3)

    batch = model.prepare_inputs(
        tokenizer,
        "hello <|image|> world",
        images=images,
    )

    assert isinstance(batch, Gemma4PreparedInputs)
    placeholder_mask = batch.input_ids == model.config.text.image_placeholder_token_id
    assert int(placeholder_mask.sum()) == 4
    assert batch.vision_tokens is not None
    assert batch.vision_token_mask is not None
    assert batch.vision_tokens.shape == (1, 4, model.config.text.hidden_size)
    assert int(batch.vision_token_mask.sum()) == 4

    forward_kwargs = batch.as_forward_kwargs()
    assert torch.equal(forward_kwargs["input_ids"], batch.input_ids)
    assert torch.equal(forward_kwargs["attention_mask"], batch.attention_mask)
    assert torch.equal(forward_kwargs["vision_tokens"], batch.vision_tokens)
    assert torch.equal(forward_kwargs["vision_token_mask"], batch.vision_token_mask)
    assert torch.equal(batch["input_ids"], batch.input_ids)
    assert batch.get("vision_tokens") is batch.vision_tokens


def test_prepared_inputs_to_preserves_masks_and_casts_embeddings() -> None:
    prepared = Gemma4PreparedInputs(
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        attention_mask=torch.tensor([[True, True, True]]),
        vision_tokens=torch.randn(1, 2, 4, dtype=torch.float32),
        vision_token_mask=torch.tensor([[True, False]]),
    )

    moved = prepared.to("cpu", dtype=torch.bfloat16)

    assert moved.input_ids.dtype == torch.long
    assert moved.attention_mask.dtype == torch.bool
    assert moved.vision_tokens is not None
    assert moved.vision_tokens.dtype == torch.bfloat16
    assert moved.vision_token_mask is not None
    assert moved.vision_token_mask.dtype == torch.bool


def test_generate_text_accepts_images(tmp_path: Path) -> None:
    tokenizer = Gemma4Tokenizer(_train_tiny_sentencepiece_model(tmp_path))
    model = Gemma4Model(
        Gemma4Config(
            text=make_tiny_text_config(),
            vision=make_tiny_vision_config(),
        )
    )
    model.eval()

    generated = model.generate_text(
        tokenizer,
        "hello <|image|> world",
        images=torch.rand(1, 4, 4, 3),
        max_new_tokens=2,
        do_sample=False,
    )

    assert isinstance(generated, str)
