from __future__ import annotations

import math
from pathlib import Path
import types
import wave

import sentencepiece as spm
import torch

from gemma4_pt_codex import Gemma4AudioProcessor, Gemma4Config, Gemma4Model, Gemma4Tokenizer
from gemma4_pt_codex.config import AudioConfig, TextConfig


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


def make_waveform(num_samples: int = 16_000) -> torch.Tensor:
    time = torch.arange(num_samples, dtype=torch.float32) / 16_000.0
    return torch.sin(2.0 * math.pi * 440.0 * time)


def train_tiny_sentencepiece_model(tmp_path: Path) -> Path:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(
        "\n".join(
            [
                "hello world",
                "audio prompt response turn",
                "simple transcription example",
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
            "<|audio|>",
            "<|audio>",
            "<audio|>",
        ],
    )
    return model_prefix.with_suffix(".model")


def write_pcm_wav(path: Path, waveform: torch.Tensor, sample_rate: int = 16_000) -> None:
    pcm = (waveform.clamp(-1.0, 1.0) * 32767.0).to(dtype=torch.int16).cpu().numpy()
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


def test_audio_processor_extracts_features_and_soft_token_counts() -> None:
    processor = Gemma4AudioProcessor.from_config(make_tiny_audio_config())

    batch = processor.preprocess(make_waveform())

    assert batch.input_features.ndim == 3
    assert batch.input_features.shape[0] == 1
    assert batch.input_features.shape[-1] == 8
    assert batch.input_features_mask.dtype == torch.bool
    assert int(batch.num_soft_tokens_per_clip[0]) > 0


def test_audio_processor_accepts_wav_path(tmp_path: Path) -> None:
    processor = Gemma4AudioProcessor.from_config(make_tiny_audio_config())
    wav_path = tmp_path / "clip.wav"
    write_pcm_wav(wav_path, make_waveform())

    batch = processor.preprocess(wav_path)

    assert batch.input_features.shape[0] == 1
    assert batch.input_features_mask.any()


def test_prepare_inputs_expands_audio_into_internal_soft_tokens(tmp_path: Path) -> None:
    tokenizer = Gemma4Tokenizer(train_tiny_sentencepiece_model(tmp_path))
    model = Gemma4Model(
        Gemma4Config(
            text=make_tiny_text_config(),
            audio=make_tiny_audio_config(),
        )
    )

    batch = model.prepare_inputs(
        tokenizer,
        "hello <|audio|> world",
        audios=make_waveform(),
    )

    placeholder_mask = batch.input_ids == model.config.text.audio_placeholder_token_id
    assert batch.audio_tokens is not None
    assert batch.audio_token_mask is not None
    assert int(placeholder_mask.sum()) == int(batch.audio_token_mask.sum())
    assert batch.audio_tokens.shape[-1] == model.config.text.hidden_size


def test_prepare_inputs_casts_audio_features_to_model_dtype(tmp_path: Path) -> None:
    tokenizer = Gemma4Tokenizer(train_tiny_sentencepiece_model(tmp_path))
    model = Gemma4Model(
        Gemma4Config(
            text=make_tiny_text_config(),
            audio=make_tiny_audio_config(),
        )
    ).to(dtype=torch.bfloat16)

    seen: dict[str, torch.dtype] = {}

    def fake_encode_audio_to_text(
            self: Gemma4Model,
            features: torch.Tensor,
            feature_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seen["dtype"] = features.dtype
        tokens = torch.zeros(
            features.shape[0],
            1,
            self.config.text.hidden_size,
            dtype=self.text.token_embedding.weight.dtype,
            device=features.device,
        )
        mask = torch.ones(features.shape[0], 1, dtype=torch.bool, device=features.device)
        return tokens, mask

    model.encode_audio_to_text = types.MethodType(fake_encode_audio_to_text, model)

    batch = model.prepare_inputs(
        tokenizer,
        "hello <|audio|> world",
        audios=make_waveform(),
    )

    assert seen["dtype"] == torch.bfloat16
    assert batch.audio_tokens is not None


def test_generate_text_accepts_audio(tmp_path: Path) -> None:
    tokenizer = Gemma4Tokenizer(train_tiny_sentencepiece_model(tmp_path))
    model = Gemma4Model(
        Gemma4Config(
            text=make_tiny_text_config(),
            audio=make_tiny_audio_config(),
        )
    )
    model.eval()

    generated = model.generate_text(
        tokenizer,
        "hello <|audio|> world",
        audios=make_waveform(),
        max_new_tokens=2,
        do_sample=False,
    )

    assert isinstance(generated, str)
