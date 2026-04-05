from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypeAlias

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torchaudio import functional as AF

from .config import AudioConfig


AudioTensorLike: TypeAlias = np.ndarray | torch.Tensor | Sequence[float]
AudioClipInput: TypeAlias = str | Path | AudioTensorLike | tuple[AudioTensorLike, int]
AudioBatchInput: TypeAlias = AudioClipInput | Sequence[AudioClipInput]


@dataclass
class Gemma4AudioBatch:
    input_features: torch.Tensor
    input_features_mask: torch.Tensor
    num_soft_tokens_per_clip: torch.Tensor

    def to(
            self,
            device: str | torch.device,
    ) -> "Gemma4AudioBatch":
        """Move the audio batch to a target device."""
        return Gemma4AudioBatch(
            input_features=self.input_features.to(device=device),
            input_features_mask=self.input_features_mask.to(device=device),
            num_soft_tokens_per_clip=self.num_soft_tokens_per_clip.to(device=device),
        )


def _to_float32_waveform(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.dtype == torch.uint8:
        return (waveform.to(dtype=torch.float32) - 128.0) / 128.0
    if waveform.dtype == torch.int16:
        return waveform.to(dtype=torch.float32) / 32768.0
    if waveform.dtype == torch.int32:
        return waveform.to(dtype=torch.float32) / 2147483648.0
    if torch.is_floating_point(waveform):
        return waveform.to(dtype=torch.float32)
    raise ValueError(f"Unsupported audio dtype: {waveform.dtype}.")


def _to_mono_waveform(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 1:
        return waveform
    if waveform.ndim != 2:
        raise ValueError(f"Expected 1D or 2D waveform tensor, got shape {tuple(waveform.shape)}.")

    if waveform.shape[0] <= 8 and waveform.shape[1] > waveform.shape[0]:
        return waveform.mean(dim=0)
    if waveform.shape[1] <= 8 and waveform.shape[0] > waveform.shape[1]:
        return waveform.mean(dim=1)
    return waveform.mean(dim=0)


def _is_audio_clip_tensor(value: object) -> bool:
    return isinstance(value, (str, Path, np.ndarray, torch.Tensor, tuple))


def _is_scalar_sequence(value: object) -> bool:
    if not isinstance(value, Sequence) or not value:
        return False
    return all(isinstance(item, (int, float, np.integer, np.floating)) for item in value)


def _load_audio_clip(
        audio: AudioClipInput,
        *,
        sample_rate: int,
) -> tuple[torch.Tensor, int]:
    if isinstance(audio, (str, Path)):
        waveform, source_rate = _load_audio_file(audio)
    elif isinstance(audio, tuple):
        if len(audio) != 2:
            raise ValueError("Audio tuples must be `(waveform, sample_rate)` pairs.")
        raw_waveform, source_rate = audio
        waveform = torch.as_tensor(raw_waveform)
    else:
        waveform = torch.as_tensor(audio)
        source_rate = sample_rate

    waveform = _to_mono_waveform(_to_float32_waveform(waveform)).contiguous()
    if source_rate != sample_rate:
        waveform = AF.resample(waveform.unsqueeze(0), source_rate, sample_rate).squeeze(0)
        source_rate = sample_rate
    return waveform, int(source_rate)


def _load_audio_file(path: str | Path) -> tuple[torch.Tensor, int]:
    audio_path = Path(path)
    try:
        return torchaudio.load(str(audio_path))
    except (OSError, RuntimeError) as exc:
        try:
            from scipy.io import wavfile
        except ImportError:
            raise RuntimeError(
                f"Failed to load audio from {audio_path} with torchaudio and SciPy is unavailable."
            ) from exc

        if audio_path.suffix.lower() != ".wav":
            raise RuntimeError(
                f"Failed to load audio from {audio_path} with torchaudio."
            ) from exc

        source_rate, waveform = wavfile.read(audio_path)
        waveform_tensor = torch.as_tensor(waveform)
        if waveform_tensor.ndim == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0)
        elif waveform_tensor.ndim == 2:
            waveform_tensor = waveform_tensor.transpose(0, 1)
        else:
            raise ValueError(f"Unsupported WAV array shape {tuple(waveform_tensor.shape)}.")
        return waveform_tensor, int(source_rate)


def _split_audio_batch(audios: AudioBatchInput) -> list[AudioClipInput]:
    if _is_scalar_sequence(audios):
        return [audios]
    if isinstance(audios, (str, Path, np.ndarray, torch.Tensor, tuple)):
        return [audios]
    if not isinstance(audios, Sequence) or not audios:
        raise TypeError("Expected an audio clip or a non-empty sequence of audio clips.")
    if _is_scalar_sequence(audios[0]):
        return [audios]
    return list(audios)


class Gemma4AudioProcessor:
    """Waveform processor for Gemma4 audio inputs using torchaudio."""

    def __init__(
            self,
            *,
            sample_rate: int = 16_000,
            num_mel_bins: int = 128,
            frame_length: int = 320,
            hop_length: int = 160,
            mel_floor: float = 1e-3,
            max_soft_tokens: int = 750,
    ) -> None:
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.mel_floor = float(mel_floor)
        self.max_soft_tokens = max_soft_tokens
        self.frame_size_for_unfold = self.frame_length + 1
        self.pad_left = self.frame_length // 2
        self.fft_length = 1 << math.ceil(math.log2(max(self.frame_length, 1)))
        self.window = torch.hann_window(self.frame_length, periodic=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.mel_filters = AF.melscale_fbanks(
                n_freqs=self.fft_length // 2 + 1,
                f_min=0.0,
                f_max=self.sample_rate / 2.0,
                n_mels=self.num_mel_bins,
                sample_rate=self.sample_rate,
                norm=None,
                mel_scale="htk",
            )

    @classmethod
    def from_config(cls, config: AudioConfig) -> "Gemma4AudioProcessor":
        return cls(
            sample_rate=config.sample_rate,
            num_mel_bins=config.num_mel_bins,
            frame_length=config.win_length,
            hop_length=config.hop_length,
            mel_floor=config.mel_floor,
        )

    def compute_num_soft_tokens(self, waveform_length: int) -> int:
        """Compute the number of valid audio soft tokens using the original JAX formula."""
        num_mel_frames = max((waveform_length - self.frame_size_for_unfold) // self.hop_length + 1, 0)
        t = num_mel_frames
        for _ in range(2):
            t_padded = t + 2
            t = (t_padded - 3) // 2 + 1
        return min(int(t), self.max_soft_tokens)

    def extract_features(
            self,
            waveform: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Convert a mono waveform into log-mel features and a valid-frame mask."""
        waveform = waveform.to(dtype=torch.float32)
        original_length = int(waveform.shape[0])
        padded = F.pad(waveform, (self.pad_left, 0))
        if padded.shape[0] < self.frame_size_for_unfold:
            padded = F.pad(padded, (0, self.frame_size_for_unfold - padded.shape[0]))

        frames = padded.unfold(0, self.frame_size_for_unfold, self.hop_length)
        total_frames = int(frames.shape[0])
        valid_frames = max((original_length - self.frame_size_for_unfold) // self.hop_length + 1, 0)
        valid_frames = min(valid_frames, total_frames)

        frames = frames[:, :-1] * self.window.to(device=waveform.device, dtype=waveform.dtype)
        spectrum = torch.fft.rfft(frames, n=self.fft_length, dim=-1).abs()
        mel_filters = self.mel_filters.to(device=waveform.device, dtype=spectrum.dtype)
        mel_spec = spectrum @ mel_filters
        log_mel = torch.log(mel_spec + self.mel_floor)

        frame_mask = torch.arange(total_frames, device=waveform.device) < valid_frames
        return log_mel, frame_mask, self.compute_num_soft_tokens(original_length)

    def preprocess(self, audios: AudioBatchInput) -> Gemma4AudioBatch:
        """Load waveforms, compute log-mel features, and pad to a batch."""
        audio_list = _split_audio_batch(audios)
        features_list: list[torch.Tensor] = []
        mask_list: list[torch.Tensor] = []
        soft_token_counts: list[int] = []

        for audio in audio_list:
            waveform, _ = _load_audio_clip(audio, sample_rate=self.sample_rate)
            features, feature_mask, num_soft_tokens = self.extract_features(waveform)
            features_list.append(features)
            mask_list.append(feature_mask)
            soft_token_counts.append(num_soft_tokens)

        max_frames = max(int(features.shape[0]) for features in features_list)
        padded_features = []
        padded_masks = []
        for features, feature_mask in zip(features_list, mask_list):
            pad_frames = max_frames - int(features.shape[0])
            if pad_frames > 0:
                features = F.pad(features, (0, 0, 0, pad_frames), value=0.0)
                feature_mask = F.pad(feature_mask, (0, pad_frames), value=False)
            padded_features.append(features)
            padded_masks.append(feature_mask)

        return Gemma4AudioBatch(
            input_features=torch.stack(padded_features, dim=0),
            input_features_mask=torch.stack(padded_masks, dim=0),
            num_soft_tokens_per_clip=torch.tensor(soft_token_counts, dtype=torch.long),
        )
