from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from .audio import Gemma4AudioTower
from .audio_processing import AudioBatchInput, Gemma4AudioBatch, Gemma4AudioProcessor
from .config import Gemma4Config
from .image_processing import Gemma4ImageBatch, ImageBatchInput
from .layers import (
    build_positions_from_mask,
    make_causal_bidirectional_mask,
    merge_flat_embeddings,
)
from .module_utils import InitContext, InitModule
from .processing import Gemma4Processor, PromptAudioInput, PromptImageInput
from .text import Gemma4TextTower, TextKVCache
from .tokenizer import Gemma4Tokenizer
from .vision import Gemma4VisionTower


CONFIG_NAME = "config.json"
SAFE_WEIGHTS_NAME = "model.safetensors"
TORCH_WEIGHTS_NAME = "pytorch_model.bin"

_LEGACY_UNCLIPPED_LINEAR_BOUND_KEYS = (
    "vision.encoder.patch_embed.input_proj.input_min",
    "vision.encoder.patch_embed.input_proj.input_max",
    "vision.encoder.patch_embed.input_proj.output_min",
    "vision.encoder.patch_embed.input_proj.output_max",
    "audio.encoder.subsampler.output_proj.input_min",
    "audio.encoder.subsampler.output_proj.input_max",
    "audio.encoder.subsampler.output_proj.output_min",
    "audio.encoder.subsampler.output_proj.output_max",
)


def _drop_legacy_unclipped_linear_bounds(state_dict: dict[str, torch.Tensor]) -> None:
    for key in _LEGACY_UNCLIPPED_LINEAR_BOUND_KEYS:
        state_dict.pop(key, None)


@dataclass
class Gemma4Output:
    logits: torch.Tensor
    hidden_states: torch.Tensor | None = None
    kv_cache: TextKVCache | None = None


@dataclass
class Gemma4PreparedInputs:
    """Prepared text and multimodal tensors ready for model forward or generation."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    vision_tokens: torch.Tensor | None = None
    vision_token_mask: torch.Tensor | None = None
    audio_tokens: torch.Tensor | None = None
    audio_token_mask: torch.Tensor | None = None

    def to(
            self,
            device: str | torch.device,
            *,
            dtype: torch.dtype | None = None,
    ) -> "Gemma4PreparedInputs":
        """Move prepared inputs to a target device and optional floating dtype."""
        return Gemma4PreparedInputs(
            input_ids=self.input_ids.to(device=device),
            attention_mask=self.attention_mask.to(device=device),
            vision_tokens=_move_optional_tensor(self.vision_tokens, device=device, dtype=dtype),
            vision_token_mask=_move_optional_tensor(self.vision_token_mask, device=device),
            audio_tokens=_move_optional_tensor(self.audio_tokens, device=device, dtype=dtype),
            audio_token_mask=_move_optional_tensor(self.audio_token_mask, device=device),
        )

    def as_forward_kwargs(self) -> dict[str, torch.Tensor]:
        """Return the non-empty tensors accepted by ``Gemma4Model.forward``."""
        kwargs: dict[str, torch.Tensor] = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }
        if self.vision_tokens is not None:
            kwargs["vision_tokens"] = self.vision_tokens
        if self.vision_token_mask is not None:
            kwargs["vision_token_mask"] = self.vision_token_mask
        if self.audio_tokens is not None:
            kwargs["audio_tokens"] = self.audio_tokens
        if self.audio_token_mask is not None:
            kwargs["audio_token_mask"] = self.audio_token_mask
        return kwargs

    def __getitem__(self, key: str) -> torch.Tensor | None:
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def get(self, key: str, default: torch.Tensor | None = None) -> torch.Tensor | None:
        return getattr(self, key, default)


def _move_optional_tensor(
        value: torch.Tensor | None,
        *,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    if value is None:
        return None
    if dtype is not None and torch.is_floating_point(value):
        return value.to(device=device, dtype=dtype)
    return value.to(device=device)


def _build_audio_token_mask(
        num_soft_tokens_per_clip: torch.Tensor,
        total_audio_tokens: int,
) -> torch.Tensor:
    counts = num_soft_tokens_per_clip.to(dtype=torch.long)
    if counts.ndim == 1:
        counts = counts.unsqueeze(1)

    batch, num_clips = counts.shape
    if num_clips == 0:
        return torch.zeros(batch, total_audio_tokens, dtype=torch.bool, device=counts.device)

    tokens_per_clip = total_audio_tokens // num_clips
    mask = torch.zeros(batch, total_audio_tokens, dtype=torch.bool, device=counts.device)
    for clip_idx in range(num_clips):
        start = clip_idx * tokens_per_clip
        positions = torch.arange(tokens_per_clip, device=counts.device)
        mask[:, start : start + tokens_per_clip] = positions.unsqueeze(0) < counts[:, clip_idx : clip_idx + 1]
    return mask


class Gemma4Model(InitModule):
    """Top-level Gemma 4 multimodal model."""

    def __init__(
            self,
            config: Gemma4Config,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.text = Gemma4TextTower(config.text, device=device, dtype=dtype)
        self.vision = (
            Gemma4VisionTower(
                config.vision,
                text_hidden_size=config.text.hidden_size,
                device=device,
                dtype=dtype,
            )
            if config.vision is not None
            else None
        )
        self.audio = (
            Gemma4AudioTower(
                config.audio,
                text_hidden_size=config.text.hidden_size,
                device=device,
                dtype=dtype,
            )
            if config.audio is not None
            else None
        )
        if not any(param.is_meta for param in self.parameters()):
            self.init_weights()

    def _init_non_persistent_buffers(self) -> None:
        """Rebuild runtime-only buffers after meta-init or state-dict assignment."""
        return

    def materialize(
            self,
            *,
            device: torch.device | str,
            dtype: torch.dtype | None = None,
            init_weights: bool = True,
            ctx: InitContext | None = None,
    ) -> "Gemma4Model":
        target_device = torch.device(device)
        if any(param.device.type == "meta" for param in self.parameters()):
            if dtype is not None:
                self.to(dtype=dtype)
            self.to_empty(device=target_device)
        else:
            self.to(device=target_device, dtype=dtype)
        if init_weights:
            self.init_weights(ctx)
        else:
            self.init_non_persistent_buffers()
        return self

    def preprocess_images(self, images: ImageBatchInput) -> Gemma4ImageBatch:
        """Convert raw images into padded patch tensors for the vision tower."""
        if self.vision is None:
            raise ValueError("This model was instantiated without a vision tower.")
        return self.vision.preprocess_images(images)

    def preprocess_audios(self, audios: AudioBatchInput) -> Gemma4AudioBatch:
        """Convert raw audio inputs into padded log-mel features for the audio tower."""
        if self.audio is None:
            raise ValueError("This model was instantiated without an audio tower.")
        return Gemma4AudioProcessor.from_config(self.config.audio).preprocess(audios)

    def build_processor(self, tokenizer: Gemma4Tokenizer) -> Gemma4Processor:
        """Create a tokenizer+image processor wrapper for multimodal prompts."""
        return Gemma4Processor(
            tokenizer=tokenizer,
            text_config=self.config.text,
            image_processor=None if self.vision is None else self.vision.encoder.image_processor,
            audio_processor=None if self.audio is None else Gemma4AudioProcessor.from_config(self.config.audio),
        )

    def prepare_inputs(
            self,
            tokenizer: Gemma4Tokenizer,
            prompt: str | list[str],
            *,
            images: PromptImageInput = None,
            audios: PromptAudioInput = None,
            add_bos: bool = True,
            add_eos: bool = False,
            padding: bool = True,
    ) -> Gemma4PreparedInputs:
        """Prepare token ids, masks, and optional vision tokens for a prompt batch."""
        processor = self.build_processor(tokenizer)
        batch = processor(
            prompt,
            images=images,
            audios=audios,
            add_bos=add_bos,
            add_eos=add_eos,
            padding=padding,
        )
        model_param = next(self.parameters())
        device = model_param.device
        model_dtype = model_param.dtype
        batch = batch.to(device)
        if batch.audio_batch is not None and batch.audio_batch.input_features.dtype != model_dtype:
            batch.audio_batch = Gemma4AudioBatch(
                input_features=batch.audio_batch.input_features.to(device=device, dtype=model_dtype),
                input_features_mask=batch.audio_batch.input_features_mask,
                num_soft_tokens_per_clip=batch.audio_batch.num_soft_tokens_per_clip,
            )
        prepared = Gemma4PreparedInputs(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        )
        if batch.image_batch is None and batch.audio_batch is None:
            return prepared
        vision_tokens = None
        vision_token_mask = None
        audio_tokens = None
        audio_token_mask = None

        if batch.image_batch is not None:
            if self.vision is None:
                raise ValueError("This model was instantiated without a vision tower.")
            vision_tokens, vision_token_mask = self.encode_images_to_text(
                batch.image_batch.pixel_values,
                batch.image_batch.image_position_ids,
            )

        if batch.audio_batch is not None:
            if self.audio is None:
                raise ValueError("This model was instantiated without an audio tower.")
            audio_tokens, audio_token_mask = self.encode_audio_to_text(
                batch.audio_batch.input_features,
                batch.audio_batch.input_features_mask,
            )
            audio_token_mask = audio_token_mask & _build_audio_token_mask(
                batch.audio_batch.num_soft_tokens_per_clip,
                audio_tokens.shape[1],
            )

        return Gemma4PreparedInputs(
            input_ids=prepared.input_ids,
            attention_mask=prepared.attention_mask,
            vision_tokens=vision_tokens,
            vision_token_mask=vision_token_mask,
            audio_tokens=audio_tokens,
            audio_token_mask=audio_token_mask,
        )

    def encode_images_to_text(
            self,
            patches_or_images: ImageBatchInput | torch.Tensor,
            positions_xy: torch.Tensor | None = None,
            output_length_overrides: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode images and project them into the text hidden space."""
        if self.vision is None:
            raise ValueError("This model was instantiated without a vision tower.")
        return self.vision.encode_to_text(
            patches_or_images,
            positions_xy,
            output_length_overrides,
        )

    def encode_audio_to_text(
            self,
            features: torch.Tensor,
            feature_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode audio features and project them into the text hidden space."""
        if self.audio is None:
            raise ValueError("This model was instantiated without an audio tower.")
        return self.audio.encode_to_text(features, feature_mask)

    def forward(
            self,
            input_ids: torch.Tensor,
            *,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
            vision_tokens: torch.Tensor | None = None,
            vision_token_mask: torch.Tensor | None = None,
            audio_tokens: torch.Tensor | None = None,
            audio_token_mask: torch.Tensor | None = None,
            return_hidden_states: bool = False,
            kv_cache: TextKVCache | None = None,
            return_kv_cache: bool = False,
    ) -> Gemma4Output:
        if attention_mask is None:
            attention_mask = input_ids != self.config.text.pad_token_id
        else:
            attention_mask = attention_mask.bool()

        hidden_states = self.text.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds

        if vision_tokens is not None:
            hidden_states = merge_flat_embeddings(
                hidden_states,
                vision_tokens,
                input_ids == self.config.text.image_placeholder_token_id,
                vision_token_mask,
            )

        if audio_tokens is not None:
            hidden_states = merge_flat_embeddings(
                hidden_states,
                audio_tokens,
                input_ids == self.config.text.audio_placeholder_token_id,
                audio_token_mask,
            )

        if kv_cache is None:
            if position_ids is None:
                position_ids = build_positions_from_mask(attention_mask)

            full_attention_mask = make_causal_bidirectional_mask(attention_mask)
            sliding_attention_mask = None
            if self.config.text.use_bidirectional_attention == "vision":
                sliding_attention_mask = make_causal_bidirectional_mask(
                    attention_mask,
                    bidirectional_mask=input_ids == self.config.text.image_placeholder_token_id,
                )
            elif self.config.text.use_bidirectional_attention == "all":
                full_attention_mask = make_causal_bidirectional_mask(
                    attention_mask,
                    bidirectional_mask=attention_mask,
                )
                sliding_attention_mask = full_attention_mask
        else:
            if position_ids is None:
                base_positions = kv_cache.valid_lengths().to(device=input_ids.device)
                relative_positions = attention_mask.long().cumsum(dim=-1) - 1
                relative_positions = relative_positions.clamp_min_(0)
                position_ids = base_positions[:, None] + relative_positions
                position_ids = position_ids.masked_fill(~attention_mask, 0)

            cached_positions = kv_cache.key_positions().to(device=input_ids.device)
            cached_mask = kv_cache.key_mask().to(device=input_ids.device)
            key_positions = torch.cat([cached_positions, position_ids], dim=-1)
            key_mask = torch.cat([cached_mask, attention_mask], dim=-1)
            full_attention_mask = self._make_attention_mask(
                query_positions=position_ids,
                key_positions=key_positions,
                query_mask=attention_mask,
                key_mask=key_mask,
            )
            sliding_attention_mask = full_attention_mask

        text_output = self.text(
            input_ids,
            inputs_embeds=hidden_states,
            position_ids=position_ids,
            query_mask=attention_mask,
            full_attention_mask=full_attention_mask,
            sliding_attention_mask=sliding_attention_mask,
            kv_cache=kv_cache,
            return_kv_cache=return_kv_cache,
        )
        if return_kv_cache:
            hidden_states, next_kv_cache = text_output
        else:
            hidden_states = text_output
            next_kv_cache = None
        logits = self.text.project_logits(hidden_states)
        if self.config.text.final_logit_softcap is not None:
            logits = (
                torch.tanh(logits / self.config.text.final_logit_softcap)
                * self.config.text.final_logit_softcap
            )
        return Gemma4Output(
            logits=logits,
            hidden_states=hidden_states if return_hidden_states else None,
            kv_cache=next_kv_cache,
        )

    @staticmethod
    def _make_attention_mask(
            query_positions: torch.Tensor,
            key_positions: torch.Tensor,
            query_mask: torch.Tensor,
            key_mask: torch.Tensor,
    ) -> torch.Tensor:
        causal = key_positions[:, None, :] <= query_positions[:, :, None]
        return query_mask[:, :, None] & key_mask[:, None, :] & causal

    @staticmethod
    def _sample_next_token(
            logits: torch.Tensor,
            *,
            do_sample: bool,
            temperature: float,
            top_k: int | None,
    ) -> torch.Tensor:
        if not do_sample or temperature == 0.0:
            return logits.argmax(dim=-1)

        scaled_logits = logits / max(temperature, 1e-5)
        if top_k is not None and top_k > 0:
            top_values, _ = torch.topk(scaled_logits, k=min(top_k, scaled_logits.shape[-1]), dim=-1)
            cutoff = top_values[:, -1].unsqueeze(-1)
            scaled_logits = scaled_logits.masked_fill(scaled_logits < cutoff, float("-inf"))
        probs = torch.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.no_grad()
    def generate(
            self,
            input_ids: torch.Tensor,
            *,
            attention_mask: torch.Tensor | None = None,
            vision_tokens: torch.Tensor | None = None,
            vision_token_mask: torch.Tensor | None = None,
            audio_tokens: torch.Tensor | None = None,
            audio_token_mask: torch.Tensor | None = None,
            max_new_tokens: int = 20,
            do_sample: bool = False,
            temperature: float = 1.0,
            top_k: int | None = None,
            eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Generate tokens using greedy decoding or top-k sampling.

        Args:
            input_ids: Prompt token ids.
            attention_mask: Prompt padding mask.
            vision_tokens: Optional projected vision tokens aligned to placeholders.
            vision_token_mask: Optional validity mask for projected vision tokens.
            audio_tokens: Optional projected audio tokens aligned to placeholders.
            audio_token_mask: Optional validity mask for projected audio tokens.
            max_new_tokens: Number of decoding steps to run.
            do_sample: Whether to sample from the next-token distribution.
            temperature: Sampling temperature.
            top_k: Optional top-k cutoff for sampling.
            eos_token_id: Optional EOS token used to stop generation.
        """
        if attention_mask is None:
            attention_mask = input_ids != self.config.text.pad_token_id
        else:
            attention_mask = attention_mask.bool()

        output = self(
            input_ids,
            attention_mask=attention_mask,
            vision_tokens=vision_tokens,
            vision_token_mask=vision_token_mask,
            audio_tokens=audio_tokens,
            audio_token_mask=audio_token_mask,
            return_kv_cache=True,
        )
        kv_cache = output.kv_cache
        generated = input_ids
        current_attention_mask = attention_mask

        last_token_index = current_attention_mask.long().sum(dim=-1) - 1
        next_logits = output.logits[
            torch.arange(output.logits.shape[0], device=input_ids.device),
            last_token_index,
        ]
        next_token = self._sample_next_token(
            next_logits,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
        )

        finished = (
            next_token == eos_token_id
            if eos_token_id is not None
            else torch.zeros_like(next_token, dtype=torch.bool)
        )

        for _ in range(max_new_tokens):
            step_tokens = next_token.unsqueeze(-1)
            generated = torch.cat([generated, step_tokens], dim=-1)
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones_like(step_tokens, dtype=torch.bool)],
                dim=-1,
            )
            if eos_token_id is not None and finished.all():
                break

            output = self(
                step_tokens,
                attention_mask=torch.ones_like(step_tokens, dtype=torch.bool),
                kv_cache=kv_cache,
                return_kv_cache=True,
            )
            kv_cache = output.kv_cache
            next_logits = output.logits[:, -1]
            next_token = self._sample_next_token(
                next_logits,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
            )
            if eos_token_id is not None:
                next_token = torch.where(
                    finished,
                    next_token.new_full((), eos_token_id),
                    next_token,
                )
                finished = finished | (next_token == eos_token_id)

        return generated

    @torch.no_grad()
    def generate_text(
            self,
            tokenizer: Gemma4Tokenizer,
            prompt: str | list[str],
            *,
            images: PromptImageInput = None,
            audios: PromptAudioInput = None,
            add_bos: bool = True,
            add_eos: bool = False,
            return_full_text: bool = False,
            skip_special_tokens: bool = True,
            **generate_kwargs,
    ) -> str | list[str]:
        """Tokenize a prompt, run generation, and decode the result."""
        batch = self.prepare_inputs(
            tokenizer,
            prompt,
            images=images,
            audios=audios,
            add_bos=add_bos,
            add_eos=add_eos,
            padding=True,
        )
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        eos_token_id = generate_kwargs.pop("eos_token_id", tokenizer.eos_token_id)
        generated = self.generate(
            input_ids,
            attention_mask=attention_mask,
            vision_tokens=batch.vision_tokens,
            vision_token_mask=batch.vision_token_mask,
            audio_tokens=batch.audio_tokens,
            audio_token_mask=batch.audio_token_mask,
            eos_token_id=eos_token_id,
            **generate_kwargs,
        )

        if return_full_text:
            decoded = self._decode_output_sequences(
                tokenizer,
                generated,
                skip_special_tokens=skip_special_tokens,
            )
        else:
            continuation = generated[:, input_ids.shape[1] :]
            decoded = tokenizer.batch_decode(continuation, skip_special_tokens=skip_special_tokens)
        return decoded[0] if isinstance(prompt, str) else decoded

    @staticmethod
    def _decode_output_sequences(
            tokenizer: Gemma4Tokenizer,
            token_ids: torch.Tensor,
            *,
            skip_special_tokens: bool,
    ) -> list[str]:
        decoded: list[str] = []
        for row in token_ids.detach().cpu().tolist():
            filtered = [int(token_id) for token_id in row if int(token_id) >= 0]
            decoded.append(tokenizer.decode(filtered, skip_special_tokens=skip_special_tokens))
        return decoded

    def save_pretrained(
            self,
            save_directory: str | Path,
            *,
            safe_serialization: bool = True,
    ) -> None:
        """Save config and weights to a local directory.

        Args:
            save_directory: Output directory.
            safe_serialization: Whether to save weights as `safetensors`.
        """
        save_directory = Path(save_directory)
        try:
            save_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OSError(f"Failed to create model directory {save_directory}.") from exc

        try:
            with (save_directory / CONFIG_NAME).open("w", encoding="utf-8") as f:
                json.dump(self.config.to_dict(), f, indent=2)
        except OSError as exc:
            raise OSError(f"Failed to write config to {save_directory / CONFIG_NAME}.") from exc

        state_dict = self.state_dict()
        if safe_serialization:
            try:
                save_safetensors(state_dict, str(save_directory / SAFE_WEIGHTS_NAME))
            except OSError as exc:
                raise OSError(
                    f"Failed to save safetensors weights to {save_directory / SAFE_WEIGHTS_NAME}."
                ) from exc
        else:
            try:
                torch.save(state_dict, save_directory / TORCH_WEIGHTS_NAME)
            except OSError as exc:
                raise OSError(
                    f"Failed to save PyTorch weights to {save_directory / TORCH_WEIGHTS_NAME}."
                ) from exc

    @classmethod
    def from_pretrained(
            cls,
            load_directory: str | Path,
            *,
            device: str | torch.device = "cpu",
            dtype: torch.dtype | None = None,
            strict: bool = True,
            attn_impl: str | None = None,
    ) -> Gemma4Model:
        """Load a model from a local directory.

        Args:
            load_directory: Directory containing config and weights.
            device: Target device for the loaded model.
            dtype: Optional parameter dtype cast after loading.
            strict: Whether to require an exact state-dict match.
            attn_impl: Optional attention implementation override for text and vision towers.
        """
        load_directory = Path(load_directory)
        config_path = load_directory / CONFIG_NAME
        if not config_path.exists():
            raise FileNotFoundError(f"Could not find {CONFIG_NAME} in {load_directory}.")

        try:
            with config_path.open(encoding="utf-8") as f:
                config = Gemma4Config.from_dict(json.load(f))
        except OSError as exc:
            raise OSError(f"Failed to read config from {config_path}.") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid config JSON in {config_path}.") from exc

        if attn_impl is not None:
            config.text.attn_impl = attn_impl
            config.text.__post_init__()
            if config.vision is not None:
                config.vision.attn_impl = attn_impl
                config.vision.__post_init__()

        safe_path = load_directory / SAFE_WEIGHTS_NAME
        torch_path = load_directory / TORCH_WEIGHTS_NAME
        if safe_path.exists():
            try:
                state_dict = load_safetensors(str(safe_path), device="cpu")
            except (OSError, RuntimeError) as exc:
                raise RuntimeError(f"Failed to load safetensors weights from {safe_path}.") from exc
        elif torch_path.exists():
            try:
                state_dict = torch.load(torch_path, map_location="cpu", weights_only=True)
            except (OSError, RuntimeError, pickle.UnpicklingError) as exc:
                raise RuntimeError(f"Failed to load PyTorch weights from {torch_path}.") from exc
        else:
            raise FileNotFoundError(
                f"Could not find {SAFE_WEIGHTS_NAME} or {TORCH_WEIGHTS_NAME} in {load_directory}."
            )

        _drop_legacy_unclipped_linear_bounds(state_dict)

        target_device = torch.device(device)
        with torch.device("meta"):
            model = cls(config)
        if dtype is not None:
            model = model.to(dtype=dtype)

        assign = target_device.type == "cpu"
        if assign:
            if dtype is not None:
                state_dict = {
                    key: value.to(dtype=dtype) if torch.is_floating_point(value) else value
                    for key, value in state_dict.items()
                }
        else:
            model.to_empty(device=target_device)
            model.init_non_persistent_buffers()

        try:
            model.load_state_dict(state_dict, strict=strict, assign=assign)
        except RuntimeError as exc:
            raise RuntimeError(f"Failed to load model weights into {type(model).__name__}.") from exc

        model.init_non_persistent_buffers()
        return model
