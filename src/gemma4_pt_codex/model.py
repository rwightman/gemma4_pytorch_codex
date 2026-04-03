from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from .audio import Gemma4AudioTower
from .config import Gemma4Config
from .layers import (
    build_positions_from_mask,
    make_causal_bidirectional_mask,
    merge_flat_embeddings,
)
from .text import Gemma4TextTower, TextKVCache
from .tokenizer import Gemma4Tokenizer
from .vision import Gemma4VisionTower


CONFIG_NAME = "config.json"
SAFE_WEIGHTS_NAME = "model.safetensors"
TORCH_WEIGHTS_NAME = "pytorch_model.bin"


@dataclass
class Gemma4Output:
    logits: torch.Tensor
    hidden_states: torch.Tensor | None = None
    kv_cache: TextKVCache | None = None


class Gemma4Model(nn.Module):
    """Top-level Gemma 4 multimodal model."""

    def __init__(self, config: Gemma4Config) -> None:
        super().__init__()
        self.config = config
        self.text = Gemma4TextTower(config.text)
        self.vision = (
            Gemma4VisionTower(config.vision, text_hidden_size=config.text.hidden_size)
            if config.vision is not None
            else None
        )
        self.audio = (
            Gemma4AudioTower(config.audio, text_hidden_size=config.text.hidden_size)
            if config.audio is not None
            else None
        )

    def encode_images_to_text(
            self,
            patches_or_images: torch.Tensor,
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
            add_bos: bool = True,
            return_full_text: bool = False,
            skip_special_tokens: bool = True,
            **generate_kwargs,
    ) -> str | list[str]:
        """Tokenize a prompt, run generation, and decode the result."""
        batch = tokenizer(
            prompt,
            add_bos=add_bos,
            padding=True,
            return_tensors="pt",
        )
        device = next(self.parameters()).device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        eos_token_id = generate_kwargs.pop("eos_token_id", tokenizer.eos_token_id)
        generated = self.generate(
            input_ids,
            attention_mask=attention_mask,
            eos_token_id=eos_token_id,
            **generate_kwargs,
        )

        if return_full_text:
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=skip_special_tokens)
        else:
            continuation = generated[:, input_ids.shape[1] :]
            decoded = tokenizer.batch_decode(continuation, skip_special_tokens=skip_special_tokens)
        return decoded[0] if isinstance(prompt, str) else decoded

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

        model = cls(config)
        safe_path = load_directory / SAFE_WEIGHTS_NAME
        torch_path = load_directory / TORCH_WEIGHTS_NAME
        if safe_path.exists():
            try:
                state_dict = load_safetensors(str(safe_path), device=str(device))
            except (OSError, RuntimeError) as exc:
                raise RuntimeError(f"Failed to load safetensors weights from {safe_path}.") from exc
        elif torch_path.exists():
            try:
                state_dict = torch.load(torch_path, map_location=device, weights_only=True)
            except (OSError, RuntimeError, pickle.UnpicklingError) as exc:
                raise RuntimeError(f"Failed to load PyTorch weights from {torch_path}.") from exc
        else:
            raise FileNotFoundError(
                f"Could not find {SAFE_WEIGHTS_NAME} or {TORCH_WEIGHTS_NAME} in {load_directory}."
            )

        try:
            model.load_state_dict(state_dict, strict=strict)
        except RuntimeError as exc:
            raise RuntimeError(f"Failed to load model weights into {type(model).__name__}.") from exc

        model.to(device=device)
        if dtype is not None:
            model.to(dtype=dtype)
        return model
