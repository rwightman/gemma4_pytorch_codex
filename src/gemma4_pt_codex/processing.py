from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeAlias

import torch

from .audio_processing import AudioBatchInput, Gemma4AudioBatch, Gemma4AudioProcessor
from .config import TextConfig
from .image_processing import (
    Gemma4ImageBatch,
    Gemma4ImageProcessor,
    ImageBatchInput,
    POSITIONS_PAD_VALUE,
)
from .tokenizer import Gemma4Tokenizer


PromptImageInput: TypeAlias = ImageBatchInput | Sequence[ImageBatchInput] | None
PromptAudioInput: TypeAlias = AudioBatchInput | Sequence[AudioBatchInput] | None


@dataclass
class Gemma4MultimodalBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    image_batch: Gemma4ImageBatch | None = None
    audio_batch: Gemma4AudioBatch | None = None

    def to(
            self,
            device: str | torch.device,
    ) -> "Gemma4MultimodalBatch":
        """Move the multimodal batch to a target device."""
        return Gemma4MultimodalBatch(
            input_ids=self.input_ids.to(device=device),
            attention_mask=self.attention_mask.to(device=device),
            image_batch=None if self.image_batch is None else self.image_batch.to(device),
            audio_batch=None if self.audio_batch is None else self.audio_batch.to(device),
        )


class Gemma4Processor:
    """Prepare Gemma4 text, image, and audio prompts using the original JAX expansion scheme."""

    def __init__(
            self,
            tokenizer: Gemma4Tokenizer,
            text_config: TextConfig,
            image_processor: Gemma4ImageProcessor | None = None,
            audio_processor: Gemma4AudioProcessor | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.text_config = text_config
        self.image_processor = image_processor
        self.audio_processor = audio_processor

    def expand_image_placeholders(
            self,
            token_ids: Sequence[int],
            num_soft_tokens_per_image: Sequence[int],
            *,
            image_token_id: int | None = None,
            begin_image_token_id: int | None = None,
            end_image_token_id: int | None = None,
            double_newline_token_ids: Sequence[int] | None = None,
    ) -> list[int]:
        """Expand visible image placeholders into JAX-style boundary + soft-token spans.

        Args:
            token_ids: Prompt token ids before multimodal expansion.
            num_soft_tokens_per_image: Number of valid vision soft tokens for each image.
            image_token_id: Visible tokenizer image placeholder id.
            begin_image_token_id: Visible begin-image token id.
            end_image_token_id: Visible end-image token id.
            double_newline_token_ids: Token ids for the `"\n\n"` prefix/suffix.

        Returns:
            The expanded token ids with internal soft image placeholders.
        """
        image_token_id = self._resolve_image_token_id(image_token_id)
        begin_image_token_id = self._resolve_required_token_id(
            begin_image_token_id,
            self.tokenizer.boi_token_id,
            "begin-image",
        )
        end_image_token_id = self._resolve_required_token_id(
            end_image_token_id,
            self.tokenizer.eoi_token_id,
            "end-image",
        )
        if double_newline_token_ids is None:
            double_newline_token_ids = self.tokenizer.encode("\n\n")

        placeholder_count = sum(int(token_id == image_token_id) for token_id in token_ids)
        if placeholder_count != len(num_soft_tokens_per_image):
            raise ValueError(
                "Mismatch between visible image placeholders and image count: "
                f"{placeholder_count} vs {len(num_soft_tokens_per_image)}."
            )

        expanded: list[int] = []
        image_idx = 0
        for token_id in token_ids:
            if token_id != image_token_id:
                expanded.append(int(token_id))
                continue

            expanded.extend(int(token) for token in double_newline_token_ids)
            expanded.append(begin_image_token_id)
            expanded.extend(
                self.text_config.image_placeholder_token_id
                for _ in range(int(num_soft_tokens_per_image[image_idx]))
            )
            expanded.append(end_image_token_id)
            expanded.extend(int(token) for token in double_newline_token_ids)
            image_idx += 1

        return expanded

    def expand_audio_placeholders(
            self,
            token_ids: Sequence[int],
            num_soft_tokens_per_audio: Sequence[int],
            *,
            audio_token_id: int | None = None,
            begin_audio_token_id: int | None = None,
            end_audio_token_id: int | None = None,
    ) -> list[int]:
        """Expand visible audio placeholders into JAX-style boundary + soft-token spans."""
        audio_token_id = self._resolve_audio_token_id(audio_token_id)
        begin_audio_token_id = self._resolve_required_token_id(
            begin_audio_token_id,
            self.tokenizer.boa_token_id,
            "begin-audio",
        )
        end_audio_token_id = self._resolve_required_token_id(
            end_audio_token_id,
            self.tokenizer.eoa_token_id,
            "end-audio",
        )

        placeholder_count = sum(int(token_id == audio_token_id) for token_id in token_ids)
        if placeholder_count != len(num_soft_tokens_per_audio):
            raise ValueError(
                "Mismatch between visible audio placeholders and audio count: "
                f"{placeholder_count} vs {len(num_soft_tokens_per_audio)}."
            )

        expanded: list[int] = []
        audio_idx = 0
        for token_id in token_ids:
            if token_id != audio_token_id:
                expanded.append(int(token_id))
                continue

            expanded.append(begin_audio_token_id)
            expanded.extend(
                self.text_config.audio_placeholder_token_id
                for _ in range(int(num_soft_tokens_per_audio[audio_idx]))
            )
            expanded.append(end_audio_token_id)
            audio_idx += 1

        return expanded

    def __call__(
            self,
            prompt: str | Sequence[str],
            *,
            images: PromptImageInput = None,
            audios: PromptAudioInput = None,
            add_bos: bool = False,
            add_eos: bool = False,
            padding: bool = True,
    ) -> Gemma4MultimodalBatch:
        """Tokenize prompt text and preprocess multimodal inputs into a model-ready batch."""
        prompts = _normalize_prompt_batch(prompt)
        image_groups = self._normalize_image_groups(prompts, images)
        audio_groups = self._normalize_audio_groups(prompts, audios)
        expanded_input_ids: list[list[int]] = []
        image_batches: list[Gemma4ImageBatch | None] = []
        audio_batches: list[Gemma4AudioBatch | None] = []

        for prompt_text, image_group, audio_group in zip(prompts, image_groups, audio_groups):
            token_ids = self.tokenizer.encode(
                prompt_text,
                add_bos=add_bos,
                add_eos=add_eos,
            )
            expanded_ids = token_ids
            image_batch = None
            audio_batch = None

            if image_group is None:
                self._raise_if_prompt_contains_visible_image_token(expanded_ids)
            else:
                if self.image_processor is None:
                    raise ValueError("This processor was created without an image processor.")
                image_batch = self.image_processor.preprocess(image_group)
                expanded_ids = self.expand_image_placeholders(
                    expanded_ids,
                    image_batch.num_soft_tokens_per_image.reshape(-1).tolist(),
                )

            if audio_group is None:
                self._raise_if_prompt_contains_visible_audio_token(expanded_ids)
            else:
                if self.audio_processor is None:
                    raise ValueError("This processor was created without an audio processor.")
                audio_batch = self.audio_processor.preprocess(audio_group)
                expanded_ids = self.expand_audio_placeholders(
                    expanded_ids,
                    audio_batch.num_soft_tokens_per_clip.reshape(-1).tolist(),
                )

            expanded_input_ids.append(expanded_ids)
            image_batches.append(image_batch)
            audio_batches.append(audio_batch)

        input_ids, attention_mask = _pad_token_lists(
            expanded_input_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            padding=padding,
        )
        return Gemma4MultimodalBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_batch=_collate_image_batches(image_batches),
            audio_batch=_collate_audio_batches(audio_batches),
        )

    def _normalize_image_groups(
            self,
            prompts: Sequence[str],
            images: PromptImageInput,
    ) -> list[ImageBatchInput | None]:
        if images is None:
            return [None] * len(prompts)

        if len(prompts) == 1:
            return [images]

        if torch.is_tensor(images):
            if images.shape[0] != len(prompts):
                raise ValueError(
                    "Tensor image batches must match the prompt batch size: "
                    f"{images.shape[0]} vs {len(prompts)}."
                )
            return [images[idx] for idx in range(images.shape[0])]

        if not isinstance(images, Sequence) or len(images) != len(prompts):
            raise ValueError(
                "Batched prompts require `images` to be a matching sequence or tensor batch."
            )

        return list(images)

    def _normalize_audio_groups(
            self,
            prompts: Sequence[str],
            audios: PromptAudioInput,
    ) -> list[AudioBatchInput | None]:
        if audios is None:
            return [None] * len(prompts)

        if len(prompts) == 1:
            return [audios]

        if not isinstance(audios, Sequence) or len(audios) != len(prompts):
            raise ValueError(
                "Batched prompts require `audios` to be a matching sequence."
            )
        return list(audios)

    def _raise_if_prompt_contains_visible_image_token(self, token_ids: Sequence[int]) -> None:
        image_token_id = self._resolve_image_token_id()
        if any(int(token_id) == image_token_id for token_id in token_ids):
            raise ValueError(
                "The prompt contains visible image placeholder tokens, but no images were provided."
            )

    def _raise_if_prompt_contains_visible_audio_token(self, token_ids: Sequence[int]) -> None:
        audio_token_id = self._resolve_audio_token_id()
        if any(int(token_id) == audio_token_id for token_id in token_ids):
            raise ValueError(
                "The prompt contains visible audio placeholder tokens, but no audio was provided."
            )

    def _resolve_image_token_id(self, token_id: int | None = None) -> int:
        token_id = self._resolve_optional_token_id(
            token_id,
            self.tokenizer.image_token_id,
            self.text_config.image_token_id,
        )
        if token_id is None:
            raise KeyError("Could not resolve the visible image placeholder token id.")
        return token_id

    def _resolve_audio_token_id(self, token_id: int | None = None) -> int:
        token_id = self._resolve_optional_token_id(
            token_id,
            self.tokenizer.audio_token_id,
            self.text_config.audio_token_id,
        )
        if token_id is None:
            raise KeyError("Could not resolve the visible audio placeholder token id.")
        return token_id

    @staticmethod
    def _resolve_required_token_id(
            override: int | None,
            tokenizer_value: int | None,
            token_name: str,
    ) -> int:
        token_id = Gemma4Processor._resolve_optional_token_id(override, tokenizer_value)
        if token_id is None:
            raise KeyError(f"Could not resolve the {token_name} token id.")
        return token_id

    @staticmethod
    def _resolve_optional_token_id(*values: int | None) -> int | None:
        for value in values:
            if value is not None:
                return int(value)
        return None


def _normalize_prompt_batch(prompt: str | Sequence[str]) -> list[str]:
    if isinstance(prompt, str):
        return [prompt]

    prompts = list(prompt)
    if not prompts:
        raise ValueError("Expected at least one prompt.")
    if not all(isinstance(prompt_text, str) for prompt_text in prompts):
        raise TypeError("All prompts must be strings.")
    return prompts


def _pad_token_lists(
        token_ids_list: Sequence[Sequence[int]],
        *,
        pad_token_id: int,
        padding: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = [len(token_ids) for token_ids in token_ids_list]
    if not lengths:
        raise ValueError("Expected at least one token sequence.")

    if not padding and len(set(lengths)) != 1:
        raise ValueError("Unpadded batching requires all token sequences to have the same length.")

    target_length = max(lengths)
    padded_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    for token_ids in token_ids_list:
        pad_length = target_length - len(token_ids)
        padded_ids.append(([pad_token_id] * pad_length) + [int(token_id) for token_id in token_ids])
        attention_masks.append(([0] * pad_length) + ([1] * len(token_ids)))

    return (
        torch.tensor(padded_ids, dtype=torch.long),
        torch.tensor(attention_masks, dtype=torch.bool),
    )


def _collate_image_batches(
        image_batches: Sequence[Gemma4ImageBatch | None],
) -> Gemma4ImageBatch | None:
    present_batches = [image_batch for image_batch in image_batches if image_batch is not None]
    if not present_batches:
        return None

    reference_batch = present_batches[0]
    max_images = max(int(image_batch.pixel_values.shape[0]) for image_batch in present_batches)
    max_patches = int(reference_batch.pixel_values.shape[-2])
    patch_dim = int(reference_batch.pixel_values.shape[-1])

    pixel_values = []
    image_position_ids = []
    num_soft_tokens_per_image = []
    for image_batch in image_batches:
        if image_batch is None:
            pixel_values.append(
                torch.zeros(
                    max_images,
                    max_patches,
                    patch_dim,
                    dtype=reference_batch.pixel_values.dtype,
                )
            )
            image_position_ids.append(
                torch.full(
                    (max_images, max_patches, 2),
                    POSITIONS_PAD_VALUE,
                    dtype=reference_batch.image_position_ids.dtype,
                )
            )
            num_soft_tokens_per_image.append(torch.zeros(max_images, dtype=torch.long))
            continue

        current_images = int(image_batch.pixel_values.shape[0])
        image_pad = max_images - current_images
        if image_pad < 0:
            raise ValueError("Encountered an image batch larger than the padded group size.")

        if image_pad == 0:
            pixel_values.append(image_batch.pixel_values)
            image_position_ids.append(image_batch.image_position_ids)
            num_soft_tokens_per_image.append(image_batch.num_soft_tokens_per_image.to(dtype=torch.long))
            continue

        pixel_values.append(
            torch.cat(
                [
                    image_batch.pixel_values,
                    torch.zeros(
                        image_pad,
                        max_patches,
                        patch_dim,
                        dtype=image_batch.pixel_values.dtype,
                        device=image_batch.pixel_values.device,
                    ),
                ],
                dim=0,
            )
        )
        image_position_ids.append(
            torch.cat(
                [
                    image_batch.image_position_ids,
                    torch.full(
                        (image_pad, max_patches, 2),
                        POSITIONS_PAD_VALUE,
                        dtype=image_batch.image_position_ids.dtype,
                        device=image_batch.image_position_ids.device,
                    ),
                ],
                dim=0,
            )
        )
        num_soft_tokens_per_image.append(
            torch.cat(
                [
                    image_batch.num_soft_tokens_per_image.to(dtype=torch.long),
                    torch.zeros(image_pad, dtype=torch.long, device=image_batch.pixel_values.device),
                ],
                dim=0,
            )
        )

    return Gemma4ImageBatch(
        pixel_values=torch.stack(pixel_values, dim=0),
        image_position_ids=torch.stack(image_position_ids, dim=0),
        num_soft_tokens_per_image=torch.stack(num_soft_tokens_per_image, dim=0),
    )


def _collate_audio_batches(
        audio_batches: Sequence[Gemma4AudioBatch | None],
) -> Gemma4AudioBatch | None:
    present_batches = [audio_batch for audio_batch in audio_batches if audio_batch is not None]
    if not present_batches:
        return None

    reference_batch = present_batches[0]
    max_clips = max(int(audio_batch.input_features.shape[0]) for audio_batch in present_batches)
    max_frames = max(int(audio_batch.input_features.shape[1]) for audio_batch in present_batches)
    num_mel_bins = int(reference_batch.input_features.shape[-1])

    input_features = []
    input_features_mask = []
    num_soft_tokens_per_clip = []
    for audio_batch in audio_batches:
        if audio_batch is None:
            input_features.append(
                torch.zeros(max_clips, max_frames, num_mel_bins, dtype=reference_batch.input_features.dtype)
            )
            input_features_mask.append(torch.zeros(max_clips, max_frames, dtype=torch.bool))
            num_soft_tokens_per_clip.append(torch.zeros(max_clips, dtype=torch.long))
            continue

        current_clips = int(audio_batch.input_features.shape[0])
        clip_pad = max_clips - current_clips
        frame_pad = max_frames - int(audio_batch.input_features.shape[1])
        features = audio_batch.input_features
        mask = audio_batch.input_features_mask
        counts = audio_batch.num_soft_tokens_per_clip.to(dtype=torch.long)

        if frame_pad > 0:
            features = torch.cat(
                [
                    features,
                    torch.zeros(
                        current_clips,
                        frame_pad,
                        num_mel_bins,
                        dtype=features.dtype,
                        device=features.device,
                    ),
                ],
                dim=1,
            )
            mask = torch.cat(
                [
                    mask,
                    torch.zeros(current_clips, frame_pad, dtype=torch.bool, device=mask.device),
                ],
                dim=1,
            )

        if clip_pad > 0:
            features = torch.cat(
                [
                    features,
                    torch.zeros(
                        clip_pad,
                        max_frames,
                        num_mel_bins,
                        dtype=features.dtype,
                        device=features.device,
                    ),
                ],
                dim=0,
            )
            mask = torch.cat(
                [
                    mask,
                    torch.zeros(clip_pad, max_frames, dtype=torch.bool, device=mask.device),
                ],
                dim=0,
            )
            counts = torch.cat(
                [
                    counts,
                    torch.zeros(clip_pad, dtype=torch.long, device=counts.device),
                ],
                dim=0,
            )

        input_features.append(features)
        input_features_mask.append(mask)
        num_soft_tokens_per_clip.append(counts)

    return Gemma4AudioBatch(
        input_features=torch.stack(input_features, dim=0),
        input_features_mask=torch.stack(input_features_mask, dim=0),
        num_soft_tokens_per_clip=torch.stack(num_soft_tokens_per_clip, dim=0),
    )
