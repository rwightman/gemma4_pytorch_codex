from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, TypeAlias

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf

from .config import VisionConfig


ImageLike: TypeAlias = Image.Image | np.ndarray | torch.Tensor
ImageBatchInput: TypeAlias = ImageLike | Sequence[ImageLike]
POSITIONS_PAD_VALUE = -1
_SUPPORTED_CHANNELS = {1, 3, 4}


@dataclass
class Gemma4ImageBatch:
    pixel_values: torch.Tensor
    image_position_ids: torch.Tensor
    num_soft_tokens_per_image: torch.Tensor

    def to(
            self,
            device: str | torch.device,
    ) -> "Gemma4ImageBatch":
        """Move the image batch to a target device."""
        return Gemma4ImageBatch(
            pixel_values=self.pixel_values.to(device=device),
            image_position_ids=self.image_position_ids.to(device=device),
            num_soft_tokens_per_image=self.num_soft_tokens_per_image.to(device=device),
        )


def get_target_dimensions(
        height: int,
        width: int,
        *,
        patch_size: int = 16,
        max_patches: int = 10_080,
        pooling_kernel_size: int = 3,
) -> tuple[int, int]:
    """Compute the JAX-style resized image shape for a patch budget.

    Args:
        height: Input image height in pixels.
        width: Input image width in pixels.
        patch_size: Patch size in pixels.
        max_patches: Maximum number of pre-pooling patches.
        pooling_kernel_size: Vision pooling kernel size.

    Returns:
        The resized `(height, width)` that preserves aspect ratio, stays within
        the patch budget, and is divisible by `pooling_kernel_size * patch_size`.
    """
    total_px = height * width
    side_mult = pooling_kernel_size * patch_size
    if total_px == 0:
        return side_mult, side_mult

    target_px = max_patches * (patch_size**2)
    factor = math.sqrt(target_px / total_px)
    ideal_height = factor * height
    ideal_width = factor * width

    target_height = int(math.floor(ideal_height / side_mult)) * side_mult
    target_width = int(math.floor(ideal_width / side_mult)) * side_mult

    if target_height == 0 and target_width == 0:
        target_height = side_mult
        target_width = side_mult
    elif target_height == 0:
        target_height = side_mult
        max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
        target_width = min(
            max(1, int(math.floor(width / height))) * side_mult,
            max_side_length,
        )
    elif target_width == 0:
        target_width = side_mult
        max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
        target_height = min(
            max(1, int(math.floor(height / width))) * side_mult,
            max_side_length,
        )

    return int(target_height), int(target_width)


def convert_image_to_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert a CHW image tensor into a flat patch sequence."""
    num_channels, image_height, image_width = image.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched_image = image.reshape(
        num_channels,
        num_patches_height,
        patch_size,
        num_patches_width,
        patch_size,
    )
    patched_image = patched_image.permute(1, 3, 2, 4, 0)
    return patched_image.reshape(num_patches_height * num_patches_width, -1)


def pad_along_first_dim(
        image: torch.Tensor,
        positions: torch.Tensor,
        target_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad patches and positions up to a fixed sequence length."""
    current_length = image.shape[0]
    padding_length = target_length - current_length
    if padding_length <= 0:
        return image, positions

    padding = [0, 0] * (image.ndim - 1) + [0, padding_length]
    pos_padding = (0, 0, 0, padding_length)
    image = torch.nn.functional.pad(image, padding, mode="constant", value=0.0)
    positions = torch.nn.functional.pad(
        positions,
        pos_padding,
        mode="constant",
        value=POSITIONS_PAD_VALUE,
    )
    return image, positions


def normalize_image_patches(patches: torch.Tensor) -> torch.Tensor:
    """Map raw `[0, 1]` image patches into the model's `[-1, 1]` space.

    If the patches already appear to be outside `[0, 1]`, they are returned
    unchanged. This keeps direct low-level patch calls workable while keeping
    patch projection free of preprocessing logic.
    """
    patches_fp32 = patches.detach().to(dtype=torch.float32)
    if float(patches_fp32.amin()) >= 0.0 and float(patches_fp32.amax()) <= 1.0:
        return (2.0 * (patches - 0.5)).contiguous()
    return patches


def _to_rgb_tensor(image: ImageLike) -> torch.Tensor:
    if isinstance(image, Image.Image):
        tensor = torch.from_numpy(np.array(image.convert("RGB")))
    elif isinstance(image, np.ndarray):
        tensor = torch.as_tensor(image)
    elif isinstance(image, torch.Tensor):
        tensor = image
    else:
        raise TypeError(f"Unsupported image type: {type(image).__name__}.")

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 3:
        raise ValueError(f"Expected a single image tensor, got shape {tuple(tensor.shape)}.")

    if tensor.shape[-1] in _SUPPORTED_CHANNELS:
        if tensor.shape[-1] == 1:
            tensor = tensor.expand(*tensor.shape[:2], 3)
        elif tensor.shape[-1] == 4:
            tensor = tensor[..., :3]
        tensor = tensor.permute(2, 0, 1)
    elif tensor.shape[0] in _SUPPORTED_CHANNELS:
        if tensor.shape[0] == 1:
            tensor = tensor.expand(3, *tensor.shape[1:])
        elif tensor.shape[0] == 4:
            tensor = tensor[:3]
    else:
        raise ValueError(f"Could not infer channel layout from image shape {tuple(tensor.shape)}.")

    tensor = tensor.contiguous()
    if torch.is_floating_point(tensor):
        tensor = tensor.to(dtype=torch.float32)
        if bool((tensor.amax() > 1.0) or (tensor.amin() < 0.0)):
            tensor = tensor / 255.0
        return tensor

    return tensor.to(dtype=torch.float32) / 255.0


def _split_tensor_batch(images: torch.Tensor) -> tuple[list[torch.Tensor], tuple[int, ...]]:
    if images.ndim < 2:
        raise ValueError(f"Expected image tensor with at least 2 dims, got {tuple(images.shape)}.")

    if (
        images.ndim in {2, 3}
        and (
            images.ndim == 2
            or images.shape[-1] in _SUPPORTED_CHANNELS
            or images.shape[0] in _SUPPORTED_CHANNELS
        )
    ):
        return [_to_rgb_tensor(images)], (1,)

    if images.ndim >= 4 and images.shape[-1] in _SUPPORTED_CHANNELS:
        batch_shape = tuple(images.shape[:-3])
        flat_images = images.reshape(-1, *images.shape[-3:])
        return [_to_rgb_tensor(image) for image in flat_images], batch_shape

    if images.ndim >= 4 and images.shape[-3] in _SUPPORTED_CHANNELS:
        batch_shape = tuple(images.shape[:-3])
        flat_images = images.reshape(-1, *images.shape[-3:])
        return [_to_rgb_tensor(image) for image in flat_images], batch_shape

    raise ValueError(f"Could not infer image layout from tensor shape {tuple(images.shape)}.")


def _split_image_batch(images: ImageBatchInput) -> tuple[list[torch.Tensor], tuple[int, ...]]:
    if isinstance(images, (Image.Image, np.ndarray, torch.Tensor)):
        if isinstance(images, torch.Tensor):
            return _split_tensor_batch(images)
        return [_to_rgb_tensor(images)], (1,)

    if not isinstance(images, Sequence) or not images:
        raise TypeError("Expected an image or a non-empty sequence of images.")

    return [_to_rgb_tensor(image) for image in images], (len(images),)


class Gemma4ImageProcessor:
    """Minimal image processor that follows the Gemma 4 JAX preprocessing path."""

    def __init__(
            self,
            patch_size: int = 16,
            max_soft_tokens: int = 280,
            pooling_kernel_size: int = 3,
            interpolation: InterpolationMode = InterpolationMode.BICUBIC,
    ) -> None:
        self.patch_size = patch_size
        self.max_soft_tokens = max_soft_tokens
        self.pooling_kernel_size = pooling_kernel_size
        self.max_patches = max_soft_tokens * pooling_kernel_size**2
        self.interpolation = interpolation

    @classmethod
    def from_config(cls, config: VisionConfig) -> "Gemma4ImageProcessor":
        output_length = config.output_length
        if isinstance(output_length, tuple):
            output_length = max(output_length)
        return cls(
            patch_size=config.patch_size,
            max_soft_tokens=int(output_length),
            pooling_kernel_size=config.pooling_kernel_size,
        )

    def get_target_dimensions(self, height: int, width: int) -> tuple[int, int]:
        """Return the preprocessed image size for the configured patch budget."""
        return get_target_dimensions(
            height,
            width,
            patch_size=self.patch_size,
            max_patches=self.max_patches,
            pooling_kernel_size=self.pooling_kernel_size,
        )

    def aspect_ratio_preserving_resize(self, image: torch.Tensor) -> torch.Tensor:
        """Resize a CHW image tensor while preserving aspect ratio."""
        height, width = image.shape[-2:]
        target_height, target_width = self.get_target_dimensions(height, width)
        if target_height == height and target_width == width:
            return image

        return tvf.resize(
            image,
            [target_height, target_width],
            interpolation=self.interpolation,
            antialias=True,
        )

    def preprocess(self, images: ImageBatchInput) -> Gemma4ImageBatch:
        """Resize, rescale, patchify, and pad images for the vision encoder.

        The resulting `pixel_values` are flat image patches in `[0, 1]`.

        Args:
            images: A single image, a batch tensor, or a flat sequence of images.

        Returns:
            A batch of padded patches and their `(x, y)` patch-grid positions.
        """
        image_list, batch_shape = _split_image_batch(images)
        pixel_values = []
        image_position_ids = []
        num_soft_tokens_per_image = []

        for image in image_list:
            image = self.aspect_ratio_preserving_resize(image)
            patch_height = image.shape[-2] // self.patch_size
            patch_width = image.shape[-1] // self.patch_size
            patches = convert_image_to_patches(image, self.patch_size)
            num_soft_tokens_per_image.append(patches.shape[0] // self.pooling_kernel_size**2)

            patch_grid = torch.meshgrid(
                torch.arange(patch_width, device=image.device),
                torch.arange(patch_height, device=image.device),
                indexing="xy",
            )
            positions = torch.stack(patch_grid, dim=-1).reshape(patches.shape[0], 2)
            patches, positions = pad_along_first_dim(
                patches,
                positions,
                target_length=self.max_patches,
            )
            pixel_values.append(patches)
            image_position_ids.append(positions)

        pixel_values_tensor = torch.stack(pixel_values, dim=0)
        image_position_ids_tensor = torch.stack(image_position_ids, dim=0)
        soft_tokens_tensor = torch.tensor(
            num_soft_tokens_per_image,
            dtype=torch.long,
            device=pixel_values_tensor.device,
        )

        return Gemma4ImageBatch(
            pixel_values=pixel_values_tensor.reshape(*batch_shape, self.max_patches, -1),
            image_position_ids=image_position_ids_tensor.reshape(*batch_shape, self.max_patches, 2),
            num_soft_tokens_per_image=soft_tokens_tensor.reshape(batch_shape),
        )
