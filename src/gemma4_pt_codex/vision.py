from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VisionConfig
from .image_processing import (
    Gemma4ImageBatch,
    Gemma4ImageProcessor,
    ImageBatchInput,
    normalize_image_patches,
)
from .layers import (
    ClippedLinear,
    K_MASK,
    RMSNorm,
    VisionRMSNorm,
    apply_multidim_rope,
    gelu_tanh,
)
from .module_utils import InitContext, InitModule, factory_kwargs, resolve_residual_init_std


POSITIONS_PAD_VALUE = -1


def _looks_like_raw_image_tensor(tensor: torch.Tensor) -> bool:
    return bool(
        (tensor.ndim >= 4 and tensor.shape[-1] in {1, 3, 4})
        or (tensor.ndim >= 4 and tensor.shape[-3] in {1, 3, 4})
        or (tensor.ndim == 3 and tensor.shape[-1] in {1, 3, 4})
    )


def patchify_images(images: torch.Tensor, patch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    *batch_dims, height, width, channels = images.shape
    if channels != 3:
        raise ValueError(f"Expected RGB images, got {channels} channels.")
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(f"Image size {(height, width)} must be divisible by patch size {patch_size}.")

    grid_h = height // patch_size
    grid_w = width // patch_size
    patches = images.view(*batch_dims, grid_h, patch_size, grid_w, patch_size, channels)
    patches = patches.permute(*range(len(batch_dims)), -5, -3, -4, -2, -1)
    patches = patches.reshape(*batch_dims, grid_h * grid_w, patch_size * patch_size * channels)

    y, x = torch.meshgrid(
        torch.arange(grid_h, device=images.device),
        torch.arange(grid_w, device=images.device),
        indexing="ij",
    )
    positions = torch.stack([x, y], dim=-1).view(grid_h * grid_w, 2)
    positions = positions.expand(*batch_dims, grid_h * grid_w, 2)
    return patches, positions


def factorized_position_embeddings(position_table: torch.Tensor, positions_xy: torch.Tensor) -> torch.Tensor:
    max_positions = position_table.shape[0]
    valid = (positions_xy >= 0) & (positions_xy < max_positions)
    safe_positions = positions_xy.clamp(min=0, max=max_positions - 1)
    x_embed = position_table[safe_positions[..., 0], 0]
    y_embed = position_table[safe_positions[..., 1], 1]
    embeds = x_embed + y_embed
    valid_xy = valid.all(dim=-1, keepdim=True)
    return embeds * valid_xy.to(dtype=embeds.dtype)


def avg_pool_by_positions(
        x: torch.Tensor,
        positions_xy: torch.Tensor,
        length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, _ = x.shape
    k = int((seq_len // length) ** 0.5)
    if k * k * length != seq_len:
        raise ValueError(f"Cannot pool shape {x.shape} to {length}.")

    padding_positions = (positions_xy == POSITIONS_PAD_VALUE).all(dim=-1)
    safe_positions = positions_xy.clamp_min(0)
    max_x = safe_positions[..., 0].amax(dim=-1, keepdim=True) + 1
    kernel_indices = torch.div(safe_positions, k, rounding_mode="floor")
    flat_kernel_indices = kernel_indices[..., 0] + (max_x // k) * kernel_indices[..., 1]
    weights = F.one_hot(flat_kernel_indices.long(), num_classes=length).to(dtype=torch.float32) / (k**2)
    weights = weights.masked_fill(padding_positions.unsqueeze(-1), 0.0)
    output = torch.matmul(weights.transpose(1, 2), x.float()).to(dtype=x.dtype)
    mask = ~(weights == 0).all(dim=1)
    return output, mask


def _make_linear(
        config: VisionConfig,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
) -> nn.Module:
    if config.use_clipped_linears:
        return ClippedLinear(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
    return nn.Linear(in_features, out_features, bias=bias, **factory_kwargs(device, dtype))


class Gemma4VisionPatchEmbed(InitModule):
    def __init__(
            self,
            config: VisionConfig,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.init_std = config.init_std
        self.position_init_std = config.position_init_std
        dd = factory_kwargs(device, dtype)
        self.input_proj = nn.Linear(config.patch_dim, config.hidden_size, bias=False, **dd)
        self.position_table = nn.Parameter(torch.empty(config.position_embedding_size, 2, config.hidden_size, **dd))

    def _init_weights(self, ctx: InitContext) -> None:
        nn.init.normal_(self.input_proj.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.position_table, mean=0.0, std=self.position_init_std, generator=ctx.generator)

    def forward(self, patches: torch.Tensor, positions_xy: torch.Tensor) -> torch.Tensor:
        hidden_states = self.input_proj(patches.to(dtype=self.input_proj.weight.dtype))
        return hidden_states + factorized_position_embeddings(
            self.position_table,
            positions_xy,
        ).to(hidden_states.dtype)


class Gemma4VisionMLP(InitModule):
    def __init__(
            self,
            config: VisionConfig,
            residual_init_std: float | None = None,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.init_std = config.init_std
        self.residual_init_std = config.init_std if residual_init_std is None else residual_init_std
        dd = factory_kwargs(device, dtype)
        self.gate_proj = _make_linear(config, config.hidden_size, config.intermediate_size, bias=False, **dd)
        self.up_proj = _make_linear(config, config.hidden_size, config.intermediate_size, bias=False, **dd)
        self.down_proj = _make_linear(config, config.intermediate_size, config.hidden_size, bias=False, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(gelu_tanh(self.gate_proj(x)) * self.up_proj(x))

    def _init_weights(self, ctx: InitContext) -> None:
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.gate_proj.bias is not None:
            nn.init.zeros_(self.gate_proj.bias)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.up_proj.bias is not None:
            nn.init.zeros_(self.up_proj.bias)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=self.residual_init_std, generator=ctx.generator)
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)


class Gemma4VisionAttention(InitModule):
    def __init__(
            self,
            config: VisionConfig,
            residual_init_std: float | None = None,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.init_std = config.init_std
        self.residual_init_std = config.init_std if residual_init_std is None else residual_init_std
        self.attn_impl = config.attn_impl
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        dd = factory_kwargs(device, dtype)

        self.q_proj = _make_linear(config, config.hidden_size, self.num_heads * self.head_dim, bias=False, **dd)
        self.k_proj = _make_linear(config, config.hidden_size, self.num_kv_heads * self.head_dim, bias=False, **dd)
        self.v_proj = _make_linear(config, config.hidden_size, self.num_kv_heads * self.head_dim, bias=False, **dd)
        self.o_proj = _make_linear(config, self.num_heads * self.head_dim, config.hidden_size, bias=False, **dd)

        self.q_norm = VisionRMSNorm(self.head_dim, eps=config.rms_norm_eps, **dd)
        self.k_norm = VisionRMSNorm(self.head_dim, eps=config.rms_norm_eps, **dd)
        self.v_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False, **dd)

    def forward(
            self,
            hidden_states: torch.Tensor,
            positions_xy: torch.Tensor,
            attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        query = self.q_norm(query)
        key = self.k_norm(key)
        value = self.v_norm(value)

        query = apply_multidim_rope(
            query,
            positions_xy,
            base_theta=self.config.rope_theta,
            scale_factor=self.config.rope_scale,
        ).transpose(1, 2)
        key = apply_multidim_rope(
            key,
            positions_xy,
            base_theta=self.config.rope_theta,
            scale_factor=self.config.rope_scale,
        ).transpose(1, 2)
        value = value.transpose(1, 2)

        if self.attn_impl == "sdpa":
            attn_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask[:, None, :, :],
                dropout_p=0.0,
                is_causal=False,
                scale=1.0,
                enable_gqa=self.num_key_value_groups > 1,
            )
        else:
            if self.num_key_value_groups > 1:
                key = key[:, :, None].expand(
                    batch_size,
                    self.num_kv_heads,
                    self.num_key_value_groups,
                    seq_len,
                    self.head_dim,
                )
                key = key.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
                value = value[:, :, None].expand(
                    batch_size,
                    self.num_kv_heads,
                    self.num_key_value_groups,
                    seq_len,
                    self.head_dim,
                )
                value = value.reshape(batch_size, self.num_heads, seq_len, self.head_dim)

            attn_logits = torch.matmul(query, key.transpose(-1, -2))
            attn_logits = attn_logits.masked_fill(~attention_mask[:, None, :, :], K_MASK)
            attn_probs = F.softmax(attn_logits.float(), dim=-1).to(dtype=query.dtype)
            attn_output = torch.matmul(attn_probs, value)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)

    def _init_weights(self, ctx: InitContext) -> None:
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.k_proj.bias is not None:
            nn.init.zeros_(self.k_proj.bias)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.v_proj.bias is not None:
            nn.init.zeros_(self.v_proj.bias)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=self.residual_init_std, generator=ctx.generator)
        if self.o_proj.bias is not None:
            nn.init.zeros_(self.o_proj.bias)
        if self.q_norm.weight is not None:
            nn.init.zeros_(self.q_norm.weight)
        if self.k_norm.weight is not None:
            nn.init.zeros_(self.k_norm.weight)
        if self.v_norm.weight is not None:
            nn.init.ones_(self.v_norm.weight)


class Gemma4VisionBlock(InitModule):
    def __init__(
            self,
            config: VisionConfig,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        dd = factory_kwargs(device, dtype)
        residual_init_std = resolve_residual_init_std(
            config.init_std,
            config.residual_init_std,
            config.use_depth_scaled_residual_init,
            config.num_layers,
        )
        self.input_norm = VisionRMSNorm(config.hidden_size, eps=config.rms_norm_eps, **dd)
        self.post_attn_norm = VisionRMSNorm(config.hidden_size, eps=config.rms_norm_eps, **dd)
        self.pre_ffn_norm = VisionRMSNorm(config.hidden_size, eps=config.rms_norm_eps, **dd)
        self.post_ffn_norm = VisionRMSNorm(config.hidden_size, eps=config.rms_norm_eps, **dd)
        self.attn = Gemma4VisionAttention(config, residual_init_std=residual_init_std, **dd)
        self.mlp = Gemma4VisionMLP(config, residual_init_std=residual_init_std, **dd)

    def forward(
            self,
            hidden_states: torch.Tensor,
            positions_xy: torch.Tensor,
            attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        hidden_states = self.attn(hidden_states, positions_xy, attention_mask)
        hidden_states = self.post_attn_norm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_ffn_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_ffn_norm(hidden_states)
        return residual + hidden_states

    def _init_weights(self, ctx: InitContext) -> None:
        if self.input_norm.weight is not None:
            nn.init.zeros_(self.input_norm.weight)
        if self.post_attn_norm.weight is not None:
            nn.init.zeros_(self.post_attn_norm.weight)
        if self.pre_ffn_norm.weight is not None:
            nn.init.zeros_(self.pre_ffn_norm.weight)
        if self.post_ffn_norm.weight is not None:
            nn.init.zeros_(self.post_ffn_norm.weight)


class Gemma4VisionPooler(InitModule):
    def __init__(
            self,
            config: VisionConfig,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        _ = device, dtype
        self.config = config
        self.root_hidden_size = math.sqrt(config.hidden_size)

    def _pool_once(
            self,
            hidden_states: torch.Tensor,
            positions_xy: torch.Tensor,
            output_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if output_length > hidden_states.shape[1]:
            raise ValueError(f"Requested {output_length} soft tokens from {hidden_states.shape[1]} patches.")

        if output_length == hidden_states.shape[1]:
            mask = ~(positions_xy == POSITIONS_PAD_VALUE).all(dim=-1)
            return hidden_states * self.root_hidden_size, mask

        pooled, mask = avg_pool_by_positions(hidden_states, positions_xy, output_length)
        return pooled * self.root_hidden_size, mask

    def forward(
            self,
            hidden_states: torch.Tensor,
            positions_xy: torch.Tensor,
            output_length_overrides: tuple[int, ...] | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        output_lengths = output_length_overrides or self.config.output_length
        if isinstance(output_lengths, int):
            output_lengths = (output_lengths,)
        return tuple(
            self._pool_once(hidden_states, positions_xy, output_length)
            for output_length in output_lengths
            if output_length <= hidden_states.shape[1]
        )


class Gemma4VisionStandardize(InitModule):
    def __init__(
            self,
            hidden_size: int,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        dd = factory_kwargs(device, dtype)
        self.scale = nn.Parameter(torch.ones(hidden_size, **dd))
        self.bias = nn.Parameter(torch.zeros(hidden_size, **dd))

    def _init_weights(self, ctx: InitContext) -> None:
        nn.init.ones_(self.scale)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.bias) * self.scale


class Gemma4VisionEncoder(InitModule):
    def __init__(
            self,
            config: VisionConfig,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        dd = factory_kwargs(device, dtype)
        self.config = config
        self.image_processor = Gemma4ImageProcessor.from_config(config)
        self.patch_embed = Gemma4VisionPatchEmbed(config, **dd)
        self.layers = nn.ModuleList([Gemma4VisionBlock(config, **dd) for _ in range(config.num_layers)])
        self.pooler = Gemma4VisionPooler(config, **dd)
        self.standardize = Gemma4VisionStandardize(config.hidden_size, **dd) if config.standardize_embeddings else None

    def preprocess_images(self, images: ImageBatchInput) -> Gemma4ImageBatch:
        """Convert raw images into padded patch tensors for the vision stack."""
        return self.image_processor.preprocess(images)

    def resolve_patch_inputs(
            self,
            patches_or_images: ImageBatchInput | torch.Tensor,
            positions_xy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve raw images or patch tensors into patch and position tensors."""
        if positions_xy is None:
            if not torch.is_tensor(patches_or_images) or _looks_like_raw_image_tensor(patches_or_images):
                image_batch = self.preprocess_images(patches_or_images)
                return image_batch.pixel_values, image_batch.image_position_ids
            raise ValueError("positions_xy is required when passing patch tensors.")

        return patches_or_images, positions_xy

    def forward(
            self,
            patches_or_images: ImageBatchInput | torch.Tensor,
            positions_xy: torch.Tensor | None = None,
            output_length_overrides: tuple[int, ...] | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        patches, positions_xy = self.resolve_patch_inputs(patches_or_images, positions_xy)

        patch_device = self.patch_embed.position_table.device
        patches = normalize_image_patches(patches).to(device=patch_device)
        positions_xy = positions_xy.to(device=patch_device)
        input_mask = ~(positions_xy == POSITIONS_PAD_VALUE).all(dim=-1)
        hidden_states = self.patch_embed(patches, positions_xy)
        attention_mask = input_mask[:, :, None] & input_mask[:, None, :]
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions_xy, attention_mask)

        outputs = self.pooler(hidden_states, positions_xy, output_length_overrides)
        if self.standardize is None:
            return outputs

        standardized: list[tuple[torch.Tensor, torch.Tensor]] = []
        for tokens, mask in outputs:
            standardized.append((self.standardize(tokens.float()).to(dtype=tokens.dtype), mask))
        return tuple(standardized)


class Gemma4VisionTower(InitModule):
    def __init__(
            self,
            config: VisionConfig,
            text_hidden_size: int | None = None,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        dd = factory_kwargs(device, dtype)
        self.config = config
        self.init_std = config.init_std
        self.encoder = Gemma4VisionEncoder(config, **dd)
        self.to_text_norm = None
        self.to_text = None
        if text_hidden_size is not None:
            self.to_text_norm = RMSNorm(config.hidden_size, eps=config.projection_norm_eps, with_scale=False, **dd)
            self.to_text = nn.Linear(config.hidden_size, text_hidden_size, bias=False, **dd)

    def forward(
            self,
            patches_or_images: ImageBatchInput | torch.Tensor,
            positions_xy: torch.Tensor | None = None,
            output_length_overrides: tuple[int, ...] | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        return self.encoder(patches_or_images, positions_xy, output_length_overrides)

    def preprocess_images(self, images: ImageBatchInput) -> Gemma4ImageBatch:
        """Convert raw images into padded patch tensors."""
        return self.encoder.preprocess_images(images)

    @staticmethod
    def _flatten_image_groups(
            patches: torch.Tensor,
            positions_xy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int] | None]:
        if patches.ndim == 5 or (patches.ndim == 4 and positions_xy.ndim == 4):
            batch, num_images = patches.shape[:2]
            return (
                patches.view(batch * num_images, *patches.shape[2:]),
                positions_xy.view(batch * num_images, *positions_xy.shape[2:]),
                (batch, num_images),
            )
        return patches, positions_xy, None

    def encode_to_text(
            self,
            patches_or_images: ImageBatchInput | torch.Tensor,
            positions_xy: torch.Tensor | None = None,
            output_length_overrides: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patches, positions = self.encoder.resolve_patch_inputs(patches_or_images, positions_xy)
        patches, positions, grouped_shape = self._flatten_image_groups(patches, positions)

        tokens, mask = self.encoder(patches, positions, output_length_overrides)[0]
        if self.to_text is not None:
            tokens = self.to_text(self.to_text_norm(tokens))

        if grouped_shape is not None:
            batch, num_images = grouped_shape
            tokens = tokens.view(batch, num_images, tokens.shape[1], tokens.shape[2])
            mask = mask.view(batch, num_images, mask.shape[1])
            tokens = tokens.flatten(1, 2)
            mask = mask.flatten(1, 2)
        return tokens, mask

    def _init_weights(self, ctx: InitContext) -> None:
        if self.to_text_norm is not None and self.to_text_norm.weight is not None:
            nn.init.ones_(self.to_text_norm.weight)
        if self.to_text is not None:
            nn.init.normal_(self.to_text.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
            if self.to_text.bias is not None:
                nn.init.zeros_(self.to_text.bias)
