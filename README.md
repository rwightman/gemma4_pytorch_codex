# gemma4-pytorch-codex

`gemma4-pytorch-codex` is a clean PyTorch implementation of the Gemma 4 model family.

It is meant to stay close to the original JAX structure while still looking like normal PyTorch code:

- text, vision, and audio towers are separate and reusable
- the top-level model is thin
- generation and KV-cache are built in
- multimodal prompt expansion is explicit instead of hidden in a large framework wrapper
- checkpoint conversion supports both Hugging Face safetensors and Orbax/JAX checkpoints

The install/import split is:

- distribution name: `gemma4-pytorch-codex`
- import package: `gemma4_pt_codex`

Typical usage:

```python
import gemma4_pt_codex as gemma4
```

## Current State

What is working today:

- text generation
- KV-cache decode
- image preprocessing and image-conditioned generation
- audio preprocessing and audio-conditioned generation
- native save/load
- conversion from local HF safetensors checkpoints
- conversion from local Orbax/JAX checkpoints

The most-tested public checkpoint path right now is the small instruction-tuned model:

- `google/gemma-4-e2b-it`

That checkpoint has been exercised locally for:

- plain text generation
- image captioning
- audio transcription-style prompts

## Install

Editable install:

```bash
cd gemma4_pytorch_codex
python -m pip install -e .
```

With PDM:

```bash
cd gemma4_pytorch_codex
pdm install
```

Conversion extras:

```bash
cd gemma4_pytorch_codex
python -m pip install -e ".[convert]"
```

Dev/test extras:

```bash
cd gemma4_pytorch_codex
python -m pip install -e ".[dev]"
```

## Loading A Checkpoint

Native checkpoints use:

- `config.json`
- `model.safetensors`
- tokenizer assets such as `tokenizer.model` or `tokenizer.json`

Load a native checkpoint:

```python
import torch

import gemma4_pt_codex as gemma4

tokenizer = gemma4.Gemma4Tokenizer.from_pretrained("/path/to/checkpoint_dir")
model = gemma4.Gemma4Model.from_pretrained(
    "/path/to/checkpoint_dir",
    dtype=torch.bfloat16,
)
model.eval()
```

If you want a different attention backend at load time:

```python
model = gemma4.Gemma4Model.from_pretrained(
    "/path/to/checkpoint_dir",
    attn_impl="sdpa",
)
```

Supported attention implementations:

- `"eager"`
- `"sdpa"`

`"sdpa"` currently applies to text and vision. Audio remains eager.

## Text Generation

The simplest path is `generate_text()`:

```python
import gemma4_pt_codex as gemma4

tokenizer = gemma4.Gemma4Tokenizer.from_pretrained("/path/to/checkpoint_dir")
model = gemma4.Gemma4Model.from_pretrained("/path/to/checkpoint_dir")
model.eval()

text = model.generate_text(
    tokenizer,
    "Write a haiku about cedar trees in coastal fog.",
    max_new_tokens=32,
    do_sample=False,
)
print(text)
```

For instruction-tuned checkpoints, use the turn format explicitly:

```python
prompt = (
    "<|turn>user\n"
    "Explain rotary position embeddings in plain English.\n"
    "<turn|>\n"
    "<|turn>model\n"
)

text = model.generate_text(
    tokenizer,
    prompt,
    max_new_tokens=128,
    do_sample=False,
)
print(text)
```

If you want lower-level control, call `prepare_inputs()` and `generate()` directly:

```python
prepared = model.prepare_inputs(
    tokenizer,
    prompt,
)

generated = model.generate(
    prepared.input_ids,
    attention_mask=prepared.attention_mask,
    max_new_tokens=128,
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id,
)

continuation = generated[:, prepared.input_ids.shape[1] :]
print(tokenizer.batch_decode(continuation, skip_special_tokens=True)[0])
```

## Image Interaction

The package includes a `Gemma4ImageProcessor` for raw image preprocessing and a higher-level
`Gemma4Processor` that handles multimodal prompt expansion.

The user-facing rule is simple:

- put one visible `<|image|>` token in the prompt for each image
- pass the actual images through `images=...`

Example captioning flow:

```python
from PIL import Image

import gemma4_pt_codex as gemma4

tokenizer = gemma4.Gemma4Tokenizer.from_pretrained("/path/to/checkpoint_dir")
model = gemma4.Gemma4Model.from_pretrained("/path/to/checkpoint_dir")
image = Image.open("/path/to/image.jpg").convert("RGB")

prompt = (
    "<|turn>user\n"
    "<|image|>Caption this image in one short sentence.\n"
    "<turn|>\n"
    "<|turn>model\n"
)

text = model.generate_text(
    tokenizer,
    prompt,
    images=image,
    max_new_tokens=64,
    do_sample=False,
)
print(text)
```

Multiple images work the same way:

- include multiple visible `<|image|>` tokens
- pass the images in the same order

Internally, the processor expands each visible image token into the Gemma4-style image span:

- `\n\n`
- `<|image>`
- internal soft-image placeholder repeated `N` times
- `<image|>`
- `\n\n`

Those internal placeholder positions are later replaced with projected vision tokens before the text
decoder runs.

### Image Preprocessing

The raw-image path is:

1. Convert to RGB
2. Resize with aspect ratio preserved
3. No crop
4. Scale pixels to `[0, 1]`
5. Patchify
6. Pad to the configured patch budget
7. Build 2D patch positions

The vision stack then applies the model-side `[0, 1] -> [-1, 1]` patch normalization before the patch
projection, matching the effective JAX behavior while keeping the patch embed layer itself clean.

If you want the processor output directly:

```python
processor = gemma4.Gemma4ImageProcessor.from_config(model.config.vision)
batch = processor.preprocess(image)

vision_tokens, vision_mask = model.encode_images_to_text(
    batch.pixel_values,
    batch.image_position_ids,
)
```

Current practical note:

- general captioning works
- OCR-style prompts may work on simple large text
- dense text-heavy images are still a harder case

## Audio Interaction

The package includes a `Gemma4AudioProcessor` and the same high-level `Gemma4Processor` handles audio
placeholder expansion.

The user-facing rule is the audio equivalent of image prompting:

- put one visible `<|audio|>` token in the prompt for each audio clip
- pass the actual clip through `audios=...`

Example transcription-style flow:

```python
import gemma4_pt_codex as gemma4

tokenizer = gemma4.Gemma4Tokenizer.from_pretrained("/path/to/checkpoint_dir")
model = gemma4.Gemma4Model.from_pretrained("/path/to/checkpoint_dir")

prompt = (
    "<|turn>user\n"
    "<|audio|>\n"
    "Transcribe this audio clip exactly.\n"
    "<turn|>\n"
    "<|turn>model\n"
)

text = model.generate_text(
    tokenizer,
    prompt,
    audios="/path/to/audio.wav",
    max_new_tokens=128,
    do_sample=False,
)
print(text)
```

You can also pass waveform data directly:

```python
import torch

waveform = torch.randn(16000 * 5)

text = model.generate_text(
    tokenizer,
    prompt,
    audios=(waveform, 16000),
    max_new_tokens=128,
    do_sample=False,
)
```

Important detail:

- if you pass a raw tensor or array without a sample rate, it is assumed to already be `16 kHz`
- if you pass `(waveform, sample_rate)` or a file path, the audio processor will resample to `16 kHz`

Internally, the processor expands each visible audio token into:

- `<|audio>`
- internal soft-audio placeholder repeated `N` times
- `<audio|>`

### Audio Preprocessing

The audio path is:

1. Load the waveform
2. Convert to mono
3. Convert to `float32`
4. Resample to `16 kHz` if needed
5. Compute log-mel features
6. Build a valid-frame mask
7. Compute the number of soft audio tokens implied by the waveform length

If you want the processor output directly:

```python
audio_processor = gemma4.Gemma4AudioProcessor.from_config(model.config.audio)
audio_batch = audio_processor.preprocess("/path/to/audio.wav")

audio_tokens, audio_mask = model.encode_audio_to_text(
    audio_batch.input_features,
    audio_batch.input_features_mask,
)
```

The current public-checkpoint path has been validated on transcription-style prompts against the small
instruction-tuned checkpoint.

## Multimodal Preparation

`Gemma4Model.prepare_inputs()` is the easiest way to see exactly what will be fed into the model:

```python
prepared = model.prepare_inputs(
    tokenizer,
    prompt,
    images=image,
    audios="/path/to/audio.wav",
)

print(prepared.input_ids.shape)
print(None if prepared.vision_tokens is None else prepared.vision_tokens.shape)
print(None if prepared.audio_tokens is None else prepared.audio_tokens.shape)
```

That returns a `Gemma4PreparedInputs` object with:

- `input_ids`
- `attention_mask`
- `vision_tokens`
- `vision_token_mask`
- `audio_tokens`
- `audio_token_mask`

It can be moved and unpacked directly:

```python
prepared = prepared.to("cuda", dtype=torch.bfloat16)
output = model(**prepared.as_forward_kwargs(), return_hidden_states=True)
```

If you want the tokenizer plus multimodal preprocessing layer directly, build a processor:

```python
processor = model.build_processor(tokenizer)
batch = processor(
    prompt,
    images=image,
    audios="/path/to/audio.wav",
    add_bos=True,
    padding=True,
)
```

## Tokenizer

`Gemma4Tokenizer` supports:

- SentencePiece tokenizer assets such as `tokenizer.model`
- HF fast-tokenizer assets such as `tokenizer.json`

Example:

```python
import gemma4_pt_codex as gemma4

tokenizer = gemma4.Gemma4Tokenizer.from_pretrained("/path/to/checkpoint_dir")
ids = tokenizer.encode("hello world", add_bos=True)
text = tokenizer.decode(ids)
```

## Convert Weights

### Convert from a local Hugging Face checkpoint

CLI:

```bash
gemma4-pt-codex-convert hf /path/to/hf_checkpoint /path/to/native_checkpoint
```

Python:

```python
import gemma4_pt_codex as gemma4

gemma4.convert_hf_checkpoint(
    "/path/to/hf_checkpoint",
    "/path/to/native_checkpoint",
)
```

Supported HF inputs:

- single-file safetensors checkpoints
- sharded safetensors checkpoints
- tokenizer assets from either SentencePiece or `tokenizer.json`

### Convert from a local Orbax/JAX checkpoint

CLI:

```bash
gemma4-pt-codex-convert orbax /path/to/orbax_checkpoint /path/to/native_checkpoint --variant gemma-4-e2b
```

Python:

```python
import gemma4_pt_codex as gemma4

gemma4.convert_orbax_checkpoint(
    "/path/to/orbax_checkpoint",
    "/path/to/native_checkpoint",
    variant="gemma-4-e2b",
)
```

The Orbax path is the closest route to the original JAX parameter tree.

## Reusable Towers

The modalities are intentionally separated.

Text-only:

```python
import gemma4_pt_codex as gemma4

config = gemma4.gemma4_e2b_config(text_only=True)
text = gemma4.Gemma4TextTower(config.text)
```

Vision:

```python
import gemma4_pt_codex as gemma4

config = gemma4.gemma4_e2b_config()
encoder = gemma4.Gemma4VisionEncoder(config.vision)
tower = gemma4.Gemma4VisionTower(config.vision, text_hidden_size=config.text.hidden_size)
```

Audio:

```python
import gemma4_pt_codex as gemma4

config = gemma4.gemma4_e2b_config()
tower = gemma4.Gemma4AudioTower(config.audio, text_hidden_size=config.text.hidden_size)
```

Top-level model layout:

```python
model.text
model.vision
model.audio
```

## Architecture Notes

This package tries to stay close to the original JAX implementation where it matters:

- released preset configs
- alternating sliding/global attention patterns
- q/k/v normalization behavior
- RoPE choices for local and global attention
- trainable layer scaling
- multimodal placeholder expansion and merge behavior
- MoE support for `26B_A4B`
- cache-aware decode

It deliberately does not copy the style of the Transformers implementation:

- fewer wrappers
- flatter ownership
- modality towers live with their own embedders/projectors
- no `einsum` in the main path

## Verification

Current local coverage includes:

- text, vision, audio, save/load, and generation smoke tests
- config creation coverage for the main released variants
- HF conversion tests
- JAX parity tests for text and vision
- image preprocessing and multimodal expansion tests
- real conversion and generation smoke tests with `google/gemma-4-e2b-it`

Run the tests with:

```bash
cd gemma4_pytorch_codex
pytest tests -q
```

## Project Layout

```text
gemma4_pytorch_codex/
  pyproject.toml
  README.md
  src/gemma4_pt_codex/
    config.py
    layers.py
    image_processing.py
    audio_processing.py
    processing.py
    text.py
    vision.py
    audio.py
    model.py
    tokenizer.py
    convert.py
  tests/
```

## Status

This is a practical local implementation, not a polished upstream distribution.

It is a good fit if you want to:

- study Gemma 4 without framework noise
- convert checkpoints and run local generation
- reuse the image or audio towers in another project
- work on a PyTorch codebase that stays relatively close to the original model structure

It is not trying to be:

- a drop-in replacement for the full Transformers ecosystem
- a large training framework
- a kitchen-sink inference server

## License

Apache 2.0, matching the project metadata in [`pyproject.toml`](./pyproject.toml).
