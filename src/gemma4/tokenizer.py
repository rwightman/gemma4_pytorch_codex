from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch


TOKENIZER_MODEL_NAME = "tokenizer.model"
TOKENIZER_CONFIG_NAME = "tokenizer_config.json"


@dataclass(frozen=True)
class Gemma4SpecialTokens:
    pad: str = "<pad>"
    eos: str = "<eos>"
    bos: str = "<bos>"
    unk: str = "<unk>"
    mask: str = "<mask>"
    sot: str = "<|turn>"
    eot: str = "<turn|>"
    image: str = "<|image|>"
    boi: str = "<|image>"
    eoi: str = "<image|>"
    audio: str = "<|audio|>"
    boa: str = "<|audio>"
    eoa: str = "<audio|>"
    soc: str = "<|channel>"
    eoc: str = "<channel|>"
    think: str = "<|think|>"
    escape: str = '<|"|>'
    str_token: str = "<|tool_response>"
    etr: str = "<tool_response|>"
    stc: str = "<|tool_call>"
    etc: str = "<tool_call|>"
    std: str = "<|tool>"
    etd: str = "<tool|>"


class Gemma4Tokenizer:
    """SentencePiece tokenizer wrapper for Gemma 4."""

    def __init__(self, model_file: str | Path) -> None:
        self.model_file = Path(model_file)

        try:
            self.sp_model = spm.SentencePieceProcessor(model_file=str(self.model_file))
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load SentencePiece model from {self.model_file}."
            ) from exc

        self.special_tokens = Gemma4SpecialTokens()

    @classmethod
    def from_pretrained(
            cls,
            path: str | Path,
    ) -> "Gemma4Tokenizer":
        """Load a tokenizer from a model file or model directory.

        Args:
            path: Tokenizer model file or directory containing tokenizer assets.
        """
        path = Path(path)
        if path.is_file():
            return cls(path)

        config_path = path / TOKENIZER_CONFIG_NAME
        if config_path.exists():
            try:
                with config_path.open(encoding="utf-8") as f:
                    config = json.load(f)
            except OSError as exc:
                raise OSError(f"Failed to read tokenizer config from {config_path}.") from exc
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid tokenizer config in {config_path}.") from exc

            model_name = config.get("model_file", TOKENIZER_MODEL_NAME)
            model_path = path / model_name
            if model_path.exists():
                return cls(model_path)

        for candidate in (
            path / TOKENIZER_MODEL_NAME,
            path / "gemma4_cleaned_262144.model",
            path / "tokenizer.spm",
        ):
            if candidate.exists():
                return cls(candidate)

        model_files = sorted(path.glob("*.model"))
        if model_files:
            return cls(model_files[0])
        raise FileNotFoundError(f"Could not find a SentencePiece tokenizer model in {path}.")

    def save_pretrained(
            self,
            save_directory: str | Path,
    ) -> None:
        """Save tokenizer assets to a directory.

        Args:
            save_directory: Destination directory for tokenizer files.
        """
        save_directory = Path(save_directory)
        try:
            save_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OSError(f"Failed to create tokenizer directory {save_directory}.") from exc

        target = save_directory / TOKENIZER_MODEL_NAME
        if self.model_file.resolve() != target.resolve():
            try:
                shutil.copyfile(self.model_file, target)
            except OSError as exc:
                raise OSError(f"Failed to copy tokenizer model to {target}.") from exc

        try:
            with (save_directory / TOKENIZER_CONFIG_NAME).open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "tokenizer_class": type(self).__name__,
                        "model_file": TOKENIZER_MODEL_NAME,
                    },
                    f,
                    indent=2,
                )
        except OSError as exc:
            raise OSError(
                f"Failed to write tokenizer config to {save_directory / TOKENIZER_CONFIG_NAME}."
            ) from exc

    def encode(
            self,
            text: str,
            *,
            add_bos: bool = False,
            add_eos: bool = False,
    ) -> list[int]:
        token_ids = list(self.sp_model.encode(text, out_type=int))
        if add_bos and self.bos_token_id is not None:
            token_ids.insert(0, self.bos_token_id)
        if add_eos and self.eos_token_id is not None:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(
            self,
            token_ids: int | list[int] | torch.Tensor,
            *,
            skip_special_tokens: bool = False,
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        elif isinstance(token_ids, int):
            token_ids = [token_ids]

        if skip_special_tokens:
            special_ids = self.all_special_token_ids
            token_ids = [
                int(token_id)
                for token_id in token_ids
                if int(token_id) not in special_ids
            ]
        return self.sp_model.decode(token_ids)

    def batch_decode(
            self,
            batch_token_ids: torch.Tensor | list[list[int]],
            *,
            skip_special_tokens: bool = False,
    ) -> list[str]:
        if isinstance(batch_token_ids, torch.Tensor):
            batch_token_ids = batch_token_ids.detach().cpu().tolist()
        return [
            self.decode(token_ids, skip_special_tokens=skip_special_tokens)
            for token_ids in batch_token_ids
        ]

    def __call__(
            self,
            text: str | list[str],
            *,
            add_bos: bool = False,
            add_eos: bool = False,
            padding: bool | str = False,
            return_tensors: str | None = None,
    ) -> dict[str, Any]:
        """Tokenize a string or batch of strings."""
        texts = [text] if isinstance(text, str) else list(text)
        encoded = [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]
        if padding:
            max_len = max(len(ids) for ids in encoded)
            pad_id = self.pad_token_id
            padded = []
            masks = []
            for ids in encoded:
                pad_len = max_len - len(ids)
                padded.append([pad_id] * pad_len + ids)
                masks.append([0] * pad_len + [1] * len(ids))
        else:
            padded = encoded
            masks = [[1] * len(ids) for ids in encoded]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.bool),
            }
        return {
            "input_ids": padded,
            "attention_mask": masks,
        }

    def token_to_id(self, token: str) -> int | None:
        token_id = int(self.sp_model.piece_to_id(token))
        if token_id < 0:
            return None
        if self.sp_model.id_to_piece(token_id) != token:
            return None
        return token_id

    def id_to_token(self, token_id: int) -> str:
        return self.sp_model.id_to_piece(int(token_id))

    @property
    def vocab_size(self) -> int:
        return int(self.sp_model.vocab_size())

    @property
    def pad_token_id(self) -> int:
        return int(self.sp_model.pad_id())

    @property
    def eos_token_id(self) -> int:
        return int(self.sp_model.eos_id())

    @property
    def bos_token_id(self) -> int:
        return int(self.sp_model.bos_id())

    @property
    def unk_token_id(self) -> int:
        return int(self.sp_model.unk_id())

    @property
    def image_token_id(self) -> int | None:
        return self.token_to_id(self.special_tokens.image)

    @property
    def boi_token_id(self) -> int | None:
        return self.token_to_id(self.special_tokens.boi)

    @property
    def eoi_token_id(self) -> int | None:
        return self.token_to_id(self.special_tokens.eoi)

    @property
    def audio_token_id(self) -> int | None:
        return self.token_to_id(self.special_tokens.audio)

    @property
    def boa_token_id(self) -> int | None:
        return self.token_to_id(self.special_tokens.boa)

    @property
    def eoa_token_id(self) -> int | None:
        return self.token_to_id(self.special_tokens.eoa)

    @property
    def all_special_token_ids(self) -> set[int]:
        optional_ids = {
            self.token_to_id(self.special_tokens.sot),
            self.token_to_id(self.special_tokens.eot),
            self.token_to_id(self.special_tokens.image),
            self.token_to_id(self.special_tokens.boi),
            self.token_to_id(self.special_tokens.eoi),
            self.token_to_id(self.special_tokens.audio),
            self.token_to_id(self.special_tokens.boa),
            self.token_to_id(self.special_tokens.eoa),
            self.token_to_id(self.special_tokens.soc),
            self.token_to_id(self.special_tokens.eoc),
            self.token_to_id(self.special_tokens.think),
            self.token_to_id(self.special_tokens.escape),
            self.token_to_id(self.special_tokens.str_token),
            self.token_to_id(self.special_tokens.etr),
            self.token_to_id(self.special_tokens.stc),
            self.token_to_id(self.special_tokens.etc),
            self.token_to_id(self.special_tokens.std),
            self.token_to_id(self.special_tokens.etd),
        }
        required_ids = {
            self.pad_token_id,
            self.eos_token_id,
            self.bos_token_id,
            self.unk_token_id,
        }
        return required_ids | {token_id for token_id in optional_ids if token_id is not None}
