from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch
from tokenizers import Tokenizer as FastTokenizer


TOKENIZER_MODEL_NAME = "tokenizer.model"
TOKENIZER_JSON_NAME = "tokenizer.json"
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
    """Tokenizer wrapper for Gemma 4.

    This wrapper supports either the canonical SentencePiece model or an HF-style
    `tokenizer.json` asset.
    """

    def __init__(self, tokenizer_file: str | Path) -> None:
        self.tokenizer_file = Path(tokenizer_file)
        self.special_tokens = Gemma4SpecialTokens()
        self.sp_model: spm.SentencePieceProcessor | None = None
        self.fast_tokenizer: FastTokenizer | None = None
        self.tokenizer_file = self._resolve_tokenizer_file(self.tokenizer_file)
        self.model_file = self.tokenizer_file

        if self.tokenizer_file.suffix == ".json":
            try:
                self.fast_tokenizer = FastTokenizer.from_file(str(self.tokenizer_file))
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"Failed to load tokenizer JSON from {self.tokenizer_file}."
                ) from exc
            self.backend = "tokenizers"
            return

        try:
            self.sp_model = spm.SentencePieceProcessor(model_file=str(self.tokenizer_file))
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load SentencePiece model from {self.tokenizer_file}."
            ) from exc

        self.backend = "sentencepiece"

    @staticmethod
    def _resolve_tokenizer_file(path: Path) -> Path:
        if path.is_file():
            return path

        config_path = path / TOKENIZER_CONFIG_NAME
        if config_path.exists():
            try:
                with config_path.open(encoding="utf-8") as f:
                    config = json.load(f)
            except OSError as exc:
                raise OSError(f"Failed to read tokenizer config from {config_path}.") from exc
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid tokenizer config in {config_path}.") from exc

            for config_key in ("model_file", "tokenizer_file"):
                token_name = config.get(config_key)
                if token_name is None:
                    continue
                token_path = path / str(token_name)
                if token_path.exists():
                    return token_path

        for candidate in (
            path / TOKENIZER_MODEL_NAME,
            path / TOKENIZER_JSON_NAME,
            path / "gemma4_cleaned_262144.model",
            path / "tokenizer.spm",
        ):
            if candidate.exists():
                return candidate

        model_files = sorted(path.glob("*.model"))
        if model_files:
            return model_files[0]
        json_files = sorted(path.glob("*.json"))
        for candidate in json_files:
            if candidate.name in {TOKENIZER_JSON_NAME, "tokenizer_fast.json"}:
                return candidate
        raise FileNotFoundError(f"Could not find a tokenizer asset in {path}.")

    @classmethod
    def from_pretrained(
            cls,
            path: str | Path,
    ) -> "Gemma4Tokenizer":
        """Load a tokenizer from a file or tokenizer directory.

        Args:
            path: Tokenizer file or directory containing tokenizer assets.
        """
        return cls(cls._resolve_tokenizer_file(Path(path)))

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

        target_name = (
            TOKENIZER_MODEL_NAME
            if self.backend == "sentencepiece"
            else TOKENIZER_JSON_NAME
        )
        target = save_directory / target_name
        if self.tokenizer_file.resolve() != target.resolve():
            try:
                shutil.copyfile(self.tokenizer_file, target)
            except OSError as exc:
                raise OSError(f"Failed to copy tokenizer asset to {target}.") from exc

        try:
            with (save_directory / TOKENIZER_CONFIG_NAME).open("w", encoding="utf-8") as f:
                config_data = {"tokenizer_class": type(self).__name__}
                if self.backend == "sentencepiece":
                    config_data["model_file"] = TOKENIZER_MODEL_NAME
                else:
                    config_data["tokenizer_file"] = TOKENIZER_JSON_NAME
                json.dump(config_data, f, indent=2)
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
        token_ids = self._encode_to_ids(text)
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
        return self._decode_ids(token_ids, skip_special_tokens=skip_special_tokens)

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
        if self.fast_tokenizer is not None:
            token_id = self.fast_tokenizer.token_to_id(token)
            return None if token_id is None else int(token_id)

        if self.sp_model is None:
            raise RuntimeError("Tokenizer backend was not initialized.")

        token_id = int(self.sp_model.piece_to_id(token))
        if token_id < 0:
            return None
        if self.sp_model.id_to_piece(token_id) != token:
            return None
        return token_id

    def id_to_token(self, token_id: int) -> str:
        if self.fast_tokenizer is not None:
            token = self.fast_tokenizer.id_to_token(int(token_id))
            if token is None:
                raise KeyError(f"Unknown token id: {token_id}.")
            return token

        if self.sp_model is None:
            raise RuntimeError("Tokenizer backend was not initialized.")
        return self.sp_model.id_to_piece(int(token_id))

    @property
    def vocab_size(self) -> int:
        if self.fast_tokenizer is not None:
            return int(self.fast_tokenizer.get_vocab_size())
        if self.sp_model is None:
            raise RuntimeError("Tokenizer backend was not initialized.")
        return int(self.sp_model.vocab_size())

    @property
    def pad_token_id(self) -> int:
        return self._required_special_id(self.special_tokens.pad, "pad")

    @property
    def eos_token_id(self) -> int:
        if self.fast_tokenizer is not None:
            return self._required_special_id(self.special_tokens.eos, "eos")
        if self.sp_model is None:
            raise RuntimeError("Tokenizer backend was not initialized.")
        return int(self.sp_model.eos_id())

    @property
    def bos_token_id(self) -> int:
        if self.fast_tokenizer is not None:
            return self._required_special_id(self.special_tokens.bos, "bos")
        if self.sp_model is None:
            raise RuntimeError("Tokenizer backend was not initialized.")
        return int(self.sp_model.bos_id())

    @property
    def unk_token_id(self) -> int:
        if self.fast_tokenizer is not None:
            return self._required_special_id(self.special_tokens.unk, "unk")
        if self.sp_model is None:
            raise RuntimeError("Tokenizer backend was not initialized.")
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

    def _encode_to_ids(self, text: str) -> list[int]:
        if self.fast_tokenizer is not None:
            return list(self.fast_tokenizer.encode(text).ids)
        if self.sp_model is None:
            raise RuntimeError("Tokenizer backend was not initialized.")
        return list(self.sp_model.encode(text, out_type=int))

    def _decode_ids(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        if self.fast_tokenizer is not None:
            return self.fast_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        if self.sp_model is None:
            raise RuntimeError("Tokenizer backend was not initialized.")
        return self.sp_model.decode(token_ids)

    def _required_special_id(self, token: str, token_name: str) -> int:
        token_id = self.token_to_id(token)
        if token_id is None:
            raise KeyError(f"Tokenizer is missing the required {token_name} token {token!r}.")
        return token_id
