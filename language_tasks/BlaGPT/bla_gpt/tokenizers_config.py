"""
Unified tokenizer configuration and wrapper for BlaGPT.

Supports both tiktoken and HuggingFace tokenizers with a consistent API.
"""

from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer selection."""

    backend: str = "tiktoken"  # "tiktoken" | "huggingface"
    name: str = "gpt2"  # tiktoken: "gpt2", "cl100k_base" | HF: model name/path
    trust_remote_code: bool = False  # For HF tokenizers that require it


class TokenizerWrapper:
    """Unified interface for tiktoken and HuggingFace tokenizers."""

    def __init__(self, config: TokenizerConfig):
        self.config = config
        self._tokenizer = None
        self._vocab_size = None
        self._eos_token_id = None

        if config.backend == "tiktoken":
            self._init_tiktoken()
        elif config.backend == "huggingface":
            self._init_huggingface()
        else:
            raise ValueError(
                f"Unknown tokenizer backend: {config.backend}. "
                "Supported: 'tiktoken', 'huggingface'"
            )

    def _init_tiktoken(self):
        """Initialize tiktoken tokenizer."""
        import tiktoken

        try:
            self._tokenizer = tiktoken.get_encoding(self.config.name)
        except KeyError:
            available = tiktoken.list_encoding_names()
            raise ValueError(
                f"Unknown tiktoken encoding: {self.config.name}. "
                f"Available: {available}"
            )

        self._vocab_size = self._tokenizer.n_vocab
        self._eos_token_id = self._tokenizer.eot_token

    def _init_huggingface(self):
        """Initialize HuggingFace tokenizer."""
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.name,
            trust_remote_code=self.config.trust_remote_code,
        )

        self._vocab_size = len(self._tokenizer)
        self._eos_token_id = self._tokenizer.eos_token_id

        # Fallback if eos_token_id is None
        if self._eos_token_id is None:
            if self._tokenizer.pad_token_id is not None:
                self._eos_token_id = self._tokenizer.pad_token_id
            else:
                self._eos_token_id = 0

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if self.config.backend == "tiktoken":
            return self._tokenizer.encode_ordinary(text)
        else:
            return self._tokenizer.encode(text, add_special_tokens=False)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode multiple texts to token IDs."""
        if self.config.backend == "tiktoken":
            return [self._tokenizer.encode_ordinary(t) for t in texts]
        else:
            encoded = self._tokenizer(
                texts, add_special_tokens=False, return_attention_mask=False
            )
            return encoded["input_ids"]

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        if self.config.backend == "tiktoken":
            return self._tokenizer.decode(tokens)
        else:
            return self._tokenizer.decode(tokens)

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self._vocab_size

    @property
    def eos_token_id(self) -> int:
        """Return end-of-sequence token ID."""
        return self._eos_token_id

    @property
    def padded_vocab_size(self) -> int:
        """Return vocab size padded to multiple of 64 for efficiency."""
        return ((self._vocab_size + 63) // 64) * 64

    def __repr__(self) -> str:
        return (
            f"TokenizerWrapper(backend={self.config.backend!r}, "
            f"name={self.config.name!r}, vocab_size={self._vocab_size})"
        )
