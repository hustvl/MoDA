"""Tests for tokenizers_config.py"""

import sys
import os

# Add bla_gpt to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bla_gpt'))

import pytest
from tokenizers_config import TokenizerWrapper, TokenizerConfig


class TestTokenizerConfig:
    """Tests for TokenizerConfig dataclass."""

    def test_default_config(self):
        config = TokenizerConfig()
        assert config.backend == "tiktoken"
        assert config.name == "gpt2"
        assert config.trust_remote_code == False

    def test_custom_config(self):
        config = TokenizerConfig(backend="huggingface", name="meta-llama/Llama-2-7b-hf")
        assert config.backend == "huggingface"
        assert config.name == "meta-llama/Llama-2-7b-hf"


class TestTokenizerWrapperTiktoken:
    """Tests for TokenizerWrapper with tiktoken backend."""

    @pytest.fixture
    def tokenizer(self):
        config = TokenizerConfig(backend="tiktoken", name="gpt2")
        return TokenizerWrapper(config)

    def test_vocab_size(self, tokenizer):
        assert tokenizer.vocab_size == 50257

    def test_eos_token_id(self, tokenizer):
        assert tokenizer.eos_token_id == 50256

    def test_padded_vocab_size(self, tokenizer):
        assert tokenizer.padded_vocab_size == 50304

    def test_encode_decode_roundtrip(self, tokenizer):
        text = "Hello, world! This is a test."
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_encode_returns_list(self, tokenizer):
        tokens = tokenizer.encode("Hello")
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)

    def test_encode_batch(self, tokenizer):
        texts = ["Hello!", "World!", "Test 123"]
        batch = tokenizer.encode_batch(texts)
        assert len(batch) == 3
        assert all(isinstance(tokens, list) for tokens in batch)

    def test_cl100k_base(self):
        config = TokenizerConfig(backend="tiktoken", name="cl100k_base")
        tok = TokenizerWrapper(config)
        assert tok.vocab_size == 100277

    def test_repr(self, tokenizer):
        r = repr(tokenizer)
        assert "tiktoken" in r
        assert "gpt2" in r
        assert "50257" in r


class TestTokenizerWrapperHuggingFace:
    """Tests for TokenizerWrapper with HuggingFace backend."""

    @pytest.fixture
    def tokenizer(self):
        config = TokenizerConfig(backend="huggingface", name="gpt2")
        return TokenizerWrapper(config)

    def test_vocab_size(self, tokenizer):
        assert tokenizer.vocab_size == 50257

    def test_eos_token_id(self, tokenizer):
        assert tokenizer.eos_token_id == 50256

    def test_encode_decode_roundtrip(self, tokenizer):
        text = "Hello, world! This is a test."
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_encode_batch(self, tokenizer):
        texts = ["Hello!", "World!"]
        batch = tokenizer.encode_batch(texts)
        assert len(batch) == 2

    def test_cross_backend_token_match(self, tokenizer):
        """Verify HF and tiktoken produce same tokens for GPT-2."""
        tik_config = TokenizerConfig(backend="tiktoken", name="gpt2")
        tik_tok = TokenizerWrapper(tik_config)

        test_text = "The quick brown fox jumps over the lazy dog."
        hf_tokens = tokenizer.encode(test_text)
        tik_tokens = tik_tok.encode(test_text)

        assert hf_tokens == tik_tokens, f"Token mismatch: HF={hf_tokens}, tiktoken={tik_tokens}"


class TestTokenizerWrapperErrors:
    """Tests for error handling."""

    def test_invalid_backend(self):
        config = TokenizerConfig(backend="invalid", name="gpt2")
        with pytest.raises(ValueError, match="Unknown tokenizer backend"):
            TokenizerWrapper(config)

    def test_invalid_tiktoken_encoding(self):
        config = TokenizerConfig(backend="tiktoken", name="nonexistent_encoding")
        with pytest.raises(ValueError, match="Unknown"):
            TokenizerWrapper(config)
