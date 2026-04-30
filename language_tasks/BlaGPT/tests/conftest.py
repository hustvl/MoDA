"""Pytest configuration and fixtures for BlaGPT tests."""

import sys
import os

# Add bla_gpt to Python path for all tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bla_gpt'))

import pytest


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (require network access)"
    )


@pytest.fixture(scope="session")
def tiktoken_tokenizer():
    """Session-scoped tiktoken tokenizer fixture."""
    from tokenizers_config import TokenizerWrapper, TokenizerConfig
    return TokenizerWrapper(TokenizerConfig(backend="tiktoken", name="gpt2"))


@pytest.fixture(scope="session")
def hf_tokenizer():
    """Session-scoped HuggingFace tokenizer fixture."""
    from tokenizers_config import TokenizerWrapper, TokenizerConfig
    return TokenizerWrapper(TokenizerConfig(backend="huggingface", name="gpt2"))
