"""Tests for data_loaders.py"""

import sys
import os
import tempfile

# Add bla_gpt to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bla_gpt'))

import numpy as np
import pytest
import torch
from data_loaders import (
    DataShard,
    ByteDataShard,
    HFDatasetConfig,
    HFStreamingDataLoader,
    generate_cache_key,
)
from tokenizers_config import TokenizerWrapper


class TestDataShard:
    """Tests for DataShard class."""

    def test_magic_number(self):
        assert DataShard.MAGIC_NUMBER == 20240520

    def test_version(self):
        assert DataShard.VERSION == 1

    def test_header_size(self):
        assert DataShard.HEADER_SIZE == 256 * 4


class TestByteDataShard:
    """Tests for ByteDataShard class."""

    def test_magic_number(self):
        assert ByteDataShard.MAGIC_NUMBER == 20240520

    def test_version(self):
        assert ByteDataShard.VERSION == 1


class TestHFDatasetConfig:
    """Tests for HFDatasetConfig dataclass."""

    def test_default_config(self):
        config = HFDatasetConfig()
        assert config.dataset_name == "HuggingFaceFW/fineweb"
        assert config.dataset_config == "sample-10BT"
        assert config.split == "train"
        assert config.text_column == "text"
        assert config.streaming == True
        assert config.shuffle_buffer_size == 10000
        assert config.seed == 42

    def test_is_coqpit_subclass(self):
        """Verify HFDatasetConfig inherits from Coqpit."""
        from coqpit import Coqpit
        config = HFDatasetConfig()
        assert isinstance(config, Coqpit)
        # Verify Coqpit methods are available
        assert hasattr(config, 'to_dict')
        assert hasattr(config, 'from_dict')

    def test_custom_config(self):
        config = HFDatasetConfig(
            dataset_name="allenai/c4",
            dataset_config="en",
            text_column="content",
            shuffle_buffer_size=5000,
        )
        assert config.dataset_name == "allenai/c4"
        assert config.dataset_config == "en"
        assert config.text_column == "content"
        assert config.shuffle_buffer_size == 5000


class TestHFStreamingDataLoaderMock:
    """Tests for HFStreamingDataLoader with mock data."""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock iterable dataset."""
        class MockDS:
            def __iter__(self):
                for i in range(100):
                    yield {"text": f"Test document {i}. " * 20}

            def shard(self, num_shards, index):
                return self

            def shuffle(self, seed, buffer_size):
                return self

        return MockDS()

    @pytest.fixture
    def tokenizer(self):
        return TokenizerWrapper()

    @pytest.fixture
    def loader_with_mock(self, mock_dataset, tokenizer, monkeypatch):
        """Create loader with mocked dataset loading."""
        def mock_load_dataset(*args, **kwargs):
            return mock_dataset

        import datasets
        monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

        config = HFDatasetConfig(shuffle_buffer_size=10)
        return HFStreamingDataLoader(config, tokenizer, batch_size=2, seq_length=32,
                                     process_rank=0, num_processes=1)

    def test_batch_shape(self, loader_with_mock):
        x, y = loader_with_mock.next_batch()
        assert x.shape == (2, 32)
        assert y.shape == (2, 32)

    def test_batch_device(self, loader_with_mock):
        x, y = loader_with_mock.next_batch()
        assert x.device.type == "cuda"
        assert y.device.type == "cuda"

    def test_batch_dtype(self, loader_with_mock):
        x, y = loader_with_mock.next_batch()
        assert x.dtype == torch.long
        assert y.dtype == torch.long

    def test_y_is_shifted_x(self, loader_with_mock):
        x, y = loader_with_mock.next_batch()
        # y should be x shifted by 1 position
        assert (y[:, :-1] == x[:, 1:]).all()

    def test_tokens_in_valid_range(self, loader_with_mock, tokenizer):
        x, y = loader_with_mock.next_batch()
        assert x.max().item() < tokenizer.vocab_size
        assert y.max().item() < tokenizer.vocab_size

    def test_multiple_batches(self, loader_with_mock):
        x1, y1 = loader_with_mock.next_batch()
        x2, y2 = loader_with_mock.next_batch()
        assert x1.shape == x2.shape
        # Batches should generally be different (not guaranteed but highly likely)

    def test_reset(self, loader_with_mock):
        x1, _ = loader_with_mock.next_batch()
        loader_with_mock.reset()
        x2, _ = loader_with_mock.next_batch()
        assert x1.shape == x2.shape


@pytest.mark.slow
class TestHFStreamingDataLoaderReal:
    """Tests for HFStreamingDataLoader with real HuggingFace dataset.

    These tests require network access and are marked as slow.
    Run with: pytest -m slow
    """

    @pytest.fixture
    def tokenizer(self):
        return TokenizerWrapper()

    @pytest.fixture
    def loader(self, tokenizer):
        config = HFDatasetConfig(
            dataset_name="HuggingFaceFW/fineweb",
            dataset_config="sample-10BT",
            shuffle_buffer_size=100,
        )
        return HFStreamingDataLoader(config, tokenizer, batch_size=2, seq_length=64,
                                     process_rank=0, num_processes=1)

    def test_real_dataset_batch_shape(self, loader):
        x, y = loader.next_batch()
        assert x.shape == (2, 64)
        assert y.shape == (2, 64)

    def test_real_dataset_on_cuda(self, loader):
        x, y = loader.next_batch()
        assert x.device.type == "cuda"

    def test_real_dataset_valid_tokens(self, loader, tokenizer):
        x, y = loader.next_batch()
        assert x.max().item() < tokenizer.vocab_size

    def test_real_dataset_decodable(self, loader, tokenizer):
        x, _ = loader.next_batch()
        sample = x[0, :10].tolist()
        decoded = tokenizer.decode(sample)
        assert isinstance(decoded, str)
        assert len(decoded) > 0


class TestGenerateCacheKey:
    """Tests for generate_cache_key function."""

    def test_deterministic(self):
        """Same config produces same key."""
        config = HFDatasetConfig(
            dataset_name="test/dataset",
            dataset_config="config1",
            split="train",
            take_samples=1000,
        )
        key1 = generate_cache_key(config, "tiktoken", "gpt2")
        key2 = generate_cache_key(config, "tiktoken", "gpt2")
        assert key1 == key2

    def test_different_dataset_produces_different_key(self):
        """Different dataset names produce different keys."""
        config1 = HFDatasetConfig(dataset_name="test/dataset1")
        config2 = HFDatasetConfig(dataset_name="test/dataset2")
        key1 = generate_cache_key(config1, "tiktoken", "gpt2")
        key2 = generate_cache_key(config2, "tiktoken", "gpt2")
        assert key1 != key2

    def test_different_config_produces_different_key(self):
        """Different dataset configs produce different keys."""
        config1 = HFDatasetConfig(dataset_config="config1")
        config2 = HFDatasetConfig(dataset_config="config2")
        key1 = generate_cache_key(config1, "tiktoken", "gpt2")
        key2 = generate_cache_key(config2, "tiktoken", "gpt2")
        assert key1 != key2

    def test_different_samples_produces_different_key(self):
        """Different take_samples produces different keys."""
        config1 = HFDatasetConfig(take_samples=500)
        config2 = HFDatasetConfig(take_samples=1000)
        key1 = generate_cache_key(config1, "tiktoken", "gpt2")
        key2 = generate_cache_key(config2, "tiktoken", "gpt2")
        assert key1 != key2

    def test_different_tokenizer_produces_different_key(self):
        """Different tokenizer produces different keys."""
        config = HFDatasetConfig()
        key1 = generate_cache_key(config, "tiktoken", "gpt2")
        key2 = generate_cache_key(config, "huggingface", "gpt2")
        assert key1 != key2

    def test_different_tokenizer_name_produces_different_key(self):
        """Different tokenizer name produces different keys."""
        config = HFDatasetConfig()
        key1 = generate_cache_key(config, "tiktoken", "gpt2")
        key2 = generate_cache_key(config, "tiktoken", "cl100k_base")
        assert key1 != key2

    def test_key_length(self):
        """Key is 16 hex characters."""
        config = HFDatasetConfig()
        key = generate_cache_key(config, "tiktoken", "gpt2")
        assert len(key) == 16
        assert all(c in "0123456789abcdef" for c in key)

    def test_none_config_handled(self):
        """None dataset_config is handled (uses 'default')."""
        config = HFDatasetConfig(dataset_config=None)
        key = generate_cache_key(config, "tiktoken", "gpt2")
        assert len(key) == 16

    def test_none_take_samples_handled(self):
        """None take_samples is handled (uses 'all')."""
        config = HFDatasetConfig(take_samples=None)
        key = generate_cache_key(config, "tiktoken", "gpt2")
        assert len(key) == 16


class TestCacheWriteLoad:
    """Tests for cache write/load functionality."""

    def test_datashard_format_roundtrip(self):
        """Test writing and loading tokens in DataShard format."""
        # Create test tokens
        tokens = np.array([100, 200, 300, 400, 500], dtype=np.uint16)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            cache_path = f.name

        try:
            # Write in DataShard format
            header = np.zeros(256, dtype=np.int32)
            header[0] = DataShard.MAGIC_NUMBER
            header[1] = DataShard.VERSION
            header[2] = len(tokens)

            with open(cache_path, "wb") as f:
                f.write(header.tobytes())
                f.write(tokens.tobytes())

            # Load back using DataShard
            loaded_tokens = DataShard.load_data_shard(cache_path)
            assert len(loaded_tokens) == len(tokens)
            assert np.array_equal(loaded_tokens, tokens)
        finally:
            os.unlink(cache_path)

    def test_large_token_roundtrip(self):
        """Test with a larger number of tokens."""
        tokens = np.arange(10000, dtype=np.uint16)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            cache_path = f.name

        try:
            header = np.zeros(256, dtype=np.int32)
            header[0] = DataShard.MAGIC_NUMBER
            header[1] = DataShard.VERSION
            header[2] = len(tokens)

            with open(cache_path, "wb") as f:
                f.write(header.tobytes())
                f.write(tokens.tobytes())

            loaded_tokens = DataShard.load_data_shard(cache_path)
            assert len(loaded_tokens) == 10000
            assert np.array_equal(loaded_tokens, tokens)
        finally:
            os.unlink(cache_path)


class TestHFDatasetConfigCacheFields:
    """Tests for HFDatasetConfig cache-related fields."""

    def test_cache_fields_default(self):
        """Test default values for cache fields."""
        config = HFDatasetConfig()
        assert config.cache_tokens == False
        assert config.cache_dir is None

    def test_cache_fields_custom(self):
        """Test custom values for cache fields."""
        config = HFDatasetConfig(
            cache_tokens=True,
            cache_dir="/tmp/my_cache",
        )
        assert config.cache_tokens == True
        assert config.cache_dir == "/tmp/my_cache"
