"""Tests for train_new.py HuggingFace streaming integration."""

import sys
import os

# Add bla_gpt to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bla_gpt'))

import pytest


class TestHyperparameters:
    """Tests for Hyperparameters dataclass HF streaming fields."""

    @pytest.fixture
    def hyperparams(self):
        from train_new import Hyperparameters
        return Hyperparameters()

    def test_use_hf_streaming_default(self, hyperparams):
        assert hyperparams.use_hf_streaming == False

    def test_hf_dataset_default(self, hyperparams):
        assert hyperparams.hf_dataset == "HuggingFaceFW/fineweb"

    def test_hf_dataset_config_default(self, hyperparams):
        assert hyperparams.hf_dataset_config == "sample-10BT"

    def test_hf_text_column_default(self, hyperparams):
        assert hyperparams.hf_text_column == "text"

    def test_hf_val_dataset_default(self, hyperparams):
        assert hyperparams.hf_val_dataset is None

    def test_hf_val_dataset_config_default(self, hyperparams):
        assert hyperparams.hf_val_dataset_config is None

    def test_hf_shuffle_buffer_default(self, hyperparams):
        assert hyperparams.hf_shuffle_buffer == 10000

    def test_tokenizer_backend_default(self, hyperparams):
        assert hyperparams.tokenizer_backend == "tiktoken"

    def test_tokenizer_name_default(self, hyperparams):
        assert hyperparams.tokenizer_name == "gpt2"

    def test_all_hf_fields_present(self, hyperparams):
        """Verify all expected HF-related fields exist."""
        expected_fields = [
            'use_hf_streaming',
            'hf_dataset',
            'hf_dataset_config',
            'hf_text_column',
            'hf_val_dataset',
            'hf_val_dataset_config',
            'hf_shuffle_buffer',
            'tokenizer_backend',
            'tokenizer_name',
        ]
        for field in expected_fields:
            assert hasattr(hyperparams, field), f"Missing field: {field}"


class TestImports:
    """Tests for module imports."""

    def test_import_hyperparameters(self):
        from train_new import Hyperparameters
        assert Hyperparameters is not None

    def test_import_trainer(self):
        from train_new import Trainer
        assert Trainer is not None

    def test_import_tee_logger(self):
        from train_new import TeeLogger
        assert TeeLogger is not None

    def test_import_data_loaders(self):
        from data_loaders import (
            DataShard,
            ByteDataShard,
            DistributedDataLoader,
            HFStreamingDataLoader,
            HFDatasetConfig,
        )
        assert DataShard is not None
        assert ByteDataShard is not None
        assert DistributedDataLoader is not None
        assert HFStreamingDataLoader is not None
        assert HFDatasetConfig is not None

    def test_import_tokenizers_config(self):
        from tokenizers_config import TokenizerConfig, TokenizerWrapper
        assert TokenizerConfig is not None
        assert TokenizerWrapper is not None
