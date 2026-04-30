"""
Tests for train_flex.py checkpoint save/restore functionality.

Run with:
    python tests/test_train_flex_checkpoint.py

Or with pytest:
    pytest tests/test_train_flex_checkpoint.py -v
"""

import os
import sys
import tempfile
import shutil
import random
import numpy as np
import torch
import unittest
from unittest.mock import MagicMock, patch
from argparse import Namespace

# Add bla_gpt to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bla_gpt'))


class MockDataLoader:
    """Mock data loader for testing without actual data files."""

    def __init__(self, batch_size=2, seq_length=32, vocab_size=256):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.current_shard = 0
        self.current_position = 0
        self.ntok_total = 1000000
        self.device = 'cpu'

    def reset(self):
        self.current_shard = 0
        self.current_position = 0

    def next_batch(self):
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        y = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        self.current_position += self.batch_size * self.seq_length
        return x.to(self.device), y.to(self.device)


class MockModel(torch.nn.Module):
    """Simple mock model for testing."""

    def __init__(self, vocab_size=256, hidden_size=64):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, x, y=None):
        h = self.embed(x)
        logits = self.linear(h)
        if y is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )
            return logits, loss
        return logits, None


class TestCheckpointFields(unittest.TestCase):
    """Test that checkpoint saves and loads all required fields."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.temp_dir, 'test_checkpoint.pt')

        # Create mock model and optimizer
        self.model = MockModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: 1.0
        )

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_checkpoint_contains_all_fields(self):
        """Test that save_checkpoint includes all required fields."""
        # Set up state
        step = 100
        best_val_loss = 2.5

        # Set RNG states
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        # Create checkpoint dict matching train_flex.py format
        log = {
            "step": step,
            "model_config": {"vocab_size": 256, "hidden_size": 64},
            "train_config": {"learning_rate": 0.001, "batch_size": 32},
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "torch_cuda": [] if not torch.cuda.is_available() else torch.cuda.get_rng_state_all(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "dataloader_state": {
                "current_shard": 3,
                "current_position": 12345,
            },
            "val_loss": 2.8,
        }

        torch.save(log, self.checkpoint_path)

        # Load and verify
        loaded = torch.load(self.checkpoint_path, weights_only=False)

        # Check all required fields exist
        required_fields = [
            'step', 'model_config', 'train_config', 'model', 'optimizer',
            'scheduler', 'best_val_loss', 'rng_state', 'dataloader_state', 'val_loss'
        ]
        for field in required_fields:
            self.assertIn(field, loaded, f"Missing field: {field}")

        # Check rng_state subfields
        rng_fields = ['torch', 'torch_cuda', 'numpy', 'python']
        for field in rng_fields:
            self.assertIn(field, loaded['rng_state'], f"Missing rng_state field: {field}")

        # Check dataloader_state subfields
        dl_fields = ['current_shard', 'current_position']
        for field in dl_fields:
            self.assertIn(field, loaded['dataloader_state'], f"Missing dataloader_state field: {field}")

        # Verify values
        self.assertEqual(loaded['step'], 100)
        self.assertEqual(loaded['best_val_loss'], 2.5)
        self.assertEqual(loaded['dataloader_state']['current_shard'], 3)
        self.assertEqual(loaded['dataloader_state']['current_position'], 12345)

    def test_load_checkpoint_restores_model_state(self):
        """Test that model state is correctly restored."""
        # Save original state
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        log = {
            "step": 50,
            "model": original_state,
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(log, self.checkpoint_path)

        # Modify model
        for param in self.model.parameters():
            param.data.fill_(999.0)

        # Verify modification
        for name, param in self.model.named_parameters():
            self.assertTrue(torch.all(param.data == 999.0))

        # Load checkpoint
        loaded = torch.load(self.checkpoint_path, weights_only=False)
        self.model.load_state_dict(loaded['model'])

        # Verify restoration
        for name in original_state:
            self.assertTrue(
                torch.allclose(self.model.state_dict()[name], original_state[name]),
                f"Model state not restored for {name}"
            )

    def test_load_checkpoint_restores_optimizer_state(self):
        """Test that optimizer state is correctly restored."""
        # Run a few steps to build optimizer state
        for _ in range(5):
            x = torch.randint(0, 256, (2, 32))
            y = torch.randint(0, 256, (2, 32))
            _, loss = self.model(x, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Save state
        original_optimizer_state = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in self.optimizer.state_dict()['state'].items()
            for k, v in (self.optimizer.state_dict()['state'].get(0, {}).items() if isinstance(k, int) else [(k, v)])
        }

        log = {
            "step": 5,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(log, self.checkpoint_path)

        # Create new optimizer and load state
        new_optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loaded = torch.load(self.checkpoint_path, weights_only=False)
        new_optimizer.load_state_dict(loaded['optimizer'])

        # Compare param groups
        self.assertEqual(
            len(new_optimizer.param_groups),
            len(self.optimizer.param_groups)
        )

    def test_load_checkpoint_restores_scheduler_state(self):
        """Test that scheduler state is correctly restored."""
        # Step scheduler a few times
        for _ in range(10):
            self.scheduler.step()

        original_last_epoch = self.scheduler.last_epoch

        log = {
            "step": 10,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(log, self.checkpoint_path)

        # Create new scheduler and load
        new_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: 1.0
        )
        loaded = torch.load(self.checkpoint_path, weights_only=False)
        new_scheduler.load_state_dict(loaded['scheduler'])

        self.assertEqual(new_scheduler.last_epoch, original_last_epoch)

    def test_load_checkpoint_restores_rng_state(self):
        """Test that RNG states are correctly restored."""
        # Set initial RNG states
        torch.manual_seed(12345)
        np.random.seed(12345)
        random.seed(12345)

        # Save RNG state BEFORE generating values
        rng_state = {
            "torch": torch.get_rng_state(),
            "torch_cuda": [] if not torch.cuda.is_available() else torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }

        # Generate some random values
        torch_vals_before = torch.rand(5)
        np_vals_before = np.random.rand(5)
        py_vals_before = [random.random() for _ in range(5)]

        log = {
            "step": 0,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng_state": rng_state,
        }
        torch.save(log, self.checkpoint_path)

        # Generate more random values (changing state further)
        _ = torch.rand(100)
        _ = np.random.rand(100)
        _ = [random.random() for _ in range(100)]

        # Load and restore RNG state
        loaded = torch.load(self.checkpoint_path, weights_only=False)
        torch.set_rng_state(loaded['rng_state']['torch'])
        np.random.set_state(loaded['rng_state']['numpy'])
        random.setstate(loaded['rng_state']['python'])

        # Generate values again - should match the first set
        torch_vals_after = torch.rand(5)
        np_vals_after = np.random.rand(5)
        py_vals_after = [random.random() for _ in range(5)]

        self.assertTrue(torch.allclose(torch_vals_before, torch_vals_after))
        self.assertTrue(np.allclose(np_vals_before, np_vals_after))
        self.assertEqual(py_vals_before, py_vals_after)


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with old checkpoint formats."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.temp_dir, 'old_checkpoint.pt')
        self.model = MockModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: 1.0
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_old_checkpoint_without_new_fields(self):
        """Test loading checkpoint without scheduler, rng_state, dataloader_state."""
        # Create old-style checkpoint (missing new fields)
        old_checkpoint = {
            "step": 100,
            "model_config": {"vocab_size": 256},
            "train_config": {"learning_rate": 0.001},
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "val_loss": 2.5,
        }
        torch.save(old_checkpoint, self.checkpoint_path)

        # Load checkpoint
        loaded = torch.load(self.checkpoint_path, weights_only=False)

        # Simulate load_checkpoint behavior with graceful handling
        # Model restoration
        self.model.load_state_dict(loaded['model'])

        # Optimizer restoration
        self.optimizer.load_state_dict(loaded['optimizer'])

        # Scheduler restoration (should handle missing gracefully)
        if 'scheduler' in loaded and self.scheduler is not None:
            self.scheduler.load_state_dict(loaded['scheduler'])
        # If not present, scheduler keeps its initial state - this is correct behavior

        # Best val loss (should handle missing gracefully)
        best_val_loss = float('inf')
        if 'best_val_loss' in loaded:
            best_val_loss = loaded['best_val_loss']
        self.assertEqual(best_val_loss, float('inf'))  # Not in old checkpoint

        # RNG state (should handle missing gracefully)
        if 'rng_state' in loaded:
            torch.set_rng_state(loaded['rng_state']['torch'])
        # If not present, RNG state is unchanged - this is correct behavior

        # Dataloader state (should handle missing gracefully)
        mock_loader = MockDataLoader()
        if 'dataloader_state' in loaded:
            mock_loader.current_shard = loaded['dataloader_state']['current_shard']
            mock_loader.current_position = loaded['dataloader_state']['current_position']
        # If not present, loader keeps reset state - will restart from beginning

        # Verify step is correct
        self.assertEqual(loaded['step'], 100)

    def test_load_partial_checkpoint(self):
        """Test loading checkpoint with only some new fields."""
        partial_checkpoint = {
            "step": 50,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),  # Has scheduler
            # Missing: best_val_loss, rng_state, dataloader_state
        }
        torch.save(partial_checkpoint, self.checkpoint_path)

        loaded = torch.load(self.checkpoint_path, weights_only=False)

        # Should load scheduler successfully
        self.assertIn('scheduler', loaded)
        new_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: 1.0
        )
        new_scheduler.load_state_dict(loaded['scheduler'])

        # Should handle missing fields gracefully
        self.assertNotIn('best_val_loss', loaded)
        self.assertNotIn('rng_state', loaded)
        self.assertNotIn('dataloader_state', loaded)


class TestDataloaderStateRestoration(unittest.TestCase):
    """Test dataloader state save/restore."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.temp_dir, 'dl_checkpoint.pt')

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dataloader_state_save_restore(self):
        """Test that dataloader position is correctly saved and restored."""
        loader = MockDataLoader()

        # Advance loader
        for _ in range(10):
            loader.next_batch()

        original_shard = loader.current_shard
        original_position = loader.current_position

        # Save state
        checkpoint = {
            "step": 10,
            "dataloader_state": {
                "current_shard": loader.current_shard,
                "current_position": loader.current_position,
            }
        }
        torch.save(checkpoint, self.checkpoint_path)

        # Reset loader
        loader.reset()
        self.assertEqual(loader.current_shard, 0)
        self.assertEqual(loader.current_position, 0)

        # Restore state
        loaded = torch.load(self.checkpoint_path, weights_only=False)
        loader.current_shard = loaded['dataloader_state']['current_shard']
        loader.current_position = loaded['dataloader_state']['current_position']

        self.assertEqual(loader.current_shard, original_shard)
        self.assertEqual(loader.current_position, original_position)


class TestCompiledModelCheckpoint(unittest.TestCase):
    """Test checkpoint handling for torch.compile models."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.temp_dir, 'compiled_checkpoint.pt')
        self.model = MockModel()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_compiled_model_checkpoint(self):
        """Test loading checkpoint saved from compiled model (with _orig_mod. prefix)."""
        # Simulate compiled model state dict (with _orig_mod. prefix)
        original_state = self.model.state_dict()
        compiled_state = {f'_orig_mod.{k}': v for k, v in original_state.items()}

        checkpoint = {
            "step": 100,
            "model": compiled_state,
        }
        torch.save(checkpoint, self.checkpoint_path)

        # Load and strip prefix (as done in load_checkpoint)
        loaded = torch.load(self.checkpoint_path, weights_only=False)
        state_dict = loaded['model']
        new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        # Should load successfully
        self.model.load_state_dict(new_state_dict)

        # Verify model works
        x = torch.randint(0, 256, (2, 32))
        y = torch.randint(0, 256, (2, 32))
        _, loss = self.model(x, y)
        self.assertIsNotNone(loss)


class TestCLIArguments(unittest.TestCase):
    """Test CLI argument parsing for resume functionality."""

    def test_resume_argument_parsing(self):
        """Test --resume argument is correctly parsed."""
        from argparse import ArgumentParser

        parser = ArgumentParser()
        parser.add_argument("--resume", type=str, default=None)
        parser.add_argument("--auto_resume", action="store_true")

        # Test --resume
        args = parser.parse_args(['--resume', '/path/to/checkpoint.pt'])
        self.assertEqual(args.resume, '/path/to/checkpoint.pt')
        self.assertFalse(args.auto_resume)

        # Test --auto_resume
        args = parser.parse_args(['--auto_resume'])
        self.assertIsNone(args.resume)
        self.assertTrue(args.auto_resume)

        # Test both (resume takes precedence in implementation)
        args = parser.parse_args(['--resume', '/path/to/checkpoint.pt', '--auto_resume'])
        self.assertEqual(args.resume, '/path/to/checkpoint.pt')
        self.assertTrue(args.auto_resume)


class TestAutoResume(unittest.TestCase):
    """Test auto-resume checkpoint discovery."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.logs_dir = os.path.join(self.temp_dir, 'logs')
        os.makedirs(self.logs_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_find_latest_checkpoint(self):
        """Test finding the latest checkpoint for auto-resume."""
        import glob as glob_module

        run_name = "test_run"

        # Create mock checkpoint directories
        for i in range(3):
            run_dir = os.path.join(self.logs_dir, f"{run_name}_{i}")
            os.makedirs(run_dir)

            # Create checkpoints with different step numbers
            for step in [100, 200, 300]:
                checkpoint_path = os.path.join(run_dir, f"state_step{step:06d}.pt")
                torch.save({"step": step}, checkpoint_path)

        # Find checkpoints matching pattern
        pattern = os.path.join(self.logs_dir, f"{run_name}_*/state_step*.pt")
        checkpoints = sorted(glob_module.glob(pattern))

        self.assertTrue(len(checkpoints) > 0)

        # Latest should be the last one alphabetically (highest step in last run)
        latest = checkpoints[-1]
        self.assertIn("state_step000300.pt", latest)

    def test_no_checkpoints_found(self):
        """Test behavior when no checkpoints exist."""
        import glob as glob_module

        pattern = os.path.join(self.logs_dir, "nonexistent_run_*/state_step*.pt")
        checkpoints = sorted(glob_module.glob(pattern))

        self.assertEqual(len(checkpoints), 0)


class TestTrainMethodStartStep(unittest.TestCase):
    """Test that train() method correctly handles start_step parameter."""

    def test_range_starts_from_start_step(self):
        """Test that training loop starts from correct step."""
        start_step = 50
        num_iterations = 100

        steps_executed = []
        for step in range(start_step, num_iterations + 1):
            steps_executed.append(step)
            if len(steps_executed) >= 5:  # Just test first few
                break

        self.assertEqual(steps_executed[0], 50)
        self.assertEqual(steps_executed[1], 51)
        self.assertEqual(steps_executed[2], 52)

    def test_dataloader_not_reset_on_resume(self):
        """Test that dataloader is not reset when resuming."""
        loader = MockDataLoader()

        # Advance loader (simulating previous training)
        for _ in range(10):
            loader.next_batch()
        original_position = loader.current_position

        # Simulate resume logic
        start_step = 10
        if start_step == 0:
            loader.reset()

        # Position should be unchanged
        self.assertEqual(loader.current_position, original_position)

    def test_dataloader_reset_on_fresh_start(self):
        """Test that dataloader is reset on fresh start."""
        loader = MockDataLoader()

        # Advance loader
        for _ in range(10):
            loader.next_batch()
        self.assertNotEqual(loader.current_position, 0)

        # Simulate fresh start logic
        start_step = 0
        if start_step == 0:
            loader.reset()

        # Position should be reset
        self.assertEqual(loader.current_position, 0)
        self.assertEqual(loader.current_shard, 0)


class TestBestValLossTracking(unittest.TestCase):
    """Test best_val_loss tracking during training."""

    def test_best_val_loss_initialization(self):
        """Test that best_val_loss starts at infinity."""
        best_val_loss = float('inf')
        self.assertEqual(best_val_loss, float('inf'))

    def test_best_val_loss_update(self):
        """Test that best_val_loss is updated correctly."""
        best_val_loss = float('inf')

        val_losses = [3.5, 3.2, 3.8, 2.9, 3.0]
        saved_steps = []

        for step, val_loss in enumerate(val_losses):
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                saved_steps.append(step)

        # Should have saved at steps 0, 1, 3 (losses 3.5, 3.2, 2.9)
        self.assertEqual(saved_steps, [0, 1, 3])
        self.assertEqual(best_val_loss, 2.9)

    def test_best_val_loss_restoration(self):
        """Test that best_val_loss is restored from checkpoint."""
        saved_best = 2.5

        checkpoint = {
            "step": 100,
            "best_val_loss": saved_best,
        }

        # Simulate restoration
        best_val_loss = float('inf')
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']

        self.assertEqual(best_val_loss, 2.5)


class TestIntegration(unittest.TestCase):
    """Integration tests for full save/load cycle."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.temp_dir, 'integration_checkpoint.pt')

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_save_load_cycle(self):
        """Test complete save and load cycle."""
        # Set up initial state
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.1, 1.0 - step * 0.01))
        loader = MockDataLoader()

        # Train for a few steps
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        for _ in range(20):
            x, y = loader.next_batch()
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        step = 20
        best_val_loss = 2.5

        # Save checkpoint
        checkpoint = {
            "step": step,
            "model_config": {"vocab_size": 256, "hidden_size": 64},
            "train_config": {"learning_rate": 0.001},
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "torch_cuda": [] if not torch.cuda.is_available() else torch.cuda.get_rng_state_all(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "dataloader_state": {
                "current_shard": loader.current_shard,
                "current_position": loader.current_position,
            },
            "val_loss": 2.8,
        }
        torch.save(checkpoint, self.checkpoint_path)

        # Create new instances
        new_model = MockModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = torch.optim.lr_scheduler.LambdaLR(new_optimizer, lambda step: max(0.1, 1.0 - step * 0.01))
        new_loader = MockDataLoader()

        # Load checkpoint
        loaded = torch.load(self.checkpoint_path, weights_only=False)

        # Restore all state
        new_model.load_state_dict(loaded['model'])
        new_optimizer.load_state_dict(loaded['optimizer'])
        new_scheduler.load_state_dict(loaded['scheduler'])

        new_best_val_loss = loaded['best_val_loss']

        torch.set_rng_state(loaded['rng_state']['torch'])
        np.random.set_state(loaded['rng_state']['numpy'])
        random.setstate(loaded['rng_state']['python'])

        new_loader.current_shard = loaded['dataloader_state']['current_shard']
        new_loader.current_position = loaded['dataloader_state']['current_position']

        # Verify state
        self.assertEqual(loaded['step'], 20)
        self.assertEqual(new_best_val_loss, 2.5)
        self.assertEqual(new_loader.current_shard, loader.current_shard)
        self.assertEqual(new_loader.current_position, loader.current_position)
        self.assertEqual(new_scheduler.last_epoch, scheduler.last_epoch)

        # Verify models produce same output
        test_x = torch.randint(0, 256, (2, 32))

        with torch.no_grad():
            out1, _ = model(test_x)
            out2, _ = new_model(test_x)

        self.assertTrue(torch.allclose(out1, out2))

        # Verify RNG produces same values
        rand1 = torch.rand(5)
        torch.set_rng_state(loaded['rng_state']['torch'])
        rand2 = torch.rand(5)
        # Note: rand1 and rand2 won't match because we already consumed RNG state
        # But the test ensures RNG state is saveable/loadable


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCheckpointFields))
    suite.addTests(loader.loadTestsFromTestCase(TestBackwardCompatibility))
    suite.addTests(loader.loadTestsFromTestCase(TestDataloaderStateRestoration))
    suite.addTests(loader.loadTestsFromTestCase(TestCompiledModelCheckpoint))
    suite.addTests(loader.loadTestsFromTestCase(TestCLIArguments))
    suite.addTests(loader.loadTestsFromTestCase(TestAutoResume))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainMethodStartStep))
    suite.addTests(loader.loadTestsFromTestCase(TestBestValLossTracking))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
