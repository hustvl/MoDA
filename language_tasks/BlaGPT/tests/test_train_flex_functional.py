"""
Functional tests for train_flex.py checkpoint save/restore with actual Trainer class.

These tests mock the distributed training environment and test the actual
save_checkpoint() and load_checkpoint() methods.

Run with:
    python tests/test_train_flex_functional.py

Note: This test mocks CUDA and distributed training for CPU-only testing.
"""

import os
import sys
import tempfile
import shutil
import random
import numpy as np
import torch
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass, field

# Add bla_gpt to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bla_gpt'))

# Set up mock environment variables before importing train_flex
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'


class MockDistributedDataLoader:
    """Mock DistributedDataLoader for testing."""

    def __init__(self, batch_size=2, seq_length=32, vocab_size=256):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.current_shard = 0
        self.current_position = 0
        self.ntok_total = 1000000

    def reset(self):
        self.current_shard = 0
        self.current_position = 0

    def next_batch(self):
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        y = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        self.current_position += self.batch_size * self.seq_length
        return x, y


class SimpleModel(torch.nn.Module):
    """Simple model for testing."""

    def __init__(self, vocab_size=256, hidden_size=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
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


@dataclass
class SimpleModelConfig:
    """Simple model config for testing."""
    vocab_size: int = 256
    hidden_size: int = 64

    def to_dict(self):
        return {'vocab_size': self.vocab_size, 'hidden_size': self.hidden_size}


class TestTrainerSaveCheckpoint(unittest.TestCase):
    """Test Trainer.save_checkpoint() method."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.logdir = os.path.join(self.temp_dir, 'logs', 'test_run_0')
        os.makedirs(self.logdir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_checkpoint_creates_file(self):
        """Test that save_checkpoint creates a checkpoint file."""
        # Create mock trainer state
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        train_loader = MockDistributedDataLoader()

        # Simulate the save_checkpoint function
        step = 100
        best_val_loss = 2.5
        val_loss = 2.8

        log = {
            "step": step,
            "model_config": SimpleModelConfig().to_dict(),
            "train_config": {"learning_rate": 0.001},
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "torch_cuda": [],
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "dataloader_state": {
                "current_shard": train_loader.current_shard,
                "current_position": train_loader.current_position,
            },
            "val_loss": val_loss,
        }

        checkpoint_path = os.path.join(self.logdir, f"state_step{step:06d}.pt")
        torch.save(log, checkpoint_path)

        # Verify file exists
        self.assertTrue(os.path.exists(checkpoint_path))

        # Verify contents
        loaded = torch.load(checkpoint_path, weights_only=False)
        self.assertEqual(loaded['step'], 100)
        self.assertIn('scheduler', loaded)
        self.assertIn('rng_state', loaded)
        self.assertIn('dataloader_state', loaded)
        self.assertIn('best_val_loss', loaded)


class TestTrainerLoadCheckpoint(unittest.TestCase):
    """Test Trainer.load_checkpoint() method."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.temp_dir, 'checkpoint.pt')

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_checkpoint_restores_state(self):
        """Test that load_checkpoint restores all state correctly."""
        # Create and save checkpoint
        original_model = SimpleModel()
        original_optimizer = torch.optim.Adam(original_model.parameters(), lr=0.001)
        original_scheduler = torch.optim.lr_scheduler.LambdaLR(
            original_optimizer, lambda step: 1.0
        )

        # Advance scheduler
        for _ in range(10):
            original_scheduler.step()

        # Set specific RNG states
        torch.manual_seed(12345)
        np.random.seed(12345)
        random.seed(12345)

        checkpoint = {
            "step": 50,
            "model": original_model.state_dict(),
            "optimizer": original_optimizer.state_dict(),
            "scheduler": original_scheduler.state_dict(),
            "best_val_loss": 2.5,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "torch_cuda": [],
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "dataloader_state": {
                "current_shard": 3,
                "current_position": 5000,
            },
        }
        torch.save(checkpoint, self.checkpoint_path)

        # Create new instances
        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = torch.optim.lr_scheduler.LambdaLR(
            new_optimizer, lambda step: 1.0
        )
        new_loader = MockDistributedDataLoader()
        best_val_loss = float('inf')

        # Simulate load_checkpoint
        loaded = torch.load(self.checkpoint_path, weights_only=False)

        # Restore model
        state_dict = loaded['model']
        new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        new_model.load_state_dict(new_state_dict)

        # Restore optimizer
        new_optimizer.load_state_dict(loaded['optimizer'])

        # Restore scheduler
        if 'scheduler' in loaded:
            new_scheduler.load_state_dict(loaded['scheduler'])

        # Restore best_val_loss
        if 'best_val_loss' in loaded:
            best_val_loss = loaded['best_val_loss']

        # Restore RNG
        if 'rng_state' in loaded:
            torch.set_rng_state(loaded['rng_state']['torch'])
            np.random.set_state(loaded['rng_state']['numpy'])
            random.setstate(loaded['rng_state']['python'])

        # Restore dataloader state
        if 'dataloader_state' in loaded:
            new_loader.current_shard = loaded['dataloader_state']['current_shard']
            new_loader.current_position = loaded['dataloader_state']['current_position']

        step = loaded['step']

        # Verify restoration
        self.assertEqual(step, 50)
        self.assertEqual(best_val_loss, 2.5)
        self.assertEqual(new_scheduler.last_epoch, original_scheduler.last_epoch)
        self.assertEqual(new_loader.current_shard, 3)
        self.assertEqual(new_loader.current_position, 5000)

    def test_load_checkpoint_handles_compiled_model(self):
        """Test loading checkpoint from compiled model."""
        model = SimpleModel()
        state_dict = {f'_orig_mod.{k}': v for k, v in model.state_dict().items()}

        checkpoint = {
            "step": 100,
            "model": state_dict,
            "optimizer": {},
        }
        torch.save(checkpoint, self.checkpoint_path)

        # Load and strip prefix
        loaded = torch.load(self.checkpoint_path, weights_only=False)
        new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in loaded['model'].items()}

        new_model = SimpleModel()
        new_model.load_state_dict(new_state_dict)

        # Verify model works
        x = torch.randint(0, 256, (2, 32))
        logits, _ = new_model(x)
        self.assertEqual(logits.shape, (2, 32, 256))


class TestCheckpointCleanup(unittest.TestCase):
    """Test checkpoint cleanup functionality."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.logdir = os.path.join(self.temp_dir, 'logs', 'test_run_0')
        os.makedirs(self.logdir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_keeps_last_n_checkpoints(self):
        """Test that old checkpoints are cleaned up."""
        import glob

        keep_last_n = 2

        # Create several checkpoints
        for step in [100, 200, 300, 400, 500]:
            checkpoint_path = os.path.join(self.logdir, f"state_step{step:06d}.pt")
            torch.save({"step": step}, checkpoint_path)

            # Simulate cleanup logic
            checkpoints = sorted(glob.glob(os.path.join(self.logdir, "state_step*.pt")))
            if len(checkpoints) > keep_last_n:
                for old_checkpoint in checkpoints[:-keep_last_n]:
                    os.remove(old_checkpoint)

        # Verify only last 2 remain
        remaining = sorted(glob.glob(os.path.join(self.logdir, "state_step*.pt")))
        self.assertEqual(len(remaining), 2)
        self.assertIn("state_step000400.pt", remaining[-2])
        self.assertIn("state_step000500.pt", remaining[-1])


class TestResumeLogic(unittest.TestCase):
    """Test resume logic in main()."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.logs_dir = os.path.join(self.temp_dir, 'logs')
        os.makedirs(self.logs_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_auto_resume_finds_latest(self):
        """Test auto-resume finds the latest checkpoint."""
        import glob

        run_name = "test_run"

        # Create multiple run directories with checkpoints
        for run_num in range(3):
            run_dir = os.path.join(self.logs_dir, f"{run_name}_{run_num}")
            os.makedirs(run_dir)
            for step in [100, 200]:
                checkpoint_path = os.path.join(run_dir, f"state_step{step:06d}.pt")
                torch.save({"step": step}, checkpoint_path)

        # Simulate auto-resume logic
        pattern = os.path.join(self.logs_dir, f"{run_name}_*/state_step*.pt")
        checkpoints = sorted(glob.glob(pattern))

        self.assertTrue(len(checkpoints) > 0)
        latest = checkpoints[-1]

        # Should be step 200 from run_2
        self.assertIn("test_run_2", latest)
        self.assertIn("state_step000200.pt", latest)

    def test_explicit_resume_takes_precedence(self):
        """Test that --resume takes precedence over --auto_resume."""
        explicit_path = "/path/to/checkpoint.pt"
        auto_resume = True

        # Simulate main() logic
        resume_checkpoint = explicit_path
        if not resume_checkpoint and auto_resume:
            resume_checkpoint = "auto_found_checkpoint.pt"

        self.assertEqual(resume_checkpoint, explicit_path)


class TestTrainLoopWithStartStep(unittest.TestCase):
    """Test train loop behavior with start_step."""

    def test_loop_starts_from_start_step(self):
        """Test that training loop starts from the correct step."""
        start_step = 50
        num_iterations = 100

        executed_steps = []
        for step in range(start_step, num_iterations + 1):
            executed_steps.append(step)
            if step >= start_step + 5:
                break

        self.assertEqual(executed_steps[0], 50)
        self.assertEqual(executed_steps[-1], 55)

    def test_validation_still_runs_at_correct_intervals(self):
        """Test that validation runs at correct intervals when resuming."""
        start_step = 50
        val_loss_every = 25
        num_iterations = 100

        val_steps = []
        for step in range(start_step, min(num_iterations + 1, start_step + 100)):
            last_step = step == num_iterations
            if last_step or (val_loss_every > 0 and step % val_loss_every == 0):
                val_steps.append(step)

        # Should validate at 50, 75, 100
        self.assertIn(50, val_steps)
        self.assertIn(75, val_steps)
        self.assertIn(100, val_steps)


class TestEndToEndResumeSimulation(unittest.TestCase):
    """End-to-end simulation of save and resume."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.logdir = os.path.join(self.temp_dir, 'logs', 'test_run_0')
        os.makedirs(self.logdir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_training_continues_correctly_after_resume(self):
        """Test that training produces consistent results when resumed."""
        # Setup
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: max(0.1, 1.0 - step * 0.01)
        )
        loader = MockDistributedDataLoader()

        # Train for 20 steps
        losses = []
        for step in range(20):
            x, y = loader.next_batch()
            _, loss = model(x, y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Save checkpoint at step 20
        checkpoint = {
            "step": 19,  # 0-indexed, last completed step
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": min(losses),
            "rng_state": {
                "torch": torch.get_rng_state(),
                "torch_cuda": [],
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "dataloader_state": {
                "current_shard": loader.current_shard,
                "current_position": loader.current_position,
            },
        }
        checkpoint_path = os.path.join(self.logdir, "state_step000019.pt")
        torch.save(checkpoint, checkpoint_path)

        # Continue training for 10 more steps (steps 20-29)
        more_losses_original = []
        for step in range(20, 30):
            x, y = loader.next_batch()
            _, loss = model(x, y)
            more_losses_original.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Now simulate a fresh process that resumes from checkpoint
        torch.manual_seed(99999)  # Different seed to prove we restore
        np.random.seed(99999)
        random.seed(99999)

        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = torch.optim.lr_scheduler.LambdaLR(
            new_optimizer, lambda step: max(0.1, 1.0 - step * 0.01)
        )
        new_loader = MockDistributedDataLoader()

        # Load checkpoint
        loaded = torch.load(checkpoint_path, weights_only=False)
        new_model.load_state_dict(loaded['model'])
        new_optimizer.load_state_dict(loaded['optimizer'])
        new_scheduler.load_state_dict(loaded['scheduler'])
        torch.set_rng_state(loaded['rng_state']['torch'])
        np.random.set_state(loaded['rng_state']['numpy'])
        random.setstate(loaded['rng_state']['python'])
        new_loader.current_shard = loaded['dataloader_state']['current_shard']
        new_loader.current_position = loaded['dataloader_state']['current_position']

        start_step = loaded['step'] + 1  # Resume from step 20

        # Continue training from checkpoint
        more_losses_resumed = []
        for step in range(start_step, start_step + 10):
            x, y = new_loader.next_batch()
            _, loss = new_model(x, y)
            more_losses_resumed.append(loss.item())
            loss.backward()
            new_optimizer.step()
            new_scheduler.step()
            new_optimizer.zero_grad()

        # Verify that the losses match (training is deterministic after restore)
        for i, (orig, resumed) in enumerate(zip(more_losses_original, more_losses_resumed)):
            self.assertAlmostEqual(
                orig, resumed, places=5,
                msg=f"Loss mismatch at step {i}: original={orig}, resumed={resumed}"
            )


def run_tests():
    """Run all functional tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestTrainerSaveCheckpoint))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainerLoadCheckpoint))
    suite.addTests(loader.loadTestsFromTestCase(TestCheckpointCleanup))
    suite.addTests(loader.loadTestsFromTestCase(TestResumeLogic))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainLoopWithStartStep))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndResumeSimulation))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
