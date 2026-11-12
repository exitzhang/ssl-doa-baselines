import pytest
import torch

from ssl_doa_baselines.training import fit


class _FakeModel(torch.nn.Module):
    """Loss is a constant chosen by the test; ``fit`` only reads its value."""

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.tensor(0.0))
        self.train_loss = 1.0
        self.val_loss = 1.0

    def training_loss(self, audio, targets):
        value = self.val_loss if not self.training else self.train_loss
        return self.p * 0.0 + value


def _loader(value):
    return [{"audio": torch.zeros(2, 4, 1), "angles": [None, None]}]


def test_patience_counts_validation_rounds_not_epochs():
    """LOCATA Trainer semantics: only eval rounds move the early-stop counter."""
    model = _FakeModel()
    model.val_loss = 1.0  # first eval improves over inf, then never again
    logs = []
    history = fit(
        model, _loader(1.0), _loader(1.0), epochs=10, device="cpu",
        lr=1e-3, scheduler=None, amp=False, patience=2, eval_every=2,
        save_dir=None, log_fn=logs.append,
    )
    # evals at epochs 1, 3, 5 -> stale 0, 1, 2 -> stop after the 5th epoch
    assert len(history) == 5
    assert sum(row["val_loss"] is not None for row in history) == 3
    assert any("early stopping at epoch 5" in line for line in logs)


def test_no_validation_uses_train_loss_monitor():
    model = _FakeModel()
    logs = []
    history = fit(
        model, _loader(1.0), None, epochs=10, device="cpu", lr=1e-3,
        scheduler=None, amp=False, patience=2, eval_every=2,
        save_dir=None, log_fn=logs.append,
    )
    assert len(history) == 3  # improved at epoch 1, stale at 2 and 3
    assert any("early stopping at epoch 3" in line for line in logs)
