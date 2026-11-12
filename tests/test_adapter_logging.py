import torch

from integrations.locata import adapter


class _FakeModel(torch.nn.Module):
    model_name = "fake"

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.tensor(0.0))
        self.config = {"name": "fake"}

    def training_loss(self, audio, targets):
        return self.p * 0.0 + 1.0


def test_train_native_writes_train_log(tmp_path, monkeypatch):
    """Native baselines must log epochs to <out_dir>/train.log like TF-Mamba."""
    batch = {"audio": torch.zeros(2, 4, 1), "angles": [None, None]}
    monkeypatch.setattr(adapter, "build_angle_dataset", lambda *a, **k: None)
    monkeypatch.setattr(adapter, "build_loader", lambda *a, **k: [batch])

    adapter.train_native(
        _FakeModel(),
        {"train_root": "train", "val_root": "val"},
        {"epochs": 2, "patience": None, "batch_size": 2, "num_workers": 0,
         "eval_every": 1},
        out_dir=str(tmp_path), seed=42, device="cpu",
    )

    log = (tmp_path / "train.log").read_text()
    assert "epoch 1/2" in log and "epoch 2/2" in log
    assert (tmp_path / "best.pth").exists()
