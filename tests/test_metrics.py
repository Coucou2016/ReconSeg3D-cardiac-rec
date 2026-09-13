"""Metrics edge-case tests."""

import numpy as np
import torch

from reconseg3d.training.metrics import _safe_auc, compute_metrics, mace_metrics


def test_auc_single_class_returns_nan():
    y = np.array([0, 0, 0])
    s = np.array([0.1, 0.2, 0.3])
    assert np.isnan(_safe_auc(y, s))


def test_auc_binary():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    auc = _safe_auc(y, s)
    assert auc == 1.0


def test_mace_metrics():
    logits = torch.tensor([2.0, -2.0])
    labels = torch.tensor([1.0, 0.0])
    m = mace_metrics(logits, labels)
    assert m["mace_accuracy"] == 1.0
    assert m["mace_auc"] == 1.0


def test_compute_metrics_smoke():
    batch = {
        "volume": torch.randn(2, 1, 4, 8, 16, 16),
        "segmentation": torch.randint(0, 5, (2, 8, 16, 16)),
        "mace": torch.tensor([0.0, 1.0]),
    }
    outputs = {
        "segmentation": torch.randn(2, 5, 8, 16, 16),
        "reconstruction": batch["volume"],
        "mace_logits": torch.tensor([0.5, -0.5]),
    }
    m = compute_metrics(outputs, batch, num_classes=5)
    assert "dice_mean" in m
    assert "mace_auc" in m
    # Perfect recon should score high on global SSIM proxy (data_range-aware).
    assert m["recon_ssim"] > 0.9
    assert m["recon_ssim_proxy"] > 0.9


def test_pool_ranking_metrics_epoch_level():
    """Mean-of-batch AUC is invalid; Trainer pools logits then scores once."""
    from reconseg3d.training.trainer import Trainer

    extras = [
        {"mace_logits": torch.tensor([2.0, -2.0]), "mace": torch.tensor([1.0, 0.0])},
        {"mace_logits": torch.tensor([-1.0, 1.5]), "mace": torch.tensor([0.0, 1.0])},
    ]
    pooled = Trainer._pool_ranking_metrics(extras)
    assert pooled["mace_auc"] == 1.0
    assert pooled["mace_accuracy"] == 1.0
