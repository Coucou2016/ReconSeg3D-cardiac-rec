"""Metrics: physical HD95 spacing, ED↔ES label propagation, SSIM proxy naming."""

import numpy as np
import torch

from reconseg3d.training.metrics import (
    compose_flows_between,
    ed_es_label_propagation_metrics,
    hd95_binary,
    propagate_label_with_flow,
)


def test_hd95_spacing_anisotropic():
    """Spacing (8,1,1) vs (1,1,1) scales distance along D."""
    pred = np.zeros((8, 8, 8), dtype=bool)
    tgt = np.zeros((8, 8, 8), dtype=bool)
    pred[2, 4, 4] = True
    pred[2, 4, 5] = True
    pred[2, 5, 4] = True
    tgt[4, 4, 4] = True
    tgt[4, 4, 5] = True
    tgt[4, 5, 4] = True
    vox = hd95_binary(pred, tgt, spacing=(1.0, 1.0, 1.0))
    mm = hd95_binary(pred, tgt, spacing=(8.0, 1.0, 1.0))
    assert vox == vox and mm == mm
    # Separation is mainly along z (D); 8 mm spacing should inflate HD95
    assert mm > vox * 3.0


def test_label_propagation_known_translation():
    """Warp a blob by +1 voxel dx; Dice vs manually shifted GT is high."""
    label = torch.zeros(1, 8, 8, 12, dtype=torch.long)
    label[0, 3:6, 3:6, 5:8] = 1
    shifted = torch.zeros_like(label)
    shifted[0, 3:6, 3:6, 4:7] = 1  # content moved toward -x under +dx pull
    flow = torch.zeros(1, 3, 8, 8, 12)
    flow[:, 2] = 1.0
    prop = propagate_label_with_flow(label, flow, num_classes=2)
    # Interior overlap should be strong
    inter = ((prop == 1) & (shifted == 1)).sum().item()
    union = ((prop == 1) | (shifted == 1)).sum().item()
    dice = 2 * inter / max(union, 1)
    assert dice > 0.7


def test_ed_es_propagation_api_synthetic():
    """API smoke: identity flow → near-perfect Dice (same ED/ES masks)."""
    seg = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    seg[0, 2:6, 2:6, 2:6] = 1
    # T=3 → 2 adjacent flows
    flow = torch.zeros(1, 3, 2, 8, 8, 8)
    flow_bwd = torch.zeros_like(flow)
    m = ed_es_label_propagation_metrics(
        seg,
        seg,
        flow,
        ed_index=0,
        es_index=2,
        flow_bwd=flow_bwd,
        num_classes=2,
        compute_hd95=False,
    )
    assert m["prop_ed2es_dice_mean"] > 0.99


def test_compose_flows_between_identity():
    flow = torch.zeros(1, 3, 3, 4, 4, 4)
    c = compose_flows_between(flow, 0, 3)
    assert torch.allclose(c, torch.zeros_like(c))
