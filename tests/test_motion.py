"""Motion warp + cycle losses."""

import torch

from reconseg3d.models.losses import MultiTaskLoss
from reconseg3d.models.motion import (
    MotionNet,
    cycle_consistency_loss,
    flow_to_grid,
    warp_consistency_loss,
    warp_volume,
)
from reconseg3d.models.reconseg3d import ReconSeg3D


def test_warp_identity_near_zero():
    vol = torch.randn(2, 1, 8, 16, 16)
    flow = torch.zeros(2, 3, 8, 16, 16)
    warped = warp_volume(vol, flow)
    assert torch.allclose(warped, vol, atol=1e-5)


def test_flow_to_grid_channel_order_and_scale():
    """Channels are (dz, dy, dx); grid last-dim is (x, y, z) for grid_sample."""
    d, h, w = 4, 5, 6
    flow = torch.zeros(1, 3, d, h, w)
    flow[:, 2] = 1.0  # +1 voxel in x
    grid = flow_to_grid(flow)
    assert grid.shape == (1, d, h, w, 3)
    # center voxel: x should shift by 2/(W-1)
    expected_dx = 2.0 / (w - 1)
    mid = (d // 2, h // 2, w // 2)
    # identity x at mid ≈ linspace; delta on channel 0 (x)
    base_x = torch.linspace(-1.0, 1.0, w)[mid[2]]
    assert torch.isclose(grid[0, mid[0], mid[1], mid[2], 0], base_x + expected_dx, atol=1e-5)


def test_warp_plus_one_voxel_x_shifts_content_left():
    """Positive dx samples from +x → content appears shifted toward -x."""
    vol = torch.zeros(1, 1, 4, 4, 8)
    vol[..., 4] = 1.0  # spike at x=4
    flow = torch.zeros(1, 3, 4, 4, 8)
    flow[:, 2] = 1.0
    warped = warp_volume(vol, flow)
    # After sampling from +1 x, intensity at x=3 should rise toward the spike.
    assert float(warped[..., 3].mean()) > float(vol[..., 3].mean())
    assert float(warped[..., 4].mean()) < float(vol[..., 4].mean()) + 1e-3


def test_flow_to_grid_singleton_axis_finite():
    flow = torch.zeros(1, 3, 1, 4, 4)
    flow[:, 0] = 0.5
    grid = flow_to_grid(flow)
    assert torch.isfinite(grid).all()


def test_warp_and_cycle_finite_backward():
    recon = torch.randn(2, 1, 4, 8, 16, 16, requires_grad=True)
    net = MotionNet(in_channels=1, base_channels=4)
    fwd, bwd = net(recon)
    assert fwd is not None and bwd is not None
    warp_l = warp_consistency_loss(recon, fwd, bwd)
    cyc_l = cycle_consistency_loss(recon, fwd, bwd)
    total = warp_l + cyc_l
    assert torch.isfinite(total)
    total.backward()
    assert recon.grad is not None
    assert torch.isfinite(recon.grad).all()


def test_multitask_motion_terms():
    model = ReconSeg3D(clinical_dim=0, per_frame_recon=True, base_channels=8)
    criterion = MultiTaskLoss(num_seg_classes=5, w_warp=0.1, w_cycle=0.1, w_volsmooth=0.05, alpha1=0.1)
    x = torch.randn(1, 1, 4, 8, 16, 16)
    seg = torch.randint(0, 5, (1, 8, 16, 16))
    mace = torch.tensor([1.0])
    out = model(x)
    losses = criterion(
        out.reconstruction,
        out.segmentation,
        out.mace_logits,
        x,
        seg,
        mace,
        flow=out.flow,
        flow_bwd=out.flow_bwd,
        seg_sequence=out.seg_sequence,
    )
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert torch.isfinite(losses["warp"])
    assert torch.isfinite(losses["cycle"])


def test_volume_curve_loss_normalized_scale():
    """Fractional volumes keep volsmooth O(1), not O((DHW)^2)."""
    from reconseg3d.models.motion import volume_curve_loss

    # Uniform logits → flat curve → ~0; noisy logits on large grid must stay modest.
    logits = torch.zeros(2, 5, 8, 16, 32, 32)
    logits[:, 1] = 0.5
    logits[:, 2] = 0.5
    flat = volume_curve_loss(logits)
    assert float(flat) < 1e-6

    noisy = torch.randn(2, 5, 8, 16, 32, 32)
    v = volume_curve_loss(noisy)
    assert torch.isfinite(v)
    # Raw voxel-count formulation exceeded 1e2–1e3; fraction form is << 1 typically.
    assert float(v) < 1.0
