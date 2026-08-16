"""Differentiable 3D motion (displacement) and volume warping."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from reconseg3d.models.blocks import ConvBlock3D


def identity_grid(d: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Normalized grid (D, H, W, 3) in grid_sample (x, y, z) order."""
    zz = torch.linspace(-1.0, 1.0, d, device=device, dtype=dtype)
    yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    grid_z, grid_y, grid_x = torch.meshgrid(zz, yy, xx, indexing="ij")
    return torch.stack((grid_x, grid_y, grid_z), dim=-1)


def flow_to_grid(flow: torch.Tensor) -> torch.Tensor:
    """
    Convert voxel displacement (B, 3, D, H, W) with channels (dz, dy, dx)
    into a sampling grid (B, D, H, W, 3) for ``grid_sample`` (align_corners=True).

    Scaling uses ``2 / max(dim - 1, 1)`` so singleton axes (D/H/W == 1) stay finite.
    Flow is interpreted as the inverse-sampling displacement expected by ``grid_sample``:
    a positive ``dx`` shifts the sampling location toward +x (source content appears
    to move toward -x in the warped volume).
    """
    b, _, d, h, w = flow.shape
    base = identity_grid(d, h, w, flow.device, flow.dtype).unsqueeze(0).expand(b, -1, -1, -1, -1)
    dz, dy, dx = flow[:, 0], flow[:, 1], flow[:, 2]
    scale_z = 2.0 / max(d - 1, 1)
    scale_y = 2.0 / max(h - 1, 1)
    scale_x = 2.0 / max(w - 1, 1)
    disp = torch.stack((dx * scale_x, dy * scale_y, dz * scale_z), dim=-1)
    return base + disp


def warp_volume(volume: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp (B, C, D, H, W) with displacement (B, 3, D, H, W) via ``grid_sample``."""
    grid = flow_to_grid(flow)
    return F.grid_sample(volume, grid, mode="bilinear", padding_mode="border", align_corners=True)


class MotionNet(nn.Module):
    """Predict 3D displacement between consecutive reconstructed frames."""

    def __init__(self, in_channels: int = 1, base_channels: int = 8) -> None:
        super().__init__()
        c = max(base_channels, 4)
        self.encoder = nn.Sequential(
            ConvBlock3D(in_channels * 2, c),
            ConvBlock3D(c, c),
        )
        self.flow_head = nn.Conv3d(c, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

    def forward_pair(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """src/tgt: (B, C, D, H, W) -> flow (B, 3, D, H, W) taking src toward tgt."""
        return self.flow_head(self.encoder(torch.cat([src, tgt], dim=1)))

    def forward(self, recon: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Args:
            recon: (B, C, T, D, H, W)
        Returns:
            flow_fwd, flow_bwd each (B, 3, T-1, D, H, W), or (None, None) if T < 2.
        """
        if recon.ndim != 6:
            raise ValueError(f"Expected (B,C,T,D,H,W), got {tuple(recon.shape)}")
        b, c, t, d, h, w = recon.shape
        if t < 2:
            return None, None
        src = recon[:, :, :-1].permute(0, 2, 1, 3, 4, 5).reshape(b * (t - 1), c, d, h, w)
        tgt = recon[:, :, 1:].permute(0, 2, 1, 3, 4, 5).reshape(b * (t - 1), c, d, h, w)
        fwd = self.forward_pair(src, tgt).view(b, t - 1, 3, d, h, w).permute(0, 2, 1, 3, 4, 5)
        bwd = self.forward_pair(tgt, src).view(b, t - 1, 3, d, h, w).permute(0, 2, 1, 3, 4, 5)
        return fwd.contiguous(), bwd.contiguous()


def warp_consistency_loss(recon: torch.Tensor, flow_fwd: torch.Tensor, flow_bwd: torch.Tensor | None = None) -> torch.Tensor:
    """||Vhat_{t+1} - warp(Vhat_t, u_t)||_1 (+ backward if provided)."""
    _b, _c, t, _d, _h, _w = recon.shape
    if t < 2:
        return recon.sum() * 0.0
    src = recon[:, :, :-1]
    tgt = recon[:, :, 1:]
    bt = src.shape[0] * src.shape[2]
    c = src.shape[1]
    d, h, w = src.shape[3:]
    src_f = src.permute(0, 2, 1, 3, 4, 5).reshape(bt, c, d, h, w)
    tgt_f = tgt.permute(0, 2, 1, 3, 4, 5).reshape(bt, c, d, h, w)
    flow_f = flow_fwd.permute(0, 2, 1, 3, 4, 5).reshape(bt, 3, d, h, w)
    warped = warp_volume(src_f, flow_f)
    loss = F.l1_loss(warped, tgt_f)
    if flow_bwd is not None:
        flow_b = flow_bwd.permute(0, 2, 1, 3, 4, 5).reshape(bt, 3, d, h, w)
        warped_b = warp_volume(tgt_f, flow_b)
        loss = loss + F.l1_loss(warped_b, src_f)
        loss = loss * 0.5
    return loss


def cycle_consistency_loss(recon: torch.Tensor, flow_fwd: torch.Tensor, flow_bwd: torch.Tensor) -> torch.Tensor:
    """
    Image-cycle consistency: ||warp(warp(V_t, u_fwd), u_bwd) - V_t||_1.

    This is not coordinate-composed inverse-consistent flow regularization; it
    only enforces that independently predicted fwd/bwd fields reconstruct the
    intensity after a round-trip warp.
    """
    _b, _c, t, _d, _h, _w = recon.shape
    if t < 2:
        return recon.sum() * 0.0
    src = recon[:, :, :-1]
    tgt = recon[:, :, 1:]
    bt = src.shape[0] * src.shape[2]
    c = src.shape[1]
    d, h, w = src.shape[3:]
    src_f = src.permute(0, 2, 1, 3, 4, 5).reshape(bt, c, d, h, w)
    tgt_f = tgt.permute(0, 2, 1, 3, 4, 5).reshape(bt, c, d, h, w)
    flow_f = flow_fwd.permute(0, 2, 1, 3, 4, 5).reshape(bt, 3, d, h, w)
    flow_b = flow_bwd.permute(0, 2, 1, 3, 4, 5).reshape(bt, 3, d, h, w)
    cyc_src = warp_volume(warp_volume(src_f, flow_f), flow_b)
    cyc_tgt = warp_volume(warp_volume(tgt_f, flow_b), flow_f)
    return 0.5 * (F.l1_loss(cyc_src, src_f) + F.l1_loss(cyc_tgt, tgt_f))


def volume_curve_loss(seg_logits_seq: torch.Tensor, lv_index: int = 1, rv_index: int = 2) -> torch.Tensor:
    """Soft second difference of LV/RV volume *fractions* over T (needs T >= 3).

    Counts are divided by D*H*W so the loss is O(1) and does not explode with
    spatial resolution (raw voxel counts made ``w_volsmooth`` dominate early smoke).
    """
    if seg_logits_seq.ndim != 6:
        raise ValueError(f"Expected (B,K,T,D,H,W), got {tuple(seg_logits_seq.shape)}")
    t = seg_logits_seq.shape[2]
    if t < 3:
        return seg_logits_seq.sum() * 0.0
    probs = F.softmax(seg_logits_seq, dim=1)
    k = probs.shape[1]
    spatial = float(probs.shape[3] * probs.shape[4] * probs.shape[5])
    terms = []
    for idx in (lv_index, rv_index):
        if idx >= k:
            continue
        # (B, T) fractional chamber volumes in [0, 1]
        frac = probs[:, idx].sum(dim=(2, 3, 4)) / max(spatial, 1.0)
        d2 = frac[:, 2:] - 2.0 * frac[:, 1:-1] + frac[:, :-2]
        terms.append((d2 ** 2).mean())
    if not terms:
        return seg_logits_seq.sum() * 0.0
    return sum(terms) / len(terms)


def temporal_seg_smoothness(seg_logits_seq: torch.Tensor) -> torch.Tensor:
    """L1 smoothness of softmax maps across adjacent frames."""
    if seg_logits_seq.shape[2] < 2:
        return seg_logits_seq.sum() * 0.0
    probs = F.softmax(seg_logits_seq, dim=1)
    return (probs[:, :, 1:] - probs[:, :, :-1]).abs().mean()
