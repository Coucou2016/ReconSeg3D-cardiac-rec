"""Volume transforms for 4D cardiac tensors."""

from __future__ import annotations

import torch


def normalize_intensity(volume: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Zero-mean unit-variance per sample."""
    mean = volume.mean()
    std = volume.std().clamp_min(eps)
    return (volume - mean) / std


def random_flip_3d(volume: torch.Tensor, mask: torch.Tensor | None = None, p: float = 0.5) -> tuple:
    """Random flip along D/H/W axes."""
    vol_dims = [-3, -2, -1]
    mask_dims = [0, 1, 2]
    for vd, md in zip(vol_dims, mask_dims):
        if torch.rand(1).item() < p:
            volume = torch.flip(volume, dims=[vd])
            if mask is not None:
                mask = torch.flip(mask, dims=[md])
    if mask is None:
        return volume
    return volume, mask


def apply_train_transforms(
    volume: torch.Tensor,
    mask: torch.Tensor,
    clinical: torch.Tensor | None = None,
) -> tuple:
    volume = normalize_intensity(volume)
    volume, mask = random_flip_3d(volume, mask)
    return volume, mask, clinical
