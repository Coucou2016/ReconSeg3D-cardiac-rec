"""Fake ACDC / MM-WHS / EMIDEC loaders."""

from pathlib import Path

import numpy as np
import torch

from reconseg3d.data.acdc import ACDCDataset, make_fake_acdc
from reconseg3d.data.emidec import EMIDECDataset, make_fake_emidec
from reconseg3d.data.mmwhs import MMWHSDataset, make_fake_mmwhs, map_mmwhs_labels


def test_fake_acdc_loader(tmp_path: Path):
    root = make_fake_acdc(tmp_path / "acdc", n_patients=4, spatial=(8, 16, 16), n_frames=4)
    ds = ACDCDataset(root, num_frames=4, spatial_size=(8, 16, 16), clinical_dim=4, train=False, split="train")
    assert len(ds) >= 1
    sample = ds[0]
    assert sample["volume"].shape == (1, 4, 8, 16, 16)
    assert sample["segmentation"].shape == (8, 16, 16)
    assert sample["clinical"].shape == (4,)
    assert int(sample["phenotype"].item()) in range(5)
    assert "seg_frame_indices" in sample
    assert sample["segmentation_sequence"].shape[0] == 4
    assert sample["seg_valid_mask"].dtype == torch.bool


def test_fake_mmwhs_loader(tmp_path: Path):
    root = make_fake_mmwhs(tmp_path / "mmwhs", n_cases=2, spatial=(8, 16, 16))
    mapped = map_mmwhs_labels(np.array([0, 500, 600, 205]))
    assert list(mapped) == [0, 1, 2, 3]
    ds = MMWHSDataset(root, num_frames=1, spatial_size=(8, 16, 16), clinical_dim=0, train=False, split="train")
    sample = ds[0]
    assert sample["volume"].shape == (1, 1, 8, 16, 16)
    labels = set(sample["segmentation"].unique().tolist())
    assert labels.issubset({0, 1, 2, 3})


def test_fake_emidec_loader(tmp_path: Path):
    root = make_fake_emidec(tmp_path / "emidec", n_cases=2, spatial=(8, 16, 16))
    ds = EMIDECDataset(root, num_frames=1, spatial_size=(8, 16, 16), stack_scar_channel=True, train=False)
    sample = ds[0]
    assert sample["volume"].shape[0] == 2
    assert sample["volume"].shape[-3:] == (8, 16, 16)
    assert sample["segmentation"].max() >= 3
