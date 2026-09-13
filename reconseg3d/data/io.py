"""NIfTI helpers shared by ACDC / MM-WHS / EMIDEC loaders."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# NIfTI-1 datatype codes used by the numpy fallback (no nibabel required for tests).
_NIFTI_DT_INT16 = 4
_NIFTI_DT_INT32 = 8
_NIFTI_DT_FLOAT32 = 16
_NIFTI_DT_FLOAT64 = 64
_NIFTI_DT_UINT8 = 2


def _open_maybe_gzip(path: Path, mode: str):
    if path.suffix == ".gz" or path.name.endswith(".nii.gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def _numpy_save_nifti(path: Path, array: np.ndarray) -> None:
    arr = np.ascontiguousarray(array)
    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype("<f4", copy=False)
        datatype, bitpix = _NIFTI_DT_FLOAT32, 32
    elif arr.dtype == np.uint8:
        arr = arr.astype("<u1", copy=False)
        datatype, bitpix = _NIFTI_DT_UINT8, 8
    else:
        arr = arr.astype("<i2", copy=False)
        datatype, bitpix = _NIFTI_DT_INT16, 16
    ndim = arr.ndim
    if ndim < 1 or ndim > 7:
        raise ValueError(f"NIfTI fallback supports 1–7D, got ndim={ndim}")
    header = bytearray(348)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<h", header, 40, ndim)
    for i, size in enumerate(arr.shape):
        struct.pack_into("<h", header, 42 + i * 2, int(size))
    struct.pack_into("<h", header, 70, datatype)
    struct.pack_into("<h", header, 72, bitpix)
    for i in range(8):
        struct.pack_into("<f", header, 76 + i * 4, 1.0)
    struct.pack_into("<f", header, 108, 352.0)
    header[344:348] = b"n+1\x00"
    payload = bytes(header) + b"\x00\x00\x00\x00" + arr.tobytes(order="F")
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_maybe_gzip(path, "wb") as f:
        f.write(payload)


def _numpy_load_nifti(path: Path) -> np.ndarray:
    with _open_maybe_gzip(path, "rb") as f:
        raw = f.read()
    if len(raw) < 352:
        raise ValueError(f"File too small to be NIfTI: {path}")
    ndim = struct.unpack_from("<h", raw, 40)[0]
    shape = tuple(struct.unpack_from("<h", raw, 42 + i * 2)[0] for i in range(max(int(ndim), 1)))
    datatype = struct.unpack_from("<h", raw, 70)[0]
    vox_offset = int(struct.unpack_from("<f", raw, 108)[0])
    dtype_map = {
        _NIFTI_DT_UINT8: np.dtype("<u1"),
        _NIFTI_DT_INT16: np.dtype("<i2"),
        _NIFTI_DT_INT32: np.dtype("<i4"),
        _NIFTI_DT_FLOAT32: np.dtype("<f4"),
        _NIFTI_DT_FLOAT64: np.dtype("<f8"),
    }
    if datatype not in dtype_map:
        raise ValueError(f"Unsupported NIfTI datatype {datatype} in {path}")
    data = np.frombuffer(raw, dtype=dtype_map[datatype], offset=max(vox_offset, 352))
    return np.reshape(data[: int(np.prod(shape))], shape, order="F").copy()


def load_nifti(path: str | Path) -> np.ndarray:
    path = Path(path)
    try:
        import nibabel as nib

        return np.asanyarray(nib.load(str(path)).dataobj)
    except ImportError:
        return _numpy_load_nifti(path)


def save_nifti(path: str | Path, array: np.ndarray, affine: np.ndarray | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import nibabel as nib

        img = nib.Nifti1Image(np.asarray(array), np.eye(4) if affine is None else affine)
        nib.save(img, str(path))
    except ImportError:
        _numpy_save_nifti(path, np.asarray(array))


def acdc_xyzt_to_ctdhw(arr: np.ndarray) -> np.ndarray:
    """ACDC-style (X, Y, Z, T) -> (C=1, T, D=Z, H=X, W=Y)."""
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D (X,Y,Z,T), got shape {arr.shape}")
    vol = np.transpose(arr, (3, 2, 0, 1))
    return vol[np.newaxis, ...].astype(np.float32, copy=False)


def xyz_to_dhw(arr: np.ndarray) -> np.ndarray:
    """(X, Y, Z) -> (D=Z, H=X, W=Y)."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D (X,Y,Z), got shape {arr.shape}")
    return np.transpose(arr, (2, 0, 1))


def resize_ctdhw(
    volume: torch.Tensor,
    spatial_size: tuple[int, int, int],
    *,
    is_label: bool = False,
) -> torch.Tensor:
    """Resize (C, T, D, H, W) spatial dims to ``spatial_size`` (D, H, W)."""
    if volume.ndim != 5:
        raise ValueError(f"Expected (C,T,D,H,W), got {tuple(volume.shape)}")
    d, h, w = spatial_size
    if tuple(volume.shape[-3:]) == (d, h, w):
        return volume
    c, t, _, _, _ = volume.shape
    flat = volume.reshape(c * t, 1, *volume.shape[-3:])
    if is_label:
        flat = F.interpolate(flat.float(), size=(d, h, w), mode="nearest")
        return flat.reshape(c, t, d, h, w).to(dtype=volume.dtype)
    flat = F.interpolate(flat, size=(d, h, w), mode="trilinear", align_corners=False)
    return flat.reshape(c, t, d, h, w)


def resize_dhw(
    vol: torch.Tensor,
    spatial_size: tuple[int, int, int],
    *,
    is_label: bool = False,
) -> torch.Tensor:
    """Resize (D, H, W) to ``spatial_size``."""
    if vol.ndim != 3:
        raise ValueError(f"Expected (D,H,W), got {tuple(vol.shape)}")
    x = vol.view(1, 1, *vol.shape).float()
    d, h, w = spatial_size
    if is_label:
        out = F.interpolate(x, size=(d, h, w), mode="nearest")
        return out.view(d, h, w).to(dtype=vol.dtype)
    out = F.interpolate(x, size=(d, h, w), mode="trilinear", align_corners=False)
    return out.view(d, h, w).to(dtype=vol.dtype)


def subsample_time(volume: torch.Tensor, num_frames: int | None) -> torch.Tensor:
    """Uniformly subsample T of (C, T, D, H, W) to ``num_frames``."""
    if num_frames is None or volume.shape[1] == num_frames:
        return volume
    t = volume.shape[1]
    if t < num_frames:
        idx = torch.linspace(0, t - 1, num_frames).round().long().clamp(0, t - 1)
        return volume[:, idx]
    idx = torch.linspace(0, t - 1, num_frames).round().long()
    return volume[:, idx]


def remap_frame_index(orig_idx: int, orig_t: int, new_t: int) -> int:
    """Map a 0-based frame index after uniform temporal subsampling."""
    if orig_t <= 1 or new_t <= 1:
        return 0
    if orig_t == new_t:
        return int(max(0, min(orig_idx, new_t - 1)))
    # Nearest neighbor in the linspace grid used by subsample_time.
    grid = torch.linspace(0, orig_t - 1, new_t)
    return int((grid - float(orig_idx)).abs().argmin().item())

