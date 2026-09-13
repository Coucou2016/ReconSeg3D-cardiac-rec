"""Evaluation metrics with safe edge-case handling."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from reconseg3d.models.motion import (
    compose_flow_sequence,
    cycle_consistency_loss,
    inverse_consistency_loss,
    jacobian_stats,
    volume_curve_loss,
    warp_consistency_loss,
    warp_volume,
)

SEG_NAME = {1: "lv", 2: "rv", 3: "myo", 4: "scar"}
HD95_MAX_SURFACE = 4000


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC with guards for single-class or tiny batches."""
    y_true = np.asarray(y_true).astype(np.int32).ravel()
    y_score = np.asarray(y_score).astype(np.float64).ravel()
    if y_true.size == 0:
        return float("nan")
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if mask.ndim != 3 or not mask.any():
        return np.zeros((0, 3), dtype=np.int32)
    pad = np.pad(mask, 1, mode="constant")
    inner = (
        pad[1:-1, 1:-1, 1:-1]
        & pad[:-2, 1:-1, 1:-1]
        & pad[2:, 1:-1, 1:-1]
        & pad[1:-1, :-2, 1:-1]
        & pad[1:-1, 2:, 1:-1]
        & pad[1:-1, 1:-1, :-2]
        & pad[1:-1, 1:-1, 2:]
    )
    surface = mask & ~inner
    if not surface.any():
        surface = mask
    return np.argwhere(surface)


def hd95_binary(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Sequence[float] | None = None,
) -> float:
    """95th-percentile Hausdorff distance.

    When ``spacing`` is ``(sz, sy, sx)`` in mm (matching D,H,W), distances are
    physical. Default ``None`` uses voxel units. Surfaces larger than
    ``HD95_MAX_SURFACE`` are subsampled.
    """
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    if pred.sum() == 0 or target.sum() == 0:
        return float("nan")
    ps = _surface_voxels(pred)
    gs = _surface_voxels(target)
    if len(ps) == 0 or len(gs) == 0:
        return float("nan")
    rng = np.random.default_rng(0)
    if len(ps) > HD95_MAX_SURFACE:
        ps = ps[rng.choice(len(ps), HD95_MAX_SURFACE, replace=False)]
    if len(gs) > HD95_MAX_SURFACE:
        gs = gs[rng.choice(len(gs), HD95_MAX_SURFACE, replace=False)]
    scale = np.ones(3, dtype=np.float64)
    if spacing is not None:
        scale = np.asarray(spacing, dtype=np.float64).reshape(3)
    delta = (ps[:, None, :] - gs[None, :, :]).astype(np.float64) * scale.reshape(1, 1, 3)
    dist = np.sqrt((delta ** 2).sum(axis=-1))
    d_pg = dist.min(axis=1)
    d_gp = dist.min(axis=0)
    return float(np.percentile(np.concatenate([d_pg, d_gp]), 95))


def dice_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Mean Dice per class; pred/target (B,D,H,W) int."""
    scores: dict[str, float] = {}
    for cls in range(num_classes):
        p = (pred == cls).float()
        t = (target == cls).float()
        inter = (p * t).sum().item()
        union = p.sum().item() + t.sum().item()
        dice = (2 * inter + eps) / (union + eps) if union > 0 else 1.0
        scores[f"dice_class_{cls}"] = dice
        name = SEG_NAME.get(cls)
        if name is not None:
            scores[f"dice_{name}"] = dice
    valid = [v for k, v in scores.items() if k.startswith("dice_class_") and (k != "dice_class_0" or num_classes == 1)]
    scores["dice_mean"] = float(np.mean(valid)) if valid else 0.0
    return scores


def hd95_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    spacing: Sequence[float] | None = None,
) -> dict[str, float]:
    """Mean HD95 over LV/RV/MYO (classes 1–3 present in ``num_classes``)."""
    pred_np = pred.detach().cpu().numpy()
    tgt_np = target.detach().cpu().numpy()
    if pred_np.ndim == 3:
        pred_np = pred_np[None]
        tgt_np = tgt_np[None]
    out: dict[str, float] = {}
    acc: dict[str, list[float]] = {}
    for b in range(pred_np.shape[0]):
        for cls in range(1, min(num_classes, 4)):
            val = hd95_binary(pred_np[b] == cls, tgt_np[b] == cls, spacing=spacing)
            name = SEG_NAME.get(cls, str(cls))
            acc.setdefault(name, []).append(val)
    all_vals: list[float] = []
    for name, vals in acc.items():
        finite = [v for v in vals if v == v]
        out[f"hd95_{name}"] = float(np.mean(finite)) if finite else float("nan")
        all_vals.extend(finite)
    out["hd95_mean"] = float(np.mean(all_vals)) if all_vals else float("nan")
    return out


def iou_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6) -> float:
    ious = []
    for cls in range(1, num_classes):
        p = pred == cls
        t = target == cls
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union > 0:
            ious.append((inter + eps) / (union + eps))
    return float(np.mean(ious)) if ious else 0.0


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float | None = None) -> float:
    pred = pred.float()
    target = target.float()
    mse = F.mse_loss(pred, target).item()
    if mse <= 0:
        return 99.0
    if data_range is None:
        data_range = float((target.max() - target.min()).item()) or 1.0
    return float(10.0 * math.log10((data_range ** 2) / mse))


def ssim_global(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float | None = None,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Global (window-free) SSIM **proxy**. Not a 3D sliding-window SSIM.

    Do not claim windowed SSIM in paper tables until a windowed implementation
    is added. Stabilizers scale with ``data_range`` (Wang et al.).
    """
    x = pred.float().reshape(-1)
    y = target.float().reshape(-1)
    if data_range is None:
        data_range = float((y.max() - y.min()).item()) or 1.0
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    mu_x = x.mean()
    mu_y = y.mean()
    var_x = x.var(unbiased=False)
    var_y = y.var(unbiased=False)
    cov = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    return float((num / den.clamp_min(1e-12)).item())


def ef_proxy_from_seg_sequence(seg_seq: torch.Tensor, lv_index: int = 1) -> float:
    """EF ≈ (max LV voxels − min LV voxels) / max over T. ``seg_seq`` (B,K,T,D,H,W) logits or (B,T,D,H,W) labels."""
    if seg_seq.ndim == 6:
        labels = seg_seq.argmax(dim=1)
    else:
        labels = seg_seq
    counts = (labels == lv_index).float().sum(dim=(2, 3, 4))
    edv = counts.max(dim=1).values
    esv = counts.min(dim=1).values
    ef = (edv - esv) / edv.clamp_min(1.0)
    return float(ef.mean().item())


def concordance_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Harrell C-index. NaN if no comparable pairs."""
    risk = np.asarray(risk, dtype=np.float64).ravel()
    time = np.asarray(time, dtype=np.float64).ravel()
    event = np.asarray(event, dtype=np.float64).ravel()
    n = risk.size
    conc = 0.0
    total = 0.0
    for i in range(n):
        if event[i] <= 0:
            continue
        for j in range(n):
            if time[i] >= time[j]:
                continue
            total += 1.0
            if risk[i] > risk[j]:
                conc += 1.0
            elif risk[i] == risk[j]:
                conc += 0.5
    if total <= 0:
        return float("nan")
    return float(conc / total)


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 200,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap mean and (1-alpha) CI. Returns (mean, lo, hi). Tiny-n safe."""
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = rng if rng is not None else np.random.default_rng(0)
    n = values.size
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        means[b] = rng.choice(values, size=n, replace=True).mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(values.mean()), lo, hi


def mace_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    probs = torch.sigmoid(logits.detach()).cpu().numpy()
    y = labels.detach().cpu().numpy()
    pred_bin = (probs >= threshold).astype(np.int32)
    tp = int(((pred_bin == 1) & (y == 1)).sum())
    tn = int(((pred_bin == 0) & (y == 0)).sum())
    fp = int(((pred_bin == 1) & (y == 0)).sum())
    fn = int(((pred_bin == 0) & (y == 1)).sum())
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    acc = (tp + tn) / max(len(y), 1)
    return {
        "mace_auc": _safe_auc(y, probs),
        "mace_sensitivity": float(sens),
        "mace_specificity": float(spec),
        "mace_accuracy": float(acc),
    }


def compose_flows_between(
    flow_fwd: torch.Tensor,
    t_src: int,
    t_tgt: int,
) -> torch.Tensor:
    """
    Compose adjacent forward flows from frame ``t_src`` to ``t_tgt``.

    ``flow_fwd``: (B, 3, T-1, D, H, W).
    """
    if t_tgt == t_src:
        b, _, _, d, h, w = flow_fwd.shape
        return torch.zeros(b, 3, d, h, w, device=flow_fwd.device, dtype=flow_fwd.dtype)
    if t_tgt < t_src:
        raise ValueError("compose_flows_between requires t_tgt >= t_src; use flow_bwd for reverse")
    flows = [flow_fwd[:, :, i] for i in range(t_src, t_tgt)]
    return compose_flow_sequence(flows)


def propagate_label_with_flow(
    label: torch.Tensor,
    flow: torch.Tensor,
    *,
    num_classes: int | None = None,
) -> torch.Tensor:
    """Warp an integer label volume with a pull displacement (nearest via one-hot)."""
    if label.ndim != 4:
        raise ValueError(f"label must be (B,D,H,W), got {tuple(label.shape)}")
    k = int(num_classes) if num_classes is not None else int(label.max().item()) + 1
    one_hot = F.one_hot(label.long().clamp_min(0), num_classes=max(k, 1)).permute(0, 4, 1, 2, 3).float()
    warped = warp_volume(one_hot, flow)
    return warped.argmax(dim=1)


def ed_es_label_propagation_metrics(
    seg_ed: torch.Tensor,
    seg_es: torch.Tensor,
    flow_fwd: torch.Tensor,
    ed_index: torch.Tensor | int,
    es_index: torch.Tensor | int,
    flow_bwd: torch.Tensor | None = None,
    num_classes: int = 4,
    spacing: Sequence[float] | None = None,
    compute_hd95: bool = True,
) -> dict[str, float]:
    """
    Warp ED GT mask to ES (and reverse) with predicted flows; report Dice (+ HD95).

    Real ACDC subject-level tables remain **待补充** until licensed data are mounted;
    this API is unit-tested with synthetic known warps.
    """
    if isinstance(ed_index, int):
        ed_i = ed_index
        es_i = int(es_index)  # type: ignore[arg-type]
    else:
        ed_i = int(ed_index.reshape(-1)[0].item())
        es_i = int(es_index.reshape(-1)[0].item())  # type: ignore[union-attr]

    out: dict[str, float] = {}
    if es_i >= ed_i:
        flow_ed_to_es = compose_flows_between(flow_fwd, ed_i, es_i)
    elif flow_bwd is not None:
        flow_ed_to_es = compose_flows_between(flow_bwd, es_i, ed_i)
    else:
        flow_ed_to_es = torch.zeros_like(flow_fwd[:, :, 0])

    prop_es = propagate_label_with_flow(seg_ed, flow_ed_to_es, num_classes=num_classes)
    dice_fwd = dice_per_class(prop_es, seg_es.long(), num_classes)
    out["prop_ed2es_dice_mean"] = dice_fwd["dice_mean"]
    for name in ("lv", "rv", "myo"):
        key = f"dice_{name}"
        if key in dice_fwd:
            out[f"prop_ed2es_{key}"] = dice_fwd[key]

    if flow_bwd is not None and es_i > ed_i:
        flow_es_to_ed = compose_flow_sequence([flow_bwd[:, :, i] for i in range(es_i - 1, ed_i - 1, -1)])
        prop_ed = propagate_label_with_flow(seg_es, flow_es_to_ed, num_classes=num_classes)
        dice_bwd = dice_per_class(prop_ed, seg_ed.long(), num_classes)
        out["prop_es2ed_dice_mean"] = dice_bwd["dice_mean"]

    if compute_hd95:
        try:
            hd = hd95_per_class(prop_es, seg_es.long(), num_classes, spacing=spacing)
            out["prop_ed2es_hd95_mean"] = hd["hd95_mean"]
        except Exception:
            out["prop_ed2es_hd95_mean"] = float("nan")
    return out


def compute_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    num_classes: int = 5,
    compute_hd95: bool = True,
    spacing: Sequence[float] | None = None,
) -> dict[str, float]:
    seg_pred = outputs["segmentation"].argmax(dim=1)
    seg_tgt = batch["segmentation"].long()
    if seg_tgt.device != seg_pred.device:
        seg_tgt = seg_tgt.to(seg_pred.device)
    metrics: dict[str, Any] = dice_per_class(seg_pred, seg_tgt, num_classes)
    metrics["iou_foreground"] = iou_per_class(seg_pred, seg_tgt, num_classes)
    if compute_hd95:
        try:
            metrics.update(hd95_per_class(seg_pred, seg_tgt, num_classes, spacing=spacing))
        except Exception:
            metrics["hd95_mean"] = float("nan")

    recon = outputs.get("reconstruction")
    vol = batch.get("volume_target", batch["volume"])
    if recon is not None:
        vol = vol.to(device=recon.device, dtype=recon.dtype)
        metrics["recon_mae"] = F.l1_loss(recon, vol, reduction="mean").item()
        metrics["recon_psnr"] = psnr(recon, vol)
        # Global SSIM proxy — not windowed SSIM.
        metrics["recon_ssim_proxy"] = ssim_global(recon, vol)
        metrics["recon_ssim"] = metrics["recon_ssim_proxy"]  # backward-compatible alias

    if "mace_logits" in outputs and "mace" in batch:
        metrics.update(mace_metrics(outputs["mace_logits"], batch["mace"].to(outputs["mace_logits"].device)))

    if "mace_logits" in outputs and "time" in batch and "event" in batch:
        risk = outputs["mace_logits"].detach().cpu().numpy()
        metrics["c_index"] = concordance_index(
            risk,
            batch["time"].detach().cpu().numpy(),
            batch["event"].detach().cpu().numpy(),
        )

    if "phenotype_logits" in outputs and "phenotype" in batch:
        logits_p = outputs["phenotype_logits"]
        pred_p = logits_p.argmax(dim=1).detach().cpu().numpy()
        y_p = batch["phenotype"].detach().cpu().numpy()
        metrics["phenotype_acc"] = float((pred_p == y_p).mean()) if y_p.size else float("nan")
        if logits_p.shape[1] > 1:
            probs = torch.softmax(logits_p.detach(), dim=-1)[:, 1].cpu().numpy()
            y_minf = (y_p == 1).astype(np.int32)
            metrics["phenotype_auc"] = _safe_auc(y_minf, probs)
        else:
            metrics["phenotype_auc"] = float("nan")

    seg_seq = outputs.get("seg_sequence")
    if seg_seq is not None:
        metrics["ef_proxy"] = ef_proxy_from_seg_sequence(seg_seq)
        try:
            metrics["vol_curve"] = float(volume_curve_loss(seg_seq).item())
        except Exception:
            metrics["vol_curve"] = float("nan")

    flow = outputs.get("flow")
    flow_bwd = outputs.get("flow_bwd")
    if recon is not None and flow is not None and recon.shape[2] > 1:
        try:
            metrics["warp_error"] = float(warp_consistency_loss(recon, flow, flow_bwd).item())
        except Exception:
            metrics["warp_error"] = float("nan")
        if flow_bwd is not None:
            try:
                metrics["cycle_error"] = float(cycle_consistency_loss(recon, flow, flow_bwd).item())
            except Exception:
                metrics["cycle_error"] = float("nan")
            try:
                metrics["inv_error"] = float(inverse_consistency_loss(flow, flow_bwd).item())
            except Exception:
                metrics["inv_error"] = float("nan")
        try:
            metrics.update(jacobian_stats(flow))
        except Exception:
            metrics["jac_neg_ratio"] = float("nan")

    if (
        flow is not None
        and "ed_index" in batch
        and "es_index" in batch
        and "segmentation_sequence" in batch
    ):
        try:
            ed_i = int(batch["ed_index"].reshape(-1)[0].item())
            es_i = int(batch["es_index"].reshape(-1)[0].item())
            seg_seq_gt = batch["segmentation_sequence"]
            if seg_seq_gt.ndim == 5:
                seg_ed = seg_seq_gt[:, ed_i].to(flow.device)
                seg_es = seg_seq_gt[:, es_i].to(flow.device)
                prop = ed_es_label_propagation_metrics(
                    seg_ed,
                    seg_es,
                    flow,
                    ed_i,
                    es_i,
                    flow_bwd=flow_bwd,
                    num_classes=num_classes,
                    spacing=spacing,
                    compute_hd95=compute_hd95,
                )
                metrics.update(prop)
        except Exception:
            metrics["prop_ed2es_dice_mean"] = float("nan")

    for k, v in list(metrics.items()):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            metrics[k] = float("nan")
    return metrics
