#!/usr/bin/env python
"""Evaluate a checkpoint on val (or test when the loader split exists)."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reconseg3d.data.dataset import build_dataloader
from reconseg3d.inference.predictor import Predictor
from reconseg3d.models.reconseg3d import output_to_metric_dict
from reconseg3d.training.metrics import compute_metrics
from reconseg3d.training.trainer import EPOCH_RANKING_KEYS, Trainer
from reconseg3d.utils.config import load_config


def _json_safe(obj: dict) -> dict:
    out = {}
    for k, v in obj.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[k] = None
        else:
            out[k] = v
    return out


def _merge_prior_losses(out_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Keep train-time loss_* fields when eval overwrites metrics.json."""
    if not out_path.is_file():
        return payload
    try:
        prior = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return payload
    if not isinstance(prior, dict):
        return payload
    merged = dict(payload)
    for k, v in prior.items():
        if k.startswith("loss_") and k not in merged:
            merged[k] = v
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ReconSeg3D")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Loader split. Official ACDC test folds are TODO (5-fold); "
        "current ACDCDataset maps non-train → held-out val-like patients.",
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default=None,
        help="Write averaged metrics JSON (default: <checkpoint_dir>/metrics.json)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)
    predictor = Predictor(args.checkpoint)
    # Map "test" → val-style split until 5-fold test files exist.
    split = "val" if args.split == "test" else args.split
    if args.split == "test":
        logging.warning(
            "split=test: using val-style held-out patients (TODO: official 5-fold test lists)."
        )
    loader = build_dataloader(cfg, split)

    sums: dict[str, float] = {}
    n = 0
    extras_list: list[dict[str, Any]] = []
    for batch in loader:
        volume = batch["volume"].to(predictor.device)
        clinical = batch["clinical"].to(predictor.device) if predictor.clinical_dim > 0 else None
        seg_frame_indices = batch.get("seg_frame_indices")
        if seg_frame_indices is not None:
            seg_frame_indices = seg_frame_indices.to(predictor.device)
        out = predictor.model(volume, clinical, seg_frame_indices=seg_frame_indices)
        spacing = batch.get("spacing")
        metrics = compute_metrics(
            output_to_metric_dict(out),
            batch,
            num_classes=cfg.get("model", {}).get("num_seg_classes", 5),
            compute_hd95=bool(cfg.get("metrics", {}).get("hd95", False)),
            spacing=spacing,
        )
        for k, v in metrics.items():
            if k in EPOCH_RANKING_KEYS:
                continue
            if isinstance(v, float) and v == v:
                sums[k] = sums.get(k, 0.0) + v

        extras: dict[str, Any] = {
            "mace_logits": out.mace_logits.detach().float().cpu(),
            "mace": batch["mace"].detach().float().cpu(),
        }
        if "time" in batch and "event" in batch:
            extras["time"] = batch["time"].detach().float().cpu()
            extras["event"] = batch["event"].detach().float().cpu()
        if out.phenotype_logits is not None and "phenotype" in batch:
            extras["phenotype_logits"] = out.phenotype_logits.detach().float().cpu()
            extras["phenotype"] = batch["phenotype"].detach().long().cpu()
        extras_list.append(extras)
        n += 1

    avg = {k: v / max(n, 1) for k, v in sums.items()}
    avg.update(Trainer._pool_ranking_metrics(extras_list))
    ckpt_path = Path(args.checkpoint)
    run_name = predictor.cfg.get("run_name") or ckpt_path.parent.name
    payload = {
        "run_name": run_name,
        "split": args.split,
        "loader_split": split,
        "n_batches": n,
        **_json_safe(avg),
    }
    print(json.dumps(payload, indent=2))

    out_path = Path(args.metrics_out) if args.metrics_out else ckpt_path.parent / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _merge_prior_losses(out_path, payload)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
