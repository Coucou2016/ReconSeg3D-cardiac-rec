"""Diagnosis-stratified ACDC K-fold split helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from reconseg3d.data.acdc import (
    ACDC_GROUP_NAMES,
    discover_acdc_patients,
    parse_info_cfg,
    phenotype_from_group,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS_DIR = REPO_ROOT / "splits"


def patient_diagnosis(patient_dir: Path) -> str:
    info_path = patient_dir / "Info.cfg"
    if not info_path.is_file():
        return "NOR"
    info = parse_info_cfg(info_path)
    group = info.get("Group", "NOR").upper()
    # Normalize ARV → RV
    if group == "ARV":
        group = "RV"
    if group not in ACDC_GROUP_NAMES:
        # Map unknown to closest bucket via phenotype index.
        idx = phenotype_from_group(group)
        group = ACDC_GROUP_NAMES[min(idx, len(ACDC_GROUP_NAMES) - 1)]
    return group


def stratified_kfold_patients(
    patients: Sequence[Path],
    *,
    n_folds: int = 5,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Build diagnosis-stratified K-fold splits (NOR/MINF/DCM/HCM/RV).

    Each fold dict: ``{fold, train, val, test, diagnosis_counts}``.
    For fold ``k``, ``test`` = fold k, ``val`` = fold (k+1)%K, ``train`` = rest.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    by_diag: dict[str, list[str]] = defaultdict(list)
    for p in patients:
        by_diag[patient_diagnosis(p)].append(p.name)
    rng = np.random.default_rng(seed)
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for diag in ACDC_GROUP_NAMES:
        names = list(by_diag.get(diag, []))
        rng.shuffle(names)
        for i, name in enumerate(names):
            folds[i % n_folds].append(name)
    # Stable sort within each fold for reproducibility.
    for f in folds:
        f.sort()

    out: list[dict[str, Any]] = []
    for k in range(n_folds):
        test = list(folds[k])
        val = list(folds[(k + 1) % n_folds])
        train = []
        for j in range(n_folds):
            if j == k or j == (k + 1) % n_folds:
                continue
            train.extend(folds[j])
        train = sorted(train)
        counts = {g: 0 for g in ACDC_GROUP_NAMES}
        for name in test:
            # Recover diagnosis from original grouping.
            for g, names in by_diag.items():
                if name in names:
                    counts[g] = counts.get(g, 0) + 1
                    break
        out.append(
            {
                "fold": k,
                "n_folds": n_folds,
                "seed": seed,
                "stratify_by": list(ACDC_GROUP_NAMES),
                "train": train,
                "val": val,
                "test": test,
                "diagnosis_counts_test": counts,
            }
        )
    return out


def write_acdc_folds(
    root: str | Path | None = None,
    *,
    out_dir: str | Path | None = None,
    n_folds: int = 5,
    seed: int = 42,
    fake_if_empty: bool = True,
    n_fake: int = 20,
) -> list[Path]:
    """Write ``splits/acdc_fold{0..n-1}.json``. Creates fake ACDC tree if needed."""
    from reconseg3d.data.acdc import make_fake_acdc

    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_SPLITS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if root is None:
        root = REPO_ROOT / "data" / "acdc"
    root = Path(root)
    patients = discover_acdc_patients(root)
    if not patients and fake_if_empty:
        make_fake_acdc(root, n_patients=n_fake, spatial=(8, 16, 16), n_frames=8, seed=seed)
        patients = discover_acdc_patients(root)
    if not patients:
        raise FileNotFoundError(f"No ACDC patients under {root}")
    folds = stratified_kfold_patients(patients, n_folds=n_folds, seed=seed)
    paths: list[Path] = []
    for fold in folds:
        path = out_dir / f"acdc_fold{fold['fold']}.json"
        path.write_text(json.dumps(fold, indent=2), encoding="utf-8")
        paths.append(path)
    manifest = {
        "dataset": "ACDC",
        "n_folds": n_folds,
        "seed": seed,
        "stratify_by": list(ACDC_GROUP_NAMES),
        "n_patients": len(patients),
        "files": [p.name for p in paths],
        "note": (
            "Diagnosis-stratified CV. Real ACDC subject tables remain 待补充 "
            "until licensed data are mounted; committed lists may reflect "
            "fake CI patients when real data were absent at generation time."
        ),
    }
    (out_dir / "acdc_folds_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return paths


def load_fold_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Fold file must be a JSON object: {path}")
    for key in ("train", "val", "test"):
        if key not in data:
            raise ValueError(f"Fold file missing '{key}': {path}")
    return data


def resolve_fold_path(cfg: dict[str, Any], fold: int | None = None) -> Path | None:
    """Resolve fold JSON from data config. Returns None if folds unused."""
    data_cfg = cfg.get("data", cfg)
    fold_file = data_cfg.get("fold_file")
    if fold_file:
        p = Path(fold_file)
        if not p.is_file():
            p = REPO_ROOT / fold_file
        return p
    fold_idx = fold if fold is not None else data_cfg.get("fold")
    if fold_idx is None:
        return None
    splits_dir = Path(data_cfg.get("splits_dir", DEFAULT_SPLITS_DIR))
    if not splits_dir.is_absolute():
        splits_dir = REPO_ROOT / splits_dir
    return splits_dir / f"acdc_fold{int(fold_idx)}.json"
