# Paper plan (living)

This repo is a **research re-implementation + extension**, not a drop-in reproduction of the private AMI numbers in the original npj Digital Medicine paper.

## What the original paper did

1. **ReconSeg3D (public MM-WHS + ACDC):** 3D ViT reconstruction of dense volumes from sparse short-axis stacks, then 3D nnU-Net segmentation (LV / RV / LVM).
2. **HeartTTable (private AMI):** spatial + temporal + table transformers with class-token cross-attention, Cox PH, 5-year MACE.

We **do not claim 0.934 AUC**. That figure is tied to private AMI data and the original training recipe.

## What this codebase can verify publicly

| Table | Question | Data | Metrics | Claim status |
|-------|----------|------|---------|--------------|
| **T1** Reconstruction | Sparse SA → dense cine/3D | ACDC, MM-WHS (or synthetic smoke) | PSNR, global-SSIM proxy, MAE | Public, compact tensors by default |
| **T2** Segmentation | 4-class LV/RV/MYO | ACDC, MM-WHS | Dice per class, optional HD95 | Public; not full nnU-Net 256³ |
| **T3** Infarct phenotyping | Disease / scar proxies | ACDC 5-class + MINF; EMIDEC scar | Accuracy / AUC on phenotype or MINF | **Proxy only**, not 5-year MACE |
| **T4** Motion-consistent 4D | Per-frame recon + warp | Synthetic / ACDC 4D | Warp L1, cycle error, volume-curve, EF proxy | **Main novelty of this repo** |

Private AMI Cox / HeartTTable 5-year MACE remains **out of scope** until the user attaches a de-identified manifest (`data.source: nifti`) and reports C-index with nested validation. Do not copy the original AUC.

## Implementation vs paper-scale

| Component | Paper | This repo (default) |
|-----------|-------|---------------------|
| Grid | 256×256×128 | `(D,H,W)=(16,32,32)` smoke; config comments show paper size |
| Recon backbone | 3D ViT | Compact 3D CNN + optional transformer bottleneck (`volume_recon.py`) |
| Seg backbone | 3D nnU-Net | Compact 3D UNet (`volume_seg.py`) + joint ReconSeg3D heads |
| Time | Broadcast recon (limitation of the previous prototype) | **Per-frame decode + motion warp** (default); cycle loss is **image-cycle** (fwd/bwd warps), not coordinate-composed inverse-consistent flow |
| Risk | Cox PH on private AMI | BCE MACE (synthetic) / phenotype (ACDC) / Cox loss (API ready) |
| Multimodal fusion | HeartTTable: 3 modality Transformers + pairwise class-token cross-attn | **HeartTTable-lite**: single CLS over concatenated spatial/temporal/table KV; uses `HeartTTable.risk_logits` when `fusion: heart_ttable`. Do **not** claim parity with the paper fusion module. |

## Demo paper pipeline (no private AMI)

```powershell
cd E:\Projects\20260523-ReconSeg3D-cardiac-rec
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

# 1) Fake ACDC + MM-WHS + EMIDEC under data/
python scripts/prepare_demo_data.py

# 2) Paper configs (1 epoch smoke)
python scripts/train.py --config configs/paper_recon.yaml --epochs 1 --output-dir outputs/paper_recon_smoke
python scripts/eval.py --config configs/paper_recon.yaml --checkpoint outputs/paper_recon_smoke/last.pt
python scripts/train.py --config configs/paper_acdc.yaml --epochs 1 --output-dir outputs/paper_acdc_smoke
python scripts/eval.py --config configs/paper_acdc.yaml --checkpoint outputs/paper_acdc_smoke/last.pt

# 3) Ablation matrix (broadcast / per_frame±motion / fusion / phenotype)
python scripts/run_ablations.py --epochs 1 --output-root outputs/ablations

# 4) Table from metrics.json (T1–T4 columns when present)
python scripts/summarize_runs.py --runs-root outputs/ablations --out-md outputs/ablations/table.md --out-csv outputs/ablations/table.csv
```

Longer runs: raise `--epochs` (configs default to 2–20) or edit `train.epochs` / `train.early_stop_patience`.

### Ablation configs (`configs/ablations/`)

| Config | Intent |
|--------|--------|
| `broadcast_baseline.yaml` | `per_frame_recon: false` |
| `per_frame_no_motion.yaml` | per-frame, `w_warp`/`w_cycle` = 0 |
| `per_frame_motion.yaml` | full warp + cycle |
| `fusion_concat.yaml` | concat MLP fusion |
| `fusion_heart_ttable.yaml` | HeartTTable-lite fusion |
| `task_phenotype.yaml` | fake/real ACDC phenotype |

Subset example:

```powershell
python scripts/run_ablations.py --epochs 1 --only broadcast_baseline per_frame_motion task_phenotype
```

If `data.root` for ACDC/MM-WHS/EMIDEC is empty and `data.auto_fake: true` (default on public loaders), training auto-calls `make_fake_*`. Prefer `prepare_demo_data.py` once for a shared tree.

## Suggested run order

1. Smoke: `python scripts/train.py --config configs/default.yaml --epochs 2`
2. Sparse recon+seg+motion: `configs/paper_recon.yaml` (synthetic or swap `data.source`)
3. ACDC phenotype: `configs/paper_acdc.yaml` after `prepare_demo_data.py` or a real ACDC root
4. Ablations: `scripts/run_ablations.py`
5. EMIDEC scar stub: `data.source: emidec` (see `docs/DATA.md`)
6. Private AMI: `data.source: nifti`, `model.fusion: heart_ttable`, `model.task: cox`

## Honest gaps for a paper

- No original weights, no private AMI cohort, no nested CV / calibration plots yet.
- Compact backbones are **interface-compatible**, not compute-matched to 3D ViT / nnU-Net.
- HD95 uses a subsampled numpy surface distance (not MONAI/SimpleITK).
- SSIM is a **global** proxy, not a 3D windowed SSIM (stabilizers scale with data range).
- Volume-curve / `w_volsmooth` uses **fractional** LV/RV volumes (counts / DHW), not raw voxel counts.
- Trainer pools MACE/phenotype AUC and C-index over the full epoch (not mean-of-batch-AUC); tiny fake splits can still look extreme.
- `best.pt` selection is task-aware: phenotype → phenotype_acc/auc, cox → c_index, else mace_auc (fallback: val loss).
- Slice-sampling is a simulation of sparse SA, not vendor undersampled k-space.
- Clinical table tokens need a real feature dictionary (labs, meds, ECG) before HeartTTable is more than a fusion ablation.
- Demo AUC / Dice on fake NIfTIs are **pipeline checks only** — do not cite as clinical performance.
