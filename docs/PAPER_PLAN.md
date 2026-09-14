# Paper plan (living)

This repo is a **research re-implementation + extension**, not a drop-in reproduction of the private AMI numbers in the original npj Digital Medicine paper.

## What the original paper did

1. **ReconSeg3D (public MM-WHS + ACDC):** 3D ViT reconstruction of dense volumes from sparse short-axis stacks, then 3D nnU-Net segmentation (LV / RV / LVM).
2. **HeartTTable (private AMI):** spatial + temporal + table transformers with class-token cross-attention, Cox PH, 5-year MACE.

We **do not claim 0.934 AUC**. That figure is tied to private AMI data and the original training recipe.

## Main publication line (this revision)

**Geometry- and motion-constrained 4D reconstruction/segmentation from sparse SA cine**, with:

| Term | Role |
|------|------|
| `L_inv` | True inverse consistency `‖u+W(v,u)‖ + ‖v+W(u,v)‖` (same pull grid as image warp) |
| `L_smooth` | `‖∇u‖²` |
| `L_jac` | `ReLU(ε − det J)` folding penalty |
| `L_periodic` / `L_loop` | **Closed-cycle** composition including edge `T-1→0` ≈ Id |
| `L_ed_ref` | ED-anchored composed-path inverse consistency (`φ_{k→ED}` vs `φ_{ED→k}`) |
| Image-cycle `w_cycle` | **Auxiliary** intensity round-trip only |
| `w_volsmooth` | **Demoted** physiological volume-curve regularizer |
| MACE / Cox / HeartTTable | **Supplementary / extensibility** (`w_mace:0`, `w_cox:0` in publication configs) |

Configs:
- Publication: `configs/publication_*.yaml` (also mirrored under `configs/publication/`)
- Smoke: `configs/smoke_motion.yaml` (also `configs/smoke/`)
- Phenotype proxy: `configs/proxy/proxy_acdc_phenotype.yaml` (`clinical_dim: 3`; shim `configs/paper_acdc.yaml`)

All publication configs set `allow_fake_data: false` and honest `task:` (`reconstruction` / `motion` / `segmentation`) with `selection.metric` / `mode` for checkpointing.

### Deformation convention

Displacement channels `(dz, dy, dx)` in voxels → `grid_sample` with `align_corners=True`, scale `2/max(dim-1,1)`. Pull composition: `compose_pull(u,v) = v + W(u,v)`.

**Closed-cycle MotionNet:** predicts **T** pairs for `T` frames (adjacent `0..T-2` plus closing `T-1→0`). Do **not** call adjacent-only (T-1) compose a "full cycle".

**ED-anchored path:** left-fold **adjacent** fields into `φ_{k→ED}` / `φ_{ED→k}`; apply the same inverse-consistency residual (`w_ed_ref`). ED/ES from ACDC `Info.cfg`; temporal subsample uses `select_time_indices_keep_anchors` so original ED/ES frames are always kept (prefer full-T + temporal mask when GPU allows).

**Clinical inputs:** ACDC `clinical_base` is height/weight/nb only — phenotype/Group is never a feature (proxy configs only).

### Checkpoint selection

| task | default metric | mode |
|------|----------------|------|
| motion | `prop_ed2es_dice_mean` | max (fallback `loss_total` min) |
| segmentation | `dice_mean` | max |
| reconstruction | `recon_mae` | min |
| joint | `loss_total` | min |

Override with `selection.metric` / `selection.mode` in YAML.

## What this codebase can verify publicly

| Table | Question | Data | Metrics | Claim status |
|-------|----------|------|---------|--------------|
| **T1** Reconstruction | Sparse SA → dense cine/3D | ACDC, MM-WHS (or synthetic smoke) | PSNR, **global-SSIM proxy** (`recon_ssim_proxy`), MAE | Public; do not claim windowed SSIM yet |
| **T2** Segmentation | 4-class LV/RV/MYO at ED/ES | ACDC | Dice; HD95 with batch `spacing` when present | Public; ED/ES supervised only |
| **T3** Motion / label prop | ED↔ES warp of GT masks | ACDC / synthetic | Prop Dice + HD95 (mm), **per-patient** then aggregate | API ready; real-table **待补充** if only fake ACDC |
| **T4** Geometry regularizers | inv / smooth / jac / periodic | Synthetic / ACDC 4D | inv_error, jac_neg_ratio, loop | Implemented (true closed cycle) |
| **Supp.** Phenotype / Cox / MACE | Proxy risk | ACDC phenotype; private AMI | Acc / C-index | Not main line |

## Implementation vs paper-scale

| Component | Paper | This repo (default) |
|-----------|-------|---------------------|
| Grid | 256×256×128 | `(D,H,W)=(16,32,32)` smoke |
| Motion | — | Closed-cycle inverse-consistent pull fields + smooth/jac/periodic; image-cycle auxiliary |
| SVF | — | **TODO:** `model.use_svf` stub + scaling-and-squaring in `MotionNet` (off by default) |
| Risk | Cox on private AMI | Off in publication configs |
| Fusion | HeartTTable full | HeartTTable-lite ablation only |

## Tracked TODOs (deferred)

- [ ] Full VoxelMorph / TransMorph / MulViMotion / FlowReg baselines on licensed ACDC
- [ ] M&Ms download + multi-site tables
- [ ] 5-seed CI + official 5-fold CV file lists for paper tables
- [ ] Enable / ablate SVF (`use_svf: true`) with scaling-and-squaring
- [x] ED-reference motion path (adjacent + ED-anchored `w_ed_ref`) — implemented 2026-09-14
- [x] Closed-cycle `L_periodic` (T pairs incl. closing edge) — 2026-09-14 P0 round 2
- [x] Anchor-preserving temporal subsample + phenotype leakage fix — 2026-09-14
- [ ] Windowed 3D SSIM if claiming SSIM in main tables
- [ ] ConvLSTM temporal experiments
- [ ] 256³ / paper-scale resolution runs
- [ ] Nested CV / calibration for any future survival claims

## Demo paper pipeline (no private AMI)

```powershell
cd E:\Projects\20260523-ReconSeg3D-cardiac-rec
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python scripts/prepare_demo_data.py
python scripts/train.py --config configs/smoke_motion.yaml --epochs 1 --output-dir outputs/smoke_motion
python scripts/train.py --config configs/paper_recon.yaml --epochs 1 --output-dir outputs/paper_recon_smoke
```

Publication configs with `allow_fake_data: false` **hard-error** if `source: synthetic` or if ACDC/MM-WHS/EMIDEC roots are empty. Use `smoke_motion.yaml` for synthetic CI.

Eval: `scripts/eval.py` passes `seg_frame_indices` / ED/ES into the model and metrics. `--split test` currently maps to val-style held-out patients (**TODO:** official 5-fold test lists).

## Honest gaps

- No original weights, no private AMI cohort.
- Global SSIM is a **proxy** (`recon_ssim_proxy` only — no `recon_ssim` alias).
- HD95 uses batch `spacing` `(sz,sy,sx)` mm when present; demo NIfTIs often unit spacing.
- Fake/demo metrics remain **DEMO-labeled** — do not invent real-table numbers.
- Volume-curve is a light physiological regularizer, not a primary motion claim.
