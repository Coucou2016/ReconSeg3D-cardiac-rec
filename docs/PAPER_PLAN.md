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
| `L_loop` | Full-cycle adjacent composition ≈ Id (coarse for short T) |
| Image-cycle `w_cycle` | **Auxiliary** intensity round-trip only |
| `w_volsmooth` | **Demoted** physiological volume-curve regularizer |
| MACE / Cox / HeartTTable | **Supplementary / extensibility** (`w_mace:0`, `w_cox:0` in `configs/publication_*.yaml`) |

Configs: `configs/publication_recon.yaml`, `publication_motion.yaml`, `publication_seg.yaml`.

### Deformation convention

Displacement channels `(dz, dy, dx)` in voxels → `grid_sample` with `align_corners=True`, scale `2/max(dim-1,1)`. Pull composition: `compose_pull(u,v) = v + W(u,v)`. ED/ES indices from ACDC `Info.cfg` (no silent `t//2`); unlabeled frames use motion self-supervision only.

## What this codebase can verify publicly

| Table | Question | Data | Metrics | Claim status |
|-------|----------|------|---------|--------------|
| **T1** Reconstruction | Sparse SA → dense cine/3D | ACDC, MM-WHS (or synthetic smoke) | PSNR, **global-SSIM proxy**, MAE | Public; do not claim windowed SSIM yet |
| **T2** Segmentation | 4-class LV/RV/MYO at ED/ES | ACDC | Dice; HD95 with physical spacing when known | Public; ED/ES supervised only |
| **T3** Motion / label prop | ED↔ES warp of GT masks | ACDC / synthetic | Prop Dice + HD95 (mm) | API ready; real-table **待补充** if only fake ACDC |
| **T4** Geometry regularizers | inv / smooth / jac / loop | Synthetic / ACDC 4D | inv_error, jac_neg_ratio, loop | Implemented |
| **Supp.** Phenotype / Cox / MACE | Proxy risk | ACDC phenotype; private AMI | Acc / C-index | Not main line |

## Implementation vs paper-scale

| Component | Paper | This repo (default) |
|-----------|-------|---------------------|
| Grid | 256×256×128 | `(D,H,W)=(16,32,32)` smoke |
| Motion | — | Inverse-consistent pull fields + smooth/jac/loop; image-cycle auxiliary |
| SVF | — | **TODO:** `model.use_svf` stub + scaling-and-squaring in `MotionNet` (off by default) |
| Risk | Cox on private AMI | Off in publication configs |
| Fusion | HeartTTable full | HeartTTable-lite ablation only |

## Tracked TODOs (deferred)

- [ ] Full VoxelMorph / TransMorph / MulViMotion baselines on licensed ACDC
- [ ] M&Ms download + multi-site tables
- [ ] 5-seed CI for paper tables
- [ ] Enable / ablate SVF (`use_svf: true`) with scaling-and-squaring
- [ ] ED-reference motion path (warp all frames to ED) beyond adjacent+loop
- [ ] Windowed 3D SSIM if claiming SSIM in main tables
- [ ] Nested CV / calibration for any future survival claims

## Demo paper pipeline (no private AMI)

```powershell
cd E:\Projects\20260523-ReconSeg3D-cardiac-rec
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python scripts/prepare_demo_data.py
python scripts/train.py --config configs/publication_motion.yaml --epochs 1 --output-dir outputs/pub_motion_smoke
python scripts/train.py --config configs/paper_recon.yaml --epochs 1 --output-dir outputs/paper_recon_smoke
```

Publication configs with `allow_fake_data: false` **hard-error** if ACDC/MM-WHS/EMIDEC roots are empty.

## Honest gaps

- No original weights, no private AMI cohort.
- Global SSIM is a **proxy**, not windowed SSIM.
- HD95 supports physical spacing `(sz,sy,sx)` mm; demo often voxel units.
- Fake/demo metrics remain **DEMO-labeled** — do not invent real-table numbers.
- Volume-curve is a light physiological regularizer, not a primary motion claim.
