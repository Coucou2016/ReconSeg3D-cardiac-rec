# Evidence authenticity audit / 证据真实性审计

**Purpose / 目的:** One-to-one mapping between manuscript numbers and local measured artefacts.  
文稿数字与本地实测产物的一一对应。

**Date / 日期:** 2026-11-07  

**Manuscript spines / 文稿脊柱:**
- English: `docs/paper/MANUSCRIPT_FINAL.md` → `docs/paper/MANUSCRIPT_FINAL.docx`
- Chinese (primary deliverable this pass): `docs/paper/MANUSCRIPT_FINAL_ZH.md` → `docs/paper/MANUSCRIPT_FINAL_ZH.docx`  
  Built via `scripts/build_manuscript_docx_zh.py` with heading-based **目录** (static TOC + Word TOC field). Five SciencePlots PNGs from `artifacts/paper/figures/` embedded. DEMO metrics only — no 0.934 / 0.897.

---

## 1. Hard non-claims / 硬性不主张（never from original npj private AMI）

| Quantity | Original paper (Gao et al., *npj Digit. Med.*) | This repo / manuscript |
|----------|-----------------------------------------------|-------------------------|
| 5-year time-dependent AUC **0.934** | Private AMI external test | **Not used as a result** |
| C-index **0.897** | Private AMI | **Not used as a result** |
| Cohort n≈4511 / external n≈1029 | Private | **Not used** |
| Full HeartTTable three-modality pairwise cross-attn | Clinical product | **Not claimed** (HeartTTable-lite ablation only) |
| ACDC/MM-WHS challenge leaderboard Dice | Original training context | **No invented ACDC clinical tables** |

Boundary language may mention that private AMI discrimination exists in prior clinical work. Those headline scores are never copied into Results.

---

## 2. DEMO vs real labels / 演示与真实标签

| Artefact | Label | How decided |
|----------|-------|-------------|
| `outputs/ablations_smoke_v2/table.csv` | **DEMO** | Short smoke on fake/demo tensors |
| `outputs/metrics.json` | **DEMO** | Joint eval on demo tensors (geometry keys present) |
| `outputs/paper_recon_smoke/metrics.json` | **DEMO** | Named paper smoke run |
| `outputs/paper_acdc_smoke/metrics.json` | **DEMO** | Local ACDC-**named** placeholder tree, not CREATIS |
| `outputs/post_fix_smoke/metrics.json` | **DEMO** | Legacy pre-geometry-key logging era |
| `data/acdc/`, `data/mmwhs/`, `data/emidec/` | **DEMO placeholders** | Tiny synthetic volumes |
| Licensed ACDC subject CV tables | **Absent from measured set** | No licensed mount |
| Private AMI MACE | **Out of scope** | No data |

---

## 3. Numbers in MANUSCRIPT_FINAL(_ZH) → source files → code

### 3.1 Ablation table (PSNR / MAE / Dice mean) — Results §5.1 / Table 2 / Fig. 2

| MS cell (rounded) | Exact source | Computing path |
|-------------------|--------------|----------------|
| broadcast_baseline 19.08 / 0.693 / 0.168 | PSNR=19.075101, MAE=0.692593, dice_mean=0.168156 | `scripts/run_ablations.py` → train/eval → metrics |
| per_frame_no_motion 19.15 / 0.692 / 0.170 | 19.150563 / 0.691882 / 0.169577 | same |
| per_frame_motion 19.15 / 0.695 / 0.159 | 19.147806 / 0.695264 / 0.158844 | same |
| fusion_concat 19.09 / 0.692 / 0.139 | 19.092749 / 0.692152 / 0.139436 | same |
| fusion_heart_ttable 19.26 / 0.689 / 0.189 | 19.260390 / 0.689495 / 0.189020 | same |
| Myocardium Dice 0.567 (HeartTTable-lite) | `dice_myo` in same CSV | same |
| Fig. 2 / Fig. 4 | `scripts/make_paper_figures.py` | reads same CSV |

File: `outputs/ablations_smoke_v2/table.csv`

### 3.2 Geometry logs — Results §5.2 / Table 3 / Fig. 3

| MS metric | Source key | File | Code |
|-----------|------------|------|------|
| Inverse-consistency 0.00177 | `inv_error` / `loss_inv` = 0.001768695 | `outputs/metrics.json` | `reconseg3d/models/motion.py::inverse_consistency_loss` |
| Cycle composition 0.00177 | `loss_loop` = 0.001768317 | same | `loop_consistency_loss` |
| Smoothness 6.1e-8 | `loss_smooth` = 6.127958585e-8 | same | `smoothness_loss` |
| Folding penalty 0.0 | `loss_jac` = 0.0 | same | `folding_penalty` |
| Neg Jac ratio 0.0 | `jac_neg_ratio` | same | `jacobian_stats` |
| Mean Jac det 0.9998 | `jac_det_mean` = 0.999822199 | same | `jacobian_stats` |
| Image-cycle 2.3e-5 | `cycle_error` = 2.339757e-5 | same | intensity cycle aux (not \(L_\mathrm{inv}\)) |
| PSNR 18.80 | `recon_psnr` = 18.795277 | same | recon metrics |
| `ssim_3d` 0.515 | `ssim_3d` = 0.514883 | same | windowed SSIM |
| `recon_ssim_proxy` 0.0098 | `recon_ssim_proxy` = 0.009767781 | same | global proxy |
| ED↔ES prop Dice 1.0 | `prop_ed2es_dice_mean`, `prop_es2ed_dice_mean` | same | label propagation API |

### 3.3 Named checkpoints — Results §5.4 / Table 4

| MS row | File | Exact keys |
|--------|------|------------|
| paper_recon_smoke | `outputs/paper_recon_smoke/metrics.json` | PSNR=19.027001, dice_mean=0.0480199, warp_error=0.000293835, cycle_error=0.000464230 |
| paper_acdc_smoke | `outputs/paper_acdc_smoke/metrics.json` | PSNR=17.389618, dice_mean=0.0175126, warp_error=8.57433e-5, cycle_error=8.33719e-5, hd95_lv=15.684387 |

Eval entry: `scripts/eval.py`.

### 3.4 Supplementary ablation spreads — Results §5.5

Derived from the same CSV (no new invention):

- PSNR span 19.075–19.260 → 0.18 dB
- MAE span 0.6895–0.6953
- Dice mean span 0.1394–0.1890
- Synthetic `mace_auc`/`c_index` fields (e.g. 0.625 / 0.5556) logged on some arms → manuscript states pipeline-only, excluded from clinical claim

### 3.5 Figures

| Figure | Generator | Data |
|--------|-----------|------|
| Fig. 1 schematic | `scripts/make_paper_figures.py` | Non-quantitative |
| Fig. 2–4 | same | `ablations_smoke_v2/table.csv` (+ Fig. 3b geometry JSON) |
| Fig. 5 claim boundary | same | Qualitative ledger |
| Outputs | `artifacts/paper/figures/*.{png,pdf,svg}` | SciencePlots `science`+`nature`+`no-latex`; Times New Roman |
| Meta | `artifacts/paper/figures/figure_meta.json` | |

### 3.6 Methods ↔ code correspondence

| Manuscript statement | Code symbol |
|----------------------|-------------|
| Pull compose \(u\circ v=v+W(u,v)\) | `compose_pull` |
| True \(L_\mathrm{inv}\) | `inverse_consistency_loss` |
| Smoothness | `smoothness_loss` |
| Jacobian folding | `folding_penalty`, `jacobian_determinant` |
| Periodic composition with return step T-1→0 | `loop_consistency_loss` (T pairs, last closes) |
| ED-anchored composed path | `ed_reference_consistency_loss`, `ed_anchored_paths` |
| Linear ED paths strip closing edge | `adjacent_flow_stack` |
| Loss weights | `reconseg3d/models/losses.py` (`w_inv`, `w_smooth`, `w_jac`, `w_loop`, `w_ed_ref`) |
| ED/ES labeled-phase seg | `labeled_phase_seg_loss` |
| Publication gates (MACE/Cox zero) | publication YAML + `w_mace`/`w_cox` defaults |

---

## 4. Commands / 再生命令

```text
python scripts/make_paper_figures.py
python scripts/build_manuscript_docx.py
python scripts/build_manuscript_docx_zh.py
python -m pytest tests/test_motion.py tests/test_metrics.py tests/test_label_propagation.py -q --assert=plain
# with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 on hosts with plugin noise
```

**OA reference PDFs (structure imitation only / 仅结构仿写，不抄正文):**

- `artifacts/paper/references/voxelmorph_balakrishnan2019.pdf` (TMI OA-style / IEEE)
- `artifacts/paper/references/krebs2020_latent_motion_matrix.pdf` (TMI)
- `artifacts/paper/references/deeptag_ye2021.pdf` (CVPR)
- `artifacts/paper/references/groupwise_dynamic_mri_frontiers2022.pdf` (Frontiers OA)
- `artifacts/paper/references/dalca2019_diffeomorphic.pdf` (MedIA)

---

## 5. Paper vs report separation / 论文与过程报告分离

| Deliverable | Paths | Rules |
|-------------|-------|-------|
| Academic paper (ZH primary) | `docs/paper/MANUSCRIPT_FINAL_ZH.md`, `MANUSCRIPT_FINAL_ZH.docx` | No local Windows paths; DEMO labels; no session tone |
| Academic paper (EN) | `docs/paper/MANUSCRIPT_FINAL.md`, `MANUSCRIPT_FINAL.docx` | same |
| Process report | `artifacts/report/*` | Workspace paths OK |
| This audit | `docs/paper/EVIDENCE_AUDIT.md` | Authenticity ledger with paths |

中文镜像说明亦见 `docs/paper/EVIDENCE_AUDIT_ZH.md`（若存在则与本文件同步）。

---

## 6. Sync checklist / 同步清单

- [x] MS tables cite only DEMO sources listed above
- [x] Figs regenerable from same CSV/JSON
- [x] Original 0.934 / 0.897 excluded from Results
- [x] No “待补充/待确定/待确认” wording in manuscript body
- [x] Chinese Word built with 目录 (static + TOC field)
- [x] Five figures embedded
- [ ] Licensed ACDC subject tables (blocker: no licensed mount)
- [ ] Full pytest green on clean env (host `numcodecs`/`zarr` noise possible)
