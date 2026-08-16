# Motion-consistent 4D cardiac reconstruction and segmentation on public CMR proxies: a methods extension of ReconSeg3D

**Working manuscript draft (methods paper)**  
**Status:** Preprint-ready scaffold with smoke/demo results only. Clinical MACE claims deferred.  
**Target genre:** Nature-family / MedIA-style methods · English  

---

## Abstract

Sparse short-axis (SA) cine cardiac magnetic resonance (CMR) stacks undersample the three-dimensional anatomy of the beating heart, motivating dense spatiotemporal reconstruction before structure-aware analysis. Recent multimodal work (ReconSeg3D + HeartTTable) showed that reconstructed bi-ventricular cine volumes can support long-horizon major adverse cardiovascular event (MACE) prediction on a large private acute myocardial infarction (AMI) cohort, but that clinical performance figure (5-year time-dependent AUC 0.934) is not reproducible without the original data and training recipe. Here we present a **public, motion-consistent 4D methods extension** of a lightweight ReconSeg3D-style codebase: per-frame volumetric decoding, differentiable warp regularization, and an **image-cycle** consistency loss (forward/backward warps of image intensities—not coordinate-composed inverse-consistent flow), jointly trained with multi-structure segmentation and optional phenotype or risk heads. HeartTTable-style fusion is retained as **HeartTTable-lite** (a single class token attending to concatenated spatial/temporal/table keys), used only as a fusion ablation. On synthetic and auto-fake public-proxy smoke runs we verify that the pipeline logs reconstruction (PSNR/MAE), segmentation Dice, and motion proxies (warp L1, image-cycle error, ejection-fraction proxy). We do **not** claim private-AMI MACE parity. Real ACDC / MM-WHS / EMIDEC subject-level tables remain **待补充** pending mounted datasets.

**Keywords:** cardiac MRI; 4D reconstruction; motion consistency; multi-task learning; public benchmarks; methods reproducibility

---

## 1. Introduction

Cardiac cine magnetic resonance imaging (CMR) is typically acquired as a stack of two-dimensional short-axis (SA) slices. Slice gaps, through-plane motion, and temporal undersampling leave a sparse observation of bi-ventricular anatomy. Dense three-dimensional (3D) reconstruction across the cardiac cycle—effectively a four-dimensional (4D) volume—can improve anatomical continuity for segmentation and downstream phenotyping.

Gao et al. (*npj Digital Medicine*, 2026) introduced ReconSeg3D to reconstruct temporally resolved bi-ventricular volumes from SA stacks and HeartTTable to fuse imaging with tabular clinical variables for 5-year MACE after percutaneous coronary intervention in AMI[^1]. That study’s headline discrimination metrics are tied to a private cohort of thousands of patients and are therefore **out of scope** for any local reimplementation that lacks those data.

**Gap.** Lightweight public reimplementations often broadcast a single reconstructed volume across time or omit explicit motion constraints, which weakens claims about “spatiotemporal” fidelity even when reconstruction and segmentation heads are present.

**Contribution.** We provide and evaluate a motion-consistent 4D training stack on public proxies:

1. Per-frame 3D reconstruction with temporal mixing.
2. MotionNet warp losses plus image-cycle consistency, with optional LV/RV volume-curve smoothing and an ejection-fraction (EF) proxy.
3. Compact segmentation and phenotype/risk heads suitable for ACDC-style proxies.
4. HeartTTable-lite as a fusion ablation only—**not** a claim of pairwise three-modality class-token cross-attention parity.
5. An honest claim boundary and smoke ablation matrix for reconstruction, segmentation, and motion metrics.

---

## 2. Related work

**Reconstruction.** Deep learning has become central to accelerated and robust MRI reconstruction[^2]. Cardiac applications increasingly couple reconstruction with motion or segmentation rather than treating them in isolation.

**Joint recon–motion–seg.** Unified frameworks that estimate groupwise motion while refining reconstruction and segmentation demonstrate benefits of inter-task dependency[^3]. Decoupled motion–shape models address sparse-slice 4D myocardium reconstruction[^4]. Memory-based networks improve whole-sequence 4D cine segmentation by exploiting spatiotemporal continuity[^5].

**Multimodal survival.** Neural Cox models such as DeepSurv generalize linear proportional-hazards risk scores[^6]. HeartTTable combines spatiotemporal imaging tokens with tabular variables via transformer cross-attention on private AMI data[^1]. Our public codebase exposes a Cox loss API and a lite fusion module but does not reproduce that clinical study.

---

## 3. Methods

### 3.1 Problem setup

Let \(x \in \mathbb{R}^{B \times C \times T \times D \times H \times W}\) denote a cine volume batch (channels \(C\), frames \(T\), depth \(D\), height \(H\), width \(W\)). Sparse SA sampling is simulated by zeroing unobserved slices. The model predicts a dense reconstruction \(\hat{x}\), optional flow fields for warping consecutive frames, multi-class segmentations (LV, RV, myocardium; optional scar), and task heads for phenotype classification, binary MACE (synthetic), or Cox partial likelihood (when event times are available).

### 3.2 Per-frame reconstruction

Unlike broadcasting a single decoded volume across \(T\), the default path encodes each frame with a compact 3D CNN (optional transformer bottleneck) and decodes per-frame volumes. Temporal mixing couples neighboring frames without collapsing the time axis.

### 3.3 Motion consistency

A MotionNet predicts voxel displacements \((d_z, d_y, d_x)\). Displacements are scaled to grid coordinates with dimension guards for singleton axes. We regularize:

- **Warp L1** between warped source and target intensities.
- **Image-cycle** error: compose forward and backward warps on images and penalize deviation from the original (distinct from inverse-consistent flow-field composition).
- Optional **volume-curve** smoothness on fractional LV/RV volumes and an **EF proxy**.

### 3.4 Segmentation and task heads

A compact 3D UNet-style head predicts multi-class masks. Phenotype (ACDC-style), BCE MACE, or Cox risk can be enabled via configuration. Metrics pool at epoch level; tiny fake splits can yield unstable AUC / C-index and must not be over-interpreted.

### 3.5 HeartTTable-lite (ablation)

When `fusion: heart_ttable`, a single CLS token attends over concatenated spatial, temporal, and table key–value streams (`HeartTTable.risk_logits`). This is **not** the original three class-token pairwise cross-attention design and is reported only as a fusion ablation against concat-MLP fusion.

### 3.6 Training protocol (smoke)

Configs under `configs/` set compact grids (e.g., \(16\times32\times32\)) and 1–2 epoch smoke runs. Public loaders may auto-synthesize fake NIfTI when roots are empty. Subject-level cross-validation, physical-space HD95, and calibration plots are specified for future real-data tables (**待补充**).

---

## 4. Experiments

### 4.1 Datasets

| Dataset | Role | Status in this draft |
|---------|------|----------------------|
| Synthetic / auto-fake | Pipeline verification | Used for smoke metrics |
| ACDC | Recon/seg/phenotype proxy | Mount path ready; real results 待补充 |
| MM-WHS | Static seg / table fusion | Real results 待补充; motion off / \(T{=}1\) recommended |
| EMIDEC | Scar / infarct proxy | Stub; 待补充 |
| Private AMI | 5-year MACE | Out of scope |

### 4.2 Ablations

Broadcast baseline; per-frame without motion weights; per-frame with warp/cycle; concat vs HeartTTable-lite; phenotype task. Summaries written by `scripts/summarize_runs.py`.

### 4.3 Metrics

Reconstruction: PSNR, global-SSIM proxy, MAE. Segmentation: per-class Dice; optional subsampled HD95. Motion: warp L1, image-cycle error, EF proxy. Phenotype/MACE/C-index when applicable—**demo values are not clinical**.

---

## 5. Results

### 5.1 Smoke ablation summary (demo only)

Quantitative panels in Figures 2–4 are generated with SciencePlots from `outputs/ablations_smoke_v2/table.csv`. Representative reconstruction PSNR on the three motion-relevant runs is approximately 19.1 dB with MAE ≈ 0.69 under the compact fake-data regime. Motion runs log non-zero warp and image-cycle errors; EF proxies remain small on synthetic volumes. Segmentation Dice on fake NIfTI is low and class-imbalanced (RV often near zero)—consistent with under-trained smoke rather than a clinical ceiling.

**Phenotype smoke:** logged phenotype accuracy 0.0 with AUC 1.0 on a tiny fake split illustrates metric instability; we treat this as a **pipeline check**, not a result.

**Paper recon / ACDC smoke checkpoints** similarly show low Dice and mid-teens–19 dB PSNR after one epoch (see `outputs/paper_*_smoke/metrics.json`).

### 5.2 Figures

- **Fig. 1** Pipeline schematic (roles of recon, motion, seg, lite fusion).
- **Fig. 2** Reconstruction PSNR/MAE ablation.
- **Fig. 3** Warp / image-cycle / EF proxies.
- **Fig. 4** Dice heatmap across ablations.
- **Fig. 5** Claim-boundary diagram vs original clinical paper.

Real ACDC cine volume curves, confidence intervals, and physical HD95: **待补充**.

---

## 6. Discussion

The practical contribution of this codebase is a **reproducible motion-consistent 4D training contract** on public proxies, with documentation that prevents silent overclaim of the original AMI discrimination numbers. Image-cycle naming matters for peer review: reviewers familiar with registration literature will expect inverse consistency of flow fields; we instead regularize intensity cycles after warping.

HeartTTable-lite is intentionally under-powered relative to the original multimodal module. Building full three-CLS pairwise attention before real AMI data would optimize for a claim we cannot evaluate publicly.

**Limitations.** Compact backbones; simulated sparse SA (not vendor k-space); approximate HD95/SSIM; no nested CV; no private AMI; Chinese/English bilingual materials in the companion research report for local review, while this draft remains English for Nature-family methods style.

---

## 7. Conclusions

We describe a methods-oriented, motion-consistent 4D extension of a ReconSeg3D-style pipeline for public cardiac MRI proxies, with HeartTTable-lite as a fusion ablation and an explicit refusal to claim private AMI AUC 0.934. Smoke ablations demonstrate that reconstruction, segmentation, and motion metrics are logged end-to-end. Completing subject-level public-data tables and—only with authorized private cohorts—survival evaluation remain future work.

---

## Data availability

Public benchmarks (ACDC, MM-WHS, EMIDEC) follow their licenses (`docs/DATA.md`). Smoke metrics and SciencePlots figure sources live under `outputs/` and `artifacts/paper/figures/`. No private AMI data are distributed with this repository.

## Code availability

Local research codebase under the project root (no public remote required for this draft). Install via `requirements.txt` (includes SciencePlots for figure regeneration).

## References

[^1]: Gao, Q. et al. 3D Spatiotemporal cardiac reconstruction for predicting MACE in acute myocardial infarction. *npj Digit. Med.* **9**, 325 (2026). https://doi.org/10.1038/s41746-026-02449-0

[^2]: Hammernik, K. et al. Deep learning for accelerated and robust MRI reconstruction. *MAGMA* (2024). https://doi.org/10.1007/s10334-024-01173-8

[^3]: Qian, P. et al. Unified Deep Learning for Simultaneous Cardiac Cine MRI Reconstruction, Motion Estimation and Segmentation. *IEEE ISBI* (2024). https://doi.org/10.1109/ISBI56570.2024.10635390

[^4]: Yuan, X. et al. 4D Myocardium Reconstruction with Decoupled Motion and Shape Model. *ICCV* (2023).

[^5]: Continuous Spatio-Temporal Memory Networks for 4D Cardiac Cine MRI Segmentation. arXiv:2410.23191 (2024).

[^6]: Katzman, J. L. et al. DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. *BMC Med. Res. Methodol.* **18**, 24 (2018). https://doi.org/10.1186/s12874-018-0482-1

---

## Notes (nature-writing)

**Claim-evidence map**

| Claim | Evidence | Status |
|-------|----------|--------|
| Motion-consistent 4D stack exists and logs metrics | Code + smoke tables + Figs 2–3 | Supported (demo regime) |
| Public clinical superiority vs baselines | Real ACDC folds | Needs evidence |
| HeartTTable parity / AMI AUC 0.934 | — | Out of scope / rejected |

**Assumptions or missing inputs:** real ACDC/MM-WHS/EMIDEC on disk; physical HD95; nested CV; private AMI for survival claims.

**Why this structure:** methods argument chain (problem → method → ablation → boundary); clinical multimodal paper deferred.
