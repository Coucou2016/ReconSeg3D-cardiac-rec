# Motion-consistent 4D reconstruction and segmentation for sparse short-axis CMR: a public methods extension of ReconSeg3D

**Working manuscript (methods)**  
**Status:** Publication-oriented draft. Quantitative panels report **demo-regime pipeline metrics only**. Subject-level public-benchmark tables remain **to be completed** after licensed dataset mounts.  
**Target genre:** Nature-family / Medical Image Analysis–style methods · English  
**Code availability:** https://github.com/Coucou2016/ReconSeg3D-cardiac-rec  

---

## Abstract

Sparse short-axis (SA) cine cardiac magnetic resonance (CMR) undersamples three-dimensional anatomy across the cardiac cycle, motivating dense spatiotemporal reconstruction before structure-aware analysis. Gao et al. (*npj Digital Medicine*, 2026) showed that reconstructed bi-ventricular cine volumes can support long-horizon major adverse cardiovascular event (MACE) prediction on a large private acute myocardial infarction (AMI) cohort; that clinical discrimination figure is tied to private data and training and is **not claimed here**. We present a public, motion-consistent four-dimensional (4D) methods extension of a lightweight ReconSeg3D-style stack: per-frame volumetric decoding, differentiable warp regularization, and an **image-cycle** consistency loss that penalizes intensity residual after forward and backward warps—distinct from coordinate-composed inverse-consistent flow. The stack is trained jointly with multi-structure segmentation and optional phenotype or risk heads. HeartTTable-style multimodal fusion is retained only as **HeartTTable-lite** (a single class token attending to concatenated spatial, temporal, and tabular keys) and reported as a fusion ablation. On controlled demo-scale public-proxy tensors we verify end-to-end logging of reconstruction (PSNR/MAE), segmentation Dice, and motion proxies (warp L1, image-cycle error, ejection-fraction proxy). Local challenge-named folders contain synthetic placeholders rather than licensed ACDC, MM-WHS, or EMIDEC volumes; real subject-level tables remain to be completed. We do not claim private-AMI MACE parity.

**Keywords:** cardiac MRI; 4D reconstruction; motion consistency; multi-task learning; public benchmarks; methods reproducibility

---

## 1. Introduction

Cardiac cine magnetic resonance imaging (CMR) is typically acquired as a stack of two-dimensional short-axis (SA) slices. Slice gaps, through-plane motion, and temporal undersampling leave a sparse observation of bi-ventricular anatomy. Dense three-dimensional (3D) reconstruction across the cardiac cycle—effectively a four-dimensional (4D) volume—can improve anatomical continuity for segmentation and downstream phenotyping.

Gao et al. (*npj Digital Medicine*, 2026) introduced ReconSeg3D to reconstruct temporally resolved bi-ventricular volumes from SA stacks and HeartTTable to fuse imaging with tabular clinical variables for five-year MACE after percutaneous coronary intervention in AMI[^1]. That study’s headline discrimination metrics are tied to a private cohort of thousands of patients and are therefore out of scope for any public reimplementation that lacks those data.

**Gap.** Lightweight public reimplementations often broadcast a single reconstructed volume across time or omit explicit motion constraints, which weakens claims about spatiotemporal fidelity even when reconstruction and segmentation heads are present. Concurrent lines of work emphasize whole-sequence continuity for 4D segmentation[^5], joint recon–motion–seg unrolling[^3], and mesh- or tetrahedra-based 4D heart recovery from sparse views[^7][^8], but a compact, auditable ReconSeg3D-style training contract with named image-cycle regularization and an honest multimodal ablation boundary remains useful for reproducible methods research.

**Contribution.** We provide and evaluate a motion-consistent 4D training stack on public proxies:

1. Per-frame 3D reconstruction with temporal mixing (default path; broadcast baseline retained for ablation).
2. Motion-network warp losses plus image-cycle consistency, with optional left-/right-ventricular volume-curve smoothing and an ejection-fraction (EF) proxy.
3. Compact segmentation and phenotype/risk heads suitable for ACDC-style phenotype proxies when real data are available.
4. HeartTTable-lite as a fusion ablation only—not a claim of pairwise three-modality class-token cross-attention parity.
5. An explicit claim boundary separating demo-regime pipeline verification from subject-level public benchmarks and from private AMI survival evaluation.

---

## 2. Related work

**Reconstruction.** Deep learning has become central to accelerated and robust MRI reconstruction[^2]. Cardiac applications increasingly couple reconstruction with motion or segmentation rather than treating them in isolation.

**Joint recon–motion–seg.** Unified frameworks that estimate groupwise motion while refining reconstruction and segmentation demonstrate benefits of inter-task dependency[^3]. Decoupled motion–shape models address sparse-slice 4D myocardium reconstruction[^4]. Memory-based networks improve whole-sequence 4D cine segmentation by exploiting spatiotemporal continuity[^5]. Complementary mesh and deformable-tetrahedra approaches recover 4D cardiac geometry from multi-view or sparse intraoperative observations[^7][^8].

**Multimodal survival.** Neural Cox models such as DeepSurv generalize linear proportional-hazards risk scores[^6]. HeartTTable combines spatiotemporal imaging tokens with tabular variables via transformer cross-attention on private AMI data[^1]. Our public codebase exposes a Cox loss API and a lite fusion module but does not reproduce that clinical study.

**Positioning.** Relative to Gao et al.[^1], this draft is a methods extension focused on motion-consistent training and public-proxy evaluation, not a clinical outcome paper. Relative to Qian et al.[^3], Ye et al.[^5], and recent 4D mesh/tetrahedra recoveries[^7][^8], we do not claim superior Dice or mesh accuracy on challenge leaderboards; we claim a transparent lightweight contract—per-frame reconstruction, warp and image-cycle regularization, and a lite fusion ablation—with reproducible demo-regime logs.

---

## 3. Methods

### 3.1 Problem setup

Let \(x \in \mathbb{R}^{B \times C \times T \times D \times H \times W}\) denote a cine volume batch (channels \(C\), frames \(T\), depth \(D\), height \(H\), width \(W\)). Sparse SA sampling is simulated by zeroing unobserved slices along the through-plane axis. The model predicts a dense reconstruction \(\hat{x}\), optional displacement fields for warping consecutive frames, multi-class segmentations (left ventricle, right ventricle, myocardium; optional scar), and task heads for phenotype classification, binary event prediction on synthetic labels, or Cox partial likelihood when event times are available.

### 3.2 Per-frame reconstruction

Unlike broadcasting a single decoded volume across \(T\), the default path encodes each frame with a compact 3D convolutional backbone (optional transformer bottleneck) and decodes per-frame volumes. Temporal mixing couples neighboring frames without collapsing the time axis. Compact demonstration grids (for example \(D{\times}H{\times}W = 16{\times}32{\times}32\) with \(T{=}8\)) are used for pipeline verification; paper-scale grids remain configuration options for future subject-level experiments.

### 3.3 Motion consistency

A motion network predicts voxel displacements \((d_z, d_y, d_x)\). Displacements are scaled to normalized grid coordinates with dimension guards for singleton axes. We regularize:

- **Warp intensity error** between warped source and target intensities (L1).
- **Image-cycle error:** compose forward and backward warps on **image intensities** and penalize deviation from the original frame. This is distinct from inverse-consistent composition of flow fields in classical registration; we name the term image-cycle to avoid reviewer misreading.
- Optional **volume-curve** smoothness on fractional left-/right-ventricular volumes and an **EF proxy** derived from end-diastolic and end-systolic fractional volumes.

### 3.4 Segmentation and task heads

A compact 3D U-Net–style head predicts multi-class masks. Phenotype (ACDC-style), binary event, or Cox risk heads can be enabled by configuration. Discrimination metrics are pooled at the epoch level; tiny synthetic splits can yield unstable area-under-curve or concordance estimates and must not be over-interpreted as clinical performance.

### 3.5 HeartTTable-lite (ablation)

When multimodal fusion is enabled in the lite setting, a single class token attends over concatenated spatial, temporal, and tabular key–value streams. This design is intentionally weaker than the original three class-token pairwise cross-attention module and is reported only as a fusion ablation against a concatenation–multilayer-perceptron baseline.

### 3.6 Training and evaluation protocol

Demonstration configurations use compact spatial grids and short training schedules for end-to-end metric logging. Public loaders may synthesize placeholder volumes when licensed roots are absent. Subject-level cross-validation, physical-space surface-distance metrics, and calibration plots are specified for future tables after licensed mounts and are marked incomplete in the present draft.

---

## 4. Experiments

### 4.1 Datasets and honesty inventory

| Dataset | Role | Status in this draft |
|---------|------|----------------------|
| Synthetic / auto-fake tensors | Pipeline verification | **Used** for all quantitative panels below |
| ACDC (challenge) | Reconstruction, segmentation, phenotype proxy | **Not mounted.** Local challenge-named folders hold demo-scale synthetic NIfTI placeholders (eight subjects; compact \(32{\times}32{\times}16{\times}8\) volumes). Official download requires institutional registration |
| MM-WHS | Static segmentation / table fusion | Demo-scale placeholders; motion off / \(T{=}1\) recommended when real data arrive |
| EMIDEC | Scar / infarct proxy | Demo-scale placeholders; subject-level tables incomplete |
| Private AMI | Five-year MACE | Out of scope (example manifest only; no cohort redistribution) |

### 4.2 Ablations

We compare: broadcast reconstruction baseline; per-frame decoding without motion loss weights; per-frame decoding with warp and image-cycle terms; concatenation versus HeartTTable-lite fusion; and a phenotype-task configuration. Summaries are aggregated from measured demo-regime ablation logs.

### 4.3 Metrics

Reconstruction: peak signal-to-noise ratio (PSNR), global structural-similarity proxy, mean absolute error (MAE). Segmentation: per-class Dice; optional subsampled Hausdorff-95 (approximate on compact grids). Motion: warp L1, image-cycle error, EF proxy. Phenotype / event / concordance indices when applicable—demo values are not clinical performance.

---

## 5. Results

### 5.1 Demo-regime ablation summary

All numbers in this section are **demo-regime pipeline metrics** measured on synthetic public-proxy tensors after short training schedules. They establish that reconstruction, segmentation, and motion terms are logged end-to-end; they are **not** estimates of clinical accuracy on licensed challenge cohorts.

Representative reconstruction PSNR on motion-relevant runs is approximately 19.1 dB with MAE approximately 0.69 under the compact fake-data regime. Motion-enabled runs log non-zero warp and image-cycle errors. Segmentation Dice on synthetic labels is low and class-imbalanced (right ventricle often near zero), consistent with under-trained demonstration schedules rather than a clinical ceiling.

**Table 1. Reconstruction and motion metrics (demo regime).**

| Configuration | PSNR (dB) | MAE | Warp L1 | Image-cycle | EF proxy |
|---------------|-----------|-----|---------|-------------|----------|
| Broadcast baseline | 19.08 | 0.693 | — | — | — |
| Per-frame, no motion weights | 19.15 | 0.692 | \(9.7{\times}10^{-4}\) | ≈0 | 0.088 |
| Per-frame + motion | 19.15 | 0.695 | \(1.1{\times}10^{-3}\) | \(3.2{\times}10^{-4}\) | 0.081 |
| Fusion (concatenation) | 19.09 | 0.692 | \(5.0{\times}10^{-4}\) | \(1.1{\times}10^{-4}\) | 0.146 |
| Fusion (HeartTTable-lite) | 19.26 | 0.689 | \(1.3{\times}10^{-3}\) | \(7.6{\times}10^{-4}\) | 0.068 |

**Table 2. Segmentation Dice (demo regime).**

| Configuration | Dice LV | Dice RV | Dice myocardium | Dice mean |
|---------------|---------|---------|-----------------|-----------|
| Broadcast baseline | 0.227 | ≈0 | 0.278 | 0.168 |
| Per-frame, no motion weights | 0.014 | ≈0 | 0.495 | 0.170 |
| Per-frame + motion | 0.038 | ≈0 | 0.439 | 0.159 |
| Fusion (HeartTTable-lite) | ≈0 | ≈0 | 0.567 | 0.189 |
| Phenotype task | 0.053 | ≈0 | ≈0 | 0.018 |

**Phenotype diagnostics.** Logged phenotype accuracy of 0.0 with area under the curve of 1.0 on a tiny synthetic split illustrates metric instability; we treat this as a pipeline check, not a scientific result. Columns corresponding to synthetic event AUC or concordance in demo logs are likewise diagnostics.

One-epoch demonstration checkpoints on recon- and phenotype-oriented configurations similarly show low Dice and mid-teens to ~19 dB PSNR.

### 5.2 Figures

- **Fig. 1** Pipeline schematic showing reconstruction, motion, segmentation, and lite-fusion roles.
- **Fig. 2** Reconstruction PSNR and MAE across ablations (**demo regime**).
- **Fig. 3** Warp, image-cycle, and EF proxies (**demo regime**).
- **Fig. 4** Dice heatmap across ablations (**demo regime**).
- **Fig. 5** Claim-boundary diagram versus the original clinical multimodal study.

Real ACDC cine volume curves, confidence intervals, and physical-space Hausdorff-95 remain to be completed after licensed mounts.

---

## 6. Discussion

The practical contribution of this work is a reproducible motion-consistent 4D training contract on public proxies, with documentation that prevents silent overclaim of the original AMI discrimination numbers. Image-cycle naming matters for peer review: reviewers familiar with registration literature will expect inverse consistency of flow fields; we instead regularize intensity cycles after warping.

HeartTTable-lite is intentionally under-powered relative to the original multimodal module. Building full three-class-token pairwise attention before real AMI data would optimize for a claim we cannot evaluate publicly.

Relative to whole-sequence memory segmentation[^5], unrolled joint recon–motion–seg[^3], and mesh/tetrahedra 4D recovery[^7][^8], our novelty claim is scoped to an auditable lightweight ReconSeg3D extension with explicit motion losses and fusion ablation boundaries—not leaderboard dominance.

**Limitations.**

1. Compact convolutional backbones are interface-compatible with, but not compute-matched to, paper-scale 3D vision transformers or full nnU-Net grids.
2. Sparse SA sampling is a slice-masking simulation, not vendor undersampled \(k\)-space.
3. Global structural-similarity and subsampled Hausdorff-95 are approximate proxies on compact grids.
4. No nested cross-validation, calibration plots, or licensed challenge subject-level tables are reported yet.
5. Challenge-named local folders remain synthetic placeholders until licensed ACDC / MM-WHS / EMIDEC data are mounted.
6. Private AMI survival evaluation and the original five-year discrimination figures remain out of scope.
7. Clinical tabular vocabularies needed for a full HeartTTable-style study are not redistributed here.

---

## 7. Conclusions

We describe a methods-oriented, motion-consistent 4D extension of a ReconSeg3D-style pipeline for public cardiac MRI proxies, with HeartTTable-lite as a fusion ablation and an explicit refusal to claim private AMI clinical discrimination as ours. Measured demo-regime ablations demonstrate that reconstruction, segmentation, and motion metrics are logged end-to-end. Completing subject-level public-data tables after licensed dataset mounts—and, only with authorized private cohorts, survival evaluation—remain future work.

---

## Data availability

Public benchmarks (ACDC, MM-WHS, EMIDEC) follow their respective licenses and access procedures (see repository data documentation). Official ACDC access requires institutional registration; this repository does not redistribute challenge volumes. Demo-regime metrics and figure sources are archived with the accompanying research report and figure bundle. No private AMI data are distributed with this repository. Challenge-named folders shipped for continuous integration contain synthetic placeholders only.

## Code availability

https://github.com/Coucou2016/ReconSeg3D-cardiac-rec  

Install dependencies listed in the repository requirements file (includes SciencePlots for figure regeneration). Demonstration figures can be regenerated with the repository figure script; the companion research report documents engineering inventory and process notes outside the scope of this manuscript.

## References

[^1]: Gao, Q. et al. 3D Spatiotemporal cardiac reconstruction for predicting MACE in acute myocardial infarction. *npj Digit. Med.* **9**, 325 (2026). https://doi.org/10.1038/s41746-026-02449-0

[^2]: Hammernik, K. et al. Deep learning for accelerated and robust MRI reconstruction. *MAGMA* (2024). https://doi.org/10.1007/s10334-024-01173-8

[^3]: Qian, P. et al. Unified Deep Learning for Simultaneous Cardiac Cine MRI Reconstruction, Motion Estimation and Segmentation. *IEEE ISBI* (2024). https://doi.org/10.1109/ISBI56570.2024.10635390

[^4]: Yuan, X. et al. 4D Myocardium Reconstruction with Decoupled Motion and Shape Model. *ICCV* (2023).

[^5]: Ye, M., Xin, B., Axel, L. & Metaxas, D. Continuous Spatio-Temporal Memory Networks for 4D Cardiac Cine MRI Segmentation. In *WACV* 9514–9524 (2025). Also arXiv:2410.23191.

[^6]: Katzman, J. L. et al. DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. *BMC Med. Res. Methodol.* **18**, 24 (2018). https://doi.org/10.1186/s12874-018-0482-1

[^7]: Chen, Y., Yang, J., Mercadier, D. S., Le, H. & Fua, P. MedTet: an online motion model for 4D heart reconstruction. arXiv:2412.02589 (2024).

[^8]: Chen, Y. et al. End-to-end 4D heart mesh recovery across full-stack and sparse cardiac MRI. *Trans. Mach. Learn. Res.* (2026). Also arXiv:2509.12090.

---

## Notes (nature-writing; not for submission body)

**Axes:** `task=manuscript` · `paper_type=methods` · `language=en` · `journal=nature-family` (methods framing).

**Claim–evidence map**

| Claim | Evidence | Status |
|-------|----------|--------|
| Motion-consistent 4D stack exists and logs metrics | Code + measured demo tables + Figs 2–3 | Supported (demo regime) |
| Public clinical superiority vs baselines | Real ACDC folds | Needs evidence (incomplete) |
| HeartTTable parity / private AMI clinical AUC | — | Out of scope / rejected |

**Assumptions or missing inputs:** licensed ACDC/MM-WHS/EMIDEC on disk; physical HD95; nested CV; private AMI for survival claims.

**Why this structure:** methods argument chain (problem → method → ablation → boundary); clinical multimodal paper deferred.
