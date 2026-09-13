# Geometry- and motion-constrained 4D reconstruction and segmentation from sparse short-axis CMR

**Working manuscript (methods)**  
**Status:** Major-revision framing. Quantitative panels that use synthetic / auto-fake tensors are **DEMO-regime only**. Subject-level public-benchmark tables remain **待补充** after licensed dataset mounts.  
**Target genre:** Nature-family / Medical Image Analysis–style methods · English  
**Code availability:** https://github.com/Coucou2016/ReconSeg3D-cardiac-rec  

---

## Abstract

Sparse short-axis (SA) cine cardiac magnetic resonance (CMR) undersamples three-dimensional anatomy across the cardiac cycle. Dense spatiotemporal reconstruction can restore anatomical continuity for segmentation and functional analysis, but unconstrained deep warps may fold tissue and break cycle topology. Gao et al. (*npj Digital Medicine*, 2026) showed that reconstructed bi-ventricular cine volumes can support long-horizon major adverse cardiovascular event (MACE) prediction on a private acute myocardial infarction (AMI) cohort; that clinical discrimination figure is tied to private data and is **not claimed here**. We present a public methods extension that treats **geometry- and motion-constrained 4D recovery from sparse SA** as the main scientific line: per-frame volumetric decoding; pull-field warps sharing a single `grid_sample` convention (`align_corners=True`, channels `(dz,dy,dx)`); true inverse consistency \(L_\mathrm{inv}=\|u+W(v,u)\|+\|v+W(u,v)\|\); spatial smoothness; a Jacobian folding penalty \(\mathrm{ReLU}(\varepsilon-\det J)\); and a periodic loop residual for adjacent compositions. Image-cycle intensity residuals and ventricular volume-curve smoothing are retained only as **auxiliary / physiological** regularizers. Segmentation is supervised at labeled end-diastolic (ED) and end-systolic (ES) phases when available; unlabeled frames rely on motion self-supervision. HeartTTable-style fusion, Cox survival, and binary MACE heads are **decoupled** from the publication path (`w_\mathrm{mace}=w_\mathrm{cox}=0`). On controlled demo-scale tensors we verify end-to-end logging of reconstruction (PSNR/MAE, global-SSIM **proxy**), ED/ES Dice, inverse-consistency / Jacobian statistics, and ED↔ES label-propagation Dice/HD95 APIs. Licensed ACDC / MM-WHS / EMIDEC subject tables remain to be completed. We do not claim private-AMI MACE parity.

**Keywords:** cardiac MRI; 4D reconstruction; inverse-consistent registration; Jacobian regularity; multi-task learning; public benchmarks

---

## 1. Introduction

Cardiac cine magnetic resonance imaging (CMR) is typically acquired as a stack of two-dimensional short-axis (SA) slices. Slice gaps, through-plane motion, and temporal undersampling leave a sparse observation of bi-ventricular anatomy. Dense three-dimensional (3D) reconstruction across the cardiac cycle—effectively a four-dimensional (4D) volume—can improve anatomical continuity for segmentation and downstream phenotyping.

Gao et al. (*npj Digital Medicine*, 2026) introduced ReconSeg3D to reconstruct temporally resolved bi-ventricular volumes from SA stacks and HeartTTable to fuse imaging with tabular clinical variables for five-year MACE after percutaneous coronary intervention in AMI[^1]. That study’s headline discrimination metrics are tied to a private cohort and are therefore out of scope for any public reimplementation that lacks those data.

**Gap.** Lightweight public reimplementations often broadcast a single reconstructed volume across time, use mid-cycle frames as silent segmentation references, or regularize only intensity cycles after warping. Registration literature instead emphasizes inverse-consistent displacement composition, diffeomorphic (non-folding) maps, and periodic cardiac constraints[^9][^10][^11]. Concurrent cardiac lines couple reconstruction with motion or mesh dynamics[^3][^4][^7][^8][^12], but a compact, auditable ReconSeg3D-style contract with named geometric losses and an honest multimodal/survival boundary remains useful.

**Contribution.**

1. Geometry-constrained 4D training: inverse consistency, smoothness, Jacobian folding, and loop consistency under one pull-warp convention.
2. ED/ES-aware multi-phase supervision; unlabeled phases via motion self-supervision (no silent `t//2` reference).
3. ED↔ES label-propagation evaluation (Dice + physical-spacing HD95 API).
4. Publication configs that zero MACE/Cox weights; HeartTTable-lite as fusion ablation only.
5. Explicit claim boundary: demo-regime pipeline verification ≠ licensed subject tables ≠ private AMI survival.

---

## 2. Related work

**Learned registration.** VoxelMorph-style CNNs predict dense displacements with similarity and smoothness losses[^9]. TransMorph and related transformers improve capacity for large deformations[^10]. Inverse-consistency and diffeomorphic integration (scaling-and-squaring of stationary velocity fields) reduce folding[^9][^11].

**Cardiac motion and 4D recovery.** Groupwise / multi-view motion models and MulViMotion-style cine registration target beat-to-beat consistency[^12]. Unified recon–motion–seg unrolling[^3], decoupled motion–shape myocardium recovery[^4], memory-based whole-sequence segmentation[^5], Neural ODE continuous dynamics, and mesh/tetrahedra recoveries (MedTet / TetHeart-style)[^7][^8] address complementary geometry. Our stack is intentionally lightweight and registration-literate rather than mesh-complete.

**Multimodal survival.** DeepSurv-style Cox models[^6] and HeartTTable[^1] motivate optional risk heads. We expose APIs but keep them off the main publication path.

**Positioning.** Relative to Gao et al.[^1], this draft is a methods extension focused on geometric motion constraints and public-proxy evaluation. We do not claim leaderboard dominance versus VoxelMorph/TransMorph baselines (tracked TODO) or private AMI discrimination.

---

## 3. Methods

### 3.1 Problem setup

Let \(x \in \mathbb{R}^{B \times C \times T \times D \times H \times W}\) denote a cine volume batch. Sparse SA sampling is simulated by zeroing unobserved slices. The model predicts a dense reconstruction \(\hat{x}\), adjacent pull displacements \(u_t,v_t\), multi-class segmentations at labeled phases, and optional task heads.

### 3.2 Per-frame reconstruction

The default path encodes each frame with a compact 3D convolutional backbone and decodes per-frame volumes. Compact demonstration grids (for example \(16{\times}32{\times}32\), \(T{=}8\)) verify the pipeline; paper-scale grids remain configuration options.

### 3.3 Deformation convention and geometric losses

Displacements use channels \((d_z,d_y,d_x)\) in voxels, converted to `grid_sample` coordinates with `align_corners=True` and scale \(2/\max(\mathrm{dim}-1,1)\). Vector fields are warped with the **same** sampler as images (`warp_vector`). Pull composition is \(u\circ v = v + W(u,v)\).

We minimize:

- **Inverse consistency** \(L_\mathrm{inv}=\|u+W(v,u)\|+\|v+W(u,v)\|\) (primary geometric cycle).
- **Smoothness** \(L_\mathrm{smooth}=\|\nabla u\|^2\).
- **Folding penalty** \(L_\mathrm{jac}=\mathrm{ReLU}(\varepsilon-\det J)\) with \(J=I+\nabla u\).
- **Loop consistency** \(L_\mathrm{loop}\): composition of adjacent forward fields over the available \(T\) ≈ identity (coarse for short \(T\)).
- **ED-anchored path** \(L_\mathrm{ed\_ref}\): compose adjacent fields into \(\phi_{k\to\mathrm{ED}}\) / \(\phi_{\mathrm{ED}\to k}\) and apply the same inverse-consistency residual (`w_ed_ref`).
- **Auxiliary image-cycle**: intensity residual after fwd/bwd warps (explicitly *not* \(L_\mathrm{inv}\)).
- **Demoted volume-curve**: second-difference of fractional LV/RV volumes (physiological soft prior).

Optional stationary velocity fields with scaling-and-squaring are stubbed (`use_svf`) and disabled by default.

### 3.4 ED/ES segmentation

When ACDC-style `Info.cfg` provides ED/ES indices, batch fields expose `seg_frame_indices`, `segmentation_sequence`, and `seg_valid_mask`. Cross-entropy/Dice are applied only on labeled frames. The primary logits head gathers the ED index (not mid-cycle).

### 3.5 Evaluation

Reconstruction: PSNR, MAE, **global SSIM proxy** (not windowed SSIM). Segmentation: Dice; HD95 with optional physical spacing \((s_z,s_y,s_x)\) mm. Motion: inverse-consistency residual, negative Jacobian ratio, ED↔ES label-propagation Dice/HD95. Real subject-level propagation tables are marked **待补充** under demo-only data.

### 3.6 HeartTTable-lite and risk heads (supplementary)

A single class token attending to concatenated spatial/temporal/table streams is retained as a fusion ablation. Publication YAML sets `w_mace: 0` and `w_cox: 0`.

---

## 4. Experiments

### 4.1 Datasets and honesty inventory

| Dataset | Role | Status |
|---------|------|--------|
| Synthetic / auto-fake | Pipeline / geometry unit tests | **DEMO** |
| ACDC | ED/ES recon–seg–motion | Licensed mounts **待补充**; local challenge-named folders may be placeholders |
| MM-WHS / EMIDEC | Static / scar proxies | Placeholders until licensed |
| Private AMI | Five-year MACE | Out of scope |

### 4.2 Ablations and publication configs

Compare broadcast vs per-frame; geometry weights on/off; concat vs HeartTTable-lite (supplementary). Use `configs/publication_recon.yaml`, `publication_motion.yaml`, `publication_seg.yaml` for the main line (`allow_fake_data: false` hard-errors empty public roots).

### 4.3 Metrics

See §3.5. Demo-regime numbers in prior drafts remain pipeline checks only and are not upgraded to clinical performance claims here.

---

## 5. Results

Subject-level licensed tables: **待补充**. Unit tests cover identity/exact-inverse flows, +1-voxel warp direction, ED/ES index wiring, label-propagation API with known translation, and HD95 spacing `(1,1,1)` vs `(8,1,1)`. Any DEMO ablation PSNR/Dice logged in the companion research report must stay labeled as demo.

---

## 6. Discussion

Naming image-cycle separately from \(L_\mathrm{inv}\) prevents registration-literate misreading. Demoting volume-curve and zeroing MACE/Cox on the main path keeps the novelty claim aligned with geometry-constrained 4D recovery.

**Limitations.** Compact backbones; simulated sparse SA (not \(k\)-space); short-\(T\) loop is approximate; SVF path stubbed; VoxelMorph/TransMorph/MulViMotion baselines and multi-seed CI deferred; global SSIM proxy only; no private AMI evaluation.

---

## 7. Conclusions

We describe a geometry- and motion-constrained 4D extension of a ReconSeg3D-style public pipeline for sparse SA CMR, with ED/ES-aware supervision and an explicit refusal to claim private AMI clinical discrimination. Completing licensed subject tables and external registration baselines remains future work.

---

## Data availability

Public benchmarks follow their licenses (see repository `docs/DATA.md`). No private AMI data are redistributed. Challenge-named CI folders may contain synthetic placeholders only.

## Code availability

https://github.com/Coucou2016/ReconSeg3D-cardiac-rec  

## References

[^1]: Gao, Q. et al. 3D Spatiotemporal cardiac reconstruction for predicting MACE in acute myocardial infarction. *npj Digit. Med.* **9**, 325 (2026). https://doi.org/10.1038/s41746-026-02449-0

[^2]: Hammernik, K. et al. Deep learning for accelerated and robust MRI reconstruction. *MAGMA* (2024). https://doi.org/10.1007/s10334-024-01173-8

[^3]: Qian, P. et al. Unified Deep Learning for Simultaneous Cardiac Cine MRI Reconstruction, Motion Estimation and Segmentation. *IEEE ISBI* (2024). https://doi.org/10.1109/ISBI56570.2024.10635390

[^4]: Yuan, X. et al. 4D Myocardium Reconstruction with Decoupled Motion and Shape Model. *ICCV* (2023).

[^5]: Ye, M., Xin, B., Axel, L. & Metaxas, D. Continuous Spatio-Temporal Memory Networks for 4D Cardiac Cine MRI Segmentation. In *WACV* 9514–9524 (2025). Also arXiv:2410.23191.

[^6]: Katzman, J. L. et al. DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. *BMC Med. Res. Methodol.* **18**, 24 (2018). https://doi.org/10.1186/s12874-018-0482-1

[^7]: Chen, Y., Yang, J., Mercadier, D. S., Le, H. & Fua, P. MedTet: an online motion model for 4D heart reconstruction. arXiv:2412.02589 (2024).

[^8]: Chen, Y. et al. End-to-end 4D heart mesh recovery across full-stack and sparse cardiac MRI. *Trans. Mach. Learn. Res.* (2026). Also arXiv:2509.12090.

[^9]: Balakrishnan, G. et al. VoxelMorph: a learning framework for deformable medical image registration. *IEEE TMI* (2019).

[^10]: Chen, J. et al. TransMorph: Transformer for unsupervised medical image registration. *Med. Image Anal.* (2022).

[^11]: Dalca, A. V. et al. Unsupervised learning of probabilistic diffeomorphic registration for images and surfaces. *Med. Image Anal.* (2019).

[^12]: Related multi-view / MulViMotion-style cardiac cine registration literature (citation to be finalized against the accepted venue list).

---

## Notes (not for submission body)

**Claim–evidence map**

| Claim | Evidence | Status |
|-------|----------|--------|
| Geometry losses + ED/ES wiring exist | Code + unit tests (58 passed) | Supported |
| Public clinical superiority | Licensed ACDC folds | Incomplete (**待补充**) |
| Private AMI AUC | — | Rejected / out of scope |
