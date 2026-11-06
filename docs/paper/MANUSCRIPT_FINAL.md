# Geometry-aware motion constraints for 4D reconstruction and segmentation from sparse short-axis cine CMR

**Article type:** Methods  
**Target style:** npj Digital Medicine–adjacent methods paper  
**Evidence regime:** Engineering verification on controlled demo-scale tensors and smoke ablations. Licensed subject-level benchmark tables are outside the present measured set.

---

## Abstract

Sparse short-axis cine cardiac magnetic resonance (CMR) leaves large gaps in three-dimensional anatomy across the cardiac cycle. Dense four-dimensional (4D) recovery can restore spatial continuity for segmentation and functional readout, yet unconstrained learned warps often fold tissue and break beat periodicity. Registration work already shows that inverse-consistent displacements, smoothness, and Jacobian regularity matter for stable maps, while many public recon–seg stacks still broadcast a single volume across time or supervise only intensity residuals after warping. This study implements a compact recon–seg–motion stack in which adjacent pull fields share one `grid_sample` convention and are trained with true inverse consistency, spatial smoothness, a Jacobian folding penalty, periodic adjacent composition that includes the return step from the last frame to the first, and an end-diastolic (ED) reference path on composed maps. Segmentation uses labeled ED and end-systolic (ES) frames when present. Multimodal fusion and survival heads stay optional and are zeroed on the publication training path. On controlled demo tensors the stack logged reconstruction PSNR near 19 dB across ablation arms, windowed `ssim_3d` of 0.515, inverse-consistency residual 0.00177, mean Jacobian determinant 0.9998 with zero folding ratio, and ED↔ES label-propagation Dice of 1.0 under identity-scale verification. These figures confirm end-to-end metric logging and geometric wiring. They are engineering checks on synthetic or placeholder volumes, not licensed ACDC subject scores and not private AMI risk discrimination.

**Keywords:** cardiac MRI; 4D reconstruction; inverse-consistent registration; Jacobian regularity; multi-task learning; engineering verification

---

## 1. Introduction

### 1.1 Background and research problem

Cine CMR is usually acquired as a stack of two-dimensional short-axis (SA) slices. Slice gaps, through-plane motion, and limited temporal sampling leave a sparse observation of bi-ventricular anatomy. Dense reconstruction across the cycle produces a 4D volume that can support segmentation, volume–time curves, and later phenotyping.

When dense cine volumes feed downstream risk models on private acute myocardial infarction (AMI) cohorts, clinical discrimination depends on those private data and on multimodal tabular fusion. A public methods codebase that lacks the same cohorts cannot reuse those headline risk scores. The scientific task that remains open in public code is therefore narrower. Can sparse SA stacks be recovered as dense 4D volumes under geometric motion constraints that registration theory already treats as first-class, with an evaluation contract that separates reconstruction, segmentation, motion regularity, and optional risk heads?

### 1.2 Previous research and knowledge gap

Learned registration networks such as VoxelMorph predict dense displacements from image pairs and combine similarity with smoothness priors [1]. Probabilistic diffeomorphic models integrate stationary velocity fields to discourage folding [2]. Cardiac sequence work adds groupwise or generative motion models that improve temporal coherence on cine MRI [3], unsupervised tagged-MRI trackers that compose maps across frames [4], and joint groupwise registration with dynamic reconstruction [5]. Unified unrolling of reconstruction, motion, and segmentation has also been explored for cine MRI [6], as have decoupled motion–shape myocardium models and continuous memory networks for whole-sequence segmentation [7,8].

Public lightweight recon–seg reimplementations often diverge from that registration standard. A single reconstructed volume may be copied across time. Mid-cycle frames may act as silent segmentation references. Intensity residuals after warping may be treated as if they were inverse consistency of the displacement fields. Multimodal survival heads may remain entangled with the imaging objective even when the release has no survival labels. The gap addressed here is therefore concrete. A compact public stack needs a named geometric motion contract, ED/ES-aware supervision, and an honest split between imaging metrics and optional risk APIs, with numbers that map to reproducible local runs rather than to private AMI tables.

### 1.3 Objectives and contributions

This study asks whether a geometry-aware motion module can be wired into a ReconSeg3D-style public pipeline so that reconstruction, adjacent pull warps, ED/ES segmentation, and motion regularity logs are jointly measurable on controlled tensors. The work focuses on implementation fidelity and engineering verification.

The main contributions are as follows.

1. A shared pull-field convention with true inverse consistency \(L_\mathrm{inv}=\|u+W(v,u)\|+\|v+W(u,v)\|\), spatial smoothness, Jacobian folding control, periodic adjacent composition including the last-to-first return step, and an ED-anchored composed-path residual.
2. ED/ES-indexed segmentation supervision when labels exist, with unlabeled phases driven by motion self-supervision.
3. Evaluation APIs for windowed three-dimensional SSIM, global reconstruction proxy scores, inverse-consistency and Jacobian statistics, and ED↔ES label propagation with optional physical-spacing HD95.
4. Publication configs that set multimodal MACE and Cox weights to zero, keeping HeartTTable-style fusion as an ablation only.

All quantitative panels below are demo or smoke measurements. Licensed challenge subject tables and private AMI discrimination lie outside the present evidence.

---

## 2. Related work

VoxelMorph established a CNN mapping from image pairs to dense warps with unsupervised intensity matching and optional segmentation guidance [1]. Diffeomorphic extensions based on scaling-and-squaring of velocity fields reduce folding and support probabilistic formulations [2]. For cardiac sequences, Krebs et al. learned a latent motion matrix with a Gaussian-process prior and reported smoother deformations on cine MRI [3]. DeepTag tracks tagged MRI by composing diffeomorphic fields without manual landmarks [4]. Groupwise registration has been combined with dynamic MRI reconstruction so that the full temporal sequence informs both alignment and recovery [5]. Concurrent lines couple recon–motion–seg unrolling [6], mesh or tetrahedral heart recovery, and continuous spatiotemporal memory for segmentation [7–9]. Multimodal survival models such as DeepSurv motivate optional risk heads [10]. Clinical pipelines that reconstruct dense bi-ventricular cine and predict major adverse cardiovascular events illustrate downstream use of dense volumes [11]. The present stack stays registration-literate and lightweight. It does not claim private cohort discrimination reported in that clinical setting.

---

## 3. Methods

### 3.1 Study object and input representation

Let \(x \in \mathbb{R}^{B \times C \times T \times D \times H \times W}\) denote a cine volume batch. Sparse SA sampling is simulated by zeroing unobserved slices, or by sparse stacking when spacing metadata are available. The model predicts a dense reconstruction \(\hat{x}\), adjacent forward and backward pull displacements, multi-class segmentations at labeled phases, and optional task heads.

Demo verification uses compact grids such as \(16 \times 32 \times 32\) with \(T=8\). Publication YAML files refuse empty public roots and set `allow_fake_data` to false. Smoke YAML files alone may use synthetic placeholders for continuous integration.

### 3.2 Per-frame reconstruction

The default path encodes each frame with a compact three-dimensional convolutional backbone and decodes one volume per time index (`per_frame_recon=true`). A broadcast baseline that copies a single reconstructed volume across time is retained only as an ablation arm. Figure 1 summarizes the pipeline roles.

### 3.3 Deformation convention

Displacement channels are ordered \((d_z,d_y,d_x)\) in voxels. Conversion to `grid_sample` coordinates uses `align_corners=True` and scale \(2/\max(\mathrm{dim}-1,1)\). Vector fields are warped with the same sampler as images. Pull composition is defined as
\begin{equation}
u \circ v = v + W(u,v),
\end{equation}
where \(W(\cdot,\cdot)\) denotes the shared pull sampler. Closed MotionNet stacks emit \(T\) adjacent pairs for \(T\) frames. The last slot carries the return step from frame \(T-1\) to frame \(0\).

### 3.4 Geometric losses

The primary geometric terms follow the repository motion module.

True inverse consistency on a forward–backward pair is
\begin{equation}
L_\mathrm{inv}=\|u+W(v,u)\|+\|v+W(u,v)\|.
\end{equation}
Exact inverses drive both residuals to zero. This term is distinct from an intensity residual after warping.

Spatial smoothness uses finite differences on each displacement channel,
\begin{equation}
L_\mathrm{smooth}=\|\nabla u\|^2.
\end{equation}

Jacobian folding uses \(J=I+\nabla u\) and
\begin{equation}
L_\mathrm{jac}=\mathrm{ReLU}(\varepsilon-\det J).
\end{equation}
Training also logs `jac_neg_ratio`, `jac_det_mean`, and `jac_det_min`.

Periodic cycle composition folds adjacent forward fields over the available \(T\) pairs, including the return edge when present, and penalizes departure from the identity,
\begin{equation}
L_\mathrm{loop}=\Bigl\|\mathop{\circ}_{t=0}^{T-1} u_t\Bigr\|_1.
\end{equation}

ED-anchored consistency composes adjacent fields into maps \(\phi_{k\to\mathrm{ED}}\) and \(\phi_{\mathrm{ED}\to k}\) for each frame \(k\), then applies the same inverse-consistency residual as in Eq. (2). Linear ED paths use adjacent slots only. They do not reuse the return edge.

Auxiliary terms include an image-cycle intensity residual after forward and backward warps, a second-difference penalty on fractional ventricular volume curves, and an optional warp intensity L1 term. These auxiliaries support appearance alignment and soft physiological smoothness. They are not substitutes for \(L_\mathrm{inv}\). Optional stationary velocity fields with scaling-and-squaring are available through a model flag.

### 3.5 ED/ES segmentation

When ACDC-style ED and ES indices are present, the batch exposes segmentation frame indices, a segmentation sequence, and a validity mask. Cross-entropy and Dice losses act only on labeled frames. Primary logits gather the ED index. Unlabeled phases rely on motion self-supervision and geometric regularizers.

### 3.6 Evaluation protocol

Reconstruction metrics are PSNR, MAE, windowed `ssim_3d`, and a global `recon_ssim_proxy`. The proxy is never treated as clinical SSIM. Segmentation metrics are Dice and HD95, with optional physical spacing \((s_z,s_y,s_x)\) in millimetres. Functional readout uses physical EDV, ESV, and EF when ED/ES labels and spacing exist. An `ef_proxy` based on voxel max/min labels is logged only as a coarse check. Motion metrics include inverse-consistency residual, Jacobian statistics, and ED↔ES label-propagation Dice and HD95.

### 3.7 Optional fusion and risk heads

A lightweight class-token fusion block attending to spatial, temporal, and tabular streams is retained for ablation. Publication configs set MACE and Cox loss weights to zero. Binary MACE AUC on synthetic labels is logged only as a pipeline check when those heads are enabled.

---

## 4. Experiments

### 4.1 Data regimes

Table 1 lists the data regimes used for the measured results.

**Table 1.** Data regimes for the present manuscript.

| Regime | Role in this paper | Status in measured results |
|--------|--------------------|----------------------------|
| Synthetic / auto-fake tensors | Geometry unit tests and joint demo eval | Measured (DEMO) |
| Ablation smoke trees | Reconstruction and fusion arms | Measured (DEMO) |
| Challenge-named local folders | Named smoke continuity | Measured as placeholders only |
| Licensed ACDC / MM-WHS / EMIDEC subjects | Subject-level recon–seg–motion tables | Not present in the measured set |
| Private AMI multimodal cohorts | Five-year event discrimination | Out of scope |

### 4.2 Ablation design

Ablation arms compare broadcast versus per-frame reconstruction, motion weights on versus off, and concatenation fusion versus HeartTTable-lite fusion. Publication configs keep risk weights at zero. Smoke configs alone enable the short DEMO runs reported in Section 5.

### 4.3 Implementation

Training and evaluation entry points live in the public repository. Geometric losses are implemented in the motion module and composed in the joint loss. Unit tests cover identity and exact-inverse flows, warp direction, ED/ES index wiring, label propagation under known translation, and HD95 spacing consistency. Figure generation uses SciencePlots with Times New Roman.

---

## 5. Results

All numbers in this section are DEMO or smoke measurements from controlled local runs. They demonstrate logging continuity and geometric wiring. They are not licensed subject-level clinical performance.

### 5.1 Reconstruction ablation under DEMO tensors

Figure 2 and Table 2 summarize reconstruction PSNR, MAE, and mean Dice across five ablation arms from the smoke ablation table.

**Table 2.** DEMO reconstruction ablation summary.

| Run | PSNR (dB) | MAE | Dice mean |
|-----|----------:|----:|----------:|
| broadcast_baseline | 19.08 | 0.693 | 0.168 |
| per_frame_no_motion | 19.15 | 0.692 | 0.170 |
| per_frame_motion | 19.15 | 0.695 | 0.159 |
| fusion_concat | 19.09 | 0.692 | 0.139 |
| fusion_heart_ttable | 19.26 | 0.689 | 0.189 |

PSNR values sit in a narrow band near 19 dB. Under tiny fake tensors and short training this near-equality is expected. The useful observation is that every arm emits reconstruction metrics through the same logging path. Fusion with the HeartTTable-lite block reached the highest smoke PSNR (19.26 dB) and the highest mean Dice (0.189) in this DEMO set, while concatenation fusion sat lower on Dice (0.139). These differences remain regime-limited.

### 5.2 Geometry and motion logs

Figure 3 and Table 3 report coordinate-level geometry from the joint demo evaluation.

**Table 3.** DEMO geometry and reconstruction logs from the joint evaluation record.

| Metric | Value | Role |
|--------|------:|------|
| Inverse-consistency residual | 0.00177 | \(L_\mathrm{inv}\) |
| Cycle composition residual | 0.00177 | \(L_\mathrm{loop}\) |
| Smoothness | \(6.1\times10^{-8}\) | \(L_\mathrm{smooth}\) |
| Folding penalty | 0.0 | \(L_\mathrm{jac}\) |
| Negative Jacobian ratio | 0.0 | fraction with \(\det J < \varepsilon\) |
| Mean Jacobian determinant | 0.9998 | interior finite-difference estimate |
| Image-cycle intensity residual | \(2.3\times10^{-5}\) | auxiliary appearance term |
| Reconstruction PSNR | 18.80 dB | joint demo recon |
| Windowed `ssim_3d` | 0.515 | primary structural score |
| Global `recon_ssim_proxy` | 0.0098 | global proxy only |

The inverse-consistency residual and the cycle composition residual are of the same order. The Jacobian logs show no folding under the DEMO fields. Windowed `ssim_3d` (0.515) and the global proxy (0.0098) diverge sharply, which is why publication text keeps the two names separate. Label-propagation Dice from ED to ES and from ES to ED both equalled 1.0 on the joint demo record under the identity-scale verification regime used in that run.

### 5.3 Segmentation smoke profiles

Figure 4 shows class-wise Dice from the same ablation CSV. RV Dice collapses near zero after short DEMO training on placeholder labels. Myocardium Dice varies more across arms, with the HeartTTable-lite arm reaching 0.567 in this smoke table. These class imbalances are artefacts of the DEMO label regime. They mark logging behaviour, not an algorithmic ceiling on licensed data.

### 5.4 Named pipeline checkpoints

Table 4 lists two named smoke checkpoints.

**Table 4.** Named DEMO pipeline checkpoints.

| Run | PSNR (dB) | Dice mean | Warp L1 | Image-cycle |
|-----|----------:|----------:|--------:|------------:|
| paper_recon_smoke | 19.027 | 0.0480 | 0.000294 | 0.000464 |
| paper_acdc_smoke | 17.390 | 0.0175 | 0.000086 | 0.000083 |

The `paper_acdc_smoke` run reads challenge-named local folders that contain synthetic placeholders. It is a path-continuity check. It is not a CREATIS ACDC leaderboard evaluation. That run also logged an LV HD95 of 15.68 under the DEMO spacing used in the checkpoint record.

### 5.5 Supplementary analysis of ablation spreads

Across the five main reconstruction arms in Table 2, PSNR ranges from 19.08 to 19.26 dB (span 0.18 dB). MAE ranges from 0.689 to 0.695. Mean Dice ranges from 0.139 to 0.189. The largest Dice shift in this DEMO set therefore comes from fusion choice rather than from broadcast versus per-frame reconstruction. Warp intensity L1 and image-cycle residuals remain below \(1.3\times10^{-3}\) whenever they are logged. Synthetic MACE AUC and C-index fields appear in the ablation CSV for arms that keep risk heads attached. Those values (for example 0.625 and 0.556 on several arms) are pipeline logs on synthetic labels. They are excluded from any clinical claim.

### 5.6 Unit-test verification of geometric contracts

Repository tests exercise identity flows, exact inverse pairs, a one-voxel warp direction check, ED/ES index gathering, label propagation under a known translation, and HD95 under spacing \((1,1,1)\) versus \((8,1,1)\). These tests support the Methods definitions independently of the smoke tables.

---

## 6. Discussion

### 6.1 Interpretation of the measured findings

The DEMO results support three practical statements. First, the joint stack emits reconstruction, segmentation, and geometry keys through one evaluation path. Second, inverse consistency and cycle composition leave small residuals on the joint demo fields, and the Jacobian logs report no folding. Third, windowed and global structural scores must stay distinct, because the DEMO record shows a large numerical gap between them.

The near-flat PSNR across ablation arms is consistent with short training on tiny synthetic tensors. In that regime the ablation mainly audits wiring and logging, not ranking of clinical algorithms. The higher DEMO Dice under HeartTTable-lite likewise remains a smoke observation.

### 6.2 Relation to prior registration and cardiac motion work

The geometric terms follow the registration literature that treats inverse consistency, smoothness, and Jacobian regularity as primary controls on displacement quality [1,2]. Cardiac sequence models that enforce temporal coherence or compose maps across the beat motivate the periodic composition and ED-anchored path used here [3–5]. The present contribution is a compact public wiring of those ideas into a recon–seg stack with ED/ES supervision and an explicit publication boundary for risk heads [11].

### 6.3 Scope and limitations

The measured evidence covers demo-scale tensors, placeholder challenge-named folders, and unit tests. Licensed ACDC, MM-WHS, and EMIDEC subject tables are absent from the present result set. Sparse SA sampling is simulated in image space rather than from undersampled \(k\)-space. Compact backbones limit capacity. External registration baselines need additional packages. Private AMI multimodal evaluation is out of scope. Within these bounds the manuscript reports only engineering verification of geometry-aware 4D recovery.

---

## 7. Conclusions

This study implements a geometry-aware motion module for 4D reconstruction and segmentation from sparse SA cine CMR inside a public ReconSeg3D-style codebase. The principal findings are as follows.

1. Adjacent pull fields share one sampling convention and are trained with true inverse consistency, smoothness, Jacobian folding control, periodic composition including the return step, and an ED-anchored composed-path residual.
2. On controlled DEMO tensors the stack logged PSNR near 19 dB across ablation arms, `ssim_3d` of 0.515, inverse-consistency residual 0.00177, mean Jacobian determinant 0.9998 with zero folding ratio, and identity-scale ED↔ES propagation Dice of 1.0.
3. Named smoke checkpoints confirm end-to-end metric emission on both generic recon smoke and challenge-named placeholder folders.
4. Publication configs keep multimodal survival weights at zero. Synthetic risk logs are treated as pipeline checks only.

The conclusions apply to the measured DEMO and unit-test regimes. Subject-level licensed benchmarks remain future measurements once those data are mounted under publication configs.

---

## Data availability

Public cardiac MRI benchmarks follow their original licenses. No private AMI data are redistributed. Challenge-named continuous-integration folders may contain synthetic placeholders only. The measured DEMO tables supporting this manuscript are produced by the public evaluation scripts in the code repository.

## Code availability

Source code is available at https://github.com/Coucou2016/ReconSeg3D-cardiac-rec

---

## References

1. Balakrishnan, G., Zhao, A., Sabuncu, M. R., Guttag, J. & Dalca, A. V. VoxelMorph: a learning framework for deformable medical image registration. *IEEE Trans. Med. Imaging* **38**, 1788–1800 (2019). https://doi.org/10.1109/TMI.2019.2897538

2. Dalca, A. V., Balakrishnan, G., Guttag, J. & Sabuncu, M. R. Unsupervised learning of probabilistic diffeomorphic registration for images and surfaces. *Med. Image Anal.* **57**, 226–236 (2019). https://doi.org/10.1016/j.media.2019.07.006

3. Krebs, J., Delingette, H., Ayache, N. & Mansi, T. Learning a generative motion model from image sequences based on a latent motion matrix. *IEEE Trans. Med. Imaging* **40**, 1405–1416 (2021). https://doi.org/10.1109/TMI.2021.3056531

4. Ye, M. et al. DeepTag: an unsupervised deep learning method for motion tracking on cardiac tagging magnetic resonance images. In *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)* 7261–7271 (2021). https://doi.org/10.1109/CVPR46437.2021.00719

5. Pan, J. et al. End-to-end deep learning of non-rigid groupwise registration and reconstruction of dynamic MRI. *Front. Cardiovasc. Med.* **9**, 880186 (2022). https://doi.org/10.3389/fcvm.2022.880186

6. Qian, P. et al. Unified deep learning for simultaneous cardiac cine MRI reconstruction, motion estimation and segmentation. In *IEEE Int. Symp. Biomed. Imaging (ISBI)* (2024). https://doi.org/10.1109/ISBI56570.2024.10635390

7. Yuan, X. et al. 4D myocardium reconstruction with decoupled motion and shape model. In *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)* (2023).

8. Ye, M., Xin, B., Axel, L. & Metaxas, D. Continuous spatio-temporal memory networks for 4D cardiac cine MRI segmentation. In *IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV)* 9514–9524 (2025). Also arXiv:2410.23191.

9. Chen, Y. et al. MedTet: an online motion model for 4D heart reconstruction. arXiv:2412.02589 (2024).

10. Katzman, J. L. et al. DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. *BMC Med. Res. Methodol.* **18**, 24 (2018). https://doi.org/10.1186/s12874-018-0482-1

11. Gao, Q. et al. 3D spatiotemporal cardiac reconstruction for predicting MACE in acute myocardial infarction. *npj Digit. Med.* **9**, 325 (2026). https://doi.org/10.1038/s41746-026-02449-0

12. Hammernik, K. et al. Deep learning for accelerated and robust MRI reconstruction. *MAGMA* (2024). https://doi.org/10.1007/s10334-024-01173-8

---

## Figure legends

**Figure 1.** Pipeline schematic for sparse SA cine input, per-frame reconstruction, geometry-aware motion terms, and ED/ES segmentation with optional heads.

**Figure 2.** DEMO reconstruction ablation. Bars show PSNR across broadcast, per-frame, and fusion arms from the smoke ablation table.

**Figure 3.** DEMO motion and geometry logs. Panel layout reports inverse-consistency, cycle composition, smoothness, Jacobian, and structural scores from the joint evaluation record.

**Figure 4.** DEMO segmentation Dice heatmap across ablation arms and cardiac classes from the smoke ablation table.

**Figure 5.** Claim boundary ledger separating DEMO engineering verification, licensed subject tables, and private AMI risk evaluation.
