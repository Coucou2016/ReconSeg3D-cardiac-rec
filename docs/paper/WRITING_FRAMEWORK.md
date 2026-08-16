# Writing architecture & innovation-framed paper outline

**Axes (nature-writing):** `task=manuscript` · `paper_type=methods` · `language=en` · `journal=nature-family` (npj Digital Medicine / MedIA-style methods)

**One-sentence argument:** On public cardiac MRI proxies (and synthetic smoke), we show that a motion-consistent per-frame 4D ReconSeg3D pipeline with explicit warp and image-cycle regularization provides a reproducible reconstruction–segmentation–motion evaluation stack, with HeartTTable-lite retained only as a fusion ablation—without claiming private AMI 5-year MACE AUC 0.934 or full HeartTTable parity.

---

## 1. Recommended venue / genre

| Option | Fit | Notes |
|--------|-----|-------|
| **Methods / software+methods** (MedIA, IEEE TMI, Frontiers in CV Med, arXiv first) | **Best** | Novelty is the **motion-consistent 4D** public pipeline + honest ablations |
| npj Digit. Med.–style clinical outcome paper | Poor until private AMI | Would require nested CV, calibration, external AMI MACE — **out of scope now** |
| Short technical note / code companion | Good interim | Emphasize reproducibility + claim boundary vs Gao et al. 2026 |

**Do not** submit as a clinical MACE superiority paper using smoke AUC/Dice.

---

## 2. Innovation framing (honest)

### Core novelty (claim)
- **Motion-consistent 4D:** per-frame 3D decode + differentiable warp + **image-cycle** (not inverse-consistent flow composition) + optional volume-curve / EF proxies.
- Public-facing tables T1–T4 (recon / seg / phenotype proxy / motion) on ACDC / MM-WHS / EMIDEC **when real data are mounted**; smoke otherwise labeled demo.

### Explicit non-claims
- No private AMI cohort; **do not report 0.934** AUC or C-index 0.897 from the original paper as ours.
- HeartTTable-lite ≠ three-modality pairwise class-token cross-attention; fusion is an **ablation**, not the product story.
- Compact CNN/UNet ≠ paper-scale 3D ViT / nnU-Net 256³.
- Demo metrics = pipeline checks only.

### Positioning vs original npj paper (Gao et al., *npj Digit. Med.* 2026; DOI `10.1038/s41746-026-02449-0`)
- Original: ReconSeg3D → HeartTTable on **n=4511 AMI** → 5-year MACE.
- This work: **methods extension** focusing on temporal consistency / motion losses on **public** data; risk head API ready for Cox when private data arrive.

### Related literature anchors (independently verified DOIs / arXiv)
| Topic | Citation | Role in outline |
|-------|----------|-----------------|
| Original clinical multimodal MACE | Gao et al. 2026, DOI 10.1038/s41746-026-02449-0 | Boundary / related work |
| Joint recon–motion–seg | Qian et al. ISBI 2024, DOI 10.1109/ISBI56570.2024.10635390 | Method neighbors |
| 4D myocardium recon | Yuan et al. ICCV 2023 | Motion–shape decoupling neighbor |
| Whole-sequence 4D seg | CSTM arXiv:2410.23191 | Temporal continuity neighbor |
| MRI recon DL review | Magma 2024, DOI 10.1007/s10334-024-01173-8 | Background |
| Cox DNN survival | DeepSurv, DOI 10.1186/s12874-018-0482-1 | Risk-head background (future AMI) |

---

## 3. Section architecture (methods paper)

1. **Title / Abstract** — motion-consistent 4D; public data; non-claim of AMI AUC.
2. **Introduction** — sparse SA cine gaps → need dense 4D; gap = missing explicit motion consistency in lightweight public reimplementations; contribution bullets.
3. **Related work** — recon; motion/registration; 4D seg; multimodal survival (HeartTTable / DeepSurv); position as methods not clinical parity.
4. **Methods**
   - Data contracts (ACDC / MM-WHS / EMIDEC / synthetic)
   - Per-frame recon encoder–decoder
   - MotionNet, warp scaling, image-cycle definition
   - Seg heads; phenotype / BCE / Cox APIs
   - HeartTTable-lite (single CLS over concat KV)
   - Losses & training protocol
5. **Experiments** — smoke protocol now; planned subject-level CV on real ACDC; ablation matrix (broadcast / PF±motion / fusion / phenotype).
6. **Results** — T1–T4; mark 待补充 where real data missing; smoke figures clearly labeled.
7. **Discussion** — what motion losses buy; failure modes; why not claim MACE.
8. **Limitations & reproducibility**
9. **Data / Code availability**

---

## 4. Figure panel plan (SciencePlots)

| Fig | Claim | Panels | Data source |
|-----|-------|--------|-------------|
| **Fig. 1** | Pipeline roles | Schematic boxes | Architecture (non-quantitative) |
| **Fig. 2** | Recon ablation | PSNR, MAE bars | `outputs/ablations_smoke_v2` |
| **Fig. 3** | Motion consistency | Warp/cycle + EF proxy | same |
| **Fig. 4** | Seg Dice | Heatmap LV/RV/MYO/mean | same |
| **Fig. 5** | Claim boundary | Supported vs out-of-scope | Qualitative ledger |
| **Fig. ED1** 待补充 | Real ACDC ED/ES + cine motion | Volume curves, Dice CI | Real ACDC when available |
| **Fig. ED2** 待补充 | Calibration / C-index | Cox on private AMI | Out of scope until data |

All rendered with **SciencePlots** (`science` + `nature` + Times New Roman for Latin).

---

## 5. Tables (planned)

| Table | Content | Status |
|-------|---------|--------|
| T1 Recon | PSNR / SSIM / MAE | Smoke present; real ACDC 待补充 |
| T2 Seg | Dice / HD95 | Smoke; physical-mm HD95 待补充 |
| T3 Phenotype | Acc / AUC on ACDC classes | Proxy only; smoke AUC unstable |
| T4 Motion | Warp L1, image-cycle, EF proxy | Smoke present |
| T5 Fusion | Concat vs HeartTTable-lite | Ablation only |

---

## 6. Advisor consultation status

- **Prior ChatGPT audit** (2026-08-16): accept HeartTTable-lite naming; motion as novelty; defer 3-CLS; real ACDC primary — see `artifacts/chatgpt_handoff/reports/20260816_chatgpt_audit_response.md`.
- **Five-round maturation (2026-08-16):** Live ChatGPT Plus **0/5** (browser MCP tabs vanish). Substitute rounds under `reports/rounds/`.
- **GitHub-MD five rounds (2026-08-17):** Briefs pushed under `artifacts/chatgpt_handoff/github_briefs/` + `ASK_CHATGPT.md`. Workflow: short ChatGPT message → open GitHub blob/raw → structured reply. Browser MCP still fails (`No browser tab available` / vanished viewId). Independent WebSearch + nature-writing/polishing matured the manuscript; round logs in `reports/rounds2/`.
- Public GitHub for advisor fetch: https://github.com/Coucou2016/ReconSeg3D-cardiac-rec
- Intended chat: https://chatgpt.com/c/6a808651-37cc-83ea-a63d-2d2539a48d07

### On-disk data honesty

| Tree | Verdict |
|------|---------|
| `data/acdc` (8 pts, ~0.49 MB 4D, 32×32×16×8) | **Demo/fake** |
| `data/mmwhs`, `data/emidec` | **Demo/fake** |
| Official ACDC | CREATIS registration required — download **blocked** without credentials |
| Smoke CSV / metrics.json | **Measured** demo numbers (OK for DEMO tables) |

---

## 7. Acceptance criteria for draft manuscript

- [x] Methods paper architecture
- [x] Explicit non-claims of 0.934 / full HeartTTable
- [x] SciencePlots figures from real local smoke logs
- [x] Demo inventory + CREATIS download blocker documented
- [x] CSTM cited as Ye et al. WACV 2025
- [ ] Real ACDC subject-level results (blocked: challenge data not licensed/mounted)
- [ ] Nested CV / calibration (deferred)
