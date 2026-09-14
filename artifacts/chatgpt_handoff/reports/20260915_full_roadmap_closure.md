# Full roadmap closure (2026-09-15)

Exhaustive checklist for round-1/round-2 leftovers + PAPER_PLAN Stage-2/3 items.
Honesty: no invented clinical/table numbers; licensed ACDC/M&Ms subject tables remain **待补充**.

## Tests

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v
============================= 96 passed =============================
```

## A. Tiny leftovers

| ID | Item | Status |
|----|------|--------|
| A1 | `MultiTaskLoss` docstring: `w_loop` = closed-cycle / `L_periodic` | **Done** |
| A2 | Clean misleading `task: mace` where unused; honest task + selection | **Done** (default/smoke/paper_recon/ablations; HeartTTable keeps `mace` when `w_mace>0`) |
| A3 | Remove dangerous `recon_ssim` alias; only `recon_ssim_proxy` / `ssim_3d` | **Done** (metrics + figure_meta rename) |

## B. Stage-2 evaluation infrastructure

| ID | Item | Status |
|----|------|--------|
| B1 | 3D local/windowed SSIM → `ssim_3d` + tests | **Done** |
| B2 | Physical EDV/ESV/EF (mL/%) via spacing + ED/ES; proxy labeled | **Done** |
| B3 | Patient-level pipeline → `case_metrics.csv` / `summary_metrics.json` / `bootstrap_ci.json` | **Done** (`scripts/eval.py`) |
| B4 | Trainer sample-weighted aggregation (not naive batch-mean) | **Done** |
| B5 | 5-fold diagnosis-stratified ACDC splits + loader | **Done** (`splits/acdc_fold{0-4}.json`, `fold`/`fold_file`) |
| B6 | Multi-seed harness (3–5 seeds) | **Done** (`scripts/run_multiseed.py`) |
| B7 | Publication vs smoke vs proxy config hygiene; hires configs; deprecate `paper_*` as formal | **Done** |
| B8 | Physical sparse SA (`slice_trans_mm`, `sampling_ratio`) | **Done** |
| B9 | Eval test split + phase indices/spacing | **Done** |

## C. Stage-3 methods / baselines

| ID | Item | Status |
|----|------|--------|
| C1 | Temporal backends: TemporalConv / ConvLSTM3D / temporal attention + configs | **Done** |
| C2 | SVF usable (`use_svf` + scaling-and-squaring; `publication_motion_svf.yaml`) | **Done** |
| C3a | Recon-only / no-motion baseline | **Done** |
| C3b | Compact VoxelMorph-style in-repo | **Done** |
| C3c | FlowReg adapter + README (not vendored; no fake numbers) | **Done** (`docs/BASELINES.md`) |
| C4 | Related-work positioning locked (Neural ODE, TetHeart, FlowReg, VoxelMorph, TransMorph) | **Done** |
| C5 | M&Ms loader stub + publication hard-fail + download docs | **Done** |

## D. Docs / hygiene

| ID | Item | Status |
|----|------|--------|
| D1 | README architecture sync | **Done** |
| D2 | PAPER_PLAN Done vs externally blocked | **Done** |
| D3 | LICENSE/CITATION | **OK** (unchanged; still accurate) |
| D4 | SciencePlots figure_meta metrics key rename | **Done** (historical DEMO values; `ssim_3d` null until re-run) |
| D5 | ChatGPT brief MD | **Done** (see github_briefs) |

## E. Tests & push

| ID | Item | Status |
|----|------|--------|
| E1 | Expanded tests (SSIM3D, EF mL, folds, aggregate, docstring, selection) | **Done** |
| E2 | Full pytest green | **Done** (96 passed) |
| E3 | Commit + push (code/docs; no huge data) | **Done** (this turn) |

## Externally blocked (infra complete; fail loud / 待补充)

| Item | Reason |
|------|--------|
| Real ACDC subject-level publication tables | CREATIS / challenge license + multi-GB download not on disk |
| M&Ms multi-site tables | Download wall / credentials; loader hard-fails in publication mode |
| FlowReg / TransMorph / MulViMotion **numeric** comparisons | Need external install + licensed data; adapters/docs only |
| Full 256³ training wall-clock | GPU + licensed data; `publication_recon_hires.yaml` provided |
| Private AMI / 0.934 | Out of scope — never claimed |

## Config map (honest)

| Role | Path |
|------|------|
| Smoke/CI | `configs/smoke/smoke_motion.yaml` |
| Publication | `configs/publication/*.yaml` (+ root shims) |
| Proxy phenotype | `configs/proxy/proxy_acdc_phenotype.yaml` |
| DEMO shim | `configs/paper_recon.yaml`, `paper_acdc.yaml` — **not** formal results |
| Baselines | `configs/baselines/` |
| Temporal ablations | `configs/ablations/temporal_*.yaml` |

## Non-claims

- No 0.934 / private AMI.
- No invented Dice/HD95/EF on missing licensed data.
- Smoke/DEMO metrics stay labeled.
