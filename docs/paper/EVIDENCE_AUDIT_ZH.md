# 证据真实性审计（中文说明）

完整英中对照表见同目录 `EVIDENCE_AUDIT.md`。本文件强调中文终稿交付与数字来源。

**日期：** 2026-11-07  
**中文文稿：** `MANUSCRIPT_FINAL_ZH.md` → `MANUSCRIPT_FINAL_ZH.docx`（含目录）  
**构建脚本：** `scripts/build_manuscript_docx_zh.py`

## 硬性不主张

原文 npj 私有急性心肌梗死 **AUC 0.934**、**C-index 0.897** 及对应队列规模**不作为**本文结果。

## 本文实测数字 → 文件 → 代码

| 文稿位置 | 代表数值 | 文件 | 代码入口 |
|----------|----------|------|----------|
| 表2 消融 | PSNR≈19.08–19.26；Dice≈0.139–0.189 | `outputs/ablations_smoke_v2/table.csv` | `scripts/run_ablations.py` |
| 表2 心肌 Dice 0.567 | `dice_myo`（fusion_heart_ttable） | 同上 | 同上 |
| 表3 逆一致性 0.00177 | `inv_error` / `loss_inv` | `outputs/metrics.json` | `inverse_consistency_loss` |
| 表3 周期合成 0.00177 | `loss_loop` | 同上 | `loop_consistency_loss` |
| 表3 雅可比均值 0.9998、折叠 0 | `jac_det_mean`、`jac_neg_ratio`、`loss_jac` | 同上 | `jacobian_stats` / `folding_penalty` |
| 表3 ssim_3d 0.515 | `ssim_3d` | 同上 | 窗式 SSIM |
| ED↔ES 传播 Dice 1.0 | `prop_ed2es_*` / `prop_es2ed_*` | 同上 | 标签传播 API |
| 表4 paper_recon_smoke | PSNR 19.027 | `outputs/paper_recon_smoke/metrics.json` | `scripts/eval.py` |
| 表4 paper_acdc_smoke | PSNR 17.390；HD95_lv 15.68 | `outputs/paper_acdc_smoke/metrics.json` | 同上 |

**证据体制：** 全部标为演示或冒烟工程核验。授权受试者表未挂载。合成风险字段（如 0.625 / 0.556）仅流水线日志。

## 方法与代码对应

真实 \(L_\mathrm{inv}\)、平滑、雅可比、含 T-1→0 返回边的周期合成、舒张末期锚定路径、发表门控（`w_mace`/`w_cox`=0）分别对应 `reconseg3d/models/motion.py` 与 `losses.py` 中的具名函数与权重。

## OA 参考 PDF（结构仿写）

见 `artifacts/paper/references/` 下 VoxelMorph、Dalca MedIA、Krebs TMI、DeepTag CVPR、Frontiers 组间动态磁共振等 PDF。
