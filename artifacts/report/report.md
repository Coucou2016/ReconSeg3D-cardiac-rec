# 运动一致四维心脏磁共振重建与分割（ReconSeg3D）研究汇报

> 冒烟/演示指标；不得声称私有 AMI AUC 0.934。插图见 `artifacts/paper/figures/`（SciencePlots）。

## 摘要

本汇报对应公开代理上的运动一致 4D ReconSeg3D 方法学扩展；HeartTTable-lite 仅作融合消融。

## 冒烟消融表（ablations_smoke_v2）

| run | PSNR | MAE | Dice mean | Warp | Cycle | EF | MACE AUC* |
|---|---:|---:|---:|---:|---:|---:|---:|
| broadcast_baseline | 19.08 | 0.6926 | 0.1682 | — | — | — | 0.6250 |
| fusion_concat | 19.09 | 0.6922 | 0.1394 | 0.0005 | 0.0001 | 0.1457 | 0.6250 |
| fusion_heart_ttable | 19.26 | 0.6895 | 0.1890 | 0.0013 | 0.0008 | 0.0676 | 0.0000 |
| per_frame_motion | 19.15 | 0.6953 | 0.1588 | 0.0011 | 0.0003 | 0.0806 | 0.6250 |
| per_frame_no_motion | 19.15 | 0.6919 | 0.1696 | 0.0010 | 0.0000 | 0.0883 | 0.6250 |
| task_phenotype | 17.39 | 0.6946 | 0.0176 | 0.0001 | 0.0001 | 0.0008 | 0.0000 |

图1–图5：`fig1_pipeline.png` … `fig5_claim_boundary.png`。

完整自包含 HTML：`artifacts/report/report.html`。

## 局限

- 真实 ACDC 等未挂载（待补充）
- 浏览器 MCP 不可用，ChatGPT 本回合未实时对话
- NO_GIT_REPO，本地交付
