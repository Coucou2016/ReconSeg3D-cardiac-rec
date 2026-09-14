# ReconSeg3D-cardiac-rec

**Geometry- and motion-constrained 4D reconstruction/segmentation from sparse SA CMR** (public methods extension).  
稀疏短轴 CMR 的几何/运动约束 4D 重建与分割方法扩展。

This repository is a **public methods codebase**. Main line = recon + seg + inverse-consistent motion + function proxies.  
It is **not** a claim of the original paper’s private-AMI **0.934 AUC**. HeartTTable / Cox / MACE heads are **supplementary**.

## 项目简介 / Overview

1. **Per-frame 3D reconstruction** `(B, C, T, D, H, W)` from sparse SA (broadcast ablation retained)
2. **Geometry-aware motion**: true `L_inv`, smoothness, Jacobian folding, **closed-cycle** `L_periodic` (T pairs incl. `T-1→0`); image-cycle and volume-curve are auxiliary / physiological only
3. **ED/ES-aware segmentation** (ACDC labeled phases; anchor-preserving temporal subsample; unlabeled frames via motion self-supervision)
4. **Optional** phenotype / Cox / MACE / HeartTTable-lite (off in `configs/publication_*.yaml`; phenotype proxy under `configs/proxy/`)

See [docs/PAPER_PLAN.md](docs/PAPER_PLAN.md) for claim boundaries. Demo metrics are **DEMO-labeled**.

## 安装 / Install

```powershell
cd E:\Projects\20260523-ReconSeg3D-cardiac-rec
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## 快速训练 / Quick train (synthetic, per-frame + motion)

```powershell
python scripts/train.py --config configs/default.yaml --epochs 2 --output-dir outputs
```

Sparse-SA recon + seg + motion losses:

```powershell
python scripts/train.py --config configs/paper_recon.yaml --epochs 2 --output-dir outputs/paper_recon
```

ACDC phenotype proxy (fake or real root — see [docs/DATA.md](docs/DATA.md)):

```powershell
python scripts/prepare_demo_data.py
python scripts/train.py --config configs/proxy/proxy_acdc_phenotype.yaml --epochs 2 --output-dir outputs/proxy_acdc
# shim still works: configs/paper_acdc.yaml
```

## Paper experiments (demo)

Local Table-style ablations **without private AMI** (fake ACDC/MM-WHS/EMIDEC under `data/`):

```powershell
python scripts/prepare_demo_data.py
python scripts/train.py --config configs/paper_recon.yaml --epochs 1 --output-dir outputs/paper_recon_smoke
python scripts/train.py --config configs/paper_acdc.yaml --epochs 1 --output-dir outputs/paper_acdc_smoke
python scripts/run_ablations.py --epochs 1 --output-root outputs/ablations
python scripts/summarize_runs.py --runs-root outputs/ablations --out-md outputs/ablations/table.md --out-csv outputs/ablations/table.csv
```

Ablation YAMLs live in `configs/ablations/` (broadcast vs per-frame, warp/cycle on/off, concat vs HeartTTable, phenotype).  
Eval writes `outputs/<run>/metrics.json`. **Do not claim private-AMI 0.934 AUC** from these demo numbers. Details: [docs/PAPER_PLAN.md](docs/PAPER_PLAN.md).

## 评估 / Evaluate

```powershell
python scripts/eval.py --config configs/default.yaml --checkpoint outputs/last.pt
# writes <checkpoint_dir>/metrics.json by default
```

## 推理 / Predict

```powershell
python scripts/predict.py --checkpoint outputs/last.pt --config configs/default.yaml --out-dir predictions
```

Exports reconstruction (and flow when present). With `model.task: phenotype`, also prints/saves phenotype probabilities.
## 测试 / Tests

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest tests/ -v
```

（部分 Conda 环境需禁用自动插件，见 `pytest.ini`。）

## 数据格式

见 [docs/DATA.md](docs/DATA.md)。标准张量布局为 **`(B, C, T, D, H, W)`**。

Point a real ACDC download at `configs/paper_acdc.yaml`:

```yaml
data:
  source: acdc
  root: E:/data/ACDC/training
```

## 模型架构（简述）

```
Input (B,C,T,D,H,W)  [optional sparse SA zeros]
  → per-frame 3D CNN encoder + temporal mix
  → per-frame recon decoder → (B, C, T, D, H, W)
  → MotionNet 3D flow + warp/cycle
  → seg head → (B, K, D, H, W) and optional (B, K, T, D, H, W)
  → concat MLP  or  HeartTTable (spatial + temporal + table + CLS cross-attn)
       └─ MACE / Cox risk  and/or  phenotype
```

Configs: `configs/default.yaml`, `configs/paper_recon.yaml`, `configs/paper_acdc.yaml`, `configs/ablations/*.yaml`.  
`model.per_frame_recon: true` (default). Broadcast ablation: `per_frame_recon: false`.  
`model.fusion`: `concat` (ablation) | `heart_ttable`.  
`model.task`: `mace` | `cox` | `phenotype`.

Standalone compact 3D ViT-style recon / UNet seg: `reconseg3d.models.volume_recon.CompactVolumeRecon`, `volume_seg.VolumeUNet3D` (paper-scale 256×256×128 is a config comment, not the smoke default).

## 复现 / Reproducibility

- 全局种子：`seed` in config（`reconseg3d.utils.seed.set_seed`）
- Windows 建议 `data.num_workers: 0`
- 小样本验证时 MACE AUC / C-index / phenotype AUC 可能为 `NaN`（单类或无可比 pair），属预期行为
- Checkpoints store `cfg`, `run_name`, `best_auc` / `best_val_loss`; runs also write `config_snapshot.yaml` + `metrics.json`

## 限制与假设 / Limitations

| 项目 | 说明 |
|------|------|
| 合成数据 | 仅用于流水线验证，不代表真实 AMI 分布 |
| 重建 | 默认 **逐帧解码**；旧版时间维广播仍可作为消融 |
| 骨干 | Compact CNN/UNet+small transformer，不是原论文 3D ViT / nnU-Net 256³ |
| MACE AUC | **不报告 0.934**；公开验证是重建/分割与 ACDC MINF、EMIDEC scar 代理 |
| Cox / HeartTTable | 接口已就绪；5 年 MACE 需要私有 AMI 队列 |
| HD95 / SSIM | numpy 表面距离（大网格子采样）；SSIM 为全局代理 |
| NIfTI | 需安装 `nibabel`；体素间距重采样未做 scanner-faithful 配准 |

## 目录结构

```
reconseg3d/     # 模型、数据、训练、推理
configs/        # YAML（含 ablations/）
scripts/        # train / eval / predict / prepare_demo_data / run_ablations / summarize_runs
tests/          # pytest
docs/           # DATA.md, PAPER_PLAN.md
```
