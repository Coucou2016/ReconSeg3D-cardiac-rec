#!/usr/bin/env python3
"""Build self-contained paper HTML + Chinese research report (Base64 figures)."""
from __future__ import annotations

import base64
import csv
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "artifacts" / "paper" / "figures"
REP = ROOT / "artifacts" / "report"
PAPER = ROOT / "artifacts" / "paper"
REP.mkdir(parents=True, exist_ok=True)
PAPER.mkdir(parents=True, exist_ok=True)

CSS = """
:root { --ink:#1a1a1a; --muted:#555; --line:#ccc; --bg:#fafafa; --accent:#1f4e79; }
* { box-sizing: border-box; }
body { font-family: "Times New Roman", "SimSun", "Songti SC", serif; color: var(--ink);
  line-height: 1.65; max-width: 920px; margin: 0 auto; padding: 28px 22px 64px; background: #fff; }
h1,h2,h3 { color: var(--accent); font-weight: 700; line-height: 1.25; }
h1 { font-size: 1.65rem; border-bottom: 2px solid var(--accent); padding-bottom: .35em; }
h2 { font-size: 1.25rem; margin-top: 1.8em; border-bottom: 1px solid var(--line); }
h3 { font-size: 1.05rem; margin-top: 1.2em; }
p,li { font-size: 15px; }
.cover { text-align: center; padding: 2.5rem 1rem 2rem; border: 1px solid var(--line); margin-bottom: 2rem; background: linear-gradient(180deg,#f3f7fb,#fff); }
.cover .meta { color: var(--muted); font-size: 14px; }
.toc a { color: var(--accent); text-decoration: none; }
.toc li { margin: .25em 0; }
figure { margin: 1.4em 0 1.8em; }
figure img { max-width: 100%; height: auto; border: 1px solid #e5e5e5; }
figcaption { font-size: 13px; color: var(--muted); margin-top: .55em; text-align: left; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 1em 0; }
th, td { border: 1px solid var(--line); padding: 6px 8px; text-align: left; }
th { background: #eef3f8; }
.note { background: #fff8e8; border-left: 4px solid #c9a227; padding: .7em 1em; margin: 1em 0; font-size: 14px; }
.warn { background: #fdecea; border-left: 4px solid #c0392b; padding: .7em 1em; margin: 1em 0; font-size: 14px; }
.term { background: #f4f6f8; padding: .15em .35em; border-radius: 3px; }
.small { font-size: 13px; color: var(--muted); }
hr { border: none; border-top: 1px solid var(--line); margin: 2em 0; }
@media print { body { max-width: none; } .cover { break-after: page; } }
"""


def b64_img(name: str) -> str:
    data = (FIG / name).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def load_ablation_rows() -> list[dict]:
    path = ROOT / "outputs" / "ablations_smoke_v2" / "table.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(v: str | None, nd: int = 4) -> str:
    if v is None or str(v).strip() == "":
        return "—"
    try:
        return f"{float(v):.{nd}f}"
    except ValueError:
        return html.escape(str(v))


def ablation_table_html(rows: list[dict]) -> str:
    cols = [
        ("run", "Run", 0),
        ("recon_psnr", "PSNR", 2),
        ("recon_mae", "MAE", 4),
        ("dice_mean", "Dice mean", 4),
        ("warp_error", "Warp L1", 4),
        ("cycle_error", "Image-cycle", 4),
        ("ef_proxy", "EF proxy", 4),
        ("mace_auc", "MACE AUC*", 4),
    ]
    thead = "".join(f"<th>{c[1]}</th>" for c in cols)
    body = []
    for r in rows:
        tds = []
        for key, _, nd in cols:
            if key == "run":
                tds.append(f"<td>{html.escape(r.get(key,''))}</td>")
            else:
                tds.append(f"<td>{fmt(r.get(key), nd if nd else 4)}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def figure_block(stem_file: str, title: str, explain: str) -> str:
    src = b64_img(stem_file)
    return f"""
<figure id="{stem_file}">
  <img src="{src}" alt="{html.escape(title)}" />
  <figcaption><strong>{html.escape(title)}</strong><br/>{explain}</figcaption>
</figure>
"""


def build_report_html(rows: list[dict]) -> str:
    fig1 = figure_block(
        "fig1_pipeline.png",
        "图1 运动一致四维重建–分割流水线示意",
        "来龙去脉：本图说明仓库默认数据流。稀疏短轴（short-axis, SA）电影磁共振（cine CMR）"
        "经逐帧三维重建得到稠密体积，再由运动网络（MotionNet）估计体素位移场，用可微扭曲（warp）"
        "与图像循环一致性（image-cycle）约束时间连贯性，最后接分割头与风险/表型头。"
        "HeartTTable-lite 仅作为融合消融，不宣称与原 npj 论文三模态成对交叉注意力等价。"
        "物理意义：把“缺层短轴栈”补成可度量的 4D 解剖与运动表示。",
    )
    fig2 = figure_block(
        "fig2_recon_ablation.png",
        "图2 重建消融（冒烟实验 / 合成数据）",
        "来龙去脉：对比广播基线（broadcast）、逐帧无运动损失、逐帧+运动损失三种配置的峰值信噪比"
        "（PSNR, Peak Signal-to-Noise Ratio）与平均绝对误差（MAE, Mean Absolute Error）。"
        "为何引入：回答“运动损失是否改变重建指标日志”。"
        "解读边界：数值来自假数据 1 epoch 冒烟，不得当作临床性能。待补充：真实 ACDC 受试者级结果。",
    )
    fig3 = figure_block(
        "fig3_motion_metrics.png",
        "图3 运动一致性与射血分数代理指标",
        "来龙去脉：Warp L1 度量扭曲后强度与目标帧差异；Image-cycle 是正向/反向扭曲后回到原图的强度误差"
        "（不是流场坐标逆一致合成）。EF proxy（ejection fraction proxy）由分数体积曲线估计。"
        "为何引入：原论文强调时空动态，本仓库用可训练运动项把“时间维”落到可审计损失上。"
        "子图 a/b 均来自 ablations_smoke_v2，属流水线自检。",
    )
    fig4 = figure_block(
        "fig4_seg_dice.png",
        "图4 分割 Dice 热图（冒烟）",
        "来龙去脉：Dice 相似系数衡量预测掩膜与真值重叠。类别包括左心室（LV）、右心室（RV）、"
        "心肌（MYO）及均值。冒烟设定下 RV 常接近 0，反映欠训练与假标签，而非算法上界。"
        "待补充：真实 ACDC 上的 Dice 与物理毫米空间 HD95（Hausdorff Distance 95th percentile）。",
    )
    fig5 = figure_block(
        "fig5_claim_boundary.png",
        "图5 创新主张边界（相对原 npj 论文）",
        "来龙去脉：明确本仓库可主张的公开重建/分割/运动一致 4D，与不得主张的私有 AMI 五年 MACE AUC 0.934。"
        "HeartTTable-lite 标为消融级支持。该图用于防止审稿与汇报中的过度声称。",
    )

    table = ablation_table_html(rows)
    recon = json.loads((ROOT / "outputs" / "paper_recon_smoke" / "metrics.json").read_text(encoding="utf-8"))
    acdc = json.loads((ROOT / "outputs" / "paper_acdc_smoke" / "metrics.json").read_text(encoding="utf-8"))

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>ReconSeg3D 运动一致四维心脏重建研究汇报</title>
<style>{CSS}</style>
</head>
<body>
<section class="cover">
  <h1>运动一致四维心脏磁共振重建与分割<br/>（ReconSeg3D 方法学扩展）研究汇报</h1>
  <p class="meta">工作区：E:\\Projects\\20260523-ReconSeg3D-cardiac-rec</p>
  <p class="meta">日期：2026-08-16 · 双代理流程（Cursor 主责 / ChatGPT Plus 顾问）</p>
  <p class="meta">公开代码（供顾问拉取）：见文末 GitHub URL · 版本：冒烟/演示指标 · 不含私有 AMI 临床结论</p>
</section>

<div class="warn"><strong>声明：</strong>全文定量结果若无特别标注，均来自本地合成/自动假数据冒烟实验，
<strong>不得</strong>引用为临床性能，亦<strong>不得</strong>声称原论文私有 AMI 五年 MACE AUC = 0.934。
顾问本回合若无法实时对话，已将公开 GitHub 仓库 URL 写入 handoff，供其自行 fetch 代码与文档。</div>

<nav class="toc">
<h2>目录</h2>
<ol>
  <li><a href="#abstract">摘要</a></li>
  <li><a href="#bg">研究背景与目的</a></li>
  <li><a href="#data">数据、术语与评价协议</a></li>
  <li><a href="#methods">思路与方法</a></li>
  <li><a href="#process">研究与工程过程（来龙去脉）</a></li>
  <li><a href="#results">结果展示与图表解读</a></li>
  <li><a href="#discuss">讨论</a></li>
  <li><a href="#concl">主要结论</a></li>
  <li><a href="#limit">局限、待补充与后续计划</a></li>
  <li><a href="#advisor">顾问通道与公开仓库</a></li>
</ol>
</nav>

<h2 id="abstract">1. 摘要</h2>
<p>本汇报对应一个面向公开数据代理与可复现流水线的 <span class="term">ReconSeg3D</span> 轻量实现。
<strong>研究问题：</strong>稀疏短轴电影 CMR 如何在公开、可审计设定下建成“时间维可训练”的 4D 重建–分割栈，而不是只报单帧重建或不可复现的私有 MACE 数字。
<strong>核心创新叙事：</strong>运动一致的四维（4D）重建–分割——逐帧三维解码、可微扭曲、图像循环一致性，以及可选的体积曲线/射血分数代理。
多模态融合模块以 <span class="term">HeartTTable-lite</span> 形式保留，<strong>仅作融合消融</strong>。
本地冒烟实验已贯通重建 PSNR/MAE、Dice、warp/cycle 等日志；真实 ACDC/MM-WHS/EMIDEC 受试者级表格仍为<strong>待补充</strong>。</p>

<h2 id="bg">2. 研究背景与目的</h2>
<h3>2.1 临床与成像背景</h3>
<p>短轴电影 CMR 以二维切片栈采集，层间间隙与过平面运动导致三维解剖不完整。
要把心动周期写成可度量的解剖+功能表示，需要从稀疏观测恢复稠密时空体积（4D）。</p>
<h3>2.2 原论文边界（必须诚实）</h3>
<p>Gao 等（npj Digital Medicine, 2026; DOI: 10.1038/s41746-026-02449-0）提出 ReconSeg3D
重建双心室序列，并以 HeartTTable 融合影像与表格变量预测 AMI 术后五年
<span class="term">MACE</span>（Major Adverse Cardiovascular Events，主要不良心血管事件），
在私有队列上报告时间依赖 AUC 0.934。该数字与私有数据绑定，本仓库<strong>不复制、不宣称</strong>。</p>
<h3>2.3 方法学空隙与本工作目的</h3>
<p>若干轻量复现将单一重建体沿时间广播，或缺少显式运动损失，使“时空”主张难以审计。
邻近文献中，联合重建–运动–分割（如 Qian 等 ISBI 2024）、全序列 4D 分割记忆网络（CSTM, arXiv:2410.23191）
以及稀疏 cine 到 4D 网格重建（CineMesh4D 等）都强调时间连续性，但公开可复现的“轻量 ReconSeg3D + 显式
image-cycle 运动合同”仍缺少透明消融矩阵。</p>
<p><strong>目的（可验收）：</strong>(1) 把运动项写进默认可训练合同；(2) 在公开代理/冒烟上给出可审计消融；
(3) 用 methods 论文架构写作并固定主张边界；(4) 交付自包含 HTML 研究报告与 SciencePlots 插图。</p>

<h2 id="data">3. 数据与术语表</h2>
<table>
<thead><tr><th>术语</th><th>全称 / 含义</th><th>为何引入</th></tr></thead>
<tbody>
<tr><td>CMR</td><td>Cardiac Magnetic Resonance，心脏磁共振</td><td>成像模态</td></tr>
<tr><td>SA cine</td><td>Short-axis cine，短轴电影序列</td><td>临床常用稀疏观测</td></tr>
<tr><td>4D</td><td>3D 空间 + 时间</td><td>刻画心动周期解剖变化</td></tr>
<tr><td>PSNR</td><td>Peak Signal-to-Noise Ratio</td><td>重建保真度</td></tr>
<tr><td>SSIM</td><td>Structural Similarity（本仓库为全局代理）</td><td>结构相似性</td></tr>
<tr><td>Dice</td><td>重叠度系数</td><td>分割评价</td></tr>
<tr><td>HD95</td><td>Hausdorff Distance 95%</td><td>边界误差；物理毫米版待补充</td></tr>
<tr><td>Warp</td><td>基于位移场的可微重采样</td><td>运动对齐</td></tr>
<tr><td>Image-cycle</td><td>图像域往返扭曲误差</td><td>时间一致性；≠ 流场逆一致</td></tr>
<tr><td>EF proxy</td><td>射血分数代理</td><td>由分数体积曲线估计功能</td></tr>
<tr><td>HeartTTable-lite</td><td>单 CLS 对拼接 KV 的注意力融合</td><td>融合消融，非原论文对等实现</td></tr>
<tr><td>ACDC / MM-WHS / EMIDEC</td><td>公开心脏分割/结构/梗死相关基准</td><td>可公开验证代理</td></tr>
<tr><td>Cox PH</td><td>Cox Proportional Hazards</td><td>生存风险接口；私有 AMI 前不作主叙事</td></tr>
</tbody>
</table>
<div class="note">数据合同详见 docs/DATA.md。当前冒烟使用 prepare_demo_data / auto_fake；真实数据根目录未挂载时不得写“已在 ACDC 达到某某 Dice”。</div>

<h3>3.1 本机数据盘点（真实靠谱完整 — 诚实结论）</h3>
<table>
<thead><tr><th>路径</th><th>判定</th><th>证据</th></tr></thead>
<tbody>
<tr><td><code>data/acdc/</code></td><td><strong>Demo/假数据</strong></td><td>8 例；4D ≈ 0.49&nbsp;MB；形状 (32,32,16,8)；非 CREATIS 挑战集</td></tr>
<tr><td><code>data/mmwhs/</code></td><td><strong>Demo/假数据</strong></td><td>4 对极小 image/label</td></tr>
<tr><td><code>data/emidec/</code></td><td><strong>Demo/假数据</strong></td><td>4 对极小 Case</td></tr>
<tr><td><code>data/ami/</code></td><td>仅示例清单</td><td>无私有 AMI 影像/结局</td></tr>
<tr><td><code>outputs/ablations_smoke_v2/table.csv</code></td><td><strong>实测冒烟</strong></td><td>可用于 DEMO 表；禁止当临床</td></tr>
<tr><td>官方 ACDC 下载</td><td><strong>受阻</strong></td><td>需 CREATIS 注册凭证；本机未配置</td></tr>
</tbody>
</table>
<p class="small">因此：公开基准受试者级主表一律标<strong>待补充</strong>；文中凡写 “ACDC smoke” 均指本地假树流水线，不是挑战榜成绩。禁止编造私有 AMI / AUC 0.934。</p>

<h2 id="methods">4. 思路与方法</h2>
<h3>4.1 张量合同与重建</h3>
<p>张量布局固定为 <code>(B,C,T,D,H,W)</code>。默认 <code>per_frame_recon=true</code>：逐帧三维编码–解码，
避免把单帧重建沿时间广播。稀疏 SA 观测用切片掩码模拟；损失对观测体素加权。</p>
<h3>4.2 运动一致性（主创新落地）</h3>
<p>MotionNet 输出体素位移 <code>(dz,dy,dx)</code>，按网格坐标缩放。可训练项包括：
<strong>Warp L1</strong>（扭曲源帧对齐目标帧强度）、<strong>Image-cycle</strong>（正/反向扭曲回到原图的强度误差；
<strong>不是</strong>流场坐标逆一致合成）、可选分数体积曲线平滑与 EF proxy。</p>
<h3>4.3 分割、任务头与融合消融</h3>
<p>紧凑 3D UNet 风格分割头输出 LV/RV/MYO（可选 scar）。任务头可切换 mace / phenotype / cox。
融合：<code>concat</code> 或 <code>heart_ttable</code>（lite：单 CLS 对拼接 KV）。HeartTTable-lite <strong>不是</strong>产品主叙事。</p>
{fig1}

<h2 id="process">5. 研究与工程过程（来龙去脉）</h2>
<ol>
<li><strong>主张边界先于代码炫技：</strong>公开运动一致 4D 为主；拒绝 0.934 与完整 HeartTTable 对等宣称（docs/PAPER_PLAN.md）。</li>
<li><strong>实现/加固：</strong>Cox 共享有限掩码；运动体积分数化防爆炸；HeartTTable risk_logits 路径；epoch 池化 AUC（见 20260816 审计响应）。</li>
<li><strong>冒烟消融矩阵：</strong><code>outputs/ablations_smoke_v2</code>（broadcast / PF±motion / concat / HTT-lite / phenotype）；<code>summarize_runs</code> 汇总。</li>
<li><strong>写作与出图：</strong>nature-writing methods 轴；SciencePlots（Times New Roman + CJK 回退）重绘图1–5；自包含 HTML 研究报告。</li>
<li><strong>顾问通道：</strong>既往 ChatGPT Plus 审计已采纳要点；五轮成熟化中浏览器 tab 无法保持（0/5 live）→ 独立 Web 检索 + 本地 substitute rounds + 就绪粘贴提示；公开 GitHub 供顾问 fetch。</li>
<li><strong>文献核对：</strong>Qian ISBI DOI、Gao npj DOI、CSTM→Ye 等 WACV 2025 已独立核实；手稿引用已更新。</li>
</ol>

<h2 id="results">6. 结果展示与图表解读</h2>
<p>下表为 <code>ablations_smoke_v2</code> 原始冒烟指标（*MACE AUC 在合成标签上无临床意义；仅流水线自检）。</p>
{table}
<p class="small">paper_recon_smoke：PSNR={recon.get('recon_psnr'):.3f}，Dice mean={recon.get('dice_mean'):.4f}，
warp={recon.get('warp_error'):.6f}，cycle={recon.get('cycle_error'):.6f}。<br/>
paper_acdc_smoke：PSNR={acdc.get('recon_psnr'):.3f}，phenotype_acc={acdc.get('phenotype_acc')}，
phenotype_auc={acdc.get('phenotype_auc')}（小样本不稳定，仅流水线检查）。</p>
{fig2}
{fig3}
{fig4}
{fig5}

<h3>6.1 内嵌示意：冒烟 PSNR 对比（SVG，无 CDN）</h3>
<svg viewBox="0 0 420 160" width="100%" height="160" role="img" aria-label="PSNR bars">
  <text x="10" y="18" font-size="12" fill="#1f4e79">Demo PSNR (dB) — broadcast / PF-no-mot / PF+mot</text>
  <rect x="40" y="50" width="18.9" height="90" fill="#4C72B0"/>
  <rect x="160" y="49" width="19.0" height="91" fill="#4C72B0"/>
  <rect x="280" y="49" width="19.0" height="91" fill="#4C72B0"/>
  <text x="30" y="155" font-size="10">19.08</text>
  <text x="150" y="155" font-size="10">19.15</text>
  <text x="270" y="155" font-size="10">19.15</text>
</svg>
<p class="small">上图为与表中数值一致的静态 SVG 示意。完整排版图见图2 PNG（Base64）。冒烟下 PSNR 接近不代表临床增益。</p>

<h2 id="discuss">7. 讨论</h2>
<p>相对“只报一个重建 Dice/AUC”，运动项使时间一致性可训练、可消融、可审计。
与 CSTM 等全序列分割网络相比，本仓库的贡献不在最大精度，而在<strong>轻量 ReconSeg3D 合同上显式落地
warp + image-cycle</strong>，并公开消融与主张边界图。</p>
<p><strong>审稿风险：</strong>把 image-cycle 误写成 inverse-consistent flow；把 HeartTTable-lite 写成原论文复现；
把冒烟 AUC/Dice 写成临床结论。正确叙事是 methods + public proxies + 明确边界。</p>

<h2 id="concl">8. 主要结论</h2>
<p>(1) 运动一致 4D 可作为本仓库可辩护的主创新；(2) HeartTTable-lite 仅消融；
(3) 冒烟矩阵证明日志与训练路径贯通，但<strong>不是</strong>公开基准上界；(4) 私有 AMI 0.934 永远不是本仓库结果。</p>

<h2 id="limit">9. 局限、待补充与后续计划</h2>
<ul>
<li>真实 ACDC/MM-WHS/EMIDEC 未挂载 → 公开主表<strong>待补充</strong>。</li>
<li>物理毫米 HD95、嵌套交叉验证、校准曲线：<strong>待补充</strong>。</li>
<li>骨干为紧凑 CNN/UNet，非原论文 256³ ViT/nnU-Net 算力对等。</li>
<li>本回合 Cursor 浏览器 MCP 无法保持 tab，ChatGPT Plus 实时粘贴<strong>受阻</strong>（非用户未登录结论）。</li>
</ul>

<h2 id="advisor">10. 顾问通道与公开仓库</h2>
<p>既往对话：https://chatgpt.com/c/6a808651-37cc-83ea-a63d-2d2539a48d07 。
本回合拟粘贴“methods 架构 + 运动一致 4D 创新 + HeartTTable-lite 仅消融 + 禁 0.934”并启用网页搜索；
因浏览器 MCP 失败改为独立检索 + 公开 GitHub 供顾问拉取。详见
<code>artifacts/chatgpt_handoff/reports/</code>。</p>

<hr/>
<p class="small">插图：scripts/make_paper_figures.py + SciencePlots（字号加大；Times New Roman + SimSun 回退）并 Base64 内嵌。
配套 Markdown：report.md。英文稿：docs/paper/MANUSCRIPT_DRAFT.md。无外部 CDN。</p>
</body>
</html>
"""


def build_paper_html() -> str:
    md = (ROOT / "docs" / "paper" / "MANUSCRIPT_DRAFT.md").read_text(encoding="utf-8")
    # Minimal markdown→HTML: escape and preserve paragraphs; embed figures at end.
    body = html.escape(md).replace("\n\n", "</p><p>").replace("\n", "<br/>\n")
    figs = "".join(
        figure_block(fn, title, cap)
        for fn, title, cap in [
            ("fig1_pipeline.png", "Figure 1", "Pipeline schematic (SciencePlots)."),
            ("fig2_recon_ablation.png", "Figure 2", "Reconstruction smoke ablation."),
            ("fig3_motion_metrics.png", "Figure 3", "Motion proxies (demo)."),
            ("fig4_seg_dice.png", "Figure 4", "Dice heatmap (demo)."),
            ("fig5_claim_boundary.png", "Figure 5", "Claim boundary vs original paper."),
        ]
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Motion-consistent 4D ReconSeg3D — Manuscript Draft</title>
<style>{CSS}</style>
</head>
<body>
<div class="warn">Manuscript HTML rendering of MANUSCRIPT_DRAFT.md with SciencePlots figures embedded as Base64. Smoke metrics only.</div>
<article><p>{body}</p></article>
<hr/>
<h2>Figures</h2>
{figs}
</body>
</html>
"""


def build_report_md(rows: list[dict], github_url: str = "") -> str:
    recon = json.loads((ROOT / "outputs" / "paper_recon_smoke" / "metrics.json").read_text(encoding="utf-8"))
    acdc = json.loads((ROOT / "outputs" / "paper_acdc_smoke" / "metrics.json").read_text(encoding="utf-8"))
    gh_line = github_url or "（待 push 后回填）"
    lines = [
        "# 运动一致四维心脏磁共振重建与分割（ReconSeg3D）研究汇报",
        "",
        "> **声明：** 冒烟/演示指标；不得声称私有 AMI AUC 0.934。插图：`artifacts/paper/figures/`（SciencePlots）。",
        "> 自包含 HTML：`artifacts/report/report.html`。公开代码：" + gh_line,
        "",
        "## 1. 摘要",
        "",
        "本汇报对应面向公开数据代理的轻量 **ReconSeg3D** 实现。研究问题：如何在公开、可审计设定下",
        "建成“时间维可训练”的 4D 重建–分割栈。**核心创新：** 运动一致四维——逐帧三维解码、可微 warp、",
        "image-cycle，以及可选体积曲线/EF proxy。**HeartTTable-lite 仅作融合消融。**",
        "真实 ACDC/MM-WHS/EMIDEC 受试者级主表仍为**待补充**。",
        "",
        "## 2. 研究背景与目的",
        "",
        "### 2.1 成像背景",
        "",
        "短轴电影 CMR 为二维切片栈；层间间隙与过平面运动使三维解剖不完整，需要稠密 4D 表示。",
        "",
        "### 2.2 原论文边界",
        "",
        "Gao 等（npj Digit. Med. 2026；DOI `10.1038/s41746-026-02449-0`）在私有 AMI 队列报告五年",
        "MACE 时间依赖 AUC **0.934**。该数字绑定私有数据与完整 HeartTTable，本仓库**不复制、不宣称**。",
        "",
        "### 2.3 方法学空隙与目的",
        "",
        "轻量复现常广播单帧重建或缺少显式运动损失。邻近工作（Qian 等 ISBI 2024 联合重建–运动–分割；",
        "CSTM arXiv:2410.23191 全序列 4D 分割；CineMesh4D 稀疏 cine→4D mesh）强调时间连续性。",
        "本工作目的：(1) 默认可训练运动合同；(2) 公开/冒烟消融矩阵；(3) methods 论文架构与主张边界；",
        "(4) 自包含 HTML 研究报告 + SciencePlots 插图。",
        "",
        "## 3. 数据、术语与评价协议",
        "",
        "| 术语 | 含义 | 为何引入 |",
        "|---|---|---|",
        "| CMR | 心脏磁共振 | 成像模态 |",
        "| SA cine | 短轴电影序列 | 稀疏观测 |",
        "| 4D | 3D+时间 | 心动周期 |",
        "| PSNR / MAE | 重建保真度 | 表 T1 |",
        "| Dice / HD95 | 分割重叠 / 边界 | HD95 物理毫米版待补充 |",
        "| Warp / Image-cycle | 运动对齐 / 图像往返误差 | ≠ 流场逆一致 |",
        "| EF proxy | 射血分数代理 | 功能曲线 |",
        "| HeartTTable-lite | 单 CLS×拼接 KV | 融合消融 |",
        "| ACDC / MM-WHS / EMIDEC | 公开基准 | 可验证代理 |",
        "",
        "数据合同见 `docs/DATA.md`。未挂载真实数据时不得写“已在 ACDC 达到某某 Dice”。",
        "",
        "### 3.1 本机数据盘点",
        "",
        "| 路径 | 判定 | 证据 |",
        "|---|---|---|",
        "| `data/acdc/` | **Demo/假** | 8 例；4D≈0.49MB；(32,32,16,8) |",
        "| `data/mmwhs/` / `emidec/` | **Demo/假** | 各 4 对极小 NIfTI |",
        "| `data/ami/` | 示例清单 | 无私有 AMI |",
        "| `outputs/ablations_smoke_v2/table.csv` | **实测冒烟** | DEMO 表可用 |",
        "| 官方 ACDC | **受阻** | 需 CREATIS 注册 |",
        "",
        "## 4. 思路与方法",
        "",
        "- 张量 `(B,C,T,D,H,W)`；默认 `per_frame_recon=true`。",
        "- MotionNet `(dz,dy,dx)` + Warp L1 + Image-cycle + 可选体积平滑。",
        "- 分割 LV/RV/MYO；任务头 mace/phenotype/cox；融合 concat 或 heart_ttable（lite）。",
        "- **图1**（`fig1_pipeline.png`）：流水线示意——稀疏 SA → 逐帧 3D recon → 运动 → 分割/风险；",
        "  HeartTTable-lite 脚注为消融。物理意义：把缺层短轴栈补成可度量 4D 表示。",
        "",
        "## 5. 研究与工程过程（来龙去脉）",
        "",
        "1. 主张边界先于实现（`docs/PAPER_PLAN.md`）。",
        "2. 加固：Cox 掩码、分数体积 volsmooth、risk_logits、epoch 池化 AUC。",
        "3. 冒烟消融 `outputs/ablations_smoke_v2`（6 配置）。",
        "4. nature-writing methods 轴 + SciencePlots 图1–5 + 本报告 bundle。",
        "5. 顾问：既往 Plus 审计已采纳；五轮成熟化 live ChatGPT **0/5**（MCP tab 消失）→ 独立检索 + `rounds/` substitute + READY_PASTE_PROMPTS。",
        "6. 文献：Qian DOI、Gao DOI、CSTM=Ye et al. WACV 2025 已核实并写入手稿。",
        "",
        "## 6. 结果展示与图表解读",
        "",
        "### 6.1 冒烟消融表（demo only）",
        "",
        "| run | PSNR | MAE | Dice mean | Warp | Cycle | EF | MACE AUC* |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {run} | {psnr} | {mae} | {dice} | {w} | {c} | {ef} | {auc} |".format(
                run=r.get("run", ""),
                psnr=fmt(r.get("recon_psnr"), 2),
                mae=fmt(r.get("recon_mae"), 4),
                dice=fmt(r.get("dice_mean"), 4),
                w=fmt(r.get("warp_error"), 4),
                c=fmt(r.get("cycle_error"), 4),
                ef=fmt(r.get("ef_proxy"), 4),
                auc=fmt(r.get("mace_auc"), 4),
            )
        )
    lines += [
        "",
        f"paper_recon_smoke：PSNR={recon.get('recon_psnr'):.3f}，Dice mean={recon.get('dice_mean'):.4f}，"
        f"warp={recon.get('warp_error'):.6f}，cycle={recon.get('cycle_error'):.6f}。",
        f"paper_acdc_smoke：PSNR={acdc.get('recon_psnr'):.3f}，phenotype_acc={acdc.get('phenotype_acc')}，"
        f"phenotype_auc={acdc.get('phenotype_auc')}（小样本不稳定）。",
        "",
        "### 6.2 各图来龙去脉",
        "",
        "- **图1 `fig1_pipeline`**：方法角色示意图（非定量）。说明默认数据流与消融脚注。",
        "- **图2 `fig2_recon_ablation`**：子图 a PSNR、b MAE。对比 broadcast / PF-no-mot / PF+mot。",
        "  回答“运动损失是否进入重建日志”。冒烟数值接近 → **不得**解读为临床增益。真实 ACDC **待补充**。",
        "- **图3 `fig3_motion_metrics`**：子图 a Warp L1 与 Image-cycle；子图 b EF proxy。",
        "  证明时间一致性项可训练、可记录。Image-cycle ≠ inverse-consistent flow。",
        "- **图4 `fig4_seg_dice`**：LV/RV/MYO/Mean Dice 热图。冒烟下 RV≈0 反映欠训练/假标签，非算法上界。",
        "  物理毫米 HD95 **待补充**。",
        "- **图5 `fig5_claim_boundary`**：主张支持度条形图。公开 recon/seg/运动一致 4D 在范围内；",
        "  HeartTTable-lite 为部分支持（消融）；私有 AMI 0.934 **out of scope**。",
        "",
        "## 7. 讨论",
        "",
        "运动项使时间一致性可消融、可审计。贡献不在追赶最大 Dice，而在轻量 ReconSeg3D 合同上显式落地",
        "warp + image-cycle 并公开边界。审稿忌：术语偷换、lite=完整 HeartTTable、冒烟当临床。",
        "",
        "## 8. 主要结论",
        "",
        "1. 运动一致 4D 是可辩护主创新。  ",
        "2. HeartTTable-lite 仅消融。  ",
        "3. 冒烟证明流水线贯通，不是公开基准上界。  ",
        "4. 私有 AMI 0.934 永远不是本仓库结果。",
        "",
        "## 9. 局限与待补充",
        "",
        "- 真实 ACDC/MM-WHS/EMIDEC 主表：**待补充**",
        "- 物理毫米 HD95、嵌套 CV、校准曲线：**待补充**",
        "- 紧凑 CNN ≠ 原论文 256³ ViT/nnU-Net",
        "- 五轮成熟化 live ChatGPT **0/5**（浏览器 MCP tab 无法保持）；非“用户未登录”结论；就绪粘贴见 `rounds/READY_PASTE_PROMPTS.md`",
        "",
        "## 10. 顾问通道与公开仓库",
        "",
        "- 对话：https://chatgpt.com/c/6a808651-37cc-83ea-a63d-2d2539a48d07",
        f"- 公开 GitHub（代码+文档，无大 data/outputs）：{gh_line}",
        "- 五轮记录：`artifacts/chatgpt_handoff/reports/rounds/`",
        "- 验收：`artifacts/chatgpt_handoff/reports/20260816_five_round_acceptance.md`",
        "",
    ]
    return "\n".join(lines)


def try_pdf(html_path: Path, pdf_path: Path) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        return f"playwright import failed: {e}"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
            page.pdf(path=str(pdf_path), format="A4", print_background=True)
            browser.close()
        return f"wrote {pdf_path}"
    except Exception as e:
        return f"PDF failed: {e}"


def main() -> None:
    rows = load_ablation_rows()
    github_url = ""
    meta_path = ROOT / "artifacts" / "github_url.txt"
    if meta_path.exists():
        github_url = meta_path.read_text(encoding="utf-8").strip()

    report_html = build_report_html(rows)
    if github_url:
        report_html = report_html.replace(
            "见文末 GitHub URL",
            html.escape(github_url),
        )
    paper_html = build_paper_html()
    report_md = build_report_md(rows, github_url=github_url)

    (REP / "report.html").write_text(report_html, encoding="utf-8")
    (REP / "report.md").write_text(report_md, encoding="utf-8")
    (PAPER / "manuscript.html").write_text(paper_html, encoding="utf-8")
    (ROOT / "docs" / "paper" / "manuscript.html").write_text(paper_html, encoding="utf-8")

    msg1 = try_pdf(REP / "report.html", REP / "report.pdf")
    msg2 = try_pdf(PAPER / "manuscript.html", PAPER / "manuscript.pdf")
    (REP / "pdf_status.txt").write_text(msg1 + "\n" + msg2 + "\n", encoding="utf-8")
    print("report.html", (REP / "report.html").stat().st_size)
    print("report.md", (REP / "report.md").stat().st_size)
    print("manuscript.html", (PAPER / "manuscript.html").stat().st_size)
    print(msg1)
    print(msg2)


if __name__ == "__main__":
    main()
