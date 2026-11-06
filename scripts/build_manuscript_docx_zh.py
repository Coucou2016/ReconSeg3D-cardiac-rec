#!/usr/bin/env python3
"""Build Chinese Word manuscript with heading-based TOC from MANUSCRIPT_FINAL_ZH.md."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor, Twips

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "docs" / "paper" / "MANUSCRIPT_FINAL_ZH.md"
FIG = ROOT / "artifacts" / "paper" / "figures"
OUT = ROOT / "docs" / "paper" / "MANUSCRIPT_FINAL_ZH.docx"

LATIN = "Times New Roman"
CJK = "SimSun"


def set_run_font(run, *, name: str = LATIN, east: str = CJK, size: int = 11, bold: bool = False, italic: bool = False):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:ascii"), name)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), east)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic


def add_para(
    doc,
    text: str,
    *,
    style: str | None = None,
    size: int = 11,
    bold: bool = False,
    italic: bool = False,
    space_after: float = 8,
    first_line: float | None = 0.35,
    align=None,
):
    p = doc.add_paragraph(style=style) if style else doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    if first_line is not None and not style:
        pf.first_line_indent = Inches(first_line)
    if align is not None:
        p.alignment = align
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold, italic=italic)
    return p


def ensure_heading_styles(doc: Document):
    for level, size in ((1, 14), (2, 12), (3, 11)):
        name = f"Heading {level}"
        try:
            style = doc.styles[name]
        except KeyError:
            style = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        style.font.name = LATIN
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        if style._element.rPr is None:
            style._element.get_or_add_rPr()
        style._element.rPr.rFonts.set(qn("w:ascii"), LATIN)
        style._element.rPr.rFonts.set(qn("w:hAnsi"), LATIN)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), CJK)


def add_heading_styled(doc, text: str, level: int):
    style_name = f"Heading {level}"
    p = doc.add_paragraph(style=style_name)
    pf = p.paragraph_format
    pf.space_before = Pt(14 if level == 1 else 10)
    pf.space_after = Pt(6)
    pf.first_line_indent = Inches(0)
    # clear default run if any
    if p.runs:
        p.runs[0].text = text
        set_run_font(p.runs[0], size={1: 14, 2: 12, 3: 11}.get(level, 11), bold=True)
    else:
        run = p.add_run(text)
        set_run_font(run, size={1: 14, 2: 12, 3: 11}.get(level, 11), bold=True)
    return p


def add_toc_field(paragraph):
    """Insert a Word TOC field that updates via right-click → Update Field."""
    run = paragraph.add_run()
    r = run._r

    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")

    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = ' TOC \\o "1-2" \\h \\z \\u '

    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")

    r.append(fld_begin)
    r.append(instr)
    r.append(fld_sep)

    placeholder = paragraph.add_run("（请在 Microsoft Word 中右键目录并选择“更新域”以刷新页码）")
    set_run_font(placeholder, size=10, italic=True)

    run_end = paragraph.add_run()
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run_end._r.append(fld_end)


def add_static_toc(doc, headings: list[tuple[int, str]]):
    """Visible generated TOC (heading text) so the deliverable is readable before field update."""
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(8)
    r = title.add_run("目录")
    set_run_font(r, size=14, bold=True)

    note = doc.add_paragraph()
    note.paragraph_format.space_after = Pt(6)
    note.paragraph_format.first_line_indent = Inches(0)
    nr = note.add_run("下列条目由一级、二级标题自动生成。文中亦嵌入可更新的 Word 目录域。")
    set_run_font(nr, size=9, italic=True)

    for level, text in headings:
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_after = Pt(2)
        pf.first_line_indent = Inches(0)
        pf.left_indent = Inches(0.25 * (level - 1))
        tab_stops = pf.tab_stops
        tab_stops.add_tab_stop(Inches(6.0), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
        run = p.add_run(text)
        set_run_font(run, size=11 if level == 1 else 10, bold=(level == 1))

    # TOC field for Word update
    field_para = doc.add_paragraph()
    field_para.paragraph_format.space_before = Pt(8)
    field_para.paragraph_format.first_line_indent = Inches(0)
    add_toc_field(field_para)

    # page break after TOC
    doc.add_page_break()


def add_figure(doc, stem: str, caption: str, width: float = 6.2):
    png = FIG / f"{stem}.png"
    if not png.exists():
        add_para(doc, f"[缺少图件: {stem}]", first_line=0, italic=True)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run()
    run.add_picture(str(png), width=Inches(width))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(10)
    cap.paragraph_format.first_line_indent = Inches(0)
    r = cap.add_run(caption)
    set_run_font(r, size=10, italic=True)


def parse_table_block(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            continue
        rows.append(cells)
    return rows


def add_table(doc, rows: list[list[str]]):
    if not rows:
        return
    ncols = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=ncols)
    table.style = "Table Grid"
    for i, row in enumerate(rows):
        for j in range(ncols):
            cell = table.cell(i, j)
            text = row[j] if j < len(row) else ""
            cell.text = ""
            p = cell.paragraphs[0]
            run = p.add_run(text)
            set_run_font(run, size=9, bold=(i == 0))
    doc.add_paragraph()


FIGURE_INSERTS = {
    "图1概括流水线角色。": (
        "fig1_pipeline",
        "图1. 稀疏短轴电影输入、逐帧重建、几何感知运动项，以及带可选头的舒张末期与收缩末期分割流水线示意。",
    ),
    "图2与表2汇总冒烟消融表中五臂的重建峰值信噪比、平均绝对误差与平均 Dice。": (
        "fig2_recon_ablation",
        "图2. 演示重建消融。柱状图为冒烟消融表中广播、逐帧与融合臂的峰值信噪比。",
    ),
    "图3与表3报告联合演示评价中的坐标级几何量。": (
        "fig3_motion_metrics",
        "图3. 演示运动与几何日志。图面报告联合评价记录中的逆一致性、周期合成、平滑、雅可比与结构分。",
    ),
    "图4给出同一消融 CSV 的类别 Dice。": (
        "fig4_seg_dice",
        "图4. 冒烟消融表中跨消融臂与心脏类别的演示分割 Dice 热图。",
    ),
}


def clean_inline(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = text.replace(r"\( ", "(").replace(r" \)", ")")
    return text


def collect_headings(lines: list[str]) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    skip_meta = True
    for raw in lines:
        raw = raw.rstrip()
        if skip_meta:
            if raw.startswith("# "):
                skip_meta = False
            continue
        if raw.startswith("## "):
            out.append((1, raw[3:].strip()))
        elif raw.startswith("### "):
            out.append((2, raw[4:].strip()))
    return out


def build(md_path: Path = MD, out_path: Path = OUT) -> Path:
    md = md_path.read_text(encoding="utf-8")
    lines = md.splitlines()
    headings = collect_headings(lines)

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)

    style = doc.styles["Normal"]
    style.font.name = LATIN
    style.font.size = Pt(11)
    style._element.rPr.rFonts.set(qn("w:ascii"), LATIN)
    style._element.rPr.rFonts.set(qn("w:hAnsi"), LATIN)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), CJK)

    ensure_heading_styles(doc)

    i = 0
    title_done = False
    toc_inserted = False
    in_table = False
    table_buf: list[str] = []
    skip_meta = True
    claim_boundary_inserted = False

    while i < len(lines):
        line = lines[i]
        raw = line.rstrip()

        if skip_meta:
            if raw.startswith("# "):
                skip_meta = False
            else:
                i += 1
                continue

        if raw.startswith("---"):
            i += 1
            continue

        if not title_done and raw.startswith("**") and ("：" in raw or ":" in raw):
            add_para(doc, clean_inline(raw.strip("*")), size=10, italic=True, first_line=0, space_after=2)
            i += 1
            continue

        if raw.startswith("# "):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(12)
            run = p.add_run(raw[2:].strip())
            set_run_font(run, size=16, bold=True)
            title_done = True
            i += 1
            continue

        if raw.startswith("## "):
            if in_table:
                add_table(doc, parse_table_block(table_buf))
                table_buf, in_table = [], False
            if title_done and not toc_inserted:
                add_static_toc(doc, headings)
                toc_inserted = True
            heading = raw[3:].strip()
            add_heading_styled(doc, heading, 1)
            if heading.startswith("6.") and not claim_boundary_inserted:
                add_figure(
                    doc,
                    "fig5_claim_boundary",
                    "图5. 主张边界账本。区分演示工程核验、授权受试者表与私有急性心肌梗死风险评价。",
                )
                claim_boundary_inserted = True
            i += 1
            continue

        if raw.startswith("### "):
            if in_table:
                add_table(doc, parse_table_block(table_buf))
                table_buf, in_table = [], False
            add_heading_styled(doc, raw[4:].strip(), 2)
            i += 1
            continue

        if raw.startswith("|"):
            in_table = True
            table_buf.append(raw)
            i += 1
            continue
        elif in_table:
            add_table(doc, parse_table_block(table_buf))
            table_buf, in_table = [], False

        if not raw:
            i += 1
            continue

        if raw.startswith("$$") or raw.startswith("\\begin{equation}"):
            block = [raw]
            i += 1
            while i < len(lines) and not (
                lines[i].strip().startswith("$$") or lines[i].strip().startswith("\\end{equation}")
            ):
                block.append(lines[i])
                i += 1
            if i < len(lines):
                block.append(lines[i])
            eq = " ".join(b.strip() for b in block)
            eq = eq.replace("\\begin{equation}", "").replace("\\end{equation}", "").replace("$$", "").strip()
            add_para(doc, eq, first_line=0, italic=True, size=10)
            i += 1
            continue

        if raw.startswith("**表") or raw.startswith("**Table"):
            add_para(doc, clean_inline(raw), first_line=0, bold=True, size=10)
            i += 1
            continue

        if raw.startswith("**图") or raw.startswith("**Figure"):
            add_para(doc, clean_inline(raw), first_line=0, italic=True, size=10)
            i += 1
            continue

        text = clean_inline(raw)
        add_para(doc, text, first_line=0.35 if not re.match(r"^\d+\.\s", text) else 0.0)

        for key, (stem, cap) in FIGURE_INSERTS.items():
            if key in text:
                add_figure(doc, stem, cap)

        i += 1

    if in_table:
        add_table(doc, parse_table_block(table_buf))

    if not claim_boundary_inserted:
        add_figure(
            doc,
            "fig5_claim_boundary",
            "图5. 主张边界账本。区分演示工程核验、授权受试者表与私有急性心肌梗死风险评价。",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--md", type=Path, default=MD)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    path = build(args.md, args.out)
    print(f"Wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
