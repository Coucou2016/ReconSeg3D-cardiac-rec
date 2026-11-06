#!/usr/bin/env python3
"""Build submission-style Word manuscript from MANUSCRIPT_FINAL.md + SciencePlots figures."""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "docs" / "paper" / "MANUSCRIPT_FINAL.md"
FIG = ROOT / "artifacts" / "paper" / "figures"
OUT = ROOT / "docs" / "paper" / "MANUSCRIPT_FINAL.docx"


def set_run_font(run, name: str = "Times New Roman", size: int = 11, bold: bool = False, italic: bool = False):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic


def add_para(doc, text: str, *, style: str | None = None, size: int = 11, bold: bool = False, italic: bool = False, space_after: float = 8, first_line: float | None = 0.35):
    p = doc.add_paragraph(style=style) if style else doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    if first_line is not None and not style:
        pf.first_line_indent = Inches(first_line)
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold, italic=italic)
    return p


def add_heading_custom(doc, text: str, level: int):
    sizes = {1: 14, 2: 12, 3: 11}
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(14 if level == 1 else 10)
    pf.space_after = Pt(6)
    pf.first_line_indent = Inches(0)
    run = p.add_run(text)
    set_run_font(run, size=sizes.get(level, 11), bold=True)
    return p


def add_figure(doc, stem: str, caption: str, width: float = 6.2):
    png = FIG / f"{stem}.png"
    if not png.exists():
        add_para(doc, f"[Missing figure: {stem}]", first_line=0, italic=True)
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
    "Figure 1 summarizes the pipeline roles.": (
        "fig1_pipeline",
        "Figure 1. Pipeline schematic for sparse SA cine input, per-frame reconstruction, geometry-aware motion terms, and ED/ES segmentation with optional heads.",
    ),
    "Figure 2 and Table 2 summarize reconstruction PSNR, MAE, and mean Dice across five ablation arms from the smoke ablation table.": (
        "fig2_recon_ablation",
        "Figure 2. DEMO reconstruction ablation. Bars show PSNR across broadcast, per-frame, and fusion arms from the smoke ablation table.",
    ),
    "Figure 3 and Table 3 report coordinate-level geometry from the joint demo evaluation.": (
        "fig3_motion_metrics",
        "Figure 3. DEMO motion and geometry logs from the joint evaluation record.",
    ),
    "Figure 4 shows class-wise Dice from the same ablation CSV.": (
        "fig4_seg_dice",
        "Figure 4. DEMO segmentation Dice heatmap across ablation arms and cardiac classes from the smoke ablation table.",
    ),
}


def clean_inline(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = text.replace(r"\( ", "(").replace(r" \)", ")")
    return text


def build() -> Path:
    md = MD.read_text(encoding="utf-8")
    lines = md.splitlines()

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")

    i = 0
    title_done = False
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

        # metadata lines under title
        if not title_done and raw.startswith("**") and ":" in raw:
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
            heading = raw[3:].strip()
            add_heading_custom(doc, heading, 1)
            if heading.startswith("6.") and not claim_boundary_inserted:
                add_figure(
                    doc,
                    "fig5_claim_boundary",
                    "Figure 5. Claim boundary ledger separating DEMO engineering verification, licensed subject tables, and private AMI risk evaluation.",
                )
                claim_boundary_inserted = True
            i += 1
            continue

        if raw.startswith("### "):
            if in_table:
                add_table(doc, parse_table_block(table_buf))
                table_buf, in_table = [], False
            add_heading_custom(doc, raw[4:].strip(), 2)
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
            # collect equation block
            block = [raw]
            i += 1
            while i < len(lines) and not (lines[i].strip().startswith("$$") or lines[i].strip().startswith("\\end{equation}")):
                block.append(lines[i])
                i += 1
            if i < len(lines):
                block.append(lines[i])
            eq = " ".join(b.strip() for b in block)
            eq = eq.replace("\\begin{equation}", "").replace("\\end{equation}", "").replace("$$", "").strip()
            add_para(doc, eq, first_line=0, italic=True, size=10)
            i += 1
            continue

        if raw.startswith("**Table"):
            add_para(doc, clean_inline(raw), first_line=0, bold=True, size=10)
            i += 1
            continue

        if raw.startswith("**Figure"):
            add_para(doc, clean_inline(raw), first_line=0, italic=True, size=10)
            i += 1
            continue

        # numbered refs / conclusions
        text = clean_inline(raw)
        # strip markdown list markers for numbered contributions already written as "1. "
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
            "Figure 5. Claim boundary ledger separating DEMO engineering verification, licensed subject tables, and private AMI risk evaluation.",
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path} ({path.stat().st_size} bytes)")
