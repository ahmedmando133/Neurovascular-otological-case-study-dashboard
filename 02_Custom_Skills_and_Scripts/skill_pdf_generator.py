#!/usr/bin/env python3
"""
Skill 5: Automated Publication-Grade Clinical PDF Generator
===========================================================
Programmatically compiles structured Markdown reports, 7 Consultant Audits,
quantitative clinical telemetry, SVG vector charts, CLAHE/ROI neuroimaging,
and cropped Ophthalmology/Vascular plates into three shareable, print-ready A4 PDFs stored in:
  d:\\mum\\05_Shareable_PDF_Reports\\

Outputs:
  1. 01_MDT_Executive_Summary_and_Action_Plan.pdf  (strictly 2-page consultant handover)
  2. 02_Consultant_7_Board_Audit_Reports.pdf       (complete 7-specialist 3-block audits + evidence plates)
  3. 03_Patient_Case_Study_Full_Report.pdf         (full Fellowship monograph EN + Egyptian AR)

Uses local Chromium/Edge headless rendering (HarfBuzz + ICU BiDi for native
English/Arabic typography) paired with PyMuPDF for running headers/footers,
metadata, TOC bookmarks, and deterministic page-count verification.
"""

from __future__ import annotations

import base64
import html
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pymupdf

WORKSPACE_ROOT = Path(r"d:\mum")
OUTPUT_DIR = WORKSPACE_ROOT / "05_Shareable_PDF_Reports"
AUDITS_DIR = WORKSPACE_ROOT / "03_Individual_Consultant_Audits"
DELIVERABLES_DIR = WORKSPACE_ROOT / "04_Final_MDT_Deliverables"
SCRIPTS_DIR = WORKSPACE_ROOT / "02_Custom_Skills_and_Scripts"
ROI_DIR = (
    WORKSPACE_ROOT
    / "01_Structured_and_Renamed_Inputs"
    / "02_Extracted_and_Enhanced_MRI_Frames"
    / "04_CLAHE_ROI_Zoom_Left_Mastoid_Sigmoid_and_CPA"
)
MRA_DIR = (
    WORKSPACE_ROOT
    / "01_Structured_and_Renamed_Inputs"
    / "02_Extracted_and_Enhanced_MRI_Frames"
    / "02_MRA_Intracranial_Arteries"
)
MRV_DIR = (
    WORKSPACE_ROOT
    / "01_Structured_and_Renamed_Inputs"
    / "02_Extracted_and_Enhanced_MRI_Frames"
    / "03_MRV_Venous_Sinuses_and_Left_IJV"
)
OPHTH_DIR = (
    WORKSPACE_ROOT
    / "01_Structured_and_Renamed_Inputs"
    / "09_Ophthalmology_Fundus_and_OpticNerve"
)
VASC_LL_DIR = (
    WORKSPACE_ROOT
    / "01_Structured_and_Renamed_Inputs"
    / "10_Peripheral_Vascular_LowerLimb_Doppler"
)

PATIENT_HEADER = "Amal Mokhtar Mohamed Ibrahim (63.5yo F) | DOB: 1963-04-09 | ID: 26304090300188"

# Complete 24-hour Holter hourly data verified in skill_clinical_quantifier.py
HOLTER_HOURLY_SERIES = [
    ("09:00", 63, 9, 0, 0),
    ("10:00", 68, 15, 0, 0),
    ("11:00", 73, 69, 2, 9),
    ("12:00", 71, 17, 0, 1),
    ("13:00", 73, 60, 0, 4),   # Max -2.1 mm ST depression CH3 @ 13:11
    ("14:00", 75, 49, 0, 1),
    ("15:00", 67, 8, 1, 0),
    ("16:00", 64, 21, 1, 1),   # 1 V-run
    ("17:00", 67, 36, 0, 1),
    ("18:00", 63, 21, 19, 19),
    ("19:00", 60, 22, 4, 4),
    ("20:00", 59, 10, 3, 3),
    ("21:00", 68, 17, 1, 0),
    ("22:00", 62, 8, 3, 3),
    ("23:00", 61, 34, 6, 6),   # 4-beat V-run @ 23:26:38
    ("00:00", 65, 29, 0, 0),   # 1 V-run
    ("01:00", 56, 5, 4, 6),
    ("02:00", 57, 9, 43, 51),  # Min HR 51 bpm @ 02:06
    ("03:00", 61, 25, 3, 4),
    ("04:00", 64, 54, 0, 0),
    ("05:00", 64, 28, 0, 0),
    ("06:00", 70, 71, 0, 0),   # Morning surge starts
    ("07:00", 70, 76, 0, 1),
    ("08:00", 72, 164, 0, 0),  # Peak 164 PVCs (3.78% burden) + V-run @ 104 bpm
]

COMMON_PRINT_CSS = """
@page {
  size: A4 portrait;
  margin: 15mm 12mm 15mm 12mm;
}
* { box-sizing: border-box; }
body {
  font-family: 'Segoe UI', 'Calibri', 'Tahoma', 'Arial', sans-serif;
  color: #1e293b;
  font-size: 9.4pt;
  line-height: 1.45;
  margin: 0;
  padding: 0;
}
h1, h2, h3, h4 {
  color: #0f172a;
  margin-top: 0.7em;
  margin-bottom: 0.35em;
  break-after: avoid;
  page-break-after: avoid;
}
h1 { font-size: 15pt; border-bottom: 2px solid #0284c7; padding-bottom: 4px; }
h2 {
  font-size: 12pt;
  background: #f1f5f9;
  border-left: 4px solid #0284c7;
  padding: 5px 9px;
  margin-top: 12px;
}
h3 { font-size: 10.5pt; color: #0369a1; margin-top: 9px; }
h4 { font-size: 9.8pt; color: #334155; }
p { margin: 0 0 6px 0; }
ul, ol { margin: 3px 0 8px 0; padding-left: 18px; }
li { margin-bottom: 3px; }
[dir="rtl"] ul, [dir="rtl"] ol { padding-left: 0; padding-right: 18px; }
code {
  font-family: 'Consolas', 'Courier New', monospace;
  font-size: 8.3pt;
  background: #f1f5f9;
  color: #0f172a;
  padding: 0.5px 3.5px;
  border-radius: 3px;
  border: 1px solid #cbd5e1;
  direction: ltr;
  unicode-bidi: isolate;
  display: inline;
  box-decoration-break: clone;
  overflow-wrap: anywhere;
  word-break: break-word;
}
a { color: #0369a1; text-decoration: none; overflow-wrap: anywhere; word-break: break-word; }
table {
  width: 100%;
  max-width: 100%;
  border-collapse: collapse;
  margin: 7px 0 10px 0;
  font-size: 8.0pt;
  line-height: 1.35;
}
table.fixed-cols {
  table-layout: fixed;
}
th, td {
  border: 1px solid #cbd5e1;
  padding: 4px 5.5px;
  vertical-align: top;
  text-align: left;
  overflow-wrap: anywhere;
  word-break: break-word;
}
td code, td a, th {
  overflow-wrap: anywhere;
  word-break: break-word;
  white-space: normal;
}
th {
  background: #0f172a;
  color: #ffffff;
  font-weight: 600;
}
tr { break-inside: avoid; page-break-inside: avoid; }
tr:nth-child(even) td { background: #f8fafc; }
.masthead {
  background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
  color: #ffffff;
  padding: 12px 14px;
  border-radius: 6px;
  margin-bottom: 10px;
}
.masthead h1 {
  color: #ffffff;
  border: none;
  margin: 0 0 4px 0;
  padding: 0;
  font-size: 14pt;
}
.masthead .sub { color: #bae6fd; font-size: 8.8pt; }
.masthead .meta {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 6px;
  font-size: 8.2pt;
  color: #e2e8f0;
  border-top: 1px solid rgba(255,255,255,0.2);
  padding-top: 5px;
}
.alert-red {
  background: #fef2f2;
  border: 1px solid #fca5a5;
  border-left: 5px solid #dc2626;
  border-radius: 4px;
  padding: 7px 10px;
  margin: 6px 0 9px 0;
  break-inside: avoid;
}
.alert-red .title { color: #991b1b; font-weight: 700; font-size: 9.2pt; margin-bottom: 3px; }
.alert-blue {
  background: #eff6ff;
  border: 1px solid #93c5fd;
  border-left: 4px solid #2563eb;
  border-radius: 4px;
  padding: 7px 10px;
  margin: 7px 0;
  break-inside: avoid;
}
.arabic-box {
  direction: rtl;
  text-align: right;
  font-family: 'Segoe UI', 'Tahoma', 'Arial', sans-serif;
  background: #f0f9ff;
  border: 1px solid #bae6fd;
  border-right: 4px solid #0284c7;
  border-radius: 5px;
  padding: 8px 11px;
  margin: 7px 0 10px 0;
  line-height: 1.55;
  font-size: 9.2pt;
}
.arabic-box .ar-title {
  color: #0369a1;
  font-weight: 700;
  font-size: 9.8pt;
  margin-bottom: 4px;
}
.grid-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-bottom: 7px;
}
.card {
  border: 1px solid #cbd5e1;
  border-radius: 5px;
  padding: 7px 9px;
  background: #ffffff;
  break-inside: avoid;
}
.card-title {
  font-weight: 700;
  color: #0f172a;
  font-size: 9.2pt;
  border-bottom: 1px solid #e2e8f0;
  padding-bottom: 3px;
  margin-bottom: 4px;
}
.badge {
  display: inline-block;
  padding: 1px 5px;
  border-radius: 3px;
  font-size: 7.3pt;
  font-weight: 700;
  text-transform: uppercase;
  white-space: nowrap;
  word-break: normal;
}
.b-red { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
.b-amber { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
.b-blue { background: #dbeafe; color: #1e40af; border: 1px solid #93c5fd; }
.b-green { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
.figure-box {
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  padding: 8px;
  margin: 8px 0 12px 0;
  background: #ffffff;
  break-inside: avoid;
  page-break-inside: avoid;
}
.figure-box img {
  width: 100%;
  height: auto;
  display: block;
  border-radius: 4px;
  border: 1px solid #e2e8f0;
}
.figure-caption {
  font-size: 8pt;
  color: #475569;
  margin-top: 5px;
  line-height: 1.35;
}
.page-break { page-break-after: always; break-after: page; }
blockquote {
  margin: 7px 0;
  padding: 7px 10px;
  background: #f8fafc;
  border-left: 4px solid #475569;
  font-size: 8.8pt;
}
blockquote[dir="rtl"] {
  border-left: 1px solid #cbd5e1;
  border-right: 4px solid #0284c7;
  background: #f0f9ff;
}
.consultant-banner {
  background: #0f172a;
  color: #ffffff;
  padding: 9px 12px;
  border-radius: 5px;
  margin-top: 8px;
  margin-bottom: 8px;
  break-after: avoid;
}
.consultant-banner .num {
  color: #38bdf8;
  font-size: 8pt;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.6px;
}
.consultant-banner h2 {
  background: transparent;
  border: none;
  color: #ffffff;
  margin: 2px 0 0 0;
  padding: 0;
  font-size: 12pt;
}
"""


def find_headless_browser() -> Path:
    """Locate Chrome or Microsoft Edge binary for CSS Paged Media PDF rendering."""
    candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ]
    for path in candidates:
        if path.exists():
            return path
    for cmd in ("chrome", "msedge"):
        found = shutil.which(cmd)
        if found:
            return Path(found)
    raise FileNotFoundError("Neither Chrome nor Microsoft Edge was found for headless PDF generation.")


def has_arabic(text: str) -> bool:
    """Return True if string contains Arabic Unicode characters."""
    return bool(re.search(r"[\u0600-\u06FF]", text))


def is_arabic_line(text: str) -> bool:
    """Return True only when a line/paragraph is predominantly Arabic RTL prose (not an English line with one Arabic name)."""
    stripped_code = re.sub(r"`[^`]+`", "", text)
    ar_chars = len(re.findall(r"[\u0600-\u06FF]", stripped_code))
    if ar_chars == 0:
        return False
    en_chars = len(re.findall(r"[A-Za-z]", stripped_code))
    starts_arabic = bool(re.match(r"^[\s>*\-#\d.()\[\]]*[\u0600-\u06FF]", stripped_code))
    return (ar_chars > en_chars) or starts_arabic


def img_to_data_uri(image_path: Path) -> str:
    """Encode local PNG/JPG image as base64 data URI for self-contained HTML->PDF rendering."""
    suffix = image_path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in ("jpg", "jpeg") else "png"
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{data}"


def format_inline_markdown(text: str, wrap_ltr_in_rtl: bool = False) -> str:
    """Convert inline markdown (**bold**, *italic*, `code`, [label](url)) safely to HTML."""
    code_tokens: list[str] = []

    def _stash_code(m: re.Match[str]) -> str:
        idx = len(code_tokens)
        code_tokens.append(f"<code>{html.escape(m.group(1))}</code>")
        return f"\x01{idx}\x01"

    text = re.sub(r"`([^`]+)`", _stash_code, text)

    link_tokens: list[str] = []

    def _stash_link(m: re.Match[str]) -> str:
        idx = len(link_tokens)
        label = html.escape(m.group(1))
        url = html.escape(m.group(2), quote=True)
        link_tokens.append(f'<a href="{url}">{label}</a>')
        return f"\x02{idx}\x02"

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _stash_link, text)
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", escaped)

    for idx, tok in enumerate(link_tokens):
        escaped = escaped.replace(f"\x02{idx}\x02", tok)
    for idx, tok in enumerate(code_tokens):
        escaped = escaped.replace(f"\x01{idx}\x01", tok)

    if wrap_ltr_in_rtl:
        # Isolate embedded Latin/clinical spans ONLY in text nodes outside HTML tags/<a>/<code>
        segments = re.split(r"(<code\b[^>]*>.*?</code>|<a\b[^>]*>.*?</a>|<[^>]+>)", escaped, flags=re.DOTALL)
        wrapped: list[str] = []
        ltr_pattern = re.compile(
            r"(?<![A-Za-z0-9\x03])"
            r"(?:(?:[<>≤≥]=?|[+\-])?\s*\d+(?:[.,:]\d+)*(?:\s*[–\-/]\s*\d+(?:[.,:]\d+)*)?\s+)?"
            r"[A-Za-z][A-Za-z0-9\-–+/.:()%<>=≤≥³²¹⁺⁻½]*"
            r"(?:\s+(?:[<>≤≥=]?\s*\d[\d.,:/\-–+%]*\s+)?[A-Za-z0-9\-–+/.:()%<>=≤≥³²¹⁺⁻½]*[A-Za-z0-9)%+\-³²¹⁺⁻½])*"
        )
        for seg in segments:
            if not seg or seg.startswith("<"):
                wrapped.append(seg)
            else:
                raw_seg = html.unescape(seg)
                ltr_tokens: list[str] = []

                def _stash_ltr(m: re.Match[str]) -> str:
                    i_tok = len(ltr_tokens)
                    ltr_tokens.append(f'<bdi dir="ltr">{html.escape(m.group(0))}</bdi>')
                    return f"\x03{i_tok}\x03"

                raw_seg = ltr_pattern.sub(_stash_ltr, raw_seg)
                raw_seg = html.escape(raw_seg)
                for i_tok, tok in enumerate(ltr_tokens):
                    raw_seg = raw_seg.replace(f"\x03{i_tok}\x03", tok)
                wrapped.append(raw_seg)
        escaped = "".join(wrapped)

    return escaped


def _build_colgroup(header_cells: list[str]) -> tuple[str, bool]:
    """Return optional <colgroup> and whether table should use fixed layout for wide filenames/paths."""
    n = len(header_cells)
    joined = " | ".join(h.lower() for h in header_cells)
    if n == 6 and "semantic" in joined and "finding" in joined:
        widths = ["4%", "16%", "28%", "17%", "9%", "26%"]
        return "<colgroup>" + "".join(f'<col style="width:{w}">' for w in widths) + "</colgroup>", True
    if n >= 4:
        return "", True
    return "", False


def markdown_to_html(md_text: str) -> str:
    """Convert structured clinical Markdown (headings, tables, lists, blockquotes, RTL prose) into clean HTML."""
    lines = md_text.splitlines()
    out: list[str] = []
    i = 0
    in_ul = False
    in_ol = False
    in_ol_li = False
    in_sub_ul = False
    in_ar_section = False

    def close_sub_ul() -> None:
        nonlocal in_sub_ul, in_ol_li
        if in_sub_ul:
            out.append("</ul>")
            in_sub_ul = False
        if in_ol_li:
            out.append("</li>")
            in_ol_li = False

    def close_lists() -> None:
        nonlocal in_ul, in_ol
        close_sub_ul()
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    def close_ar_box() -> None:
        nonlocal in_ar_section
        close_lists()
        if in_ar_section:
            out.append("</div>")
            in_ar_section = False

    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        indent = len(raw) - len(raw.lstrip())

        if not stripped:
            close_sub_ul()
            # Peek next non-blank line to keep <ol> numbering continuous across blank lines
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if in_ol and j < len(lines) and (re.match(r"^\d+\.\s+", lines[j].strip()) or (len(lines[j]) - len(lines[j].lstrip()) >= 2 and re.match(r"^[-*]\s+", lines[j].strip()))):
                i += 1
                continue
            close_lists()
            i += 1
            continue

        if stripped == "---":
            close_ar_box()
            out.append('<hr style="border:none;border-top:1px solid #cbd5e1;margin:10px 0;">')
            i += 1
            continue

        # Headings
        hm = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if hm:
            close_ar_box()
            level = len(hm.group(1))
            htext = hm.group(2).strip()
            if has_arabic(htext) and not re.search(r"[A-Za-z]{4,}", re.sub(r"\([^)]*\)", "", htext)):
                in_ar_section = True
                out.append('<div class="arabic-box" dir="rtl" lang="ar">')
                out.append(f'<div class="ar-title">{format_inline_markdown(htext, True)}</div>')
            else:
                out.append(f"<h{level}>{format_inline_markdown(htext)}</h{level}>")
            i += 1
            continue

        # Tables
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            close_lists()
            header_cells = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            colgroup_html, use_fixed = _build_colgroup(header_cells)
            tbl_cls = ' class="fixed-cols"' if use_fixed else ""
            out.append(f"<table{tbl_cls}>{colgroup_html}<thead><tr>")
            for hc in header_cells:
                out.append(f"<th>{format_inline_markdown(hc)}</th>")
            out.append("</tr></thead><tbody>")
            for row in rows:
                out.append("<tr>")
                for ci, cell in enumerate(row):
                    cell_ar = is_arabic_line(cell)
                    out.append(f"<td>{format_inline_markdown(cell, cell_ar)}</td>")
                out.append("</tr>")
            out.append("</tbody></table>")
            continue

        # Blockquotes
        if stripped.startswith(">"):
            close_lists()
            bq_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                bq_lines.append(lines[i].strip()[1:].strip())
                i += 1
            bq_joined = " ".join(bq_lines)
            is_ar_bq = is_arabic_line(bq_joined)
            dir_attr = ' dir="rtl" lang="ar"' if is_ar_bq else ""
            out.append(f"<blockquote{dir_attr}>")
            for bq in bq_lines:
                if bq:
                    out.append(f"<p>{format_inline_markdown(bq, is_ar_bq)}</p>")
            out.append("</blockquote>")
            continue

        # Bullet lists (supports nested sub-bullets inside numbered <ol> items)
        bm = re.match(r"^[-*]\s+(.*)$", stripped)
        if bm:
            item_text = bm.group(1)
            item_ar = is_arabic_line(item_text)
            li_dir = ' dir="rtl" lang="ar"' if (item_ar and not in_ar_section) else ""
            if in_ol and indent >= 2:
                if not in_sub_ul:
                    out.append('<ul style="margin-top:3px;margin-bottom:4px;">')
                    in_sub_ul = True
                out.append(f"<li{li_dir}>{format_inline_markdown(item_text, item_ar or in_ar_section)}</li>")
            else:
                close_sub_ul()
                if in_ol:
                    out.append("</ol>")
                    in_ol = False
                if not in_ul:
                    out.append("<ul>")
                    in_ul = True
                out.append(f"<li{li_dir}>{format_inline_markdown(item_text, item_ar or in_ar_section)}</li>")
            i += 1
            continue

        # Numbered lists
        nm = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if nm:
            close_sub_ul()
            if in_ul:
                out.append("</ul>")
                in_ul = False
            num = int(nm.group(1))
            if not in_ol:
                start_attr = f' start="{num}"' if num != 1 else ""
                out.append(f"<ol{start_attr}>")
                in_ol = True
            item_text = nm.group(2)
            item_ar = is_arabic_line(item_text)
            li_dir = ' dir="rtl" lang="ar"' if (item_ar and not in_ar_section) else ""
            out.append(f"<li{li_dir}>{format_inline_markdown(item_text, item_ar or in_ar_section)}")
            in_ol_li = True
            i += 1
            continue

        # Normal paragraph
        close_lists()
        p_ar = is_arabic_line(stripped)
        p_dir = ' dir="rtl" lang="ar" class="arabic-box"' if (p_ar and not in_ar_section) else ""
        out.append(f"<p{p_dir}>{format_inline_markdown(stripped, p_ar or in_ar_section)}</p>")
        i += 1

    close_ar_box()
    return "\n".join(out)


def build_holter_circadian_svg() -> str:
    """Create crisp vector SVG chart of 24-hour PVCs & Avg HR highlighting the 06:00-08:59 AM surge."""
    w, h = 720, 232
    pad_l, pad_r, pad_t, pad_b = 44, 38, 34, 34
    cw = w - pad_l - pad_r
    ch = h - pad_t - pad_b
    max_pvc = 220.0
    n = len(HOLTER_HOURLY_SERIES)
    bar_slot = cw / n
    bar_w = bar_slot * 0.68

    parts = [
        f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;background:#ffffff;border:1px solid #cbd5e1;border-radius:6px;">',
        f'<text x="{w/2}" y="17" text-anchor="middle" font-size="11" font-weight="700" fill="#0f172a" font-family="Segoe UI,Arial">24-Hour Holter Circadian PVC Distribution (Total = 857 PVCs) &amp; Mean Heart Rate</text>',
        # Morning surge shaded band (indices 21, 22, 23 = 06:00, 07:00, 08:00)
        f'<rect x="{pad_l + 21*bar_slot - 2:.1f}" y="{pad_t}" width="{3*bar_slot + 4:.1f}" height="{ch}" fill="#fee2e2" opacity="0.65"/>',
        f'<text x="{pad_l + 22.5*bar_slot:.1f}" y="{pad_t+10}" text-anchor="middle" font-size="8" font-weight="700" fill="#991b1b" font-family="Segoe UI,Arial">06-09 AM Surge</text>',
        f'<text x="{pad_l + 22.5*bar_slot:.1f}" y="{pad_t+20}" text-anchor="middle" font-size="7.5" fill="#991b1b" font-family="Segoe UI,Arial">(311 PVCs / 36.29%)</text>',
        # St depression marker @ 13:00 (idx 4)
        f'<rect x="{pad_l + 4*bar_slot:.1f}" y="{pad_t}" width="{bar_slot:.1f}" height="{ch}" fill="#fef3c7" opacity="0.65"/>',
        f'<text x="{pad_l + 4.5*bar_slot:.1f}" y="{pad_t+10}" text-anchor="middle" font-size="7.5" font-weight="700" fill="#92400e" font-family="Segoe UI,Arial">13:11 ST -2.1mm</text>',
    ]

    # Grid lines & Y-axis ticks (PVCs left, HR right)
    for ytick in (0, 55, 110, 165, 220):
        yy = pad_t + ch - (ytick / max_pvc) * ch
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{w-pad_r}" y2="{yy:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l-5}" y="{yy+3:.1f}" text-anchor="end" font-size="8" fill="#475569" font-family="Segoe UI,Arial">{ytick}</text>')
    parts.append(f'<text x="12" y="{pad_t + ch/2}" text-anchor="middle" font-size="8.5" font-weight="600" fill="#0369a1" transform="rotate(-90 12 {pad_t + ch/2})" font-family="Segoe UI,Arial">PVCs / hr</text>')

    hr_pts: list[str] = []
    for idx, (hr_str, avg_hr, ve, st2, st3) in enumerate(HOLTER_HOURLY_SERIES):
        x0 = pad_l + idx * bar_slot + (bar_slot - bar_w) / 2
        bh = (ve / max_pvc) * ch
        by = pad_t + ch - bh
        color = "#dc2626" if hr_str in ("06:00", "07:00", "08:00") else ("#d97706" if hr_str == "13:00" else "#0284c7")
        parts.append(f'<rect x="{x0:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" rx="2" fill="{color}"/>')
        if ve >= 45:
            parts.append(f'<text x="{x0 + bar_w/2:.1f}" y="{by-3:.1f}" text-anchor="middle" font-size="7.5" font-weight="700" fill="{color}" font-family="Segoe UI,Arial">{ve}</text>')
        if idx % 2 == 0 or idx == 23:
            parts.append(f'<text x="{x0 + bar_w/2:.1f}" y="{h - pad_b + 12}" text-anchor="middle" font-size="7.2" fill="#334155" font-family="Segoe UI,Arial">{hr_str[:2]}h</text>')
        # HR line mapped 40-100 bpm
        hx = pad_l + (idx + 0.5) * bar_slot
        hy = pad_t + ch - ((avg_hr - 40) / 60.0) * ch
        hr_pts.append(f"{hx:.1f},{hy:.1f}")

    parts.append(f'<polyline fill="none" stroke="#0f172a" stroke-width="1.8" points="{" ".join(hr_pts)}"/>')
    for pt in hr_pts:
        px, py = pt.split(",")
        parts.append(f'<circle cx="{px}" cy="{py}" r="2.2" fill="#0f172a"/>')
    parts.append(f'<text x="{w - 6}" y="{pad_t + ch/2}" text-anchor="middle" font-size="8.5" font-weight="600" fill="#0f172a" transform="rotate(90 {w-6} {pad_t + ch/2})" font-family="Segoe UI,Arial">Avg HR (40–100 bpm)</text>')
    parts.append(f'<text x="{w/2}" y="{h-5}" text-anchor="middle" font-size="8" fill="#475569" font-family="Segoe UI,Arial">Blue/Red Bars: Hourly PVCs (Peak 08:00 = 164 PVCs, 3.78% burden, V-run @ 104 bpm) | Black Line: Mean HR (Avg 65 bpm)</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def build_audiology_svg() -> str:
    """Create crisp SVG chart illustrating the Definitive 2026-11-03 Masked PTA & GSI-39 Tympanometry Baseline."""
    return """<svg viewBox="0 0 720 155" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;background:#ffffff;border:1px solid #cbd5e1;border-radius:6px;">
  <text x="360" y="16" text-anchor="middle" font-size="10.2" font-weight="700" fill="#0f172a" font-family="Segoe UI,Arial">Definitive Audiological Baseline (2026-11-03): GSI-39 Tympanometry &amp; Masked Pure Cochlear SNHL</text>
  <!-- Left panel: 2026-11-03 Inter-Aural Tymp Compliance Bars (2.13x Asymmetry) -->
  <line x1="65" y1="30" x2="65" y2="120" stroke="#94a3b8"/>
  <line x1="65" y1="120" x2="365" y2="120" stroke="#94a3b8"/>
  <!-- 3.0 cm3 graph ceiling & 1.6 cm3 ULN line -->
  <line x1="65" y1="45" x2="365" y2="45" stroke="#dc2626" stroke-dasharray="4 3" stroke-width="1"/>
  <line x1="65" y1="80" x2="365" y2="80" stroke="#94a3b8" stroke-dasharray="2 3" stroke-width="0.9"/>
  <text x="61" y="48" text-anchor="end" font-size="7.5" fill="#dc2626" font-family="Segoe UI,Arial">3.0 cm³</text>
  <text x="61" y="83" text-anchor="end" font-size="7.5" fill="#475569" font-family="Segoe UI,Arial">1.6 ULN</text>
  <text x="61" y="122" text-anchor="end" font-size="7.5" fill="#475569" font-family="Segoe UI,Arial">0.0</text>
  <!-- Right Ear 2026-11-03: 1.5 cm3 Normal Type A -->
  <rect x="98" y="82.5" width="76" height="37.5" fill="#2563eb" rx="2"/>
  <text x="136" y="77" text-anchor="middle" font-size="7.6" fill="#1e3a8a" font-weight="700" font-family="Segoe UI,Arial">1.5 cm³ (Type A)</text>
  <text x="136" y="104" text-anchor="middle" font-size="6.8" fill="#ffffff" font-weight="600" font-family="Segoe UI,Arial">Refutes 'Rt As'</text>
  <text x="136" y="133" text-anchor="middle" font-size="8" font-weight="600" fill="#0f172a" font-family="Segoe UI,Arial">Right Peak (cm³)</text>
  <!-- Left Ear 2026-11-03: 3.2 cm3 Extreme Type Ad -->
  <rect x="235" y="40" width="76" height="80" fill="#dc2626" rx="2"/>
  <text x="273" y="35" text-anchor="middle" font-size="7.6" fill="#991b1b" font-weight="700" font-family="Segoe UI,Arial">3.2 cm³ (Type Ad!)</text>
  <text x="273" y="76" text-anchor="middle" font-size="7.2" fill="#ffffff" font-weight="700" font-family="Segoe UI,Arial">2.13x Left/Right</text>
  <text x="273" y="133" text-anchor="middle" font-size="8" font-weight="600" fill="#0f172a" font-family="Segoe UI,Arial">Left Peak (cm³)</text>
  <text x="215" y="147" text-anchor="middle" font-size="7.3" fill="#475569" font-family="Segoe UI,Arial">Lt 3.2 cm³ (&gt;3.0 graph ceiling) + 0 dB ABG = monomeric TM amplifying Lt venous flow (replaces 2024 notes)</text>

  <!-- Right panel: Key Audiometric Metrics Card -->
  <rect x="388" y="27" width="318" height="117" rx="5" fill="#f8fafc" stroke="#cbd5e1"/>
  <text x="400" y="43" font-size="8.4" font-weight="700" fill="#0369a1" font-family="Segoe UI,Arial">2026-11-03 Definitive Masked AC/BC &amp; GSI-39 Audit:</text>
  <text x="400" y="58" font-size="7.8" fill="#1e293b" font-family="Segoe UI,Arial">• Air-Bone Gap: 0 dB Bilaterally (Pure SNHL; intact ossicles)</text>
  <text x="400" y="72" font-size="7.8" fill="#1e293b" font-family="Segoe UI,Arial">• Right Ear: 20–70 dB (3f 40.0) | SRT 25 dB | WRS 84% (@ 65/55)</text>
  <text x="400" y="86" font-size="7.8" fill="#1e293b" font-family="Segoe UI,Arial">• Left Ear: 50–80 dB (3f 54.2) | SRT 50 dB | WRS 76% (@ 90/80)</text>
  <text x="400" y="100" font-size="7.8" fill="#991b1b" font-weight="700" font-family="Segoe UI,Arial">• GSI-39: I 1000 Hz = NT bilat; Rt 1.5 cm³ refutes handwritten 'Rt As'</text>
  <text x="400" y="114" font-size="7.6" fill="#475569" font-family="Segoe UI,Arial">• Prior Hx Metz (45 dB SL) -> OHC ischemia; Holter ST/PVCs threaten cochlea</text>
  <text x="400" y="129" font-size="7.6" fill="#0f766e" font-weight="600" font-family="Segoe UI,Arial">• Hearing Aid Rule: Bilateral Open-Fit WDRC RIC, OSPL90 &lt;= 95–100 dB</text>
</svg>"""


def render_html_to_pdf(
    html_content: str,
    dest_pdf: Path,
    browser_path: Path,
    doc_title: str,
    doc_subtitle: str,
    expected_pages: int | None = None,
) -> int:
    """Compile HTML string to A4 PDF via headless browser and stamp PyMuPDF headers/footers/metadata."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp_path = Path(tmpdir)
        html_file = tmp_path / "render.html"
        raw_pdf = tmp_path / "raw.pdf"
        html_file.write_text(html_content, encoding="utf-8")

        cmd = [
            str(browser_path),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            f"--print-to-pdf={raw_pdf}",
            html_file.as_uri(),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if not raw_pdf.exists():
            # Retry legacy --headless flag if --headless=new behaves differently
            cmd[1] = "--headless"
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if not raw_pdf.exists():
            raise RuntimeError(f"Failed rendering {dest_pdf.name}: {res.stderr}")

        doc = pymupdf.open(raw_pdf)
        try:
            total_pages = len(doc)

            if expected_pages is not None and total_pages != expected_pages:
                raise AssertionError(
                    f"{dest_pdf.name} rendered {total_pages} pages, expected strictly {expected_pages} pages."
                )

            # Stamp vector headers/footers on every page without obscuring content margins
            for page_idx, page in enumerate(doc, start=1):
                rect = page.rect
                pw, ph = rect.width, rect.height
                page.draw_line(pymupdf.Point(30, 22), pymupdf.Point(pw - 30, 22), color=(0.75, 0.80, 0.87), width=0.6)
                page.insert_text(
                    pymupdf.Point(30, 17),
                    PATIENT_HEADER,
                    fontsize=7.0,
                    color=(0.35, 0.42, 0.52),
                    fontname="helv",
                )
                page.insert_textbox(
                    pymupdf.Rect(pw - 260, 8, pw - 30, 21),
                    doc_subtitle,
                    fontsize=7.0,
                    color=(0.01, 0.41, 0.63),
                    fontname="helv",
                    align=pymupdf.TEXT_ALIGN_RIGHT,
                )
                page.draw_line(pymupdf.Point(30, ph - 24), pymupdf.Point(pw - 30, ph - 24), color=(0.75, 0.80, 0.87), width=0.6)
                page.insert_text(
                    pymupdf.Point(30, ph - 14),
                    "Confidential MDT Clinical Dossier — Prepared for Treating Specialists",
                    fontsize=7.0,
                    color=(0.45, 0.50, 0.58),
                    fontname="helv",
                )
                page.insert_textbox(
                    pymupdf.Rect(pw - 160, ph - 22, pw - 30, ph - 8),
                    f"Page {page_idx} of {total_pages}",
                    fontsize=7.4,
                    color=(0.15, 0.23, 0.36),
                    fontname="helv",
                    align=pymupdf.TEXT_ALIGN_RIGHT,
                )

            doc.set_metadata({
                "title": doc_title,
                "author": "MDT Clinical Board & Autonomous Biomedical Data Engineer",
                "subject": PATIENT_HEADER,
                "keywords": "Pulsatile Tinnitus, CSVD, Holter PVC Surge, Legal Blindness, Tympanic Hypercompliance, SNHL",
            })
            doc.save(dest_pdf, garbage=4, deflate=True)
            return total_pages
        finally:
            doc.close()


def build_executive_summary_html() -> str:
    """Construct concise, tightly calibrated 2-page PDF handover brief (Page 1 & Page 2)."""
    extra_css = """
    @page { size: A4 portrait; margin: 10mm 9.5mm 10.5mm 9.5mm; }
    body { font-size: 7.45pt; line-height: 1.24; }
    code { font-size: 7.1pt; padding: 0px 3px; }
    .masthead { padding: 6px 10px; margin-bottom: 5px; }
    .masthead h1 { font-size: 11.4pt; margin-bottom: 2px; }
    .masthead .sub { font-size: 7.4pt; }
    .masthead .meta { margin-top: 3px; padding-top: 3px; font-size: 7.1pt; gap: 7px; }
    .alert-red { padding: 4px 7px; margin: 0 0 4px 0; }
    .alert-red .title { font-size: 7.8pt; margin-bottom: 1.5px; }
    h2 { font-size: 9.0pt; padding: 2.5px 6px; margin: 4.5px 0 3px 0; }
    .grid-2 { gap: 6px; margin-bottom: 4px; }
    .card { padding: 4.5px 6px; }
    .card-title { font-size: 7.8pt; margin-bottom: 2px; padding-bottom: 2px; }
    table { font-size: 7.1pt; margin: 3px 0 4.5px 0; line-height: 1.21; }
    th, td { padding: 2.5px 4px; }
    ul { margin: 1.5px 0 2px 0; padding-left: 13px; }
    li { margin-bottom: 1.5px; }
    .arabic-box { padding: 4.5px 7px; margin: 3px 0; font-size: 7.6pt; line-height: 1.35; }
    .arabic-box .ar-title { font-size: 8.1pt; margin-bottom: 1.5px; }
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>{COMMON_PRINT_CSS}\n{extra_css}</style>
</head>
<body>

<!-- ==================== PAGE 1 ==================== -->
<div class="masthead">
  <h1>MULTIDISCIPLINARY MEDICAL BOARD (MDT) — EXECUTIVE CLINICAL HANDOVER BRIEF</h1>
  <div class="sub">Integrated Audit for Treating Neurology, Neurosurgery, Neuro-Intervention, Cardiology, Endocrinology &amp; ENT Consultants</div>
  <div class="meta">
    <span><strong>Patient:</strong> Amal Mokhtar Mohamed Ibrahim (آمال مختار محمد إبراهيم)</span>
    <span><strong>Age/Sex:</strong> 63.5 yrs / Female</span>
    <span><strong>Verified DOB:</strong> 1963-04-09 (ID: 26304090300188; MRI clerical: 1963-03-27)</span>
    <span><strong>Audit Date:</strong> 2026-11-15 | 69 Verified Files (2011–2026) | 7 Consultants</span>
  </div>
</div>

<div class="alert-red">
  <div class="title">CRITICAL PROCEDURAL &amp; DIAGNOSTIC SAFETY ALERTS (READ BEFORE ORDERING TESTS OR DRUGS)</div>
  <ul>
    <li><strong>1. Bilateral Legal Blindness (&lt; 1/60, MOH 2021) + 2026 Eye Synthesis:</strong> Topcon OD clear (<code>RNFL 40 &micro;m</code>) vs OS silicone <code>Qual 2</code> false <code>"90 &micro;m"</code> trap (true 2019 <code>64 &micro;m</code>, CSO OPs <code>0.74 &micro;V</code>), YET <strong>Flash VEP preserved OU (10.29 / 7.99 &micro;V)</strong> -> Do <strong>NOT</strong> remove OS oil blindly; run dark-only VNG + <strong>fresh 2026 B-Scan + IOP</strong>!</li>
    <li><strong>2. Defer Immediate Catheter DSA:</strong> <code>5 mm</code> Left Carotid Bulb Plaque (Carotid IMT <code>0.13 cm</code>) + unverified eGFR on Metformin + <code>-2.1 mm</code> silent myocardial ischemia. Run bedside compression &amp; non-invasive 4D-MRA / Temporal Bone HRCT+CTA/CTV gatekeeper first.</li>
    <li><strong>3. Circadian Morning PVC Surge &amp; Glycemic Link:</strong> <code>36.29%</code> of all PVCs (<code>311/857</code>; peak <code>164/hr = 3.78%</code> &amp; <code>104 bpm</code> V-run @ <code>08:00 AM</code>) cluster at <code>06:00–08:59 AM</code> on <code>80 IU/day</code> insulin; Dr. Galhom prescribed non-selective <code>Inderal 10 mg BID</code> which masks insulin hypoglycemia!</li>
  </ul>
</div>

<h2>1. Multimodal Diagnostic Audit &amp; Quantitative Evidence Synthesis</h2>
<div class="grid-2">
  <div class="card">
    <div class="card-title">Neuroimaging (MRI 2026-11-05, CT 2024-07-27) &amp; Duplex (2026-11-07)</div>
    <ul>
      <li><strong>MRV (Film06, Verified Markers R/L):</strong> Right-dominant drainage; diffusely attenuated/hypoplastic <strong>Lt Transverse, Sigmoid &amp; IJV</strong> (<code>L/R ratio = 0.48</code>). <strong>Zero thrombosis</strong>; no giant sigmoid diverticulum.</li>
      <li><strong>MRA 1.0-mm Source Slices (Video01, 513 slices + CLAHE ROI1b/c):</strong> Clear aerated mastoids; patent Circle of Willis; <strong>NO high-flow dAVF, cortical venous reflux, or dilated occipital feeder</strong> (slow Cognard I below 1.5T TOF threshold not ruled out).</li>
      <li><strong>Brain Parenchyma (ROI3):</strong> Hypertensive-diabetic <strong>CSVD (Fazekas PV 1–2 / DWM 1–2)</strong> + chronic right basal ganglia, Lt frontal &amp; Rt parietal lacunes. <strong>DWI/ADC negative</strong>; <strong>MS excluded</strong> (0 Dawson fingers; spared CC/brainstem).</li>
      <li><strong>Orbits &amp; CT Continuity (ROI2/5b):</strong> Normal IACs/CPAs; bilateral ethmoid sinusitis; <strong>Lt intraocular silicone oil (2.59x)</strong> confirmed persistent since <code>2024-07-27</code> non-contrast CT + thin optic nerves.</li>
      <li><strong>Carotid &amp; Leg Duplex (69 Audited Files, 2011–2026):</strong> Carotid IMT <code>0.13 cm</code> + <strong>5 mm Lt bulb plaque</strong> (2011 leg Doppler <em>[HISTORICAL]</em> confirms no PAD/DVT).</li>
    </ul>
  </div>
  <div class="card">
    <div class="card-title">24h Holter (2026-11-05), Def. Audiology (2026-11-03), EEG &amp; Exam</div>
    <ul>
      <li><strong>Holter Rate/Rhythm:</strong> <code>95,454</code> beats; HR avg <code>65 bpm</code> (51–99); 0 pauses &gt;2.5s; <strong>SDNN 101 ms</strong>; normal QTc (max <code>326 ms</code>).</li>
      <li><strong>Ectopy &amp; ST Ischemia:</strong> <strong>857 PVCs (0.898%)</strong>, 22 V-pairs, <strong>4 V-runs (up to 104 bpm)</strong>; 153 SVEs. Genuine <strong>-2.1 mm horizontal/downsloping ST depression @ 13:11 CH3</strong> — exacerbates cochlear/CSVD hypoperfusion.</li>
      <li><strong>Definitive Audiology (2026-11-03 Masked AC/BC + GSI-39, ROI5a):</strong> <strong>0 dB ABG</strong> (Pure SNHL: Rt 20–70 dB, WRS <strong>84%</strong>; Lt 50–80 dB, WRS <strong>76%</strong>); Rt <code>1.5 cm³</code> Type A (refutes handwritten <code>'Rt As'</code>), <strong>Lt 3.2 cm³ Type Ad (&gt;3.0 ceiling, I 1000 Hz NT)</strong> -&gt; amplifies venous sound!</li>
      <li><strong>Movement &amp; EEG (2024-07-17 &amp; Dr. Galhom 2026-11-03):</strong> 2h IMC EEG (9 epochs) shows normal 9 Hz alpha + somatic EMG tremor bursts (<strong>0 epileptiform spikes</strong>). Dr. Galhom logged <code>Essential tremor + Titubation</code> (vs prior PD/Sinemet + BG lacune + morning glycemic dips).</li>
    </ul>
  </div>
</div>

<h2>2. Definitive Board Resolution of the 4 Core Clinical Dilemmas</h2>
<table>
  <thead><tr><th style="width:20%">Clinical Dilemma</th><th style="width:42%">Integrated Pathophysiological Mechanism</th><th style="width:38%">MDT Consensus Decision</th></tr></thead>
  <tbody>
    <tr><td><strong>1. Left Pulsatile Tinnitus &amp; DSA Debate</strong></td><td>Turbulent flow across attenuated Lt Transverse-Sigmoid-IJV (<code>L/R 0.48</code>) + mastoid emissary vein, amplified by <strong>3.2 cm³ flaccid Lt eardrum</strong> &amp; nocturnal supine CVP rise. Firm posterior pressure collapses BOTH venous &amp; occipital artery!</td><td><strong>Defer direct DSA.</strong> Step 1: Light anterior IJV vs pinpoint occipital compression + auscultation -&gt; Step 2A Medical/sinus/BP trial (if venous) or Step 2B <strong>4D-MRA / Temporal HRCT+CTA/CTV</strong> (eGFR check + 48h Metformin hold) -&gt; Step 3 Conditional DSA only if positive.</td></tr>
    <tr><td><strong>2. White Matter &amp; Tremor Conflict</strong></td><td><strong>CSVD Fazekas 1–2</strong> (NOT MS; 2017 McDonald DIS negative, echoed by 12-yr <strong>2014 Optovue baseline</strong>). Tremor = <strong>ET + Titubation</strong> (Dr. Galhom) + BG lacune + <strong>80 IU/d AM dips</strong>.</td><td>No lumbar puncture/MS workup. Run formal <strong>MDS-UPDRS off/on Sinemet</strong> (freed from Cetrovit L-Tyrosine LAT1 competition) + <strong>14d CGM</strong>; keep <strong>Brain-Ox/Ginkgo/Escitalopram STOPPED</strong>.</td></tr>
    <tr><td><strong>3. Disabling "4-Hit" Vertigo</strong></td><td>Hit 1: Lt labyrinthine micro-infarct; Hit 2: <strong>Bilateral Legal Blindness (&lt;1/60)</strong>; Hit 3: CSVD + neuropathy (prior <strong>Meclizine</strong> STOPPED); Hit 4: PVCs / -2.1 mm ST / BP &amp; glucose dips.</td><td>Keep <strong>Meclizine DISCONTINUED</strong> (unlock VRT); continue active <strong>Nasonex</strong>; perform <strong>Dark-Only VNG</strong> (positional + calorics only); start <strong>Non-Visual VRT</strong>.</td></tr>
    <tr><td><strong>4. Auditory / Type Ad Paradox</strong></td><td>Non-progressive prior cochlear end-artery infarct spared CN VIII (<code>WRS 76%</code>) + historical Metz (<code>45 dB SL</code>). Left monomeric drum laxity (<code>3.2 cm³</code>) &amp; Rt <code>1.5 cm³</code> refutes hand-noted <code>'Rt As'</code> (0 dB ABG).</td><td>Fit <strong>Bilateral Open-Fit WDRC RIC Hearing Aids</strong> (soft vented domes; avoid occlusion!) with strict <strong>OSPL90 &lt;= 95–100 dB SPL</strong> on Left Ear + binaural navigation safety.</td></tr>
  </tbody>
</table>

<div class="arabic-box" dir="rtl" lang="ar">
  <div class="ar-title">ملخص تنفيذي سريع للسبعة استشاريين (الورقة 1 — التشخيص المتكامل لـ 69 ملفاً 2011–2026)</div>
  <strong>الحقائق الحاسمة (69 ملفاً):</strong> (1) كف البصر <span dir="ltr">&lt; 1/60</span> مع ضمور اليمين <span dir="ltr">40 &micro;m</span> وزيت الشمال (<span dir="ltr">OPs 0.74 &micro;V</span> مقابل <span dir="ltr">VEP 7.99 &micro;V</span> الحي) يمنع تتبع الضوء بالـ <span dir="ltr">VNG</span>. (2) الرنين والـ <span dir="ltr">MRA</span> وتخطيط المخ المطوّل (9 مقاطع: نشاط ألفا <span dir="ltr">9 Hz</span> بدون صرع) نفوا وجود <span dir="ltr">MS</span> أو ناسور عالي التدفق أو بؤر صرعية أو جلطات حادة، وأثبتوا <span dir="ltr">CSVD Fazekas 1–2</span> وجلطة <span dir="ltr">Basal Ganglia</span> وضيق وريدي شمال (<span dir="ltr">L/R 0.48</span>) تضخمه طبلية <span dir="ltr">2026-11-03</span> المرنة (<span dir="ltr">3.2 cm³ Type Ad</span> مقابل <span dir="ltr">1.5 cm³ Type A</span> باليمين؛ <span dir="ltr">0 dB ABG</span>، <span dir="ltr">WRS 84%/76%</span>). (3) الهولتر (<span dir="ltr">-2.1 mm ST</span> و <span dir="ltr">36.29%</span> من الـ <span dir="ltr">857 PVC</span> بين 6–9 ص) يفاقم نقص التروية القوقعية والطنين، والبلاك الـ <span dir="ltr">5 mm</span> بالسباتي الشمال يوجب تأجيل القسطرة <span dir="ltr">DSA</span>.
</div>

<div class="page-break"></div>

<!-- ==================== PAGE 2 ==================== -->
<h2>3. Specialty-by-Specialty Actionable Requests &amp; Bedside Gatekeeper Protocol</h2>
<div class="grid-2">
  <div class="card">
    <div class="card-title"><span class="badge b-red">Priority 1</span> Cardiology &amp; Vascular Risk Team</div>
    <ul>
      <li><strong>Urgent Labs:</strong> Serum <code>K+</code>, <code>Mg2+</code>, <code>Cr/eGFR</code> (rule out HCTZ hypokalemia fueling 857 PVCs/4 V-runs).</li>
      <li><strong>Ischemia Evaluation:</strong> <strong>TTE</strong> + <strong>Pharmacological Stress Echo (Dobutamine) or SPECT</strong> (treadmill invalid due to &lt;1/60 blindness) for <code>-2.1 mm</code> @ 13:11 CH3 ST depression.</li>
      <li><strong>Antihypertensive &amp; Lipid Upgrade:</strong> Stop <code>Angiopress Comp</code> (Captopril/HCTZ OD) &amp; avoid proposed <code>Cidamex</code>! Switch to long-acting <strong>Perindopril 5 mg or Telmisartan 40–80 mg</strong> + <strong>Cardioselective Bisoprolol 2.5–5 mg AM</strong> (reviewing Propranolol); upgrade <code>Zatorva 20</code> -&gt; <strong>Atorvastatin 40–80 mg nocte</strong> (LDL &lt; 55).</li>
    </ul>
  </div>
  <div class="card">
    <div class="card-title"><span class="badge b-amber">Priority 1</span> Endocrinology &amp; Glycemic Audit</div>
    <ul>
      <li><strong>Suspected Occult Morning Hypoglycemia:</strong> Audit <code>Toujeo 30U + Apidra 50U = 80 IU/d</code> + <code>Gliptus Plus</code> via <strong>14-day CGM (FreeStyle Libre) or 7-point SMBG</strong> targeting <code>03:00–09:00 AM</code> to halt adrenergic tremor &amp; 08:00 AM PVC spike (<code>3.78%</code>).</li>
      <li><strong>Renal &amp; SGLT2i Review:</strong> Check <code>HbA1c</code>, <code>Urine ACR</code>, and baseline <code>eGFR</code> (&gt;=45 required for Metformin; <strong>HOLD Metformin 48h</strong> for any contrast CTA/DSA). Consider adding <strong>Empagliflozin 10 mg</strong> for cardiorenal benefit.</li>
    </ul>
  </div>
  <div class="card">
    <div class="card-title"><span class="badge b-blue">Priority 2</span> Neurology, Neurosurgery &amp; Neuro-Intervention</div>
    <ul>
      <li><strong>Bedside Tinnitus Test:</strong> Mastoid auscultation + <strong>Light anterior Lt IJV compression</strong> (never heavy bilateral carotid pressure over 5 mm plaque!) vs <strong>Pinpoint Lt Occipital artery pulse compression</strong>.</li>
      <li><strong>Gated Imaging / DSA Rule:</strong> Only if arterial bruit persists or medical/sinus trial fails -&gt; <strong>4D-MRA</strong> or <strong>Temporal Bone HRCT + CTA/CTV</strong> (eGFR check + 48h Metformin hold) BEFORE conditional DSA.</li>
      <li><strong>Movement Reconciliation:</strong> Perform off/on <strong>Sinemet MDS-UPDRS test</strong> before tapering Levodopa; monitor ET/Titubation response if transitioning Propranolol -&gt; Bisoprolol (keep Propranolol LA only under active CGM).</li>
    </ul>
  </div>
  <div class="card">
    <div class="card-title"><span class="badge b-green">Priority 2</span> ENT, Neurotology &amp; Audiology</div>
    <ul>
      <li><strong>Med Cleanup &amp; Sinus Care:</strong> Keep <strong>Meclizine</strong>, <strong>Cetrovit Ginkgo</strong>, <strong>Brain-Ox</strong> &amp; <strong>Escitalopram</strong> <strong>STOPPED</strong> (sub <strong>pure Citicoline 500 + Vit B1/B6/B12</strong>). <strong>Continue active Nasonex (Mometasone 50 mcg)</strong> + saline (IOP check 4–6w) + 30 deg head elevation.</li>
      <li><strong>Blindness-Adapted VNG &amp; VRT:</strong> Skip saccade/pursuit/OPK; run <strong>dark Dix-Hallpike + bithermal calorics</strong>. Start supervised <strong>non-visual somatosensory VRT</strong> &amp; home fall-proofing.</li>
      <li><strong>Binaural Hearing Aids &amp; Eye Guardrails (C6/C7):</strong> Bilateral <strong>Open-Fit WDRC RIC</strong> (vented domes, Lt <code>OSPL90 &lt;= 95–100 dB</code>). Order <strong>2026 B-Scan OU + IOP</strong>; topical drops ONLY (<strong>NEVER oral Cidamex</strong> atop HCTZ!); keep <code>BP &gt;= 115/70</code> for <code>40 &micro;m</code> nerve &amp; focus low-vision care on <strong>OD</strong>.</li>
    </ul>
  </div>
</div>

<h2>4. Critical Drug-Interaction Hazard Audit &amp; Modified Regimen</h2>
<table>
  <thead><tr><th style="width:20%">Drug / Regimen Audited</th><th style="width:14%">Severity</th><th style="width:33%">Specific Physiological / Pharmacological Hazard</th><th style="width:33%">Modified MDT Action &amp; Monitoring</th></tr></thead>
  <tbody>
    <tr><td><strong>Angiopress Comp</strong> (Captopril 50/HCTZ 25 OD) + <strong>Cidamex</strong></td><td><span class="badge b-red">CONTRAINDICATED</span></td><td>Dual renal K+ wasting -&gt; severe <strong>hypokalemia triggering PVCs/V-runs</strong>; short Captopril t1/2 causes nocturnal BP rebound.</td><td><strong>STOP HCTZ &amp; AVOID Cidamex.</strong> Replace with <strong>Perindopril 5 mg</strong> (or Telmisartan) + <strong>Bisoprolol 2.5–5 mg</strong> after K+/eGFR check.</td></tr>
    <tr><td><strong>Aggrex 75</strong> + <strong>Cetrovit Ginkgo</strong> (10-ingred. 2-cap label) + <strong>Escitalopram</strong></td><td><span class="badge b-red">STOPPED</span></td><td>L-Tyrosine 250mg competes w/ Sinemet @ LAT1 + spikes PVCs/BP; Ginkgo 60mg + Aspirin bleeds; SSRI+HCTZ drops Na+/QT.</td><td><strong>STOP Cetrovit Ginkgo &amp; Escitalopram</strong>; keep <strong>Aggrex 75</strong>; sub <strong>pure Citicoline 500 + B1/B6/B12</strong>.</td></tr>
    <tr><td><strong>Meclizine</strong> &amp; <strong>Brain-Ox</strong> (Vincamine 30) | <strong>Nasonex (Active)</strong></td><td><span class="badge b-red">STOP / ACTIVE</span></td><td>Meclizine blocks VRT &amp; increases falls; Brain-Ox adds QT/PVC risk (-2.1mm/HCTZ) + nocturnal optic steal; Nasonex treats ethmoiditis.</td><td><strong>STOP Meclizine &amp; Brain-Ox</strong> (use Statin + Aggrex + Citicoline 500); <strong>continue active Nasonex</strong> (+ 4–6w IOP check).</td></tr>
    <tr><td><strong>Inderal 10 mg BID</strong> + <strong>Toujeo 30U + Apidra 50U</strong></td><td><span class="badge b-amber">HIGH CAUTION</span></td><td>Propranolol eases ET/tinnitus/PVCs, <strong>but β2 blockade masks insulin hypoglycemia</strong> &amp; impairs hepatic recovery!</td><td><strong>Verify CGM first</strong>; consider switching to cardioselective <strong>Bisoprolol</strong> (or keep Propranolol LA strictly with active CGM alarms).</td></tr>
    <tr><td><strong>Zatorva 20 mg</strong> + <strong>Gliptus Plus</strong> (Metformin) + <strong>Sinemet</strong></td><td><span class="badge b-blue">OPTIMIZE</span></td><td>Statin dose too weak for <code>5 mm</code> bulb plaque + CAD; Metformin risks MALA if contrast given without 48h hold.</td><td><strong>Atorvastatin 40–80 mg nocte</strong>; verify eGFR &amp; hold Metformin 48h for CTA/DSA; continue <strong>Cobal F</strong> if B12 low; test Sinemet before taper.</td></tr>
  </tbody>
</table>

<div class="grid-2">
  <div class="card">
    <div class="card-title">Phase 1 Mandatory Lab &amp; Bedside Checklist (Week 1)</div>
    <ul>
      <li>[ ] <strong>Electrolytes &amp; Renal:</strong> <code>K+</code>, <code>Na+</code>, <code>Mg2+</code>, <code>Ca2+</code>, <code>Creatinine/eGFR</code>, <code>Urine ACR</code></li>
      <li>[ ] <strong>Metabolic &amp; Secondary:</strong> <code>HbA1c</code>, <strong>14-Day CGM</strong> (<code>03–09 AM</code> focus), Fasting Lipids, <code>CBC</code>, <code>TSH</code>, <code>ALT/AST</code>, <code>Vit B12</code></li>
      <li>[ ] <strong>Bedside Differentiator:</strong> Quiet mastoid auscultation + Light anterior IJV pressure vs Pinpoint occipital groove pressure</li>
    </ul>
  </div>
  <div class="arabic-box" dir="rtl" lang="ar" style="margin:0;">
    <div class="ar-title">خطة التنفيذ وتعديل الأدوية العاجل (الورقة 2)</div>
    <strong>الأولوية القصوى:</strong> (1) <strong>تأكيد الإيقاف النهائي لـ</strong> <bdi dir="ltr">Brain-Ox</bdi> و <bdi dir="ltr">Cetrovit Ginkgo</bdi> (<bdi dir="ltr">L-Tyrosine + Ginkgo</bdi>) و <bdi dir="ltr">Meclizine</bdi> و <bdi dir="ltr">Escitalopram</bdi> مع حظر <bdi dir="ltr">Cidamex</bdi> والاستعاضة بـ <bdi dir="ltr">Citicoline 500 mg + B1/B6/B12</bdi>. (2) تحاليل <bdi dir="ltr">K+, Mg2+, Cr/eGFR, CBC, TSH, HbA1c, B12</bdi> وسنسور <bdi dir="ltr">CGM</bdi>. (3) تحويل <bdi dir="ltr">Angiopress</bdi> بعد فحص القلب لـ <bdi dir="ltr">Perindopril + Bisoprolol</bdi> و<bdi dir="ltr">Atorvastatin 40–80 mg</bdi>. (4) <strong>الاستمرار على بخاخ <bdi dir="ltr">Nasonex</bdi> النشط</strong> + <bdi dir="ltr">30°</bdi> + <bdi dir="ltr">VRT</bdi> + سماعتي <bdi dir="ltr">Open-Fit (&lt;= 100 dB)</bdi> وسونار <bdi dir="ltr">2026 + IOP</bdi>.
  </div>
</div>

</body>
</html>"""


def build_consultant_compendium_html() -> str:
    """Compile all 7 individual specialist 3-block audit files + clinical/ophthalmology plates into one HTML document."""
    audit_files = [
        ("Consultant 1 — Diagnostic Head, Neck & Neuroradiology", AUDITS_DIR / "Consultant_1_Diagnostic_Head_Neck_Neuroradiology.md", ["ROI1a_MRV_Left_Transverse_Sigmoid_IJV_CLAHE_Zoom.png", "ROI1b_MRA_3D_TOF_SourceSlices_SkullBase_Mastoid_Occipital_CLAHE.png", "ROI1c_Left_Mastoid_Sigmoid_Occipital_MicroZoom_dAVF_Audit.png"]),
        ("Consultant 2 — Multiple Sclerosis (MS) & White Matter Disorders", AUDITS_DIR / "Consultant_2_MS_and_White_Matter_Disorders.md", ["ROI3_BasalGanglia_WhiteMatter_LacunarInfarcts_CLAHE.png"]),
        ("Consultant 3 — Medical Neurology, Vascular & Movement Disorders", AUDITS_DIR / "Consultant_3_Medical_Neurology_Vascular_and_Movement_Disorders.md", [
            "ROI4_Clinical_Handwriting_and_Holter_KeyStrips_Zoom.png",
            "ROI5b_2024_CT_Brain_and_ProlongedEEG_TremorArtifacts.png",
            VASC_LL_DIR / "VASC_2011-06-21_LowerLimb_ColorDoppler_Report_DrHelmyRashad_LtCalfPerforatorIncompetence.png",
            OPHTH_DIR / "2014_04_23_Historical_Optovue_OCT" / "OPHTH_2014-04-23_Optovue_MaculaMap_EMM5_OU_RPE_Disruption_OD280_OS100_HISTORICAL.png",
            OPHTH_DIR / "2014_04_23_Historical_Optovue_OCT" / "OPHTH_2014-04-23_Optovue_OpticNerve_OS_RNFL78.36um_CDVert0.83_HISTORICAL.png",
        ]),
        ("Consultant 4 — Senior Consultant in Neurosurgery", AUDITS_DIR / "Consultant_4_Neurosurgery.md", []),
        ("Consultant 5 — Neuro-Interventional Radiology (DSA Decision Resolution)", AUDITS_DIR / "Consultant_5_Neuro_Interventional_Radiology.md", []),
        ("Consultant 6 — ENT, Neurotology & Clinical Audiology", AUDITS_DIR / "Consultant_6_ENT_Neurotology_and_Clinical_Audiology.md", ["ROI2_NeuroOtological_IAC_CPA_Orbits_Ethmoid_CLAHE.png", "ROI5a_Audiology_2026-11-03_MaskedPTA_and_GSI39_Tymp_Zoom.png"]),
        ("Consultant 7 — Ophthalmology (Vitreoretinal & Neuro-Ophthalmology)", AUDITS_DIR / "Consultant_7_Ophthalmology_Vitreoretinal_and_NeuroOphthalmology.md", [
            OPHTH_DIR / "2014_04_23_Historical_Optovue_OCT" / "OPHTH_2014-04-23_Optovue_Macula_OD_CystoidEdema_RPEDisruption_HISTORICAL.png",
            OPHTH_DIR / "2014_04_23_Historical_Optovue_OCT" / "OPHTH_2014-04-23_Optovue_MaculaMap_EMM5_OU_RPE_Disruption_OD280_OS100_HISTORICAL.png",
            OPHTH_DIR / "2014_04_23_Historical_Optovue_OCT" / "OPHTH_2014-04-23_Optovue_OpticNerve_OS_RNFL78.36um_CDVert0.83_HISTORICAL.png",
            OPHTH_DIR / "2026_06_Topcon_OCT_and_VisualField" / "OPHTH_2026-06-01_Topcon_OCT_OpticNerve_3DDisc_OD_RNFL40um_SevereAtrophy_Qual9.png",
            OPHTH_DIR / "2026_06_Topcon_OCT_and_VisualField" / "OPHTH_2026-06-01_Topcon_OCT_OpticNerve_3DDisc_OS_Qual2_MediaArtifact.png",
            OPHTH_DIR / "2026_06_Topcon_OCT_and_VisualField" / "OPHTH_2026-06-27_Zeiss_VisualField_HFA_VFI18pct_UnreliableTest_DrAlaaKhaled.png",
            OPHTH_DIR / "2026_07_CSO_VEP_and_ERG" / "OPHTH_2026-07-08_CSO_Flash_VEP_OS_Present_N1P1_7.99uV_P2_132.42ms.png",
            OPHTH_DIR / "2026_07_CSO_VEP_and_ERG" / "OPHTH_2026-07-08_CSO_ERG_OscillatoryPotentials_OD10.0uV_OS0.74uV_SevereOSIschemia.png",
            OPHTH_DIR / "2019_Historical_OCT_Referral_BScan" / "OPHTH_2019-08-20_Optopol_OCT_OS_SevereRNFLThinning_MeanTSNIT64_HISTORICAL.png",
        ]),
    ]

    sections: list[str] = []
    for idx, (label, path, roi_names) in enumerate(audit_files, start=1):
        md_raw = path.read_text(encoding="utf-8")
        # Strip top H1 since we render a stylized banner for each specialist
        lines = md_raw.splitlines()
        if lines and lines[0].startswith("# "):
            md_body = "\n".join(lines[1:])
        else:
            md_body = md_raw
        body_html = markdown_to_html(md_body)

        roi_blocks = []
        for rname in roi_names:
            rpath = rname if isinstance(rname, Path) else (ROI_DIR / rname)
            rlabel = rpath.name
            is_2014 = "2014-04-23" in rlabel
            is_hist = ("HISTORICAL" in rlabel) or rlabel.startswith("VASC_2011") or rlabel.startswith("ROI5b_2024")
            if is_2014:
                box_style = ' style="border:2px solid #F59E0B;background:rgba(245,158,11,0.12);"'
                cap_prefix = '<span class="badge b-amber">⚠️ HISTORICAL RECORD — 2014 BASELINE (PRE-RETINAL DETACHMENT / 12 YEARS OLD)</span> '
            elif is_hist:
                box_style = ' style="border:2px solid #f59e0b;background:#fffbeb;"'
                cap_prefix = '<span class="badge b-amber">HISTORICAL / OLD EXAM (2011–2024)</span> '
            else:
                box_style = ""
                cap_prefix = '<span class="badge b-blue">CURRENT 2026 EVIDENCE PLATE</span> '
            if rpath.exists():
                b64 = img_to_data_uri(rpath)
                roi_blocks.append(
                    f'<div class="figure-box"{box_style}><img src="{b64}" alt="{html.escape(rlabel)}">'
                    f'<div class="figure-caption">{cap_prefix}<code>{html.escape(rlabel)}</code></div></div>'
                )

        sections.append(
            f'<div class="page-break"></div>'
            f'<div class="consultant-banner"><div class="num">Independent Specialist Audit {idx} of 7</div><h2>{html.escape(label)}</h2></div>'
            f'{body_html}\n{"".join(roi_blocks)}'
        )

    cover_html = f"""
<div class="masthead" style="padding:9px 12px;margin-bottom:6px;">
  <h1 style="font-size:13.2pt;">7-CONSULTANT MULTIDISCIPLINARY BOARD — COMPLETE 3-BLOCK AUDIT COMPENDIUM</h1>
  <div class="sub">Blinded Primary-Source Audit of All 69 Files (2011–2026): MRI/MRA/MRV (513 Slices), 2026 Topcon OCT, Zeiss HFA, CSO VEP/ERG, 2014–2021 Eye Archive, 2011 Leg Doppler, Holter &amp; 2026-11-03 Audiology</div>
  <div class="meta">
    <span><strong>Patient:</strong> Amal Mokhtar Mohamed Ibrahim (آمال مختار محمد إبراهيم)</span>
    <span><strong>DOB:</strong> 1963-04-09 (ID: 26304090300188)</span>
    <span><strong>Audit Date:</strong> 2026-11-15 | 69 Verified Files (2011–2026) | 7 Consultants</span>
  </div>
</div>

<div class="alert-blue" style="padding:5px 8px;margin:4px 0 6px 0;font-size:8.6pt;">
  <strong>Methodological Principle:</strong> Every consultant independently audited all 69 semantically organized files in <code>01_Structured_and_Renamed_Inputs/</code> (strictly distinguishing <em>HISTORICAL 2011–2024</em> exams from <em>CURRENT 2026</em> tests) using standardized <strong>1. Conclusion | 2. Immediate Action | 3. Detailed Action Plan</strong> deliverables.
</div>

<h2 style="margin-top:6px;margin-bottom:4px;padding:4px 8px;font-size:11pt;">Inter-Consultant Consensus &amp; Conflict Resolution Matrix</h2>
<table class="fixed-cols" style="font-size:7.6pt;line-height:1.28;margin:4px 0 6px 0;">
  <thead>
    <tr><th style="width:21%">Specialist Consultant</th><th style="width:29%">Primary Independent Verdict</th><th style="width:26%">Discrepancy / Blind Spot Exposed</th><th style="width:24%">Key Decisive Recommendation</th></tr>
  </thead>
  <tbody>
    <tr><td><strong>1. Diagnostic Neuroradiology</strong></td><td>CSVD (Fazekas 1–2) + chronic BG/subcortical lacunes; Rt-dominant MRV; attenuated Lt TS/Sigmoid/IJV (0.48); clear mastoids; intact IACs; no high-flow dAVF.</td><td>Official Rology PDF omitted <strong>Left Intraocular Silicone Oil (2.59x)</strong> persisting since <strong>2024-07-27 CT</strong> &amp; optic atrophy; MRVFilm06 markers verified (PA R-on-right).</td><td>If arterial/bony lesion suspected -&gt; <strong>Temporal Bone HRCT + CTA/CTV</strong> or <strong>4D-MRA</strong>.</td></tr>
    <tr><td><strong>2. MS &amp; White Matter</strong></td><td><strong>MS &amp; CNS Vasculitis EXCLUDED.</strong> Hypertensive-Diabetic CSVD Fazekas 1–2 (0 Dawson fingers, spared corpus callosum/U-fibers/brainstem).</td><td>Prior concern over white matter foci + optic atrophy explained by retinal/vascular retinopathy, NOT demyelination; Holter ST/PVCs accelerate CSVD.</td><td>No LP/oligoclonal bands; aggressive vascular &amp; cardiac rhythm risk factor control is sole white matter DMD.</td></tr>
    <tr><td><strong>3. Medical Neurology &amp; Movement</strong></td><td>Mixed Tremor: <strong>ET + Titubation</strong> (Dr. Galhom) + BG lacune + <strong>06–09 AM insulin dips</strong> (<code>36.29%</code> PVC); <strong>2014 Optovue baseline</strong> links &gt;12y microangiopathy to CSVD.</td><td>2024 2h EEG = 9 Hz alpha (0 spikes); <strong>Brain-Ox</strong> (QT/steal), <strong>Cetrovit</strong> (L-Tyrosine LAT1/Ginkgo 60) &amp; <strong>Escitalopram</strong> (Na+/QT) STOPPED.</td><td>Off/on Sinemet <strong>MDS-UPDRS</strong> + 14d CGM; sub <strong>Citicoline 500 + B1/B6/B12</strong>; Atorvastatin 40–80 mg; active Nasonex.</td></tr>
    <tr><td><strong>4. Neurosurgery</strong></td><td>Zero tumor, aneurysm, Chiari, hydrocephalus, or high-grade Cognard IIb–IV dAVF with cortical venous reflux across 2024 CT and 2026 MRI.</td><td>Validates referring surgeon's dAVF vigilance while proving immediate surgical/hemorrhage urgency is absent; 2024 EEG confirms non-epileptic movement.</td><td>Defer invasive DSA to avoid crossing <code>5 mm</code> bulb plaque; preserve dominant Rt Transverse-Sigmoid-IJV.</td></tr>
    <tr><td><strong>5. Neuro-Interventional Radiology</strong></td><td>Resolves DSA controversy: 1.5T TOF can miss Cognard I, but blind DSA carries high thromboembolic/CA-AKI/cardiac risk.</td><td><code>5 mm</code> Lt Carotid Bulb plaque sits directly in left ECA catheter path + pending eGFR on Metformin + <code>-2.1 mm</code> ST ischemia &amp; <code>3.78%</code> AM PVC burden.</td><td>Enforce <strong>3-Step Gatekeeper</strong> (Bedside selective IJV vs Occipital -&gt; 4D-MRA/CTA after eGFR -&gt; Conditional DSA).</td></tr>
    <tr><td><strong>6. ENT, Neurotology &amp; Audiology</strong></td><td>Definitive <strong>2026-11-03 Baseline</strong>: Stable pure SNHL (0 dB ABG, Rt WRS 84%, Lt WRS 76%) + <strong>Extreme Lt Type Ad (3.2 cm³)</strong> + <strong>4-Hit Vertigo</strong>.</td><td><strong>Rt 1.5 cm³ Type A refutes handwritten 'Rt As'</strong>; Lt 3.2 cm³ amplifies venous tinnitus/PVCs; <strong>&lt;1/60 blindness invalidates optical VNG</strong>.</td><td>Keep <strong>Meclizine/Ginkgo/Brain-Ox STOPPED</strong>; continue active <strong>Nasonex</strong>; dark VNG + non-visual VRT; <strong>Open-Fit WDRC RIC (&lt;= 100 dB)</strong>.</td></tr>
    <tr><td><strong>7. Ophthalmology (Vitreoretinal &amp; Neuro-Ophth)</strong></td><td>12-yr progression [<strong>2014 BASELINE</strong> RNFL 78.36 &amp; OD CME 280 -&gt; 2019 64 &micro;m -&gt; oil OU]: <strong>OD Qual 10 (40 &micro;m)</strong>; <strong>OS OPs 0.74 &micro;V</strong>, yet <strong>VEP OU (10.29 / 7.99 &micro;V)</strong>!</td><td>Co-audited w/ Neurology C3 (no standalone 2014 fundus camera JPG uploaded); functional OS recovery unlikely &amp; ~7y oil removal risky; yet VEP + chiasm intact!</td><td>Urgent <strong>2026 B-Scan OU + IOP</strong>; topical drops ONLY (NO oral Cidamex!); check IOP 4–6w on active <strong>Nasonex</strong>; <code>BP &gt;= 115/70</code>.</td></tr>
  </tbody>
</table>

<div style="max-width:595px;margin:0 auto;">{build_audiology_svg()}</div>

<div class="arabic-box" dir="rtl" lang="ar" style="margin:5px 0 0 0;padding:6px 10px;font-size:8.7pt;line-height:1.45;">
  <div class="ar-title" style="margin-bottom:2px;font-size:9.3pt;">خلاصة اجتماع اللجنة الطبية السباعية (<bdi dir="ltr">7-Consultant MDT Board</bdi>) — إجمالي 69 ملفاً موثقاً</div>
  تم تجميع الآراء العلمية المستقلة للاستشاريين السبعة بالكامل في الصفحات التالية مع لوحات الزوم الميكروسكوبية (<bdi dir="ltr">CLAHE &amp; ROI Panels</bdi>) وفحوصات <bdi dir="ltr">2014</bdi> التاريخية الموثقة بين الأعصاب والعيون لتقديم مرجع تخصصي متكامل ومحدّث دوائياً حتى <bdi dir="ltr">15-11-2026</bdi>.
</div>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>{COMMON_PRINT_CSS}</style>
</head>
<body>
{cover_html}
{"".join(sections)}
</body>
</html>"""


def build_full_case_study_html() -> str:
    """Build complete Fellowship Monograph combining Master_MDT_Case_Study_Report.md + charts + visual plates."""
    md_text = (DELIVERABLES_DIR / "Master_MDT_Case_Study_Report.md").read_text(encoding="utf-8")
    # Strip top markdown heading + metadata block so we render stylized cover masthead first
    lines = md_text.splitlines()
    start_idx = 0
    for idx, ln in enumerate(lines):
        if ln.startswith("## SECTION 1"):
            start_idx = idx
            break
    core_html = markdown_to_html("\n".join(lines[start_idx:]))

    # Embed visual appendix plates
    plate_specs = [
        ("Plate A — MRV Left Transverse/Sigmoid/IJV Asymmetry & Marker Verification", ROI_DIR / "ROI1a_MRV_Left_Transverse_Sigmoid_IJV_CLAHE_Zoom.png"),
        ("Plate B — MRA 1.0-mm Source Slice Skull-Base & Mastoid Occipital dAVF Audit", ROI_DIR / "ROI1b_MRA_3D_TOF_SourceSlices_SkullBase_Mastoid_Occipital_CLAHE.png"),
        ("Plate C — Left Mastoid / Sigmoid Plate Micro-Zoom CLAHE Panel", ROI_DIR / "ROI1c_Left_Mastoid_Sigmoid_Occipital_MicroZoom_dAVF_Audit.png"),
        ("Plate D — Neuro-Otological Axis (IAC/CPA, Left Intraocular Silicone Oil 2.59x, Ethmoiditis)", ROI_DIR / "ROI2_NeuroOtological_IAC_CPA_Orbits_Ethmoid_CLAHE.png"),
        ("Plate E — CSVD (Fazekas 1–2) & Right Basal Ganglia Chronic Lacune", ROI_DIR / "ROI3_BasalGanglia_WhiteMatter_LacunarInfarcts_CLAHE.png"),
        ("Plate F — Prof. Dr. Ayman Galhom Clinical Notes/Rx & Holter -2.1 mm ST / Morning Surge", ROI_DIR / "ROI4_Clinical_Handwriting_and_Holter_KeyStrips_Zoom.png"),
        ("Plate G — 48-Slice Z-Stack Reconstruction from 513 Deduplicated 3D TOF Source Frames", MRA_DIR / "MRA_2026-11-05_Video01_48Slice_ZStack_ContactSheet.png"),
        ("Plate H — Definitive 2026-11-03 Masked Pure-Tone Audiometry & GSI-39 Tympanometry", ROI_DIR / "ROI5a_Audiology_2026-11-03_MaskedPTA_and_GSI39_Tymp_Zoom.png"),
        ("Plate I [HISTORICAL — 2024] — Non-Contrast Brain CT & 2024-07-17 Prolonged 2h EEG (Tremor EMG Artifacts)", ROI_DIR / "ROI5b_2024_CT_Brain_and_ProlongedEEG_TremorArtifacts.png"),
        ("Plate J [CURRENT 2026-06-01] — Topcon 3D Disc OD Severe Optic Atrophy (RNFL 40 um, Qual 9)", OPHTH_DIR / "2026_06_Topcon_OCT_and_VisualField" / "OPHTH_2026-06-01_Topcon_OCT_OpticNerve_3DDisc_OD_RNFL40um_SevereAtrophy_Qual9.png"),
        ("Plate K [CURRENT 2026-06-01 & 06-27] — Topcon OS Qual 2 Artifact ('90 um' Noise Trap)", OPHTH_DIR / "2026_06_Topcon_OCT_and_VisualField" / "OPHTH_2026-06-01_Topcon_OCT_OpticNerve_3DDisc_OS_Qual2_MediaArtifact.png"),
        ("Plate L [CURRENT 2026-07-08] — CSO ERG Oscillatory Potentials OS Collapse (0.74 vs 10.0 uV OD)", OPHTH_DIR / "2026_07_CSO_VEP_and_ERG" / "OPHTH_2026-07-08_CSO_ERG_OscillatoryPotentials_OD10.0uV_OS0.74uV_SevereOSIschemia.png"),
        ("Plate M [CURRENT 2026-07-08] — CSO Flash VEP OS Preserved Conduction (7.99 uV, P2 132.4 ms)", OPHTH_DIR / "2026_07_CSO_VEP_and_ERG" / "OPHTH_2026-07-08_CSO_Flash_VEP_OS_Present_N1P1_7.99uV_P2_132.42ms.png"),
        ("Plate N0 [⚠️ HISTORICAL RECORD — 2014 BASELINE (PRE-RETINAL DETACHMENT / 12 YEARS OLD)] — Optovue OD CME / RPE Disruption (Dual Neuro & Ophth)", OPHTH_DIR / "2014_04_23_Historical_Optovue_OCT" / "OPHTH_2014-04-23_Optovue_Macula_OD_CystoidEdema_RPEDisruption_HISTORICAL.png"),
        ("Plate N1 [⚠️ HISTORICAL RECORD — 2014 BASELINE (PRE-RETINAL DETACHMENT / 12 YEARS OLD)] — Optovue EMM5 Macula Map OD 280 / OS 100", OPHTH_DIR / "2014_04_23_Historical_Optovue_OCT" / "OPHTH_2014-04-23_Optovue_MaculaMap_EMM5_OU_RPE_Disruption_OD280_OS100_HISTORICAL.png"),
        ("Plate N2 [⚠️ HISTORICAL RECORD — 2014 BASELINE (PRE-RETINAL DETACHMENT / 12 YEARS OLD)] — Optovue OS RNFL 78.36 um / Vert C/D 0.83", OPHTH_DIR / "2014_04_23_Historical_Optovue_OCT" / "OPHTH_2014-04-23_Optovue_OpticNerve_OS_RNFL78.36um_CDVert0.83_HISTORICAL.png"),
        ("Plate N3 [HISTORICAL — 2019-08-20] — Optopol OS Severe RNFL Thinning (64 um Red, Pre-Op)", OPHTH_DIR / "2019_Historical_OCT_Referral_BScan" / "OPHTH_2019-08-20_Optopol_OCT_OS_SevereRNFLThinning_MeanTSNIT64_HISTORICAL.png"),
        ("Plate O [HISTORICAL — 2011-06-21] — Lower Limb Color Doppler Report (PAD/DVT Excluded)", VASC_LL_DIR / "VASC_2011-06-21_LowerLimb_ColorDoppler_Report_DrHelmyRashad_LtCalfPerforatorIncompetence.png"),
    ]
    plates_html: list[str] = []
    for caption, ppath in plate_specs:
        if ppath.exists():
            b64 = img_to_data_uri(ppath)
            is_2014 = "2014 BASELINE" in caption
            is_hist = is_2014 or ("[HISTORICAL" in caption)
            if is_2014:
                box_style = ' style="border:2px solid #F59E0B;background:rgba(245,158,11,0.12);"'
            elif is_hist:
                box_style = ' style="border:2px solid #f59e0b;background:#fffbeb;"'
            else:
                box_style = ""
            plates_html.append(
                f'<div class="figure-box"{box_style}><img src="{b64}" alt="{html.escape(caption)}">'
                f'<div class="figure-caption"><strong>{html.escape(caption)}</strong> — Source: <code>{html.escape(ppath.name)}</code></div></div>'
            )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>{COMMON_PRINT_CSS}</style>
</head>
<body>

<div class="masthead">
  <h1>FELLOWSHIP-LEVEL MULTIDISCIPLINARY CASE STUDY &amp; BIOMEDICAL AUDIT MONOGRAPH</h1>
  <div class="sub">7-Consultant Multi-System Neuro-Vascular, Movement Disorder, Cardiac, Neuro-Otological &amp; Vitreoretinal Synthesis Across 69 Verified Files (2011–2026)</div>
  <div class="meta">
    <span><strong>Patient:</strong> Amal Mokhtar Mohamed Ibrahim (آمال مختار محمد إبراهيم)</span>
    <span><strong>Age/Sex:</strong> 63.5 yrs / Female</span>
    <span><strong>Verified DOB:</strong> 1963-04-09 (ID: 26304090300188; MRI clerical: 27-Mar-1963)</span>
    <span><strong>Date:</strong> November 15, 2026</span>
  </div>
</div>

<div class="figure-box">
  {build_holter_circadian_svg()}
  <div class="figure-caption"><strong>Quantitative Rhythm Discovery:</strong> Hourly digitization across all 24 rows of Page 2 shows <strong>36.29% of all 857 PVCs (311 beats) surge between 06:00 and 08:59 AM</strong> (peaking at 164 PVCs / 3.78% burden with a 104 bpm V-run at 08:00 AM on 80 IU/day insulin), alongside a genuine <strong>-2.1 mm CH3 ST depression at 13:11</strong>.</div>
</div>

<div class="figure-box">
  {build_audiology_svg()}
</div>

{core_html}

<div class="page-break"></div>
<h2>SECTION 9: ENHANCED NEUROIMAGING, OPHTHALMOLOGY, DOPPLER &amp; HOLTER VISUAL APPENDIX (2011–2026)</h2>
{"".join(plates_html)}

</body>
</html>"""


def main() -> None:
    browser = find_headless_browser()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using headless Chromium renderer: {browser}")

    # 1. Concise 2-Page Executive Handover PDF (assert strictly 2 pages)
    pdf1 = OUTPUT_DIR / "01_MDT_Executive_Summary_and_Action_Plan.pdf"
    p1_pages = render_html_to_pdf(
        build_executive_summary_html(),
        pdf1,
        browser,
        "MDT Executive Summary & Action Plan — Amal Mokhtar Mohamed Ibrahim",
        "2-Page Specialist Handover Brief",
        expected_pages=2,
    )
    print(f"[OK] {pdf1.name}: {p1_pages} pages ({pdf1.stat().st_size:,} bytes)")

    # 2. 7-Consultant Compendium PDF
    pdf2 = OUTPUT_DIR / "02_Consultant_7_Board_Audit_Reports.pdf"
    p2_pages = render_html_to_pdf(
        build_consultant_compendium_html(),
        pdf2,
        browser,
        "7-Consultant Medical Board Audit Compendium — Amal Mokhtar Mohamed Ibrahim",
        "Independent 7-Specialist 3-Block Audit Reports",
        expected_pages=None,
    )
    print(f"[OK] {pdf2.name}: {p2_pages} pages ({pdf2.stat().st_size:,} bytes)")

    # 3. Complete Fellowship Case Study Monograph (EN + Egyptian AR + Plates)
    pdf3 = OUTPUT_DIR / "03_Patient_Case_Study_Full_Report.pdf"
    p3_pages = render_html_to_pdf(
        build_full_case_study_html(),
        pdf3,
        browser,
        "Full Fellowship MDT Case Study (EN & Egyptian AR) — Amal Mokhtar Mohamed Ibrahim",
        "Complete Case Study Monograph & Evidence Plates",
        expected_pages=None,
    )
    print(f"[OK] {pdf3.name}: {p3_pages} pages ({pdf3.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
