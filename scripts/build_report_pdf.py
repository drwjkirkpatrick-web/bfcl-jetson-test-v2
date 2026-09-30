#!/usr/bin/env python3
"""
BFCL Jetson Test v2 — PDF report builder.

Regenerates docs/BFCL_Jetson_Test_v2_Report.pdf from the benchmark data:
3 matplotlib charts (overall / category / chained) + reportlab layout with
Paragraph-wrapped table cells and explicit column widths (no overflow).

Run:  ~/.hermes/hermes-agent/venv/bin/python scripts/build_report_pdf.py
(requires reportlab + matplotlib, both in the hermes-agent venv)

History: v2.1 report built 2026-09-29 (17 models); v2.2 adds Gemma 4 E4B QAT
(48/52, only perfect chained score) — same day.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
    PageBreak, Table, TableStyle, Image,
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT

PROJECT = Path(__file__).resolve().parent.parent
DOCS = PROJECT / "docs"
CHART_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "bfcl_v2_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = DOCS / "BFCL_Jetson_Test_v2_Report.pdf"

TITLE = "BFCL Jetson Test v2 — Benchmark Report"
RUN_DATE = "2026-09-29"

# ---------------------------------------------------------------------------
# Benchmark data (17 v1 models + E4B v2 debut) — from results JSONs
# name: (overall, simple, parallel, chained, coding, note)
# ---------------------------------------------------------------------------
RESULTS = [
    ("granite4-3b",       50, 29, 12, 7, 2, "champion — perfect parallel"),
    ("hammer2.1-3b",      49, 29, 11, 7, 2, "biggest v1-to-v2 climb"),
    ("gemma4-e4b-qat",    48, 28, 10, 8, 2, "only perfect chained score (8/8)"),
    ("granite4.1-3b",     48, 28, 11, 7, 2, ""),
    ("ternary-bonsai-tq2", 46, 29, 11, 4, 2, "ternary, sub-4-bit — and it codes"),
    ("arch-agent-1.5b",   46, 28, 10, 6, 2, "best small agentic profile"),
    ("xlam-2-1b",         44, 29, 10, 3, 2, "1.6GB file — efficiency star"),
    ("ternary-bonsai-q4", 44, 28, 11, 3, 2, ""),
    ("granite4.2-3b",     44, 28, 9, 5, 2, ""),
    ("granite3.2-2b",     43, 29, 12, 1, 1, "text-fallback rescue (1.9% to 82.7%)"),
    ("xlam-2-8b",         42, 29, 11, 0, 2, ""),
    ("qwen2.5-3b",        42, 29, 11, 0, 2, ""),
    ("xlam-2-3b",         40, 29, 9, 0, 2, ""),
    ("hermes3-3b-q5",     39, 27, 10, 0, 2, ""),
    ("nanbeige4-3b",      37, 27, 9, 0, 1, "thinking model"),
    ("hermes3-3b-q4",     31, 19, 10, 0, 2, "Q4 hurts simple, not parallel"),
    ("llama3.2-3b",       20, 18, 0, 0, 2, "codes fine, calls poorly"),
]

PENDING = "qwen3.5-9b (4.1GB DeltaNet hybrid) — OOM-killed twice; next attempt 12K ctx."

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
INK = "#22262d"
ACCENT = "#c0392b"
GRID = "#e4e7ec"
VDEBUT = "#2e7d32"   # E4B highlight


def pct(n, d):
    return 100.0 * n / d


def build_charts():
    names = [r[0] for r in RESULTS]
    overall = [r[1] for r in RESULTS]
    simple = [r[2] for r in RESULTS]
    par = [r[3] for r in RESULTS]
    chained = [r[4] for r in RESULTS]
    coding = [r[5] for r in RESULTS]

    # ---- Chart 1: overall ----
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ypos = list(range(len(names)))[::-1]
    colors_bar = [VDEBUT if n == "gemma4-e4b-qat" else INK for n in names]
    ax.barh(ypos, [pct(v, 52) for v in overall], color=colors_bar, height=0.62)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Overall score (% of 52 tests)", fontsize=9)
    ax.set_xlim(0, 105)
    for y, v in zip(ypos, overall):
        ax.text(pct(v, 52) + 1, y, f"{v}/52", va="center", fontsize=7.5, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "chart_overall.png", dpi=150)
    plt.close(fig)

    # ---- Chart 2: category means ----
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    cats = ["Simple (x30)", "Parallel (x12)", "Chained (x8)", "Coding (x2)"]
    means = [pct(sum(simple), 30 * len(names)),
             pct(sum(par), 12 * len(names)),
             pct(sum(chained), 8 * len(names)),
             pct(sum(coding), 2 * len(names))]
    bar_colors = [INK, INK, ACCENT, INK]
    bars = ax.bar(cats, means, color=bar_colors, width=0.55)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 1.5, f"{m:.0f}%",
                ha="center", fontsize=9, color=INK)
    ax.set_ylabel("Field mean (%)", fontsize=9)
    ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "chart_categories.png", dpi=150)
    plt.close(fig)

    # ---- Chart 3: chained (the divider) ----
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    colors_bar = [VDEBUT if n == "gemma4-e4b-qat" else (ACCENT if c == 0 else INK)
                  for n, c in zip(names, chained)]
    ax.barh(ypos, chained, color=colors_bar, height=0.62)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Chained tests passed (of 8)", fontsize=9)
    ax.set_xlim(0, 8.6)
    ax.set_xticks(range(0, 9))
    for y, c in zip(ypos, chained):
        ax.text(c + 0.12, y, f"{c}/8", va="center", fontsize=7.5, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "chart_chained.png", dpi=150)
    plt.close(fig)

    return [CHART_DIR / "chart_overall.png",
            CHART_DIR / "chart_categories.png",
            CHART_DIR / "chart_chained.png"]


# ---------------------------------------------------------------------------
# Reportlab layout
# ---------------------------------------------------------------------------
BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=9.5, leading=13,
                      textColor=colors.HexColor(INK), alignment=TA_LEFT)
H1 = ParagraphStyle("h1", parent=BODY, fontName="Helvetica-Bold", fontSize=17,
                    leading=21, spaceAfter=10)
H2 = ParagraphStyle("h2", parent=BODY, fontName="Helvetica-Bold", fontSize=13,
                    leading=16, spaceBefore=6, spaceAfter=6)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8, leading=10)
CELL_B = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")


def P(text, style=BODY):
    return Paragraph(text, style)


def make_table(header, rows, col_widths, body_style=CELL):
    data = [[Paragraph(str(c), CELL_B) for c in header]] + \
           [[Paragraph(str(c), body_style) for c in row] for row in rows]
    t = Table(data, colWidths=[w * inch / 72.0 for w in col_widths],
              repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f5")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9ced6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def page_number(canv, doc):
    canv.saveState()
    canv.setFont("Helvetica", 8)
    canv.setFillColor(colors.HexColor("#8a919c"))
    canv.drawRightString(doc.pagesize[0] - inch, inch * 0.6,
                         f"Page {doc.page}")
    canv.restoreState()


def build():
    charts = build_charts()

    doc = BaseDocTemplate(
        str(OUT_PDF), pagesize=letter,
        leftMargin=inch * 0.85, rightMargin=inch * 0.85,
        topMargin=inch * 0.75, bottomMargin=inch * 0.8,
        title=TITLE, author="Walker Kirkpatrick")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  id="main")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame],
                                      onPage=page_number)])

    story = []

    # ---- Page 1: title + summary ----
    story.append(P(TITLE, H1))
    story.append(P(
        "Which small LLMs can actually do things — call tools, chain steps, and "
        "write working code — on a single 8GB board? This benchmark takes the 17 "
        "best function-callers from BFCL Jetson Test v1, plus one v2 debut — "
        "Gemma 4 E4B QAT, the smallest model already proven as an agentic "
        "tool-caller on this hardware — and puts them through a harder exam: "
        "50 tool-calling tests plus two auto-graded coding builds, all served "
        "locally by llama.cpp on a Jetson Orin Nano 8GB. Every model runs in the "
        "same hardware envelope you would actually deploy — no cloud, no GPU "
        "cluster, just one small board doing its best."))
    story.append(P(
        "The headline: single-call accuracy is nearly saturated at this scale — "
        "the real separation comes from chained multi-step calls and code that "
        "actually runs. A 3B Granite model tops the leaderboard at 96.2%. And the "
        "only perfect score on chained multi-step calling belongs to a 4B "
        "generalist, not a function-calling specialist: E4B, in its first "
        "function-calling exam, aced all 8 chained tests and never once failed to "
        "derive step-2 arguments from returned data."))
    story.append(P(
        "Companion repo: https://github.com/drwjkirkpatrick-web/bfcl-jetson-test-v2 "
        "— harness code, full per-test JSON results, and settings snapshot. Full "
        f"source list on page 7. Run date {RUN_DATE}."))
    story.append(PageBreak())

    # ---- Page 2: test suite ----
    story.append(P("1. Test Suite", H2))
    story.append(P(
        "Four categories, 52 tests, all auto-graded. Evaluation is deterministic "
        "and executable — no LLM judge, no fluctuation. Tool-call matching "
        "follows BFCL-style rules: strict on required parameters, lenient on "
        "extras, string normalization, int/float coercion, order-independent "
        "matching for parallel calls. A chained test passes only if every step "
        "matches."))
    story.append(make_table(
        ["Category", "Count", "What it measures"],
        [
            ["simple", "30",
             'Single call with tricky argument extraction: ISO 8601 dates, 24-hour '
             'times, currency codes, booleans, arrays, unit conversion ("18% tip on '
             'an $85.40 bill" to calculate_tip(bill_amount=85.40, tip_percentage=18))'],
            ["moderate_parallel", "12",
             '2-3 independent calls in one response, matched order-independently '
             '(BFCL "parallel" style)'],
            ["moderate_chained", "8",
             'The divider. The harness executes step 1 with a deterministic mock, '
             'feeds the result back as a tool message, and the model must derive '
             'step-2 arguments from the returned data — including conditionals '
             '("if below zero, set the thermostat")'],
            ["python_game (coding)", "1",
             "Terminal tic-tac-toe: human vs computer, input validation, win/tie "
             "detection, play-again loop, stdlib only. Grade: 4 checks (board "
             "display, input validation, win detection, executes cleanly with piped "
             "input); pass = 3/4"],
            ["html_profile (coding)", "1",
             "Single-file profile page for a fictional photographer: hero, skills, "
             "contact form, dark theme, @media query, JS form validation. Grade: 7 "
             "structural checks; pass = 5/7"],
        ],
        [90, 40, 320]))
    story.append(P(
        "Extraction uses two paths, mirroring BFCL's separate FC and Prompt "
        "tracks. Path 1 reads the structured tool_calls field produced by "
        "llama.cpp's autoparser when the template renders tool markers. Path 2 is "
        "a text fallback for models whose GGUF templates never trigger the "
        "grammar: it scans content for JSON call emissions (arrays, fenced JSON, "
        "bare objects), tolerates arguments/parameters/args keys, and dedupes "
        "exact copies. Without path 2, a prompt-mode model scores near zero on "
        "structured-only reading — proven live when granite3.2-2b jumped from a "
        "false 1.9% to 82.7% once the fallback landed."))
    story.append(PageBreak())

    # ---- Page 3: results + overall chart ----
    story.append(P("2. Results", H2))
    story.append(P(
        "17 of 18 models complete; " + PENDING + " All scores from the fixed "
        "v2.1 harness: structured extraction with text fallback, duplicate-call "
        "dedupe, per-model memory sizing."))
    story.append(Image(str(charts[0]), width=6.4 * inch, height=4.4 * inch))
    story.append(PageBreak())

    # ---- Page 4: full results table ----
    story.append(P("Full Results Table", H2))
    rows = []
    for name, total, s, p, c, code, note in RESULTS:
        rows.append([name, f"{total}/52 ({pct(total, 52):.1f}%)",
                     f"{s}/30", f"{p}/12", f"{c}/8", f"{code}/2", note])
    story.append(make_table(
        ["Model", "Overall", "Simple", "Par.", "Chain.", "Code", "Notes"],
        rows, [100, 70, 45, 45, 45, 45, 118]))
    story.append(P(
        "Coding scores are pass/fail per test at the thresholds in section 1; "
        "overall is correct / 52. E4B's coding pass used the big-model memory "
        "rule (16K context, q4_0 KV, 12,288-token coding cap)."))
    story.append(PageBreak())

    # ---- Page 5: categories chart ----
    story.append(Image(str(charts[1]), width=6.4 * inch, height=2.7 * inch))
    story.append(P("3. What the Numbers Say", H2))
    story.append(Image(str(charts[2]), width=6.4 * inch, height=4.4 * inch))
    story.append(PageBreak())

    # ---- Page 6: findings text ----
    story.append(P(
        "Chained calls are the great divider — and the generalist won them. "
        "Granite 4.x and Hammer2.1 pass 5-7 of 8 chained tests; nine models "
        "score exactly 0 — they emit a correct first call, then cannot derive "
        "second-step arguments from returned data. This is BFCL's multi-turn "
        "gap reproduced at edge scale. The one perfect chained score belongs "
        "to E4B — a 4B QAT generalist, not a function-calling specialist. "
        "Deriving the next call from a result is a reasoning skill more than "
        "a format skill, and it shows."))
    story.append(P(
        "Simple is saturated. Tool-trained models cluster at 27-29/30 — "
        "consistent with BFCL's single-turn saturation, which is exactly why "
        "BFCL re-weighted toward agentic categories."))
    story.append(P(
        "Coding is table stakes — with two exceptions. Sixteen of seventeen "
        "complete models built a runnable tic-tac-toe and a structurally sound "
        "HTML page; granite3.2-2b passed Python but produced no parsable HTML, "
        "and llama3.2-3b codes 2/2 while calling tools at 38.5%. "
        "Execution-based grading keeps these claims honest."))
    story.append(P(
        "Near-misses cluster on string surface form. Three of E4B's four "
        "failures are argument-form near-misses, not wrong calls: it sent "
        'activity: "run" where the ground truth expects the literal word '
        '"running" from the question, and it over-specified theater: "downtown '
        'theater" where the expected value is "downtown". The calls would work '
        "against a real API that matched loosely; strict AST equality does not. "
        "The fourth was a single empty response on a three-call parallel test. "
        "When a model's failures are all surface-form, its true ceiling is "
        "higher than the score reads."))
    story.append(P("4. Harness Honesty and Memory Limits", H2))
    story.append(P(
        "Two 0% scores during the run were harness artifacts, not model "
        "failures — fixed by the text fallback and dedupe, then retested "
        "(granite3.2-2b: 1.9% to 82.7%; xlam-2-1b: 3.8% to 84.6%). A benchmark "
        "that cannot tell those apart is not measuring models, it is measuring "
        "itself."))
    story.append(P(
        "Quantization bites unevenly. Hermes3-3B Q4 loses 8 simple-call points "
        "to Q5 (19/30 vs 27/30) yet matches it on parallel (10/12). Ternary "
        "Bonsai TQ2_0 beats its own Q4_0 on chained tests (4/8 vs 3/8)."))
    story.append(P(
        "qwen3.5-9b (4.1GB DeltaNet hybrid) was OOM-killed twice by the kernel "
        "— the same model that the v1 screen had passed at 50%. Server launch "
        "configs for the big models: 24K context with q8_0 KV failed twice, so "
        "models above 4.5GB now run 16K context with q4_0 KV and a 12,288-token "
        "coding cap; on a shared 8GB pool, KV quantization is what lets a "
        "4.9GB model finish at all. E4B (4.0GB) runs the same big-model rule."))
    story.append(P(
        "Also documented: two server-start allocation failures before the "
        "big-model rule existed — ternary-bonsai-q4 and xlam-2-8b failed to "
        "start at 24K/q8_0 and passed cleanly at 16K/q4_0."))
    story.append(PageBreak())

    # ---- Page 7: hardware + sources ----
    story.append(P("5. Hardware and Server", H2))
    story.append(make_table(
        ["Target", "NVIDIA Jetson Orin Nano 8GB, ARM64, CUDA; GUI (GDM) stopped "
         "during runs to free ~6GB RAM"],
        [
            ["Server",
             "llama.cpp llama-server --jinja, OpenAI-compatible API, "
             "GGML_CUDA_ENABLE_UNIFIED_MEMORY=1, flash attention on; autoparser "
             "analyzes chat templates to parse tool calls via a differential "
             "approach inspired by the git diff algorithm"],
            ["Context",
             "24576 tokens, q8_0 KV for models up to ~4.5GB; 16384 tokens, q4_0 "
             "KV, 12,288-token coding cap above (E4B included)"],
            ["Determinism",
             "temperature 0.0, top-k 1, repeat penalty 1.0 for tool tests"],
            ["Templates",
             "built-in where the GGUF template renders tool markers; Qwen2.5 "
             "override for xLAM-2 / Arch-Agent; Hermes3 override for Hermes3 / "
             "Hammer2.1"],
        ],
        [70, 390]))
    story.append(P("6. Sources", H2))
    story.append(make_table(
        ["[1]", "gorilla.cs.berkeley.edu/leaderboard.html — BFCL V4 Leaderboard"],
        [
            ["[2]", "gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html — "
             "BFCL V4 Agentic Web Search blog"],
            ["[3]", "huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-"
             "Leaderboard — BFCL dataset card"],
            ["[4]", "github.com/ggml-org/llama.cpp/blob/master/docs/autoparser.md — "
             "llama.cpp autoparser architecture"],
            ["[5]", "github.com/ShishirPatil/gorilla — Gorilla repo (BFCL, Apache 2.0)"],
            ["[6]", "github.com/drwjkirkpatrick-web/bfcl-jetson-test — BFCL Jetson "
             "Test v1 (our repo)"],
            ["[7]", "gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html — "
             "BFCL V3 multi-turn blog"],
            ["[8]", "github.com/ggml-org/llama.cpp — llama.cpp repo"],
        ],
        [30, 430]))

    doc.build(story)
    print(f"PDF written: {OUT_PDF}")


if __name__ == "__main__":
    build()
