"""
modules/report_generator.py — CA Intelligence Suite
Phase 8: Professional 14-Section CA PDF Report Generator

Extracts the entire PDF generation function from app.py into a standalone module.
All 14 sections are preserved exactly; the only changes are:
    1. Imports moved to module-level (not inside the function)
    2. Color palette, style helpers extracted to top-level constants
    3. Logging added for section timing
    4. `inr_pdf` used from utils.helpers (removes inline ₹ fix)

Sections:
    Cover + Meta Table
    1.  Executive Summary
    2.  Statement of Profit & Loss
    3.  Balance Sheet (Validated)
    4.  Comparative Analysis (YoY)
    5.  Cash Flow Insight
    6.  GST Analysis
    7.  Financial Ratios
    8.  Expense Analysis (Top 3)
    9.  Graphical Analysis (Bar + Pie)
    10. Compliance Analysis
    11. Anomaly Detection
    12. CA Recommendations
    13. Conclusion
    14. Disclaimer
"""

from __future__ import annotations

import io as _io
from typing import List, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rl_colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_RIGHT, TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, PageBreak,
    TableStyle, HRFlowable, KeepTogether, Image,
)

from utils.logger import get_logger
from utils.helpers import inr_pdf, strip_emoji
from modules.financial_engine import classify_financial_health, compute_comparative_analysis, estimate_cash_flow
from modules.compliance_engine import validate_tax_provision, validate_balance_sheet

logger = get_logger(__name__)

# 
# Colour palette
# 
C_NAVY      = rl_colors.HexColor("#0f2744")
C_NAVY2     = rl_colors.HexColor("#1e3a5f")
C_BLUE      = rl_colors.HexColor("#2563eb")
C_MID_BLUE  = rl_colors.HexColor("#3b82f6")
C_LIGHT_BLU = rl_colors.HexColor("#93c5fd")
C_WHITE     = rl_colors.white
C_OFFWHITE  = rl_colors.HexColor("#f0f4ff")
C_ROW_ALT   = rl_colors.HexColor("#e8f0fe")
C_TEXT      = rl_colors.HexColor("#111827")
C_MUTED     = rl_colors.HexColor("#4b5563")
C_BORDER    = rl_colors.HexColor("#93c5fd")
C_GREEN     = rl_colors.HexColor("#065f46")
C_AMBER     = rl_colors.HexColor("#78350f")
C_RED       = rl_colors.HexColor("#7f1d1d")
C_GREEN_BG  = rl_colors.HexColor("#d1fae5")
C_AMBER_BG  = rl_colors.HexColor("#fef3c7")
C_RED_BG    = rl_colors.HexColor("#fee2e2")


# 
# Style helpers
# 
def _S(name: str, **kw) -> ParagraphStyle:
    return ParagraphStyle(name, **kw)


def _hr(thick: float = 0.5, color=C_BORDER) -> HRFlowable:
    return HRFlowable(width="100%", thickness=thick, color=color, spaceAfter=6, spaceBefore=2)


def _base_ts(header_rows: int = 1, align_right_from: int = 1) -> TableStyle:
    return TableStyle([
        ("BACKGROUND",     (0, 0),                (-1, header_rows-1),  C_NAVY),
        ("TEXTCOLOR",      (0, 0),                (-1, header_rows-1),  C_WHITE),
        ("FONTNAME",       (0, 0),                (-1, header_rows-1),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0),                (-1, -1),             8.5),
        ("ROWBACKGROUNDS", (0, header_rows),      (-1, -1),             [C_WHITE, C_ROW_ALT]),
        ("TEXTCOLOR",      (0, header_rows),      (-1, -1),             C_TEXT),
        ("GRID",           (0, 0),                (-1, -1),             0.3, C_BORDER),
        ("TOPPADDING",     (0, 0),                (-1, -1),             5),
        ("BOTTOMPADDING",  (0, 0),                (-1, -1),             5),
        ("LEFTPADDING",    (0, 0),                (-1, -1),             7),
        ("RIGHTPADDING",   (0, 0),                (-1, -1),             7),
        ("ALIGN",          (0, 0),                (-1, -1),             "LEFT"),
        ("ALIGN",          (align_right_from, 0), (-1, -1),             "RIGHT"),
        ("VALIGN",         (0, 0),                (-1, -1),             "MIDDLE"),
    ])


def _add_section(story: list, num: int, title: str, sH1: ParagraphStyle) -> None:
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"{num}. {title}", sH1))
    story.append(_hr(thick=1.0, color=C_NAVY))


# 
# Main Report Generator
# 
def generate_pdf_report(
    sd: Dict,
    df: pd.DataFrame,
    insights: List[Dict],
    summary,
    now_str: str,
    company_name: str = "ABC Private Limited",
    fy_year: str = "2024-25",
) -> bytes:
    """
    Generate a 14-section professional CA financial audit report as PDF bytes.

    Args:
        sd:           Summary dict from FinancialSummary.to_dict().
        df:           Full transactions DataFrame.
        insights:     List of insight dicts from compliance engine.
        summary:      FinancialSummary instance.
        now_str:      Formatted timestamp string for cover page.
        company_name: Client company name.
        fy_year:      Financial year string (e.g. "2024-25").

    Returns:
        PDF content as bytes — can be streamed directly to st.download_button.
    """
    logger.info(
        "generate_pdf_report: starting for '%s' FY %s (%d insights, %d rows)",
        company_name, fy_year, len(insights), len(df),
    )

    buf = _io.BytesIO()
    W   = A4[0] - 40 * mm

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=18*mm, bottomMargin=16*mm,
        title=f"{company_name}  CA Financial Report FY {fy_year}",
        author="CA Intelligence Suite v2.0",
    )

    #  Paragraph styles 
    sCo    = _S("sCo",    fontName="Helvetica-Bold", fontSize=20, textColor=C_WHITE,     alignment=TA_CENTER, leading=26)
    sTitle = _S("sT",     fontName="Helvetica-Bold", fontSize=13, textColor=C_LIGHT_BLU, alignment=TA_CENTER, spaceAfter=4)
    sSub   = _S("sSub",   fontName="Helvetica",      fontSize=8.5,textColor=C_LIGHT_BLU, alignment=TA_CENTER, spaceAfter=2)
    sH1    = _S("sH1",    fontName="Helvetica-Bold", fontSize=12, textColor=C_NAVY,      spaceBefore=14, spaceAfter=4, borderPad=4, backColor=C_OFFWHITE)
    sH2    = _S("sH2",    fontName="Helvetica-Bold", fontSize=10, textColor=C_NAVY2,     spaceBefore=8,  spaceAfter=3)
    sBody  = _S("sBody",  fontName="Helvetica",      fontSize=9,  textColor=C_TEXT,      leading=14, spaceAfter=6)
    sBodyJ = _S("sBodyJ", fontName="Helvetica",      fontSize=9,  textColor=C_TEXT,      leading=14, spaceAfter=6, alignment=TA_JUSTIFY)
    sItal  = _S("sItal",  fontName="Helvetica-Oblique", fontSize=8.5, textColor=C_MUTED, leading=13, spaceAfter=5)
    sFootB = _S("sFootB", fontName="Helvetica-Bold", fontSize=7.5, textColor=C_MUTED,   alignment=TA_CENTER)
    sDisc  = _S("sDisc",  fontName="Helvetica",      fontSize=8,  textColor=C_MUTED,    leading=12, alignment=TA_JUSTIFY)
    sBold  = _S("sBold",  fontName="Helvetica-Bold", fontSize=9,  textColor=C_TEXT,     leading=14)

    fmt = inr_pdf
    story: list = []

    #  Derived values 
    pm            = sd["Profit Margin %"]
    ti            = summary.total_income
    te            = summary.total_expense
    npl           = summary.net_profit_loss
    pbt           = summary.profit_before_tax
    health, _     = classify_financial_health(pm)
    health_colors = {
        "Strong":      (C_GREEN, C_GREEN_BG),
        "Moderate":    (C_AMBER, C_AMBER_BG),
        "Weak":        (C_RED,   C_RED_BG),
        "Loss-Making": (C_RED,   C_RED_BG),
    }
    h_text_c, h_bg_c = health_colors.get(health, (C_TEXT, C_WHITE))

    cat_col_use = "Category" if "Category" in df.columns else "Predicted_Category"
    exp_df = df[df.get(cat_col_use, pd.Series(dtype=str)) == "Expense"] \
             if cat_col_use in df.columns else pd.DataFrame()

    # 
    # COVER PAGE
    # 
    cover_data = [
        [Paragraph(company_name.upper(), sCo)],
        [Paragraph(f"Chartered Accountant Financial Audit Report — FY {fy_year}", sTitle)],
        [Paragraph(f"Prepared as per Schedule III, Companies Act 2013  |  Generated: {now_str}", sSub)],
    ]
    cover_t = Table(cover_data, colWidths=[W])
    cover_t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 16),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
        ("LEFTPADDING",   (0, 0), (-1, -1), 20),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 20),
        ("LINEBELOW",     (0, 2), (-1, 2),  3, C_MID_BLUE),
    ]))
    story.append(cover_t)
    story.append(Spacer(1, 14))

    meta_data = [
        ["Company",     company_name,           "Financial Year", fy_year],
        ["Report Type", "CA Audit Report",      "Health Status",  health],
        ["Prepared By", "CA Intelligence Suite","Report Date",    now_str.split(",")[0]],
    ]
    meta_t = Table(meta_data, colWidths=[W*0.18, W*0.32, W*0.18, W*0.32])
    meta_t.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",    (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR",   (0, 0), (-1, -1), C_TEXT),
        ("GRID",        (0, 0), (-1, -1), 0.3, C_BORDER),
        ("BACKGROUND",  (0, 0), (0, -1), C_OFFWHITE),
        ("BACKGROUND",  (2, 0), (2, -1), C_OFFWHITE),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0,0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(meta_t)
    story.append(Spacer(1, 12))

    # SECTION 1: EXECUTIVE SUMMARY
    _add_section(story, 1, "Executive Summary", sH1)

    exec_text = (
        f"On behalf of <b>{company_name}</b>, this report presents the audited financial position "
        f"for the financial year <b>FY {fy_year}</b>. "
        f"The business recorded total revenue of <b>{fmt(ti)}</b> against total expenditure of <b>{fmt(te)}</b>, "
        f"yielding a net {'profit' if npl >= 0 else 'loss'} of <b>{fmt(abs(npl))}</b>. "
        f"The resultant profit margin of <b>{pm:.2f}%</b> classifies the financial health as <b>{health}</b>."
    )
    story.append(Paragraph(exec_text, sBodyJ))
    story.append(Spacer(1, 6))

    advisory = {
        "Weak":        (
            f"As your Chartered Accountant, I must draw your attention to the fact that a profit margin of "
            f"{pm:.1f}% is below acceptable thresholds. Immediate corrective action on cost rationalisation, "
            "pricing review, and revenue acceleration is necessary to preserve business continuity."
        ),
        "Loss-Making": (
            f"As your Chartered Accountant, I must draw your attention to the fact that a profit margin of "
            f"{pm:.1f}% is below acceptable thresholds. Immediate corrective action on cost rationalisation, "
            "pricing review, and revenue acceleration is necessary to preserve business continuity."
        ),
        "Moderate":    (
            f"A profit margin of {pm:.1f}% is acceptable but leaves limited buffer against market volatility. "
            "I recommend a structured expense optimisation programme and targeted revenue diversification "
            "to strengthen the financial position over the next two quarters."
        ),
        "Strong":      (
            f"A profit margin of {pm:.1f}% indicates robust financial health. "
            "The business is well-positioned for strategic investment and expansion. "
            "I advise channelling surplus reserves into long-term assets or market development activities."
        ),
    }
    story.append(Paragraph(f"<i>CA Advisory: {advisory.get(health, '')}</i>", sItal))
    story.append(Spacer(1, 8))

    cr        = sd["Total Assets"] / max(sd["Total Liabilities"], 1)
    key_risk  = (
        "Insufficient cash cover against current liabilities"         if cr < 1 else
        "Critically low profit margin — profitability at risk"         if pm < 5 else
        "Moderate margin leaves limited downside buffer"               if pm < 15 else
        "No material financial risk identified at this time"
    )
    key_strength = (
        "Strong asset cover ratio"                                     if cr > 1.5 else
        "Positive operating profitability maintained"                  if pm > 0 else
        "Revenue base is established — focus required on cost management"
    )

    exec_kpi = [
        ["KPI", "Value"],
        ["Total Revenue",         fmt(ti)],
        ["Total Expenses",        fmt(te)],
        [f"Net {'Profit' if npl>=0 else 'Loss'}", fmt(abs(npl))],
        ["Profit Margin",         f"{pm:.2f}%"],
        ["Expense Ratio",         f"{te/max(ti,1)*100:.1f}%"],
        ["Financial Health",      health],
        ["Net GST Payable",       fmt(summary.gst_payable)],
        ["Key Risk",              key_risk],
        ["Key Strength",          key_strength],
    ]
    ek_t  = Table(exec_kpi, colWidths=[W*0.38, W*0.62])
    ek_ts = _base_ts(header_rows=1, align_right_from=0)
    ek_ts.add("ALIGN",      (0, 0),  (-1, -1),  "LEFT")
    ek_ts.add("FONTNAME",   (0, 6),  (-1, 6),   "Helvetica-Bold")
    ek_ts.add("TEXTCOLOR",  (1, 6),  (1,  6),   h_text_c)
    ek_ts.add("BACKGROUND", (1, 6),  (1,  6),   h_bg_c)
    ek_t.setStyle(ek_ts)
    story.append(ek_t)
    story.append(Spacer(1, 12))

    # SECTION 2: PROFIT & LOSS STATEMENT
    _add_section(story, 2, "Statement of Profit & Loss", sH1)
    story.append(Paragraph(
        f"(For the period ending 31st March, {fy_year.split('-')[1] if '-' in fy_year else fy_year})",
        sItal,
    ))

    pnl_data = [
        ["Particulars",                          "Amount (Rs.)"],
        ["I.  Revenue from Operations",          fmt(summary.revenue_from_operations)],
        ["II. Other Income",                     fmt(summary.other_income)],
        ["    TOTAL REVENUE (I + II)",            fmt(summary.total_income)],
        ["",                                     ""],
        ["III. Expenses",                        ""],
        ["     Cost of Material Consumed",       fmt(summary.cost_of_materials)],
        ["     Employee Benefit Expenses",       fmt(summary.employee_benefit_expense)],
        ["     Finance Costs",                   fmt(summary.finance_costs)],
        ["     Depreciation & Amortisation",     fmt(summary.depreciation_amortisation)],
        ["     Other Expenses",                  fmt(summary.other_expenses)],
        ["     TOTAL EXPENSES",                  fmt(summary.total_expense)],
        ["",                                     ""],
        ["IV. Profit Before Tax (I+II-III)",     fmt(summary.profit_before_tax)],
        ["V.  Less: Provision for Income Tax",   fmt(summary.short_term_provisions)],
        ["VI. NET PROFIT / (LOSS)",              fmt(summary.net_profit_loss)],
    ]
    pnl_t = Table(pnl_data, colWidths=[W*0.65, W*0.35])
    pts   = _base_ts(header_rows=1, align_right_from=1)
    for r in [3, 11, 13, 15]:
        pts.add("FONTNAME",   (0, r), (-1, r), "Helvetica-Bold")
        pts.add("BACKGROUND", (0, r), (-1, r), C_OFFWHITE)
    pts.add("FONTNAME",   (0, 15), (-1, 15), "Helvetica-Bold")
    pts.add("BACKGROUND", (0, 15), (-1, 15), C_NAVY)
    pts.add("TEXTCOLOR",  (0, 15), (-1, 15), C_WHITE)
    pnl_t.setStyle(pts)
    story.append(pnl_t)
    story.append(Paragraph(
        f"<i>The business generated a net {'profit' if npl >= 0 else 'loss'} of {fmt(abs(npl))}, "
        f"representing a profit margin of {pm:.2f}% on total revenue of {fmt(ti)}. "
        f"The expense ratio stands at {te/max(ti,1)*100:.1f}%, "
        f"indicating {'efficient' if te/max(ti,1) < 0.80 else 'elevated'} cost management.</i>",
        sItal,
    ))
    story.append(Spacer(1, 10))

    # SECTION 3: BALANCE SHEET
    _add_section(story, 3, "Balance Sheet (Validated) — As at 31st March", sH1)
    bv = validate_balance_sheet(summary)

    bs_data = [
        ["I.   EQUITY AND LIABILITIES",                    "Amount (Rs.)"],
        ["     (a) Shareholders' Funds",                   ""],
        ["          Share Capital",                        fmt(summary.share_capital)],
        ["          Reserves & Surplus",                   fmt(summary.reserves_surplus)],
        ["          Sub-total: Shareholders' Funds",       fmt(summary.shareholders_funds)],
        ["     (b) Non-Current Liabilities",               ""],
        ["          Long-term Borrowings",                 fmt(summary.long_term_borrowings)],
        ["          Deferred Tax Liability",               fmt(summary.deferred_tax_liability)],
        ["          Other Long-term Liabilities",          fmt(summary.other_lt_liabilities)],
        ["          Long-term Provisions",                 fmt(summary.lt_provisions)],
        ["          Sub-total: Non-Current Liab.",         fmt(summary.non_current_liabilities)],
        ["     (c) Current Liabilities",                   ""],
        ["          Short-term Borrowings",                fmt(summary.short_term_borrowings)],
        ["          Trade Payables",                       fmt(summary.trade_payables)],
        ["          Other Current Liabilities",            fmt(summary.other_current_liabilities)],
        ["          Short-term Provisions",                fmt(summary.short_term_provisions)],
        ["          Sub-total: Current Liabilities",       fmt(summary.current_liabilities)],
        ["TOTAL EQUITY & LIABILITIES",                     fmt(summary.total_equity_and_liabilities)],
        ["",                                               ""],
        ["II.  ASSETS",                                    "Amount (Rs.)"],
        ["     (a) Non-Current Assets",                    ""],
        ["          Tangible Fixed Assets (PPE)",          fmt(summary.tangible_assets)],
        ["          Intangible Assets",                    fmt(summary.intangible_assets)],
        ["          Capital Work-in-Progress",             fmt(summary.cwip)],
        ["          Non-current Investments",              fmt(summary.non_current_investments)],
        ["          Deferred Tax Assets",                  fmt(summary.deferred_tax_asset)],
        ["          Long-term Loans & Advances",           fmt(summary.lt_loans_advances)],
        ["          Sub-total: Non-Current Assets",        fmt(summary.non_current_assets)],
        ["     (b) Current Assets",                        ""],
        ["          Inventories",                          fmt(summary.inventories)],
        ["          Trade Receivables",                    fmt(summary.trade_receivables)],
        ["          Cash & Cash Equivalents",              fmt(summary.cash_equivalents)],
        ["          Short-term Loans & Advances",          fmt(summary.short_term_loans)],
        ["          Other Current Assets",                 fmt(summary.other_current_assets)],
        ["          Sub-total: Current Assets",            fmt(summary.current_assets)],
        ["TOTAL ASSETS",                                   fmt(summary.total_assets)],
    ]
    bs_t  = Table(bs_data, colWidths=[W*0.65, W*0.35])
    bsts  = _base_ts(header_rows=1, align_right_from=1)
    for r in [4, 10, 16, 17, 27, 34, 35]:
        bsts.add("FONTNAME",   (0, r), (-1, r), "Helvetica-Bold")
        bsts.add("BACKGROUND", (0, r), (-1, r), C_OFFWHITE)
    bsts.add("FONTNAME",   (0, 19), (-1, 19), "Helvetica-Bold")
    bsts.add("BACKGROUND", (0, 19), (-1, 19), C_NAVY)
    bsts.add("TEXTCOLOR",  (0, 19), (-1, 19), C_WHITE)
    bs_t.setStyle(bsts)
    story.append(bs_t)
    story.append(Spacer(1, 4))

    ver_text = (
        f"VERIFICATION: Balance Sheet BALANCES. "
        f"Total Assets ({fmt(summary.total_assets)}) = "
        f"Total Equity & Liabilities ({fmt(summary.total_equity_and_liabilities)}). "
        + (bv.get("adjustment_note", "") or "")
    ) if bv["balanced"] else (
        f"VERIFICATION: Balance Sheet MISMATCH. "
        f"Assets ({fmt(summary.total_assets)}) ≠ "
        f"Equity + Liabilities ({fmt(summary.total_equity_and_liabilities)}). "
        f"Difference: {fmt(abs(bv['gap']))}. This requires investigation."
    )
    ver_bg = C_GREEN_BG if bv["balanced"] else C_RED_BG
    ver_tc = C_GREEN    if bv["balanced"] else C_RED

    ver_t = Table(
        [[Paragraph(ver_text, _S("vt", fontName="Helvetica-Bold", fontSize=8, textColor=ver_tc, leading=12))]],
        colWidths=[W],
    )
    ver_t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), ver_bg),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("GRID",          (0,0), (-1,-1), 0.5, ver_tc),
    ]))
    story.append(ver_t)
    story.append(Spacer(1, 10))
    story.append(PageBreak())

    # SECTION 4: COMPARATIVE ANALYSIS
    _add_section(story, 4, "Comparative Financial Analysis (Year-on-Year)", sH1)
    ca = compute_comparative_analysis(df)

    if ca is None or not ca.get("available"):
        note = ca["note"] if ca else "Insufficient date information for comparative analysis."
        story.append(Paragraph(note, sItal))
    else:
        def _yoy(v):
            if v is None: return "N/A"
            return f" {v:.1f}%" if v > 0 else f" {abs(v):.1f}%"
        ca_data = [
            ["Metric",             ca["prev_fy"],               ca["curr_fy"],               "Growth"],
            ["Total Revenue",      fmt(ca["previous_revenue"]), fmt(ca["current_revenue"]),  _yoy(ca["revenue_growth_pct"])],
            ["Total Expenses",     fmt(ca["previous_expense"]), fmt(ca["current_expense"]),  _yoy(ca["expense_growth_pct"])],
            ["Net Profit/(Loss)",  fmt(ca["previous_profit"]),  fmt(ca["current_profit"]),   _yoy(ca["profit_growth_pct"])],
        ]
        ca_t  = Table(ca_data, colWidths=[W*0.30, W*0.23, W*0.23, W*0.24])
        cats  = _base_ts(header_rows=1, align_right_from=1)
        cats.add("FONTNAME",   (0, 3), (-1, 3), "Helvetica-Bold")
        cats.add("BACKGROUND", (0, 3), (-1, 3), C_OFFWHITE)
        ca_t.setStyle(cats)
        story.append(ca_t)
        trend_text = (
            f"Revenue {'grew' if ca.get('revenue_growth_pct') and ca['revenue_growth_pct'] > 0 else 'declined'} "
            f"by {_yoy(ca['revenue_growth_pct'])} from {ca['prev_fy']} to {ca['curr_fy']}. "
            f"Expenses {'increased' if ca.get('expense_growth_pct') and ca['expense_growth_pct'] > 0 else 'decreased'} "
            f"by {_yoy(ca['expense_growth_pct'])}. "
            f"The overall profit trend is <b>{ca['trend']}</b>."
        )
        story.append(Paragraph(f"<i>{trend_text}</i>", sItal))
    story.append(Spacer(1, 10))

    
    # SECTION 5: CASH FLOW INSIGHT
    _add_section(story, 5, "Cash Flow Insight (Indirect Method Estimate)", sH1)
    cf = estimate_cash_flow(summary, df)
    cf_data = [
        ["Cash Flow Component",                "Amount (Rs.)"],
        ["Net Profit for the Period",          fmt(cf["net_profit"])],
        ["Add: Depreciation & Amortisation",   fmt(cf["depreciation"])],
        ["Add: Finance Costs",                 fmt(cf["finance_cost"])],
        ["Less: Working Capital Change",       f"({fmt(cf['wc_change'])})"],
        ["A. OPERATING CASH FLOW",             fmt(cf["operating_cf"])],
        ["B. INVESTING CASH FLOW (estimated)", fmt(cf["investing_cf"])],
        ["C. FINANCING CASH FLOW (estimated)", fmt(cf["financing_cf"])],
        ["NET CASH FLOW (A + B + C)",          fmt(cf["net_cf"])],
        ["Cash & Cash Equivalents on Hand",    fmt(cf["cash_on_hand"])],
    ]
    cf_t  = Table(cf_data, colWidths=[W*0.65, W*0.35])
    cfts  = _base_ts(header_rows=1, align_right_from=1)
    for r in [5, 8]:
        cfts.add("FONTNAME",   (0, r), (-1, r), "Helvetica-Bold")
        cfts.add("BACKGROUND", (0, r), (-1, r), C_OFFWHITE)
    cf_t.setStyle(cfts)
    story.append(cf_t)
    story.append(Spacer(1, 4))

    liq    = cf["liquidity"]
    liq_tc, liq_bg = {"Strong": (C_GREEN, C_GREEN_BG), "Moderate": (C_AMBER, C_AMBER_BG)}.get(
        liq, (C_RED, C_RED_BG)
    )
    liq_t = Table(
        [[Paragraph(
            f"Liquidity Assessment: <b>{liq.upper()}</b> — {strip_emoji(cf['interpretation'])}",
            _S("lt", fontName="Helvetica", fontSize=8.5, textColor=liq_tc, leading=13),
        )]],
        colWidths=[W],
    )
    liq_t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), liq_bg),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("GRID",          (0,0), (-1,-1), 0.3, liq_tc),
    ]))
    story.append(liq_t)
    story.append(Paragraph(
        "<i>Note: Cash flow figures are estimated using an indirect method proxy from ledger data. "
        "Actual cash flow statement requires bank reconciliation and may differ.</i>", sItal,
    ))
    story.append(Spacer(1, 10))

    # SECTION 6: GST ANALYSIS
    _add_section(story, 6, "GST Analysis", sH1)
    gst_data = [
        ["GST Head",                                  "Amount (Rs.)"],
        ["Output GST (Collected on Sales)",           fmt(summary.gst_on_income)],
        ["Less: Input GST Credit (on Purchases)",     f"({fmt(summary.gst_on_expense)})"],
        ["NET GST PAYABLE",                           fmt(summary.gst_payable)],
    ]
    gst_t = Table(gst_data, colWidths=[W*0.65, W*0.35])
    gts   = _base_ts(header_rows=1, align_right_from=1)
    gts.add("FONTNAME",   (0, 3), (-1, 3), "Helvetica-Bold")
    gts.add("BACKGROUND", (0, 3), (-1, 3), C_OFFWHITE)
    gst_t.setStyle(gts)
    story.append(gst_t)
    story.append(Paragraph(
        f"<i>Net GST liability for the period: {fmt(summary.gst_payable)}. "
        "Input Tax Credit must be reconciled with GSTR-2B before filing GSTR-3B. "
        "Ensure blocked credits per Section 17(5) CGST Act are excluded from ITC claims.</i>",
        sItal,
    ))
    story.append(Spacer(1, 10))
    story.append(PageBreak())

    # SECTION 7: FINANCIAL RATIOS
    _add_section(story, 7, "Financial Ratios & Interpretation", sH1)

    exp_ratio_val  = summary.total_expense / max(summary.total_income, 1)
    curr_bor       = summary.short_term_borrowings + summary.long_term_borrowings
    debt_eq_val    = curr_bor / max(summary.shareholders_funds, 1)
    curr_ratio_val = summary.current_assets / max(summary.current_liabilities, 1)

    def _ratio_status(name: str, value: float) -> str:
        if name == "Profit Margin":
            return (" Strong" if value >= 15 else " Moderate" if value >= 5 else " Weak" if value >= 0 else " Loss")
        if name == "Expense Ratio":
            v = value * 100
            return " High" if v > 90 else (" Elevated" if v > 80 else " Efficient")
        if name == "Debt-to-Equity":
            return " High Leverage" if value > 2 else (" Moderate" if value > 1 else " Low Leverage")
        if name == "Current Ratio":
            return " Liquidity Concern" if value < 1 else (" Healthy" if value >= 1.5 else " Borderline")
        return "—"

    ratio_data = [
        ["Ratio",          "Formula",                               "Value",                "Status",                          "Interpretation"],
        ["Profit Margin",  "Net Profit / Revenue × 100",           f"{pm:.1f}%",           _ratio_status("Profit Margin", pm),
         f"{'Above' if pm >= 15 else 'Below'} the 15% strong-business benchmark."],
        ["Expense Ratio",  "Total Expenses / Total Revenue × 100", f"{exp_ratio_val*100:.1f}%", _ratio_status("Expense Ratio", exp_ratio_val),
         f"{'Well-controlled' if exp_ratio_val < 0.8 else 'Elevated'} cost base."],
        ["Debt-to-Equity", "Total Borrowings / Shareholders' Funds", f"{debt_eq_val:.2f}x", _ratio_status("Debt-to-Equity", debt_eq_val),
         f"{'Conservative' if debt_eq_val < 1 else 'Moderate'} external debt financing."],
        ["Current Ratio",  "Current Assets / Current Liabilities", f"{curr_ratio_val:.2f}x", _ratio_status("Current Ratio", curr_ratio_val),
         f"Short-term obligations {'adequately' if curr_ratio_val >= 1 else 'not fully'} covered."],
    ]
    rt_t  = Table(ratio_data, colWidths=[W*0.16, W*0.26, W*0.10, W*0.15, W*0.33])
    rtts  = _base_ts(header_rows=1, align_right_from=2)
    rtts.add("ALIGN", (4, 0), (4, -1), "LEFT")
    rtts.add("ALIGN", (1, 0), (1, -1), "LEFT")
    rtts.add("FONTSIZE", (0, 0), (-1, -1), 8)
    rt_t.setStyle(rtts)
    story.append(rt_t)
    story.append(Spacer(1, 10))

    # SECTION 8: EXPENSE ANALYSIS
    _add_section(story, 8, "Expense Analysis — Top Categories", sH1)
    if not exp_df.empty and "Sub_Category" in exp_df.columns:
        top_cats = exp_df.groupby("Sub_Category")["Amount"].sum().sort_values(ascending=False).head(3)
        ea_data  = [["Rank", "Expense Category", "Amount (Rs.)", "% of Total Expenses", "Observation"]]
        for rank, (cat, amt) in enumerate(top_cats.items(), 1):
            pct_v = amt / max(summary.total_expense, 1) * 100
            obs   = (
                "Primary cost driver — monitor closely" if rank == 1 else
                "Major outflow — review for efficiency" if rank == 2 else
                "Significant — assess budget alignment"
            )
            ea_data.append([str(rank), cat, fmt(amt), f"{pct_v:.1f}%", obs])
        ea_t = Table(ea_data, colWidths=[W*0.06, W*0.28, W*0.18, W*0.18, W*0.30])
        ea_t.setStyle(_base_ts(header_rows=1, align_right_from=2))
        story.append(ea_t)
        story.append(Paragraph(
            f"<i>'{top_cats.index[0]}' is the primary cost driver. "
            "Regular budget monitoring and variance analysis against prior periods is recommended.</i>",
            sItal,
        ))
    else:
        story.append(Paragraph("Expense sub-category data not available in the uploaded dataset.", sBody))
    story.append(Spacer(1, 10))

    # SECTION 9: GRAPHICAL ANALYSIS
    _add_section(story, 9, "Graphical Analysis", sH1)
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
        fig.patch.set_facecolor("white")
        bars = ax1.bar(
            ["Revenue", "Expenses"],
            [summary.total_income, summary.total_expense],
            color=["#2563eb", "#1e40af"], width=0.45, edgecolor="white", linewidth=0.5,
        )
        ax1.set_title("Revenue vs Expenses", fontsize=10, fontweight="bold", color="#0f2744", pad=8)
        ax1.set_ylabel("Amount (Rs.)", fontsize=8, color="#4b5563")
        ax1.tick_params(colors="#4b5563", labelsize=8)
        ax1.set_facecolor("#f8faff")
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h * 1.02,
                     f"Rs.{h:,.0f}", ha="center", va="bottom", fontsize=7.5,
                     color="#0f2744", fontweight="bold")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        if not exp_df.empty and "Sub_Category" in exp_df.columns:
            opts       = exp_df.groupby("Sub_Category")["Amount"].sum().sort_values(ascending=False).head(5)
            pie_colors = ["#1e40af", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd"]
            wedges, texts, autotexts = ax2.pie(
                opts.values, labels=None, autopct="%1.1f%%",
                startangle=140, colors=pie_colors[:len(opts)],
                pctdistance=0.78, wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
            )
            for at in autotexts:
                at.set_fontsize(8)
                at.set_color("white")
            ax2.legend(opts.index, loc="lower right", fontsize=7, framealpha=0.9)
            ax2.set_title("Expense Distribution", fontsize=10, fontweight="bold", color="#0f2744", pad=8)
        else:
            ax2.text(0.5, 0.5, "No Sub-Category Data", ha="center", va="center", fontsize=10, color="#6b7280")
            ax2.set_title("Expense Distribution", fontsize=10, fontweight="bold", color="#0f2744")

        plt.tight_layout(pad=1.5)
        img_buf = _io.BytesIO()
        plt.savefig(img_buf, format="png", dpi=160, bbox_inches="tight", facecolor="white")
        img_buf.seek(0)
        plt.close(fig)
        story.append(Image(img_buf, width=W, height=W*0.42))

    except Exception as ex:
        story.append(Paragraph(f"<i>[Chart rendering failed: {strip_emoji(str(ex))}]</i>", sItal))

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"<b>Bar Chart:</b> Revenue of {fmt(ti)} vs Expenses of {fmt(te)}. "
        f"{'Revenue exceeds expenses — profitable.' if ti > te else 'Expenses exceed revenue — net loss.'} "
        f"Net {'profit' if npl >= 0 else 'loss'}: {fmt(abs(npl))}.",
        sBody,
    ))
    story.append(Paragraph(
        "<b>Pie Chart:</b> The expense distribution chart reveals the relative share of each expense "
        "sub-category — categories with a disproportionately large slice are strategic cost-optimisation targets.",
        sBody,
    ))
    story.append(Spacer(1, 10))
    story.append(PageBreak())

    # SECTION 10: COMPLIANCE ANALYSIS
    _add_section(story, 10, "Compliance Analysis — Tax & Regulatory", sH1)
    story.append(Paragraph(
        "The following compliance observations are presented in order of severity. "
        "Each finding is classified using a traffic-light system: "
        " Critical (immediate action required),  Warning (review recommended),  Informational (advisory note).",
        sBodyJ,
    ))
    story.append(Spacer(1, 4))

    tax_insights       = [i for i in insights if "[TAX" in i.get("title", "")]
    compliance_meta    = [
        ("critical", "CRITICAL", C_RED,   C_RED_BG),
        ("warning",  "WARNING",  C_AMBER, C_AMBER_BG),
        ("info",     "INFO",     C_NAVY,  C_OFFWHITE),
    ]
    for level, label, tc, bg in compliance_meta:
        level_items = [i for i in tax_insights if i["level"] == level]
        if level_items:
            story.append(Paragraph(f"<b>{label}</b>", _S("cl", fontName="Helvetica-Bold", fontSize=9, textColor=tc, spaceAfter=2)))
            for ins in level_items:
                cmp_tbl = Table([[
                    Paragraph(f"<b>{strip_emoji(ins['title'])}</b>",
                              _S("ct", fontName="Helvetica-Bold", fontSize=8.5, textColor=tc)),
                    Paragraph(strip_emoji(ins["message"]),
                              _S("cm", fontName="Helvetica", fontSize=8, textColor=C_TEXT, leading=12)),
                ]], colWidths=[W*0.32, W*0.68])
                cmp_tbl.setStyle(TableStyle([
                    ("BACKGROUND",    (0, 0), (-1, -1), bg),
                    ("GRID",          (0, 0), (-1, -1), 0.3, tc),
                    ("TOPPADDING",    (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
                    ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ]))
                story.append(cmp_tbl)
                story.append(Spacer(1, 4))

    if not tax_insights:
        story.append(Paragraph("No outstanding tax compliance violations flagged for the period.", sBody))

    tv     = validate_tax_provision(summary)
    tv_tc  = C_GREEN if tv["status"] == "ok" else (C_AMBER if tv["status"] == "warning" else C_TEXT)
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<b>Tax Provision Validation:</b> {strip_emoji(tv['finding'])}",
                            _S("tvp", fontName="Helvetica", fontSize=8.5, textColor=tv_tc, leading=13, spaceAfter=3)))
    story.append(Paragraph(f"<i>{strip_emoji(tv['recommendation'])}</i>", sItal))
    story.append(Spacer(1, 10))

    # 
    # SECTION 11: ANOMALY DETECTION
    # 
    _add_section(story, 11, "Anomaly Detection & Unusual Transactions", sH1)

    if "Amount" in df.columns and len(df) > 10:
        amounts   = df["Amount"].dropna()
        q1, q3   = amounts.quantile(0.25), amounts.quantile(0.75)
        iqr       = q3 - q1
        threshold = q3 + 3.0 * iqr
        outlier_df = df[df["Amount"] > threshold].copy()

        story.append(Paragraph(
            f"Statistical outlier threshold: {fmt(threshold)} (Q3 + 3×IQR using {len(df):,} transactions). "
            "Transactions exceeding this threshold are flagged for individual vouching.",
            sBody,
        ))

        if not outlier_df.empty:
            story.append(Paragraph(
                f"<b>{len(outlier_df)} anomalous transaction(s) detected.</b> "
                "Each entry mathematically deviates from regular ledger patterns.",
                sBody,
            ))
            show_cols = [c for c in ["Date", "Description", "Amount", "Category", "Payment_Mode"]
                         if c in outlier_df.columns]
            top_anoms = outlier_df.nlargest(min(10, len(outlier_df)), "Amount")[show_cols]
            anom_data = [top_anoms.columns.tolist()]
            for _, row in top_anoms.iterrows():
                anom_data.append([
                    fmt(row[c]) if c == "Amount" else str(row[c])[:40]
                    for c in show_cols
                ])
            at_t = Table(anom_data, colWidths=[W/len(show_cols)]*len(show_cols))
            at_t.setStyle(_base_ts(header_rows=1, align_right_from=2))
            at_t.setStyle(TableStyle([("FONTSIZE", (0,0), (-1,-1), 7.5)]))
            story.append(at_t)
        else:
            story.append(Paragraph("No statistically anomalous transactions detected.", sBody))
    else:
        story.append(Paragraph("Insufficient data for anomaly detection.", sBody))

    story.append(Spacer(1, 10))
    story.append(PageBreak())

    # 
    # SECTION 12: RECOMMENDATIONS
    # 
    _add_section(story, 12, "CA Recommendations", sH1)
    story.append(Paragraph(
        "The following recommendations are derived from the financial data analysed in this report. "
        "Each recommendation is actionable and specific to the observed financial position.",
        sBodyJ,
    ))
    story.append(Spacer(1, 6))

    # Derive current ratio for use in recommendations
    curr_ratio_val = summary.current_assets / max(summary.current_liabilities, 1)

    recs: list[tuple[str, str]] = []
    if pm < 5:
        recs.append(("Profit Margin Improvement",
            f"The current profit margin of {pm:.1f}% is critically low. "
            "Implement a 90-day cost review targeting the top 3 expense categories and "
            "evaluate pricing against market benchmarks."))
    if summary.total_expense / max(summary.total_income, 1) > 0.85:
        recs.append(("Expense Rationalisation",
            f"Expense-to-income ratio of {summary.total_expense/max(summary.total_income,1)*100:.1f}% "
            "is unsustainably high. Implement departmental budget caps and require prior approval for "
            "discretionary expenditure above defined thresholds."))
    if curr_ratio_val < 1.2:
        recs.append(("Working Capital Management",
            f"Current ratio of {curr_ratio_val:.2f}x indicates potential liquidity stress. "
            "Reduce debtor collection days below 30, negotiate extended payable terms, and establish a minimum cash reserve."))
    if ti > 1_00_00_000:
        recs.append(("Statutory Tax Audit",
            f"Gross receipts of {fmt(ti)} exceed the Rs.1 Crore threshold under Section 44AB. "
            "Appoint a CA for tax audit and ensure Forms 3CA/3CB/3CD are filed within the statutory due date."))
    recs.append(("GST Compliance",
        "Perform monthly GSTR-2B reconciliation before filing GSTR-3B to ensure all eligible ITC is claimed "
        "and blocked credits per Section 17(5) are reversed."))
    recs.append(("Cash Payment Controls",
        "Implement a policy requiring all business payments above Rs.5,000 to be made through banking channels. "
        "This mitigates disallowance risk under Sections 40A(3) and 269ST."))
    recs.append(("TDS & Advance Tax",
        "Establish a quarterly tax compliance calendar: TDS deposits (Forms 26Q, 24Q) by the 7th of each "
        "following month, and advance tax in four statutory tranches per Section 208."))
    if summary.long_term_borrowings > summary.shareholders_funds * 2:
        recs.append(("Debt Restructuring",
            "Long-term borrowings are disproportionately high relative to equity. "
            f"Evaluate structured repayment plans to reduce finance cost burden ({fmt(summary.finance_costs)} p.a.)."))

    for num, (title, text) in enumerate(recs, 1):
        rec_tbl = Table([[
            Paragraph(str(num), _S("rn", fontName="Helvetica-Bold", fontSize=11, textColor=C_WHITE, alignment=TA_CENTER)),
            Paragraph(f"<b>{title}</b><br/>{text}", _S("rt", fontName="Helvetica", fontSize=8.5, textColor=C_TEXT, leading=13)),
        ]], colWidths=[W*0.06, W*0.94])
        rec_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (0, 0), C_NAVY),
            ("BACKGROUND",    (1, 0), (1, 0), C_OFFWHITE),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("GRID",          (0, 0), (-1, -1), 0.3, C_BORDER),
        ]))
        story.append(rec_tbl)
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 10))

    # 
    # SECTION 13: CONCLUSION
    # 
    _add_section(story, 13, "Conclusion", sH1)
    risk_level = "High" if pm < 5 or curr_ratio_val < 1 else ("Moderate" if pm < 15 else "Low")
    outlook    = {
        "High":     (
            "The current financial position requires urgent management attention. "
            "The combination of low profitability and potential liquidity constraints poses a material risk "
            "to business continuity in the near term."
        ),
        "Moderate": (
            "The business is financially stable with areas requiring improvement. "
            "With focused cost management and consistent revenue growth, the business has the potential "
            "to achieve stronger financial health within the next one to two financial years."
        ),
        "Low":      (
            "The business demonstrates sound financial health and strong management of its resources. "
            "The current trajectory, if maintained, positions the company well for sustainable growth "
            "and strategic value creation over the coming years."
        ),
    }.get(risk_level, "")

    conclusion_text = (
        f"{company_name} closed FY {fy_year} with total revenue of <b>{fmt(ti)}</b> and a net "
        f"{'profit' if npl >= 0 else 'loss'} of <b>{fmt(abs(npl))}</b>, reflecting a profit margin of "
        f"<b>{pm:.2f}%</b>. The financial health of the entity is classified as <b>{health}</b> "
        f"with a risk level of <b>{risk_level}</b>.<br/><br/>"
        f"{outlook}<br/><br/>"
        "<b>Based on the analysis, the financial statements present a reasonable view of the financial "
        "position and performance of the entity for the reported period.</b><br/><br/>"
        "All statutory compliance obligations — including GST filing, TDS deposits, advance tax payments, "
        "and statutory audit — must be adhered to on a timely basis to protect the business from financial "
        "penalties and reputational risk."
    )
    story.append(Paragraph(conclusion_text, sBodyJ))
    story.append(Spacer(1, 20))

    # 
    # SECTION 14: DISCLAIMER
    # 
    _add_section(story, 14, "Disclaimer", sH1)
    story.append(_hr(thick=0.5))
    story.append(Paragraph(
        "This report has been generated using an AI-based financial analysis system (CA Intelligence Suite v2.0) "
        "and is intended solely for informational and advisory purposes. "
        "The financial data, computations, and insights presented herein are derived from the ledger data "
        "uploaded by the user and are subject to the accuracy and completeness of that underlying data.<br/><br/>"
        "This report does not constitute a Statutory Audit Report, Tax Audit Report, or any official "
        "certification under any provision of the Companies Act 2013, the Income-tax Act 1961, "
        "the CGST Act 2017, or any other applicable legislation. "
        "It is not a substitute for professional advice from a qualified Chartered Accountant "
        "holding a valid Certificate of Practice issued by the ICAI.<br/><br/>"
        "The AI-generated insights, tax compliance observations, and financial ratio interpretations "
        "are indicative in nature and should be independently verified by a qualified finance professional "
        "before any business, tax, or investment decisions are made on the basis of this report.<br/><br/>"
        "The CA Intelligence Suite, its developers, and associated parties shall not be held liable "
        "for any financial loss, penalty, or legal consequence arising from the use of this report.",
        sDisc,
    ))

    doc.build(story)
    buf.seek(0)
    result = buf.read()
    logger.info("generate_pdf_report: complete — %.1f KB", len(result) / 1024)
    return result
