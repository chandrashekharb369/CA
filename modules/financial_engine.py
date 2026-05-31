"""
modules/financial_engine.py — CA Intelligence Suite
Phase 5: Financial Engine — Computations & Intelligence

Provides:
    - compute_financials(df, cat_col)         → FinancialSummary
    - classify_financial_health(pm)           → (label, color_hint)
    - estimate_cash_flow(summary, df)         → cash flow dict
    - compute_comparative_analysis(df)        → YoY analysis dict
    - financial_health_alert(summary)         → cost escalation alerts

Financial correctness is the top priority in this module.
All computations delegate to BackwardChainingRuleEngine for Schedule III accuracy.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from config import (
    PROFIT_MARGIN_STRONG,
    PROFIT_MARGIN_MODERATE,
    PROFIT_MARGIN_WEAK,
)
from utils.logger import get_logger
from utils.helpers import inr_format

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core Computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_financials(df: pd.DataFrame, cat_col: str = "Category"):
    """
    Run the backward chaining rule engine and return a FinancialSummary.

    All Schedule III line items (P&L + Balance Sheet) are computed here.

    Args:
        df:      Financial transactions DataFrame.
        cat_col: Column name containing the category ('Category' or 'Predicted_Category').

    Returns:
        FinancialSummary instance (see rule_engine.FinancialSummary).
    """
    from rule_engine import run_rule_engine
    logger.info("compute_financials: running rule engine on %d rows (col='%s')", len(df), cat_col)
    try:
        summary = run_rule_engine(df, category_col=cat_col)
        logger.info(
            "compute_financials: complete — income=%.0f, expense=%.0f, PBT=%.0f",
            summary.total_income, summary.total_expense, summary.profit_before_tax,
        )
        return summary
    except Exception as exc:
        logger.error("compute_financials: rule engine failed — %s", exc, exc_info=True)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Financial Health Classification
# ─────────────────────────────────────────────────────────────────────────────
def classify_financial_health(pm: float) -> Tuple[str, str]:
    """
    Classify financial health based on profit margin percentage.

    Args:
        pm: Profit margin % (can be negative for loss-making entities).

    Returns:
        Tuple of (classification, color_hint):
            classification: 'Strong' | 'Moderate' | 'Weak' | 'Loss-Making'
            color_hint:     'green'  | 'amber'    | 'red'  | 'red'
    """
    if pm >= PROFIT_MARGIN_STRONG:
        return "Strong", "green"
    elif pm >= PROFIT_MARGIN_MODERATE:
        return "Moderate", "amber"
    elif pm >= PROFIT_MARGIN_WEAK:
        return "Weak", "red"
    else:
        return "Loss-Making", "red"


# ─────────────────────────────────────────────────────────────────────────────
# Financial Intelligence Alerts (Phase 5 — cost escalation detection)
# ─────────────────────────────────────────────────────────────────────────────
def financial_health_alerts(summary, df: pd.DataFrame) -> List[Dict]:
    """
    Generate intelligent alerts by comparing expense and revenue growth trends.

    Detects:
        - Cost escalation (expense growth > revenue growth)
        - Revenue stagnation with rising costs
        - Finance cost burden above threshold
        - Cash crunch vs GST payable

    Args:
        summary: FinancialSummary from the rule engine.
        df:      Raw transactions DataFrame (for trend analysis).

    Returns:
        List of alert dicts with keys: level, icon, title, message.
    """
    alerts: List[Dict] = []

    # ── Cost escalation risk (YoY) ────────────────────────────────────────────
    try:
        ca = compute_comparative_analysis(df)
        if ca and ca.get("available"):
            rev_growth = ca.get("revenue_growth_pct") or 0.0
            exp_growth = ca.get("expense_growth_pct") or 0.0

            if exp_growth > rev_growth and exp_growth > 0:
                alerts.append({
                    "level":   "warning",
                    "icon":    "📈",
                    "title":   "Cost Escalation Risk",
                    "message": (
                        f"Expenses grew by {exp_growth:.1f}% YoY while revenue grew by only {rev_growth:.1f}%. "
                        "Rising costs outpacing revenue is a leading indicator of margin compression. "
                        "Immediate cost rationalisation is recommended."
                    ),
                })
                logger.warning(
                    "financial_health_alerts: cost escalation — expense_growth=%.1f%% > revenue_growth=%.1f%%",
                    exp_growth, rev_growth,
                )
    except Exception:
        pass  # YoY analysis requires multi-year data — not always available

    # ── Finance cost burden ───────────────────────────────────────────────────
    if summary.total_expense > 0:
        fc_ratio = summary.finance_costs / summary.total_expense
        if fc_ratio > 0.15:
            alerts.append({
                "level":   "warning",
                "icon":    "🏦",
                "title":   "High Finance Cost Burden",
                "message": (
                    f"Finance costs represent {fc_ratio*100:.1f}% of total expenses "
                    f"({inr_format(summary.finance_costs)}). "
                    "High debt servicing costs suppress profitability. Consider refinancing at lower rates."
                ),
            })

    # ── Cash vs GST payable ───────────────────────────────────────────────────
    if summary.gst_payable > 0 and summary.cash_equivalents < summary.gst_payable:
        alerts.append({
            "level":   "critical",
            "icon":    "💰",
            "title":   "Cash Crunch vs GST Liability",
            "message": (
                f"Cash and equivalents ({inr_format(summary.cash_equivalents)}) are insufficient "
                f"to cover pending GST liability ({inr_format(summary.gst_payable)}). "
                "Immediate cash flow planning and receivables acceleration is required."
            ),
        })

    return alerts


# ─────────────────────────────────────────────────────────────────────────────
# Cash Flow Estimation (Indirect Method)
# ─────────────────────────────────────────────────────────────────────────────
def estimate_cash_flow(summary, df: pd.DataFrame) -> Dict:
    """
    Estimate operating cash flow and net liquidity using an indirect method proxy.

    Formula:
        Operating CF = Net Profit + Depreciation + Finance Costs − ΔWorking Capital
        Investing CF = −(Tangible + Intangible + CWIP Assets)
        Financing CF =  Long-term Borrowings + Short-term Borrowings − Share Capital

    Args:
        summary: FinancialSummary from the rule engine.
        df:      Raw transactions DataFrame.

    Returns:
        Dict with keys: net_profit, depreciation, finance_cost, wc_change,
        operating_cf, investing_cf, financing_cf, net_cf, cash_on_hand,
        liquidity, interpretation.
    """
    net_profit   = summary.net_profit_loss
    depreciation = summary.depreciation_amortisation
    finance_cost = summary.finance_costs

    wc_uses   = summary.trade_receivables + summary.inventories
    wc_source = summary.trade_payables
    wc_change = wc_uses - wc_source  # Positive = cash absorbed by working capital

    operating_cf = net_profit + depreciation + finance_cost - wc_change
    investing_cf = -(summary.tangible_assets + summary.intangible_assets + summary.cwip)
    financing_cf = (summary.long_term_borrowings + summary.short_term_borrowings + summary.share_capital)
    net_cf       = operating_cf + investing_cf + financing_cf
    cash_on_hand = summary.cash_equivalents

    # Liquidity classification
    if cash_on_hand > summary.current_liabilities * 0.5 and operating_cf > 0:
        liquidity = "Strong"
    elif operating_cf > 0:
        liquidity = "Moderate"
    else:
        liquidity = "Weak"

    interpretation = (
        f"Liquidity position is {liquidity.upper()}. "
        f"Estimated operating cash flow is {inr_format(operating_cf)}, "
        f"with {inr_format(cash_on_hand)} in cash and cash equivalents. " + (
            "The business generates adequate cash from operations to meet short-term obligations."
            if liquidity == "Strong" else (
                "Operating cash flow is positive but cash reserves may be insufficient to fully "
                "cover current liabilities. Close monitoring of receivables and payables is advisable."
                if liquidity == "Moderate" else
                "Negative operating cash flow indicates the business is consuming more cash than it "
                "generates. Immediate working capital review and cash flow planning is strongly recommended."
            )
        )
    )

    logger.info(
        "estimate_cash_flow: operating_cf=%.0f, net_cf=%.0f, liquidity=%s",
        operating_cf, net_cf, liquidity,
    )

    return {
        "net_profit":    net_profit,
        "depreciation":  depreciation,
        "finance_cost":  finance_cost,
        "wc_change":     wc_change,
        "operating_cf":  operating_cf,
        "investing_cf":  investing_cf,
        "financing_cf":  financing_cf,
        "net_cf":        net_cf,
        "cash_on_hand":  cash_on_hand,
        "liquidity":     liquidity,
        "liq_color":     {"Strong": "green", "Moderate": "amber", "Weak": "red"}.get(liquidity, "red"),
        "interpretation": interpretation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Comparative Analysis (Year-on-Year)
# ─────────────────────────────────────────────────────────────────────────────
def compute_comparative_analysis(
    df: pd.DataFrame,
    cat_col: str = "Category",
) -> Optional[Dict]:
    """
    Split the dataset by Indian fiscal year (Apr–Mar) and compute YoY metrics.

    Args:
        df:      Full transactions DataFrame with Date and Amount columns.
        cat_col: Category column to use.

    Returns:
        Dict with YoY revenue, expense, and profit figures + growth percentages.
        Returns None if Date or Amount columns are absent.
        Returns ``{"available": False, "note": str}`` if only one FY of data exists.
    """
    if "Date" not in df.columns or "Amount" not in df.columns:
        logger.warning("compute_comparative_analysis: Date or Amount column missing")
        return None

    df2 = df.copy()
    df2["Date"] = pd.to_datetime(df2["Date"], errors="coerce")
    df2 = df2.dropna(subset=["Date"])

    # Indian FY: April = start. Label = ending year e.g. "FY 2024-25"
    df2["FY"] = df2["Date"].apply(
        lambda d: f"FY {d.year}-{str(d.year+1)[-2:]}" if d.month >= 4
                  else f"FY {d.year-1}-{str(d.year)[-2:]}"
    )

    fy_list = sorted(df2["FY"].unique())
    if len(fy_list) < 2:
        return {
            "available": False,
            "note": (
                "Comparative analysis requires data spanning at least two financial years. "
                "The current dataset covers only one fiscal year — prior-year figures are unavailable."
            ),
            "fy_list": fy_list,
        }

    prev_fy = fy_list[-2]
    curr_fy = fy_list[-1]

    def _fy_total(fy_label: str, category: str) -> float:
        mask = (df2["FY"] == fy_label) & (df2[cat_col] == category)
        return float(df2.loc[mask, "Amount"].sum())

    def _growth(curr: float, prev: float) -> Optional[float]:
        return None if prev == 0 else round((curr - prev) / prev * 100, 2)

    curr_rev  = _fy_total(curr_fy, "Income")
    prev_rev  = _fy_total(prev_fy, "Income")
    curr_exp  = _fy_total(curr_fy, "Expense")
    prev_exp  = _fy_total(prev_fy, "Expense")
    curr_prof = curr_rev - curr_exp
    prev_prof = prev_rev - prev_exp

    rev_growth  = _growth(curr_rev,  prev_rev)
    exp_growth  = _growth(curr_exp,  prev_exp)
    prof_growth = _growth(curr_prof, prev_prof)

    trend = (
        "Improving" if prof_growth is not None and prof_growth > 5
        else ("Declining" if prof_growth is not None and prof_growth < -5 else "Stable")
    )

    logger.info(
        "compute_comparative_analysis: %s vs %s — rev_growth=%.1f%%, trend=%s",
        prev_fy, curr_fy, rev_growth or 0.0, trend,
    )

    return {
        "available":          True,
        "curr_fy":            curr_fy,
        "prev_fy":            prev_fy,
        "current_revenue":    curr_rev,
        "previous_revenue":   prev_rev,
        "revenue_growth_pct": rev_growth,
        "current_expense":    curr_exp,
        "previous_expense":   prev_exp,
        "expense_growth_pct": exp_growth,
        "current_profit":     curr_prof,
        "previous_profit":    prev_prof,
        "profit_growth_pct":  prof_growth,
        "trend":              trend,
    }
