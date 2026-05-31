"""
modules/compliance_engine.py — CA Intelligence Suite
Phase 6: Compliance Engine — Scoring & Insights

Provides:
    - generate_compliance_insights(summary, df)  → List[Dict] of CA insights
    - compute_compliance_score(insights)         → risk_score + compliance_score
    - validate_tax_provision(summary)            → tax adequacy check
    - validate_balance_sheet(summary)            → balance sheet equality check

Compliance Score Formula (Phase 6):
    risk_score       = critical_count × 5 + warning_count × 3
    compliance_score = max(0, 100 − risk_score)
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from config import (
    PROFIT_MARGIN_STRONG,
    PROFIT_MARGIN_MODERATE,
    GST_RATIO_HIGH,
    EXPENSE_RATIO_HIGH,
    LARGE_TXN_IQR_FACTOR,
    MIN_ROWS_FOR_ANOMALY,
    TOP_N_EXPENSE_CATS,
    TAX_RATE_LOW,
    TAX_RATE_HIGH,
    COMPLIANCE_CRITICAL_WEIGHT,
    COMPLIANCE_WARNING_WEIGHT,
)
from utils.logger import get_logger
from utils.helpers import inr_pdf as inr_format
from modules.financial_engine import classify_financial_health

logger = get_logger(__name__)


# 
# Compliance Scoring (Phase 6 requirement)
# 
def compute_compliance_score(insights: List[Dict]) -> Dict:
    """
    Compute a numerical compliance score from a list of insight dicts.

    Formula:
        risk_score       = (critical_count × 5) + (warning_count × 3)
        compliance_score = max(0, 100 − risk_score)

    Args:
        insights: List of insight dicts with 'level' key.

    Returns:
        Dict with keys:
            critical_count, warning_count, info_count,
            risk_score, compliance_score, grade (A/B/C/D/F)
    """
    critical = sum(1 for i in insights if i.get("level") == "critical")
    warning  = sum(1 for i in insights if i.get("level") == "warning")
    info     = sum(1 for i in insights if i.get("level") == "info")

    risk_score       = critical * COMPLIANCE_CRITICAL_WEIGHT + warning * COMPLIANCE_WARNING_WEIGHT
    compliance_score = max(0, 100 - risk_score)

    # Letter grade
    grade = (
        "A" if compliance_score >= 90 else
        "B" if compliance_score >= 75 else
        "C" if compliance_score >= 60 else
        "D" if compliance_score >= 40 else "F"
    )

    logger.info(
        "compute_compliance_score: score=%d/100 grade=%s (critical=%d, warning=%d)",
        compliance_score, grade, critical, warning,
    )

    return {
        "critical_count":  critical,
        "warning_count":   warning,
        "info_count":      info,
        "risk_score":      risk_score,
        "compliance_score": compliance_score,
        "grade":           grade,
    }


# 
# Tax Provision Validation
# 
def validate_tax_provision(summary) -> Dict:
    """
    Validate that the tax provision is within the 20–30% of Profit Before Tax range.

    Args:
        summary: FinancialSummary from the rule engine.

    Returns:
        Dict with keys: status ('ok'|'warning'|'n/a'), finding, recommendation,
        pbt, tax, effective_rate_pct.
    """
    pbt = summary.profit_before_tax
    tax = summary.short_term_provisions

    if pbt <= 0:
        return {
            "status":             "n/a",
            "finding":            "Tax validation not applicable — no taxable profit exists for the period.",
            "recommendation":     "No advance tax obligation arises when PBT is zero or negative.",
            "pbt": pbt, "tax": tax, "effective_rate_pct": 0.0,
        }

    if tax == 0:
        return {
            "status":  "warning",
            "finding": (
                f"No tax provision has been made against a Profit Before Tax of {inr_format(pbt)}. "
                "This appears inconsistent and requires review."
            ),
            "recommendation": (
                "A provision for income tax should be estimated at approximately 25% of PBT "
                "and recognised as a Short-term Provision in the Balance Sheet."
            ),
            "pbt": pbt, "tax": tax, "effective_rate_pct": 0.0,
        }

    effective_rate = tax / pbt
    eff_pct        = round(effective_rate * 100, 2)

    if effective_rate < TAX_RATE_LOW:
        status  = "warning"
        finding = (
            f"Tax provision of {inr_format(tax)} represents only {eff_pct}% of PBT "
            f"({inr_format(pbt)}), below the expected 20–30% range. Under-stated."
        )
        rec     = (
            "Re-examine the tax computation to include current tax, deferred tax, and applicable "
            "surcharge/cess. Ensure MAT provisions are evaluated where applicable."
        )
    elif eff_pct > 35:
        status  = "critical"
        finding = (
            f"Tax provision of {inr_format(tax)} represents {eff_pct}% of PBT "
            f"({inr_format(pbt)}). Tax provision unrealistic."
        )
        rec     = (
            "Tax provisioning exceeds the statutory maximum rate. Re-examine the tax computation "
            "immediately for mathematical errors or misclassified inputs."
        )
    elif effective_rate > TAX_RATE_HIGH:
        status  = "warning"
        finding = (
            f"Tax provision of {inr_format(tax)} represents {eff_pct}% of PBT "
            f"({inr_format(pbt)}), exceeding the expected 20–30% range. Over-stated."
        )
        rec     = (
            "Re-examine the tax computation for allowable deductions (Section 10, 80C–80U), "
            "brought-forward losses, and depreciation differences to avoid over-provisioning."
        )
    else:
        status  = "ok"
        finding = (
            f"Tax provision of {inr_format(tax)} ({eff_pct}% of PBT) is within the "
            "expected 20–30% range — provision appears reasonable."
        )
        rec     = "Continue monitoring quarterly advance tax payments per Section 208 timelines."

    return {
        "status": status, "finding": finding, "recommendation": rec,
        "pbt": pbt, "tax": tax, "effective_rate_pct": eff_pct,
    }


# 
# Balance Sheet Validation
# 
def validate_balance_sheet(summary) -> Dict:
    """
    Validate that Total Assets = Total Equity & Liabilities.

    Args:
        summary: FinancialSummary.

    Returns:
        Dict with keys: balanced (bool), assets, equity_liab, gap, note, adjustment_note.
    """
    assets     = round(summary.total_assets, 2)
    equity_lib = round(summary.total_equity_and_liabilities, 2)
    gap        = round(assets - equity_lib, 2)

    if abs(gap) < 1.0:
        return {
            "balanced":   True,
            "assets":     assets,
            "equity_liab": equity_lib,
            "gap":        gap,
            "note": (
                f"Balance Sheet balances. Total Assets ({inr_format(assets)}) = "
                f"Total Equity & Liabilities ({inr_format(equity_lib)})."
            ),
            "adjustment_note": (
                f"A residual balancing adjustment of {inr_format(summary.balance_sheet_adjustment)} "
                "has been posted to Reserves & Surplus."
            ) if abs(summary.balance_sheet_adjustment) > 1 else "",
        }
    else:
        return {
            "balanced":    False,
            "assets":      assets,
            "equity_liab": equity_lib,
            "gap":         gap,
            "note": (
                f"Balance Sheet mismatch detected. Assets ({inr_format(assets)}) ≠ "
                f"Equity + Liabilities ({inr_format(equity_lib)}). "
                f"Unreconciled difference: {inr_format(abs(gap))}."
            ),
            "adjustment_note": "",
        }


# 
# Insights Generator
# 
def generate_compliance_insights(summary, df: pd.DataFrame) -> List[Dict]:
    """
    Generate a comprehensive list of CA-grade financial and tax compliance insights.

    Covers:
        1. Profitability (net profit/loss, margin)
        2. Expense ratio
        3. Top expense sub-categories
        4. GST liability
        5. Large transaction outliers
        6. Section 40A(3) — cash expenses > ₹10,000
        7. Section 269ST — cash dealings ≥ ₹2 Lakh
        8. Section 44AB — tax audit threshold
        9. TDS obligations (194J, 194C)
        10. Advance tax (Section 208)
        11. Blocked ITC (Section 17(5) CGST)
        12. Asset-liability leverage

    Args:
        summary: FinancialSummary from the rule engine.
        df:      Raw transactions DataFrame.

    Returns:
        List of insight dicts with keys: level, icon, title, message.
    """
    insights: List[Dict] = []

    ti      = summary.total_income
    te      = summary.total_expense
    npl     = summary.net_profit_loss
    pm      = summary.profit_margin_pct
    gst_pay = summary.gst_payable
    health, _ = classify_financial_health(pm)

    #  1. Net Profit / Loss 
    if npl > 0:
        insights.append({
            "level": "info", "icon": "",
            "title": "Profitable Period",
            "message": (
                f"The business recorded a net profit of {inr_format(npl)} "
                f"(margin: {pm:.1f}%). Financial health is classified as '{health}'."
            ),
        })
    else:
        insights.append({
            "level": "critical", "icon": "",
            "title": "Net Loss Recorded",
            "message": (
                f"The business recorded a net loss of {inr_format(abs(npl))} for the period. "
                "An immediate review of cost structures, pricing strategy, and revenue diversification "
                "is strongly recommended to restore operational viability."
            ),
        })

    #  2. Profit Margin Classification 
    if pm >= PROFIT_MARGIN_STRONG:
        insights.append({
            "level": "info", "icon": "",
            "title": f"Strong Profit Margin ({pm:.1f}%)",
            "message": (
                f"Profit margin of {pm:.1f}% exceeds the 15% benchmark. "
                "Consider reinvesting surplus in expansion or long-term assets."
            ),
        })
    elif pm >= PROFIT_MARGIN_MODERATE:
        insights.append({
            "level": "warning", "icon": "",
            "title": f"Moderate Profit Margin ({pm:.1f}%)",
            "message": (
                f"Profit margin stands at {pm:.1f}%, below the 15% target. "
                "A focused cost-reduction strategy and review of discounting policies is advisable."
            ),
        })
    elif pm >= 0:
        insights.append({
            "level": "critical", "icon": "",
            "title": f"Weak Profit Margin ({pm:.1f}%)",
            "message": (
                f"Profit margin of only {pm:.1f}% is critically low. "
                "Immediate corrective action on pricing, discretionary expenditure, "
                "and operational efficiency is necessary."
            ),
        })

    #  3. Expense Ratio 
    if ti > 0:
        exp_ratio = te / ti
        if exp_ratio >= EXPENSE_RATIO_HIGH:
            insights.append({
                "level": "warning", "icon": "",
                "title": "High Expense-to-Income Ratio",
                "message": (
                    f"Total expenses constitute {exp_ratio*100:.1f}% of gross income, leaving only "
                    f"{(1-exp_ratio)*100:.1f}% as surplus. 'Other Expenses' should be reviewed."
                ),
            })

    #  4. Top Expense Sub-categories 
    cat_col = "Category" if "Category" in df.columns else "Predicted_Category"
    if cat_col in df.columns and "Sub_Category" in df.columns:
        exp_df = df[df[cat_col] == "Expense"]
        if not exp_df.empty and "Amount" in exp_df.columns:
            top = (
                exp_df.groupby("Sub_Category")["Amount"]
                .sum().sort_values(ascending=False).head(TOP_N_EXPENSE_CATS)
            )
            for sub_cat, amt in top.items():
                pct = (amt / te * 100) if te > 0 else 0
                insights.append({
                    "level": "info", "icon": "",
                    "title": f"Major Expenditure Head: {sub_cat}",
                    "message": (
                        f"'{sub_cat}' represents {inr_format(amt)} ({pct:.1f}% of total expenses). "
                        "Review whether this outflow is within approved budgetary limits."
                    ),
                })

    #  5. GST Liability 
    if ti > 0:
        gst_ratio = gst_pay / ti
        if gst_ratio > GST_RATIO_HIGH:
            insights.append({
                "level": "warning", "icon": "",
                "title": "Elevated GST Liability",
                "message": (
                    f"Net GST payable is {inr_format(gst_pay)} ({gst_ratio*100:.1f}% of gross income). "
                    "Ensure all eligible ITC from GSTR-2B is reconciled before filing GSTR-3B."
                ),
            })
        else:
            insights.append({
                "level": "info", "icon": "",
                "title": "GST Position",
                "message": (
                    f"Output GST: {inr_format(summary.gst_on_income)} | "
                    f"Input GST Credit: {inr_format(summary.gst_on_expense)} | "
                    f"Net Payable: {inr_format(gst_pay)}. GST liability is at a manageable level."
                ),
            })

    #  6. Large Transaction Outliers 
    if "Amount" in df.columns and len(df) > MIN_ROWS_FOR_ANOMALY:
        amounts   = df["Amount"].dropna()
        q1, q3   = amounts.quantile(0.25), amounts.quantile(0.75)
        iqr       = q3 - q1
        threshold = q3 + LARGE_TXN_IQR_FACTOR * iqr
        outliers  = df[df["Amount"] > threshold]
        if not outliers.empty:
            max_amt = outliers["Amount"].max()
            insights.append({
                "level": "warning", "icon": "",
                "title": f"Anomalous Large Transactions Detected ({len(outliers)})",
                "message": (
                    f"{len(outliers)} transaction(s) exceed the threshold of {inr_format(threshold)}. "
                    f"The largest is {inr_format(max_amt)}. Each requires individual vouching."
                ),
            })

    #  7a. Section 40A(3) — Cash Expenses > Rs.10,000 
    if "Payment_Mode" in df.columns and cat_col in df.columns:
        cash_exp = df[(df[cat_col] == "Expense") & (df["Payment_Mode"] == "Cash")]
        disallowed = cash_exp[cash_exp["Amount"] > 10000]
        if not disallowed.empty:
            insights.append({
                "level": "critical", "icon": "",
                "title": "[TAX COMPLIANCE] Section 40A(3) — Cash Expenditure Assessment",
                "message": (
                    f"Identified {len(disallowed)} expense transaction(s) settled in cash, each exceeding "
                    f"Rs.10,000. Aggregate at risk: {inr_format(disallowed['Amount'].sum())}. "
                    "May result in income-tax disallowance under Section 40A(3) ITA 1961."
                ),
            })

    #  7b. Section 269ST — Cash >= Rs.2 Lakh 
    if "Payment_Mode" in df.columns:
        cash_txns = df[df["Payment_Mode"] == "Cash"]
        sec269    = cash_txns[cash_txns["Amount"] >= 200000]
        if not sec269.empty:
            insights.append({
                "level": "critical", "icon": "",
                "title": "[TAX COMPLIANCE] Section 269ST — Cash Receipts/Payments Exceeding Rs.2 Lakh",
                "message": (
                    f"Detected {len(sec269)} transaction(s) of Rs.2,00,000 or more in cash. "
                    f"Total: {inr_format(sec269['Amount'].sum())}. "
                    "Penalty risk under Section 269ST — immediate policy review advised."
                ),
            })

    #  8. Section 44AB — Tax Audit Threshold 
    if ti > 1_00_00_000:
        insights.append({
            "level": "warning", "icon": "",
            "title": "[TAX COMPLIANCE] Tax Audit Threshold — Section 44AB",
            "message": (
                f"Gross receipts of {inr_format(ti)} exceed the Rs.1 Crore threshold. "
                "Statutory Tax Audit under Section 44AB is required. "
                "Ensure timely appointment of a CA and filing of Forms 3CA/3CB/3CD."
            ),
        })

    #  9. TDS Obligations 
    if "Sub_Category" in df.columns:
        import pandas as _pd
        prof_fees   = df[df["Sub_Category"].str.contains("Professional|Legal|Audit", case=False, na=False)]
        contractors = df[df["Sub_Category"].str.contains("Contract|Labour", case=False, na=False)]
        prof_total  = prof_fees["Amount"].sum()  if not prof_fees.empty  else 0
        contr_total = contractors["Amount"].sum() if not contractors.empty else 0

        tds_msgs = []
        if prof_total >= 30000:
            tds_msgs.append(f"Professional Fees: {inr_format(prof_total)} — TDS under Section 194J")
        if contr_total >= 100000:
            tds_msgs.append(f"Contractor Payments: {inr_format(contr_total)} — TDS under Section 194C")

        if tds_msgs:
            insights.append({
                "level": "warning", "icon": "",
                "title": "[TAX COMPLIANCE] TDS Obligation Check — Form 26Q",
                "message": (
                    "High-value expenditure requiring TDS verification:\n• "
                    + "\n• ".join(tds_msgs)
                    + "\nFailure to deduct TDS may result in disallowance under Section 40(a)(ia)."
                ),
            })

    #  10. Advance Tax 
    pbt = summary.profit_before_tax
    if pbt > 40000:
        est_tax = pbt * 0.25
        if est_tax >= 10000:
            insights.append({
                "level": "info", "icon": "",
                "title": "[TAX COMPLIANCE] Advance Tax Installments — Section 208",
                "message": (
                    f"Estimated income tax: {inr_format(est_tax)} (25% of PBT — indicative). "
                    "Pay advance tax in four tranches: 15% by 15-Jun, 45% by 15-Sep, "
                    "75% by 15-Dec, 100% by 15-Mar. Non-compliance attracts interest u/s 234B & 234C."
                ),
            })

    #  11. Blocked ITC 
    if "Sub_Category" in df.columns:
        blocked_cats = ["Staff Welfare Expenses", "Food & Beverages", "Vehicle Repair", "Health Insurance"]
        blocked_exp  = df[df["Sub_Category"].isin(blocked_cats)]
        if not blocked_exp.empty:
            blocked_amt = (
                blocked_exp["GST_Amount"].sum()
                if "GST_Amount" in blocked_exp.columns else 0
            )
            if blocked_amt > 0:
                insights.append({
                    "level": "critical", "icon": "",
                    "title": "[TAX COMPLIANCE] Restricted Input Tax Credit — Section 17(5) CGST Act",
                    "message": (
                        f"Detected {inr_format(blocked_amt)} in GST on restricted categories "
                        "(Staff Welfare, Food & Beverages, Vehicle Repair, Health Insurance). "
                        "ITC on these is blocked under Section 17(5) CGST Act — must be reversed in GSTR-3B."
                    ),
                })

    #  12. Asset-Liability Leverage 
    ta = summary.total_assets
    tl = summary.total_liabilities
    if tl > 0 and ta > 0:
        leverage = tl / ta
        if leverage > 2.0:
            insights.append({
                "level": "warning", "icon": "",
                "title": "High Financial Leverage",
                "message": (
                    f"Total liabilities ({inr_format(tl)}) are {leverage:.1f}× total assets ({inr_format(ta)}). "
                    "Excessive leverage increases insolvency risk. A debt-reduction plan is recommended."
                ),
            })
        else:
            insights.append({
                "level": "info", "icon": "",
                "title": "Balance Sheet Position",
                "message": (
                    f"Total Assets: {inr_format(ta)} | Total Liabilities: {inr_format(tl)} | "
                    f"Net Worth (implied): {inr_format(ta - tl)}. "
                    "The business maintains a healthy asset-to-liability ratio."
                ),
            })

    logger.info(
        "generate_compliance_insights: generated %d insights (critical=%d, warning=%d, info=%d)",
        len(insights),
        sum(1 for i in insights if i["level"] == "critical"),
        sum(1 for i in insights if i["level"] == "warning"),
        sum(1 for i in insights if i["level"] == "info"),
    )
    return insights
