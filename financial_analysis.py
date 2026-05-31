"""
Phase 5: Financial Analysis Engine — Professional CA Grade
Produces CA-grade textual insights, data validation, tax validation,
comparative analysis, cash flow estimation, and anomaly detection
from the FinancialSummary produced by the rule engine.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from rule_engine import FinancialSummary


# ─────────────────────────────────────────────────────────────────────────────
# Thresholds (configurable)
# ─────────────────────────────────────────────────────────────────────────────
PROFIT_MARGIN_STRONG   = 15.0   # % — Strong if ≥ this
PROFIT_MARGIN_MODERATE =  5.0   # % — Moderate if ≥ this, < STRONG
PROFIT_MARGIN_WEAK     =  0.0   # % — Weak if ≥ 0 but < MODERATE
TAX_RATE_LOW           = 0.20   # 20% of PBT
TAX_RATE_HIGH          = 0.30   # 30% of PBT
GST_RATIO_HIGH         = 0.15   # GST > 15% of total income → high
EXPENSE_RATIO_HIGH     = 0.80   # Expense > 80% of income → warning
LARGE_TXN_IQR_FACTOR   = 3.0   # Outlier if > Q3 + 3*IQR
TOP_N_EXPENSE_CATS     = 3      # Top N sub-categories to flag


# ─────────────────────────────────────────────────────────────────────────────
# Indian Number Format Helper
# ─────────────────────────────────────────────────────────────────────────────
def inr_format(value: float, symbol: bool = True) -> str:
    """
    Format a number in Indian lakh/crore notation.
    e.g. 63385218.98 → ₹6,33,85,218.98
    """
    prefix = "₹" if symbol else ""
    try:
        value = float(value)
        is_neg = value < 0
        value = abs(value)
        # Split integer and decimal
        int_part = int(value)
        dec_part = round(value - int_part, 2)
        dec_str = f"{dec_part:.2f}"[1:]   # ".XX"

        # Indian grouping: last 3 digits, then groups of 2
        s = str(int_part)
        if len(s) <= 3:
            formatted = s
        else:
            # last 3
            formatted = s[-3:]
            s = s[:-3]
            while s:
                formatted = s[-2:] + "," + formatted
                s = s[:-2]
        result = f"{prefix}{'-' if is_neg else ''}{formatted}{dec_str}"
        return result
    except Exception:
        return f"{prefix}{value}"


def classify_financial_health(pm: float) -> Tuple[str, str]:
    """
    Returns (classification, color_hint) based on profit margin %.
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
# Phase 1: Data Validation
# ─────────────────────────────────────────────────────────────────────────────
def validate_data_quality(df: pd.DataFrame) -> List[Dict]:
    """
    Examines the raw dataframe for data quality issues.
    Returns a list of issue dicts:
      { "type": str, "severity": "critical|warning|info",
        "description": str, "impact": str, "count": int }
    """
    issues: List[Dict] = []

    # 1. Missing values
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if not null_cols.empty:
        total_nulls = int(null_cols.sum())
        col_list = ", ".join([f"{c} ({n})" for c, n in null_cols.items()])
        issues.append({
            "type": "Missing Values",
            "severity": "warning",
            "description": f"{total_nulls} missing value(s) detected across column(s): {col_list}.",
            "impact": (
                "Missing values in 'Amount' or 'Category' columns may cause incorrect financial totals. "
                "Missing 'Date' entries will be excluded from trend analysis."
            ),
            "count": total_nulls,
        })

    # 2. Duplicate rows
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        issues.append({
            "type": "Duplicate Entries",
            "severity": "critical",
            "description": f"{dup_count} duplicate transaction row(s) detected.",
            "impact": (
                "Duplicate entries will cause double-counting of income and expense figures, "
                "leading to overstated revenue, inflated tax liability, and incorrect profit margin."
            ),
            "count": dup_count,
        })

    # 3. Negative or zero amounts in unexpected categories
    if "Amount" in df.columns and "Category" in df.columns:
        neg_amounts = df[df["Amount"] < 0]
        if not neg_amounts.empty:
            neg_cats = neg_amounts["Category"].value_counts().to_dict()
            issues.append({
                "type": "Negative Transaction Amounts",
                "severity": "warning",
                "description": (
                    f"{len(neg_amounts)} transaction(s) carry negative amounts. "
                    f"Breakdown by category: {neg_cats}."
                ),
                "impact": (
                    "Negative income entries reduce reported revenue. Negative expense entries "
                    "may represent credit notes or refunds, and should be explicitly tagged."
                ),
                "count": len(neg_amounts),
            })

        zero_amounts = df[df["Amount"] == 0]
        if not zero_amounts.empty:
            issues.append({
                "type": "Zero-Value Transactions",
                "severity": "info",
                "description": f"{len(zero_amounts)} transaction(s) have an amount of ₹0.",
                "impact": "Zero-value entries are generally harmless but inflate transaction counts and may indicate incomplete data entry.",
                "count": len(zero_amounts),
            })

    # 4. Statistical outliers (IQR method)
    if "Amount" in df.columns:
        amounts = df["Amount"].dropna()
        if len(amounts) > 10:
            q1, q3 = amounts.quantile(0.25), amounts.quantile(0.75)
            iqr = q3 - q1
            threshold = q3 + LARGE_TXN_IQR_FACTOR * iqr
            outliers = df[df["Amount"] > threshold]
            if not outliers.empty:
                max_val = outliers["Amount"].max()
                issues.append({
                    "type": "Statistical Outliers (Large Transactions)",
                    "severity": "warning",
                    "description": (
                        f"{len(outliers)} transaction(s) exceed the outlier threshold of "
                        f"{inr_format(threshold)} (Q3 + 3×IQR). Largest: {inr_format(max_val)}."
                    ),
                    "impact": (
                        "Outlier transactions may represent legitimate bulk purchases, capital expenditure, "
                        "or erroneous data entries. Each requires individual verification."
                    ),
                    "count": len(outliers),
                })

    # 5. Unknown / unclassified categories
    if "Category" in df.columns:
        valid_cats = {"Expense", "Income", "Asset", "Liability"}
        unknown = df[~df["Category"].isin(valid_cats)]
        if not unknown.empty:
            un_vals = unknown["Category"].value_counts().to_dict()
            issues.append({
                "type": "Unrecognised Category Values",
                "severity": "critical",
                "description": f"{len(unknown)} row(s) have unrecognised category values: {un_vals}.",
                "impact": (
                    "Transactions with unrecognised categories are excluded from all financial computations. "
                    "This directly understates reported revenue, expenses, assets, or liabilities."
                ),
                "count": len(unknown),
            })

    if not issues:
        issues.append({
            "type": "No Issues Detected",
            "severity": "info",
            "description": "Dataset passed all integrity checks — no missing values, duplicates, or abnormal entries detected.",
            "impact": "Data quality is satisfactory for financial analysis.",
            "count": 0,
        })

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Financial Validation
# ─────────────────────────────────────────────────────────────────────────────
def validate_tax_provision(summary: FinancialSummary) -> Dict:
    """
    Validate that tax provision is within 20–30% of Profit Before Tax.
    Returns a dict with status, finding, and recommendation.
    """
    pbt = summary.profit_before_tax
    tax = summary.short_term_provisions

    if pbt <= 0:
        return {
            "status": "n/a",
            "finding": "Tax validation not applicable — no taxable profit exists for the period.",
            "recommendation": "No advance tax obligation arises when PBT is zero or negative.",
            "pbt": pbt, "tax": tax, "effective_rate_pct": 0.0,
        }

    if tax == 0:
        return {
            "status": "warning",
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
    eff_pct = round(effective_rate * 100, 2)

    if effective_rate < TAX_RATE_LOW:
        status = "warning"
        finding = (
            f"Tax provision of {inr_format(tax)} represents only {eff_pct}% of PBT "
            f"({inr_format(pbt)}), which is below the expected 20–30% range. "
            "Tax provision appears under-stated and requires review."
        )
        rec = (
            "Re-examine the tax computation to include current tax, deferred tax, and applicable surcharge/cess. "
            "Ensure MAT provisions are evaluated where applicable."
        )
    elif effective_rate > TAX_RATE_HIGH:
        status = "warning"
        finding = (
            f"Tax provision of {inr_format(tax)} represents {eff_pct}% of PBT "
            f"({inr_format(pbt)}), which exceeds the expected 20–30% range. "
            "Tax provision appears over-stated and requires review."
        )
        rec = (
            "Re-examine the tax computation for allowable deductions (Section 10, 80C–80U), "
            "brought-forward losses, and depreciation differences to avoid over-provisioning."
        )
    else:
        status = "ok"
        finding = (
            f"Tax provision of {inr_format(tax)} ({eff_pct}% of PBT) is within the "
            "expected 20–30% range — provision appears reasonable."
        )
        rec = "Continue monitoring quarterly advance tax payments per Section 208 timelines."

    return {
        "status": status,
        "finding": finding,
        "recommendation": rec,
        "pbt": pbt, "tax": tax, "effective_rate_pct": eff_pct,
    }


def validate_balance_sheet(summary: FinancialSummary) -> Dict:
    """
    Validate that Total Assets = Total Equity & Liabilities.
    Returns { "balanced": bool, "assets": float, "equity_liab": float, "gap": float, "note": str }
    """
    assets     = round(summary.total_assets, 2)
    equity_lib = round(summary.total_equity_and_liabilities, 2)
    gap        = round(assets - equity_lib, 2)

    if abs(gap) < 1.0:   # Within ₹1 rounding tolerance
        return {
            "balanced": True,
            "assets": assets,
            "equity_liab": equity_lib,
            "gap": gap,
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
            "balanced": False,
            "assets": assets,
            "equity_liab": equity_lib,
            "gap": gap,
            "note": (
                f"Balance Sheet mismatch detected. Assets ({inr_format(assets)}) ≠ "
                f"Equity + Liabilities ({inr_format(equity_lib)}). "
                f"Unreconciled difference: {inr_format(abs(gap))}."
            ),
            "adjustment_note": "",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Comparative Analysis (Year-on-Year)
# ─────────────────────────────────────────────────────────────────────────────
def compute_comparative_analysis(df: pd.DataFrame,
                                  cat_col: str = "Category") -> Optional[Dict]:
    """
    Splits the dataset by fiscal year (April–March) and computes YoY metrics.
    Returns None if insufficient multi-year data exists.
    """
    if "Date" not in df.columns or "Amount" not in df.columns:
        return None

    df2 = df.copy()
    df2["Date"] = pd.to_datetime(df2["Date"], errors="coerce")
    df2 = df2.dropna(subset=["Date"])

    # Compute Indian FY: April = start. Year label = ending year.
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

    # Take the last two FYs
    prev_fy = fy_list[-2]
    curr_fy = fy_list[-1]

    def fy_total(fy_label, category):
        mask = (df2["FY"] == fy_label) & (df2[cat_col] == category)
        return float(df2.loc[mask, "Amount"].sum())

    def growth(curr, prev):
        if prev == 0:
            return None
        return round((curr - prev) / prev * 100, 2)

    curr_rev  = fy_total(curr_fy, "Income")
    prev_rev  = fy_total(prev_fy, "Income")
    curr_exp  = fy_total(curr_fy, "Expense")
    prev_exp  = fy_total(prev_fy, "Expense")
    curr_prof = curr_rev - curr_exp
    prev_prof = prev_rev - prev_exp

    rev_growth  = growth(curr_rev, prev_rev)
    exp_growth  = growth(curr_exp, prev_exp)
    prof_growth = growth(curr_prof, prev_prof)

    return {
        "available": True,
        "curr_fy": curr_fy,
        "prev_fy": prev_fy,
        "current_revenue":  curr_rev,
        "previous_revenue": prev_rev,
        "revenue_growth_pct": rev_growth,
        "current_expense":  curr_exp,
        "previous_expense": prev_exp,
        "expense_growth_pct": exp_growth,
        "current_profit":   curr_prof,
        "previous_profit":  prev_prof,
        "profit_growth_pct": prof_growth,
        "trend": (
            "Improving" if prof_growth is not None and prof_growth > 5
            else ("Declining" if prof_growth is not None and prof_growth < -5
                  else "Stable")
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Cash Flow Insight
# ─────────────────────────────────────────────────────────────────────────────
def estimate_cash_flow(summary: FinancialSummary, df: pd.DataFrame) -> Dict:
    """
    Estimates operating cash flow and net liquidity position using indirect method proxy.

    Operating CF (Indirect Method Proxy):
      = Net Profit + Depreciation & Amortisation + Finance Costs
        - Change in Working Capital (proxied by Trade Receivables + Inventories - Trade Payables)
    """
    net_profit   = summary.net_profit_loss
    depreciation = summary.depreciation_amortisation
    finance_cost = summary.finance_costs

    # Working capital change proxy
    wc_uses   = summary.trade_receivables + summary.inventories
    wc_source = summary.trade_payables
    wc_change = wc_uses - wc_source  # positive = cash absorbed

    operating_cf = net_profit + depreciation + finance_cost - wc_change

    # Investing CF proxy (capital expenditure in assets)
    investing_cf = -(summary.tangible_assets + summary.intangible_assets + summary.cwip)

    # Financing CF proxy (borrowings net)
    financing_cf = summary.long_term_borrowings + summary.short_term_borrowings - summary.share_capital

    net_cf = operating_cf + investing_cf + financing_cf
    cash_on_hand = summary.cash_equivalents

    # Liquidity classification
    if cash_on_hand > summary.current_liabilities * 0.5 and operating_cf > 0:
        liquidity = "Strong"
        liq_color = "green"
    elif operating_cf > 0:
        liquidity = "Moderate"
        liq_color = "amber"
    else:
        liquidity = "Weak"
        liq_color = "red"

    return {
        "net_profit":        net_profit,
        "depreciation":      depreciation,
        "finance_cost":      finance_cost,
        "wc_change":         wc_change,
        "operating_cf":      operating_cf,
        "investing_cf":      investing_cf,
        "financing_cf":      financing_cf,
        "net_cf":            net_cf,
        "cash_on_hand":      cash_on_hand,
        "liquidity":         liquidity,
        "liq_color":         liq_color,
        "interpretation": (
            f"Liquidity position is {liquidity.upper()}. "
            f"Estimated operating cash flow is {inr_format(operating_cf)}, "
            f"with {inr_format(cash_on_hand)} in cash and cash equivalents. "
            + (
                "The business generates adequate cash from operations to meet short-term obligations."
                if liquidity == "Strong"
                else (
                    "Operating cash flow is positive but cash reserves may be insufficient to fully cover current liabilities. "
                    "Close monitoring of receivables and payables cycle is advisable."
                    if liquidity == "Moderate"
                    else
                    "Negative operating cash flow indicates the business is consuming more cash than it generates. "
                    "Immediate working capital review and cash flow planning is strongly recommended."
                )
            )
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Insights Generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_insights(summary: FinancialSummary,
                       df: pd.DataFrame) -> List[Dict]:
    """
    Returns a list of insight dicts:
      { "level": "info|warning|critical", "icon": str,
        "title": str, "message": str }
    """
    insights: List[Dict] = []

    ti  = summary.total_income
    te  = summary.total_expense
    npl = summary.net_profit_loss
    pm  = summary.profit_margin_pct
    gst_pay = summary.gst_payable
    health, _ = classify_financial_health(pm)

    # ── 1. Net Profit / Loss ──────────────────────────────────────────────────
    if npl > 0:
        insights.append({
            "level": "info",
            "icon": "✅",
            "title": "Profitable Period",
            "message": (
                f"The business recorded a net profit of {inr_format(npl)} "
                f"(margin: {pm:.1f}%). Financial health is classified as "
                f"'{health}' — a positive indicator."
            ),
        })
    else:
        insights.append({
            "level": "critical",
            "icon": "🚨",
            "title": "Net Loss Recorded",
            "message": (
                f"The business recorded a net loss of {inr_format(abs(npl))} for the period. "
                "An immediate review of cost structures, pricing strategy, and revenue diversification "
                "is strongly recommended to restore operational viability."
            ),
        })

    # ── 2. Profit Margin Classification ──────────────────────────────────────
    if pm >= PROFIT_MARGIN_STRONG:
        insights.append({
            "level": "info",
            "icon": "📈",
            "title": f"Strong Profit Margin ({pm:.1f}%)",
            "message": (
                f"Profit margin of {pm:.1f}% exceeds the 15% benchmark, indicating strong operational "
                "efficiency. The business may consider reinvesting surplus in expansion or long-term assets."
            ),
        })
    elif pm >= PROFIT_MARGIN_MODERATE:
        insights.append({
            "level": "warning",
            "icon": "🔶",
            "title": f"Moderate Profit Margin ({pm:.1f}%)",
            "message": (
                f"Profit margin stands at {pm:.1f}%, which is moderate but below the 15% target. "
                "A focused cost-reduction strategy and review of discounting policies is advisable."
            ),
        })
    elif pm >= 0:
        insights.append({
            "level": "critical",
            "icon": "⚠️",
            "title": f"Weak Profit Margin ({pm:.1f}%)",
            "message": (
                f"Profit margin of only {pm:.1f}% is critically low (threshold: 5%). "
                "The business is barely covering costs. Immediate corrective action on pricing, "
                "discretionary expenditure, and operational efficiency is necessary."
            ),
        })

    # ── 3. Expense Ratio ──────────────────────────────────────────────────────
    if ti > 0:
        exp_ratio = te / ti
        if exp_ratio >= EXPENSE_RATIO_HIGH:
            insights.append({
                "level": "warning",
                "icon": "📊",
                "title": "High Expense-to-Income Ratio",
                "message": (
                    f"Total expenses constitute {exp_ratio*100:.1f}% of gross income, leaving only "
                    f"{(1-exp_ratio)*100:.1f}% as surplus. Discretionary expenditure, particularly under "
                    "'Other Expenses', should be reviewed for reduction opportunities."
                ),
            })

    # ── 4. Top Expense Sub-categories ─────────────────────────────────────────
    cat_col_det = "Category"
    if "Category" in df.columns and "Sub_Category" in df.columns:
        exp_df = df[df[cat_col_det] == "Expense"]
        if not exp_df.empty and "Amount" in exp_df.columns:
            top_cats = (
                exp_df.groupby("Sub_Category")["Amount"]
                .sum()
                .sort_values(ascending=False)
                .head(TOP_N_EXPENSE_CATS)
            )
            for sub_cat, amt in top_cats.items():
                pct = (amt / te * 100) if te > 0 else 0
                insights.append({
                    "level": "info",
                    "icon": "💡",
                    "title": f"Major Expenditure Head: {sub_cat}",
                    "message": (
                        f"'{sub_cat}' represents {inr_format(amt)} ({pct:.1f}% of total expenses). "
                        "Review whether this outflow is within approved budgetary limits and aligned with business objectives."
                    ),
                })

    # ── 5. GST Liability ──────────────────────────────────────────────────────
    if ti > 0:
        gst_ratio = gst_pay / ti
        if gst_ratio > GST_RATIO_HIGH:
            insights.append({
                "level": "warning",
                "icon": "🏛️",
                "title": "Elevated GST Liability",
                "message": (
                    f"Net GST payable is {inr_format(gst_pay)} ({gst_ratio*100:.1f}% of gross income). "
                    "Ensure all eligible Input Tax Credits (ITC) from GSTR-2B are fully reconciled "
                    "before filing GSTR-3B to mitigate excess outflow."
                ),
            })
        else:
            insights.append({
                "level": "info",
                "icon": "🏛️",
                "title": "GST Position",
                "message": (
                    f"Output GST (collected): {inr_format(summary.gst_on_income)} | "
                    f"Input GST (credit): {inr_format(summary.gst_on_expense)} | "
                    f"Net payable: {inr_format(gst_pay)}. "
                    "GST liability is at a manageable level."
                ),
            })

    # ── 6. Large Transaction Outliers ─────────────────────────────────────────
    if "Amount" in df.columns and len(df) > 10:
        amounts = df["Amount"].dropna()
        q1, q3 = amounts.quantile(0.25), amounts.quantile(0.75)
        iqr = q3 - q1
        threshold = q3 + LARGE_TXN_IQR_FACTOR * iqr
        outliers = df[df["Amount"] > threshold]
        if not outliers.empty:
            max_amt = outliers["Amount"].max()
            insights.append({
                "level": "warning",
                "icon": "🔍",
                "title": f"Anomalous Large Transactions Detected ({len(outliers)})",
                "message": (
                    f"{len(outliers)} transaction(s) exceed the statistical outlier threshold of "
                    f"{inr_format(threshold)} (Q3 + 3×IQR). "
                    f"The largest transaction is {inr_format(max_amt)}. "
                    "Each of these entries should be individually vouched and supported by appropriate documentation."
                ),
            })

    # ── 7a. Section 40A(3) IT Act ─────────────────────────────────────────────
    if "Payment_Mode" in df.columns and "Category" in df.columns:
        cash_exp = df[(df["Category"] == "Expense") & (df["Payment_Mode"] == "Cash")]
        disallowed_txns = cash_exp[cash_exp["Amount"] > 10000]
        if not disallowed_txns.empty:
            dis_total = disallowed_txns["Amount"].sum()
            insights.append({
                "level": "critical",
                "icon": "🛑",
                "title": "[TAX COMPLIANCE] Section 40A(3) — Cash Expenditure Assessment",
                "message": (
                    f"Identified {len(disallowed_txns)} expense transaction(s) settled in cash, each exceeding "
                    f"₹10,000 per day. Aggregate amount at risk: {inr_format(dis_total)}. "
                    "Such payments may result in disallowance while computing taxable business income under Section 40A(3) "
                    "of the Income-tax Act, 1961. A review of internal cash payment control procedures is strongly recommended."
                ),
            })

        # ── 7b. Section 269ST ──────────────────────────────────────────────────
        cash_txns = df[df["Payment_Mode"] == "Cash"]
        sec269_violations = cash_txns[cash_txns["Amount"] >= 200000]
        if not sec269_violations.empty:
            sec_total = sec269_violations["Amount"].sum()
            insights.append({
                "level": "critical",
                "icon": "💵",
                "title": "[TAX COMPLIANCE] Section 269ST — Cash Receipts/Payments Exceeding ₹2 Lakh",
                "message": (
                    f"Detected {len(sec269_violations)} transaction(s) of ₹2,00,000 or more in cash. "
                    f"Total identified: {inr_format(sec_total)}. "
                    "Aggregate cash dealings of ₹2 lakh or more per person/event/occasion may attract "
                    "penalty under Section 269ST of the Income-tax Act. Immediate policy review is advised."
                ),
            })

    # ── 8. Section 44AB Tax Audit Limit ──────────────────────────────────────
    if ti > 1_00_00_000:
        insights.append({
            "level": "warning",
            "icon": "📑",
            "title": "[TAX COMPLIANCE] Tax Audit Threshold — Section 44AB",
            "message": (
                f"Gross receipts of {inr_format(ti)} exceed the primary ₹1 Crore threshold. "
                "Unless aggregate cash transactions are restricted to 5% of total turnover "
                "(qualifying for the ₹10 Crore limit), a statutory Tax Audit under Section 44AB "
                "is required. Ensure timely appointment of a Chartered Accountant and submission of Forms 3CA/3CB and 3CD."
            ),
        })

    # ── 9. TDS Obligations ────────────────────────────────────────────────────
    if "Sub_Category" in df.columns:
        prof_fees = df[df["Sub_Category"].str.contains("Professional|Legal|Audit", case=False, na=False)]
        contractors = df[df["Sub_Category"].str.contains("Contract|Labour", case=False, na=False)]

        prof_total  = prof_fees["Amount"].sum()  if not prof_fees.empty  else 0
        contr_total = contractors["Amount"].sum() if not contractors.empty else 0

        tds_msgs = []
        if prof_total >= 30000:
            tds_msgs.append(f"Professional Fees: {inr_format(prof_total)} — subject to TDS under Section 194J")
        if contr_total >= 100000:
            tds_msgs.append(f"Contractor Payments: {inr_format(contr_total)} — subject to TDS under Section 194C")

        if tds_msgs:
            insights.append({
                "level": "warning",
                "icon": "✂️",
                "title": "[TAX COMPLIANCE] TDS Obligation Check — Form 26Q",
                "message": (
                    "High-value expenditure requiring TDS verification has been identified:\n• "
                    + "\n• ".join(tds_msgs)
                    + "\nFailure to deduct and remit TDS within statutory timelines may result in "
                    "expense disallowance under Section 40(a)(ia) and attract penal interest under Sections 201 and 234E."
                ),
            })

    # ── 10. Advance Tax (Section 208) ─────────────────────────────────────────
    pbt = summary.profit_before_tax
    if pbt > 40000:
        est_tax = pbt * 0.25
        if est_tax >= 10000:
            insights.append({
                "level": "info",
                "icon": "🗓️",
                "title": "[TAX COMPLIANCE] Advance Tax Installments — Section 208",
                "message": (
                    f"Estimated income tax liability approximately {inr_format(est_tax)} (25% of PBT — indicative). "
                    "Statutory advance tax must be paid in four tranches: "
                    "15% by 15-Jun, 45% by 15-Sep, 75% by 15-Dec, and 100% by 15-Mar. "
                    "Non-compliance may attract interest under Sections 234B and 234C."
                ),
            })

    # ── 11. Blocked ITC (Section 17(5) CGST) ─────────────────────────────────
    if "Sub_Category" in df.columns:
        blocked_cats = ["Staff Welfare Expenses", "Food & Beverages", "Vehicle Repair", "Health Insurance"]
        blocked_exp = df[df["Sub_Category"].isin(blocked_cats)]
        if not blocked_exp.empty:
            blocked_amt = blocked_exp["GST_Amount"].sum() if "GST_Amount" in blocked_exp.columns else 0
            if blocked_amt > 0:
                insights.append({
                    "level": "critical",
                    "icon": "🚫",
                    "title": "[TAX COMPLIANCE] Restricted Input Tax Credit — Section 17(5) CGST Act",
                    "message": (
                        f"Detected {inr_format(blocked_amt)} in GST on restricted expenditure categories "
                        "(Staff Welfare, Food & Beverages, Vehicle Repair, Health Insurance). "
                        "Under Section 17(5) of the CGST Act, Input Tax Credit on these categories is blocked. "
                        "This amount must be reversed or excluded from Form GSTR-3B filing to avoid incorrect credit claims."
                    ),
                })

    # ── 12. Asset vs Liability Leverage ──────────────────────────────────────
    ta = summary.total_assets
    tl = summary.total_liabilities
    if tl > 0 and ta > 0:
        leverage = tl / ta
        if leverage > 2.0:
            insights.append({
                "level": "warning",
                "icon": "⚖️",
                "title": "High Financial Leverage",
                "message": (
                    f"Total liabilities ({inr_format(tl)}) are {leverage:.1f}× total assets ({inr_format(ta)}). "
                    "Excessive leverage increases insolvency risk and may constrain future borrowing capacity. "
                    "A structured debt-reduction plan is recommended."
                ),
            })
        else:
            insights.append({
                "level": "info",
                "icon": "⚖️",
                "title": "Balance Sheet Position",
                "message": (
                    f"Total Assets: {inr_format(ta)} | "
                    f"Total Liabilities: {inr_format(tl)} | "
                    f"Net Worth (implied): {inr_format(ta - tl)}. "
                    "The business maintains a healthy asset-to-liability ratio."
                ),
            })

    return insights


# ─────────────────────────────────────────────────────────────────────────────
# Console Print Utility
# ─────────────────────────────────────────────────────────────────────────────
def print_insights(insights: List[Dict]):
    print("\n[Phase 5] CA Financial Insights:")
    print("=" * 70)
    for ins in insights:
        print(f"\n  {ins['icon']}  [{ins['level'].upper()}] {ins['title']}")
        print(f"     {ins['message']}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from rule_engine import run_rule_engine

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "financial_dataset_clean.csv"
    df = pd.read_csv(csv_path)

    print("\n[Phase 1] Data Quality Validation:")
    issues = validate_data_quality(df)
    for issue in issues:
        print(f"  [{issue['severity'].upper()}] {issue['type']}: {issue['description']}")

    summary = run_rule_engine(df)
    pm = summary.profit_margin_pct
    health, _ = classify_financial_health(pm)
    print(f"\nProfit Margin: {pm:.1f}% — Financial Health: {health}")

    print("\n[Phase 2] Tax Validation:")
    tv = validate_tax_provision(summary)
    print(f"  Status: {tv['status']} | {tv['finding']}")

    print("\n[Phase 2] Balance Sheet Validation:")
    bv = validate_balance_sheet(summary)
    print(f"  Balanced: {bv['balanced']} | {bv['note']}")

    print("\n[Phase 3] Comparative Analysis:")
    ca = compute_comparative_analysis(df)
    if ca and ca.get("available"):
        print(f"  Revenue Growth: {ca['revenue_growth_pct']}% | Profit Trend: {ca['trend']}")

    print("\n[Phase 3] Cash Flow Estimate:")
    cf = estimate_cash_flow(summary, df)
    print(f"  Operating CF: {inr_format(cf['operating_cf'])} | Liquidity: {cf['liquidity']}")

    insights = generate_insights(summary, df)
    print_insights(insights)
