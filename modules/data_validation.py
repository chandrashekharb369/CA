"""
modules/data_validation.py — CA Intelligence Suite
Phase 2: Data Validation & Cleaning

Provides:
    - validate_data_quality(df)  → structured issue list for display
    - clean_data(df)             → cleaned DataFrame (via utils.helpers)
    - data_quality_score(df)     → integer 0–100 quality score

All functions are pure and stateless — safe to call repeatedly.
"""

from __future__ import annotations

import pandas as pd
from typing import List, Dict

from config import (
    VALID_CATEGORIES,
    LARGE_TXN_IQR_FACTOR,
    MIN_ROWS_FOR_ANOMALY,
)
from utils.logger import get_logger
from utils.helpers import clean_data, data_quality_score, inr_format

logger = get_logger(__name__)

# Re-export helpers so callers can import everything from this module
__all__ = ["validate_data_quality", "clean_data", "data_quality_score"]


# 
# Data Quality Validation
# 
def validate_data_quality(df: pd.DataFrame) -> List[Dict]:
    """
    Examine a raw financial DataFrame for data quality issues.

    Checks performed:
        1. Missing values across all columns
        2. Duplicate transaction rows
        3. Negative / zero Amount values
        4. Statistical outliers (IQR method)
        5. Unrecognised Category values

    Args:
        df: Raw input DataFrame as uploaded by the user.

    Returns:
        List of issue dicts, each with keys:
            ``type``        — Human-readable issue name
            ``severity``    — "critical" | "warning" | "info"
            ``description`` — Detailed description string
            ``impact``      — Business / audit impact statement
            ``count``       — Number of affected rows / cells
    """
    issues: List[Dict] = []
    logger.info("validate_data_quality: scanning %d rows × %d cols", len(df), len(df.columns))

    #  1. Missing values 
    total_cells = len(df) * len(df.columns)
    total_nulls_all = int(df.isnull().sum().sum())
    completeness = 100.0 - ((total_nulls_all / total_cells) * 100) if total_cells > 0 else 100.0

    null_counts = df.isnull().sum()
    null_cols   = null_counts[null_counts > 0]
    if not null_cols.empty:
        total_nulls = int(null_cols.sum())
        col_list    = ", ".join([f"{c} ({n})" for c, n in null_cols.items()])
        issues.append({
            "type":        "Missing Values",
            "severity":    "warning",
            "description": f"Data completeness: {completeness:.1f}%. {total_nulls} missing value(s) detected across column(s): {col_list}.",
            "impact": (
                "Missing values in 'Amount' or 'Category' columns may cause incorrect financial totals. "
                "Missing 'Date' entries will be excluded from trend analysis."
            ),
            "count": total_nulls,
        })
        logger.warning("validate_data_quality: %d missing values", total_nulls)

    #  2. Duplicate rows 
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        issues.append({
            "type":        "Duplicate Entries",
            "severity":    "critical",
            "description": f"{dup_count} duplicate transaction row(s) detected.",
            "impact": (
                "Duplicate entries will cause double-counting of income and expense figures, "
                "leading to overstated revenue, inflated tax liability, and incorrect profit margin."
            ),
            "count": dup_count,
        })
        logger.warning("validate_data_quality: %d duplicate rows", dup_count)

    #  3. Negative / zero amounts 
    if "Amount" in df.columns and "Category" in df.columns:
        neg_amounts = df[df["Amount"] < 0]
        if not neg_amounts.empty:
            neg_cats = neg_amounts["Category"].value_counts().to_dict()
            issues.append({
                "type":        "Negative Transaction Amounts",
                "severity":    "warning",
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
                "type":        "Zero-Value Transactions",
                "severity":    "info",
                "description": f"{len(zero_amounts)} transaction(s) have an amount of Rs.0.",
                "impact": "Zero-value entries inflate transaction counts and may indicate incomplete data entry.",
                "count": len(zero_amounts),
            })

    #  4. Statistical outliers (IQR method) 
    if "Amount" in df.columns:
        amounts = df["Amount"].dropna()
        if len(amounts) > MIN_ROWS_FOR_ANOMALY:
            q1, q3  = amounts.quantile(0.25), amounts.quantile(0.75)
            iqr     = q3 - q1
            threshold = q3 + LARGE_TXN_IQR_FACTOR * iqr
            outliers  = df[df["Amount"] > threshold]
            if not outliers.empty:
                max_val = outliers["Amount"].max()
                issues.append({
                    "type":        "Statistical Outliers (Large Transactions)",
                    "severity":    "warning",
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

    #  5. Unrecognised categories 
    if "Category" in df.columns:
        unknown = df[~df["Category"].isin(VALID_CATEGORIES)]
        if not unknown.empty:
            un_vals = unknown["Category"].value_counts().to_dict()
            issues.append({
                "type":        "Unrecognised Category Values",
                "severity":    "critical",
                "description": f"{len(unknown)} row(s) have unrecognised category values: {un_vals}.",
                "impact": (
                    "Transactions with unrecognised categories are excluded from all financial computations. "
                    "This directly understates reported revenue, expenses, assets, or liabilities."
                ),
                "count": len(unknown),
            })

    #  No issues 
    if not issues:
        issues.append({
            "type":        "No Issues Detected",
            "severity":    "info",
            "description": f"Data completeness: {completeness:.1f}%. Dataset passed all integrity checks — no missing values, duplicates, or abnormal entries.",
            "impact":      "Data quality is satisfactory for financial analysis.",
            "count":       0,
        })
        logger.info("validate_data_quality: all checks passed")

    return issues
