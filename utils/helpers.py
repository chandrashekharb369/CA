"""
utils/helpers.py — CA Intelligence Suite
Shared pure-utility functions: formatters, cleaners, and data quality scorers.
No business logic lives here — only reusable helpers.
"""

from __future__ import annotations

import re
import pandas as pd
import numpy as np
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# INR Formatter
# ─────────────────────────────────────────────────────────────────────────────
def inr_format(value: float, symbol: bool = True) -> str:
    """
    Format a number in Indian lakh/crore notation.

    Args:
        value: Numeric value to format.
        symbol: If True, prefix with '₹'.

    Returns:
        Formatted string, e.g. ₹6,33,85,218.98

    Example:
        >>> inr_format(63385218.98)
        '₹6,33,85,218.98'
    """
    prefix = "₹" if symbol else ""
    try:
        value = float(value)
        is_neg = value < 0
        value = abs(value)

        int_part = int(value)
        dec_part = round(value - int_part, 2)
        dec_str = f"{dec_part:.2f}"[1:]   # ".XX"

        # Indian grouping: last 3 digits, then groups of 2
        s = str(int_part)
        if len(s) <= 3:
            formatted = s
        else:
            formatted = s[-3:]
            s = s[:-3]
            while s:
                formatted = s[-2:] + "," + formatted
                s = s[:-2]

        return f"{prefix}{'-' if is_neg else ''}{formatted}{dec_str}"
    except Exception:
        return f"{prefix}{value}"


def inr_pdf(value: float) -> str:
    """
    PDF-safe INR formatter. Replaces ₹ (U+20B9, unsupported by Helvetica)
    with 'Rs. ' so the glyph renders correctly in all PDF viewers.
    """
    return inr_format(value).replace("\u20b9", "Rs. ")


# ─────────────────────────────────────────────────────────────────────────────
# Emoji Stripper (for PDF-safe text)
# ─────────────────────────────────────────────────────────────────────────────
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001FA70-\U0001FAFF"
    "\U00002500-\U00002BEF"
    "\U00002300-\U000023FF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "\u20D0-\u20FF\u0300-\u036F"
    "]+",
    flags=re.UNICODE,
)


def strip_emoji(text: str) -> str:
    """
    Remove all emoji and special Unicode symbols from a string.
    Used to produce PDF-safe text for ReportLab.

    Args:
        text: Input string potentially containing emojis.

    Returns:
        Cleaned string with emojis removed.
    """
    return _EMOJI_RE.sub("", str(text)).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Normalise a transaction description for NLP.
    Lowercases, removes punctuation, collapses whitespace.

    Args:
        text: Raw description string.

    Returns:
        Cleaned lowercase string, or 'unknown' on non-string input.
    """
    if not isinstance(text, str):
        return "unknown"
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Data Cleaning (Phase 2 — clean_data requirement)
# ─────────────────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard data cleaning steps to a financial transaction DataFrame.

    Steps:
        1. Remove exact duplicate rows
        2. Fill missing Amount with column median
        3. Drop rows where Amount is negative (credit notes excluded)
        4. Fill missing Description with 'unknown transaction'
        5. Fill missing Payment_Mode with 'Bank Transfer'
        6. Parse Date column to datetime

    Args:
        df: Raw input DataFrame.

    Returns:
        Cleaned DataFrame with same schema.

    Note:
        This function is non-destructive — it returns a copy.
    """
    original_len = len(df)
    df = df.copy()

    # 1. Remove duplicates
    df = df.drop_duplicates()
    dup_removed = original_len - len(df)
    if dup_removed:
        logger.info("clean_data: removed %d duplicate rows", dup_removed)

    # 2. Fill missing Amount
    if "Amount" in df.columns:
        median_val = df["Amount"].median()
        missing_amount = df["Amount"].isnull().sum()
        if missing_amount:
            df["Amount"] = df["Amount"].fillna(median_val)
            logger.info("clean_data: filled %d missing Amount values with median %.2f",
                        missing_amount, median_val)

        # 3. Drop negative amounts (retain zeros — they may be valid)
        neg_mask = df["Amount"] < 0
        if neg_mask.any():
            logger.warning("clean_data: dropped %d rows with negative Amount", neg_mask.sum())
            df = df[~neg_mask]

    # 4. Text fields
    if "Description" in df.columns:
        df["Description"] = df["Description"].fillna("unknown transaction")
    if "Payment_Mode" in df.columns:
        df["Payment_Mode"] = df["Payment_Mode"].fillna("Bank Transfer")
    if "Vendor_Client_Name" in df.columns:
        df["Vendor_Client_Name"] = df["Vendor_Client_Name"].fillna("unknown")

    # 5. Parse dates
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    logger.info("clean_data: %d rows in → %d rows out", original_len, len(df))
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data Quality Score (Phase 2 requirement)
# ─────────────────────────────────────────────────────────────────────────────
def data_quality_score(df: pd.DataFrame) -> int:
    """
    Compute a 0–100 data quality score for a financial transaction DataFrame.

    Deductions:
        - Each missing cell:     -0.5 points
        - Each duplicate row:    -1.0 point
        - Each negative Amount:  -0.5 points
        - Each zero Amount:      -0.2 points
        - Unrecognised Category: -1.0 point per row

    Args:
        df: Input DataFrame.

    Returns:
        Integer score between 0 (worst) and 100 (perfect).

    Example:
        >>> score = data_quality_score(df)
        >>> print(f"Data Quality: {score}/100")
    """
    if df.empty:
        return 0

    n = len(df)
    penalty = 0.0

    # Missing values
    missing = df.isnull().sum().sum()
    penalty += missing * 0.5

    # Duplicate rows
    duplicates = int(df.duplicated().sum())
    penalty += duplicates * 1.0

    # Negative amounts
    if "Amount" in df.columns:
        neg = int((df["Amount"] < 0).sum())
        zero = int((df["Amount"] == 0).sum())
        penalty += neg * 0.5
        penalty += zero * 0.2

    # Unrecognised categories
    if "Category" in df.columns:
        valid_cats = {"Expense", "Income", "Asset", "Liability"}
        unknown = int((~df["Category"].isin(valid_cats)).sum())
        penalty += unknown * 1.0

    score = max(0, 100 - int(penalty))
    logger.debug("data_quality_score: %d/100 (penalty=%.1f, n=%d)", score, penalty, n)
    return score
