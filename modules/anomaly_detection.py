"""
modules/anomaly_detection.py — CA Intelligence Suite
Phase 11: Anomaly Detection Engine

Provides:
    - detect_statistical_anomalies(df)          → IQR-based outliers
    - detect_ml_anomalies(df, anomaly_model, X) → Isolation Forest results
    - get_anomaly_summary(df, anomaly_model, X) → consolidated report

Separating anomaly detection into its own module ensures:
    1. It can be tested independently of the Streamlit UI
    2. Both statistical and ML methods are available even without the model
    3. Results are consistently structured for consumption by any reporter
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import LARGE_TXN_IQR_FACTOR, MIN_ROWS_FOR_ANOMALY
from utils.logger import get_logger
from utils.helpers import inr_format

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Statistical Anomaly Detection (IQR Method)
# ─────────────────────────────────────────────────────────────────────────────
def detect_statistical_anomalies(df: pd.DataFrame) -> Dict:
    """
    Detect statistically anomalous transactions using the IQR method.

    A transaction is flagged as an outlier if:
        Amount > Q3 + LARGE_TXN_IQR_FACTOR × IQR

    Args:
        df: Financial transactions DataFrame with an 'Amount' column.

    Returns:
        Dict with keys:
            available (bool)    — True if dataset was large enough
            threshold (float)   — Computed upper bound
            count (int)         — Number of outlier transactions
            total_amount (float)— Sum of outlier amounts
            outlier_df (DataFrame) — Rows flagged as outliers
            interpretation (str)— Human-readable description

    Note:
        Returns ``{"available": False}`` if Dataset has fewer than
        ``MIN_ROWS_FOR_ANOMALY`` rows or no 'Amount' column.
    """
    if "Amount" not in df.columns:
        logger.warning("detect_statistical_anomalies: 'Amount' column not found")
        return {"available": False, "count": 0, "outlier_df": pd.DataFrame()}

    amounts = df["Amount"].dropna()
    if len(amounts) < MIN_ROWS_FOR_ANOMALY:
        return {
            "available": False,
            "count": 0,
            "outlier_df": pd.DataFrame(),
            "interpretation": f"Insufficient data ({len(amounts)} rows) for statistical outlier detection.",
        }

    q1, q3    = amounts.quantile(0.25), amounts.quantile(0.75)
    iqr       = q3 - q1
    threshold = q3 + LARGE_TXN_IQR_FACTOR * iqr

    outlier_df   = df[df["Amount"] > threshold].copy()
    outlier_count = len(outlier_df)
    total_amount  = float(outlier_df["Amount"].sum()) if not outlier_df.empty else 0.0

    if outlier_count > 0:
        max_val         = outlier_df["Amount"].max()
        interpretation  = (
            f"{outlier_count} transaction(s) exceed the statistical outlier threshold of "
            f"{inr_format(threshold)} (Q3 + 3×IQR using {len(amounts):,} data points). "
            f"The largest flagged transaction is {inr_format(max_val)}. "
            "These entries may represent capital expenditures, bulk orders, or erroneous entries "
            "and each requires individual vouching and documentation review."
        )
        logger.info(
            "detect_statistical_anomalies: %d outliers found (threshold=%s)",
            outlier_count, inr_format(threshold),
        )
    else:
        interpretation = (
            f"No statistically anomalous transactions detected. "
            f"All {len(amounts):,} entries are within the normal range "
            f"(threshold: {inr_format(threshold)})."
        )

    return {
        "available":     True,
        "threshold":     float(threshold),
        "q1":            float(q1),
        "q3":            float(q3),
        "iqr":           float(iqr),
        "count":         outlier_count,
        "total_amount":  total_amount,
        "outlier_df":    outlier_df,
        "interpretation": interpretation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ML Anomaly Detection (Isolation Forest)
# ─────────────────────────────────────────────────────────────────────────────
def detect_ml_anomalies(
    df: pd.DataFrame,
    anomaly_model,
    X: Optional[np.ndarray] = None,
) -> Dict:
    """
    Detect anomalies using the trained Isolation Forest model.

    Args:
        df:            Financial transactions DataFrame.
        anomaly_model: Fitted sklearn IsolationForest model.
        X:             Feature matrix (numpy array). If None, falls back to
                       Amount-only simple detection.

    Returns:
        Dict with keys:
            available (bool), count, anomaly_df, anomaly_scores, interpretation.
    """
    if anomaly_model is None:
        return {
            "available": False,
            "count": 0,
            "anomaly_df": pd.DataFrame(),
            "interpretation": "Isolation Forest model not loaded — ML anomaly detection unavailable.",
        }

    try:
        if X is None:
            # Fallback: use log-scaled Amount only
            amounts = df["Amount"].fillna(0)
            X = np.log1p(amounts.values.reshape(-1, 1)).astype(np.float32)

        preds  = anomaly_model.predict(X)
        scores = anomaly_model.decision_function(X)

        anomaly_mask = preds == -1
        anomaly_df   = df[anomaly_mask].copy()
        anomaly_df["Anomaly_Score"] = scores[anomaly_mask]

        count = int(anomaly_mask.sum())
        interpretation = (
            f"Isolation Forest flagged {count} transaction(s) as anomalous out of {len(df):,}. "
            f"(Contamination rate: {count/max(len(df),1)*100:.1f}%). "
            "These may represent unusual patterns in description, amount, or payment mode combinations."
            if count > 0 else
            "No anomalous transactions detected by the Isolation Forest model."
        )

        logger.info("detect_ml_anomalies: %d anomalies flagged by Isolation Forest", count)
        return {
            "available":   True,
            "count":       count,
            "anomaly_df":  anomaly_df,
            "anomaly_scores": scores,
            "interpretation": interpretation,
        }

    except Exception as exc:
        logger.error("detect_ml_anomalies: failed — %s", exc, exc_info=True)
        return {
            "available": False,
            "count": 0,
            "anomaly_df": pd.DataFrame(),
            "interpretation": f"ML anomaly detection failed: {exc}",
        }

# Consolidated Anomaly Summary
def get_anomaly_summary(
    df: pd.DataFrame,
    anomaly_model=None,
    X: Optional[np.ndarray] = None,
) -> Dict:
    """
    Run both statistical and ML anomaly detection and return a unified report.

    Args:
        df:            Financial transactions DataFrame.
        anomaly_model: Optional Isolation Forest model.
        X:             Optional feature matrix for ML detection.

    Returns:
        Dict with keys:
            statistical — result of detect_statistical_anomalies()
            ml          — result of detect_ml_anomalies()
            total_flagged — combined unique anomaly count
    """
    stat_result = detect_statistical_anomalies(df)
    ml_result   = detect_ml_anomalies(df, anomaly_model, X)

    # Compute unique anomaly count (union of both methods)
    stat_indices = set(stat_result["outlier_df"].index) if stat_result["count"] > 0 else set()
    ml_indices   = set(ml_result["anomaly_df"].index)   if ml_result["count"] > 0 else set()
    total_flagged = len(stat_indices | ml_indices)

    logger.info(
        "get_anomaly_summary: statistical=%d, ml=%d, total_unique=%d",
        stat_result["count"], ml_result["count"], total_flagged,
    )

    return {
        "statistical":    stat_result,
        "ml":             ml_result,
        "total_flagged":  total_flagged,
    }
