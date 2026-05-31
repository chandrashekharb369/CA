"""
modules/ml_model.py — CA Intelligence Suite
Phase 3: ML Transaction Classifier with Hybrid Fallback

Provides:
    - load_artifacts()         → loads TF model + sklearn transformers
    - predict_categories(df)   → runs inference with hybrid rule-engine fallback
    - _rule_engine_classify()  → fallback for low-confidence predictions

Hybrid Logic:
    If the neural network's confidence < ML_CONFIDENCE_THRESHOLD (default 0.70),
    the transaction is re-classified using a lightweight keyword-based rule,
    preventing the model from confidently mislabelling unusual transactions.
"""

from __future__ import annotations

import os
import json
import re
import warnings
import numpy as np
import pandas as pd
import joblib
from typing import Optional, Dict

from config import (
    ARTIFACTS_DIR, MODEL_PATH, TFIDF_PATH, SCALER_PATH,
    PM_ENCODER_PATH, ANOMALY_MODEL_PATH, METRICS_PATH,
    LABEL2IDX, IDX2LABEL, ML_CONFIDENCE_THRESHOLD,
)
from utils.logger import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Artifact Loader
# ─────────────────────────────────────────────────────────────────────────────
def load_artifacts() -> Optional[Dict]:
    """
    Load all model artifacts: TF model, TF-IDF, scaler, encoder, anomaly model.

    Returns:
        Dict with keys: model, tfidf, scaler, pm_enc, anomaly_model, metrics.
        Returns None if the primary model file does not exist.
        Returns {"error": str} on unexpected exception.
    """
    if not os.path.exists(MODEL_PATH):
        logger.warning("load_artifacts: model not found at %s", MODEL_PATH)
        return None

    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")

        logger.info("load_artifacts: loading TF model from %s", MODEL_PATH)
        model    = tf.keras.models.load_model(MODEL_PATH)
        tfidf    = joblib.load(TFIDF_PATH)
        scaler   = joblib.load(SCALER_PATH)
        pm_enc   = joblib.load(PM_ENCODER_PATH)

        anomaly_model = None
        if os.path.exists(ANOMALY_MODEL_PATH):
            anomaly_model = joblib.load(ANOMALY_MODEL_PATH)
            logger.info("load_artifacts: anomaly model loaded")

        metrics = {}
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, encoding="utf-8") as f:
                metrics = json.load(f)

        logger.info("load_artifacts: all artifacts loaded successfully")
        return {
            "model": model,
            "tfidf": tfidf,
            "scaler": scaler,
            "pm_enc": pm_enc,
            "anomaly_model": anomaly_model,
            "metrics": metrics,
        }

    except Exception as exc:
        logger.error("load_artifacts: failed — %s", exc, exc_info=True)
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Internal: rule-engine keyword fallback classifier
# ─────────────────────────────────────────────────────────────────────────────
_KEYWORD_RULES: list[tuple[str, str]] = [
    # Pattern → Category
    (r"salary|payroll|wage|compensation|bonus|gratuity", "Expense"),
    (r"rent|lease|electricity|power|internet|insurance", "Expense"),
    (r"raw material|purchase|supplier|vendor|inventory", "Expense"),
    (r"loan interest|bank charge|overdraft|finance cost", "Expense"),
    (r"depreciation|amortis|capital consumable",          "Expense"),
    (r"sales|revenue|service income|client receipt",      "Income"),
    (r"other income|interest received|dividend",          "Income"),
    (r"plant|machinery|vehicle|equipment|furniture|land", "Asset"),
    (r"software|patent|trademark|goodwill",               "Asset"),
    (r"trade receivable|debtor|advance paid",             "Asset"),
    (r"cash|bank balance|fixed deposit",                  "Asset"),
    (r"loan taken|borrowing|debenture|term loan",         "Liability"),
    (r"creditor|payable|advance received",                "Liability"),
    (r"provision|tax payable|gst payable",                "Liability"),
]

_COMPILED_RULES = [(re.compile(pat, re.IGNORECASE), cat) for pat, cat in _KEYWORD_RULES]


def _rule_engine_classify(description: str, amount: float) -> str:
    """
    Lightweight keyword-based fallback classifier.
    Activated when the neural network confidence is below threshold.

    Args:
        description: Transaction description string.
        amount:      Transaction amount.

    Returns:
        Predicted category: 'Expense' | 'Income' | 'Asset' | 'Liability'
    """
    for pattern, category in _COMPILED_RULES:
        if pattern.search(description):
            return category
    # Default heuristic: high-value umatched → Asset, else Expense
    return "Asset" if amount > 500_000 else "Expense"


# ─────────────────────────────────────────────────────────────────────────────
# Feature Builder
# ─────────────────────────────────────────────────────────────────────────────
def _build_features(df: pd.DataFrame, artifacts: Dict):
    """
    Build the feature matrix (TF-IDF + numeric) for the neural network.

    Args:
        df:        Input DataFrame with Description, Amount, Payment_Mode, Date.
        artifacts: Loaded model artifacts dict.

    Returns:
        Dense numpy array of shape (n_samples, input_dim).
    """
    from scipy.sparse import hstack, csr_matrix

    def _clean(t):
        if not isinstance(t, str):
            return "unknown"
        t = t.lower().strip()
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    desc     = df["Description"].fillna("unknown transaction").apply(_clean)
    amount   = df["Amount"].fillna(df["Amount"].median())
    pay_mode = df["Payment_Mode"].fillna("Bank Transfer")

    df2           = df.copy()
    df2["Date"]   = pd.to_datetime(df2.get("Date", "2024-04-01"), errors="coerce")
    month         = df2["Date"].dt.month.fillna(1).astype(int)
    is_weekend    = df2["Date"].dt.dayofweek.isin([5, 6]).astype(int)

    pm_enc   = artifacts["pm_enc"]
    classes  = list(pm_enc.classes_)
    pm_codes = pay_mode.apply(lambda x: pm_enc.transform([x])[0] if x in classes else 0)

    tfidf_mat     = artifacts["tfidf"].transform(desc)
    log_amount    = np.log1p(amount)
    amount_scaled = artifacts["scaler"].transform(log_amount.values.reshape(-1, 1))

    num_feat = csr_matrix(
        np.column_stack([
            amount_scaled,
            pm_codes.values,
            month.values,
            is_weekend.values,
        ]).astype(np.float32)
    )
    return hstack([tfidf_mat, num_feat]).toarray().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main Inference Function
# ─────────────────────────────────────────────────────────────────────────────
def predict_categories(
    df: pd.DataFrame,
    artifacts: Dict,
    confidence_threshold: float = ML_CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    """
    Run hybrid ML inference on a financial transaction DataFrame.

    Hybrid Logic (Phase 3):
        1. Build feature matrix from Description, Amount, Payment_Mode, Date.
        2. Run the trained TensorFlow neural network.
        3. For any prediction with confidence < ``confidence_threshold``,
           fall back to the keyword rule engine to avoid confident mislabelling.
        4. Attach anomaly flags if the Isolation Forest model is available.

    Args:
        df:                   Input DataFrame.
        artifacts:            Dict returned by ``load_artifacts()``.
        confidence_threshold: Confidence threshold for hybrid fallback (default 0.70).

    Returns:
        DataFrame with additional columns:
            - ``Predicted_Category`` — Final category label
            - ``Confidence_%``       — Model confidence (0–100)
            - ``Is_Anomaly_Predicted`` — bool (if anomaly model available)
            - ``Anomaly_Score``      — float (if anomaly model available)
    """
    logger.info("predict_categories: running inference on %d rows (threshold=%.2f)",
                len(df), confidence_threshold)

    try:
        X = _build_features(df, artifacts)
    except Exception as exc:
        logger.error("predict_categories: feature build failed — %s", exc, exc_info=True)
        result_df = df.copy()
        result_df["Predicted_Category"] = df.get("Category", "Expense")
        result_df["Confidence_%"] = 0.0
        return result_df

    # ── Neural network inference ───────────────────────────────────────────────
    proba      = artifacts["model"].predict(X, verbose=0)
    preds      = np.argmax(proba, axis=1)
    confidence = np.max(proba, axis=1)

    # ── Hybrid fallback: low confidence → keyword rule engine ─────────────────
    fallback_count = 0
    descriptions   = df["Description"].fillna("").tolist()
    amounts        = df["Amount"].fillna(0).tolist()

    for i, conf in enumerate(confidence):
        if conf < confidence_threshold:
            fallback_cat = _rule_engine_classify(descriptions[i], amounts[i])
            preds[i]     = LABEL2IDX.get(fallback_cat, preds[i])
            fallback_count += 1

    if fallback_count:
        logger.info("predict_categories: %d/%d predictions used keyword fallback",
                    fallback_count, len(df))

    result_df = df.copy()
    result_df["Predicted_Category"] = [IDX2LABEL[p] for p in preds]
    result_df["Confidence_%"]       = (confidence * 100).round(1)

    # ── Anomaly detection ─────────────────────────────────────────────────────
    anomaly_model = artifacts.get("anomaly_model")
    if anomaly_model is not None:
        try:
            anomaly_preds = anomaly_model.predict(X)
            result_df["Is_Anomaly_Predicted"] = np.where(anomaly_preds == -1, True, False)
            result_df["Anomaly_Score"]         = anomaly_model.decision_function(X)
            anom_count = int((anomaly_preds == -1).sum())
            logger.info("predict_categories: %d anomalies flagged by Isolation Forest", anom_count)
        except Exception as exc:
            logger.warning("predict_categories: anomaly detection failed — %s", exc)

    logger.info("predict_categories: complete. Fallback rate: %.1f%%",
                fallback_count / max(len(df), 1) * 100)
    return result_df
