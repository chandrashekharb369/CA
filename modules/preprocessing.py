"""
modules/preprocessing.py — CA Intelligence Suite
Phase 9: Preprocessing Module (Production Wrapper)

Wraps the training-time preprocess.py with production runtime utilities:
    - clean_data_chunked() — streaming / large-file chunk processor
    - preprocess_for_inference() — fast runtime feature build (no disk I/O)
    - data_quality_report() — consolidated quality stats dict

The training pipeline (preprocess.py at root) is unchanged and still used
by run_pipeline.py. This module handles RUNTIME data flowing into the app.
"""

from __future__ import annotations

import re
from typing import Generator, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    CHUNK_SIZE,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_MIN_DF,
)
from utils.logger import get_logger
from utils.helpers import clean_data, data_quality_score, clean_text

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Chunk Processing (Phase 9 — large dataset support)
# ─────────────────────────────────────────────────────────────────────────────
def clean_data_chunked(
    filepath: str,
    chunksize: int = CHUNK_SIZE,
) -> Generator[pd.DataFrame, None, None]:
    """
    Stream-process a large CSV file in chunks, cleaning each chunk independently.

    This avoids loading the entire file into memory — suitable for datasets
    with 100,000+ rows.

    Args:
        filepath:  Path to the CSV file.
        chunksize: Number of rows per chunk (default from config.CHUNK_SIZE).

    Yields:
        Cleaned DataFrame chunks of size ``chunksize``.

    Example:
        chunks = list(clean_data_chunked("large_ledger.csv"))
        df = pd.concat(chunks, ignore_index=True)
    """
    logger.info("clean_data_chunked: reading '%s' in chunks of %d", filepath, chunksize)
    total_rows = 0
    chunk_num  = 0

    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        chunk_num  += 1
        cleaned     = clean_data(chunk)
        total_rows += len(cleaned)
        logger.debug("clean_data_chunked: chunk %d → %d rows", chunk_num, len(cleaned))
        yield cleaned

    logger.info("clean_data_chunked: processed %d chunks, total %d rows", chunk_num, total_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Runtime Feature Builder (no disk I/O — uses already-loaded artifacts)
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_for_inference(df: pd.DataFrame, artifacts: dict) -> np.ndarray:
    """
    Transform a raw transactions DataFrame into a neural-network feature matrix
    using the already-loaded sklearn artifacts (TF-IDF, scaler, encoder).

    This is the inference-time analogue of preprocess.py's ``preprocess()``
    function — it uses fitted transformers rather than fitting new ones.

    Feature vector:
        [TF-IDF(Description) | log(Amount)_scaled | Payment_Mode_enc | Month | Is_Weekend]

    Args:
        df:        Raw or cleaned transactions DataFrame.
        artifacts: Dict returned by ``modules.ml_model.load_artifacts()``.

    Returns:
        Dense float32 numpy array of shape (n_rows, input_dim).

    Raises:
        KeyError: If required artifact keys (tfidf, scaler, pm_enc) are missing.
    """
    from scipy.sparse import hstack, csr_matrix

    logger.info("preprocess_for_inference: building features for %d rows", len(df))

    # ── Text features ─────────────────────────────────────────────────────────
    desc       = df["Description"].fillna("unknown transaction").apply(clean_text)
    tfidf_mat  = artifacts["tfidf"].transform(desc)

    # ── Numeric features ──────────────────────────────────────────────────────
    amount     = df["Amount"].fillna(df["Amount"].median())
    log_amount = np.log1p(amount)
    amount_scaled = artifacts["scaler"].transform(log_amount.values.reshape(-1, 1))

    # ── Categorical features ──────────────────────────────────────────────────
    pay_mode = df["Payment_Mode"].fillna("Bank Transfer")
    pm_enc   = artifacts["pm_enc"]
    classes  = list(pm_enc.classes_)
    pm_codes = pay_mode.apply(lambda x: pm_enc.transform([x])[0] if x in classes else 0)

    # ── Temporal features ─────────────────────────────────────────────────────
    dates      = pd.to_datetime(df.get("Date", "2024-04-01"), errors="coerce")
    month      = dates.dt.month.fillna(1).astype(int)
    is_weekend = dates.dt.dayofweek.isin([5, 6]).astype(int)

    # ── Assemble feature matrix ───────────────────────────────────────────────
    num_feat = csr_matrix(
        np.column_stack([
            amount_scaled,
            pm_codes.values,
            month.values,
            is_weekend.values,
        ]).astype(np.float32)
    )
    X = hstack([tfidf_mat, num_feat]).toarray().astype(np.float32)
    logger.info("preprocess_for_inference: feature matrix shape %s", X.shape)
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Data Quality Report
# ─────────────────────────────────────────────────────────────────────────────
def data_quality_report(df: pd.DataFrame) -> dict:
    """
    Generate a consolidated data quality statistics dictionary.

    Suitable for displaying in a dashboard header or logging.

    Args:
        df: Input DataFrame.

    Returns:
        Dict with keys:
            rows, columns, missing_cells, missing_pct, duplicates,
            negative_amounts, zero_amounts, quality_score.
    """
    rows           = len(df)
    cols           = len(df.columns)
    missing_cells  = int(df.isnull().sum().sum())
    missing_pct    = round(missing_cells / max(rows * cols, 1) * 100, 2)
    duplicates     = int(df.duplicated().sum())
    neg_amounts    = int((df["Amount"] < 0).sum()) if "Amount" in df.columns else 0
    zero_amounts   = int((df["Amount"] == 0).sum()) if "Amount" in df.columns else 0
    score          = data_quality_score(df)

    report = {
        "rows":            rows,
        "columns":         cols,
        "missing_cells":   missing_cells,
        "missing_pct":     missing_pct,
        "duplicates":      duplicates,
        "negative_amounts": neg_amounts,
        "zero_amounts":    zero_amounts,
        "quality_score":   score,
    }
    logger.info(
        "data_quality_report: %d rows, score=%d/100, missing=%d, dupes=%d",
        rows, score, missing_cells, duplicates,
    )
    return report
