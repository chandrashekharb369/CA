"""
config.py — CA Intelligence Suite
Centralised configuration: all constants, paths, and thresholds in one place.
Any module needing a constant should import from here — never hard-code values.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Directory Paths
# ─────────────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = "model_artifacts"
TRAIN_DIR     = "train_dataset"
TEST_DIR      = "test_dataset"
DATA_DIR      = "data"
RULES_DIR     = "rules"
LOGS_DIR      = "logs"

# Ensure runtime directories exist
for _d in (ARTIFACTS_DIR, DATA_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Model Artifact Paths
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH          = os.path.join(ARTIFACTS_DIR, "ca_model.h5")
TFIDF_PATH          = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.pkl")
SCALER_PATH         = os.path.join(ARTIFACTS_DIR, "amount_scaler.pkl")
PM_ENCODER_PATH     = os.path.join(ARTIFACTS_DIR, "payment_mode_encoder.pkl")
ANOMALY_MODEL_PATH  = os.path.join(ARTIFACTS_DIR, "anomaly_model.pkl")
METRICS_PATH        = os.path.join(ARTIFACTS_DIR, "metrics.json")
FEATURES_PATH       = os.path.join(ARTIFACTS_DIR, "features.pkl")

# Compliance rules JSON
COMPLIANCE_RULES_PATH = os.path.join(RULES_DIR, "compliance_rules.json")

# ─────────────────────────────────────────────────────────────────────────────
# Category Label Mappings
# ─────────────────────────────────────────────────────────────────────────────
LABEL2IDX: dict[str, int] = {
    "Expense":   0,
    "Income":    1,
    "Asset":     2,
    "Liability": 3,
}
IDX2LABEL: dict[int, str] = {v: k for k, v in LABEL2IDX.items()}

VALID_CATEGORIES = frozenset(LABEL2IDX.keys())

# ─────────────────────────────────────────────────────────────────────────────
# ML Inference Thresholds
# ─────────────────────────────────────────────────────────────────────────────
# When model confidence is below this, hybrid logic falls back to rule engine
ML_CONFIDENCE_THRESHOLD: float = 0.70

# ─────────────────────────────────────────────────────────────────────────────
# Financial Health Thresholds (Profit Margin %)
# ─────────────────────────────────────────────────────────────────────────────
PROFIT_MARGIN_STRONG:   float = 15.0   # ≥ 15% → Strong
PROFIT_MARGIN_MODERATE: float =  5.0   # ≥  5% → Moderate
PROFIT_MARGIN_WEAK:     float =  0.0   # ≥  0% → Weak  (else Loss-Making)

# ─────────────────────────────────────────────────────────────────────────────
# Tax Validation Thresholds
# ─────────────────────────────────────────────────────────────────────────────
TAX_RATE_LOW:  float = 0.20   # 20% of PBT — lower bound
TAX_RATE_HIGH: float = 0.30   # 30% of PBT — upper bound

# ─────────────────────────────────────────────────────────────────────────────
# GST & Expense Ratio Thresholds
# ─────────────────────────────────────────────────────────────────────────────
GST_RATIO_HIGH:     float = 0.15   # Net GST > 15% of income → warning
EXPENSE_RATIO_HIGH: float = 0.80   # Expenses > 80% of income → warning

# ─────────────────────────────────────────────────────────────────────────────
# Anomaly Detection
# ─────────────────────────────────────────────────────────────────────────────
LARGE_TXN_IQR_FACTOR: float = 3.0    # Outlier if Amount > Q3 + factor × IQR
MIN_ROWS_FOR_ANOMALY:  int   = 10    # Minimum rows needed for IQR analysis
TOP_N_EXPENSE_CATS:    int   = 3     # Top N expense sub-categories to flag

# ─────────────────────────────────────────────────────────────────────────────
# Compliance Scoring Weights (Phase 6)
# ─────────────────────────────────────────────────────────────────────────────
COMPLIANCE_CRITICAL_WEIGHT: int = 5
COMPLIANCE_WARNING_WEIGHT:  int = 3
# Formula: risk_score = critical × 5 + warning × 3
# compliance_score = max(0, 100 − risk_score)

# ─────────────────────────────────────────────────────────────────────────────
# Visualisation Colour Palette (Plotly / Matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
CAT_COLORS: dict[str, str] = {
    "Expense":   "#1e40af",
    "Income":    "#3b82f6",
    "Asset":     "#93c5fd",
    "Liability": "#60a5fa",
}

MONO_SEQ: list[str] = [
    "#0d1b36", "#1e3a5f", "#1e40af", "#2563eb",
    "#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe",
]

PLOT_LAYOUT: dict = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#e6edf3"),
    margin=dict(l=30, r=30, t=50, b=30),
)

# ─────────────────────────────────────────────────────────────────────────────
# Data Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES: int = 500
TFIDF_NGRAM_RANGE: tuple = (1, 2)
TFIDF_MIN_DF: int = 2
CHUNK_SIZE: int = 10_000   # Rows per chunk for large-file processing

# ─────────────────────────────────────────────────────────────────────────────
# Training Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_EPOCHS:     int   = 30
TRAIN_BATCH_SIZE: int   = 256
TRAIN_LR:         float = 1e-3
TRAIN_TEST_SPLIT: float = 0.20
ISOLATION_FOREST_CONTAMINATION: float = 0.05

# ─────────────────────────────────────────────────────────────────────────────
# Application Meta
# ─────────────────────────────────────────────────────────────────────────────
APP_TITLE:   str = "CA Intelligence Suite"
APP_ICON:    str = "🏦"
APP_VERSION: str = "2.0.0"
