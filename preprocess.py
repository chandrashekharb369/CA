"""
Phase 2: Data Preprocessing Pipeline
Cleans, normalizes, encodes, and vectorizes the synthetic dataset
for use by the machine learning model.
"""

import os
import pandas as pd
import numpy as np
import joblib
import re
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings("ignore")

# Paths

TRAIN_DIR     = "train_dataset"
ARTIFACTS_DIR = "model_artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

LABEL2IDX = {"Expense": 0, "Income": 1, "Asset": 2, "Liability": 3}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}


# Text cleaning

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return "unknown"
    text = text.lower().strip()
    # Remove punctuation/special chars, keep alphanumeric + space
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Main preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(file_path: str = None, data_dir: str = TRAIN_DIR):
    print("[Phase 2] Data Preprocessing...")
    
    import glob
    if file_path:
        csv_files = [file_path]
    else:
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        # Exclude any consolidated/clean files so we don't duplicate
        csv_files = [f for f in csv_files if "clean" not in f]
    
    if not csv_files:
        print(f"  ❌ No CSV files found")
        return None, None, None
        
    print(f"  Found {len(csv_files)} datasets. Consolidating...")
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    print(f"  Loaded combined dataset: {len(df):,} rows × {len(df.columns)} columns")

    # ── 1. Handle missing values ──────────────────────────────────────────────
    df["Description"]        = df["Description"].fillna("unknown transaction")
    df["Vendor_Client_Name"] = df["Vendor_Client_Name"].fillna("unknown")
    df["Payment_Mode"]       = df["Payment_Mode"].fillna("Bank Transfer")
    print(f"  ✓ Missing values filled (remaining: {df.isnull().sum().sum()})")

    # ── 2. Normalise text ─────────────────────────────────────────────────────
    df["Description_clean"] = df["Description"].apply(clean_text)
    print("  ✓ Descriptions normalised")

    # ── 2.b Temporal features ──────────────────────────────────────────────────
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Month"] = df["Date"].dt.month.fillna(1).astype(int)
    df["Is_Weekend"] = df["Date"].dt.dayofweek.isin([5, 6]).astype(int)
    print("  ✓ Temporal features engineered (Month, Is_Weekend)")

    # ── 3. Encode Payment_Mode ────────────────────────────────────────────────
    pm_encoder = LabelEncoder()
    df["Payment_Mode_enc"] = pm_encoder.fit_transform(df["Payment_Mode"].astype(str))
    joblib.dump(pm_encoder, os.path.join(ARTIFACTS_DIR, "payment_mode_encoder.pkl"))
    print("  ✓ Payment_Mode encoded")

    # ── 4. Encode target (Category) ───────────────────────────────────────────
    df["Category_enc"] = df["Category"].map(LABEL2IDX)
    print("  ✓ Target categories encoded")

    # ── 5. Normalise Amount ───────────────────────────────────────────────────
    scaler = StandardScaler()
    df["Log_Amount"] = np.log1p(df["Amount"].fillna(0))
    df["Amount_scaled"] = scaler.fit_transform(df[["Log_Amount"]])
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "amount_scaler.pkl"))
    print("  ✓ Amount normalised (Log + StandardScaler)")

    # ── 6. TF-IDF on Description ──────────────────────────────────────────────
    tfidf = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        min_df=2,
        strip_accents="unicode",
    )
    tfidf_matrix = tfidf.fit_transform(df["Description_clean"])
    joblib.dump(tfidf, os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.pkl"))
    print(f"  ✓ TF-IDF matrix: {tfidf_matrix.shape}")

    # ── 7. Build feature matrix ───────────────────────────────────────────────
    # [TF-IDF features | Amount_scaled | Payment_Mode_enc | Month | Is_Weekend]
    numeric_features = csr_matrix(
        df[["Amount_scaled", "Payment_Mode_enc", "Month", "Is_Weekend"]].values.astype(np.float32)
    )
    X = hstack([tfidf_matrix, numeric_features])
    y = df["Category_enc"].values

    print(f"  ✓ Final feature matrix: {X.shape}")

    # ── 8. Save cleaned dataframe ─────────────────────────────────────────────
    df.to_csv("train_dataset/financial_dataset_clean.csv", index=False)
    joblib.dump({"X": X, "y": y}, os.path.join(ARTIFACTS_DIR, "features.pkl"))
    print(f"  ✓ Saved: train_dataset/financial_dataset_clean.csv & {ARTIFACTS_DIR}/features.pkl")

    # Class distribution check
    print("\n  Class distribution in encoded targets:")
    for label, idx in LABEL2IDX.items():
        cnt = (y == idx).sum()
        print(f"    {label:<12}: {cnt:,} ({cnt/len(y)*100:.1f}%)")

    return X, y, df


def load_preprocessed():
    """Load saved preprocessed features."""
    data = joblib.load(os.path.join(ARTIFACTS_DIR, "features.pkl"))
    return data["X"], data["y"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None, help="Path to a specific CSV file to preprocess")
    args = parser.parse_args()
    preprocess(file_path=args.file)
