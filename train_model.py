"""
Phase 3: Neural Network Training with Backpropagation
Trains a multi-layer Dense neural network (TensorFlow/Keras) to classify
financial transactions into: Expense, Income, Asset, Liability.

Uses:
  - TF-IDF (description) + Amount + Payment_Mode as input
  - Categorical cross-entropy + Adam optimiser (backpropagation)
  - 80/20 train/test split
  - Saves trained model + metrics
"""

import os
import pickle
import json
import warnings
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.ensemble import IsolationForest

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ─────────────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = "model_artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

LABEL2IDX  = {"Expense": 0, "Income": 1, "Asset": 2, "Liability": 3}
IDX2LABEL  = {v: k for k, v in LABEL2IDX.items()}
NUM_CLASSES = 4

EPOCHS     = 30
BATCH_SIZE = 256
LR         = 1e-3


# ─────────────────────────────────────────────────────────────────────────────
def build_model(input_dim: int, num_classes: int = NUM_CLASSES) -> keras.Model:
    """Dense feed-forward neural network with Dropout regularisation."""
    inputs = keras.Input(shape=(input_dim,), name="features")

    x = layers.Dense(512, activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)

    x = layers.Dense(256, activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.30)(x)

    x = layers.Dense(64, activation="relu")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="category")(x)

    model = keras.Model(inputs, outputs, name="CA_Classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History – CA Transaction Classifier", fontsize=14, fontweight="bold")

    # Accuracy
    axes[0].plot(history.history["accuracy"],    label="Train Accuracy",  color="#2196F3")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy",   color="#FF5722")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train Loss",  color="#2196F3")
    axes[1].plot(history.history["val_loss"], label="Val Loss",   color="#FF5722")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Training curve saved → {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    labels = list(LABEL2IDX.keys())
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Confusion Matrix – CA Classifier", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Confusion matrix saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
def train(company_csv: str = None):
    print("[Phase 3] Training Neural Network (Backpropagation)...")

    if company_csv:
        import preprocess
        print(f"  --> Preprocessing & Training on specific ledger: {company_csv}")
        X_sparse, y, _ = preprocess.preprocess(file_path=company_csv)
        if X_sparse is None:
            return
    else:
        # ── Load preprocessed features ────────────────────────────────────────────
        data = joblib.load(os.path.join(ARTIFACTS_DIR, "features.pkl"))
        X_sparse = data["X"]
        y        = data["y"]

    # Convert sparse → dense numpy (necessary for Keras)
    X = X_sparse.toarray().astype(np.float32)
    print(f"  Features: {X.shape}, Labels: {y.shape}")

    # ── Train / Test split ────────────────────────────────────────────────────
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
    except ValueError:
        print("  ⚠️ Stratified split failed. Falling back to random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        
    print(f"  Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_model(input_dim=X_train.shape[1])
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )

    # ── Train (backpropagation via Adam) ──────────────────────────────────────
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred       = np.argmax(y_pred_proba, axis=1)

    test_acc = accuracy_score(y_test, y_pred)
    print(f"\n  ═══ Test Accuracy: {test_acc * 100:.2f}% ═══")
    
    unique_labels = np.unique(y_test)
    target_names = [list(LABEL2IDX.keys())[i] for i in unique_labels]
    
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names))

    # ── Save model & metrics ──────────────────────────────────────────────────
    model_path = os.path.join(ARTIFACTS_DIR, "ca_model.h5")
    model.save(model_path)
    print(f"  ✓ Model saved → {model_path}")

    # ── Anomaly Detection (Isolation Forest) ─────────────────────────────────
    print("\n  Training Anomaly Detection Engine (Isolation Forest)...")
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(X)
    joblib.dump(iso_forest, os.path.join(ARTIFACTS_DIR, "anomaly_model.pkl"))
    print(f"  ✓ Anomaly model saved → {ARTIFACTS_DIR}/anomaly_model.pkl")

    metrics = {
        "test_accuracy": float(test_acc),
        "epochs_run": len(history.history["accuracy"]),
        "classes": list(LABEL2IDX.keys()),
        "input_dim": int(X_train.shape[1]),
        "classification_report": classification_report(
            y_test, y_pred,
            labels=unique_labels,
            target_names=target_names,
            output_dict=True,
        ),
    }
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Metrics saved → {ARTIFACTS_DIR}/metrics.json")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_history(history, os.path.join(ARTIFACTS_DIR, "training_history.png"))
    plot_confusion_matrix(y_test, y_pred,
                          os.path.join(ARTIFACTS_DIR, "confusion_matrix.png"))

    print(f"\n[Phase 3] ✅ Training complete — Accuracy: {test_acc*100:.2f}%")
    return model, metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train CA Model")
    parser.add_argument("--company", type=str, default=None, help="Path to a specific company CSV file to train on")
    args = parser.parse_args()
    
    train(company_csv=args.company)
