"""
train_model.py
==============
Trains a Gradient Boosting classifier on the PhiUSIIL dataset using ONLY
the URL-string features that can also be computed at runtime for any new URL.

Run once:  python train_model.py
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from feature_extractor import FeatureExtractor

# --- Constants ----------------------------------------------------------------
DATASET_PATH = "PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_PATH   = "model.pkl"
SAMPLE_SIZE  = 60_000   # balanced rows per class (30k legit + 30k phishing)

# Features to use from the CSV (must also be computable from URL at runtime)
CSV_FEATURE_COLS = FeatureExtractor.FEATURE_NAMES


# --- Helpers ------------------------------------------------------------------

def load_real_data():
    """Load the PhiUSIIL CSV and select URL-only feature columns."""
    print(f"  Loading {DATASET_PATH} ...")
    df = pd.read_csv(DATASET_PATH)

    available_cols = [c for c in CSV_FEATURE_COLS if c in df.columns]
    missing = [c for c in CSV_FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  WARNING: Missing columns (will skip): {missing}")

    df_sub = df[available_cols + ["label"]].copy()
    df_sub.dropna(inplace=True)

    n_legit = (df_sub.label == 0).sum()
    n_phish = (df_sub.label == 1).sum()
    print(f"  Loaded {len(df_sub):,} rows | Legit: {n_legit:,}  Phishing: {n_phish:,}")
    return df_sub, available_cols


def balance_and_sample(df, n):
    legit  = df[df.label == 0]
    phish  = df[df.label == 1]
    half   = n // 2
    lsamp  = legit.sample(min(half, len(legit)), random_state=42)
    psamp  = phish.sample(min(half, len(phish)), random_state=42)
    return pd.concat([lsamp, psamp]).sample(frac=1, random_state=42)


def train():
    print("\n==============================================")
    print("       PhishGuard AI  -  Model Trainer")
    print("==============================================\n")

    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset not found at '{DATASET_PATH}'.")
        print("  Please download 'PhiUSIIL_Phishing_URL_Dataset.csv'")
        print("  from Kaggle and place it in this directory, then re-run.")
        sys.exit(1)

    # Load data
    df, feature_cols = load_real_data()
    df = balance_and_sample(df, SAMPLE_SIZE)
    print(f"  Balanced sample: {len(df):,} rows\n")

    X = df[feature_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train
    print("  Training Gradient Boosting classifier (this may take ~60s)...")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42,
        ))
    ])
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n  [OK] Validation Accuracy: {acc*100:.2f}%")
    print()
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))

    # Save model + metadata
    payload = {"model": model, "meta": {"feature_names": feature_cols}}
    joblib.dump(payload, MODEL_PATH)
    print(f"  [OK] Model saved -> {MODEL_PATH}\n")


if __name__ == "__main__":
    train()
