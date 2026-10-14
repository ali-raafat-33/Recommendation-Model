"""
train.py — trains the Co-Found recommendation model on real data.
Run:  python model/train.py --data /path/to/matches_all_*.csv
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler

MODEL_DIR = Path(__file__).parent

# ── Columns that must be dropped before training (leakage / ids) ──────────
DROP_COLS = ["id", "feedback_type", "label_not_relevant"]

# ── Columns the notebook treats as numeric ────────────────────────────────
NUMERIC_COLS = [
    "skill_overlap", "skills_count_a", "complementarity",
    "unique_skills_a", "unique_skills_b", "user_a_verified",
    "user_b_verified", "both_identity_verified",
    "overlapping_skills_count", "covered_skills_count",
    "location_match",
]

TARGET = "label_relevant"


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded  {df.shape[0]:,} rows × {df.shape[1]} cols")

    df_clean = df.copy()
    df_clean.drop(columns=[c for c in DROP_COLS if c in df_clean.columns], inplace=True)

    # Encode match_type
    df_clean["match_type"] = df_clean["match_type"].fillna("unknown")
    df_clean["match_type"] = df_clean["match_type"].map(
        {"collaborator": 0, "project": 1, "unknown": -1}
    )

    # Coerce numeric columns
    for col in NUMERIC_COLS:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Fill NaNs with column median
    for col in df_clean.columns[df_clean.isnull().any()]:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    print(f"Cleaned {df_clean.shape[0]:,} rows × {df_clean.shape[1]} cols")
    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_skills"] = df["skills_count_a"] + df["skills_count_b"]
    df["overlap_ratio"] = np.where(
        df["total_skills"] > 0,
        df["overlapping_skills_count"] / df["total_skills"], 0,
    )
    df["trust_score"]    = df["user_a_verified"].fillna(0) + df["user_b_verified"].fillna(0)
    df["skill_gap"]      = abs(df["skills_count_a"] - df["skills_count_b"])
    df["readiness"]      = df["project_accepting"] + df["team_openness"]
    df["local_verified"] = df["same_location"] * df["user_identity_verified"]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="/mnt/user-data/uploads/matches_all_20260515_211601.csv",
        help="Path to the matches CSV",
    )
    args = parser.parse_args()

    df = load_and_clean(args.data)
    df = engineer_features(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    feature_cols = list(X.columns)
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Label distribution:\n{y.value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Coerce any remaining object cols
    for col in X_train.select_dtypes(include="object").columns:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col]  = pd.to_numeric(X_test[col],  errors="coerce")

    for col in X_train.columns:
        if X_train[col].isnull().any():
            med = X_train[col].median()
            X_train[col] = X_train[col].fillna(med)
            X_test[col]  = X_test[col].fillna(med)

    scaler = MinMaxScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train_sc, y_train)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(rf, X_train_sc, y_train, cv=cv, scoring="roc_auc")
    print(f"\n5-Fold CV ROC-AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

    y_pred      = rf.predict(X_test_sc)
    y_pred_prob = rf.predict_proba(X_test_sc)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)

    print("\n" + "=" * 55)
    print("Classification Report")
    print("=" * 55)
    print(classification_report(y_test, y_pred, target_names=["Not Relevant", "Relevant"]))
    print(f"ROC-AUC: {auc:.4f}")

    # Save artefacts
    joblib.dump(rf,           MODEL_DIR / "rf_model.pkl")
    joblib.dump(scaler,       MODEL_DIR / "scaler.pkl")
    joblib.dump(feature_cols, MODEL_DIR / "feature_cols.pkl")

    # Save column medians for inference-time imputation
    col_medians = {col: float(X_train[col].median()) for col in feature_cols}
    joblib.dump(col_medians, MODEL_DIR / "col_medians.pkl")

    print(f"\n✅  Artefacts saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
