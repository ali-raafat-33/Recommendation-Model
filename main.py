"""
Co-Found Recommendation API
============================
Trained on:  matches_all_20260515_211601.csv  (1 288 rows, 25 cols)
Model:       RandomForestClassifier (n=300, max_depth=8, class_weight=balanced)
Features:    27 (raw + engineered)
ROC-AUC:     ~0.69 on hold-out

Endpoints
---------
GET  /health
POST /predict          — score ONE match pair
POST /predict/batch    — score up to 1 000 pairs
POST /retrain          — retrain from a new CSV upload
"""

import io
import os
import tempfile
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Security, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator, model_validator

# ── Config ────────────────────────────────────────────────────────────────
API_SECRET  = os.getenv("API_SECRET", "dev-secret")
MODEL_DIR   = Path(__file__).parent / "model"

# ── Load artefacts ────────────────────────────────────────────────────────
def _load():
    rf          = joblib.load(MODEL_DIR / "rf_model.pkl")
    scaler      = joblib.load(MODEL_DIR / "scaler.pkl")
    feature_cols: list[str] = joblib.load(MODEL_DIR / "feature_cols.pkl")
    col_medians: dict       = joblib.load(MODEL_DIR / "col_medians.pkl")
    return rf, scaler, feature_cols, col_medians

rf, scaler, FEATURE_COLS, COL_MEDIANS = _load()

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Co-Found Recommendation API",
    description=(
        "Predicts whether a collaborator or project match is **relevant** for a user.\n\n"
        "Trained on real match data from Co-Found (1 288 rows).\n\n"
        "**Auth:** `Authorization: Bearer <API_SECRET>` — default dev secret: `dev-secret`"
    ),
    version="2.0.0",
)

_bearer = HTTPBearer()

def verify(creds: HTTPAuthorizationCredentials = Security(_bearer)):
    if creds.credentials != API_SECRET:
        raise HTTPException(status_code=401, detail="Invalid API secret.")


# ── Pydantic request schemas ──────────────────────────────────────────────

class CollaboratorInput(BaseModel):
    """All feature fields from the real dataset for a user-to-user match."""
    match_type: Literal["collaborator"] = "collaborator"

    # Core skill features
    skill_overlap:            float = Field(..., ge=0, le=1,  example=0.25,
                                            description="Jaccard similarity: |A∩B|/|A∪B|")
    complementarity:          float = Field(..., ge=0, le=1,  example=0.75,
                                            description="1 − skill_overlap")
    overlapping_skills_count: int   = Field(..., ge=0,        example=3)
    skills_count_a:           int   = Field(..., ge=0,        example=8,
                                            description="Total skills of user A (receiver)")
    skills_count_b:           int   = Field(..., ge=0,        example=7,
                                            description="Total skills of user B (matched)")
    unique_skills_a:          Optional[int]   = Field(None,   example=8)
    unique_skills_b:          Optional[int]   = Field(None,   example=7)

    # Identity / location
    user_a_verified:          int   = Field(..., ge=0, le=1,  example=1)
    user_b_verified:          int   = Field(..., ge=0, le=1,  example=1)
    both_identity_verified:   int   = Field(..., ge=0, le=1,  example=1)
    location_match:           int   = Field(..., ge=0, le=1,  example=0)
    same_location:            int   = Field(..., ge=0, le=1,  example=0)
    user_identity_verified:   int   = Field(0,  ge=0, le=1,  example=0)

    # Engagement signals (if known; default = not yet seen)
    compatibility_score:      float = Field(0.0, ge=0, le=1, example=0.42)
    viewed:                   int   = Field(0,   ge=0, le=1)
    saved:                    int   = Field(0,   ge=0, le=1)
    action_taken:             int   = Field(0,   ge=0, le=1)

    # Project fields → not applicable for collaborator; set to neutral
    project_accepting:        int   = Field(0,   ge=0, le=1)
    team_openness:            float = Field(0.0, ge=0, le=1)
    covered_skills_count:     int   = Field(0,   ge=0)


class ProjectInput(BaseModel):
    """All feature fields from the real dataset for a user-to-project match."""
    match_type: Literal["project"] = "project"

    # Skill coverage
    skill_overlap:            float = Field(0.0, ge=0, le=1,  example=0.0)
    complementarity:          float = Field(0.0, ge=0, le=1,  example=0.0)
    overlapping_skills_count: int   = Field(0,   ge=0,        example=0)
    skills_count_a:           int   = Field(..., ge=0,        example=10,
                                            description="User's total skills")
    skills_count_b:           int   = Field(0,   ge=0,        example=0)
    unique_skills_a:          Optional[int]   = Field(None,   example=10)
    unique_skills_b:          Optional[int]   = Field(None,   example=0)
    covered_skills_count:     int   = Field(..., ge=0,        example=6,
                                            description="User skills matching project requirements")

    # Project readiness
    project_accepting:        int   = Field(..., ge=0, le=1,  example=1)
    team_openness:            float = Field(..., ge=0, le=1,  example=0.5,
                                            description="(max_team − current) / max_team")

    # Identity / location
    user_identity_verified:   int   = Field(..., ge=0, le=1,  example=1)
    same_location:            int   = Field(0,   ge=0, le=1,  example=0)
    location_match:           int   = Field(0,   ge=0, le=1,  example=0)
    user_a_verified:          int   = Field(0,   ge=0, le=1)
    user_b_verified:          int   = Field(0,   ge=0, le=1)
    both_identity_verified:   int   = Field(0,   ge=0, le=1)

    # Engagement signals
    compatibility_score:      float = Field(0.0, ge=0, le=1,  example=0.61)
    viewed:                   int   = Field(0,   ge=0, le=1)
    saved:                    int   = Field(0,   ge=0, le=1)
    action_taken:             int   = Field(0,   ge=0, le=1)


class PredictRequest(BaseModel):
    """Wrapper that accepts either a collaborator or project input."""
    data: CollaboratorInput | ProjectInput = Field(
        ...,
        discriminator="match_type",
        description="Use `match_type: collaborator` or `match_type: project`",
    )


class BatchPredictRequest(BaseModel):
    pairs: list[CollaboratorInput | ProjectInput] = Field(
        ..., min_length=1, max_length=1000,
        description="Up to 1 000 match pairs"
    )

    @field_validator("pairs", mode="before")
    @classmethod
    def tag_match_type(cls, v):
        return v


# ── Feature engineering (mirrors notebook exactly) ───────────────────────

def _engineer(d: dict) -> dict:
    """Apply the same feature engineering steps as the notebook."""
    row = dict(d)

    # defaults for missing keys
    row.setdefault("unique_skills_a", row.get("skills_count_a", 0))
    row.setdefault("unique_skills_b", row.get("skills_count_b", 0))
    if row["unique_skills_a"] is None:
        row["unique_skills_a"] = row.get("skills_count_a", 0)
    if row["unique_skills_b"] is None:
        row["unique_skills_b"] = row.get("skills_count_b", 0)

    # match_type encoding
    mt = row.get("match_type", "collaborator")
    row["match_type"] = 0 if mt == "collaborator" else (1 if mt == "project" else -1)

    # engineered features
    total = row.get("skills_count_a", 0) + row.get("skills_count_b", 0)
    row["total_skills"]   = total
    row["overlap_ratio"]  = (row.get("overlapping_skills_count", 0) / total
                             if total > 0 else 0.0)
    row["trust_score"]    = row.get("user_a_verified", 0) + row.get("user_b_verified", 0)
    row["skill_gap"]      = abs(row.get("skills_count_a", 0) - row.get("skills_count_b", 0))
    row["readiness"]      = row.get("project_accepting", 0) + row.get("team_openness", 0)
    row["local_verified"] = row.get("same_location", 0) * row.get("user_identity_verified", 0)

    return row


def _to_vector(d: dict) -> list:
    """Convert a feature dict to the exact ordered vector the model expects."""
    engineered = _engineer(d)
    return [float(engineered.get(col, COL_MEDIANS.get(col, 0))) for col in FEATURE_COLS]


def _predict(input_obj) -> dict:
    """Run one prediction and return a rich response dict."""
    raw = input_obj.model_dump()
    match_type = raw.get("match_type", "collaborator")

    vec    = _to_vector(raw)
    df_row = pd.DataFrame([vec], columns=FEATURE_COLS)
    scaled = scaler.transform(df_row)
    prob   = float(rf.predict_proba(scaled)[0][1])
    label  = int(rf.predict(scaled)[0])
    score  = round(float(np.clip(prob, 0.01, 1.0)), 4)

    # Which features drive this prediction most?
    importances = dict(zip(FEATURE_COLS, rf.feature_importances_))
    if match_type == "collaborator":
        reason_keys = ["skill_overlap", "complementarity", "location_match",
                       "both_identity_verified", "trust_score", "overlap_ratio"]
    else:
        reason_keys = ["covered_skills_count", "team_openness", "project_accepting",
                       "user_identity_verified", "readiness", "skills_count_a"]

    engineered = _engineer(raw)
    match_reasons = {
        k: round(float(engineered.get(k, 0)), 4)
        for k in reason_keys if k in engineered
    }

    top_features = sorted(
        importances.items(), key=lambda x: x[1], reverse=True
    )[:5]

    return {
        "match_type":                match_type,
        "is_relevant":               bool(label),
        "relevant_probability":      score,
        "compatibility_score":       score,
        "match_reasons":             match_reasons,
        "top_model_features":        [{"feature": k, "importance": round(v, 4)}
                                      for k, v in top_features],
    }


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"], summary="Health check")
def health():
    """Returns model status and feature count. No auth required."""
    return {
        "status":        "ok",
        "model":         "RandomForestClassifier",
        "n_features":    len(FEATURE_COLS),
        "feature_cols":  FEATURE_COLS,
        "version":       "2.0.0",
    }


@app.post(
    "/predict",
    tags=["Prediction"],
    summary="Score a single match pair",
    response_description="Relevance score, probability, and match reasons",
)
def predict_one(req: PredictRequest, _=Depends(verify)):
    """
    Score **one** match pair (collaborator or project).

    Set `match_type` to `"collaborator"` for user-to-user matches
    or `"project"` for user-to-project matches — the required fields differ.

    Returns:
    - **is_relevant** — boolean prediction
    - **relevant_probability** — model confidence (0–1)
    - **compatibility_score** — same as `relevant_probability`, clipped to [0.01, 1.0]
    - **match_reasons** — key feature values that drove the prediction
    - **top_model_features** — top 5 most important features overall
    """
    return {"status": "success", "data": _predict(req.data)}


@app.post(
    "/predict/batch",
    tags=["Prediction"],
    summary="Score up to 1 000 match pairs at once",
)
def predict_batch(req: BatchPredictRequest, _=Depends(verify)):
    """
    Batch-score match pairs. Results are returned in the **same order** as the input.

    Mix of collaborator and project pairs is supported in one request.
    """
    results = [_predict(p) for p in req.pairs]
    return {
        "status": "success",
        "count":  len(results),
        "data":   results,
    }


@app.post(
    "/retrain",
    tags=["Model Management"],
    summary="Retrain the model from a new CSV file",
)
async def retrain(
    file: UploadFile = File(..., description="New matches CSV (same columns as training data)"),
    _=Depends(verify),
):
    """
    Upload a new `matches_all_*.csv` and retrain the model in place.

    The server reloads the new artefacts immediately after training completes.
    The CSV must have the same columns as the original training file.
    """
    global rf, scaler, FEATURE_COLS, COL_MEDIANS

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=422, detail="Only .csv files are accepted.")

    contents = await file.read()

    # Write to a temp file and run the training script
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    import subprocess, sys
    result = subprocess.run(
        [sys.executable, str(MODEL_DIR / "train.py"), "--data", tmp_path],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"Training failed:\n{result.stderr[-2000:]}",
        )

    # Reload artefacts
    rf, scaler, FEATURE_COLS, COL_MEDIANS = _load()

    return {
        "status":  "success",
        "message": "Model retrained and reloaded successfully.",
        "log":     result.stdout[-3000:],
    }
