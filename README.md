# Co-Found Recommendation API

FastAPI service wrapping the **RandomForest recommendation model** trained on `matches_all_20260515_211601.csv`.

| Metric | Value |
|---|---|
| Training rows | 1 288 |
| Features | 27 (19 raw + 6 engineered) |
| CV ROC-AUC | 0.69 ± 0.02 |
| Hold-out ROC-AUC | 0.68 |

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model on your CSV (run once)
python model/train.py --data matches_all_20260515_211601.csv

# 3. Start the server
uvicorn main:app --reload --port 8000
```

Interactive docs → http://localhost:8000/docs

Set a real secret in production:
```bash
API_SECRET=your-real-secret uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Authentication

Every endpoint (except `/health`) requires:
```
Authorization: Bearer dev-secret
```
Override with `API_SECRET` env var.

---

## Endpoints

### `GET /health`
No auth needed.
```bash
curl http://localhost:8000/health
```
```json
{
  "status": "ok",
  "model": "RandomForestClassifier",
  "n_features": 27,
  "feature_cols": ["match_type", "compatibility_score", ...],
  "version": "2.0.0"
}
```

---

### `POST /predict` — score one pair

#### Collaborator match (user ↔ user)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer dev-secret" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "match_type": "collaborator",
      "skill_overlap": 0.25,
      "complementarity": 0.75,
      "overlapping_skills_count": 3,
      "skills_count_a": 8,
      "skills_count_b": 7,
      "unique_skills_a": 8,
      "unique_skills_b": 7,
      "user_a_verified": 1,
      "user_b_verified": 1,
      "both_identity_verified": 1,
      "location_match": 0,
      "same_location": 0,
      "user_identity_verified": 0,
      "compatibility_score": 0.42,
      "viewed": 1,
      "saved": 0,
      "action_taken": 0,
      "project_accepting": 0,
      "team_openness": 0.0,
      "covered_skills_count": 0
    }
  }'
```

#### Project match (user ↔ project)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer dev-secret" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "match_type": "project",
      "skills_count_a": 10,
      "covered_skills_count": 7,
      "project_accepting": 1,
      "team_openness": 0.6,
      "user_identity_verified": 1,
      "same_location": 0,
      "compatibility_score": 0.61,
      "viewed": 0,
      "saved": 0,
      "action_taken": 0
    }
  }'
```

#### Response (both types)
```json
{
  "status": "success",
  "data": {
    "match_type": "collaborator",
    "is_relevant": false,
    "relevant_probability": 0.1823,
    "compatibility_score": 0.1823,
    "match_reasons": {
      "skill_overlap": 0.25,
      "complementarity": 0.75,
      "location_match": 0,
      "both_identity_verified": 1,
      "trust_score": 2.0,
      "overlap_ratio": 0.2
    },
    "top_model_features": [
      {"feature": "compatibility_score", "importance": 0.1832},
      {"feature": "overlapping_skills_count", "importance": 0.0941},
      {"feature": "skill_overlap", "importance": 0.0887},
      {"feature": "complementarity", "importance": 0.0821},
      {"feature": "covered_skills_count", "importance": 0.0743}
    ]
  }
}
```

---

### `POST /predict/batch` — score up to 1 000 pairs

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Authorization: Bearer dev-secret" \
  -H "Content-Type: application/json" \
  -d '{
    "pairs": [
      {
        "match_type": "collaborator",
        "skill_overlap": 0.4,
        "complementarity": 0.6,
        "overlapping_skills_count": 4,
        "skills_count_a": 9,
        "skills_count_b": 6,
        "user_a_verified": 1,
        "user_b_verified": 0,
        "both_identity_verified": 0,
        "location_match": 1,
        "same_location": 1,
        "compatibility_score": 0.55,
        "viewed": 1,
        "saved": 1,
        "action_taken": 0,
        "project_accepting": 0,
        "team_openness": 0.0,
        "covered_skills_count": 0
      },
      {
        "match_type": "project",
        "skills_count_a": 12,
        "covered_skills_count": 9,
        "project_accepting": 1,
        "team_openness": 0.75,
        "user_identity_verified": 1,
        "same_location": 0,
        "compatibility_score": 0.71
      }
    ]
  }'
```

**Python example:**
```python
import requests

pairs = [
    {
        "match_type": "collaborator",
        "skill_overlap": 0.3, "complementarity": 0.7,
        "overlapping_skills_count": 2,
        "skills_count_a": 5, "skills_count_b": 6,
        "user_a_verified": 1, "user_b_verified": 1,
        "both_identity_verified": 1,
        "location_match": 0, "same_location": 0,
        "compatibility_score": 0.35,
        "viewed": 0, "saved": 0, "action_taken": 0,
        "project_accepting": 0, "team_openness": 0.0,
        "covered_skills_count": 0,
    },
    # ... more pairs
]

resp = requests.post(
    "http://localhost:8000/predict/batch",
    json={"pairs": pairs},
    headers={"Authorization": "Bearer dev-secret"},
)
results = resp.json()["data"]
for r in results:
    print(r["match_type"], r["is_relevant"], r["compatibility_score"])
```

---

### `POST /retrain` — retrain from a new CSV

```bash
curl -X POST http://localhost:8000/retrain \
  -H "Authorization: Bearer dev-secret" \
  -F "file=@matches_all_new.csv"
```

The server retrains in place and reloads artefacts immediately. The CSV must have the same column names as the original training file.

---

## Feature Reference

These are the exact column names from `matches_all_20260515_211601.csv` used as model inputs:

| Feature | Type | Description |
|---|---|---|
| `match_type` | collaborator / project | Encoded as 0 / 1 |
| `compatibility_score` | float 0–1 | Previously assigned score |
| `viewed` | 0/1 | User opened the match |
| `saved` | 0/1 | User bookmarked the match |
| `action_taken` | 0/1 | User submitted feedback |
| `skill_overlap` | float 0–1 | Jaccard: \|A∩B\| / \|A∪B\| |
| `complementarity` | float 0–1 | 1 − skill_overlap |
| `overlapping_skills_count` | int | Raw count of shared skills |
| `skills_count_a` | int | Total skills of user A |
| `skills_count_b` | int | Total skills of user B / project |
| `covered_skills_count` | int | Skills covering project requirements |
| `team_openness` | float 0–1 | (max−current)/max team size |
| `project_accepting` | 0/1 | Project open for applications |
| `user_identity_verified` | 0/1 | User passed identity check |
| `user_a_verified` | 0/1 | User A verified |
| `user_b_verified` | 0/1 | User B verified |
| `both_identity_verified` | 0/1 | Both users verified |
| `location_match` | 0/1 | Same location string |
| `same_location` | 0/1 | Same location (derived) |
| `unique_skills_a` | int | Unique skills of user A |
| `unique_skills_b` | int | Unique skills of user B |

**Engineered at inference time** (no need to send):

| Feature | Formula |
|---|---|
| `total_skills` | `skills_count_a + skills_count_b` |
| `overlap_ratio` | `overlapping_skills_count / total_skills` |
| `trust_score` | `user_a_verified + user_b_verified` |
| `skill_gap` | `abs(skills_count_a − skills_count_b)` |
| `readiness` | `project_accepting + team_openness` |
| `local_verified` | `same_location × user_identity_verified` |

---

## Project Layout

```
cofound_v2/
├── main.py              ← FastAPI app (all endpoints)
├── requirements.txt
├── README.md
└── model/
    ├── train.py         ← training script
    ├── rf_model.pkl     ← trained RandomForestClassifier
    ├── scaler.pkl       ← MinMaxScaler (fitted on training data)
    ├── feature_cols.pkl ← ordered list of 27 feature names
    └── col_medians.pkl  ← per-column medians for inference imputation
```
