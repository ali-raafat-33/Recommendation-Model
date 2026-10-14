# Co-Found — ML Team Documentation

> **Audience:** ML engineers working on the recommendation model.
> This document covers everything you need: the database schema that backs the system, every API endpoint you interact with, what to send, what you get back, and the full training cycle from data pull to pushing recommendations live.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Authentication](#2-authentication)
3. [Database Schema](#3-database-schema)
   - [matches](#matches)
   - [match_feedback](#match_feedback)
   - [users (relevant columns)](#users-relevant-columns)
   - [user_skills](#user_skills)
   - [projects (relevant columns)](#projects-relevant-columns)
   - [project_skills](#project_skills)
4. [Training Data — Features & Labels](#4-training-data--features--labels)
   - [Collaborator model features](#collaborator-model-features)
   - [Project model features](#project-model-features)
   - [Label columns](#label-columns)
5. [ML API Endpoints](#5-ml-api-endpoints)
   - [GET /ml/dataset/stats](#get-mldatasetstats)
   - [POST /ml/dataset/generate](#post-mldatasetgenerate)
   - [GET /ml/dataset/export](#get-mldatasetexport)
   - [POST /ml/matches/ingest](#post-mlmatchesingest)
6. [User-Facing Match Endpoints](#6-user-facing-match-endpoints)s
   - [GET /matches](#get-matches)
   - [PATCH /matches/{id}/view](#patch-matchesidview)
   - [PATCH /matches/{id}/save](#patch-matchesidsave)
   - [POST /matches/{id}/feedback](#post-matchesidfeedback)
7. [The Full Training Cycle](#7-the-full-training-cycle)
8. [The Feedback Loop](#8-the-feedback-loop)
9. [Scoring Weights Reference](#9-scoring-weights-reference)
10. [Common Mistakes & Edge Cases](#10-common-mistakes--edge-cases)

---

## 1. System Overview

Co-Found has two match types:

- **Collaborator** — user-to-user matching. A user is recommended another user as a potential co-founder or team member.
- **Project** — user-to-project matching. A user is recommended a project to join.

The ML model is responsible for two things:

1. **Scoring** — computing a `compatibility_score` (0.0–1.0) for each user+target pair and populating the `match_reasons` feature breakdown.
2. **Delivery** — pushing scored pairs back to the platform via the ingest API so end users see them.

The platform handles everything else: displaying matches to users, collecting their feedback (view, save, explicit rating), and making that feedback available for your next training cycle via the export API.

```
Your model                  Platform                   End users
──────────                  ────────                   ─────────
                            existing users/projects
GET  /ml/dataset/export  ←  match + feedback data
Train model
POST /ml/matches/ingest  →  writes to matches table →  GET /matches
                                                        view / save / feedback
GET  /ml/dataset/export  ←  new feedback as labels  ←
```

---

## 2. Authentication

All `/ml/*` endpoints use a **shared service secret**, separate from user tokens.

```
Authorization: Bearer <ML_SERVICE_SECRET>
```

You should have this secret from the platform team. It maps to the `ML_SERVICE_SECRET` environment variable on the server. If you get a `401`, contact the platform team — the secret may have been rotated.

Do not use a user Sanctum token for these endpoints. They are different auth systems.

---

## 3. Database Schema

You do not query the database directly — you use the API. But understanding the schema tells you exactly what raw signals exist and how they become the feature columns you receive in the export.

### `matches`

The central table. One row = one match recommendation for one user.

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | Primary key |
| `user_id` | UUID | The user who receives this recommendation |
| `matched_user_id` | UUID \| null | Populated when `match_type = collaborator` |
| `matched_project_id` | UUID \| null | Populated when `match_type = project` |
| `match_type` | enum | `collaborator` or `project` |
| `compatibility_score` | decimal(3,2) | **Your model writes this.** Range 0.00–1.00 |
| `match_reasons` | JSON | **Your model writes this.** Feature breakdown object (see below) |
| `viewed` | boolean | Set by platform when user opens the match |
| `viewed_at` | timestamp \| null | First view timestamp |
| `saved` | boolean | User explicitly saved this match |
| `action_taken` | boolean | Set to `true` when user submits feedback |
| `expires_at` | timestamp \| null | After this date the match is no longer shown |
| `created_at` | timestamp | When the record was created |

**What you write:** `compatibility_score`, `match_reasons`, `expires_at`, `user_id`, `match_type`, `matched_user_id` / `matched_project_id`.

**What you read back (as training signals):** `viewed`, `saved`, `action_taken` — these are implicit engagement signals even before a user submits explicit feedback.

**Upsert behaviour:** if you push a match for a user+target pair that already has a non-expired record, the platform updates `compatibility_score`, `match_reasons`, and `expires_at` in place. It does not reset `viewed`, `saved`, or `action_taken`.

---

### `match_feedback`

Explicit user ratings. One row per user per match — the unique constraint on `(match_id, user_id)` enforces this.

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | Primary key |
| `match_id` | UUID | FK → matches.id |
| `user_id` | UUID | The user who gave the feedback |
| `feedback_type` | enum | `relevant`, `not_relevant`, `already_connected`, `not_interested` |
| `created_at` | timestamp | When feedback was submitted |

**Signal interpretation for training:**

| `feedback_type` | Meaning | Suggested label |
|---|---|---|
| `relevant` | User explicitly says "good match" | Positive (1) |
| `not_relevant` | User explicitly says "bad match" | Negative (0) |
| `already_connected` | Match was correct but already known | Treat as positive, but flag — could indicate data staleness |
| `not_interested` | User passed for reasons unrelated to match quality | Ambiguous — consider excluding from binary classification |

The export API converts these to `label_relevant` and `label_not_relevant` boolean columns automatically. The raw `feedback_type` string is also included so you can apply your own weighting.

---

### `users` (relevant columns)

You do not receive full user records. The export surfaces the signals that matter:

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | Referenced by `matches.user_id` and `matches.matched_user_id` |
| `location` | string \| null | Free-text city/country. Used for `location_match` feature |
| `identity_verified` | boolean | Whether the user passed identity verification |
| `identity_verification_level` | enum \| null | `none`, `basic`, `advanced` |
| `account_status` | enum | `active` users only appear in matches |

`identity_verified = true` with level `advanced` is the strongest trust signal. Both users being verified is computed as `both_identity_verified` in the export.

---

### `user_skills`

Each user can have multiple skill rows. These are the primary feature source for skill-based matching.

| Column | Type | Notes |
|---|---|---|
| `user_id` | UUID | FK → users |
| `skill_name` | string | Free-text, max 100 chars (e.g. `"React"`, `"Python"`) |
| `proficiency_level` | int | 1–5 scale |
| `years_experience` | decimal(3,1) \| null | e.g. `3.5` |
| `is_approved` | boolean | Only approved skills are used in matching |

In the export, the raw skill lists are collapsed to scalar counts: `skills_count_a`, `skills_count_b`, `overlapping_skills_count`. If you want the raw skill names for embedding-based approaches, you would need direct DB read access — contact the platform team.

---

### `projects` (relevant columns)

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | Referenced by `matches.matched_project_id` |
| `status` | enum | `planning`, `active`, `on_hold`, `completed`, `cancelled` |
| `is_accepting_applications` | boolean | Whether the project is open for new members |
| `current_team_size` | int | Current number of team members |
| `team_size_max` | int \| null | Maximum team size. Used to compute `team_openness` |

`team_openness` is computed as `(team_size_max - current_team_size) / team_size_max`. A value of `1.0` means the team is empty; `0.0` means full.

---

### `project_skills`

Each project can require multiple skills. Mirrored structure to `user_skills`.

| Column | Type | Notes |
|---|---|---|
| `project_id` | UUID | FK → projects |
| `skill_name` | string | Matched against `user_skills.skill_name` |
| `proficiency_required` | int | 1–5 scale |
| `is_required` | boolean | Only required skills are used in the `skill_coverage` feature |

`skill_coverage` is the fraction of a project's required skills that the user possesses: `count(intersection) / count(project_required_skills)`.

---

## 4. Training Data — Features & Labels

The export endpoint returns one flat row per match. Features vary by `match_type`.

### Collaborator model features

These features are present when `match_type = collaborator`.

| Feature column | Type | Description |
|---|---|---|
| `skill_overlap` | float 0–1 | Jaccard similarity of skill sets: `|A ∩ B| / |A ∪ B|` |
| `complementarity` | float 0–1 | Inverse of overlap: `(|A ∪ B| - |A ∩ B|) / |A ∪ B|`. High = different but non-redundant skill sets |
| `overlapping_skills_count` | int | Raw count of shared skills |
| `skills_count_a` | int | Total skills of the user receiving the recommendation |
| `skills_count_b` | int | Total skills of the matched user |
| `location_match` | 0 or 1 | 1 if both users have the same `location` string |
| `both_identity_verified` | 0 or 1 | 1 if both users have `identity_verified = true` |
| `user_a_verified` | 0 or 1 | Identity verification status of the recommending user |
| `user_b_verified` | 0 or 1 | Identity verification status of the matched user |
| `same_location` | 0 or 1 | Derived from live user data at export time (same as `location_match`) |

> **Note on `skill_overlap` vs `complementarity`:** These are not independent — they sum to 1.0. Include both in your feature set but be aware of the multicollinearity if you are using linear models. For tree-based models this is not an issue.

---

### Project model features

These features are present when `match_type = project`.

| Feature column | Type | Description |
|---|---|---|
| `skill_coverage` | float 0–1 | Fraction of the project's required skills the user has |
| `covered_skills_count` | int | Raw count of matched required skills |
| `required_skills_count` | int | Total required skills on the project |
| `user_skills_count` | int | Total skills of the user |
| `team_openness` | float 0–1 | `(max - current) / max`. 1.0 = empty team, 0.0 = full |
| `project_accepting` | 0 or 1 | Whether `is_accepting_applications = true` |
| `user_identity_verified` | 0 or 1 | Identity verification status of the user |
| `same_location` | 0 or 1 | Currently always 0 — location matching is not implemented for project matches |

---

### Label columns

These are present on every row regardless of match type.

| Label column | Type | Description |
|---|---|---|
| `compatibility_score` | float 0–1 | The score your model previously wrote (or the synthetic ground truth). Use as regression target |
| `feedback_type` | string | Raw feedback value. Empty string `""` if no feedback yet |
| `label_relevant` | 0 or 1 | 1 when `feedback_type = relevant`. Binary classification target |
| `label_not_relevant` | 0 or 1 | 1 when `feedback_type = not_relevant` |
| `viewed` | 0 or 1 | Implicit positive signal — user opened the match |
| `saved` | 0 or 1 | Stronger implicit positive signal — user bookmarked it |
| `action_taken` | 0 or 1 | Whether any explicit action was taken (feedback submitted) |

**For regression training:** use `compatibility_score` as the target. Works best on synthetic data from the generate endpoint where ground truth scores are computed from known features.

**For classification training:** use `label_relevant` as the target. This only has signal on rows where `feedback_type` is non-empty. Use the `with_feedback_only=true` filter when training classifiers to avoid training on empty labels.

---

## 5. ML API Endpoints

Base URL: `https://api.cofound.io/api/v1`

All requests require: `Authorization: Bearer <ML_SERVICE_SECRET>`

---

### GET /ml/dataset/stats

**Purpose:** Check the current state of the dataset before deciding what to do next. Call this before every training run.

**Request:** No parameters.

**Response:**

```json
{
  "status": "success",
  "data": {
    "total_matches": 1842,
    "by_type": {
      "collaborator": 1104,
      "project": 738
    },
    "total_feedback": 612,
    "feedback_rate": 0.3323,
    "feedback_distribution": {
      "relevant": 287,
      "not_relevant": 198,
      "already_connected": 74,
      "not_interested": 53
    },
    "score_stats": {
      "avg_score": 0.5841,
      "min_score": 0.01,
      "max_score": 0.9700
    }
  }
}
```

**What to look for:**

- `feedback_rate` below ~0.15 means you do not have enough labelled data for meaningful classification training. Consider generating synthetic data or waiting for more organic feedback.
- `by_type` imbalance is normal — collaborator matches typically outnumber project matches.
- `score_stats.avg_score` significantly higher than 0.5 may indicate score inflation; significantly lower may indicate the scoring weights need review.

---

### POST /ml/dataset/generate

**Purpose:** Generate a synthetic labelled dataset when real user feedback is insufficient or when you want to bootstrap training from scratch. Uses the same feature computation and scoring weights as the production model to create internally consistent synthetic ground truth.

**When to use it:** Before initial model training, or when `feedback_rate` from `/stats` is too low to train on.

**When not to use it:** Do not run with `fresh: true` against production data. Use it only on a staging environment, or if you are intentionally resetting synthetic data.

**Request body (all fields optional):**

```json
{
  "users": 100,
  "projects": 40,
  "collaborator_pairs": 400,
  "project_pairs": 300,
  "fresh": false
}
```

| Field | Default | Notes |
|---|---|---|
| `users` | 100 | Synthetic user profiles to create. Range 10–500 |
| `projects` | 40 | Synthetic projects to create. Range 5–200 |
| `collaborator_pairs` | 400 | Collaborator match rows to generate. Range 10–5000 |
| `project_pairs` | 300 | Project match rows to generate. Range 10–5000 |
| `fresh` | false | If `true`, wipes all existing match and feedback data first. **Destructive.** |

**Response:**

```json
{
  "status": "success",
  "message": "Training dataset generated successfully.",
  "data": {
    "users": 100,
    "projects": 40,
    "collaborator_matches": 400,
    "project_matches": 300
  }
}
```

**Notes:**

- Generation runs synchronously. For the maximum sizes (5000+5000 pairs) expect 30–60 seconds. The API will not time out, but you should use a generous HTTP timeout on your client.
- The synthetic scores are computed from the same feature weights documented in [Section 8](#8-scoring-weights-reference). They are not random — they are derived from the generated skill/location/verification data, so the features and labels are internally consistent training pairs.
- Simulated feedback is attached to ~60% of generated matches with a score-correlated distribution: high-scoring matches are more likely to receive `relevant` feedback.

---

### GET /ml/dataset/export

**Purpose:** Pull the full training dataset. This is your primary data source for both training and evaluation.

**Query parameters:**

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `format` | `csv` \| `json` | `json` | CSV streams as a file download. JSON returns inline |
| `type` | `collaborator` \| `project` | _(all)_ | Filter to a single match type. Omit to get both |
| `min_score` | float 0–1 | `0` | Exclude rows below this score threshold |
| `with_feedback_only` | boolean | `false` | Only return rows that have user feedback |

**Example requests:**

```
# Full dataset as JSON
GET /ml/dataset/export

# Collaborator matches only, CSV download
GET /ml/dataset/export?type=collaborator&format=csv

# Labelled rows only — best for classification training
GET /ml/dataset/export?with_feedback_only=true

# High-confidence rows for regression training
GET /ml/dataset/export?min_score=0.6
```

**JSON response structure:**

```json
{
  "status": "success",
  "meta": {
    "total": 1842,
    "type": "all",
    "min_score": 0,
    "with_feedback_only": false
  },
  "data": [
    {
      "id": "uuid",
      "match_type": "collaborator",
      "compatibility_score": 0.7823,
      "viewed": 1,
      "saved": 0,
      "action_taken": 1,
      "feedback_type": "relevant",
      "label_relevant": 1,
      "label_not_relevant": 0,
      "user_identity_verified": 1,
      "same_location": 0,
      "skill_overlap": 0.333,
      "complementarity": 0.667,
      "overlapping_skills_count": 2,
      "skills_count_a": 5,
      "skills_count_b": 7,
      "location_match": 0,
      "both_identity_verified": 1,
      "user_a_verified": 1,
      "user_b_verified": 1
    }
  ]
}
```

**CSV format:** Same columns as the JSON `data` array, one row per match, standard RFC 4180 CSV. The header row is included. Recommended for pandas/numpy ingestion:

```python
import pandas as pd
import requests

resp = requests.get(
    "https://api.cofound.io/api/v1/ml/dataset/export",
    params={"format": "csv", "with_feedback_only": "true"},
    headers={"Authorization": f"Bearer {ML_SERVICE_SECRET}"},
    stream=True,
)
df = pd.read_csv(resp.raw)
```

**Rows with no feedback (`feedback_type = ""`):** These rows have `label_relevant = 0` and `label_not_relevant = 0`. This is not a negative signal — it means the label is unknown. Do not use them as negative examples in classification training. Either filter them out with `with_feedback_only=true` or handle them as unlabelled samples.

---

### POST /ml/matches/ingest

**Purpose:** Push your model's scored match pairs back to the platform. This is how your recommendations become visible to end users.

**Behaviour:**

- If a non-expired match already exists for the same `(user_id, match_type, matched_user_id / matched_project_id)` triple, it is **updated**: `compatibility_score`, `match_reasons`, and `expires_at` are replaced. `viewed`, `saved`, `action_taken` are left untouched.
- If no such match exists, a new record is **created**.
- Up to **1,000 records per request**. Batch larger inference runs across multiple calls.
- The entire batch is wrapped in a database transaction — if any record fails validation, nothing is written.

**Request body:**

```json
{
  "matches": [
    {
      "user_id": "uuid-of-the-user-receiving-the-recommendation",
      "match_type": "collaborator",
      "matched_user_id": "uuid-of-the-recommended-user",
      "compatibility_score": 0.87,
      "match_reasons": {
        "skill_overlap": 0.60,
        "complementarity": 0.80,
        "location_match": 1,
        "both_identity_verified": 1
      },
      "expires_at": "2026-08-01T00:00:00Z"
    },
    {
      "user_id": "uuid-of-the-user",
      "match_type": "project",
      "matched_project_id": "uuid-of-the-project",
      "compatibility_score": 0.74,
      "match_reasons": {
        "skill_coverage": 0.75,
        "team_openness": 0.50,
        "project_accepting": 1,
        "user_identity_verified": 1
      },
      "expires_at": "2026-08-01T00:00:00Z"
    }
  ]
}
```

**Field rules:**

| Field | Required | Notes |
|---|---|---|
| `user_id` | Yes | Must be a valid, existing user UUID |
| `match_type` | Yes | `collaborator` or `project` |
| `matched_user_id` | Yes if `match_type = collaborator` | Must be a valid, existing user UUID. Do not send the same UUID as `user_id` |
| `matched_project_id` | Yes if `match_type = project` | Must be a valid, existing project UUID |
| `compatibility_score` | Yes | Float 0.0–1.0 |
| `match_reasons` | Yes | Object. Keys shown to users to explain the match. Use the same keys as the feature columns — see below |
| `expires_at` | Yes | ISO 8601 datetime. Must be in the future. Recommended: 30–60 days out |

**`match_reasons` keys and what they display to users:**

The platform surfaces `match_reasons` directly in the UI. Use these exact keys so the frontend can render them with the correct labels:

For collaborator matches: `skill_overlap`, `complementarity`, `location_match`, `both_identity_verified`

For project matches: `skill_coverage`, `team_openness`, `project_accepting`, `user_identity_verified`

**Response:**

```json
{
  "status": "success",
  "message": "Matches ingested successfully.",
  "data": {
    "created": 312,
    "updated": 88
  }
}
```

**Batching strategy for large inference runs:**

```python
BATCH_SIZE = 1000

for i in range(0, len(scored_pairs), BATCH_SIZE):
    batch = scored_pairs[i : i + BATCH_SIZE]
    resp = requests.post(
        "https://api.cofound.io/api/v1/ml/matches/ingest",
        json={"matches": batch},
        headers={"Authorization": f"Bearer {ML_SERVICE_SECRET}"},
    )
    resp.raise_for_status()
    result = resp.json()["data"]
    print(f"Batch {i // BATCH_SIZE + 1}: {result['created']} created, {result['updated']} updated")
```

---

## 6. User-Facing Match Endpoints

These endpoints are **not called by the ML service**. They are called by the platform frontend on behalf of end users. They are documented here because they are the mechanism through which user behaviour becomes training signal — understanding them tells you exactly what data you can expect in your next export.

Auth for these endpoints uses a user Sanctum token, not the ML service secret. You cannot and should not call them from your model infrastructure.

---

### GET /matches

**Called by:** Frontend, on behalf of a logged-in user.

**What it does:** Returns the authenticated user's match queue — the list of `collaborator` and `project` matches currently visible to them, sorted by `compatibility_score` descending by default.

**Why it matters to you:** This is where your ingest output surfaces. After `POST /ml/matches/ingest`, the matches you pushed appear here immediately. The user sees `compatibility_score` and the `match_reasons` breakdown you provided. If your scores are poor or `match_reasons` keys are missing, this is where it becomes visible.

**Engagement signals generated:** Every time this endpoint is called, the frontend will subsequently call `/view` on matches the user opens. The rate at which users open matches vs scroll past them is an implicit quality signal you can read back via `viewed` in the export.

---

### PATCH /matches/{id}/view

**Called by:** Frontend, when a user opens a match card to read the full profile.

**What it does:** Sets `viewed = true` and records `viewed_at` on the match row. Idempotent — calling it multiple times only sets the timestamp once.

**Why it matters to you:** `viewed = 1` in your export means the user thought the match was worth opening. Combined with `saved` and `action_taken`, it gives you an implicit engagement ladder even for users who never submit explicit feedback:

| Signal | Meaning |
|---|---|
| `viewed = 0` | User never opened the match — either not shown yet, or skipped immediately |
| `viewed = 1, saved = 0, action_taken = 0` | User looked but took no further action — weak implicit negative or indifference |
| `viewed = 1, saved = 1` | User bookmarked the match — strong implicit positive |
| `viewed = 1, action_taken = 1` | User submitted explicit feedback — strongest signal, check `feedback_type` |

Do not treat `viewed = 0` rows as negatives. A match that has never been viewed has an unknown label, not a negative one. The user may simply not have scrolled that far in their queue.

---

### PATCH /matches/{id}/save

**Called by:** Frontend, when a user bookmarks a match.

**What it does:** Toggles `saved` on the match. Calling it again unsaves.

**Why it matters to you:** `saved = 1` is your strongest implicit positive signal — the user explicitly chose to keep this match for later. In the export, rows where `saved = 1` are high-confidence positives even without explicit feedback. If your model is struggling with sparse labelled data, `saved` rows are worth up-weighting as pseudo-labels.

Note that `saved` can toggle back to `0` if the user changes their mind. The export reflects the current state, not a history of saves/unsaves. A row with `saved = 0` may have been saved and then unsaved — you cannot distinguish this from a match that was never saved.

---

### POST /matches/{id}/feedback

**Called by:** Frontend, when a user explicitly rates a match.

**What it does:** Creates a `match_feedback` row with one of four values: `relevant`, `not_relevant`, `already_connected`, `not_interested`. Also sets `action_taken = true` on the match. One submission per user per match — the API returns 409 on duplicate attempts.

**Why it matters to you:** This is your primary source of explicit training labels. Every feedback submission becomes a row in `match_feedback` and appears in the `feedback_type` column of your export.

The four values and how to treat them:

| `feedback_type` | Training interpretation |
|---|---|
| `relevant` | Explicit positive. Use as `label_relevant = 1`. High-confidence signal. |
| `not_relevant` | Explicit negative. Use as `label_not_relevant = 1`. High-confidence signal. |
| `already_connected` | The match was topically correct but the pair already knew each other. Treat as a positive for quality purposes, but flag it — high rates here suggest your candidate pool may be too narrow or that connection data is stale. |
| `not_interested` | The user passed for personal, timing, or availability reasons — not a quality judgement. Exclude from binary classification or treat as unlabelled. Do not treat as `label_not_relevant`. |

The feedback loop is fully automatic — once a user submits feedback here, it will appear in your next `GET /ml/dataset/export` call with no action required on your side.

---

## 7. The Full Training Cycle

This is the recommended flow for a full training and deployment cycle.

### Step 1 — Assess the dataset

```
GET /ml/dataset/stats
```

Check `total_matches`, `feedback_rate`, and `feedback_distribution`. If `feedback_rate` is below 0.15 and you need labelled data, go to Step 2. Otherwise skip to Step 3.

### Step 2 — Generate synthetic data (if needed)

```
POST /ml/dataset/generate
{ "collaborator_pairs": 500, "project_pairs": 300 }
```

Only on staging, or when bootstrapping. Do not use `fresh: true` in production unless you have explicitly confirmed with the platform team.

### Step 3 — Export training data

```
# For regression (uses all rows, feedback or not)
GET /ml/dataset/export?format=csv

# For classification (labelled rows only)
GET /ml/dataset/export?format=csv&with_feedback_only=true

# Train separate models per type
GET /ml/dataset/export?format=csv&type=collaborator
GET /ml/dataset/export?format=csv&type=project
```

### Step 4 — Train your model externally

The platform has no involvement here. Train, validate, and evaluate outside the platform.

Keep in mind:
- `compatibility_score` is the regression target — your model should output a value in 0.0–1.0.
- `label_relevant` is the binary classification target.
- The collaborator and project models should be trained separately — they have different feature sets and different user intents.
- Rows where `feedback_type = ""` have unknown labels. Do not treat them as negatives.

### Step 5 — Push scored matches

```
POST /ml/matches/ingest
{ "matches": [ ...up to 1000 scored pairs... ] }
```

After this call, users will see your new scores at `GET /matches` immediately. You should push matches for every active user — the platform handles surfacing only the highest-scoring unexpired matches in the user's queue.

### Step 6 — Monitor implicit engagement

Repeat `GET /ml/dataset/stats` after 24–48 hours. Check whether `viewed`, `saved`, and `action_taken` rates are healthy. Rising `feedback_rate` over time means the engagement loop is working and your next training cycle will have more labelled data.

---

## 8. The Feedback Loop

Understanding how feedback flows through the system matters for training data quality.

```
You push a match via POST /ml/matches/ingest
        ↓
User sees it at GET /matches
        ↓
User views it   → match.viewed = true    (implicit signal)
User saves it   → match.saved = true     (implicit signal, stronger)
User submits    → match_feedback row     (explicit label)
feedback           with feedback_type
        ↓
Next GET /ml/dataset/export
  └── feedback_type column populated
  └── label_relevant / label_not_relevant computed
```

**Key points:**

- `viewed` and `saved` update automatically — you do not need to do anything to capture them.
- Explicit feedback (`feedback_type`) requires the user to actively rate the match in the UI. Not all users do this. Expect feedback rates of 15–40% depending on engagement.
- `already_connected` feedback means your model identified the right person/project but the data was stale. If you see this at >10% it is worth investigating whether your training data includes recently formed connections.
- `not_interested` is ambiguous — it is not a signal that the match was poor quality, only that the user does not want to act on it right now. Use caution including it as a negative label.

---

## 9. Scoring Weights Reference

These are the weights currently used by the synthetic data generator. If you train a model that outperforms these, the platform team will update them. In the meantime, these are the ground truth weights that labelled your synthetic training data.

**Collaborator score:**

```
score = (complementarity        × 0.35)
      + (skill_overlap          × 0.25)
      + (location_match         × 0.15)
      + (both_identity_verified × 0.15)
      + (min(skills_count_a / 8, 1.0) × 0.05)
      + (min(skills_count_b / 8, 1.0) × 0.05)
```

**Project score:**

```
score = (skill_coverage         × 0.50)
      + (team_openness          × 0.20)
      + (project_accepting      × 0.15)
      + (user_identity_verified × 0.15)
```

Scores are clamped to the range [0.01, 1.00] and rounded to 4 decimal places.

These weights reflect the product team's current hypothesis about what drives a good match. Your model should treat them as a baseline to improve on, not a constraint to reproduce.

---

## 10. Common Mistakes & Edge Cases

**Sending `matched_user_id` for a project match (or vice versa)**
The API validates the type/target combination. A `collaborator` match must have `matched_user_id`; a `project` match must have `matched_project_id`. Sending the wrong one returns a `422`.

**Sending `user_id = matched_user_id`**
The API does not currently reject this but it produces a self-match that will never be meaningful. Filter these out before calling ingest.

**`expires_at` in the past**
The API rejects this with a `422`. Matches with past expiry will not be shown to users even if you somehow write them. Always set `expires_at` to at least 7 days in the future.

**Export rows where `feedback_type = ""` treated as negatives**
These rows have unknown labels, not negative labels. Using them as `label_relevant = 0` will add noise to your classifier. Either exclude them with `with_feedback_only=true` or apply a semi-supervised approach.

**Score values outside 0–1**
The API will reject scores outside this range with a `422`. Clamp your model output before calling ingest.

**`match_reasons` missing keys**
The platform frontend uses specific keys from `match_reasons` to display explanations to users. If you omit or rename a key, the feature will simply not be displayed — the ingest call will still succeed, but users will see an incomplete explanation. Use the keys documented in the ingest section.

**Large inference batches timing out**
The ingest endpoint processes up to 1,000 records per call in a single transaction. For inference runs covering tens of thousands of users, batch into 1,000-record chunks as shown in the batching example. Do not try to send everything in one call.

**`fresh: true` on generate in production**
This deletes all existing match and feedback data including real user feedback. Only use it on staging or at the explicit request of the platform team.
