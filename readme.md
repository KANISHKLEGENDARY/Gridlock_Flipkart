# Flipkart GRiD 7.0 — Traffic Demand Prediction

![Leaderboard Score](https://img.shields.io/badge/Leaderboard%20Score-86.57%2F100-brightgreen)
![Model](https://img.shields.io/badge/Model-XGBoost-blue)
![Language](https://img.shields.io/badge/Language-Python%203.12-yellow)
![Competition](https://img.shields.io/badge/Competition-Flipkart%20GRiD%207.0-orange)

A competitive machine learning solution for Flipkart GRiD 7.0 Round 1 — predicting real-time traffic demand across geographic locations using spatiotemporal feature engineering, leak-free cross-validation, and gradient boosted trees.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Evaluation Metric](#evaluation-metric)
- [Project Architecture](#project-architecture)
- [Key Findings and Challenges](#key-findings-and-challenges)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Cross-Validation Strategy](#cross-validation-strategy)
- [Data Leakage Analysis](#data-leakage-analysis)
- [Results](#results)
- [Installation and Usage](#installation-and-usage)
- [File Structure](#file-structure)
- [Lessons Learned](#lessons-learned)

---

## Problem Statement

Cities worldwide face increasing traffic congestion that disrupts transportation and poses barriers to economic growth. This competition required building a machine learning system to predict **traffic demand** at specific geographic locations and timestamps.

Given a tabular dataset of geospatial, temporal, road infrastructure, and weather features, the task was to predict the `demand` column for 41,778 test rows and submit predictions as a CSV file.

**This is a pure ML regression competition** — no system design, no web application, no backend required for Round 1. The team that produces the highest R² on the hidden test set wins.

---

## Dataset Description

### Files

| File | Shape | Description |
|------|-------|-------------|
| `train.csv` | 77,299 × 11 | Labelled training data |
| `test.csv` | 41,778 × 10 | Unlabelled test data (no demand column) |
| `sample_submission.csv` | 5 × 2 | Submission format reference |

### Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `Index` | int64 | Unique row identifier |
| `geohash` | object | Compressed geographic coordinate string (e.g. `qp02z1`) |
| `day` | int64 | Sequential day counter in the dataset |
| `timestamp` | object | 15-minute time slot (e.g. `0:0`, `13:30`) |
| `RoadType` | object | Type of road at the location (e.g. Residential, Highway) |
| `NumberofLanes` | int64 | Number of lanes at the location |
| `LargeVehicles` | object | Whether large vehicles are permitted (`Allowed` / `Not Allowed`) |
| `Landmarks` | object | Whether a notable landmark is nearby (`Yes` / `No`) |
| `Temperature` | float64 | Temperature at the location in degrees |
| `Weather` | object | Weather condition (`Sunny`, `Rainy`, `Foggy`, `Snowy`) |
| `demand` | float64 | **Target variable** — normalised traffic demand score (0 to ~1) |

### Critical Dataset Observations

**Day distribution:**
```
Day 48 → 69,427 rows  (89.8% of training data)
Day 49 →  7,872 rows  (10.2% of training data)
```

**Timestamp distribution:**
```
Train day 49 timestamps: 0:00 → 2:45  (midnight to 3am only)
Test  day 49 timestamps: 2:15 → 13:45 (includes peak morning hours)
```

This structural mismatch between train and test timestamps was the **central challenge** of this competition and required careful feature engineering to address.

**Geohash coverage:**
```
Train geohashes: 1,249 unique locations
Test geohashes:  1,190 unique locations
Unseen in test:  10 geohashes (only 0.8%)
```

**Demand distribution:**
```
Min:    0.000001  (near-zero traffic)
Mean:   0.094     (low-moderate traffic)
Median: 0.048     (most locations have low demand)
Max:    1.000     (fully saturated road)
Skewness: 3.73    (heavily right-skewed)
```

Demand is a normalised continuous score between 0 and 1 representing what fraction of a road's capacity is being used. It is NOT a percentage, NOT a raw vehicle count.

---

## Evaluation Metric

```python
score = max(0, 100 * metrics.r2_score(actual, predicted))
```

R² (coefficient of determination) measures how well predictions explain the variance in actual demand. A score of 100 means perfect prediction. A score of 0 means the model is no better than always predicting the mean. Negative scores mean the model is worse than predicting the mean.

**Important:** R² must be computed on the **original demand scale** (not on any transformed version) to match the competition formula exactly.

---

## Project Architecture

```
Raw Data
    │
    ├── Null Imputation
    │   ├── RoadType    → mode per geohash group
    │   ├── Temperature → median per (geohash, timestamp) group
    │   └── Weather     → mode per (geohash, day) group
    │
    ├── Time Feature Extraction
    │   ├── hour, minute, time_slot (0–95)
    │   ├── day_of_week, is_weekend
    │   ├── is_peak (rush hours), is_night
    │   └── Cyclical encodings (sin/cos)
    │
    ├── Geohash Processing
    │   ├── Decode to lat/lng coordinates
    │   └── Extract prefix levels (4, 3, 2 chars)
    │
    ├── Label Encoding
    │   └── RoadType, LargeVehicles, Landmarks, Weather
    │
    ├── GroupKFold Cross-Validation (5 folds by geohash)
    │   │
    │   └── Per Fold (computed from training fold only):
    │       ├── Geo demand statistics (mean, std, max)
    │       ├── Geo × hour interaction demand
    │       ├── Geo × time_slot interaction demand
    │       ├── Day 48 reference demand (hour + slot level)
    │       ├── Geohash prefix neighbourhood demand
    │       ├── Weather target encoding
    │       └── KNN spatial demand (5 nearest neighbours)
    │
    ├── XGBoost Regressor
    │   ├── n_estimators: 5000
    │   ├── learning_rate: 0.01
    │   ├── max_depth: 8
    │   └── early_stopping_rounds: 200
    │
    └── Submission
        ├── Clip negative predictions to 0
        └── submission.csv (41,778 × 2)
```

---

## Key Findings and Challenges

### Finding 1 — Day is a sequential counter, not a calendar day

`day = 48` does not mean the 48th day of a month. It means the 48th consecutive day since data collection started. The dataset only contains 2 unique day values (48 and 49), making day-based temporal splitting meaningless.

### Finding 2 — Timestamp represents 15-minute slots

Each timestamp like `0:0`, `0:15`, `13:30` represents a 15-minute recording interval. There are 96 unique timestamps per day (24 hours × 4 slots per hour). The dataset records demand for multiple geohashes at each timestamp.

### Finding 3 — Demand is a normalised capacity utilisation score

Demand values between 0 and 1 represent what fraction of a road's maximum capacity is being used. Values slightly above 1.0 represent over-capacity conditions. It is NOT measured in vehicles per hour or any physical unit.

### Finding 4 — Critical temporal train/test mismatch

The single most important finding of this project:

```
Train contains:
  Day 48: all 96 timestamps (full day, 0:00 to 23:45)
  Day 49: only 9 timestamps (0:00 to 2:45, midnight only)

Test contains:
  Day 49: timestamps from 2:15 to 13:45 (includes peak hours 10am-1pm)
```

The model trained almost entirely on Day 48 patterns but needed to predict Day 49 peak hours. This caused predictions to underestimate demand for peak morning slots — directly reducing the leaderboard score.

### Finding 5 — Target leakage was the biggest trap

Multiple feature engineering approaches that gave excellent out-of-fold R² scores (up to 0.9975) turned out to be severely leaky. The leakage came from computing geo demand statistics from the full training set and then using those same statistics to evaluate on validation folds — where the validation geohashes had contributed their own demand to the statistics being used to predict them.

The fix was computing all demand-derived features **exclusively from the training portion of each fold**, never using validation fold rows to compute any statistics.

---

## Feature Engineering

All features marked with `[FOLD-SAFE]` are recomputed from the training portion of each fold to prevent data leakage.

### Time Features

```python
df["hour"]       = df["timestamp"].str.split(":").str[0].astype(int)
df["minute"]     = df["timestamp"].str.split(":").str[1].astype(int)
df["time_slot"]  = df["hour"] * 4 + df["minute"] // 15   # 0 to 95
df["day_of_week"] = df["day"] % 7
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
df["is_peak"]    = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
df["is_night"]   = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
```

**Cyclical encoding** prevents the model from thinking hour 23 is far from hour 0:

```python
df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
df["slot_sin"]   = np.sin(2 * np.pi * df["time_slot"] / 96)
df["slot_cos"]   = np.cos(2 * np.pi * df["time_slot"] / 96)
df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
```

### Geohash Decoding

Geohash strings like `qp02z1` are compressed GPS coordinate codes. Decoding them to latitude and longitude lets the model understand spatial proximity between locations:

```python
import pygeohash as gh

def decode_geo(g):
    try:
        lat, lng = gh.decode(g)
        return lat, lng
    except:
        return np.nan, np.nan

train["lat"], train["lng"] = zip(*train["geohash"].map(decode_geo))
```

### Geo Demand Statistics `[FOLD-SAFE]`

For each geohash location, compute historical demand statistics:

```python
gs = ref_df.groupby("geohash")["demand"].agg(["mean", "std", "max"]).reset_index()
gs.columns = ["geohash", "geo_demand_mean", "geo_demand_std", "geo_demand_max"]
```

`geo_demand_mean` alone is one of the strongest features — it captures the baseline traffic level for each location regardless of time.

### Geo × Hour Interaction `[FOLD-SAFE]`

Average demand at each location for each specific hour of the day:

```python
gh_f = ref_df.groupby(["geohash", "hour"])["demand"].mean().reset_index()
gh_f.columns = ["geohash", "hour", "geo_hour_demand"]
```

This is significantly more powerful than `geo_demand_mean` because it captures the fact that a busy location at 10am might be quiet at midnight.

### Geo × Time Slot Interaction `[FOLD-SAFE]`

Same as above but at 15-minute granularity (96 slots per day instead of 24 hours):

```python
gs_f = ref_df.groupby(["geohash", "time_slot"])["demand"].mean().reset_index()
gs_f.columns = ["geohash", "time_slot", "geo_slot_demand"]
```

### Day 48 Reference Features `[FOLD-SAFE]`

Since the test set is Day 49 peak hours, and the model has almost no Day 49 data at those hours in training, the best proxy is what each location did at the same time on Day 48:

```python
# Hour level
d48h = ref_df[ref_df["day"] == 48].groupby(["geohash", "hour"])["demand"].mean().reset_index()
d48h.columns = ["geohash", "hour", "day48_hour_demand"]

# Slot level (finer granularity)
d48s = ref_df[ref_df["day"] == 48].groupby(["geohash", "time_slot"])["demand"].mean().reset_index()
d48s.columns = ["geohash", "time_slot", "day48_slot_demand"]
```

**Why this works:** Traffic demand at 10am on a Tuesday is highly correlated with demand at 10am on the previous day at the same location. Day 48 provides the best reference for predicting Day 49 peak hours.

### Geohash Prefix Neighbourhood Features `[FOLD-SAFE]`

Geohashes that share a common prefix are physically close to each other. This creates a natural spatial hierarchy:

```
qp02z1  ← exact location
qp02    ← neighbourhood (4-char prefix)
qp0     ← district (3-char prefix)
qp      ← city (2-char prefix)
```

```python
for df in [train, test]:
    df["geo_prefix_4"] = df["geohash"].str[:4]
    df["geo_prefix_3"] = df["geohash"].str[:3]
    df["geo_prefix_2"] = df["geohash"].str[:2]

for prefix in ["geo_prefix_4", "geo_prefix_3", "geo_prefix_2"]:
    ps = ref_df.groupby(prefix)["demand"].mean().reset_index()
    ps.columns = [prefix, f"{prefix}_demand_mean"]
```

This provides multi-level spatial fallback for the 10 unseen test geohashes — instead of falling back to the global mean, they use their neighbourhood average.

### KNN Spatial Demand `[FOLD-SAFE]`

For any location, the 5 nearest locations (by lat/lng and hour) provide a weighted estimate of expected demand:

```python
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=5, weights="distance")
knn.fit(ref_df[["lat", "lng", "hour"]].values, ref_df["demand"].values)
df["knn_demand"] = knn.predict(df[["lat", "lng", "hour"]].values)
```

Distance-weighted so closer neighbours have more influence than farther ones.

### Weather Target Encoding `[FOLD-SAFE]`

Instead of label encoding Weather as arbitrary numbers (Sunny=0, Rainy=1), target encoding replaces each weather category with the average demand it historically causes:

```python
wte = ref_df.groupby("Weather")["demand"].mean()
df["weather_target_enc"] = df["Weather"].map(wte).fillna(fold_mean)
```

This encodes the actual relationship between weather and demand rather than just giving categories arbitrary numbers.

### Categorical Label Encoding

Standard label encoding for low-cardinality categorical columns. Fitted on combined train+test to prevent unseen label errors:

```python
from sklearn.preprocessing import LabelEncoder

for col in ["RoadType", "LargeVehicles", "Landmarks", "Weather"]:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col + "_enc"] = le.transform(train[col].astype(str))
    test[col  + "_enc"] = le.transform(test[col].astype(str))
```

### Null Imputation Strategy

| Column | Null Count | Strategy |
|--------|-----------|----------|
| `RoadType` | 600 | Mode per geohash group → global mode fallback |
| `Temperature` | 2,495 | Median per (geohash, timestamp) → geohash median → global median |
| `Weather` | 797 | Mode per (geohash, day) → global mode fallback |

All imputation statistics computed from train only, then applied to test.

---

## Model Architecture

### XGBoost Regressor

```python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=5000,
    learning_rate=0.01,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.05,
    reg_lambda=1.0,
    early_stopping_rounds=200,
    eval_metric="rmse",
    device="cpu",
    random_state=42
)
```

**Parameter rationale:**

| Parameter | Value | Reason |
|-----------|-------|--------|
| `n_estimators` | 5000 | High ceiling — early stopping controls actual count |
| `learning_rate` | 0.01 | Low rate — more careful learning, better generalisation |
| `max_depth` | 8 | Deeper trees to capture complex location × time interactions |
| `subsample` | 0.8 | Row subsampling — prevents overfitting |
| `colsample_bytree` | 0.8 | Feature subsampling per tree — prevents overfitting |
| `min_child_weight` | 3 | Minimum samples per leaf — prevents splits on noise |
| `reg_alpha` | 0.05 | L1 regularisation — encourages sparse feature use |
| `reg_lambda` | 1.0 | L2 regularisation — penalises large weights |
| `early_stopping_rounds` | 200 | Stop if no improvement for 200 rounds |

### Full Feature List

```python
FEATURES = [
    # Time features
    "hour", "minute", "time_slot",
    "hour_sin", "hour_cos", "slot_sin", "slot_cos", "minute_sin", "minute_cos",
    "day", "day_of_week", "is_weekend", "is_peak", "is_night",

    # Location features
    "lat", "lng",

    # Road infrastructure
    "RoadType_enc", "NumberofLanes", "LargeVehicles_enc", "Landmarks_enc",

    # Weather
    "Temperature", "weather_target_enc",

    # Geo demand statistics (fold-safe)
    "geo_demand_mean", "geo_demand_std", "geo_demand_max",

    # Geo × time interactions (fold-safe)
    "geo_hour_demand", "geo_slot_demand",

    # Day 48 reference (fold-safe)
    "day48_hour_demand", "day48_slot_demand",

    # Neighbourhood features (fold-safe)
    "geo_prefix_4_demand_mean", "geo_prefix_3_demand_mean", "geo_prefix_2_demand_mean",

    # Spatial KNN (fold-safe)
    "knn_demand",
]
```

Total features: 32

---

## Cross-Validation Strategy

### Why GroupKFold by Geohash

Standard KFold randomly splits rows — this means the same geohash location appears in both training and validation, causing the model to see the location's demand history and then predict the same location. This inflates OOF R² significantly.

GroupKFold by geohash ensures each validation fold contains geohash locations that are **completely absent** from the training fold:

```python
from sklearn.model_selection import GroupKFold

groups = train["geohash"].values
gkf    = GroupKFold(n_splits=5)

for fold, (tr_idx, val_idx) in enumerate(gkf.split(train, train["demand"], groups)):
    # val_idx contains ~250 geohashes never seen in tr_idx
```

```
Fold 1: train on geohashes 251-1249, validate on geohashes 1-250
Fold 2: train on geohashes 1-250 + 501-1249, validate on 251-500
...
```

This matches the actual test scenario where the model must predict for locations it has seen before (99.2% of test) and a few it hasn't (0.8% of test).

### Leak-Free Feature Computation

All demand-derived features are recomputed **inside each fold** using only the training portion:

```python
def compute_demand_features(target_df, ref_df):
    """
    Compute all demand-derived features for target_df
    using statistics from ref_df only.
    """
    df = target_df.copy()
    fold_mean = ref_df["demand"].mean()

    # All statistics computed from ref_df
    # Applied to target_df via merge
    ...
    return df

for fold, (tr_idx, val_idx) in enumerate(gkf.split(...)):
    tr_df  = train.iloc[tr_idx].copy()
    val_df = train.iloc[val_idx].copy()

    # val_df features use tr_df as reference — no leakage
    tr_df_feat  = compute_demand_features(tr_df,  ref_df=tr_df)
    val_df_feat = compute_demand_features(val_df, ref_df=tr_df)
    test_feat   = compute_demand_features(test,   ref_df=tr_df)
```

---

## Data Leakage Analysis

This was the most technically challenging aspect of the project. Multiple forms of leakage were identified and fixed:

### Leakage Type 1 — Geo statistics computed from full train

**Symptom:** OOF R² of 0.98, leaderboard score of 83.33 (15-point gap)

**Cause:** `geo_demand_mean` was computed from all 77,299 train rows. Validation fold rows contributed their own demand to the statistic, then the model used that statistic to predict those same rows.

**Fix:** Recompute geo statistics inside each fold using only training fold rows.

### Leakage Type 2 — KNN fitted on full train

**Symptom:** OOF inflated, leaderboard score lower than expected

**Cause:** KNN was fitted on all train data including validation rows. Predicting on those same validation rows gave artificially good KNN features.

**Fix:** Fit KNN inside each fold on training fold only, predict on validation fold.

### Leakage Type 3 — Prefix statistics from full train

**Symptom:** Subtle OOF inflation

**Cause:** Neighbourhood demand means computed from all geohashes including validation ones.

**Fix:** Compute prefix statistics inside each fold.

### Leakage Type 4 — Weather target encoding from full train

**Symptom:** Minor OOF inflation

**Cause:** Weather mean demand included validation fold demand values.

**Fix:** Compute weather target encoding inside each fold.

### Leakage Type 5 — Day48 features not separated properly

**Symptom:** OOF R² of 0.9975, clearly unrealistic

**Cause:** Day 48 demand statistics for a geohash were computed from the full train, including rows that were in the validation fold. For Day 48 rows in the validation fold, the feature `day48_hour_demand` literally contained the average of values including themselves.

**Fix:** Compute day48 statistics from training fold only inside the fold loop.

### Summary of Leakage Impact

| Version | Approach | OOF R² | Leaderboard | Gap |
|---------|----------|---------|-------------|-----|
| v1 | Leaky geo features | 0.9804 | 83.33 | 14.7 pts |
| v2 | Leak-free + day48 | 0.9117 | 77.03 | -6.3 pts |
| v3 | Restored baseline | ~0.83 | 83.33 | ~0 pts |
| v4 | Full leak-free pipeline | ~0.87 | 86.57 | ~0 pts |

The shrinking gap between OOF and leaderboard scores confirms progressively better leakage removal.

---

## Results

### Final Leaderboard Score

```
Score: 86.57 / 100
Metric: max(0, 100 * R²_score(actual, predicted))
```

### Score Progression

```
Attempt 1 (wrong target, nulls present):    -32.29   ← negative R²
Attempt 2 (log1p target, basic features):   +83.33   ← first real baseline
Attempt 3 (day48 features, over-engineered): 77.03   ← regression from leakage
Attempt 4 (restored baseline):               83.33   ← back to best
Attempt 5 (full leak-free pipeline):         86.57   ← final score ✓
```

### Fold-Level Performance (Final Model)

```
Fold 1 | 250 val geohashes | RMSE: 0.117 | stopped at ~560 iterations
Fold 2 | 250 val geohashes | RMSE: 0.135 | stopped at ~604 iterations
Fold 3 | 250 val geohashes | RMSE: 0.112 | stopped at ~680 iterations
Fold 4 | 250 val geohashes | RMSE: 0.144 | stopped at ~775 iterations
Fold 5 | 249 val geohashes | RMSE: 0.158 | stopped at ~732 iterations
```

All folds triggered early stopping (did not run to 5000 iterations) — confirming the model is genuinely converging rather than memorising.

---

## Installation and Usage

### Requirements

```bash
pip install pandas numpy scikit-learn xgboost pygeohash
```

### Full dependency list

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
pygeohash>=1.2.0
```

### Running the pipeline

```bash
# Place train.csv and test.csv in the working directory
python solution.py

# Output: submission.csv (41778 x 2)
```

### Expected output

```
Loading datasets...
Train: (77299, 11), Test: (41778, 10)
Handling nulls...
Time features...
Decoding geohashes...
Label encoding...
Prefix features...
Computing test geo features from full train...
KNN feature for test...
Training with leak-free GroupKFold...

--- Fold 1 ---
[0]    validation_0-rmse: 0.117
[500]  validation_0-rmse: 0.112
[560]  validation_0-rmse: 0.112
...

OOF R² (original scale): 0.87
Estimated leaderboard:   87.xx / 100
Shape: (41778, 2)
Saved submission.csv!
```

---

## File Structure

```
flipkart-grid-traffic-demand/
│
├── solution.py              ← Complete pipeline (load → preprocess → train → submit)
├── train.csv                ← Training data (77,299 × 11)
├── test.csv                 ← Test data (41,778 × 10)
├── sample_submission.csv    ← Submission format reference
├── submission.csv           ← Final predictions (generated by solution.py)
└── README.md                ← This file
```

---

## Lessons Learned

### 1. Always understand your data before modelling

The `day` column appeared to be a temporal feature but only had 2 unique values. The `timestamp` column had a structural mismatch between train and test. Understanding these facts before building any model saved hours of debugging later.

### 2. OOF score must match leaderboard score

A 15-point gap between OOF (98.04) and leaderboard (83.33) is not a model quality problem — it is a leakage problem. When OOF is too optimistic, stop adding features and start removing leakage.

### 3. Target leakage is subtle in aggregation features

Computing `geo_demand_mean` sounds harmless — it is just an average. But when that average includes the rows you are trying to predict, it is leakage. The fix is always: compute statistics from the training portion of each fold, never from the full dataset.

### 4. Evaluation metric must match competition formula exactly

The competition uses `r2_score(actual, predicted)` on the original demand scale. Computing R² on a log-transformed or Box-Cox-transformed scale gives a completely different number that does not reflect your real leaderboard position. Always evaluate in the same space the competition evaluates.

### 5. Temporal structure requires temporal validation

Standard KFold is wrong for time-structured data. GroupKFold by geohash was the correct choice here because the actual test challenge was predicting demand at known locations at unseen timestamps — not predicting demand at completely new locations.

### 6. Simple models on good features beat complex models on bad features

XGBoost with 32 well-engineered features outperformed every approach tried with raw features and complex architectures. Feature engineering contributed more to the score than any model hyperparameter change.

### 7. Never overwrite your best submission

Every new approach should be saved as a new file (submission_v2.csv, submission_v3.csv). Submit new versions only after confirming the local estimated score exceeds the current best leaderboard score.

---

## Future Improvements

```
1. Add LightGBM in parallel → ensemble XGB + LGB (expected +1-2 points)
2. Optuna hyperparameter search → automated optimal params
3. Better temporal validation → train on hours 0-2, validate on hours 3+
   (mirrors actual test scenario more accurately)
4. Feature importance analysis → drop low-importance features
5. CatBoost as third ensemble model → handles categoricals natively
6. Pseudo-labelling → use high-confidence test predictions as extra training data
```

---

## Author

**Kanishk Garg**
BTech AI/ML, NIIT University
[LinkedIn](https://www.linkedin.com/in/kanishk-garg-7b120936b/) | [GitHub](https://github.com/KANISHKLEGENDARY)

---

*Built for Flipkart GRiD 7.0 — Round 1 Machine Learning Competition*