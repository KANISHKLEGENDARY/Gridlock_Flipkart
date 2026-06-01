import numpy as np
import pandas as pd
import pygeohash as gh
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import os
import warnings
warnings.filterwarnings('ignore')

print("Loading datasets...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(f"Original Train shape: {train.shape}, Test shape: {test.shape}")

# Handling null values
print("Handling null values...")
road_mode = train.groupby("geohash")["RoadType"].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
global_road_mode = train["RoadType"].mode()[0]

def fill_roadtype(row, mode_map, global_mode):
    if pd.isna(row["RoadType"]):
        return mode_map.get(row["geohash"], global_mode)
    return row["RoadType"]

train["RoadType"] = train.apply(fill_roadtype, axis=1, mode_map=road_mode, global_mode=global_road_mode)
test["RoadType"] = test.apply(fill_roadtype, axis=1, mode_map=road_mode, global_mode=global_road_mode)

temp_map = train.groupby(["geohash", "timestamp"])["Temperature"].median()
geo_temp_map = train.groupby("geohash")["Temperature"].median()
global_temp = train["Temperature"].median()

def fill_temp(row):
    if pd.isna(row["Temperature"]):
        val = temp_map.get((row["geohash"], row["timestamp"]), np.nan)
        if pd.isna(val):
            val = geo_temp_map.get(row["geohash"], global_temp)
        return val
    return row["Temperature"]

train["Temperature"] = train.apply(fill_temp, axis=1)
test["Temperature"] = test.apply(fill_temp, axis=1)

weather_map = train.groupby(["geohash", "day"])["Weather"].agg(
    lambda x: x.mode()[0] if not x.mode().empty else np.nan
)
global_weather = train["Weather"].mode()[0]

def fill_weather(row):
    if pd.isna(row["Weather"]):
        val = weather_map.get((row["geohash"], row["day"]), np.nan)
        if pd.isna(val):
            val = global_weather
        return val
    return row["Weather"]

train["Weather"] = train.apply(fill_weather, axis=1)
test["Weather"] = test.apply(fill_weather, axis=1)

# Time features with bug fixed!
print("Extracting time features (bug fixed)...")
for df in [train, test]:
    df["hour"] = df["timestamp"].str.split(":").str[0].astype(int)
    df["minute"] = df["timestamp"].str.split(":").str[1].astype(int)
    # BUG FIX: divide by 15 to get time_slot in [0, 95]
    df["time_slot"] = df["hour"] * 4 + df["minute"] // 15
    df["day_of_week"] = df["day"] % 7
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["slot_sin"] = np.sin(2 * np.pi * df["time_slot"] / 96)
    df["slot_cos"] = np.cos(2 * np.pi * df["time_slot"] / 96)

# Geohash decoding
print("Decoding geohashes...")
def decode_geo(g):
    try:
        lat, lng = gh.decode(g)
        return lat, lng
    except:
        return np.nan, np.nan

train["lat"], train["lng"] = zip(*train["geohash"].map(decode_geo))
test["lat"],  test["lng"]  = zip(*test["geohash"].map(decode_geo))

geo_stats = train.groupby("geohash")["demand"].agg(["mean","median","std","max"]).reset_index()
geo_stats.columns = ["geohash", "geo_demand_mean", "geo_demand_median", "geo_demand_std", "geo_demand_max"]

train = train.merge(geo_stats, on="geohash", how="left")
test  = test.merge(geo_stats,  on="geohash", how="left")

for col in ["geo_demand_mean", "geo_demand_median", "geo_demand_std", "geo_demand_max"]:
    test[col] = test[col].fillna(train["demand"].mean())

# geo x hour and geo x time_slot
print("Geo demand by hour and slot features...")
geo_hour = train.groupby(["geohash", "hour"])["demand"].mean().reset_index()
geo_hour.columns = ["geohash", "hour", "geo_hour_demand"]
train = train.merge(geo_hour, on=["geohash", "hour"], how="left")
test  = test.merge(geo_hour,  on=["geohash", "hour"], how="left")
train["geo_hour_demand"] = train["geo_hour_demand"].fillna(train["geo_demand_mean"])
test["geo_hour_demand"]  = test["geo_hour_demand"].fillna(test["geo_demand_mean"])

geo_slot = train.groupby(["geohash", "time_slot"])["demand"].mean().reset_index()
geo_slot.columns = ["geohash", "time_slot", "geo_slot_demand"]
train = train.merge(geo_slot, on=["geohash", "time_slot"], how="left")
test  = test.merge(geo_slot,  on=["geohash", "time_slot"], how="left")
train["geo_slot_demand"] = train["geo_slot_demand"].fillna(train["geo_demand_mean"])
test["geo_slot_demand"]  = test["geo_slot_demand"].fillna(test["geo_demand_mean"])

# Label encodings
print("Label encoding categorical features...")
for col in ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather']:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col + "_enc"] = le.transform(train[col].astype(str))
    test[col + "_enc"] = le.transform(test[col].astype(str))

# Target encoding for Weather
weather_mean = train.groupby('Weather')['demand'].mean()
train['weather_target_enc'] = train['Weather'].map(weather_mean)
test['weather_target_enc'] = test['Weather'].map(weather_mean)
test['weather_target_enc'] = test['weather_target_enc'].fillna(train["demand"].mean())

# Impute Temperature with geohash-level median
temp_median = train.groupby('geohash')['Temperature'].median()
global_temp_median = train['Temperature'].median()
train['Temperature'] = train.apply(
    lambda r: temp_median.get(r['geohash'], global_temp_median) if pd.isna(r['Temperature']) else r['Temperature'],
    axis=1
)
test['Temperature'] = test.apply(
    lambda r: temp_median.get(r['geohash'], global_temp_median) if pd.isna(r['Temperature']) else r['Temperature'],
    axis=1
)

# Prefix features
print("Adding neighborhood prefix features...")
train["geo_prefix_4"] = train["geohash"].str[:4]
train["geo_prefix_3"] = train["geohash"].str[:3]
train["geo_prefix_2"] = train["geohash"].str[:2]
test["geo_prefix_4"]  = test["geohash"].str[:4]
test["geo_prefix_3"]  = test["geohash"].str[:3]
test["geo_prefix_2"]  = test["geohash"].str[:2]

for prefix in ["geo_prefix_4", "geo_prefix_3", "geo_prefix_2"]:
    stats_df = train.groupby(prefix)["demand"].mean().reset_index()
    stats_df.columns = [prefix, f"{prefix}_demand_mean"]
    train = train.merge(stats_df, on=prefix, how="left")
    test  = test.merge(stats_df,  on=prefix, how="left")
    # Fill unseen prefixes with global mean
    train[f"{prefix}_demand_mean"] = train[f"{prefix}_demand_mean"].fillna(train["demand"].mean())
    test[f"{prefix}_demand_mean"]  = test[f"{prefix}_demand_mean"].fillna(train["demand"].mean())

# KNN feature
print("KNN location feature...")
knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn.fit(train[["lat","lng","hour"]].values, train["demand"].values)
train["knn_demand"] = knn.predict(train[["lat","lng","hour"]].values)
test["knn_demand"]  = knn.predict(test[["lat","lng","hour"]].values)

# Recompute global day48 features for the test set
print("Computing day 48 demand features...")
day48_demand = train[train["day"] == 48].groupby(["geohash", "hour"])["demand"].mean().reset_index()
day48_demand.columns = ["geohash", "hour", "day48_hour_demand"]
test = test.merge(day48_demand, on=["geohash","hour"], how="left")
test["day48_hour_demand"] = test["day48_hour_demand"].fillna(test["geo_demand_mean"])

day48_slot = train[train["day"] == 48].groupby(["geohash", "time_slot"])["demand"].mean().reset_index()
day48_slot.columns = ["geohash", "time_slot", "day48_slot_demand"]
test = test.merge(day48_slot, on=["geohash","time_slot"], how="left")
test["day48_slot_demand"] = test["day48_slot_demand"].fillna(test["geo_demand_mean"])

# Setup cross-validation
print("Setting up leak-free GroupKFold cross-validation...")
FEATURES = [
    "hour", "minute", "time_slot", "hour_sin", "hour_cos",
    "slot_sin", "slot_cos", "day", "day_of_week", "is_weekend",
    "lat", "lng",
    "RoadType_enc", "NumberofLanes", "LargeVehicles_enc",
    "Landmarks_enc", "Temperature", "weather_target_enc",
    "geo_demand_mean", "geo_demand_std",
    "geo_hour_demand", "geo_slot_demand",
    "day48_hour_demand", "day48_slot_demand",
    "geo_prefix_4_demand_mean", "geo_prefix_3_demand_mean", "geo_prefix_2_demand_mean",
    "knn_demand"
]

groups = train["geohash"].values
gkf = GroupKFold(n_splits=5)

# Array for OOF predictions of XGB, LGB, CAT
oof_xgb = np.zeros(len(train))
oof_lgb = np.zeros(len(train))
oof_cat = np.zeros(len(train))

test_xgb = np.zeros(len(test))
test_lgb = np.zeros(len(test))
test_cat = np.zeros(len(test))

for fold, (tr_idx, val_idx) in enumerate(gkf.split(train, train["demand"], groups)):
    print(f"\n--- Fold {fold+1} ---")
    
    tr_df  = train.iloc[tr_idx].copy()
    val_df = train.iloc[val_idx].copy()

    # Recompute day48 features from training fold only
    d48_hour = tr_df[tr_df["day"]==48].groupby(["geohash","hour"])["demand"].mean().reset_index()
    d48_hour.columns = ["geohash","hour","day48_hour_demand"]

    d48_slot = tr_df[tr_df["day"]==48].groupby(["geohash","time_slot"])["demand"].mean().reset_index()
    d48_slot.columns = ["geohash","time_slot","day48_slot_demand"]

    # Apply to train fold
    tr_df = tr_df.drop(columns=["day48_hour_demand","day48_slot_demand"], errors='ignore')
    tr_df = tr_df.merge(d48_hour, on=["geohash","hour"],      how="left")
    tr_df = tr_df.merge(d48_slot, on=["geohash","time_slot"], how="left")
    tr_df["day48_hour_demand"] = tr_df["day48_hour_demand"].fillna(tr_df["geo_demand_mean"])
    tr_df["day48_slot_demand"] = tr_df["day48_slot_demand"].fillna(tr_df["geo_demand_mean"])

    # Apply to val fold
    val_df = val_df.drop(columns=["day48_hour_demand","day48_slot_demand"], errors='ignore')
    val_df = val_df.merge(d48_hour, on=["geohash","hour"],      how="left")
    val_df = val_df.merge(d48_slot, on=["geohash","time_slot"], how="left")
    val_df["day48_hour_demand"] = val_df["day48_hour_demand"].fillna(val_df["geo_demand_mean"])
    val_df["day48_slot_demand"] = val_df["day48_slot_demand"].fillna(val_df["geo_demand_mean"])

    X_tr  = tr_df[FEATURES].values
    y_tr  = tr_df["demand"].values
    X_val = val_df[FEATURES].values
    y_val = val_df["demand"].values

    # 1. XGBoost
    print("Training XGBoost...")
    model_xgb = XGBRegressor(
        n_estimators=3000, learning_rate=0.015, max_depth=7,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0,
        early_stopping_rounds=150, eval_metric="rmse",
        device="cpu", random_state=42
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_val)
    test_xgb += model_xgb.predict(test[FEATURES].values) / 5

    # 2. LightGBM
    print("Training LightGBM...")
    model_lgb = LGBMRegressor(
        n_estimators=3000, learning_rate=0.015, max_depth=7, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1
    )
    model_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[])
    oof_lgb[val_idx] = model_lgb.predict(X_val)
    test_lgb += model_lgb.predict(test[FEATURES].values) / 5

    # 3. CatBoost
    print("Training CatBoost...")
    model_cat = CatBoostRegressor(
        iterations=3000, learning_rate=0.02, depth=7,
        l2_leaf_reg=3.0, eval_metric='RMSE', random_seed=42,
        verbose=False
    )
    model_cat.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=150)
    oof_cat[val_idx] = model_cat.predict(X_val)
    test_cat += model_cat.predict(test[FEATURES].values) / 5

# Clip all predictions
oof_xgb = np.maximum(oof_xgb, 0)
oof_lgb = np.maximum(oof_lgb, 0)
oof_cat = np.maximum(oof_cat, 0)

actual = train["demand"].values

print("\n=== MODEL PERFORMANCE (R2 raw scale) ===")
print(f"XGBoost OOF R2  : {r2_score(actual, oof_xgb):.4f}")
print(f"LightGBM OOF R2 : {r2_score(actual, oof_lgb):.4f}")
print(f"CatBoost OOF R2 : {r2_score(actual, oof_cat):.4f}")

# Find optimal blending weights using SLSQP or grid search
best_score = 0
best_w = (0.4, 0.3, 0.3)

for w1 in np.arange(0, 1.05, 0.05):
    for w2 in np.arange(0, 1.05 - w1, 0.05):
        w3 = 1.0 - w1 - w2
        blend_oof = w1 * oof_xgb + w2 * oof_lgb + w3 * oof_cat
        score = r2_score(actual, blend_oof)
        if score > best_score:
            best_score = score
            best_w = (w1, w2, w3)

print(f"\nBest Blend Weights: XGB={best_w[0]:.2f}, LGB={best_w[1]:.2f}, CAT={best_w[2]:.2f}")
print(f"Best Blended OOF R2 Score: {best_score:.4f}")
print(f"Estimated Leaderboard score: {max(0, 100 * best_score):.2f} / 100")

# Generate final ensemble predictions
print("Generating final blended test predictions...")
final_xgb = np.maximum(test_xgb, 0)
final_lgb = np.maximum(test_lgb, 0)
final_cat = np.maximum(test_cat, 0)

final_blend = best_w[0] * final_xgb + best_w[1] * final_lgb + best_w[2] * final_cat

# Clip predictions
final_blend = np.minimum(np.maximum(final_blend, 0), 1.0)

submission = pd.DataFrame({
    "Index": test["Index"],
    "demand": final_blend
})
submission.to_csv("submission_ensemble_blend.csv", index=False)
print("Saved submission_ensemble_blend.csv successfully!")
