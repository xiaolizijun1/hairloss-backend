"""
train_classification_v2.py

Purpose:
- Produce an optimized training pipeline for the Hair Loss project.
- Keep original train_classification.py untouched (v1). This script (v2) reproduces
  a v1-like baseline for fair comparison, then runs optimized experiments:
    * improved feature handling (keep Age and Stress as continuous, optional bins)
    * drop constant / zero-importance features
    * GridSearchCV for XGBoost (scoring='recall')
    * train RandomForest, optimized XGBoost, and a Stacking ensemble (RF + XGB -> LR)
- Select best model (primary: recall, secondary: f1), save model + meta.
- Print a comparison summary.

How to run:
    python src/train_classification_v2.py

Outputs:
  models/best_model_v2.joblib
  models/model_meta_v2.joblib
  models/feature_schema_v2.json

Notes:
- Requires xgboost installed for the best results. If xgboost not available, it falls back to GradientBoosting.
- Uses 70/30 stratified split (same as v1).
- Grid search is limited to a conservative grid by default to keep runtime modest; expand if you have more time/CPU.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from time import time
from pprint import pprint

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

# try import xgboost; fall back gracefully
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception as e:
    XGB_AVAILABLE = False
    from sklearn.ensemble import GradientBoostingClassifier as XGBFallback
    print("Warning: xgboost not available; will use GradientBoosting as fallback. Install xgboost for best results.")

# ---- Config / paths ----
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_CANDIDATES = [
    os.path.join(ROOT, "data", "Predict Hair Fall.csv"),
    os.path.join(ROOT, "data", "Predict Hair Fall.csv"),
    os.path.join(ROOT, "data", "predict_hair_fall_clean.csv"),
    os.path.join(ROOT, "data", "predict_hair_fall.csv"),
]
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

OUT_MODEL = os.path.join(MODELS_DIR, "best_model_v2.joblib")
OUT_META = os.path.join(MODELS_DIR, "model_meta_v2.joblib")
OUT_SCHEMA = os.path.join(MODELS_DIR, "feature_schema_v2.json")

RANDOM_STATE = 42
TEST_SIZE = 0.30
CV = 4  # 4-fold CV for grid search to balance speed and reliability

# ---- Utility helpers ----
def find_data():
    for p in DATA_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No dataset found. Checked: {DATA_CANDIDATES}")

def to_agegroup(x):
    try:
        xv = float(x)
    except:
        return 0
    if xv <= 25: return 0
    elif xv <= 35: return 1
    elif xv <= 45: return 2
    else: return 3

def stress_bucket_from_raw(x):
    """
    Keep continuous stress if numeric; otherwise map textual Low/Medium/High to 0/1/2.
    We'll preserve both continuous (Stress_continuous) and bucket (Stress3) for experiments.
    """
    if pd.isna(x): return np.nan
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("low","l"): return 0
        if s in ("medium","med","m"): return 1
        if s in ("high","h"): return 2
        # if string numeric
        try:
            xv = float(s)
            if xv <= 3: return 0
            elif xv <= 7: return 1
            else: return 2
        except:
            return np.nan
    try:
        xv = float(x)
        if xv <= 3: return 0
        elif xv <= 7: return 1
        else: return 2
    except:
        return np.nan

def map_yes_no(series):
    # robust mapping: accept "Yes"/"No"/True/False/"1"/"0" etc.
    s = series.copy()
    s = s.replace({
        "Yes":1, "No":0, "YES":1, "NO":0, "yes":1, "no":0,
        "Y":1, "N":0, "y":1, "n":0,
        True:1, False:0, "True":1, "False":0
    })
    # If still non-numeric, attempt numeric coercion then fillna(0)
    s = pd.to_numeric(s, errors='coerce')
    s = s.fillna(0).astype(int)
    return s

def summarize_metrics(y_true, y_pred, probs=None, label=""):
    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, probs) if probs is not None else None
    except:
        auc = None
    print(f"--- {label} ---")
    print(f"Recall: {rec:.4f}, Precision: {prec:.4f}, F1: {f1:.4f}, AUC: {auc}")
    print(classification_report(y_true, y_pred))
    return {"recall": rec, "precision": prec, "f1": f1, "auc": auc}

# ---- Load and preprocess (v2 strategy) ----
print("loading data...")
DATA_PATH = find_data()
print("Using dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
orig_columns = df.columns.tolist()
print("Original columns:", orig_columns)

# normalize column names
df.columns = [c.strip().replace(" ", "_").replace("&","and").replace("-","_") for c in df.columns]

# find target
if "Hair_Loss" not in df.columns:
    if "HairLoss" in df.columns:
        df.rename(columns={"HairLoss":"Hair_Loss"}, inplace=True)
    elif "Hair Loss" in df.columns:
        df.rename(columns={"Hair Loss":"Hair_Loss"}, inplace=True)
    elif "bald_prob" in df.columns:
        # fallback: convert probability to binary by 0.5
        df["Hair_Loss"] = (df["bald_prob"] >= 0.5).astype(int)
    else:
        raise RuntimeError("Target column not found. Need Hair_Loss or HairLoss or bald_prob")

# Fill obvious NA with 0 for binary-like fields; continuous we will impute more carefully
df = df.copy()
df.fillna(value=0, inplace=True)

# Create Age continuous + AgeGroup (v2)
if "Age" in df.columns:
    df["Age_cont"] = pd.to_numeric(df["Age"], errors='coerce').fillna(df["Age"].median())
    df["AgeGroup"] = df["Age_cont"].apply(lambda x: to_agegroup(x))
else:
    df["Age_cont"] = 0.0
    df["AgeGroup"] = 0

# Stress: create continuous if exists (try to coerce), plus bucket Stress3
if "Stress" in df.columns:
    # try numeric continuous
    df["Stress_cont"] = pd.to_numeric(df["Stress"], errors='coerce')
    # if all NaN, we'll keep bucket only
    df["Stress3"] = df["Stress"].apply(stress_bucket_from_raw)
    # if Stress_cont has many NaNs, fill them with median
    df["Stress_cont"] = df["Stress_cont"].fillna(df["Stress_cont"].median() if not df["Stress_cont"].isna().all() else df["Stress3"].median())
    df["Stress3"] = df["Stress3"].fillna(df["Stress_cont"].apply(lambda x: 0 if x<=3 else (1 if x<=7 else 2)))
else:
    df["Stress_cont"] = 5.0
    df["Stress3"] = 1

# Canonical feature list (v2) - starts from v1 and we will optionally prune later
FEATURES_V2 = [
    "Age_cont",  # keep continuous Age
    "AgeGroup",
    "Stress_cont",  # continuous stress kept as information-rich
    "Stress3",
    "Genetics",
    "Hormonal_Changes",
    "Poor_Hair_Care_Habits",
    "Environmental_Factors",
    "Smoking",
    "Weight_Loss",
    "Nutritional_Deficiencies",
    "Medical_Conditions",
    "Medications_and_Treatments"
]

# Attempt to map fuzzy columns in df to canon names; if missing create zeros
mapped = {}
for canon in FEATURES_V2:
    if canon in df.columns:
        mapped[canon] = canon
    else:
        # look for similar
        found = None
        for c in df.columns:
            if canon.lower().replace("_","") in c.lower().replace("_",""):
                found = c; break
        if found:
            mapped[canon] = found
        else:
            # create zero column (or sensible default)
            if canon in ("Age_cont","Stress_cont"):
                df[canon] = 0.0
            else:
                df[canon] = 0
            mapped[canon] = canon

print("Mapped feature columns (v2):")
pprint(mapped)

# Convert binary-like fields robustly
binary_list = [c for c in FEATURES_V2 if c not in ("Age_cont","AgeGroup","Stress_cont","Stress3")]
for b in binary_list:
    df[mapped[b]] = map_yes_no(df[mapped[b]])

# Build X,y for v2
X_v2 = df[[mapped[f] for f in FEATURES_V2]].apply(pd.to_numeric, errors='coerce').fillna(0)
y = pd.to_numeric(df["Hair_Loss"], errors='coerce').fillna(0).astype(int)

print("Final X_v2 shape:", X_v2.shape, "y dist:", y.value_counts().to_dict())

# ---- reproduce baseline model (v1-like) for fair comparison ----
print("\n=== Baseline (v1-like) XGBoost reproduction ===")
# baseline uses AgeGroup and Stress3 and the simpler feature set (like earlier script)
FEATURES_V1_LIKE = [
    "AgeGroup", "Genetics", "Hormonal_Changes", "Medical_Conditions",
    "Medications_and_Treatments", "Nutritional_Deficiencies", "Stress3",
    "Poor_Hair_Care_Habits", "Environmental_Factors", "Smoking", "Weight_Loss"
]
X_v1 = df[[mapped.get(f,f) for f in FEATURES_V1_LIKE]].apply(pd.to_numeric, errors='coerce').fillna(0)

X_train_v1, X_test_v1, y_train_v1, y_test_v1 = train_test_split(X_v1, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

# build baseline xgb with v1 params
if XGB_AVAILABLE:
    baseline_xgb = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4, use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE)
else:
    baseline_xgb = GradientBoostingClassifier(n_estimators=400, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE)

baseline_xgb.fit(X_train_v1.values if XGB_AVAILABLE else X_train_v1, y_train_v1.values if XGB_AVAILABLE else y_train_v1)
preds_baseline = baseline_xgb.predict(X_test_v1.values if XGB_AVAILABLE else X_test_v1)
probs_baseline = baseline_xgb.predict_proba(X_test_v1.values)[:,1] if hasattr(baseline_xgb, "predict_proba") else None
metrics_baseline = summarize_metrics(y_test_v1, preds_baseline, probs_baseline, label="Baseline XGBoost (v1-like)")

# ---- V2: optimized experiments ----
print("\n=== V2: optimized experiments ===")
X_train, X_test, y_train, y_test = train_test_split(X_v2, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
print("Train/test shapes (v2):", X_train.shape, X_test.shape)

# Option: prune columns that are constant or near-constant
const_cols = [c for c in X_train.columns if X_train[c].nunique() <= 1]
if const_cols:
    print("Dropping constant columns (no variance):", const_cols)
    X_train.drop(columns=const_cols, inplace=True); X_test.drop(columns=const_cols, inplace=True)

# Train RandomForest (strong baseline)
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)
preds_rf = rf.predict(X_test)
probs_rf = rf.predict_proba(X_test)[:,1] if hasattr(rf, "predict_proba") else None
metrics_rf = summarize_metrics(y_test, preds_rf, probs_rf, label="RandomForest (v2 baseline)")

# Train an optimized XGBoost via GridSearch (focus on recall)
print("\nStarting GridSearch for XGBoost (recall)...")
if XGB_AVAILABLE:
    estimator = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE)
else:
    estimator = GradientBoostingClassifier(random_state=RANDOM_STATE)

param_grid = {
    "n_estimators": [300, 500, 800],
    "learning_rate": [0.01, 0.03, 0.05],
    "max_depth": [3, 4, 5],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0] if XGB_AVAILABLE else [1.0]
}

# limit verbosity and runtime; scoring recall because business wants recall up
grid = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=CV, scoring="recall", verbose=1, n_jobs=-1)
t0 = time()
grid.fit(X_train.values if XGB_AVAILABLE else X_train, y_train.values if XGB_AVAILABLE else y_train)
print(f"GridSearch done in {time()-t0:.1f}s. Best params:", grid.best_params_)
best_xgb = grid.best_estimator_

# evaluate optimized xgb
if XGB_AVAILABLE:
    preds_xgb_opt = best_xgb.predict(X_test.values)
    probs_xgb_opt = best_xgb.predict_proba(X_test.values)[:,1]
else:
    preds_xgb_opt = best_xgb.predict(X_test)
    probs_xgb_opt = best_xgb.predict_proba(X_test)[:,1]
metrics_xgb_opt = summarize_metrics(y_test, preds_xgb_opt, probs_xgb_opt, label="Optimized XGBoost (GridSearch)")

# Check feature importances and optionally drop zero-importance columns, retrain small XGB
if hasattr(best_xgb, "feature_importances_"):
    fi = pd.Series(best_xgb.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\nTop feature importances (optimized XGB):")
    print(fi.to_string())
    zero_imp = fi[fi <= 0.0].index.tolist()
    if zero_imp:
        print("Zero-importance features (will drop and retrain small xgb):", zero_imp)
        # drop and retrain quick
        X_train_pruned = X_train.drop(columns=zero_imp)
        X_test_pruned = X_test.drop(columns=zero_imp)
        if XGB_AVAILABLE:
            best_xgb.fit(X_train_pruned.values, y_train.values)
            preds_xgb_pruned = best_xgb.predict(X_test_pruned.values)
            probs_xgb_pruned = best_xgb.predict_proba(X_test_pruned.values)[:,1]
        else:
            best_xgb.fit(X_train_pruned, y_train)
            preds_xgb_pruned = best_xgb.predict(X_test_pruned)
            probs_xgb_pruned = best_xgb.predict_proba(X_test_pruned)[:,1]
        metrics_xgb_pruned = summarize_metrics(y_test, preds_xgb_pruned, probs_xgb_pruned, label="Optimized XGB (pruned features)")
else:
    print("No feature_importances_ on best_xgb (maybe fallback).")

# ---- Stacking ensemble (RF + XGB -> LR) ----
print("\nTraining stacking ensemble (RF + XGB) ...")
estimators_stack = [
    ("rf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))
]
if XGB_AVAILABLE:
    estimators_stack.append(("xgb", XGBClassifier(**grid.best_params_, use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE)))
else:
    estimators_stack.append(("gb", GradientBoostingClassifier(**grid.best_params_, random_state=RANDOM_STATE)))

stack = StackingClassifier(estimators=estimators_stack, final_estimator=LogisticRegression(), passthrough=True, n_jobs=-1)
stack.fit(X_train.values if XGB_AVAILABLE else X_train, y_train.values if XGB_AVAILABLE else y_train)
preds_stack = stack.predict(X_test.values if XGB_AVAILABLE else X_test)
probs_stack = stack.predict_proba(X_test.values if XGB_AVAILABLE else X_test)[:,1] if hasattr(stack,"predict_proba") else None
metrics_stack = summarize_metrics(y_test, preds_stack, probs_stack, label="Stacking Ensemble (RF + XGB -> LR)")

# ---- Select best model among: baseline_xgb, rf, best_xgb, stack
candidates = {
    "baseline_xgb_v1": {"model": baseline_xgb, "metrics": metrics_baseline},
    "rf_v2": {"model": rf, "metrics": metrics_rf},
    "xgb_opt_v2": {"model": best_xgb, "metrics": metrics_xgb_opt},
    "stack_v2": {"model": stack, "metrics": metrics_stack}
}

# selection rule: primary recall, secondary f1
best_name = None
best_tuple = (-1.0, -1.0)
for name, info in candidates.items():
    m = info["metrics"]
    tup = (m["recall"], m["f1"])
    if tup > best_tuple:
        best_tuple = tup
        best_name = name

best_info = candidates[best_name]
print("\n=== SELECTION RESULT ===")
print("Best model:", best_name, "metrics:", best_info["metrics"])

# Save best model and metadata
best_model_to_save = best_info["model"]
joblib.dump(best_model_to_save, OUT_MODEL)
meta = {
    "model_name": best_name,
    "selection_metric": {"primary":"recall","secondary":"f1"},
    "features_v2": list(X_train.columns),
    "mapped": mapped,
    "grid_best_params": getattr(grid, "best_params_", None)
}
joblib.dump(meta, OUT_META)
with open(OUT_SCHEMA, "w") as f:
    json.dump(list(X_train.columns), f, indent=2)

print("Saved best_model_v2 ->", OUT_MODEL)
print("Saved model_meta_v2 ->", OUT_META)
print("Saved feature_schema_v2 ->", OUT_SCHEMA)

# Final comparison printout
print("\n===== FINAL COMPARISON SUMMARY =====")
print("Baseline (v1-like) metrics:")
pprint(metrics_baseline)
print("\nRandomForest (v2) metrics:")
pprint(metrics_rf)
print("\nOptimized XGBoost (v2) metrics:")
pprint(metrics_xgb_opt)
print("\nStacking (v2) metrics:")
pprint(metrics_stack)
print("\nSelected model:", best_name, "with metrics:", best_info["metrics"])

print("\nDone. You can now start backend/app.py to serve models/best_model_v2.joblib")
