# src/train_classification_v4.py
"""
train_classification_v4.py - Final optimized training pipeline

Main ideas:
- Keep v1 feature spirit (AgeGroup + several binary features) but:
  - Use Stress as continuous numeric (Stress_num) instead of coarse bucket
  - Remove clearly noisy features (Nutritional_Deficiencies, Medical_Conditions, Medications_and_Treatments)
- Use SMOTE on training set (if imblearn installed) or fallback to simple random upsampling
- Train baseline (v1-like) XGBoost for comparison and optimized XGBoost_v4
- Selection rule: primary=recall, secondary=f1
- Save: models/best_model_v4.joblib, model_meta_v4.joblib, feature_schema_v4.json
"""

import os, json, joblib
import pandas as pd, numpy as np
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# try import xgboost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# try import SMOTE
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except Exception:
    SMOTE_AVAILABLE = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_CANDIDATES = [
    os.path.join(ROOT, "data", "Predict Hair Fall (1).csv"),
    os.path.join(ROOT, "data", "Predict Hair Fall.csv"),
    os.path.join(ROOT, "data", "predict_hair_fall_clean.csv"),
]
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

OUT_MODEL = os.path.join(MODELS_DIR, "best_model_v4.joblib")
OUT_META = os.path.join(MODELS_DIR, "model_meta_v4.joblib")
OUT_SCHEMA = os.path.join(MODELS_DIR, "feature_schema_v4.json")

RANDOM_STATE = 42
TEST_SIZE = 0.30

def find_data():
    for p in DATA_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No dataset found. Put CSV in data/")

def map_yesno(s):
    return s.replace({
        "Yes":1,"No":0,"yes":1,"no":0,"Y":1,"N":0,"true":1,"false":0,"True":1,"False":0
    })

def to_agegroup(x):
    try:
        xv = float(x)
    except:
        return 0
    if xv <= 25: return 0
    elif xv <= 35: return 1
    elif xv <= 45: return 2
    else: return 3

def stress_to_num(x):
    # prefer numeric; if text Low/Medium/High convert to representative numeric (2,5,8)
    if pd.isna(x): return np.nan
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("low","l"): return 2.0
        if s in ("medium","med","m"): return 5.0
        if s in ("high","h"): return 8.0
        try:
            return float(s)
        except:
            return np.nan
    try:
        return float(x)
    except:
        return np.nan

def summarize(y_true, y_pred, probs=None, label=""):
    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = None
    if probs is not None:
        try:
            auc = roc_auc_score(y_true, probs)
        except:
            auc = None
    print(f"--- {label} ---")
    print(f"Recall: {rec:.4f}, Precision: {prec:.4f}, F1: {f1:.4f}, AUC: {auc}")
    print(classification_report(y_true, y_pred))
    return {"recall": rec, "precision": prec, "f1": f1, "auc": auc}

# -------------------------
# Load & preproc
# -------------------------
DATA_PATH = find_data()
print("Using dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().replace(" ", "_").replace("&","and").replace("-","_") for c in df.columns]

# normalize target
if "Hair_Loss" not in df.columns:
    if "HairLoss" in df.columns:
        df.rename(columns={"HairLoss":"Hair_Loss"}, inplace=True)
    elif "Hair Loss" in df.columns:
        df.rename(columns={"Hair Loss":"Hair_Loss"}, inplace=True)
    elif "bald_prob" in df.columns:
        df["Hair_Loss"] = (df["bald_prob"] >= 0.5).astype(int)
    else:
        raise RuntimeError("Target not found")

# fill NA broadly
df.fillna(value=np.nan, inplace=True)  # keep NaNs for continuous handling

# AgeGroup (keep), remove Age_cont to avoid conflict
df["AgeGroup"] = df["Age"].apply(to_agegroup) if "Age" in df.columns else 0

# Stress_num: continuous representative (prefer numeric, else map Low/Med/High to 2/5/8)
df["Stress_num"] = df["Stress"].apply(stress_to_num) if "Stress" in df.columns else np.nan
# if still NaN fill with median of non-nulls or 5
if df["Stress_num"].isna().all():
    df["Stress_num"] = 5.0
else:
    df["Stress_num"] = df["Stress_num"].fillna(df["Stress_num"].median())

# Define final feature set (v4): based on v1 but cleaned (drop noisy cols)
FEATURES_V4 = [
    "AgeGroup",
    "Stress_num",
    "Genetics",
    "Hormonal_Changes",
    "Poor_Hair_Care_Habits",
    "Environmental_Factors",
    "Smoking",
    "Weight_Loss"
]

# Remove noisy raw columns if present
for c in ["Nutritional_Deficiencies", "Medical_Conditions", "Medications_and_Treatments"]:
    if c in df.columns:
        # drop entirely to avoid noise
        df.drop(columns=[c], inplace=True)

# Map binary fields to 0/1 robustly
for col in ["Genetics","Hormonal_Changes","Poor_Hair_Care_Habits","Environmental_Factors","Smoking","Weight_Loss"]:
    if col in df.columns:
        df[col] = df[col].astype(str).replace({"nan":0}).map({
            "Yes":1,"No":0,"yes":1,"no":0,"Y":1,"N":0,"True":1,"False":0,"true":1,"false":0
        }).replace({None:0}).fillna(0)
        # coerce to numeric
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    else:
        df[col] = 0

# Build X,y
X = df[FEATURES_V4].copy()
y = pd.to_numeric(df["Hair_Loss"], errors='coerce').fillna(0).astype(int)

print("Final features (v4):", FEATURES_V4)
print("X shape:", X.shape, "y dist:", y.value_counts().to_dict())

# -------------------------
# Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
print("Train/test shapes:", X_train.shape, X_test.shape)

# -------------------------
# Oversampling: SMOTE or fallback
# -------------------------
print("SMOTE available:", SMOTE_AVAILABLE)
if SMOTE_AVAILABLE:
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print("After SMOTE, class dist:", pd.Series(y_res).value_counts().to_dict())
else:
    # simple random upsampling minority class on train
    from sklearn.utils import resample
    df_train = pd.concat([X_train, y_train], axis=1)
    majority = df_train[df_train["Hair_Loss"]==0]
    minority = df_train[df_train["Hair_Loss"]==1]
    if len(minority)==0 or len(majority)==0:
        X_res, y_res = X_train, y_train
    else:
        minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=RANDOM_STATE)
        df_bal = pd.concat([majority, minority_upsampled])
        y_res = df_bal["Hair_Loss"]
        X_res = df_bal.drop(columns=["Hair_Loss"])

# -------------------------
# Baseline (v1-like) XGBoost reproduction for comparison
# -------------------------
print("\nTraining baseline XGBoost (v1-like) for comparison...")
if XGB_AVAILABLE:
    baseline = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4, use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE)
else:
    from sklearn.ensemble import GradientBoostingClassifier
    baseline = GradientBoostingClassifier(n_estimators=400, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE)

# Use v1-like features: AgeGroup + Stress_num + binaries (same as our FEATURES_V4 minus any extra)
X_v1 = X_train.copy()  # we already set X columns similarly
baseline.fit(X_v1.values if XGB_AVAILABLE else X_v1, y_train.values if XGB_AVAILABLE else y_train)
pred_b = baseline.predict(X_test.values if XGB_AVAILABLE else X_test)
prob_b = baseline.predict_proba(X_test.values)[:,1] if hasattr(baseline, "predict_proba") else None
metrics_baseline = summarize(y_test, pred_b, prob_b, label="Baseline (v1-like)")

# -------------------------
# Optimized XGBoost v4 (recommended)
# -------------------------
print("\nTraining optimized XGBoost v4...")
if XGB_AVAILABLE:
    xgb_v4 = XGBClassifier(
        n_estimators=350,
        learning_rate=0.08,
        max_depth=3,
        subsample=0.85,
        colsample_bytree=0.75,
        min_child_weight=3,
        gamma=0.1,
        scale_pos_weight=1.2,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )
    xgb_v4.fit(X_res.values, y_res.values)
    preds_x4 = xgb_v4.predict(X_test.values)
    probs_x4 = xgb_v4.predict_proba(X_test.values)[:,1]
else:
    # fallback: RandomForest tuned
    xgb_v4 = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    xgb_v4.fit(X_res, y_res)
    preds_x4 = xgb_v4.predict(X_test)
    probs_x4 = xgb_v4.predict_proba(X_test)[:,1]

metrics_x4 = summarize(y_test, preds_x4, probs_x4, label="XGBoost_v4 (optimized)")

# -------------------------
# Also train LogisticRegression (reference)
# -------------------------
print("\nTraining LogisticRegression (reference)...")
lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_res.values if hasattr(X_res, "values") else X_res, y_res.values if hasattr(y_res, "values") else y_res)
pred_lr = lr.predict(X_test.values if hasattr(X_test, "values") else X_test)
probs_lr = lr.predict_proba(X_test.values)[:,1] if hasattr(lr, "predict_proba") else None
metrics_lr = summarize(y_test, pred_lr, probs_lr, label="LogisticRegression (v4 ref)")

# -------------------------
# Candidate selection
# -------------------------
candidates = {
    "baseline_v1": {"model": baseline, "metrics": metrics_baseline},
    "xgb_v4": {"model": xgb_v4, "metrics": metrics_x4},
    "lr_v4": {"model": lr, "metrics": metrics_lr}
}

# choose best by (recall, f1)
best_name = None
best_tuple = (-1.0, -1.0)
for name, info in candidates.items():
    m = info["metrics"]
    tup = (m["recall"], m["f1"])
    if tup > best_tuple:
        best_tuple = tup
        best_name = name

print("\n=== Selection result ===")
print("Best model:", best_name, "metrics:", candidates[best_name]["metrics"])

# Save selected model & meta & schema
joblib.dump(candidates[best_name]["model"], OUT_MODEL)
meta = {
    "selected": best_name,
    "features": FEATURES_V4,
    "notes": "v4: Stress_num continuous, AgeGroup kept, noisy fields removed, SMOTE applied during training"
}
joblib.dump(meta, OUT_META)
with open(OUT_SCHEMA, "w") as f:
    json.dump(FEATURES_V4, f, indent=2)

print("Saved:", OUT_MODEL, OUT_META, OUT_SCHEMA)

# Print final feature importances when available
if hasattr(candidates[best_name]["model"], "feature_importances_"):
    fi = pd.Series(candidates[best_name]["model"].feature_importances_, index=FEATURES_V4).sort_values(ascending=False)
    print("\nFeature importances (selected model):\n", fi.to_string())
