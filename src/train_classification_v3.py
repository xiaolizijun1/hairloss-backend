# backend/train_classification.py  (v3 optimized)
import os, json, joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import recall_score, precision_score, f1_score, classification_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "Predict Hair Fall.csv")
OUT_DIR = os.path.join(ROOT, "models")
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# 1. Helpers
# -----------------------------
def age_group(x):
    try: x = float(x)
    except: return 0
    if x <= 25: return 0
    elif x <= 35: return 1
    elif x <= 45: return 2
    return 3

def stress_bucket(x):
    if pd.isna(x): return 1
    try:
        if isinstance(x, str):
            s = x.strip().lower()
            if s in ("low","l","0","1"): return 0
            if s in ("medium","m","2"): return 1
            if s in ("high","h","3"): return 2
        xv = float(x)
        if xv <= 3: return 0
        elif xv <= 7: return 1
        else: return 2
    except:
        return 1

def yesno(series):
    return series.replace({
        "Yes":1,"No":0,"yes":1,"no":0,
        "Y":1,"N":0,"TRUE":1,"FALSE":0,
        "True":1,"False":0
    }).fillna(series).apply(
        lambda v: 1 if str(v).strip() in ("1","1.0") else
                  0 if str(v).strip() in ("0","0.0","") else 0
    )

# -----------------------------
# 2. Load
# -----------------------------
df = pd.read_csv(DATA)
df.columns = [c.strip().replace(" ", "_").replace("&","and").replace("-","_") for c in df.columns]

# normalize target
if "Hair_Loss" not in df.columns:
    if "HairLoss" in df.columns:
        df.rename(columns={"HairLoss":"Hair_Loss"}, inplace=True)
    elif "Hair Loss" in df.columns:
        df.rename(columns={"Hair Loss":"Hair_Loss"}, inplace=True)

df.fillna(0, inplace=True)

# -----------------------------
# 3. Feature engineering (v3)
# -----------------------------
# KEEP effective features only
FEATURES = [
    "AgeGroup",
    "Age_cont",                  # NEW (continuous)
    "Genetics",
    "Hormonal_Changes",
    "Stress3",
    "Poor_Hair_Care_Habits",
    "Environmental_Factors",
    "Smoking",
    "Weight_Loss"
]

# Add AgeGroup & continuous age
df["AgeGroup"] = df["Age"].apply(age_group) if "Age" in df.columns else 0
df["Age_cont"] = pd.to_numeric(df["Age"], errors='coerce').fillna(30)

# Stress bucket
df["Stress3"] = df["Stress"].apply(stress_bucket) if "Stress" in df.columns else 1

# Remove noisy features (these exist but we ignore them)
NOISE = [
    "Nutritional_Deficiencies",
    "Medical_Conditions",
    "Medications_and_Treatments"
]
for n in NOISE:
    if n in df.columns:
        df.drop(columns=[n], inplace=True)

# yes/no → 0/1
binary_cols = [
    "Genetics", "Hormonal_Changes",
    "Poor_Hair_Care_Habits", "Environmental_Factors",
    "Smoking", "Weight_Loss"
]
for col in binary_cols:
    if col not in df.columns:
        df[col] = 0
    df[col] = yesno(df[col]).astype(int)

# final dataset
X = df[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
y = df["Hair_Loss"].astype(int)

print("\n=== FINAL FEATURES (v3) ===")
print(X.head())
print("X shape:", X.shape)
print("y dist:", y.value_counts().to_dict())

# -----------------------------
# 4. Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# -----------------------------
# 5. Models
# -----------------------------
models = {}

# DecisionTree (baseline)
models["DecisionTree"] = DecisionTreeClassifier(max_depth=4, min_samples_split=10, random_state=42)

# RandomForest
models["RandomForest"] = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Logistic Regression (great for weak signal)
models["LogisticRegression"] = LogisticRegression(
    class_weight="balanced",
    C=1.0,
    max_iter=1000
)

# Optimized XGBoost
try:
    from xgboost import XGBClassifier
    print("\nXGBoost available.")
    models["XGBoost_v3"] = XGBClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=1.2,   # IMPORTANT
        eval_metric="logloss",
        random_state=42
    )
except:
    print("XGBoost not available.")

# -----------------------------
# 6. Train & evaluate
# -----------------------------
results = {}
for name, model in models.items():
    print(f"\nTraining {name} ...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    recall = recall_score(y_test, preds)
    prec = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    results[name] = {
        "model": model,
        "recall": recall,
        "precision": prec,
        "f1": f1
    }

    print(f"{name} -> recall={recall:.4f}, precision={prec:.4f}, f1={f1:.4f}")
    print(classification_report(y_test, preds))

# -----------------------------
# 7. Select best
# -----------------------------
best_name = None
best_metric = (-1, -1)

for name, r in results.items():
    metric = (r["recall"], r["f1"])
    if metric > best_metric:
        best_metric = metric
        best_name = name

best_model = results[best_name]["model"]
print("\n=== SELECTED BEST MODEL (v3) ===")
print(best_name, results[best_name])

# -----------------------------
# 8. Save artifacts
# -----------------------------
joblib.dump(best_model, os.path.join(OUT_DIR, "best_model_v3.joblib"))

with open(os.path.join(OUT_DIR, "feature_schema_v3.json"), "w") as f:
    json.dump(FEATURES, f, indent=2)

joblib.dump({
    "features": FEATURES,
    "age_bins": [25,35,45],
    "model": best_name
}, os.path.join(OUT_DIR, "model_meta_v3.joblib"))

print("\nSaved to:", OUT_DIR)

# Feature importance if exists
if hasattr(best_model, "feature_importances_"):
    print("\n=== Feature Importances ===")
    fi = pd.Series(best_model.feature_importances_, index=FEATURES)
    print(fi.sort_values(ascending=False))
