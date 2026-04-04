import os, json, joblib
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CANDIDATES = [
    os.path.join(ROOT, "data", "Predict Hair Fall.csv"),

]
OUT_DIR = os.path.join(ROOT, "models")
os.makedirs(OUT_DIR, exist_ok=True)

# Helpers
def find_data():
    for p in DATA_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No dataset found. Put CSV in data/ and try.")

def age_group(x):
    try:
        x = float(x)
    except:
        return 0
    if x <= 25: return 0
    elif x <= 35: return 1
    elif x <= 45: return 2
    else: return 3

def stress_bucket(x):
    # map original stress (1-10 or 'Low'/'Medium'/'High') into 3 buckets:
    # 0: low, 1: medium, 2: high
    if pd.isna(x): return 1
    try:
        if isinstance(x, str):
            s = x.strip().lower()
            if s in ("low","l","1","0"): return 0
            if s in ("medium","med","mid","m","2"): return 1
            if s in ("high","h","3"): return 2
        xv = float(x)
        if xv <= 3: return 0
        elif xv <= 7: return 1
        else: return 2
    except:
        return 1

def map_yes_no(series):
    out = series.replace({
        "Yes": 1, "No": 0,
        "Low": 0, "Medium": 1, "High": 2
    })

    out = out.infer_objects(copy=False)
    return out


# Load
DATA_PATH = find_data()
print("Using dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().replace(" ", "_").replace("&","and").replace("-","_") for c in df.columns]

# target column (already 0/1)
df = df.dropna(subset=["Hair_Loss"])
y = df["Hair_Loss"].astype(int)
df.fillna(0, inplace=True)  # fill broad NA

# Apply AgeGroup and Stress3
if "Age" in df.columns:
    df["AgeGroup"] = df["Age"].apply(age_group)
else:
    df["AgeGroup"] = 0

if "Stress" in df.columns:
    df["Stress3"] = df["Stress"].apply(stress_bucket)
else:
    df["Stress3"] = 1

# Candidate canonical features (order matters)
FEATURES = [
    "AgeGroup",
    "Genetics",
    "Hormonal_Changes",
    "Medical_Conditions",
    "Medications_and_Treatments",
    "Nutritional_Deficiencies",
    "Stress3",
    "Poor_Hair_Care_Habits",
    "Environmental_Factors",
    "Smoking",
    "Weight_Loss"
]

# Try to map dataset column names to canonical features, add missing with zeros
mapped = {}
cols_lower = {c.lower(): c for c in df.columns}
for f in FEATURES:
    # direct
    if f in df.columns:
        mapped[f] = f
        continue
    # fuzzy checks
    lf = f.lower()
    found = None
    for c in df.columns:
        if lf.replace("_","") in c.lower().replace("_",""):
            found = c; break
    if found:
        mapped[f] = found
    else:
        # add a zero column
        df[f] = 0
        mapped[f] = f

# For mapped features that are categorical yes/no, convert to 0/1
binary_candidates = [mapped[f] for f in FEATURES if f not in ("AgeGroup","Stress3")]
for col in binary_candidates:
    # attempt mapping yes/no to 0/1; leave numeric as is
    df[col] = map_yes_no(df[col])
    # finally coerce to numeric (if still text, convert to 0)
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# final X,y
X = df[[mapped[f] for f in FEATURES]].apply(pd.to_numeric, errors='coerce').fillna(0)
y = pd.to_numeric(df["Hair_Loss"], errors='coerce').fillna(0).astype(int)

print("Final features used:", list(X.columns))
print("X shape:", X.shape, "y dist:", y.value_counts().to_dict())

# split 70/30 stratified
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

# models to train
models = {}
models["DecisionTree"] = DecisionTreeClassifier(max_depth=5, min_samples_split=8, random_state=42)
models["RandomForest"] = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)

# try xgboost, fallback to GradientBoosting
try:
    from xgboost import XGBClassifier

    models["XGBoost"] = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        eval_metric="logloss",
        random_state=42
    )
    print("XGBoost available and will be used.")
except Exception as e:
    print("XGBoost not available, fallback to GradientBoosting. Err:", e)
    models["GradientBoosting"] = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)

results = {}
for name, model in models.items():
    print("Training", name)
    if name == "XGBoost":
        model.fit(X_train.values, y_train.values)
        preds = model.predict(X_test.values)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    recall = recall_score(y_test, preds)
    precision = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    results[name] = {"model": model, "recall": recall, "precision": precision, "f1": f1}
    print(f"{name} -> recall={recall:.4f}, precision={precision:.4f}, f1={f1:.4f}")
    print(classification_report(y_test, preds))

# select best by recall then f1
best = None
best_metric = (-1.0, -1.0)
# Prioritize recall over F1 to minimize missed hair loss cases (false negatives more costly)
for name, r in results.items():
    metric = (r["recall"], r["f1"])
    if metric > best_metric:
        best_metric = metric; best = name

best_model = results[best]["model"]
print("Selected best model:", best)

# save artifacts
joblib.dump(best_model, os.path.join(OUT_DIR, "best_model.joblib"))
meta = {
    "features": FEATURES,
    "mapped_features": [mapped[f] for f in FEATURES],
    "age_bins": [25,35,45],
    "stress_map": {"low":0,"medium":1,"high":2},
    "model_name": best
}
joblib.dump(meta, os.path.join(OUT_DIR, "model_meta.joblib"))
with open(os.path.join(OUT_DIR, "feature_schema.json"), "w") as f:
    json.dump([mapped[f] for f in FEATURES], f, indent=2)

print("Saved artifacts to", OUT_DIR)
if hasattr(best_model, "feature_importances_"):
    import pandas as _pd
    fi = _pd.Series(best_model.feature_importances_, index=[mapped[f] for f in FEATURES]).sort_values(ascending=False)
    print("Feature importances:\n", fi.to_string())
