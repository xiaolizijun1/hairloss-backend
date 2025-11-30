# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os, joblib, json, pandas as pd, numpy as np

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE, "..", "models"))
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
META_PATH = os.path.join(MODELS_DIR, "model_meta.joblib")
SCHEMA_PATH = os.path.join(MODELS_DIR, "feature_schema.json")

if not os.path.exists(BEST_MODEL_PATH):
    raise RuntimeError("best_model.joblib not found. Run training first.")

best_model = joblib.load(BEST_MODEL_PATH)
meta = joblib.load(META_PATH)  # features, mapped_features, age_bins, stress_map
FEATURES = meta["features"]
MAPPED_FEATURES = meta["mapped_features"]
AGE_BINS = meta.get("age_bins", [25,35,45])

def to_agegroup(age):
    try:
        x = float(age)
    except:
        return 0
    if x <= 25: return 0
    elif x <= 35: return 1
    elif x <= 45: return 2
    else: return 3

def to_stress3(v):
    if v is None: return 1
    try:
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("low","l"): return 0
            if s in ("medium","med","m"): return 1
            if s in ("high","h"): return 2
        xv = float(v)
        if xv <= 3: return 0
        elif xv <= 7: return 1
        else: return 2
    except:
        return 1

def to_binary(v):
    if v is None: return 0
    if isinstance(v,(int,float)) and int(v) in (0,1): return int(v)
    s = str(v).strip().lower()
    if s in ("yes","y","true","1","1.0"): return 1
    return 0

# Strict validation rules for canonical features
def validate_payload(payload):
    # Accept either canonical features or raw age/stress
    # If payload has raw Age and Stress we convert them; otherwise require canonical fields.
    # We'll build a canonical dict 'canon' with keys = FEATURES
    canon = {}
    # If Age present as raw, convert
    if "Age" in payload:
        canon["AgeGroup"] = to_agegroup(payload.get("Age"))
    elif "AgeGroup" in payload:
        canon["AgeGroup"] = int(payload.get("AgeGroup"))
    else:
        return None, "Missing Age/AgeGroup"

    if "Stress" in payload:
        canon["Stress3"] = to_stress3(payload.get("Stress"))
    elif "Stress3" in payload:
        canon["Stress3"] = int(payload.get("Stress3"))
    else:
        return None, "Missing Stress/Stress3"

    # For all other binary features, accept Yes/No/0/1/true/false
    for f in FEATURES:
        if f in ("AgeGroup", "Stress3"): continue
        if f in payload:
            canon[f] = to_binary(payload.get(f))
        else:
            # Try to accept mapped names too (meta contains mapped_features)
            # We'll accept either canonical or original mapped column; else default 0
            canon[f] = 0

    # Validate ranges
    if not (0 <= int(canon["AgeGroup"]) <= 3):
        return None, "AgeGroup out of range 0..3"
    if not (0 <= int(canon["Stress3"]) <= 2):
        return None, "Stress3 out of range 0..2"
    for f in FEATURES:
        if f in ("AgeGroup","Stress3"): continue
        if canon[f] not in (0,1):
            return None, f"{f} invalid value"

    return canon, None

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.json
    if not payload:
        return jsonify({"error":"Empty request or wrong Content-Type (must be application/json)"}), 400

    canon, err = validate_payload(payload)
    if err:
        return jsonify({"error": err}), 400

    # Build DataFrame in mapped column order used in training
    row = {}
    for canon_name, mapped_name in zip(FEATURES, MAPPED_FEATURES):
        val = canon.get(canon_name, 0)
        row[mapped_name] = val

    df = pd.DataFrame([row]).astype(float)
    X = df[MAPPED_FEATURES].values

    try:
        if hasattr(best_model, "predict_proba"):
            prob = float(best_model.predict_proba(X)[0][1])
            pred = int(best_model.predict(X)[0])
        else:
            pred = int(best_model.predict(X)[0])
            prob = float(pred)
    except Exception as e:
        return jsonify({"error": "Model prediction error: " + str(e)}), 500

    return jsonify({"prediction": pred, "probability": round(prob,4)})

@app.route("/")
def home():
    return "Hair Loss Prediction API (validation enabled)."

if __name__ == "__main__":
    print("Using model:", BEST_MODEL_PATH)
    app.run(debug=True, port=5000)
