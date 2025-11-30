import pandas as pd
import numpy as np
import os
import joblib

CLEAN = os.path.join(os.path.dirname(__file__), "..", "data", "Predict Hair Fall.csv")
OUT_PIPE = os.path.join(os.path.dirname(__file__), "..", "models", "preprocessing.joblib")
SCHEMA = os.path.join(os.path.dirname(__file__), "..", "models", "feature_schema.json")

def build_feature_set(path=CLEAN):
    df = pd.read_csv(path)

    # rename
    df.rename(columns=lambda x: x.strip().replace(" ", "_"), inplace=True)

    y = df["Hair_Loss"].astype(int)

    # ----------------------------
    # 1. Binary Yes/No → 1/0
    # ----------------------------
    binary_cols = [
        "Genetics", "Hormonal_Changes", "Medical_Conditions",
        "Medications_and_Treatments", "Nutritional_Deficiencies",
        "Stress", "Poor_Hair_Care_Habits", "Environmental_Factors",
        "Smoking", "Weight_Loss"
    ]

    for col in binary_cols:
        if col in df:
            df[col] = df[col].map({"Yes": 1, "No": 0, "Low": 0, "Medium": 1, "High": 2}).fillna(0)

    # ----------------------------
    # 2. Age
    # ----------------------------
    if "Age" in df:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(df["Age"].median())

    # ----------------------------
    # 3. Final features
    # ----------------------------
    X = df.drop(columns=["Hair_Loss"])

    # Save schema
    import json
    json.dump({"columns": list(X.columns)}, open(SCHEMA, "w"))

    joblib.dump(None, OUT_PIPE)  # no pipeline, keep simple processing
    return X.values, y.values, list(X.columns)

if __name__ == "__main__":
    X, y, cols = build_feature_set()
    print("X shape:", X.shape)
