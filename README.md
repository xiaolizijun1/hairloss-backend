# Hair Loss Risk Prediction — Full-Stack ML App

A full-stack machine learning web application that predicts hair loss risk based on user health and lifestyle inputs. Built with a React front-end, Flask REST API backend, and deployed to Vercel.

🔗 **Live Demo:** [hairloss-frontend.vercel.app](https://hairloss-frontend.vercel.app)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React, HTML5, CSS3 |
| Backend | Python, Flask, REST API |
| ML Models | scikit-learn, XGBoost, Pandas |
| Deployment | Vercel (frontend), Render (backend) |

---

## Project Structure

hairloss-backend/
├── app.py                  # Flask REST API entry point
├── best_model.joblib       # Trained XGBoost model (serialized)
├── model_meta.joblib       # Preprocessing metadata & encoders
├── feature_schema.json     # Input feature schema for validation
├── models/                 # Model training experiments
└── requirements.txt

hairloss-frontend/
├── src/
│   ├── App.js              # Main React component
│   └── ...                 # UI components & form logic
└── package.json

---

## ML Pipeline

## Dataset
**Source:** [Hair Loss Factors Dataset](https://www.kaggle.com/code/ayaabdalsalam/hair-loss-analysis/input?scriptVersionId=217216355) via Kaggle
Features include age, stress level, genetics, and other lifestyle indicators.
Model is pre-trained（best_model.joblib） — clone the repo and run directly without re-downloading data.

### Models Compared
Three classifiers were trained and evaluated on the same dataset:

| Model | Notes |
|-------|-------|
| Decision Tree | Baseline interpretable model |
| Random Forest | Ensemble baseline |
| **XGBoost** | **Best F1-score — selected for production** |

### Feature Engineering
- **Age grouping:** binned continuous age into categorical risk brackets
- **Stress-level encoding:** ordinal encoding of self-reported stress levels
- **Lifestyle factors:** diet, sleep, physical activity encoded as categorical features
- **Stratified train/test split** (70/30) to preserve class distribution

### Evaluation
- Primary metric: **F1-score** (balances precision and recall for imbalanced classes)
- Secondary: Precision, Recall, Confusion Matrix
- Consistent preprocessing pipeline between training and production inference

---

## API Endpoint

POST /predict
Content-Type: application/json

{
  "age": 32,
  "stress_level": "high",
  "diet": "poor",
  ...
}

Response:
{
  "prediction": "At Risk",
  "probability": 0.78
}

---

## Running Locally

### Backend
git clone https://github.com/xiaolizijun1/hairloss-backend
cd hairloss-backend
pip install -r requirements.txt
python app.py
# API running at http://localhost:5000

### Frontend
git clone https://github.com/xiaolizijun1/hairloss-frontend
cd hairloss-frontend
npm install
npm start
# App running at http://localhost:3000

---

## Key Takeaways

- End-to-end ML deployment: from notebook to production web app
- Reproducible preprocessing pipeline ensures training/inference consistency
- XGBoost outperformed both Decision Tree and Random Forest baselines
