"""
app.py
------
Flask REST API + web server for the Student Dropout Prediction project.

Routes
    GET  /            prediction form (frontend/index.html)
    GET  /dashboard   analytics dashboard (frontend/analytics.html)
    POST /predict     JSON in -> risk prediction out
    GET  /analytics   dataset statistics + model metrics (JSON)
    GET  /health      quick status check

Run:    python backend/app.py        then open http://localhost:5000
Deploy: gunicorn --chdir backend app:app
"""

import json
import math
import os
import sys

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
ML_DIR = os.path.join(ROOT_DIR, "ml")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
DATASET_PATH = os.path.join(ROOT_DIR, "dataset", "student_data.csv")
MODEL_PATH = os.path.join(ML_DIR, "student_model.pkl")
METRICS_PATH = os.path.join(ML_DIR, "metrics.json")

# reuse the SAME data-preparation code that was used for training
sys.path.insert(0, ML_DIR)
from features import make_target, prepare, to_binary  # noqa: E402

app = Flask(__name__)

# ---------------------------------------------------------------------
# Load the trained model ONCE at startup (not on every request)
# ---------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise SystemExit("Model not found. Train it first:  python ml/train_model.py")

bundle = joblib.load(MODEL_PATH)
model = bundle["pipeline"]
FEATURES = bundle["features"]
MODEL_NAME = bundle["model_name"]
PASS_MARK = bundle["pass_mark"]

# allowed ranges for validation
RANGES = {"age": (10, 35), "studytime": (1, 4), "failures": (0, 10), "absences": (0, 200)}
REQUIRED = ["age", "studytime", "failures", "absences", "schoolsup", "famsup"]


# ---------------------------------------------------------------------
# Small CORS helper so the HTML pages also work if opened from another
# origin during development (e.g. a different port or Live Server).
# ---------------------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = os.environ.get("CORS_ORIGIN", "*")
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def risk_level(probability):
    if probability >= 0.6:
        return "High"
    if probability >= 0.4:
        return "Medium"
    return "Low"


def validate(payload):
    """Return (clean_row_dict, list_of_errors)."""
    errors, row = [], {}
    if not isinstance(payload, dict):
        return None, ["Send a JSON object."]

    for field in REQUIRED:
        if field not in payload or payload[field] in ("", None):
            errors.append(f"'{field}' is required.")

    for field, (low, high) in RANGES.items():
        if field in payload and payload[field] not in ("", None):
            try:
                value = float(payload[field])
                if not math.isfinite(value) or not (low <= value <= high):
                    raise ValueError
                row[field] = value
            except (TypeError, ValueError):
                errors.append(f"'{field}' must be a number between {low} and {high}.")

    for field in ("schoolsup", "famsup"):
        if field in payload and payload[field] not in ("", None):
            value = to_binary(payload[field])
            if math.isnan(value):
                errors.append(f"'{field}' must be 1/0 or yes/no.")
            else:
                row[field] = value

    row["school"] = str(payload.get("school") or "GP").strip().upper()
    return row, errors


# ---------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------
@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/dashboard")
def dashboard():
    return send_from_directory(FRONTEND_DIR, "analytics.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})


# ---------------------------------------------------------------------
# API
# ---------------------------------------------------------------------
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return ("", 204)

    row, errors = validate(request.get_json(silent=True))
    if errors:
        return jsonify({"error": "Invalid input", "details": errors}), 400

    frame = prepare(pd.DataFrame([row]))[FEATURES]      # same cleaning as training
    probability = float(model.predict_proba(frame)[0][1])
    return jsonify({
        "dropout_risk": int(probability >= 0.5),
        "risk_level": risk_level(probability),
        "probability": round(probability, 4),
        "model": MODEL_NAME,
    })


@app.route("/analytics", methods=["GET"])
def analytics():
    if not os.path.exists(DATASET_PATH):
        return jsonify({"error": "Dataset file not found"}), 404

    data = pd.read_csv(DATASET_PATH)
    data.columns = [c.strip() for c in data.columns]
    data["at_risk"] = make_target(data["G3"])

    total = int(len(data))
    at_risk = int(data["at_risk"].sum())

    by_failures = (data.groupby("failures")["at_risk"].mean() * 100).round(1)
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)

    return jsonify({
        "total_students": total,
        "dropout": at_risk,                 # kept for backward compatibility
        "at_risk": at_risk,
        "safe": total - at_risk,
        "at_risk_percent": round(100 * at_risk / total, 1) if total else 0,
        "risk_by_failures": {str(k): float(v) for k, v in by_failures.items()},
        "label_rule": f"At risk = final grade G3 below {PASS_MARK} (out of 20)",
        "model": MODEL_NAME,
        "metrics": metrics,
    })


if __name__ == "__main__":
    # debug is OFF by default; set FLASK_DEBUG=1 only on your own machine
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)),
            debug=os.environ.get("FLASK_DEBUG") == "1")
