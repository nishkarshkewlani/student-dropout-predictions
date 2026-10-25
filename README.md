# Student Academic-Risk Prediction

A small end-to-end machine-learning demo that estimates whether a student may be at risk of receiving a low final grade, based on selected academic and support indicators. It includes data preparation, exploratory analysis, model comparison, a Flask API, and a browser-based prediction form/dashboard.

> **Label clarification:** The supplied dataset does not contain a true dropout/withdrawal label. In this project, `at_risk = 1` means the final grade `G3` is below 10 out of 20. Therefore, this is a **low-final-grade / academic-risk proxy**, not a validated student-dropout predictor.

## Features

- Clean and prepare student-record data.
- Engineer additional indicators from support, prior failures, and absences.
- Generate exploratory data-analysis charts.
- Train and compare a Decision Tree and Gaussian Naive Bayes model.
- Evaluate with accuracy, precision, recall, F1, ROC-AUC (when calculable), and a confusion matrix.
- Save the trained pipeline and metrics.
- Serve predictions through a Flask API.
- Provide a prediction page and analytics dashboard.

## Target definition and input features

### Target

`at_risk = 1` when `G3 < 10`; otherwise `at_risk = 0`.

This is a proxy label based on final grade. It should not be described as actual dropout unless a real withdrawal/dropout outcome is added to the dataset.

### Features

The project uses selected fields such as school, age, study time, past failures, absences, school support, and family support. Feature preparation also derives:

| Engineered feature | Meaning |
|---|---|
| `support_count` | Number of support indicators enabled |
| `has_failures` | Whether the student has any recorded past failures |
| `high_absences` | Whether absences meet the configured threshold (10 or more) |

Earlier grades `G1` and `G2` are excluded to reduce target leakage from grades closely related to `G3`. The `sex` field is also excluded from the model inputs.

## Tech stack

- **Language:** Python
- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn
- **Model persistence:** joblib
- **API and web server:** Flask
- **Visualizations:** Matplotlib
- **Frontend:** HTML, JavaScript/Chart.js (dashboard)
- **Deployment:** Gunicorn / Render configuration

## Project structure

```text
student-dropout-predictions-main/
├── dataset/
│   └── student_data.csv
├── ml/
│   ├── features.py        # shared cleaning, feature engineering, target creation
│   └── train_model.py     # EDA, training, comparison, metrics/model saving
├── backend/
│   └── app.py             # Flask pages and API endpoints
├── frontend/
│   ├── index.html         # prediction form
│   └── analytics.html     # analytics dashboard
├── requirements.txt
└── Procfile
```

Generated files such as `ml/student_model.pkl`, `ml/metrics.json`, and EDA charts are created by the training script and may not be present until training has run.

## Setup and run locally

Use Python 3.10 or 3.11 if your installed scikit-learn version causes compatibility issues.

```bash
# 1. Clone the repository and enter its directory
git clone <YOUR_REPOSITORY_URL>
cd student-dropout-predictions-main

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux (run this instead of the Windows command)
source .venv/bin/activate

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Train models and generate the model bundle, metrics, and EDA charts
python ml/train_model.py

# 5. Start the Flask application
python backend/app.py
```

Open `http://127.0.0.1:5000`.

**Important:** Run training before starting the web app. The backend expects the trained model file to exist.

## Application routes

| Route | Purpose |
|---|---|
| `GET /` | Prediction form |
| `GET /dashboard` | Analytics dashboard |
| `POST /predict` | Returns a risk prediction for supplied student features |
| `GET /analytics` | Returns dataset statistics and model metrics |
| `GET /health` | Health/status endpoint |

### Example prediction request

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "school": "GP",
    "age": 17,
    "studytime": 2,
    "failures": 1,
    "absences": 8,
    "schoolsup": 0,
    "famsup": 1
  }'
```

The response format is defined by `backend/app.py`; actual risk level and probability depend on the trained model and input.

## Model evaluation

The training script compares:
- **Decision Tree:** an interpretable tree-based classifier with class balancing and parameter search when the data supports cross-validation.
- **Gaussian Naive Bayes:** a probabilistic classifier based on conditional-independence assumptions.

Metrics are saved to `ml/metrics.json` when training completes. Report results only after running the code, and always include the number of rows and test examples used.

| Metric | What it tells you |
|---|---|
| Accuracy | Fraction of all test predictions that are correct |
| Precision | Fraction of predicted at-risk students who are truly at risk under the proxy label |
| Recall | Fraction of at-risk students under the proxy label that the model identifies |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Ranking discrimination across thresholds; only defined here when the test split contains both classes |

For an early-support use case, recall for the at-risk class is important, but it should be considered alongside precision and the cost of false alarms.

## Data and result caveats

The included CSV is a small demonstration dataset. A very small dataset or test split can produce unstable metrics, misleadingly high accuracy, or an undefined ROC-AUC. Do not present results from a tiny sample as evidence of real-world predictive performance.

For a stronger evaluation:
1. Use a sufficiently large, documented dataset with a genuine outcome label.
2. Keep a representative held-out test set and use cross-validation where feasible.
3. Report class balance, sample counts, confusion matrix, and per-class precision/recall.
4. Check calibration, subgroup performance, data quality, and potential bias.
5. Compare against a simple baseline before claiming the model adds value.

## Deployment notes

The repository includes a `Procfile` for Gunicorn deployment. For Render, configure the build to install requirements and train the model, then start the Flask app using the command specified by the repository's deployment configuration. Ensure the training and serving environments use compatible scikit-learn versions.

## Limitations and responsible use

- The target is a final-grade threshold proxy, not actual dropout.
- Small sample sizes make model performance estimates unreliable.
- Risk probabilities are model outputs and may not be calibrated probabilities of a real-world event.
- The tool should be used, at most, to support human review and early assistance—not to punish, label, or deny opportunities to students.
- Any real deployment should require validated data, privacy safeguards, transparency, fairness checks, and human oversight.
