"""
train_model.py
--------------
End-to-end training workflow for the Student Performance & Dropout Prediction
project.

    1. Load and CLEAN the dataset
    2. EDA  (charts saved to ml/eda/)
    3. FEATURE ENGINEERING (see features.py)
    4. Train and compare TWO classifiers: Decision Tree and Naive Bayes
    5. Evaluate with accuracy, precision, recall, F1, ROC-AUC, confusion matrix
    6. Save the best model + metrics for the Flask API

Run from anywhere:
    python ml/train_model.py
"""

import json
import os
import warnings
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")  # draw charts to files, no display needed (works in Codespaces)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from features import (CATEGORICAL, FEATURES, NUMERIC, PASS_MARK, TARGET_SOURCE,
                       make_target, prepare)

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "student_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "student_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")
EDA_DIR = os.path.join(BASE_DIR, "eda")
RANDOM_STATE = 42
MIN_ROWS_RECOMMENDED = 100


def banner(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


# ---------------------------------------------------------------------
# 1. LOAD + CLEAN
# ---------------------------------------------------------------------
def load_and_clean():
    banner("1. LOADING AND CLEANING DATA")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {os.path.abspath(DATASET_PATH)}")

    raw = pd.read_csv(DATASET_PATH)
    raw.columns = [c.strip() for c in raw.columns]
    print(f"Rows loaded      : {len(raw)}")
    print(f"Columns          : {list(raw.columns)}")

    if TARGET_SOURCE not in raw.columns:
        raise ValueError(f"Dataset must contain the column '{TARGET_SOURCE}'")

    # drop rows with no final grade (cannot create a label for them)
    before = len(raw)
    raw[TARGET_SOURCE] = pd.to_numeric(raw[TARGET_SOURCE], errors="coerce")
    raw = raw.dropna(subset=[TARGET_SOURCE]).reset_index(drop=True)
    print(f"Rows without G3  : {before - len(raw)} removed")

    print(f"Duplicate rows   : {int(raw.duplicated().sum())} (kept: students can look identical)")
    print("Missing values per column:")
    print(raw.isna().sum().to_string())

    data = prepare(raw)                      # cleaning + feature engineering
    data["at_risk"] = make_target(data[TARGET_SOURCE])
    return data


# ---------------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------------
def run_eda(data):
    banner("2. EXPLORATORY DATA ANALYSIS")
    os.makedirs(EDA_DIR, exist_ok=True)

    counts = data["at_risk"].value_counts().sort_index()
    safe, risk = int(counts.get(0, 0)), int(counts.get(1, 0))
    print(f"Class balance    : safe={safe}  at_risk={risk}  ({risk / len(data):.1%} at risk)")
    print("\nAverages by risk group:")
    cols = ["age", "studytime", "failures", "absences", "support_count"]
    print(data.groupby("at_risk")[cols].mean().round(2).to_string())

    # chart 1: class balance
    plt.figure(figsize=(4, 3.5))
    plt.bar(["Safe", "At risk"], [safe, risk], color=["#2e7d32", "#c62828"])
    plt.title("Class balance")
    plt.ylabel("Students")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "class_balance.png"), dpi=120)
    plt.close()

    # chart 2: at-risk rate by number of past failures
    rate = data.groupby("failures")["at_risk"].mean() * 100
    plt.figure(figsize=(5, 3.5))
    plt.bar(rate.index.astype(int).astype(str), rate.values, color="#1565c0")
    plt.title("At-risk rate by past failures")
    plt.xlabel("Past failures")
    plt.ylabel("% at risk")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "risk_by_failures.png"), dpi=120)
    plt.close()

    # chart 3: absences by risk group
    plt.figure(figsize=(5, 3.5))
    for label, color, name in [(0, "#2e7d32", "Safe"), (1, "#c62828", "At risk")]:
        subset = data.loc[data["at_risk"] == label, "absences"].dropna()
        if len(subset):
            plt.hist(subset, bins=15, alpha=0.6, color=color, label=name)
    plt.title("Absences by risk group")
    plt.xlabel("Absences")
    plt.ylabel("Students")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "absences_by_risk.png"), dpi=120)
    plt.close()

    # chart 4: correlation heatmap
    corr_cols = NUMERIC + ["at_risk"]
    corr = data[corr_cols].corr()
    plt.figure(figsize=(7, 6))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="correlation")
    plt.xticks(range(len(corr_cols)), corr_cols, rotation=75, ha="right")
    plt.yticks(range(len(corr_cols)), corr_cols)
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "correlation.png"), dpi=120)
    plt.close()

    print(f"\nCharts saved in  : {os.path.abspath(EDA_DIR)}")
    return {"safe": safe, "at_risk": risk}


# ---------------------------------------------------------------------
# 3. MODELS
# ---------------------------------------------------------------------
def one_hot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:                      # older scikit-learn versions
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline(classifier):
    preprocess = ColumnTransformer([
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", one_hot())]), CATEGORICAL),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC),
    ])
    return Pipeline([("prep", preprocess), ("clf", classifier)])


def evaluate(pipe, X_test, y_test):
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]
    result = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "roc_auc": None,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
    }
    if y_test.nunique() == 2:
        result["roc_auc"] = round(float(roc_auc_score(y_test, prob)), 4)
    return result


def feature_importances(pipe):
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return {}
    names = pipe.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    pairs = sorted(zip(names, clf.feature_importances_), key=lambda p: -p[1])
    return {n: round(float(v), 4) for n, v in pairs if v > 0}


def train_and_compare(data):
    banner("3. TRAINING AND COMPARING MODELS")
    X = data[FEATURES]
    y = data["at_risk"]
    n_rows = len(data)
    min_class = int(y.value_counts().min()) if y.nunique() == 2 else 0

    warnings_list = []
    if y.nunique() < 2:
        raise ValueError("Only one class found in the data. Need both safe and at-risk students.")
    if n_rows < MIN_ROWS_RECOMMENDED:
        msg = (f"Only {n_rows} rows: metrics are NOT reliable. "
               f"Use the full dataset ({MIN_ROWS_RECOMMENDED}+ rows) before quoting results.")
        warnings_list.append(msg)
        print("WARNING:", msg)

    # stratified split keeps the same risk percentage in train and test
    can_stratify = min_class >= 2 and n_rows >= 10
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE,
        stratify=y if can_stratify else None)
    print(f"Train rows: {len(X_train)}   Test rows: {len(X_test)}   stratified: {can_stratify}")

    # cross-validation folds: limited by the smallest class in the training data
    n_splits = min(5, int(y_train.value_counts().min()))
    use_cv = n_splits >= 2
    if not use_cv:
        warnings_list.append("Too few at-risk (or safe) students for cross-validation; "
                             "models were fit with default settings.")
        print("WARNING:", warnings_list[-1])

    candidates = {
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {"clf__max_depth": [3, 4, 5, 6], "clf__min_samples_leaf": [1, 5, 10]}),
        "Naive Bayes": (
            GaussianNB(),
            {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]}),
    }

    results, fitted = {}, {}
    for name, (clf, grid) in candidates.items():
        pipe = build_pipeline(clf)
        if use_cv:
            search = GridSearchCV(pipe, grid, cv=n_splits, scoring="f1")
            search.fit(X_train, y_train)
            best, cv_f1 = search.best_estimator_, float(search.best_score_)
            params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        else:
            best = pipe.fit(X_train, y_train)
            cv_f1, params = None, {}
        metrics = evaluate(best, X_test, y_test)
        metrics["cv_f1"] = None if cv_f1 is None else round(cv_f1, 4)
        metrics["best_params"] = params
        results[name], fitted[name] = metrics, best

        print(f"\n--- {name} ---")
        print(f"best params : {params}")
        print(f"CV F1       : {metrics['cv_f1']}")
        print(f"accuracy    : {metrics['accuracy']}   precision: {metrics['precision']}   "
              f"recall: {metrics['recall']}   f1: {metrics['f1']}   roc_auc: {metrics['roc_auc']}")
        tn, fp = metrics["confusion_matrix"][0]
        fn, tp = metrics["confusion_matrix"][1]
        print(f"confusion matrix  TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    # pick the model by cross-validated F1 (fallback: test F1). F1 balances
    # catching at-risk students (recall) against false alarms (precision).
    def score(name):
        m = results[name]
        return (m["cv_f1"] if m["cv_f1"] is not None else m["f1"], m["recall"])

    best_name = max(results, key=score)
    banner(f"BEST MODEL: {best_name}")
    return best_name, fitted[best_name], results, warnings_list, len(X_train), len(X_test)


# ---------------------------------------------------------------------
# 4. SAVE
# ---------------------------------------------------------------------
def save(best_name, best_pipe, results, balance, warnings_list, n_train, n_test, n_rows):
    importances = feature_importances(best_pipe)
    if importances:
        print("Top factors driving risk (decision tree importances):")
        for feat, value in list(importances.items())[:6]:
            print(f"  {feat:15s} {value:.3f}")

    bundle = {
        "pipeline": best_pipe,
        "features": FEATURES,
        "model_name": best_name,
        "pass_mark": PASS_MARK,
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, MODEL_PATH)

    report = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "rows": n_rows,
        "train_rows": n_train,
        "test_rows": n_test,
        "class_balance": balance,
        "label_rule": f"at_risk = 1 when final grade G3 < {PASS_MARK} (out of 20)",
        "features_used": FEATURES,
        "best_model": best_name,
        "models": results,
        "feature_importances": importances,
        "warnings": warnings_list,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nModel saved   : {os.path.abspath(MODEL_PATH)}")
    print(f"Metrics saved : {os.path.abspath(METRICS_PATH)}")


def main():
    data = load_and_clean()
    balance = run_eda(data)
    best_name, best_pipe, results, warnings_list, n_train, n_test = train_and_compare(data)
    save(best_name, best_pipe, results, balance, warnings_list, n_train, n_test, len(data))
    print("\nDone. Next:  python backend/app.py   then open http://localhost:5000")


if __name__ == "__main__":
    main()
