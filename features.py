"""
features.py
-----------
Shared data preparation used by BOTH training (ml/train_model.py) and the
Flask API (backend/app.py).

Keeping this logic in ONE place guarantees that a student entered in the web
form is cleaned and engineered exactly like the rows the model was trained on.
That removes the classic "training vs serving mismatch" bug.
"""

import numpy as np
import pandas as pd

# Columns that must exist in dataset/student_data.csv (G3 is the target source).
BASE_COLUMNS = ["school", "age", "studytime", "failures", "schoolsup", "famsup", "absences"]
TARGET_SOURCE = "G3"          # final grade out of 20
PASS_MARK = 10                # G3 below this => "at risk" (academic dropout risk)

# Features given to the model.
# NOTE: "sex" is deliberately NOT used. Predicting risk from gender would be
# unfair, and G1/G2 (earlier grades) are not used either because they would
# make the task almost trivial (target leakage).
CATEGORICAL = ["school"]
NUMERIC = [
    "age", "studytime", "failures", "absences",      # raw numeric inputs
    "schoolsup", "famsup",                            # yes/no -> 1/0
    "support_count", "has_failures", "high_absences",  # engineered features
]
FEATURES = CATEGORICAL + NUMERIC

HIGH_ABSENCE_THRESHOLD = 10   # 10+ absences counts as "high absences"

_YES = {"yes", "y", "true", "1", "1.0"}
_NO = {"no", "n", "false", "0", "0.0"}


def to_binary(value):
    """Convert yes/no style values (yes, no, 1, 0, True, False) to 1.0 / 0.0.
    Anything unrecognised becomes NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().lower()
    if text in _YES:
        return 1.0
    if text in _NO:
        return 0.0
    return np.nan


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw columns and add engineered features.

    Works on the training CSV (text values such as 'yes'/'GP') and on a single
    row coming from the web form (numbers or text).
    """
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    missing = [c for c in BASE_COLUMNS if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    # ---- cleaning -------------------------------------------------------
    # school: tidy text, keep missing values as NaN so the imputer can fill them
    school = d["school"].astype("string").str.strip().str.upper()
    d["school"] = school.astype(object).where(school.notna(), np.nan)

    # numeric columns: convert text to numbers, bad values become NaN
    for col in ["age", "studytime", "failures", "absences"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["absences"] = d["absences"].clip(lower=0)     # negative absences are impossible

    # yes/no columns -> 1/0
    for col in ["schoolsup", "famsup"]:
        d[col] = d[col].map(to_binary)

    # ---- feature engineering -------------------------------------------
    # how many support systems the student has (0, 1 or 2)
    d["support_count"] = d["schoolsup"] + d["famsup"]
    # has the student failed any class before?
    d["has_failures"] = (d["failures"] > 0).astype(float).where(d["failures"].notna(), np.nan)
    # is the student frequently absent?
    d["high_absences"] = (
        (d["absences"] >= HIGH_ABSENCE_THRESHOLD).astype(float).where(d["absences"].notna(), np.nan)
    )
    return d


def make_target(g3: pd.Series) -> pd.Series:
    """1 = at risk (final grade below the pass mark), 0 = safe."""
    return (pd.to_numeric(g3, errors="coerce") < PASS_MARK).astype(int)
