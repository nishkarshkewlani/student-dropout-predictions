# Student Performance & Dropout Prediction

An end-to-end machine learning project that identifies students at academic risk
from historical educational indicators, so advisors can intervene early.

**Workflow:** data cleaning -> exploratory data analysis -> feature engineering ->
Decision Tree vs Naive Bayes comparison -> Flask REST API -> web form and analytics dashboard.

## What "at risk" means

The dataset has no real dropout column. A student is labelled **at risk when the final grade
G3 is below 10 (out of 20)**, the pass mark. The model predicts the probability of that outcome
from earlier information (study time, past failures, absences, support, age, school).
G1/G2 (earlier grades) and sex are deliberately **not** used: G1/G2 would leak the answer and
sex would make the prediction unfair.

## Project structure

```
student-dropout-prediction/
├── dataset/student_data.csv     training data (school, sex, age, studytime, failures,
│                                schoolsup, famsup, absences, G3)
├── ml/
│   ├── features.py              shared cleaning + feature engineering (training AND API)
│   ├── train_model.py           EDA, training, comparison, saves model and metrics
│   ├── student_model.pkl        created by train_model.py
│   ├── metrics.json             created by train_model.py
│   └── eda/                     EDA charts, created by train_model.py
├── backend/app.py               Flask API + serves the web pages
├── frontend/
│   ├── index.html               prediction form
│   └── analytics.html           dashboard (Chart.js)
├── requirements.txt
└── Procfile                     gunicorn command for deployment
```

## Run it

```bash
pip install -r requirements.txt
python ml/train_model.py          # prints metrics, saves model + EDA charts
python backend/app.py             # then open http://localhost:5000
```

Pages: `/` predictor, `/dashboard` analytics. API: `POST /predict`, `GET /analytics`, `GET /health`.

Example request:

```bash
curl -X POST localhost:5000/predict -H "Content-Type: application/json" \
  -d '{"school":"GP","age":17,"studytime":2,"failures":1,"absences":8,"schoolsup":0,"famsup":1}'
```

## Feature engineering

| Feature | Meaning |
| --- | --- |
| support_count | school support + family support (0, 1 or 2) |
| has_failures | 1 if the student has any past failure |
| high_absences | 1 if absences are 10 or more |

## Model comparison

Fill this table from the output of `python ml/train_model.py` (also saved in `ml/metrics.json`).
Report the held-out test numbers and the dataset size next to them.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Decision Tree | | | | | |
| Naive Bayes | | | | | |

Recall on the at-risk class matters most: missing a struggling student costs more than a false alarm.

## Deploy (Render)

* Build command: `pip install -r requirements.txt && python ml/train_model.py`
* Start command: `gunicorn --chdir backend app:app --bind 0.0.0.0:$PORT`

Training during the build keeps the saved model and the server's scikit-learn version in sync.

## Limitations

* The label is a proxy for risk, not actual withdrawal from school.
* Small datasets give unreliable metrics; use the full dataset.
* A prediction is a screening aid: it should trigger support, never a penalty.
