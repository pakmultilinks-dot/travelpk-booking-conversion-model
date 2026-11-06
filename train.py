"""Train and evaluate the Travel.pk booking conversion model.

Data: synthetic only (see generate_data.py and README). No real Travel.pk
booking data was available. The test set is held out with a stratified
split and is never used for training, so every metric below is measured
on unseen data.

Models: XGBoost classifier and a logistic regression baseline.
Outputs: model files, test predictions, results/metrics.json and
results/metrics.txt, feature importance.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.20
DATA_PATH = "data/travelpk_bookings_synthetic.csv"
MODEL_PATH = "model_xgboost.pkl"
BASELINE_PATH = "model_baseline.pkl"
RESULTS_DIR = "results"

NUMERIC = [
    "price_pkr",
    "days_advance",
    "party_size",
    "trip_duration_days",
    "pages_viewed",
    "time_on_site_min",
    "returning_visitor",
    "weekend_departure",
]
CATEGORICAL = ["trip_type", "device", "channel"]
TARGET = "booked"


def build_preprocessor():
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def feature_names(preprocessor):
    names = list(NUMERIC)
    cat = preprocessor.named_transformers_["cat"]
    names += list(cat.get_feature_names_out(CATEGORICAL))
    return names


def evaluate(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    print("Rows:", len(df), "| base conversion rate: %.4f" % df[TARGET].mean())

    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print("Train rows:", len(X_train), "| test rows:", len(X_test),
          "| test positives:", int(y_test.sum()))

    # XGBoost model. Hyperparameters picked by 5-fold cross-validation on
    # the training data (CV ROC-AUC 0.7508): depth 3, lr 0.05, 200 trees,
    # min_child_weight 5.
    xgb = Pipeline(
        [
            ("prep", build_preprocessor()),
            (
                "model",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    min_child_weight=5,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    xgb_metrics = evaluate(y_test, xgb_pred, xgb_proba)

    # Logistic regression baseline, same preprocessing.
    baseline = Pipeline(
        [
            ("prep", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    baseline.fit(X_train, y_train)
    base_pred = baseline.predict(X_test)
    base_proba = baseline.predict_proba(X_test)[:, 1]
    base_metrics = evaluate(y_test, base_pred, base_proba)

    # Feature importance (gain) from the fitted XGBoost model.
    booster = xgb.named_steps["model"].get_booster()
    gain = booster.get_score(importance_type="gain")
    names = feature_names(xgb.named_steps["prep"])
    importance = [
        {"feature": names[int(k[1:])], "gain": round(float(v), 2)}
        for k, v in gain.items()
    ]
    importance.sort(key=lambda d: -d["gain"])
    total_gain = sum(d["gain"] for d in importance) or 1.0
    for d in importance:
        d["gain_share_pct"] = round(100.0 * d["gain"] / total_gain, 2)

    joblib.dump(xgb, MODEL_PATH)
    joblib.dump(baseline, BASELINE_PATH)
    test_out = X_test.copy()
    test_out["actual"] = y_test.values
    test_out["predicted"] = xgb_pred
    test_out["probability"] = np.round(xgb_proba, 4)
    test_out.to_csv(os.path.join(RESULTS_DIR, "test_predictions.csv"), index=False)

    best = (
        "xgboost"
        if xgb_metrics["roc_auc"] >= base_metrics["roc_auc"]
        else "logistic_regression_baseline"
    )
    results = {
        "data": {
            "rows": len(df),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "base_conversion_rate": round(float(df[TARGET].mean()), 4),
            "seed": RANDOM_STATE,
            "synthetic": True,
        },
        "xgboost": xgb_metrics,
        "logistic_regression_baseline": base_metrics,
        "feature_importance_gain": importance,
        "best_model": best,  # by test ROC-AUC
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    lines = [
        "Travel.pk Booking Conversion Model - test set results",
        "Data: synthetic only, 6000 rows, base conversion rate %.4f"
        % df[TARGET].mean(),
        "Test set: %d rows, %d positives (stratified, unseen in training)"
        % (len(X_test), int(y_test.sum())),
        "",
        "XGBoost: " + json.dumps(xgb_metrics),
        "Logistic regression baseline: " + json.dumps(base_metrics),
        "",
        "Feature importance (XGBoost gain):",
    ]
    for d in importance:
        lines.append("  %s: gain %.2f (%.2f%%)" % (d["feature"], d["gain"], d["gain_share_pct"]))
    with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print("XGBoost metrics:", json.dumps(xgb_metrics))
    print("Baseline metrics:", json.dumps(base_metrics))
    print("Top 5 features:")
    for d in importance[:5]:
        print("  ", d["feature"], d["gain"], str(d["gain_share_pct"]) + "%")
    print("Saved:", MODEL_PATH, ",", BASELINE_PATH, ", results/metrics.json")


if __name__ == "__main__":
    main()
