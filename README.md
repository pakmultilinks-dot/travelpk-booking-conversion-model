# Travel.pk Booking Conversion Prediction Model

Predicts whether a Travel.pk visitor will complete a booking (1) or not (0),
from session and trip features. The shipped model is XGBoost, served behind
a FastAPI endpoint. A logistic regression baseline is trained alongside it
for comparison.

## Data: synthetic prototype, no real Travel.pk data

No real Travel.pk booking data was available for this task, so the model is
a prototype trained on fully synthetic data. Nothing in the results below
is a real Travel.pk measurement.

- Generator: `generate_data.py`, fixed seed 42, reproducible.
- File: `data/travelpk_bookings_synthetic.csv`, 6,000 rows.
- Base conversion rate: 12.37% (742 bookings out of 6,000 sessions).
- Target: `booked`, 1 when the visitor completed a booking, 0 when not.
- Features (11): `trip_type` (flight, hotel, tour, umrah, holiday_package),
  `price_pkr`, `days_advance`, `party_size`, `trip_duration_days`,
  `device` (mobile, desktop), `channel` (organic, paid_search, social,
  direct), `pages_viewed`, `time_on_site_min`, `returning_visitor`,
  `weekend_departure`.
- The synthetic signal is documented in `generate_data.py`: conversion rises
  with pages viewed, time on site, returning visitors and cheaper trips, and
  includes realistic non-linear structure (pages viewed saturates after 12,
  price hurts mobile shoppers more, returning direct visitors convert better,
  booking too early or too late is worse than a middle window).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

```bash
# 1. Regenerate the synthetic data (seed 42, reproducible)
python generate_data.py

# 2. Train and evaluate (stratified 80/20 split, test set unseen)
python train.py

# 3. Start the API
uvicorn api:app --port 8000

# 4. Test the API
python test_api.py
```

## Results

Measured on the held-out test set: 1,200 rows, 148 positives, stratified,
never used in training. Threshold 0.5 for all precision, recall and F1
figures.

| Metric | XGBoost | Logistic regression baseline |
| --- | --- | --- |
| Accuracy | 0.8725 | 0.8783 |
| Precision | 0.4000 | 0.5625 |
| Recall | 0.0676 | 0.0608 |
| F1 | 0.1156 | 0.1098 |
| ROC-AUC | 0.7280 | 0.7405 |
| Confusion matrix (TN, FP, FN, TP) | 1037, 15, 138, 10 | 1045, 7, 139, 9 |

Full metrics and test predictions: `results/metrics.json`,
`results/metrics.txt`, `results/test_predictions.csv`.

## Best model

By test ROC-AUC, the best model is the logistic regression baseline
(0.7405 vs 0.7280 for XGBoost), and it also leads on accuracy and
precision. XGBoost leads only on F1 (0.1156 vs 0.1098) and recall
(0.0676 vs 0.0608). The baseline wins here because the synthetic data
generator is mostly a logistic function, which plays to a linear model's
strengths; the tree model has little extra non-linear signal to find in
this synthetic data. With real booking data, which has stronger
interactions and thresholds, XGBoost is expected to do better. The API
ships the XGBoost model as the task requires, and both models are saved
(`model_xgboost.pkl`, `model_baseline.pkl`).

Recall is low for both models at the 0.5 threshold because the positive
class is only 12.37% of sessions. For a real deployment the threshold
should be tuned for the business cost of missing a booker versus
flagging a non-booker.

## Feature importance (XGBoost, gain)

| Feature | Gain | Share |
| --- | --- | --- |
| price_pkr | 11.43 | 11.24% |
| time_on_site_min | 10.86 | 10.68% |
| returning_visitor | 9.30 | 9.14% |
| pages_viewed | 7.60 | 7.47% |
| trip_type_hotel | 7.13 | 7.01% |

Full list: `results/metrics.json`. Price, time on site and returning
visitor status drive the prediction most; party size and channel matter
least in this synthetic data.

## API

`GET /health` returns `{"status": "ok", "model": "xgboost", ...}`.

`POST /predict` accepts one visitor session and returns the booking
probability and label:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"trip_type": "flight", "price_pkr": 28000, "days_advance": 45,
       "party_size": 2, "trip_duration_days": 4, "device": "desktop",
       "channel": "direct", "pages_viewed": 18, "time_on_site_min": 22.5,
       "returning_visitor": 1, "weekend_departure": 0}'
```

Response:

```json
{"booking_probability": 0.6909, "predicted_label": 1, "predicted_text": "likely to book"}
```

Verified API test (`results/api_test.json`, run 6 Nov 2026):

| Case | Probability | Label |
| --- | --- | --- |
| Likely to book (desktop, direct, returning, 18 pages, 22.5 min) | 0.6909 | 1, likely to book |
| Unlikely to book (Umrah, PKR 420,000, 3 days ahead, 2 pages, 0.8 min) | 0.0031 | 0, unlikely to book |
| Edge case, minimal engagement (hotel, PKR 12,000, 1 page, 0.2 min) | 0.0407 | 0, unlikely to book |

## What I would improve

1. Real data first. Replace the synthetic set with real Travel.pk sessions
   and bookings, with consent and privacy handling, and re-train. Every
   number in this README changes when real data arrives.
2. More features. Search destination, route and date flexibility, fare
   class, promo use, payment method, past booking history, and session
   sequence features instead of single-session aggregates.
3. Threshold and class imbalance. Tune the decision threshold on a
   validation set for the business goal (for example maximise F1, or fix
   recall at 80%), and test class weights or resampling.
4. Calibration. Calibrate predicted probabilities (Platt or isotonic) so
   the probability means what it says before it drives offers or ranking.
5. Better validation. Time-based split instead of random, so the model is
   tested on future weeks, plus cross-validation for stable estimates.
6. Monitoring. Track live conversion, prediction drift and feature drift,
   and retrain on a schedule or when drift crosses a limit.
7. A/B test. Ship the model to a small share of traffic first and measure
   real booking lift against the current experience before full rollout.
8. Model comparison. Add gradient boosting variants and a small neural
   net, and pick by ROC-AUC and calibration on real data, not by default.

## Limitations

- All data is synthetic; results do not describe real Travel.pk visitors.
- The synthetic base rate (12.37%) is higher than typical real travel
  booking conversion, so probabilities are not calibrated to real traffic.
- Only 11 features, no real user history, destination or payment data.
- Single random split, no time-based validation yet.
- The API runs locally only; no live deployment is included.
