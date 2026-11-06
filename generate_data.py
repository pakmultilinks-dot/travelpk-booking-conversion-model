"""Generate the synthetic Travel.pk booking dataset.

No real Travel.pk booking data is available, so this script creates a
fully synthetic prototype dataset with a fixed seed. Every row is a
simulated visitor session on Travel.pk (flights, hotels, tours, holiday
packages, Umrah). The label `booked` is 1 when the visitor completed a
booking and 0 when they did not.

The synthetic signal is deliberately realistic but simple: conversion
rises with more pages viewed, more time on site, earlier booking, a
returning visitor, and cheaper trips, and falls with higher prices.
It also contains realistic non-linear structure, because real booking
behaviour is not a straight line: pages viewed saturates after about
12 pages, price hurts mobile shoppers more than desktop shoppers,
returning visitors who arrive direct convert better than the sum of the
two effects, and booking too far ahead or at the last minute is worse
than a middle window. Nothing here is a real Travel.pk measurement.
"""
import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 6000
OUT_PATH = "data/travelpk_bookings_synthetic.csv"

rng = np.random.default_rng(SEED)

trip_types = ["flight", "hotel", "tour", "umrah", "holiday_package"]
devices = ["mobile", "desktop"]
channels = ["organic", "paid_search", "social", "direct"]

trip = rng.choice(trip_types, size=N_ROWS, p=[0.34, 0.26, 0.14, 0.12, 0.14])
device = rng.choice(devices, size=N_ROWS, p=[0.68, 0.32])
channel = rng.choice(channels, size=N_ROWS, p=[0.35, 0.25, 0.22, 0.18])

# Price in PKR depends on trip type, with a long right tail.
base_price = {
    "flight": 45000,
    "hotel": 28000,
    "tour": 95000,
    "umrah": 320000,
    "holiday_package": 180000,
}
price = np.array([rng.lognormal(np.log(base_price[t]), 0.45) for t in trip])
price = np.round(np.clip(price, 8000, 900000)).astype(int)

days_advance = rng.integers(1, 121, size=N_ROWS)
party_size = rng.integers(1, 7, size=N_ROWS)
trip_duration = rng.integers(1, 15, size=N_ROWS)
pages_viewed = rng.integers(1, 26, size=N_ROWS)
time_on_site = np.round(rng.exponential(6.0, size=N_ROWS), 1)  # minutes
time_on_site = np.clip(time_on_site, 0.2, 90.0)
returning = rng.choice([0, 1], size=N_ROWS, p=[0.62, 0.38])
weekend_departure = rng.choice([0, 1], size=N_ROWS, p=[0.70, 0.30])

# Latent booking probability: logistic in a small set of drivers, plus
# realistic non-linear structure (saturation, interactions, sweet spot).
log_price = np.log(price / 50000.0)
is_mobile = (device == "mobile").astype(float)
is_direct = (channel == "direct").astype(float)
logit = (
    -4.35
    + 0.16 * np.minimum(pages_viewed, 12)
    + 0.10 * time_on_site
    + 0.55 * returning
    - 0.55 * log_price
    - 0.35 * log_price * is_mobile
    + 0.50 * returning * is_direct
    - 0.00008 * (days_advance - 45.0) ** 2
    - 0.06 * party_size
    + 0.25 * (device == "desktop")
    + 0.20 * (channel == "direct")
    + 0.15 * (channel == "organic")
    - 0.30 * (trip == "umrah")
)
prob = 1.0 / (1.0 + np.exp(-logit))
booked = (rng.random(N_ROWS) < prob).astype(int)

df = pd.DataFrame(
    {
        "trip_type": trip,
        "price_pkr": price,
        "days_advance": days_advance,
        "party_size": party_size,
        "trip_duration_days": trip_duration,
        "device": device,
        "channel": channel,
        "pages_viewed": pages_viewed,
        "time_on_site_min": time_on_site,
        "returning_visitor": returning,
        "weekend_departure": weekend_departure,
        "booked": booked,
    }
)

df.to_csv(OUT_PATH, index=False)
print("Wrote", OUT_PATH, "rows:", len(df))
print("Base conversion rate: %.4f" % df["booked"].mean())
print(df["trip_type"].value_counts().to_string())
print(df.groupby("booked").size().to_string())
