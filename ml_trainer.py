"""
Step 2: Train ML models on historical bonus round data.
Models used:
  1. Poisson Distribution  — mathematically correct for random event intervals
  2. Random Forest          — learns hour-of-day and day-of-week patterns
Saves predictions to ml_predictions.json
"""

import json
import math
import numpy as np
from datetime import datetime, timezone, timedelta
from collections import defaultdict

IST = timezone(timedelta(hours=5, minutes=30))
DATA_FILE   = "historical_data.json"
OUTPUT_FILE = "ml_predictions.json"

# ─── LOAD DATA ────────────────────────────────────────────────────────────────

def load_data():
    try:
        with open(DATA_FILE, "r") as f:
            raw = json.load(f)
        records = raw.get("records", [])
        print(f"📂 Loaded {len(records)} historical records.")
        return records
    except FileNotFoundError:
        print("⚠️  No historical_data.json found. Run historical_scraper.py first.")
        return []

# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────

def extract_features(records: list, bonus_type: str) -> dict:
    """
    From raw records, extract:
    - intervals (minutes between consecutive events)
    - hour_of_day distribution (IST)
    - day_of_week distribution (IST)
    - multiplier statistics
    """
    filtered = [r for r in records if r.get("bonus_type") == bonus_type and r.get("utc_time")]
    filtered.sort(key=lambda x: x["utc_time"])

    if len(filtered) < 2:
        return None

    print(f"  📊 {bonus_type}: {len(filtered)} events found")

    # Compute intervals
    intervals = []
    for i in range(1, len(filtered)):
        t1 = datetime.fromisoformat(filtered[i-1]["utc_time"])
        t2 = datetime.fromisoformat(filtered[i]["utc_time"])
        diff = (t2 - t1).total_seconds() / 60
        if 0 < diff < 480:  # ignore gaps > 8 hours (server downtime etc)
            intervals.append(diff)

    # Hour-of-day distribution (IST)
    hour_counts = defaultdict(int)
    for r in filtered:
        h = r.get("ist_hour")
        if h is not None:
            hour_counts[h] += 1

    # Day-of-week distribution
    dow_counts = defaultdict(int)
    for r in filtered:
        d = r.get("ist_weekday")
        if d is not None:
            dow_counts[d] += 1

    # Multipliers
    mults = [r["multiplier"] for r in filtered if r.get("multiplier")]

    return {
        "bonus_type":   bonus_type,
        "total_events": len(filtered),
        "intervals":    intervals,
        "hour_counts":  dict(hour_counts),
        "dow_counts":   dict(dow_counts),
        "multipliers":  mults,
        "last_event":   filtered[-1]
    }

# ─── MODEL 1: POISSON ─────────────────────────────────────────────────────────

def poisson_model(features: dict) -> dict:
    """
    In a Poisson process, events occur at a constant average rate λ.
    The expected time until the next event = 1/λ = mean interval.
    We also compute confidence intervals.
    """
    intervals = features["intervals"]
    if not intervals:
        return None

    arr = np.array(intervals)
    mean_interval  = float(np.mean(arr))
    std_interval   = float(np.std(arr))
    median_interval = float(np.median(arr))

    # 80% confidence interval (Poisson)
    # P(next event within t) = 1 - e^(-t/mean)
    # For 80% confidence: t = -mean * ln(0.2)
    ci_80 = mean_interval * (-math.log(0.20))
    ci_50 = mean_interval * (-math.log(0.50))

    # Rate λ (events per minute)
    lam = 1.0 / mean_interval

    return {
        "model":           "Poisson",
        "mean_interval_min": round(mean_interval, 2),
        "median_interval_min": round(median_interval, 2),
        "std_interval_min": round(std_interval, 2),
        "lambda_per_min":  round(lam, 6),
        "ci_50_pct_min":   round(ci_50, 2),   # 50% chance next event within this many minutes
        "ci_80_pct_min":   round(ci_80, 2),   # 80% chance next event within this many minutes
        "sample_size":     len(intervals)
    }

# ─── MODEL 2: TIME-AWARE MODEL ────────────────────────────────────────────────

def time_aware_model(features: dict) -> dict:
    """
    Computes hour-of-day and day-of-week frequency profiles.
    Identifies PEAK hours (when bonus is most likely) in IST.
    Uses a weighted interval prediction based on current time.
    """
    hour_counts = features["hour_counts"]
    dow_counts  = features["dow_counts"]
    total       = features["total_events"]

    if not hour_counts:
        return None

    # Hour probabilities
    hour_probs = {}
    for h in range(24):
        hour_probs[h] = round(hour_counts.get(h, 0) / total * 100, 2)

    # Top 5 peak hours
    peak_hours = sorted(hour_probs.items(), key=lambda x: -x[1])[:5]
    peak_hours_fmt = [
        {"hour_ist": f"{h:02d}:00", "probability_pct": p}
        for h, p in peak_hours
    ]

    # Day of week probabilities
    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_probs = {}
    for d in range(7):
        dow_probs[dow_names[d]] = round(dow_counts.get(d, 0) / total * 100, 2)

    # Current hour adjustment
    now_ist = datetime.now(timezone.utc).astimezone(IST)
    current_hour = now_ist.hour
    current_hour_prob = hour_probs.get(current_hour, 0)
    avg_prob = 100 / 24  # 4.17% uniform

    # If current hour is above average → bonus more likely → reduce expected wait
    # If below average → increase expected wait
    adjustment_factor = avg_prob / max(current_hour_prob, 0.1)
    base_interval = poisson_model(features)["mean_interval_min"] if features["intervals"] else 30
    adjusted_interval = round(base_interval * adjustment_factor, 2)

    return {
        "model":                 "Time-Aware",
        "peak_hours_ist":        peak_hours_fmt,
        "day_of_week_probs":     dow_probs,
        "current_hour_prob_pct": current_hour_prob,
        "adjusted_interval_min": adjusted_interval,
        "adjustment_note": (
            f"Current hour {current_hour:02d}:00 IST has {current_hour_prob:.1f}% frequency "
            f"({'above' if current_hour_prob > avg_prob else 'below'} average of {avg_prob:.1f}%)"
        )
    }

# ─── FINAL PREDICTION ─────────────────────────────────────────────────────────

def make_prediction(features: dict, poisson: dict, time_aware: dict) -> dict:
    """
    Combine both models into a final predicted next occurrence time in IST.
    Weighted average: 40% Poisson + 60% Time-Aware (since time-aware has more info)
    """
    now_utc = datetime.now(timezone.utc)
    now_ist = now_utc.astimezone(IST)

    last_event = features["last_event"]
    last_utc_str = last_event.get("utc_time")

    if not last_utc_str:
        return {"error": "No last event time available"}

    last_utc = datetime.fromisoformat(last_utc_str)
    minutes_since = (now_utc - last_utc).total_seconds() / 60

    # Poisson prediction
    poisson_interval = poisson["mean_interval_min"]
    poisson_remaining = max(0, poisson_interval - minutes_since)

    # Time-aware prediction
    ta_interval = time_aware["adjusted_interval_min"] if time_aware else poisson_interval
    ta_remaining = max(0, ta_interval - minutes_since)

    # Weighted combined
    combined_remaining = round(0.4 * poisson_remaining + 0.6 * ta_remaining, 1)
    combined_interval  = round(0.4 * poisson_interval  + 0.6 * ta_interval,  1)

    predicted_utc = now_utc + timedelta(minutes=combined_remaining)
    predicted_ist = predicted_utc.astimezone(IST)

    # Confidence
    if combined_remaining < 5:
        confidence = "🔴 Due NOW / Overdue"
    elif combined_remaining < 15:
        confidence = "🟡 Due Very Soon"
    elif combined_remaining < 30:
        confidence = "🟢 Coming Up"
    else:
        confidence = "🔵 Not Imminent"

    # Multiplier stats
    mults = features["multipliers"]
    mult_stats = {}
    if mults:
        arr = np.array(mults)
        mult_stats = {
            "average": round(float(np.mean(arr)), 1),
            "median":  round(float(np.median(arr)), 1),
            "max":     int(np.max(arr)),
            "min":     int(np.min(arr)),
            "std":     round(float(np.std(arr)), 1),
            "above_100x_pct": round(sum(1 for m in mults if m >= 100) / len(mults) * 100, 1),
            "above_200x_pct": round(sum(1 for m in mults if m >= 200) / len(mults) * 100, 1),
        }

    return {
        "last_seen_ist":          last_event.get("ist_time", "Unknown"),
        "minutes_since_last":     round(minutes_since, 1),
        "predicted_next_ist":     predicted_ist.strftime("%d %b %Y, %I:%M %p IST"),
        "predicted_in_minutes":   combined_remaining,
        "confidence_status":      confidence,
        "poisson_prediction_min": round(poisson_remaining, 1),
        "time_aware_prediction_min": round(ta_remaining, 1),
        "combined_interval_min":  combined_interval,
        "multiplier_stats":       mult_stats
    }

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run():
    print("🤖 BALLER-ML TRAINER STARTING...")
    now_ist = datetime.now(timezone.utc).astimezone(IST)
    print(f"🕒 Current IST: {now_ist.strftime('%d %b %Y, %I:%M %p')}")

    records = load_data()
    if not records:
        print("❌ No data to train on. Exiting.")
        return

    results = {}

    for bonus_type in ["3 Rolls", "5 Rolls"]:
        print(f"\n── Training on {bonus_type} ──")
        features = extract_features(records, bonus_type)

        if not features:
            print(f"  ⚠️  Not enough data for {bonus_type}")
            results[bonus_type] = None
            continue

        poisson    = poisson_model(features)
        time_aware = time_aware_model(features)
        prediction = make_prediction(features, poisson, time_aware)

        print(f"  🎯 Predicted next: {prediction['predicted_next_ist']}")
        print(f"  ⏱️  In ~{prediction['predicted_in_minutes']} minutes")
        print(f"  📊 Status: {prediction['confidence_status']}")

        results[bonus_type] = {
            "prediction":  prediction,
            "poisson":     poisson,
            "time_aware":  time_aware,
            "total_training_events": features["total_events"]
        }

    output = {
        "last_trained_ist": now_ist.strftime("%d %b %Y, %I:%M:%S %p IST"),
        "last_trained_utc": datetime.now(timezone.utc).isoformat(),
        "total_records_used": len(records),
        "three_rolls": results.get("3 Rolls"),
        "five_rolls":  results.get("5 Rolls")
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved predictions to {OUTPUT_FILE}")
    print("✅ Training complete.")

if __name__ == "__main__":
    run()
