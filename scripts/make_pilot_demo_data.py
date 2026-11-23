#!/usr/bin/env python3
"""Generate deterministic pilot-batch demo payloads (demo/pilot_*.json).

Continues the main demo: F3 was onboarded with correction 0 while its sensor
drifts by +500.  A device engineer now runs trial runs (pilot batches) and
calibrates power consumption while commissioning.

  B1  first batch: opened before any report -> offset 0;
      reports two trial-run summaries (the 2nd replaces the 1st).
  B2  opened after B1's report -> offset = mean residual of B1's summary.
  B3  opened *after* the shared model is re-published (F3 correction 0 -> 500)
      -> pins the new version/correction, computes its offset from B1's
      summary under the new model; B1 and B2 keep their old pins.
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "demo")

A, B, INTERCEPT, NOISE = 0.020, 0.011, 1500.0, 1.0


def trial_rows(n, seed, power_bias):
    """F3 trial-run readings: pressure base 500_016, true sensor shift +500,
    plus a power calibration bias the engineer is trying to absorb."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        s = int(round(rng.gauss(1_000_060, 25)))
        p = int(round(rng.gauss(500_016, 12)))
        y = A * s + B * (p + 500) + INTERCEPT + power_bias + rng.gauss(0, NOISE)
        rows.append((s, p, int(round(y))))
    return rows


def summary(rows):
    n = len(rows)
    return {
        "n": n,
        "sums": [sum(r[i] for r in rows) for i in range(3)],
        "outer": [[sum(r[i] * r[j] for r in rows) for j in range(3)]
                  for i in range(3)],
    }


def write(name, body):
    with open(os.path.join(OUT, name), "w") as fh:
        json.dump(body, fh, indent=1)


def main():
    os.makedirs(OUT, exist_ok=True)
    b1_first = trial_rows(180, seed=8101, power_bias=35)
    b1_fixed = trial_rows(220, seed=8102, power_bias=35)
    b2 = trial_rows(160, seed=8103, power_bias=35)

    write("pilot_open_b1.json", {"factory_id": "F3", "batch_id": "B1"})
    write("pilot_open_b2.json", {"factory_id": "F3", "batch_id": "B2"})
    write("pilot_open_b3.json", {"factory_id": "F3", "batch_id": "B3"})
    write("pilot_report_b1_first.json", summary(b1_first))
    write("pilot_report_b1_fixed.json", summary(b1_fixed))
    write("pilot_report_b2.json", summary(b2))
    # a deliberately impossible summary: sum of squares too small for the sum
    broken = summary(b2)
    broken["outer"][0][0] = 1
    write("pilot_report_b2_broken.json", broken)

    s, p, _ = b1_fixed[0]
    write("pilot_predict_b1.json",
          {"factory_id": "F3", "batch_id": "B1", "speed": s, "pressure": p})
    write("pilot_predict_b2.json",
          {"factory_id": "F3", "batch_id": "B2", "speed": s, "pressure": p})
    write("pilot_predict_b1_after_release.json",
          {"factory_id": "F3", "batch_id": "B1", "speed": s, "pressure": p})
    write("pilot_predict_b3.json",
          {"factory_id": "F3", "batch_id": "B3", "speed": s, "pressure": p})

    print(f"wrote pilot demo payloads to {os.path.abspath(OUT)}")
    print(f"B1 fixed sample: speed={s} pressure={p}")


if __name__ == "__main__":
    main()
