"""
generate_fall_risk_data.py
--------------------------
Generates a synthetic Synthea-style fall risk dataset (CSV).
No external EHR access required.

Usage:
    python data/generate_fall_risk_data.py            # 2000 patients (default)
    python data/generate_fall_risk_data.py --n 5000   # custom size
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Clinical priors (evidence-based weights)
# ---------------------------------------------------------------------------
# Risk factors drawn from Morse Fall Scale + CDC STEADI framework:
#   age, Parkinson's, dementia, sedatives, diuretics, low sodium, low Hgb

def generate_patients(n: int = 2000) -> pd.DataFrame:
    # ── Demographics ────────────────────────────────────────────────────────
    age   = RNG.integers(18, 96, size=n)
    sex   = RNG.choice(["M", "F"], size=n)
    bmi   = RNG.normal(27.5, 5.5, size=n).clip(15, 55).round(1)

    # ── Diagnoses (prevalence rises with age) ──────────────────────────────
    age_norm = (age - 18) / (95 - 18)          # 0→1
    has_parkinsons    = RNG.random(n) < (0.01 + 0.09 * age_norm)
    has_osteoporosis  = RNG.random(n) < (0.05 + 0.35 * age_norm)
    has_diabetes      = RNG.random(n) < (0.08 + 0.22 * age_norm)
    has_dementia      = RNG.random(n) < (0.00 + 0.18 * age_norm)
    has_depression    = RNG.random(n) < 0.15
    has_hypertension  = RNG.random(n) < (0.10 + 0.50 * age_norm)

    # ── Medications (correlated with diagnoses) ─────────────────────────────
    on_sedatives         = RNG.random(n) < (0.10 + 0.20 * has_dementia.astype(float) + 0.10 * has_depression.astype(float))
    on_diuretics         = RNG.random(n) < (0.10 + 0.40 * has_hypertension.astype(float))
    on_antihypertensives = has_hypertension & (RNG.random(n) < 0.80)
    on_anticoagulants    = RNG.random(n) < (0.05 + 0.15 * age_norm)

    # ── Vitals & Labs ───────────────────────────────────────────────────────
    systolic_bp  = RNG.normal(130, 18, size=n).clip(80, 200).round(0).astype(int)
    diastolic_bp = RNG.normal(82,  12, size=n).clip(50, 120).round(0).astype(int)
    heart_rate   = RNG.normal(76,  14, size=n).clip(40, 130).round(0).astype(int)
    # BUN elevated in dehydration/renal impairment → fall risk
    bun          = RNG.normal(18,   8, size=n).clip(5,  80).round(1)
    # Sodium: hyponatremia (<135) increases fall risk
    sodium       = RNG.normal(138,  4, size=n).clip(120, 148).round(1)
    # Hemoglobin: anemia → dizziness → falls
    hemoglobin   = RNG.normal(13.5, 2, size=n).clip(6,  18).round(1)

    # ── Functional ─────────────────────────────────────────────────────────
    prior_fall   = RNG.random(n) < (0.15 + 0.30 * age_norm)    # strongest single predictor
    uses_assistive_device = RNG.random(n) < (0.05 + 0.35 * age_norm)

    # ── Synthesize label with clinical logic ───────────────────────────────
    # Logit-based model so risk is probabilistic, not deterministic
    logit = (
        -3.5
        + 0.045  * (age - 65)
        + 0.30   * (bmi < 18.5).astype(float)           # underweight
        + 1.50   * has_parkinsons.astype(float)
        + 0.60   * has_osteoporosis.astype(float)
        + 0.40   * has_diabetes.astype(float)
        + 1.20   * has_dementia.astype(float)
        + 0.50   * has_depression.astype(float)
        + 1.10   * on_sedatives.astype(float)
        + 0.70   * on_diuretics.astype(float)
        + 0.40   * on_antihypertensives.astype(float)
        + 0.55   * (sodium < 135).astype(float)          # hyponatremia
        + 0.45   * (hemoglobin < 10).astype(float)       # anemia
        + 0.35   * (bun > 30).astype(float)              # elevated BUN
        + 2.00   * prior_fall.astype(float)
        + 0.90   * uses_assistive_device.astype(float)
        + RNG.normal(0, 0.4, size=n)                      # residual noise
    )
    prob = 1 / (1 + np.exp(-logit))
    fall_risk = (RNG.random(n) < prob).astype(int)

    df = pd.DataFrame({
        # Demographics
        "age":    age,
        "sex":    sex,
        "bmi":    bmi,
        # Diagnoses
        "has_parkinsons":    has_parkinsons.astype(int),
        "has_osteoporosis":  has_osteoporosis.astype(int),
        "has_diabetes":      has_diabetes.astype(int),
        "has_dementia":      has_dementia.astype(int),
        "has_depression":    has_depression.astype(int),
        "has_hypertension":  has_hypertension.astype(int),
        # Medications
        "on_sedatives":          on_sedatives.astype(int),
        "on_diuretics":          on_diuretics.astype(int),
        "on_antihypertensives":  on_antihypertensives.astype(int),
        "on_anticoagulants":     on_anticoagulants.astype(int),
        # Vitals / Labs
        "systolic_bp":  systolic_bp,
        "diastolic_bp": diastolic_bp,
        "heart_rate":   heart_rate,
        "bun":          bun,
        "sodium":       sodium,
        "hemoglobin":   hemoglobin,
        # Functional
        "prior_fall":            prior_fall.astype(int),
        "uses_assistive_device": uses_assistive_device.astype(int),
        # Target
        "fall_risk": fall_risk,
    })

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000, help="Number of patients")
    args = parser.parse_args()

    out_path = Path(__file__).parent / "fall_risk_patients.csv"
    df = generate_patients(args.n)
    df.to_csv(out_path, index=False)

    prevalence = df["fall_risk"].mean()
    print(f"✓ Generated {len(df):,} patients → {out_path}")
    print(f"  Fall risk prevalence: {prevalence:.1%}  ({df['fall_risk'].sum()} cases)")
    print(f"  Columns: {', '.join(df.columns)}")
