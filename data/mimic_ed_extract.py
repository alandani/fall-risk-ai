"""
mimic_ed_extract.py
-------------------
Extracts fall risk features from MIMIC-IV-ED Demo (or full MIMIC-IV-ED).

Download the data first:
    cd data/
    wget -r -N -c -np https://physionet.org/files/mimic-iv-ed-demo/2.2/

Then run:
    python data/mimic_ed_extract.py

Output: data/fall_risk_patients_mimic.csv  (same schema as fall_risk_patients.csv)

Feature coverage from MIMIC-IV-ED:
  ✓ sex, systolic_bp, diastolic_bp, heart_rate
  ✓ has_parkinsons, has_osteoporosis, has_diabetes, has_dementia,
    has_depression, has_hypertension
  ✓ on_sedatives, on_diuretics, on_antihypertensives, on_anticoagulants
  ✓ prior_fall (from patient's previous ED visits)
  ✓ fall_risk label (from chief complaint + ICD W00-W19)
  ~ age, bmi, sodium, hemoglobin, bun, uses_assistive_device
    → imputed from population medians (not available in ED tables)
    → upgrade to full MIMIC-IV for these after credentialing
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
# Auto-detect the downloaded data directory
CANDIDATES = [
    SCRIPT_DIR / "physionet.org/files/mimic-iv-ed-demo/2.2/ed",
    SCRIPT_DIR / "physionet.org/files/mimic-iv-ed-demo/2.2",
    SCRIPT_DIR / "physionet.org/files/mimic-iv-ed/2.2/ed",
    SCRIPT_DIR / "physionet.org/files/mimic-iv-ed/2.2",
    SCRIPT_DIR / "mimic-iv-ed-demo/2.2/ed",
    SCRIPT_DIR / "mimic-iv-ed-demo/2.2",
    SCRIPT_DIR / "mimic-iv-ed/2.2/ed",
    SCRIPT_DIR / "mimic-iv-ed/2.2",
]

# ── ICD code prefixes → diagnosis flags ──────────────────────────────────────
ICD_MAP = {
    "has_parkinsons":    ["G20"],
    "has_osteoporosis":  ["M80", "M81"],
    "has_diabetes":      ["E10", "E11"],
    "has_dementia":      ["F01", "F02", "F03", "G30"],
    "has_depression":    ["F32", "F33"],
    "has_hypertension":  ["I10"],
}

# ICD codes W00–W19 = falls (ICD-10)
FALL_ICD_PREFIXES = [f"W{str(i).zfill(2)}" for i in range(0, 20)]

# ── Medication keyword lists ──────────────────────────────────────────────────
MED_MAP = {
    "on_sedatives": [
        "lorazepam", "diazepam", "alprazolam", "clonazepam", "midazolam",
        "temazepam", "oxazepam", "triazolam", "zolpidem", "zaleplon",
        "eszopiclone", "chlordiazepoxide", "haloperidol", "quetiapine",
        "olanzapine", "risperidone", "promethazine", "diphenhydramine",
    ],
    "on_diuretics": [
        "furosemide", "lasix", "hydrochlorothiazide", "hctz",
        "spironolactone", "bumetanide", "bumex", "torsemide",
        "metolazone", "chlorthalidone", "indapamide", "amiloride",
    ],
    "on_antihypertensives": [
        "lisinopril", "enalapril", "ramipril", "captopril", "benazepril",
        "amlodipine", "nifedipine", "diltiazem", "verapamil", "felodipine",
        "metoprolol", "atenolol", "carvedilol", "bisoprolol", "labetalol",
        "losartan", "valsartan", "irbesartan", "olmesartan", "candesartan",
        "hydralazine", "clonidine", "doxazosin", "prazosin", "terazosin",
    ],
    "on_anticoagulants": [
        "warfarin", "coumadin", "apixaban", "eliquis", "rivaroxaban",
        "xarelto", "dabigatran", "pradaxa", "heparin", "enoxaparin",
        "lovenox", "fondaparinux", "edoxaban",
    ],
}

# ── Imputed medians for features not available in ED tables ──────────────────
# Values from clinical literature / CDC STEADI population norms
IMPUTED_DEFAULTS = {
    "age":                   65.0,
    "bmi":                   27.5,
    "sodium":               138.0,
    "hemoglobin":            13.5,
    "bun":                   18.0,
    "uses_assistive_device":  0,
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def find_data_dir(override: str | None = None) -> Path:
    if override:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"Data directory not found: {p}")
        return p
    for c in CANDIDATES:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not find MIMIC-IV-ED data. Run:\n"
        "  cd data/\n"
        "  wget -r -N -c -np https://physionet.org/files/mimic-iv-ed-demo/2.2/\n"
        "Or pass --data-dir <path>"
    )


def load_table(data_dir: Path, name: str) -> pd.DataFrame:
    """Load a table, supporting both .csv.gz and .csv."""
    for ext in [".csv.gz", ".csv"]:
        p = data_dir / (name + ext)
        if p.exists():
            print(f"  Loading {p.name} …")
            return pd.read_csv(p, compression="gzip" if ext == ".csv.gz" else None,
                               low_memory=False)
    raise FileNotFoundError(f"Table '{name}' not found in {data_dir}")


def icd_flags(diag_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each stay_id, create binary columns for each diagnosis category.
    Uses ICD-10 prefix matching. ICD-9 codes are excluded (icd_version == 9).
    """
    diag10 = diag_df[diag_df["icd_version"] == 10].copy()
    diag10["icd_code"] = diag10["icd_code"].str.strip().str.upper()

    stay_flags = pd.DataFrame({"stay_id": diag_df["stay_id"].unique()})
    stay_flags = stay_flags.set_index("stay_id")

    for col, prefixes in ICD_MAP.items():
        pattern = "|".join(f"^{p}" for p in prefixes)
        matched = diag10[diag10["icd_code"].str.match(pattern, na=False)]["stay_id"].unique()
        stay_flags[col] = stay_flags.index.isin(matched).astype(int)

    # Fall ICD label
    fall_pattern = "|".join(f"^{p}" for p in FALL_ICD_PREFIXES)
    fall_stays = diag10[diag10["icd_code"].str.match(fall_pattern, na=False)]["stay_id"].unique()
    stay_flags["fall_icd"] = stay_flags.index.isin(fall_stays).astype(int)

    return stay_flags.reset_index()


def med_flags(medrecon_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each stay_id, create binary columns for each medication class.
    Matches on lower-cased medication name.
    """
    med = medrecon_df[["stay_id", "name"]].copy()
    med["name_lower"] = med["name"].str.lower().fillna("")

    stay_flags = pd.DataFrame({"stay_id": med["stay_id"].unique()})
    stay_flags = stay_flags.set_index("stay_id")

    for col, keywords in MED_MAP.items():
        pattern = "|".join(keywords)
        matched = med[med["name_lower"].str.contains(pattern, na=False)]["stay_id"].unique()
        stay_flags[col] = stay_flags.index.isin(matched).astype(int)

    return stay_flags.reset_index()


def fall_from_complaint(triage_df: pd.DataFrame) -> pd.Series:
    """Return a boolean Series (indexed by stay_id) for fall-related chief complaints."""
    fall_re = re.compile(
        r"\bfall(s|en|ing)?\b|\bfell\b|\bmechanical fall\b|\bground.?level fall\b",
        re.IGNORECASE,
    )
    mask = triage_df["chiefcomplaint"].fillna("").str.contains(fall_re)
    return triage_df.set_index("stay_id")["chiefcomplaint"].where(mask).notna()


def prior_fall_flag(edstays_df: pd.DataFrame, fall_label: pd.Series) -> pd.Series:
    """
    For each stay, check if the same patient had a prior ED visit flagged as a fall.
    fall_label: Series indexed by stay_id (True/False).
    Returns Series indexed by stay_id.
    """
    stays = edstays_df[["stay_id", "subject_id", "intime"]].copy()
    stays["intime"] = pd.to_datetime(stays["intime"])
    stays["is_fall"] = stays["stay_id"].map(fall_label).fillna(False)
    stays = stays.sort_values(["subject_id", "intime"])

    # Cumulative falls *before* current visit per patient
    stays["cum_falls"] = stays.groupby("subject_id")["is_fall"].cumsum()
    # Shift so current visit doesn't count itself
    stays["prior_fall"] = (
        stays.groupby("subject_id")["cum_falls"].shift(1).fillna(0) > 0
    ).astype(int)

    return stays.set_index("stay_id")["prior_fall"]


# ── Main extraction ───────────────────────────────────────────────────────────
def extract(data_dir: Path, out_path: Path) -> pd.DataFrame:
    print(f"\nData directory : {data_dir}")
    print(f"Output         : {out_path}\n")

    # ── Load tables ──────────────────────────────────────────────────────────
    edstays  = load_table(data_dir, "edstays")
    triage   = load_table(data_dir, "triage")
    diag     = load_table(data_dir, "diagnosis")

    # medrecon is optional (not always in demo)
    try:
        medrecon = load_table(data_dir, "medrecon")
        has_meds = True
    except FileNotFoundError:
        print("  medrecon not found — medication flags will be 0")
        has_meds = False

    # ── Base: one row per ED stay ─────────────────────────────────────────────
    base = edstays[["stay_id", "subject_id", "gender", "intime"]].copy()
    base["sex"] = base["gender"].str.upper().str[0].map({"M": "M", "F": "F"}).fillna("F")

    # ── Vitals from triage ────────────────────────────────────────────────────
    vitals = triage[["stay_id", "sbp", "dbp", "heartrate"]].copy()
    vitals.columns = ["stay_id", "systolic_bp", "diastolic_bp", "heart_rate"]
    # Clip to physiologically plausible ranges
    vitals["systolic_bp"]  = pd.to_numeric(vitals["systolic_bp"],  errors="coerce").clip(80,  200)
    vitals["diastolic_bp"] = pd.to_numeric(vitals["diastolic_bp"], errors="coerce").clip(50,  120)
    vitals["heart_rate"]   = pd.to_numeric(vitals["heart_rate"],   errors="coerce").clip(30,  200)
    # Fill missing vitals with population medians
    vitals["systolic_bp"].fillna(130,  inplace=True)
    vitals["diastolic_bp"].fillna(82,  inplace=True)
    vitals["heart_rate"].fillna(76,    inplace=True)

    # ── Diagnosis flags ───────────────────────────────────────────────────────
    dx_flags = icd_flags(diag)

    # ── Medication flags ──────────────────────────────────────────────────────
    if has_meds:
        med_flag_df = med_flags(medrecon)
    else:
        med_flag_df = pd.DataFrame({"stay_id": base["stay_id"]})
        for col in MED_MAP:
            med_flag_df[col] = 0

    # ── Fall label ────────────────────────────────────────────────────────────
    complaint_fall = fall_from_complaint(triage)        # Series[bool] by stay_id
    prior_fall     = prior_fall_flag(edstays, complaint_fall)

    # ── Merge everything ──────────────────────────────────────────────────────
    df = (
        base
        .merge(vitals,      on="stay_id", how="left")
        .merge(dx_flags,    on="stay_id", how="left")
        .merge(med_flag_df, on="stay_id", how="left")
    )

    # Attach fall signals
    df["fall_complaint"] = df["stay_id"].map(complaint_fall).fillna(False).astype(int)
    df["fall_icd"]       = df.get("fall_icd", pd.Series(0, index=df.index)).fillna(0).astype(int)
    df["prior_fall"]     = df["stay_id"].map(prior_fall).fillna(0).astype(int)

    # Final label: fall if chief complaint OR ICD code indicates fall
    df["fall_risk"] = ((df["fall_complaint"] == 1) | (df["fall_icd"] == 1)).astype(int)

    # ── Impute missing features (not available in ED tables) ─────────────────
    for col, val in IMPUTED_DEFAULTS.items():
        df[col] = val

    # Fill any remaining NaN diagnosis/med flags with 0
    flag_cols = list(ICD_MAP.keys()) + list(MED_MAP.keys())
    df[flag_cols] = df[flag_cols].fillna(0).astype(int)

    # ── Select and order final columns (matches fall_risk_patients.csv schema) ─
    out_cols = [
        "age", "sex", "bmi",
        "has_parkinsons", "has_osteoporosis", "has_diabetes",
        "has_dementia", "has_depression", "has_hypertension",
        "on_sedatives", "on_diuretics", "on_antihypertensives", "on_anticoagulants",
        "systolic_bp", "diastolic_bp", "heart_rate",
        "bun", "sodium", "hemoglobin",
        "prior_fall", "uses_assistive_device",
        "fall_risk",
    ]
    df = df[out_cols].copy()

    # ── Report ────────────────────────────────────────────────────────────────
    n_total  = len(df)
    n_falls  = df["fall_risk"].sum()
    prevalence = df["fall_risk"].mean()

    print(f"\n{'─'*50}")
    print(f"Total stays        : {n_total:,}")
    print(f"Fall cases         : {n_falls:,}  ({prevalence:.1%})")
    print(f"Missing values     : {df.isnull().sum().sum()}")
    print(f"\nDiagnosis prevalence:")
    for col in ICD_MAP:
        print(f"  {col:<25} {df[col].mean():.1%}")
    print(f"\nMedication prevalence:")
    for col in MED_MAP:
        print(f"  {col:<25} {df[col].mean():.1%}")
    print(f"\nImputed features (not in ED tables):")
    for col in IMPUTED_DEFAULTS:
        print(f"  {col}")
    print(f"\n⚠️  Upgrade to full MIMIC-IV for real age, BMI, and lab values.")
    print(f"{'─'*50}")

    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved → {out_path}")
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract fall risk features from MIMIC-IV-ED Demo")
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to MIMIC-IV-ED folder containing edstays.csv.gz etc."
    )
    parser.add_argument(
        "--out", type=str,
        default=str(SCRIPT_DIR / "fall_risk_patients_mimic.csv"),
        help="Output CSV path"
    )
    args = parser.parse_args()

    data_dir = find_data_dir(args.data_dir)
    extract(data_dir, Path(args.out))
