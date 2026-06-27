"""
llm_advisor.py
--------------
Standalone LLM advisor for fall risk predictions.

Flow:
  patient dict  →  score (XGBoost)  →  top SHAP features
                →  clinical prompt  →  LM Studio (Gemma 4)
                →  { explanation, recommendations }

LM Studio exposes an OpenAI-compatible API at http://localhost:1234/v1
Make sure LM Studio is running with Gemma 4 loaded before calling this module.

Usage:
    from llm_advisor import FallRiskAdvisor

    advisor = FallRiskAdvisor()
    result  = advisor.advise(patient)
    print(result["explanation"])
    print(result["recommendations"])
"""

import json
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from openai import OpenAI

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR  = Path(__file__).parent / "models"
SCHEMA_PATH = MODEL_DIR / "model_schema.json"

# ── LM Studio connection ─────────────────────────────────────────────────────
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY  = "lm-studio"        # LM Studio accepts any non-empty key
DEFAULT_MODEL      = "gemma-3-4b-it"    # adjust to match your loaded model name

# ── Risk thresholds ──────────────────────────────────────────────────────────
RISK_HIGH   = 0.65
RISK_MEDIUM = 0.35


def _risk_label(score: float) -> str:
    if score >= RISK_HIGH:
        return "HIGH"
    if score >= RISK_MEDIUM:
        return "MODERATE"
    return "LOW"


def _build_prompt(
    patient: dict,
    risk_score: float,
    top_shap: list[tuple[str, float]],   # [(feature_name, shap_value), ...]
) -> str:
    """
    Build a concise, structured clinical prompt.
    Top SHAP features are translated to plain-English clinical language.
    """
    label = _risk_label(risk_score)

    # ── Feature → clinical label map ─────────────────────────────────────────
    feature_labels = {
        "age":                   "Patient age",
        "bmi":                   "Body mass index (BMI)",
        "prior_fall":            "History of prior fall",
        "has_parkinsons":        "Parkinson's disease",
        "has_dementia":          "Dementia",
        "has_osteoporosis":      "Osteoporosis",
        "has_diabetes":          "Diabetes",
        "has_depression":        "Depression",
        "has_hypertension":      "Hypertension",
        "on_sedatives":          "Currently on sedatives/benzodiazepines",
        "on_diuretics":          "Currently on diuretics",
        "on_antihypertensives":  "Currently on antihypertensives",
        "on_anticoagulants":     "Currently on anticoagulants",
        "uses_assistive_device": "Uses assistive device (walker/cane)",
        "sodium":                "Serum sodium level",
        "hemoglobin":            "Hemoglobin (anemia marker)",
        "bun":                   "Blood urea nitrogen (BUN)",
        "systolic_bp":           "Systolic blood pressure",
        "diastolic_bp":          "Diastolic blood pressure",
        "heart_rate":            "Heart rate",
        "sex_M":                 "Sex (male)",
        "sex_F":                 "Sex (female)",
    }

    # Format top contributing factors
    factors_text = "\n".join(
        f"  - {feature_labels.get(name, name)}: "
        f"{'increases' if val > 0 else 'decreases'} risk "
        f"(impact score: {abs(val):.3f})"
        for name, val in top_shap
    )

    # Format patient summary
    dx = [k.replace("has_", "").replace("_", " ").title()
          for k, v in patient.items() if k.startswith("has_") and v == 1]
    meds = [k.replace("on_", "").replace("_", " ").title()
            for k, v in patient.items() if k.startswith("on_") and v == 1]

    prompt = f"""You are a clinical decision support assistant specializing in fall prevention.

## Patient Summary
- Age: {patient.get('age')} | Sex: {patient.get('sex')} | BMI: {patient.get('bmi')}
- Diagnoses: {', '.join(dx) if dx else 'None documented'}
- Active medications: {', '.join(meds) if meds else 'None documented'}
- Prior fall history: {'Yes' if patient.get('prior_fall') else 'No'}
- Uses assistive device: {'Yes' if patient.get('uses_assistive_device') else 'No'}
- Key labs: Na {patient.get('sodium')} mEq/L | Hgb {patient.get('hemoglobin')} g/dL | BUN {patient.get('bun')} mg/dL
- Vitals: BP {patient.get('systolic_bp')}/{patient.get('diastolic_bp')} mmHg | HR {patient.get('heart_rate')} bpm

## AI Risk Assessment
- Fall risk score: {risk_score:.1%}
- Risk level: **{label}**

## Top Contributing Factors (from SHAP analysis)
{factors_text}

## Your Task
Respond with ONLY a valid JSON object — no markdown, no code fences, no extra text.

The JSON must have exactly these two keys:
- "explanation": 2–3 sentences in plain clinical language explaining why this patient received a {label} fall risk score, referencing the specific contributing factors above.
- "recommendations": a single string containing 4–6 specific, evidence-based interventions as a newline-separated bullet list (each line starting with "- "), tailored to this patient's risk factors. Be concrete — name specific medication classes to review, referrals to make, environmental modifications, or monitoring parameters.

Keep the tone professional and suitable for a clinical handoff note.

Example format (do not copy content, only structure):
{{"explanation": "...", "recommendations": "- ...\\n- ...\\n- ..."}}
"""
    return prompt.strip()


class FallRiskAdvisor:
    """
    Loads saved XGBoost model + preprocessor, scores patients,
    and generates LLM-powered clinical explanations and recommendations.
    """

    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
        lm_studio_url: str = LM_STUDIO_BASE_URL,
        model_name: str = DEFAULT_MODEL,
        top_n_shap: int = 6,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        self.top_n_shap   = top_n_shap
        self.model_name   = model_name
        self.temperature  = temperature
        self.max_tokens   = max_tokens

        # Load schema
        schema = json.loads((model_dir / "model_schema.json").read_text())
        self.feature_names = schema["feature_names"]
        self.num_features  = schema["num_features"]
        self.bin_features  = schema["bin_features"]
        self.cat_features  = schema["cat_features"]
        self.all_features  = self.num_features + self.bin_features + self.cat_features

        # Load ML artifacts
        self.model        = joblib.load(model_dir / "fall_risk_xgb.joblib")
        self.preprocessor = joblib.load(model_dir / "preprocessor.joblib")
        self.explainer    = shap.TreeExplainer(self.model)

        # LM Studio client (OpenAI-compatible)
        self.client = OpenAI(base_url=lm_studio_url, api_key=LM_STUDIO_API_KEY)

        print(f"FallRiskAdvisor ready — model: {self.model_name} @ {lm_studio_url}")

    def score(self, patient: dict) -> tuple[float, list[tuple[str, float]]]:
        """
        Returns (risk_score, top_shap_features).
        top_shap_features: [(feature_name, shap_value), ...] sorted by |impact|
        """
        df = pd.DataFrame([patient])[self.all_features]
        X  = self.preprocessor.transform(df)

        risk_score = float(self.model.predict_proba(X)[0, 1])

        sv = self.explainer(X)
        shap_series = pd.Series(sv.values[0], index=self.feature_names)
        top_shap = (
            shap_series.abs()
            .nlargest(self.top_n_shap)
            .index
            .tolist()
        )
        top_shap_pairs = [(f, float(shap_series[f])) for f in top_shap]

        return risk_score, top_shap_pairs

    def _call_llm(self, prompt: str) -> str:
        """Call LM Studio and return the raw text response."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise, evidence-based clinical decision support assistant. "
                        "Always ground recommendations in the specific patient data provided. "
                        "Never fabricate lab values or diagnoses not in the summary."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        msg = response.choices[0].message
        # Thinking models (e.g. Gemma 4) put chain-of-thought in reasoning_content
        # and the actual answer in content. If content is empty the model ran out of
        # tokens during reasoning — extract the JSON directly from reasoning_content.
        content = (msg.content or "").strip()
        if not content:
            reasoning = getattr(msg, "reasoning_content", None) or ""
            import re
            match = re.search(r'\{.*\}', reasoning, re.DOTALL)
            content = match.group(0) if match else ""
        return content

    def _parse_response(self, text: str) -> dict:
        """
        Parse the JSON response from the LLM.
        Falls back to returning the full text as explanation if JSON parsing fails.
        """
        import re

        explanation     = ""
        recommendations = ""

        # Strip markdown code fences if the model adds them anyway
        cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip(), flags=re.MULTILINE).strip()

        try:
            data            = json.loads(cleaned)
            explanation     = data.get("explanation", "").strip()
            recommendations = data.get("recommendations", "").strip()
        except (json.JSONDecodeError, AttributeError):
            # Last-resort fallback: show the raw response so the user can debug
            explanation = text

        return {
            "explanation":      explanation,
            "recommendations":  recommendations,
            "full_response":    text,
        }

    def advise(self, patient: dict) -> dict:
        """
        Full pipeline: score → SHAP → prompt → LLM → parsed output.

        Returns:
            {
                "risk_score":       float,          # 0.0–1.0
                "risk_label":       str,            # HIGH / MODERATE / LOW
                "top_shap":         list[tuple],    # [(name, shap_val), ...]
                "explanation":      str,
                "recommendations":  str,
                "full_response":    str,
            }
        """
        risk_score, top_shap = self.score(patient)
        prompt   = _build_prompt(patient, risk_score, top_shap)
        raw_text = self._call_llm(prompt)
        parsed   = self._parse_response(raw_text)

        return {
            "risk_score":      risk_score,
            "risk_label":      _risk_label(risk_score),
            "top_shap":        top_shap,
            **parsed,
        }

    def advise_batch(self, patients: list[dict]) -> list[dict]:
        """Score and advise a list of patients."""
        return [self.advise(p) for p in patients]
