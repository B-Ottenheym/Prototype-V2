from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

EXPERIMENT_MODE: bool = True

CONDITIONS = [
    "Black box",
    "SHAP",
    "Anchors",
    "Tegenfeitelijk",
    "Surrogaatmodel (beslisboom)",
]

QUALTRICS_BASE_URL = "https://qualtricsxmp3nt7g5jg.qualtrics.com/jfe/form/SV_862H83gyRZHYjoa"

@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    title: str
    narrative_markdown: str
    image_path: str | None
    features: dict


# --- Scenario(s) ---
# Start with ONE standardised scenario to match your current procedure.
# You can add more later by appending to SCENARIOS.
SCENARIOS = [
    Scenario(
        scenario_id="S1",
        title="Scenario 1",
        narrative_markdown=(
            """### Project scenario\n"""
            "U beoordeelt een bouwproject in de fase voorafgaand aan de uitvoering.\n\n"
            "Op basis van beschikbare projectinformatie wordt een inschatting gemaakt van het risico op projectvertraging.\n\n"
            "HIER SCENARIO BESCHRIJVING\n"
        ),
        image_path=None,  # e.g., "assets/scenario_S1.png"
        features={
            # TODO: Replace with the fixed feature values you want to show.
            # These keys should correspond to your *raw* input columns before encoding.
            # Example numeric:
            "planned_duration_days": 365,
            "contract_value_million": 20,
            "project_size_m2": 20000,
            # Example categorical:
            "project_type": "Wonen",
            "contract_award_method": "Laagste prijs",
            "contract_type": "Lump-sum",
            "use_of_bim": 1,
            "consultant_prior_collaboration": 1,
            # Add remaining features as needed...
        },
    )
]


# --- Precomputed artifacts directory ---
# Put pre-generated images/text/tables here, keyed by scenario_id and condition.
# For example:
#   artifacts/S1/shap.png
#   artifacts/S1/anchors.txt
#   artifacts/S1/counterfactual.csv
#   artifacts/S1/surrogate.png
ARTIFACTS_DIR = Path("artifacts")


def artifact_path(scenario_id: str, condition: str) -> Path:
    base = ARTIFACTS_DIR / scenario_id
    if condition == "SHAP":
        return base / "shap.png"
    if condition == "Anchors":
        return base / "anchors.txt"
    if condition == "Tegenfeitelijk":
        return base / "counterfactual.csv"
    if condition == "Surrogaatmodel (beslisboom)":
        return base / "surrogate.png"
    return base / ""  # black_box