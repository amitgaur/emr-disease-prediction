from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EICUPaths:
    root: Path = Path("data/eicu")

    @property
    def patient(self) -> Path:
        return self.root / "patient.csv"

    @property
    def diagnosis(self) -> Path:
        return self.root / "diagnosis.csv"

    @property
    def medication(self) -> Path:
        return self.root / "medication.csv"

    @property
    def lab(self) -> Path:
        return self.root / "lab.csv"

    @property
    def vital_periodic(self) -> Path:
        return self.root / "vitalPeriodic.csv"

    @property
    def vital_aperiodic(self) -> Path:
        return self.root / "vitalAperiodic.csv"


TEMPORAL_WINDOWS_HOURS = {
    "24h": 24,
    "48h": 48,
    "7d": 168,
}

LAB_FEATURES = [
    "glucose", "BUN", "creatinine", "potassium", "sodium",
    "Hgb", "Hct", "WBC x 1000", "platelets x 1000",
    "bicarbonate", "calcium", "chloride", "magnesium",
    "phosphate", "total bilirubin", "albumin",
    "AST (SGOT)", "ALT (SGPT)", "troponin - I",
    "PT - Loss INR", "lactate",
]

VITAL_FEATURES = [
    "heartrate", "systemicsystolic", "systemicdiastolic",
    "systemicmean", "respiration", "sao2", "temperature",
]

VITAL_APERIODIC_FEATURES = [
    "noninvasivesystolic", "noninvasivediastolic", "noninvasivemean",
]

MEDICATION_DRUG_CLASSES = [
    "Cardiovascular", "Analgesics", "Anti-infectives",
    "Gastrointestinal", "Endocrine", "Respiratory",
    "Hematologic", "Central Nervous System", "Electrolytes",
    "Nutritional", "Dermatological", "Musculoskeletal",
]

CCS_CATEGORY_MAP = {
    range(1, 10): "Infectious",
    range(11, 47): "Neoplasms",
    range(47, 75): "Endocrine/Metabolic",
    range(76, 84): "Blood",
    range(85, 95): "Mental",
    range(95, 134): "Nervous/Sense",
    range(134, 145): "Nervous/Sense",
    range(145, 159): "Circulatory",
    range(159, 165): "Respiratory",
    range(165, 176): "Digestive",
    range(176, 197): "Genitourinary",
    range(197, 200): "Pregnancy",
    range(200, 213): "Skin",
    range(213, 225): "Musculoskeletal",
    range(225, 236): "Congenital",
    range(236, 245): "Perinatal",
    range(245, 260): "Symptoms/Signs",
    range(260, 261): "Residual",
    range(2601, 2622): "External",
}

READMISSION_WINDOW_DAYS = 30

TRAIN_SPLIT_YEAR = 2014
