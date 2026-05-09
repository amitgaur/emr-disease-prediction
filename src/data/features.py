from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from configs.eicu_config import (
    CCS_CATEGORY_MAP,
    MEDICATION_DRUG_CLASSES,
    VITAL_FEATURES,
    VITAL_APERIODIC_FEATURES,
)


class FeatureEngineer:

    def __init__(self):
        self._lab_scaler: StandardScaler | None = None
        self._icd9_to_ccs = _build_icd9_to_ccs()

    # ------------------------------------------------------------------
    # Lab normalization
    # ------------------------------------------------------------------

    def normalize_labs(
        self, df: pd.DataFrame, lab_columns: list[str], *, fit: bool = True
    ) -> pd.DataFrame:
        present = [c for c in lab_columns if c in df.columns]
        if not present:
            return df

        if fit:
            self._lab_scaler = StandardScaler()
            df[present] = self._lab_scaler.fit_transform(df[present])
        elif self._lab_scaler is not None:
            df[present] = self._lab_scaler.transform(df[present])
        else:
            raise RuntimeError("normalize_labs called with fit=False before fitting")

        return df

    # ------------------------------------------------------------------
    # Diagnosis grouping  (ICD-9 / ICD-10 → CCS category)
    # ------------------------------------------------------------------

    def encode_diagnoses(self, diagnoses: pd.DataFrame) -> pd.DataFrame:
        diag = diagnoses.copy()
        diag["icd_code"] = diag["icd9code"].fillna("").astype(str).str.strip()
        diag["ccs_category"] = diag["icd_code"].map(self._map_icd_to_ccs)

        pivoted = (
            diag.groupby(["patientunitstayid", "ccs_category"])
            .size()
            .unstack(fill_value=0)
        )
        pivoted.columns = [f"dx_{c}" for c in pivoted.columns]
        return pivoted.clip(upper=1)

    def _map_icd_to_ccs(self, code: str) -> str:
        if not code or code == "nan":
            return "Unknown"

        numeric = _extract_numeric_icd(code)
        if numeric is not None:
            for rng, label in self._icd9_to_ccs.items():
                if numeric in rng:
                    return label
        return "Other"

    # ------------------------------------------------------------------
    # Medication aggregation (drug class binary features)
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_medications(medications: pd.DataFrame) -> pd.DataFrame:
        meds = medications.copy()
        drug_string = meds["drugname"].fillna("").str.lower()

        records: list[dict] = []
        for stay_id, group in meds.groupby("patientunitstayid"):
            row: dict = {"patientunitstayid": stay_id}
            combined = " ".join(group["drugname"].fillna("").str.lower())
            for cls in MEDICATION_DRUG_CLASSES:
                row[f"med_{cls}"] = int(cls.lower() in combined)
            row["med_unique_count"] = group["drugname"].nunique()
            row["med_total_orders"] = len(group)
            records.append(row)

        if not records:
            return pd.DataFrame(columns=["patientunitstayid"])

        return pd.DataFrame(records).set_index("patientunitstayid")

    # ------------------------------------------------------------------
    # Vital sign aggregation  (min / max / mean / std)
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_vitals(
        vitals_periodic: pd.DataFrame,
        vitals_aperiodic: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        agg_frames: list[pd.DataFrame] = []

        if not vitals_periodic.empty:
            cols = [c for c in VITAL_FEATURES if c in vitals_periodic.columns]
            if cols:
                agg = (
                    vitals_periodic.groupby("patientunitstayid")[cols]
                    .agg(["min", "max", "mean", "std"])
                )
                agg.columns = [f"vital_{c}_{s}" for c, s in agg.columns]
                agg_frames.append(agg)

        if vitals_aperiodic is not None and not vitals_aperiodic.empty:
            cols = [c for c in VITAL_APERIODIC_FEATURES if c in vitals_aperiodic.columns]
            if cols:
                agg = (
                    vitals_aperiodic.groupby("patientunitstayid")[cols]
                    .agg(["min", "max", "mean", "std"])
                )
                agg.columns = [f"vital_{c}_{s}" for c, s in agg.columns]
                agg_frames.append(agg)

        if not agg_frames:
            return pd.DataFrame()

        return pd.concat(agg_frames, axis=1)


# ======================================================================
# Helpers
# ======================================================================

def _build_icd9_to_ccs() -> dict[range, str]:
    return dict(CCS_CATEGORY_MAP)


def _extract_numeric_icd(code: str) -> int | None:
    digits = "".join(c for c in code.split(".")[0] if c.isdigit())
    if digits:
        try:
            return int(digits)
        except ValueError:
            return None
    return None
