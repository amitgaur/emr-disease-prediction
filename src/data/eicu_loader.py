from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from configs.eicu_config import (
    EICUPaths,
    LAB_FEATURES,
    READMISSION_WINDOW_DAYS,
    TEMPORAL_WINDOWS_HOURS,
    TRAIN_SPLIT_YEAR,
)
from src.data.features import FeatureEngineer

logger = logging.getLogger(__name__)


class EICULoader:

    def __init__(self, paths: EICUPaths | None = None):
        self.paths = paths or EICUPaths()
        self.fe = FeatureEngineer()
        self._raw: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> dict[str, pd.DataFrame]:
        self._raw = {
            "patient": self._read("patient"),
            "diagnosis": self._read("diagnosis"),
            "medication": self._read("medication"),
            "lab": self._read("lab"),
            "vital_periodic": self._read("vital_periodic"),
            "vital_aperiodic": self._read("vital_aperiodic"),
        }
        for name, df in self._raw.items():
            logger.info("Loaded %s: %d rows, %d cols", name, len(df), len(df.columns))
        return self._raw

    def build_features(
        self, windows: dict[str, int] | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not self._raw:
            self.load()

        windows = windows or TEMPORAL_WINDOWS_HOURS
        patient = self._raw["patient"].copy()
        stay_ids = patient["patientunitstayid"]

        feature_frames: list[pd.DataFrame] = []

        # --- demographics ---
        demo = self._build_demographics(patient)
        feature_frames.append(demo)

        # --- labs (per temporal window) ---
        for label, hours in windows.items():
            lab_feats = self._build_lab_features(self._raw["lab"], hours)
            lab_feats = lab_feats.add_suffix(f"_{label}")
            feature_frames.append(lab_feats)

        # --- vitals ---
        vital_feats = self.fe.aggregate_vitals(
            self._raw["vital_periodic"],
            self._raw.get("vital_aperiodic"),
        )
        if not vital_feats.empty:
            feature_frames.append(vital_feats)

        # --- diagnoses ---
        dx_feats = self.fe.encode_diagnoses(self._raw["diagnosis"])
        feature_frames.append(dx_feats)

        # --- medications ---
        med_feats = self.fe.aggregate_medications(self._raw["medication"])
        feature_frames.append(med_feats)

        # --- merge everything on patientunitstayid ---
        features = feature_frames[0]
        for frame in feature_frames[1:]:
            if frame.empty:
                continue
            features = features.join(frame, how="left")

        # --- normalize labs ---
        lab_cols = [c for c in features.columns if c.startswith("lab_")]
        features = self.fe.normalize_labs(features, lab_cols)

        # --- labels ---
        labels = self._build_readmission_labels(patient)

        features, labels = self._align(features, labels)
        features = features.fillna(0)

        logger.info(
            "Feature matrix: %d patients, %d features", len(features), features.shape[1]
        )
        return features, labels

    def time_split(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        split_year: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        split_year = split_year or TRAIN_SPLIT_YEAR
        patient = self._raw["patient"].set_index("patientunitstayid")

        discharge_year = pd.to_numeric(
            patient["hospitaldischargeyear"], errors="coerce"
        )
        common = features.index.intersection(discharge_year.index)
        discharge_year = discharge_year.loc[common]

        train_ids = discharge_year[discharge_year <= split_year].index
        test_ids = discharge_year[discharge_year > split_year].index

        train_ids = train_ids.intersection(features.index)
        test_ids = test_ids.intersection(features.index)

        logger.info("Train: %d, Test: %d (split year=%d)", len(train_ids), len(test_ids), split_year)

        return (
            features.loc[train_ids],
            labels.loc[train_ids],
            features.loc[test_ids],
            labels.loc[test_ids],
        )

    # ------------------------------------------------------------------
    # Private: table readers
    # ------------------------------------------------------------------

    def _read(self, table_name: str) -> pd.DataFrame:
        path = getattr(self.paths, table_name, None)
        if path is None:
            logger.warning("No path configured for table: %s", table_name)
            return pd.DataFrame()

        path = Path(path)
        if not path.exists():
            logger.warning("File not found: %s", path)
            return pd.DataFrame()

        logger.info("Reading %s", path)
        return pd.read_csv(path, low_memory=False)

    # ------------------------------------------------------------------
    # Private: feature builders
    # ------------------------------------------------------------------

    def _build_demographics(self, patient: pd.DataFrame) -> pd.DataFrame:
        demo = patient[["patientunitstayid"]].copy()
        demo = demo.set_index("patientunitstayid")

        if "age" in patient.columns:
            demo["age"] = pd.to_numeric(
                patient["age"].replace("> 89", 90), errors="coerce"
            )

        if "gender" in patient.columns:
            demo["is_female"] = (patient["gender"].str.strip().str.lower() == "female").astype(int)

        if "ethnicity" in patient.columns:
            eth_dummies = pd.get_dummies(patient["ethnicity"], prefix="eth")
            eth_dummies.index = demo.index
            demo = demo.join(eth_dummies)

        if "admissionweight" in patient.columns:
            demo["weight"] = pd.to_numeric(patient["admissionweight"], errors="coerce")

        if "admissionheight" in patient.columns:
            demo["height"] = pd.to_numeric(patient["admissionheight"], errors="coerce")

        if "unitdischargeoffset" in patient.columns:
            demo["los_hours"] = patient["unitdischargeoffset"].values / 60.0

        return demo

    def _build_lab_features(
        self, lab: pd.DataFrame, window_hours: int
    ) -> pd.DataFrame:
        if lab.empty:
            return pd.DataFrame()

        offset_col = "labresultoffset"
        if offset_col not in lab.columns:
            return pd.DataFrame()

        windowed = lab[lab[offset_col] <= window_hours * 60].copy()

        lab_cols = [c for c in LAB_FEATURES if c in windowed["labname"].unique()]
        pivoted = windowed[windowed["labname"].isin(lab_cols)].copy()

        pivoted["labresult"] = pd.to_numeric(pivoted["labresult"], errors="coerce")
        pivoted = pivoted.dropna(subset=["labresult"])

        agg = pivoted.groupby(["patientunitstayid", "labname"])["labresult"].agg(
            ["mean", "std", "min", "max", "last"]
        )
        agg = agg.unstack(level="labname")
        agg.columns = [f"lab_{name}_{stat}" for stat, name in agg.columns]

        medians = pivoted.groupby("labname")["labresult"].median()
        for col in agg.columns:
            lab_name = col.split("_", 1)[1].rsplit("_", 1)[0]
            if lab_name in medians.index:
                agg[col] = agg[col].fillna(medians[lab_name])

        return agg

    def _build_readmission_labels(self, patient: pd.DataFrame) -> pd.DataFrame:
        pt = patient[["patientunitstayid", "patienthealthsystemstayid"]].copy()
        pt = pt.set_index("patientunitstayid")

        if "unitdischargeoffset" in patient.columns:
            pt["discharge_offset"] = patient["unitdischargeoffset"].values

        readmitted = pd.Series(0, index=pt.index, name="readmission_30d")

        for hsid, group in pt.groupby("patienthealthsystemstayid"):
            if len(group) < 2:
                continue
            sorted_group = group.sort_values("discharge_offset")
            offsets = sorted_group["discharge_offset"].values
            for i in range(len(offsets) - 1):
                gap_days = (offsets[i + 1] - offsets[i]) / (60 * 24)
                if gap_days <= READMISSION_WINDOW_DAYS:
                    readmitted.loc[sorted_group.index[i]] = 1

        labels = pd.DataFrame({"readmission_30d": readmitted})
        logger.info(
            "Readmission labels: %d positive / %d total (%.1f%%)",
            readmitted.sum(),
            len(readmitted),
            100 * readmitted.mean(),
        )
        return labels

    # ------------------------------------------------------------------
    # Private: alignment
    # ------------------------------------------------------------------

    @staticmethod
    def _align(
        features: pd.DataFrame, labels: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        common = features.index.intersection(labels.index)
        return features.loc[common], labels.loc[common]
