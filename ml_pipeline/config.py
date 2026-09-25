"""Shared configuration for the Mitifall fall-detection ML pipeline.

Everything that the feature extractor and the evaluator need to agree on lives
here, so the firmware side has a single place to mirror (window length, axis
order, feature order).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent

PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
OUTPUT_DIR = PIPELINE_DIR / "outputs"

FEATURES_CSV = OUTPUT_DIR / "features.csv"
FEATURE_NAMES_TXT = OUTPUT_DIR / "feature_names.txt"
FOLD_METRICS_CSV = OUTPUT_DIR / "loso_fold_metrics.csv"
SUMMARY_CSV = OUTPUT_DIR / "loso_summary.csv"
CONFUSION_JSON = OUTPUT_DIR / "confusion_matrices.json"
REPORT_MD = OUTPUT_DIR / "loso_report.md"

# --------------------------------------------------------------------------- #
# Data layout
# --------------------------------------------------------------------------- #

# Class directories inside dataset/processed/ and their binary labels.
CLASS_DIRS: dict[str, int] = {"ADL": 0, "Fall": 1}

LABEL_NAMES = {0: "ADL", 1: "Fall"}

TIME_COLUMN = "TimeStamp"  # milliseconds

ACC_AXES = ("Ax", "Ay", "Az")
GYR_AXES = ("Gx", "Gy", "Gz")
MAG_AXES = ("Mx", "My", "Mz")
SENSOR_AXES = ACC_AXES + GYR_AXES + MAG_AXES

REQUIRED_COLUMNS = (TIME_COLUMN, *SENSOR_AXES)

# Columns carried through the feature table that are *not* model inputs.
METADATA_COLUMNS = (
    "subject_id",
    "label",
    "class_dir",
    "activity",
    "trial",
    "source_file",
    "window_index",
    "window_start_ms",
    "window_end_ms",
    "n_samples",
)

# --------------------------------------------------------------------------- #
# Windowing / labelling
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WindowConfig:
    """Sliding-window segmentation parameters.

    The UMAFall wrist stream runs at roughly 20 Hz with jittery timestamps, so
    windows are defined in *milliseconds* rather than in sample counts. A 2.0 s
    window therefore holds ~40 samples even when the effective rate drifts.
    """

    window_ms: int = 2000
    overlap: float = 0.5  # 50 % overlap -> 1.0 s step
    min_samples: int = 10  # discard windows too sparse for stable statistics
    allow_partial: bool = True  # emit one short window if a file is < window_ms

    # "file"   -> every window of a Fall recording is labelled 1 (dataset default)
    # "impact" -> only windows overlapping the impact peak are labelled 1
    fall_label_mode: str = "file"
    impact_halfwidth_ms: int = 1000

    @property
    def step_ms(self) -> int:
        step = int(round(self.window_ms * (1.0 - self.overlap)))
        return max(step, 1)


# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #

RANDOM_STATE = 42

# Defaults tuned for a resource-constrained ESP32-C3 target: a shallow, small
# forest converts to a compact C header, and the linear models are trivially
# portable.
RF_PARAMS = dict(
    n_estimators=60,
    max_depth=10,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
)

SVM_RBF_PARAMS = dict(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced")
SVM_LINEAR_PARAMS = dict(kernel="linear", C=1.0, class_weight="balanced")
LOGREG_PARAMS = dict(C=1.0, max_iter=2000, class_weight="balanced")

MODEL_NAMES = ("random_forest", "svm_rbf", "svm_linear", "logistic_regression")
