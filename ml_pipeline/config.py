from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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

CLASS_DIRS: dict[str, int] = {"ADL": 0, "Fall": 1}

LABEL_NAMES = {0: "ADL", 1: "Fall"}

TIME_COLUMN = "TimeStamp"

ACC_AXES = ("Ax", "Ay", "Az")
GYR_AXES = ("Gx", "Gy", "Gz")
MAG_AXES = ("Mx", "My", "Mz")
SENSOR_AXES = ACC_AXES + GYR_AXES + MAG_AXES

REQUIRED_COLUMNS = (TIME_COLUMN, *SENSOR_AXES)

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

@dataclass(frozen=True)
class WindowConfig:

    window_ms: int = 2000
    overlap: float = 0.5
    min_samples: int = 10
    allow_partial: bool = True

    fall_label_mode: str = "file"
    impact_halfwidth_ms: int = 1000

    @property
    def step_ms(self) -> int:
        step = int(round(self.window_ms * (1.0 - self.overlap)))
        return max(step, 1)

RANDOM_STATE = 42

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
