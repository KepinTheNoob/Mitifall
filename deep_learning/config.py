from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
OUTPUT_DIR = PACKAGE_DIR / "outputs"

WINDOW_CACHE = OUTPUT_DIR / "windows_cache.npz"
FINAL_CHECKPOINT = OUTPUT_DIR / "model_final.pt"
ONNX_MODEL = OUTPUT_DIR / "mitifall_fall_dl.onnx"
ONNX_INT8_MODEL = OUTPUT_DIR / "mitifall_fall_dl.int8.onnx"
MODEL_CARD = OUTPUT_DIR / "model_card.json"

SAMPLE_RATE_HZ = 50
SAMPLE_PERIOD_MS = 1000.0 / SAMPLE_RATE_HZ
WINDOW_SAMPLES = 100
HOP_SAMPLES = 50

LOWPASS_CUTOFF_HZ = 8.0
LOWPASS_ORDER = 2

RAW_ACC_COLUMNS = ("Ax", "Ay", "Az")
RAW_GYR_COLUMNS = ("Gx", "Gy", "Gz")

CHANNEL_SETS: dict[str, tuple[str, ...]] = {

    "spec": ("ax", "ay", "az", "acc_mag", "tilt_deg"),

    "spec_gyro": ("ax", "ay", "az", "acc_mag", "tilt_deg", "gx", "gy", "gz", "gyr_mag"),
}
BARO_CHANNEL = "dh_m"
BARO_PRESSURE_COLUMN = "Pressure_hPa"

ML_INFERENCE_THRESHOLD = 0.30

VALIDATION_POOL = (2, 4, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19)
VALIDATION_SUBJECTS_PER_FOLD = 3

FINAL_VALIDATION_SUBJECTS = (5, 15, 17)

RANDOM_SEED = 42

ENSEMBLE_SIZE = 1

@dataclass(frozen=True)
class ModelConfig:
    arch: str = "tcn"
    channels: int = 32
    levels: int = 3
    kernel: int = 5
    dropout: float = 0.15
    heads: int = 2
    lstm_hidden: int = 32

@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 60
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 10
    lr_patience: int = 4
    lr_factor: float = 0.5
    min_lr: float = 1e-5
    loss: str = "focal"
    focal_gamma: float = 2.0
    threads: int = 6

@dataclass(frozen=True)
class AugmentConfig:
    enabled: bool = True

    p_jitter: float = 0.5
    jitter_sigma_g: float = 0.02

    p_scale: float = 0.5
    scale_low: float = 0.7
    scale_high: float = 1.15
    gravity_window: int = 25

    p_warp: float = 0.4
    warp_sigma: float = 0.15
    warp_knots: int = 4

    p_rotate: float = 0.5
    rotate_max_deg: float = 12.0
