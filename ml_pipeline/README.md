# Mitifall ML pipeline

Windowed feature extraction and Leave-One-Subject-Out benchmarking for wrist-worn
fall detection, targeting an ESP32-C3 deployment.

## Layout

| File | Role |
|------|------|
| `config.py` | Paths, window parameters, axis order, model hyperparameters |
| `data_loader.py` | Recording discovery, `Subject_(\d+)` parsing, ADL/Fall labelling, CSV loading |
| `feature_extraction.py` | Sliding-window segmentation and the 53-feature block |
| `train_evaluate.py` | LOSO-CV benchmark, fall-detection metrics, confusion matrices, reports |
| `run_pipeline.py` | End-to-end runner (extraction + evaluation) |
| `outputs/` | Generated artifacts (features, metrics, report) |

## Prerequisites

Input data must already exist in `dataset/processed/{ADL,Fall}/` with columns
`[TimeStamp, Ax, Ay, Az, Gx, Gy, Gz, Mx, My, Mz]`. Build it with
`python dataset/process_dataset.py`.

```bash
python -m venv .venv && source .venv/bin/activate.fish
pip install -r requirements.txt
```

## Usage

```bash
python ml_pipeline/run_pipeline.py                     # extract features + benchmark
python ml_pipeline/run_pipeline.py --reuse-features     # benchmark only
python ml_pipeline/feature_extraction.py                # extraction only
python ml_pipeline/train_evaluate.py                    # benchmark only
```

Useful flags:

```bash
--window-seconds 2.0 --overlap 0.5     # segmentation geometry
--models random_forest svm_rbf          # benchmark a subset
--fall-label-mode impact                # see "Label granularity" below
```

## Segmentation

The wrist stream runs at ~20 Hz with jittery timestamps, so windows are defined
in **milliseconds**, not sample counts: a 2.0 s window with 50 % overlap steps
every 1.0 s and holds ~40 samples. Windows with fewer than `min_samples` (10)
samples are dropped. Each recording (~15 s) yields ~13 windows; the full dataset
yields ~9.7 k windows.

## Feature block (53 values per window)

* 9 axes (`Ax..Mz`) x `{mean, std, min, max, var}` -> 45
* acceleration vector magnitude `sqrt(Ax^2+Ay^2+Az^2)`: `{max, mean, std}` -> 3
* angular velocity vector magnitude: `{max, mean, std}` -> 3
* signal magnitude area (mean of summed absolute axes) for accelerometer and
  gyroscope -> 2

All are single-pass time-domain statistics, computable on-device over a ring
buffer: no FFT, no filtering, no cross-axis terms. `std` uses the population
definition (ddof = 0) to match a naive embedded implementation. `var` is retained
alongside `std` even though it is redundant by construction.

The ordered feature names are written to `outputs/feature_names.txt` — the
firmware feature vector must follow exactly that order.

## Evaluation: LOSO-CV

`LeaveOneGroupOut` grouped on `subject_id` gives one fold per subject (19 folds).
Windows from one recording are highly correlated and windows from one subject
share gait and posture signatures, so a random split would leak subject identity
and inflate every score.

Metrics, per fold and aggregated:

* **sensitivity / recall** = TP / (TP + FN) — falls actually detected
* **specificity** = TN / (TN + FP) — false-alarm behaviour
* F1, precision, accuracy, balanced accuracy

Two aggregation views are reported:

* **pooled** — metrics recomputed from the confusion matrix summed over all
  folds (every window counted once). This is the headline number.
* **fold-averaged** — NaN-aware mean ± std across subjects, showing
  per-subject variability.

Six of the 19 subjects (1, 3, 9, 10, 12, 13) contributed no Fall recordings.
Their folds contain no positives, so sensitivity, precision and F1 are
mathematically undefined there and are reported as NaN rather than 0.0, and
excluded from the fold averages. The `n_folds_<metric>` columns in
`loso_summary.csv` record how many folds each average is based on.

## Label granularity

`--fall-label-mode file` (default) follows the dataset ground truth: every window
of a Fall recording is labelled 1. Since a fall trial is ~15 s of mostly normal
movement around a sub-second impact, roughly two thirds of the positive windows
contain no fall event, which caps achievable sensitivity.

`--fall-label-mode impact` labels only windows overlapping ±1 s of the peak
acceleration magnitude as positive, which is closer to what an on-device detector
must actually discriminate. Use it for a realistic performance estimate; use
`file` for comparability with the dataset-level convention.

## Outputs

| Artifact | Contents |
|----------|----------|
| `outputs/features.csv` | One row per window: metadata + 53 features |
| `outputs/feature_names.txt` | Ordered feature contract for the firmware |
| `outputs/loso_fold_metrics.csv` | Per model, per subject: confusion counts + metrics |
| `outputs/loso_summary.csv` | Pooled and fold-averaged summary table |
| `outputs/confusion_matrices.json` | Aggregated confusion matrix per model |
| `outputs/loso_report.md` | Human-readable benchmark report |

## Model export to firmware

```bash
python ml_pipeline/export_model.py                  # 53-feature 9-DoF model
python ml_pipeline/export_model.py --axes acc_gyro  # 38-feature 6-DoF model
```

Writes `include/RandomForest.h` (flat-array forest with a probability API) and
`include/features_order.json` (the feature contract), plus
`outputs/threshold_sweep.csv` with LOSO out-of-fold sensitivity/specificity per
decision threshold. The exporter refuses to write unless its interpreter
reproduces scikit-learn's `predict_proba` on the whole dataset.

Firmware build environments (see `platformio.ini`):

| Environment | Entry point | Purpose |
|---|---|---|
| `scanner` | `src/scanner_main.cpp` | I2C scan, sensor ID, pin sweep, actuator test |
| `raw_stream` | `src/raw_stream_main.cpp` | 50 Hz 9-DoF CSV capture |
| `ml_inference` | `src/inference_main.cpp` | On-device inference and alarm |

```bash
pio run -e ml_inference -t upload
pio device monitor -e ml_inference
```
