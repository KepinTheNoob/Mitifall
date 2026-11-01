# deep_learning — time-series fall detection (PyTorch → ONNX)

Sequence models that read the raw 2 s accelerometer (and optionally gyroscope)
window directly, benchmarked **head-to-head** against the Random Forest that
`[env:ml_inference]` runs on the ESP32-C3, and exported to ONNX for an edge
gateway or a streaming microservice.

This is a Python module like `ml_pipeline/`, not a PlatformIO environment: the
models run off the wearable (phone, Raspberry Pi, server) on the CSV stream that
`[env:raw_stream]` produces. The forest remains the on-device model; see
[§4.3](#43-on-the-microcontroller) for what porting would take.

---

## 1. Results

Leave-One-Subject-Out over all 19 UMAFall subjects, pooled over the 9 683 windows
both models are scored on (2 689 Fall / 6 994 ADL). Both models are trained on the
same 15 subjects per fold and both get their decision threshold from the same 3
validation subjects (Youden's J); the test subject is never seen by either.
"Soft-fall sens." is sensitivity on the impact windows of the softest third of
fall recordings (peak |a| ≤ 5.04 g, 140 windows).

| Model | Accuracy | F1 macro | F1 Fall | Sensitivity | Specificity | ROC-AUC | PR-AUC | Soft-fall sens. |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `ml_inference` RF @ 0.30 (as deployed) | 0.842 | 0.816 | 0.747 | **0.837** | 0.844 | 0.906 | 0.854 | **0.914** |
| `ml_inference` RF @ val-tuned threshold | 0.877 | 0.846 | 0.778 | 0.777 | 0.915 | 0.906 | 0.854 | 0.771 |
| TCN + attention (5 ch.) — *pre-specified primary* | 0.862 | 0.833 | 0.763 | 0.799 | 0.887 | 0.926 | 0.861 | 0.843 |
| CNN-BiLSTM (5 ch.) | 0.870 | 0.840 | 0.770 | 0.784 | 0.903 | 0.927 | 0.861 | 0.850 |
| CNN-BiLSTM (5 ch.) × 3 seeds | 0.867 | 0.836 | 0.764 | 0.775 | 0.902 | 0.925 | 0.863 | 0.879 |
| **CNN-BiLSTM + gyro (9 ch.) — shipped** | **0.880** | **0.851** | **0.784** | 0.783 | **0.918** | **0.928** | **0.867** | 0.757 |

5 ch. = `ax, ay, az, |a|, tilt` (the requested set; there is no barometer in
UMAFall, see §3.3). 9 ch. adds `gx, gy, gz, |ω|`. The forest uses 53 features from
accelerometer, gyroscope **and** magnetometer.

**What the numbers support.** Differences below are DL − RF from the table;
brackets are 95 % intervals from a subject-level bootstrap (2 000 resamples of
the 19 subjects). Full tables are in each run's `report.md`.

- **Ranking (ROC-AUC) is better in every deep configuration**, by +0.020 to
  +0.022. The interval excludes zero for CNN-BiLSTM ([+0.004, +0.041]) and
  CNN-BiLSTM + gyro ([+0.007, +0.039]), not for the other two.
- **Shipped model vs the forest with the same threshold rule:** higher on every
  thresholded metric — accuracy +0.004, F1 macro +0.005, F1 Fall +0.007,
  sensitivity +0.006, specificity +0.003 — but each of those intervals includes
  zero, and it is **lower** on soft-fall sensitivity (−0.014, [−0.153, +0.112]).
- **Shipped model vs the forest as deployed (threshold 0.30):** accuracy +0.038
  [+0.012, +0.062], F1 macro +0.035 [+0.006, +0.063] and specificity +0.074
  [+0.043, +0.101] are clearly higher; sensitivity is clearly lower (−0.054
  [−0.106, −0.012]). That is an operating-point difference: 0.30 buys the forest
  its sensitivity with 1 090 false alarms against the shipped model's 575.
- **Soft landings:** the requested-channel models catch more soft falls than the
  forest at its val-tuned threshold (+0.07 to +0.11; the 3-seed ensemble's
  interval [+0.014, +0.197] excludes zero). Adding the gyroscope removed that
  advantage.
- **At the same operating point** (the deep model's ROC curve read at the forest's
  specificity or sensitivity) the deep models are ahead in 14 of 16 comparisons,
  by up to +0.036, but no interval excludes zero:

  | Forest operating point | Metric | RF | TCN | CNN-BiLSTM | ×3 seeds | + gyro |
  |---|---|:---:|:---:|:---:|:---:|:---:|
  | RF @ 0.30 | sensitivity at equal specificity | 0.837 | 0.863 | 0.859 | 0.847 | 0.863 |
  | RF @ 0.30 | specificity at equal sensitivity | 0.844 | 0.880 | 0.873 | 0.858 | 0.871 |
  | RF @ val-tuned | sensitivity at equal specificity | 0.777 | 0.788 | 0.771 | 0.784 | 0.813 |
  | RF @ val-tuned | specificity at equal sensitivity | 0.915 | 0.921 | 0.909 | 0.917 | 0.936 |

  For the shipped model all four are ahead (+0.021 to +0.035) with interval
  lower bounds between −0.014 and −0.008.

**Bottom line.** On this dataset the deep model is *as good as or better than* the
forest on nearly every metric, and clearly better at ranking and against the
forest's deployed threshold, but it does **not** beat the forest on every metric
with statistical confidence: 19 subjects leave intervals of several points. Two
honest caveats on top: (1) the shipped configuration adds the gyroscope to the
requested channel list, and (2) it was chosen after seeing these four results —
there is no untouched hold-out set, so its margin is slightly optimistic. The TCN
row is the configuration fixed before any benchmark ran.

Runs, each reproducible with its tag:

| Run | Command | Output |
|---|---|---|
| TCN | `python deep_learning/evaluate_benchmark.py --arch tcn` | `outputs/benchmark/tcn_06e0e567/` |
| CNN-BiLSTM | `... --arch cnn_bilstm` | `outputs/benchmark/cnn_bilstm_6fb27860/` |
| CNN-BiLSTM × 3 | `... --arch cnn_bilstm --ensemble 3` | `outputs/benchmark/cnn_bilstm_ens3_9c4b2f03/` |
| CNN-BiLSTM + gyro | `... --arch cnn_bilstm --with-gyro` | `outputs/benchmark/cnn_bilstm_gyro_9680000d/` |

Add `--jobs 4 --threads 2` to train folds in parallel (about 15 min for the TCN
and 8 min for a CNN-BiLSTM on a 12-thread desktop CPU; three times that for
`--ensemble 3`).

### 1.1 Full evaluation against every classic model

`python deep_learning/evaluation_report.py` (~2 min) writes
[`outputs/evaluation/report.md`](outputs/evaluation/report.md) with 17 tables and
6 figures. It reuses the deep runs' test predictions and refits every classic
model of `ml_pipeline` on the same folds: the deployed forest (RF-53), the same
forest on accelerometer + gyroscope features (RF-38) and on accelerometer
features only (RF-19), so each deep model also meets a forest that sees the same
sensors, plus SVM-RBF, linear SVM and logistic regression. On top of the
bootstrap intervals it adds a McNemar test clustered by subject, per-subject
Wilcoxon tests, per-activity error rates, threshold-free operating points, an
event-level replay of every recording (alarm after 1 or 2 positive windows, 6 s
cooldown) and each model's compute cost.

What it shows:

- **Ranking:** every deep model ranks windows better than every classic one
  (ROC-AUC 0.925–0.928 vs at most 0.908). Against the forest built from the same
  sensors the gain is +0.019 to +0.033, every interval above zero.
- **High-sensitivity region:** at a required window sensitivity of 0.90 the
  shipped model keeps specificity 0.794 against 0.634 for RF-53 (+0.159
  [+0.037, +0.220]); at 0.95 it is 0.577 against 0.414.
- **At the thresholds actually used**, the shipped model and RF-53 with the same
  threshold rule are statistically tied (clustered McNemar p = 0.668). It clearly
  beats SVM-RBF, linear SVM, logistic regression and RF-19 on accuracy, F1,
  specificity, precision and MCC.
- **Impact windows:** it flags the window containing the impact less often than
  the forest at 0.30 (0.863 vs 0.969, clearly lower), the other side of its
  lower false-alarm rate.
- **Event level:** with one positive window, both it and RF-53 @ 0.30 detect about
  99 % of fall trials, with 124 vs 204 false alarms per hour of ADL (−79
  [−121, −37]); with two windows in a row, 60 vs 115 per hour. Against RF-53
  with the same threshold rule there is no difference (−0.5 [−31, +34] per hour).
- **Trial level, threshold-free:** recording-level ROC-AUC is 0.970–0.981 for the
  deep models and 0.968 for RF-53 with an alarm on one window, but 0.969–0.971
  against 0.975 with two windows in a row; shipped model vs RF-53 is
  +0.007 [−0.003, +0.021] and −0.005 [−0.017, +0.007]. No clear difference.
- **Rule-based detector:** at the window thresholds no learned model reaches the
  trial-level specificity of `raw_stream`'s cascade (0.942, README 9.4; the best
  learned model reaches 0.788), but the cascade misses more fall trials (0.856
  detected). Read off the trial-level curves at specificity 0.95, the deep models
  would detect 0.79–0.90 of fall trials with one window (RF-53: 0.84) and
  0.81–0.85 with two in a row (RF-53: 0.88). Those thresholds are picked on the
  test curve, so they are upper bounds.
- **Cost:** 1.28 M multiply-accumulates per window against ~560 comparisons;
  only the forest runs on the ESP32-C3.

---

## 2. Quick start

```bash
# from the repository root, after the ml_pipeline setup (README §4.3)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r deep_learning/requirements.txt

python deep_learning/evaluate_benchmark.py --arch cnn_bilstm --with-gyro --jobs 4 --threads 2
python deep_learning/evaluation_report.py                      # -> outputs/evaluation/report.md
python deep_learning/train.py --arch cnn_bilstm --with-gyro     # -> outputs/model_final.pt
python deep_learning/export_onnx.py --int8                     # -> outputs/*.onnx, model_card.json
python deep_learning/infer_stream.py --selftest 0              # streaming == batch, all recordings

# live: the model is the ONLY fall detector (rule-based detector off on the board)
pio run -e dl_stream -t upload
python deep_learning/infer_stream.py --port /dev/ttyACM0
```

Without flags, `train.py` / `evaluate_benchmark.py` build the TCN on the five
requested channels. The windows are cached in `outputs/windows_cache.npz` (12 MB,
git-ignored) on first use.

---

## 3. Design

### 3.1 Same windows, same labels

`data.py` builds one tensor per row of `ml_pipeline`'s feature table — the exact
windows the forest is trained and scored on: 2.0 s windows with a 1.0 s hop
(50 % overlap) from the same recording, same file-level label, same subject. On
the 9 683 windows, the window mean of `Ax` computed from the tensors correlates
with the forest's `Ax_mean` feature at r = 0.9989. That alignment is what makes
the benchmark head-to-head.

### 3.2 Signal path (training and streaming are identical)

```
native samples (~20 Hz UMAFall, 50 Hz firmware)
  → linear interpolation onto a uniform 50 Hz grid (20 ms)
  → causal 2nd-order Butterworth low-pass, 8 Hz, started at steady state
  → 100-sample windows (2.0 s), hop 50 (1.0 s)
  → model: |a| = √(ax²+ay²+az²), tilt = acos(az/|a|) [, |ω|], robust scaler
```

UMAFall's wrist stream has no content above its ~10 Hz Nyquist limit, so the
same 8 Hz low-pass is applied to the 50 Hz MPU-6050 stream at inference: the
model sees the bandwidth it was trained on. The derived channels and the
RobustScaler (per-channel median / IQR of the training windows) are *inside* the
network as frozen buffers, so they travel with the ONNX file and cannot drift.

### 3.3 Channels and the barometer (`Δh`)

The requested inputs are `ax, ay, az, |a|, tilt` and `Δh`. UMAFall has **no
barometer**, so no `Δh` model can be trained or evaluated on it, and none of the
results above use it. Two shortcuts were rejected: a zero-filled channel (its
weights would be untrained noise at inference) and a `Δh` synthesised from the
labels (label leakage).

The path is in place for when data exists. `[env:raw_stream]` writes
`Pressure_hPa` into its CSV; `data.py` turns it into `dh_m`, the barometric
altitude relative to the window start, and `--with-baro` adds it as a channel
(with its own jitter augmentation) in training, benchmarking, export and
streaming. Every recording must then carry pressure, so this needs a dataset of
board captures rather than UMAFall:

1. Record labelled sessions with `kStreamCsv = true`; strip the `#` / log lines
   and rename the `Timestamp_ms` column to `TimeStamp`.
2. Lay them out like UMAFall: `dataset/processed/{ADL,Fall}/`, with
   `Subject_<id>` in each file name (that is how LOSO groups them).
3. Rebuild the feature table (`python ml_pipeline/run_pipeline.py`) — the deep
   windows are defined by it — then:

```bash
python deep_learning/evaluate_benchmark.py --arch cnn_bilstm --with-gyro --with-baro
```

The plumbing was checked on a synthetic set (24 UMAFall recordings given an
artificial `Pressure_hPa` column): window building, one training epoch, ONNX
parity (6 × 10⁻⁸), and streaming windows bit-identical to the batch windows.
Firmware-style input with the first 30 pressure fields empty and a mid-stream
board reset also streams without error. That shows the code path works, not that
`Δh` helps.

### 3.4 Models

| | TCN + self-attention | CNN-BiLSTM |
|---|---|---|
| Front end | stride-2 conv stem (100 → 50 steps) | 2 × (conv k5 + BN + ReLU + max-pool) (100 → 25) |
| Temporal | 3 residual dilated conv blocks (d = 1, 2, 4, k = 5; receptive field 117 samples) + 2-head self-attention | bidirectional LSTM, 32 units per direction |
| Pooling / head | attention pooling → MLP → logit | attention pooling → MLP → logit |
| Parameters (5 ch. / 9 ch.) | 37 602 / — | 40 706 / 41 346 |
| MACs per window (5 ch. / 9 ch.) | 1.94 M / — | 1.21 M / 1.28 M |

The convolutions extract the sharp impact pattern; the dilated stack or the
BiLSTM carries the order of events across the window (upright → impact →
settling). Attention is written out as plain matrix products because PyTorch's
fused `nn.MultiheadAttention` path does not export cleanly.

### 3.5 Training

| Concern | Choice |
|---|---|
| Class imbalance (28 % Fall) | focal loss, γ = 2, α = share of ADL windows (`--loss weighted_bce` for class-weighted BCE) |
| Augmentation (per batch) | rotation ±12° (strap orientation); scaling of the dynamic component 0.7–1.15 × around a 0.5 s gravity estimate (softer, cushioned landings keep 1 g of gravity); smooth time warp σ = 0.15; jitter 0.02 g |
| Optimiser | AdamW, lr 2e-3, weight decay 1e-4, batch 128, gradient clip 1.0 |
| Schedule | ReduceLROnPlateau on validation loss (×0.5, patience 4) |
| Early stopping | validation loss, patience 10, best weights restored |
| Threshold | Youden's J on the validation subjects |

Early stopping lands early (median best epoch 5–6): after that, validation loss
rises. The likely causes are fitting to subject-specific motion and label noise —
with file-level labels, most "Fall" windows are the standing and lying around
the impact.

### 3.6 Benchmark protocol (`evaluate_benchmark.py`)

- 19 LOSO folds. Per fold, 3 validation subjects come from the subjects with
  falls (subject 18, a quarter of all windows, is never used for validation), and
  **both** models train on the remaining 15.
- The forest is `ml_pipeline`'s deployed configuration (60 trees, depth 10,
  53 features, `class_weight="balanced_subsample"`), refitted per fold.
- Reported at the forest's deployed threshold 0.30 and at a val-tuned threshold
  for each model.
- Every DL − RF difference gets a subject-level bootstrap interval; the
  matched-operating-point table compares the ROC curves at the forest's
  operating points.
- Folds are checkpointed to `folds/subject_XX.csv`, so an interrupted run resumes.

---

## 4. Export and deployment

### 4.1 ONNX (`export_onnx.py`)

| | |
|---|---|
| Input | `window` float32 `(batch, 6, 100)`: `ax, ay, az` [g], `gx, gy, gz` [°/s], 50 Hz, low-passed as in §3.2 |
| Output | `p_fall` float32 `(batch,)` |
| Threshold | 0.59 (`model_card.json → decision.threshold`) |
| File | `outputs/mitifall_fall_dl.onnx`, 247 KiB, opset 18, dynamic batch |
| Parity with PyTorch | max \|Δp\| 3.6 × 10⁻⁷ over all 9 683 windows (batched) and 1.8 × 10⁻⁷ at batch 1 (64-window sample); 100 % identical decisions |
| Latency | 0.10–0.13 ms per window (two runs), ONNX Runtime, batch 1, one thread of a desktop CPU |
| INT8 (`--int8`) | `outputs/mitifall_fall_dl.int8.onnx`, 131 KiB, static QDQ (Conv/MatMul/Gemm); 99.06 % decisions identical to fp32, mean \|Δp\| 0.014, max 0.44 |

The fp32 model is the one to deploy: it is small and fast already, and INT8
changes about 1 decision in 100. `model_card.json` records the input contract,
preprocessing, threshold, training configuration, this model's benchmark rows,
file hashes and the parity figures.

The exporter refuses to write a model card if ONNX Runtime and PyTorch disagree
by more than 10⁻⁴. That check caught a real bug during development: the exported
graph had dropped the `+ 1e-12` inside `|a|`, turning all-zero samples (a few
UMAFall files start with one) into NaN. The norm now uses `clamp_min`, which the
exporter keeps.

### 4.2 Streaming inference (`infer_stream.py`)

Reads CSV from a file or stdin — the firmware's serial output can be piped in
unchanged, because `#` lines, boot logs and any non-data line are skipped, and a
new header (board reset) restarts the stream. Per sample it interpolates onto
the 50 Hz grid, runs the low-pass with its state carried across samples, keeps a
100-sample ring buffer, and scores a window every 50 samples:

```
$ python deep_learning/infer_stream.py dataset/processed/Fall/UMAFall_Subject_02_Fall_backwardFall_1_2016-06-13_20-51-32.csv
# model mitifall_fall_dl.onnx: channels ax, ay, az, gx, gy, gz, threshold 0.59
window_start_ms,window_end_ms,p_fall,fall
160,2160,0.3966,0
1160,3160,0.8052,1
*** FALL  window 1160-3160 ms  p=0.805 (threshold 0.59)
2160,4160,0.5556,0
3160,5160,0.6385,1
...
12160,14160,0.7757,1
# 299 rows, 13 windows, 3 alert(s); skipped 0 non-data lines, dropped 0 out-of-order rows
```

Subject 2 is in the shipped model's training set, so this shows the mechanics,
not accuracy. The windows after the impact stay positive while the subject lies
on the floor: with file-level labels the model learned "this is a fall
recording", not just "this is the impact". `--refractory` (default 5 s) spaces
the alerts, `--quiet` prints alerts only, `--int8` uses the INT8 model.

`--selftest 0` replays every dataset recording through the streaming path and
compares with the batch windows the benchmark used: **9 683 / 9 683 windows
reproduced, max |Δp| 1.2 × 10⁻⁷.**

### 4.3 On the microcontroller

Not done here, and the numbers say why it is a project of its own rather than a
flag: 1.28 M multiply-accumulates per window, once a second, plus a BiLSTM, on a
160 MHz RISC-V core without FPU or DSP extensions. An int8 port through
TensorFlow Lite Micro or ESP-DL is plausible for the conv-only part (the TCN was
designed with that in mind), but would need quantization-aware training — post-
training INT8 already flips ~1 % of decisions on a PC — and an on-target latency
measurement. The forest stays the on-device model; the deep model runs on the
stream.

---

## 5. Files

| File | Role |
|---|---|
| `config.py` | paths, window geometry, channel sets, model / training / augmentation defaults |
| `data.py` | window tensors aligned with ml_pipeline, resampling, low-pass, barometric altitude |
| `augment.py` | rotation, soft-landing scaling, time warp, jitter |
| `model.py` | `InputPrep`, TCN + attention, CNN-BiLSTM, `FallEnsemble`, focal loss, MAC counter |
| `train.py` | training loop, seed ensembles, checkpoint I/O, final-model CLI |
| `metrics.py` | pooled metrics, ROC/PR-AUC, Youden threshold |
| `evaluate_benchmark.py` | LOSO head-to-head benchmark and reports |
| `evaluation_report.py` | full DL vs classic-ML evaluation over all finished runs (section 1.1) |
| `export_onnx.py` | ONNX / INT8 export, parity checks, model card |
| `infer_stream.py` | streaming inference and the streaming-vs-batch self-test |
| `outputs/benchmark/<run>/` | `report.md`, `metrics.csv`, `bootstrap_deltas.csv`, `matched_operating_points.csv`, `predictions.csv` (every test window's probabilities); `folds/` holds the resumable per-fold checkpoints and is git-ignored |
| `outputs/evaluation/` | `report.md`, one CSV per table, `figures/*.png`, `ml_predictions.csv` (cached refit of the classic models) |
| `outputs/model_final.pt` | shipped model (PyTorch) |
| `outputs/mitifall_fall_dl{,.int8}.onnx`, `outputs/model_card.json` | shipped model (ONNX) and its card |

---

## 6. Limitations

- **Dataset, not device.** Everything above is UMAFall wrist data resampled from
  ~20 Hz. Nothing has been validated on recordings from the Mitifall board yet.
- **File-level labels.** Every window of a fall recording counts as Fall, as for
  the forest, which caps attainable sensitivity and makes "Fall" partly mean
  "within a fall recording" (README §12.3).
- **19 subjects.** Bootstrap intervals are several points wide; differences of
  1–2 points between configurations are noise.
- **No barometer data.** `Δh` is implemented end to end but untrained (§3.3).
- **Selection.** Four configurations were compared on the same LOSO folds and
  the best was shipped (§1).
