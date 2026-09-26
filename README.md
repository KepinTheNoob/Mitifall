<div align="center">

# Mitifall

### Wrist-worn edge fall detection — TinyML on an ESP32-C3

A 9-DoF wearable that classifies falls **on-device** with a Random Forest compiled
to C++, alerts through a buzzer and vibration motor without waiting on the
network, and streams telemetry to Adafruit IO or Blynk IoT.

[![PlatformIO](https://img.shields.io/badge/PlatformIO-6.1-orange?logo=platformio&logoColor=white)](https://platformio.org/)
[![Framework](https://img.shields.io/badge/framework-Arduino%20ESP32-00979D?logo=arduino&logoColor=white)](https://github.com/espressif/arduino-esp32)
[![MCU](https://img.shields.io/badge/MCU-ESP32--C3%20RISC--V-E7352C?logo=espressif&logoColor=white)](https://www.espressif.com/en/products/socs/esp32-c3)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![TinyML](https://img.shields.io/badge/TinyML-Random%20Forest-6A5ACD)](#5-machine-learning-pipeline)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#14-license--credits)

</div>

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Hardware Specification & Pinout](#2-hardware-specification--pinout)
3. [Repository Structure](#3-repository-structure)
4. [Prerequisites & Installation](#4-prerequisites--installation)
5. [Machine Learning Pipeline](#5-machine-learning-pipeline)
6. [Firmware Environments: Build & Upload](#6-firmware-environments-build--upload)
7. [Cloud Dashboard Setup](#7-cloud-dashboard-setup)
8. [Troubleshooting & Common Pitfalls](#8-troubleshooting--common-pitfalls)
   — [actuators stuck on](#87-buzzer-or-motor-runs-continuously-from-power-up) · [silencing an alarm](#88-silencing-an-alarm)
9. [Measured Performance](#9-measured-performance)
10. [Resource Footprint](#10-resource-footprint)
11. [Configuration Reference](#11-configuration-reference)
12. [Known Limitations](#12-known-limitations)
13. [Verification Methodology](#13-verification-methodology)
14. [License & Credits](#14-license--credits)

---

## 1. System Architecture

The device runs a closed loop entirely on the microcontroller. Connectivity is
strictly optional: **detection and the physical alarm never depend on WiFi**.

```
┌──────────────────────────── LOLIN C3 Mini (ESP32-C3, 160 MHz RISC-V) ────────────────────────────┐
│                                                                                                   │
│   I²C @ 400 kHz                                                                                   │
│  ┌──────────────┐                                                                                 │
│  │  MPU-6050    │  ax ay az (g)                                                                   │
│  │  0x68 / 0x69 │  gx gy gz (°/s)                                                                 │
│  └──────┬───────┘                                                                                 │
│         │            ┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────┐   │
│  ┌──────┴───────┐    │   RING BUFFER   │     │  FEATURE EXTRACTOR   │     │   RANDOM FOREST    │   │
│  │ HMC5883L 0x1E│    │  100 samples    │     │  9 axes × 5 stats=45 │     │  60 trees, d≤10    │   │
│  │ QMC5883L 0x0D├───►│  2.0 s @ 50 Hz  ├────►│  |a|,|ω| max/µ/σ = 6 ├────►│  23 634 nodes      │   │
│  │ QMC5883P 0x2C│    │  hop 50 (50 %)  │     │  SMA acc + gyr   = 2 │     │  → P(fall) 0…1     │   │
│   mx my mz (µT)      └─────────────────┘     │  ─────────────────── │     └─────────┬──────────┘   │
│                       1 decision / second    │  53 features         │               │              │
│                                              └──────────────────────┘               ▼              │
│                                                                          ┌────────────────────┐    │
│                                                                          │ DECISION THRESHOLD │    │
│                                                                          │ FALL_THRESHOLD     │    │
│                                                                          │      = 0.30f       │    │
│                                                                          └─────────┬──────────┘    │
│                                          ┌─────────────────────────────────────────┴───────────┐   │
│                                          ▼                                                     ▼   │
│                            ┌──────────────────────────┐                      ┌──────────────────┐  │
│                            │   ALARM MANAGER          │                      │  TELEMETRY       │  │
│                            │   non-blocking, millis() │                      │  (best effort)   │  │
│                            │   ├─ Buzzer    GPIO4     │                      │                  │  │
│                            │   ├─ Vibration GPIO0     │                      │                  │  │
│                            │   └─ Dismiss ◄ GPIO3 btn │                      │                  │  │
│                            └──────────────────────────┘                      └────────┬─────────┘  │
└───────────────────────────────────────────────────────────────────────────────────────┼────────────┘
                                                                                        │
                    ┌───────────────────────────────────────────────────────────────────┴──┐
                    ▼                                                                      ▼
        ┌────────────────────────────┐                                    ┌────────────────────────────┐
        │  ADAFRUIT IO   (MQTT)      │                                    │  BLYNK IoT  (virtual pins) │
        │  [env:raw_stream]          │                                    │  [env:ml_inference]        │
        │  ├─ accel-mag      chart   │                                    │  ├─ V0 probability  chart  │
        │  ├─ motion-status  LED     │                                    │  ├─ V1 fall alert   LED    │
        │  └─ buzzer-command switch ─┼──► triggers alarm                  │  └─ V2 reset      button ──┼──► dismiss
        └────────────────────────────┘                                    └────────────────────────────┘
```

**Timing contract.** The main loop performs exactly one unit of work per
iteration. If a sample is due (every 20 ms) it is taken and everything else
waits; network servicing only happens in the gap between samples. Both
`Blynk.connect()` and the MQTT reconnect run on throttled paths with bounded
timeouts, so a dead broker cannot stall acquisition.

---

## 2. Hardware Specification & Pinout

### 2.1 Bill of materials

| # | Component | Role | Notes |
|:-:|-----------|------|-------|
| 1 | LOLIN C3 Mini (ESP32-C3) | MCU, WiFi, native USB | 4 MB flash, 320 KB SRAM, **no FPU** |
| 2 | MPU-6050 | 6-DoF accelerometer + gyroscope | GY-521 breakout has pull-ups |
| 3 | HMC5883L / QMC5883L / **QMC5883P** | 3-DoF magnetometer | GY-271/GY-270 boards ship any of the three — see §2.3 |
| 4 | Active buzzer | Audible alert | *Active* — drive HIGH, no PWM needed |
| 5 | Vibration motor | Haptic alert | **Requires a driver transistor** |
| 6 | TP4056 (USB-C) | 1S Li-Po charger | Prefer the DW01+FS8205 protected variant |
| 7 | 1S Li-Po (3.7 V) | Battery | 500–1000 mAh suits a wrist enclosure |
| 8 | SPDT mini slide switch | Power on/off | Wired on the **load** side, not the cell |
| 9 | Momentary push button | Silence an active alarm | GPIO3 → GND, uses the internal pull-up |

### 2.2 Signal pinout

| Signal | ESP32-C3 pin | Connects to | Direction | Notes |
|--------|:------------:|-------------|:---------:|-------|
| `SDA` | **GPIO8** | MPU-6050 SDA + magnetometer SDA | bidir | 4.7 kΩ pull-up to 3V3 |
| `SCL` | **GPIO9** | MPU-6050 SCL + magnetometer SCL | bidir | 4.7 kΩ pull-up to 3V3. GPIO9 is the BOOT strapping pin — fine for I²C once running, but if it is held LOW at reset the chip enters download mode |
| `BUZZER` | **GPIO4** | Active buzzer module `IN` | out | Idle level set by `kBuzzerActiveHigh` |
| `VIBRATION` | **GPIO0** | Motor driver gate/base | out | **Never** drive the motor directly |
| `DISMISS_BTN` | **GPIO3** | Momentary push button → `GND` | in | `INPUT_PULLUP`; pressed reads LOW |
| `3V3` | `3V3` | MPU-6050 VCC, magnetometer VCC | pwr | Both modules are 3.3 V tolerant |
| `GND` | `GND` | Common ground for all modules | pwr | Single star ground |

Pins are declared in [`include/board_config.h`](include/board_config.h) and can be
changed in one place. If your wiring differs, the `scanner` environment sweeps
candidate pin pairs and prints the working combination.

### 2.3 I²C address map

| Address | Device | Identification method | Driver |
|:-------:|--------|----------------------|--------|
| `0x68` | MPU-6050 (AD0 low) | `WHO_AM_I` (reg `0x75`) returns `0x68` | Adafruit MPU6050 |
| `0x69` | MPU-6050 (AD0 high) | same register, auto-detected | Adafruit MPU6050 |
| `0x1E` | HMC5883L (Honeywell, EOL) | ID regs `0x0A–0x0C` spell `'H' '4' '3'` | Adafruit HMC5883 |
| `0x0D` | QMC5883L (QST clone) | chip-ID reg `0x0D` returns `0xFF` | built-in, `sensor_hub.cpp` |
| `0x2C` | **QMC5883P** (QST, current) | chip-ID reg `0x00` returns `0x80` | built-in, `sensor_hub.cpp` |

Boards sold as "HMC5883L" have silently changed part three times. Honeywell
discontinued the HMC5883L, clones moved to the QMC5883L at `0x0D`, and current
stock is usually the **QMC5883P at `0x2C`**. All three have **different register
maps**, not just different addresses, so each needs its own driver. `sensor_hub`
probes `0x1E → 0x0D → 0x2C` and reports which part it found.

> Not to be confused with the **QMC6310**, a different QST magnetometer that lives
> at `0x1C`. If your scan shows `0x1C`, this firmware does not yet drive it.

### 2.4 Power & charging topology

```
                  ┌──────────────── TP4056 (USB-C, 1S Li-Po) ────────────────┐
   USB-C ─────────┤ IN+  IN-                                B+   B-  OUT+ OUT-│
   (charge only)  └────────────────────────────────────────┬────┬───┬────┬────┘
                                                           │    │   │    │
                        1S Li-Po 3.7 V  ───────────────────┘    │   │    │
                        (cell +)                                │   │    │
                        (cell −) ───────────────────────────────┘   │    │
                                                                    │    │
                         SPDT slide switch                          │    │
                        ┌──────────────────┐                        │    │
             OUT+ ──────┤ common      NO   ├────────────────────────┴────┼──► LOLIN C3 Mini  5V pin
                        └──────────────────┘                             │
             OUT- ──────────────────────────────────────────────────────┴────► LOLIN C3 Mini  GND
```

**Wiring rules that matter:**

1. **Switch the load, not the cell.** Put the slide switch between `OUT+` and the
   board's `5V` pin. Charging then works whether the device is on or off, and the
   TP4056's charge-termination circuit always sees the battery.
2. **Feed `5V`, not `3V3`.** The board's LDO regulates the 3.7–4.2 V pack down to
   3.3 V. Injecting battery voltage straight into `3V3` bypasses the regulator and
   puts an out-of-spec rail on the SoC. Expect a brown-out near ~3.5 V pack
   voltage, which is a reasonable low-battery cutoff for a 1S cell.
3. **Only one source at a time.** When the USB-C data cable is plugged into the
   *C3* for flashing, switch the battery rail **off** to avoid two supplies
   fighting on the same node. The TP4056's own USB-C port is for charging only.
4. **Use a protected TP4056 module.** The bare TP4056 has no over-discharge or
   short-circuit protection; the DW01+FS8205 variant adds both. A Li-Po without
   protection is a fire risk in a wrist enclosure.
5. **The vibration motor needs a driver.** An inductive load pulls far more than a
   GPIO can source and will kick back on switch-off. Use an NPN/MOSFET
   (e.g. S8050, 2N2222, AO3400) with a flyback diode across the motor terminals.

---

## 3. Repository Structure

```
Mitifall/
├── platformio.ini                  # 3 build environments, per-env libs & source filters
├── requirements.txt                # Python deps for the ML pipeline
├── compile_commands.json           # merged clangd DB (all 3 envs) — see §8.1
│
├── include/                        # headers + generated model + configuration
│   ├── board_config.h              # pins, 50 Hz timing, unit conversions, sensor ranges
│   ├── i2c_diagnostics.h           # bus scan, chip ID, GPIO sweeper
│   ├── sensor_hub.h                # 9-DoF acquisition in training units
│   ├── feature_extractor.h         # ring buffer + 53-feature contract
│   ├── alarm_manager.h             # non-blocking buzzer/vibration state machine
│   ├── wifi_link.h                 # non-blocking WiFi (WPA2-PSK + Enterprise/EAP)
│   ├── adafruit_io_config.h        # MQTT topics, rate budget, motion threshold (no secrets)
│   ├── blynk_config.h              # virtual-pin map, push cadence (no secrets)
│   ├── RandomForest.h              # ★ GENERATED — flat-array forest + probability API
│   ├── features_order.json         # ★ GENERATED — the 53-feature ordering contract
│   ├── secrets.example.h           # credential template (committed)
│   └── secrets.h                   # real credentials (git-ignored, never commit)
│
├── src/
│   ├── i2c_diagnostics.cpp         # shared
│   ├── sensor_hub.cpp              # shared — MPU6050 + HMC5883L/QMC5883L drivers
│   ├── feature_extractor.cpp       # shared — Arduino-free, host-testable
│   ├── alarm_manager.cpp           # shared
│   ├── wifi_link.cpp               # shared
│   ├── scanner_main.cpp            # entry point ▸ [env:scanner]
│   ├── raw_stream_main.cpp         # entry point ▸ [env:raw_stream]   (Adafruit IO)
│   └── inference_main.cpp          # entry point ▸ [env:ml_inference] (Blynk)
│
├── dataset/
│   ├── process_dataset.py          # UMAFall raw → aligned per-experiment CSVs
│   ├── raw/                        # 746 unzipped UMAFall recordings (git-ignored)
│   └── processed/                  # git-ignored
│       ├── ADL/                    # 538 files
│       └── Fall/                   # 208 files
│
└── ml_pipeline/
    ├── config.py                   # paths, window geometry, model hyperparameters
    ├── data_loader.py              # discovery, Subject_(\d+) parsing, ADL/Fall labelling
    ├── feature_extraction.py       # sliding windows → 53 features
    ├── train_evaluate.py           # LOSO-CV benchmark (RF / SVM-RBF / SVM-lin / LogReg)
    ├── export_model.py             # ★ model → include/RandomForest.h + features_order.json
    ├── run_pipeline.py             # end-to-end runner
    ├── README.md                   # pipeline-specific documentation
    └── outputs/                    # features.csv, LOSO metrics, threshold_sweep.csv, report
```

> **Path note.** The exporter lives at `ml_pipeline/export_model.py`, not
> `scripts/export_model.py` — it shares `config.py` and the feature definitions
> with the training code, which is what guarantees the firmware and the training
> pipeline cannot drift apart.

---

## 4. Prerequisites & Installation

### 4.1 Toolchain

| Tool | Version | Purpose |
|------|---------|---------|
| VS Code | latest | Editor |
| PlatformIO IDE | 6.1+ | Build, flash, monitor, library management |
| Python | 3.10+ | ML pipeline and model export |
| Git | any | Version control |

### 4.2 PlatformIO

Install the **PlatformIO IDE** extension in VS Code. It provisions its own Python
environment and toolchains on first build — the ESP32-C3 RISC-V toolchain and the
Arduino framework download automatically.

The CLI lives inside that environment. It is **not** on your shell `PATH` by
default:

```bash
~/.platformio/penv/bin/pio --version
```

Either use the full path, or run commands from the VS Code PlatformIO terminal
where `pio` is injected. Examples below use plain `pio`.

### 4.3 Python environment

```bash
python -m venv .venv
source .venv/bin/activate        # fish: source .venv/bin/activate.fish
pip install -r requirements.txt
```

`requirements.txt` pulls `pandas`, `numpy`, `scikit-learn` and `micromlgen`.

> **`micromlgen` / `m2cgen` are not required.** `export_model.py` is a
> self-contained transpiler. micromlgen's random-forest port emits nested
> `if`/`else` returning only an `argmax` class label — there is no probability to
> threshold, which this project depends on. m2cgen does emit probabilities but
> unrolls every split into straight-line code, inflating flash and compile time
> for a 60-tree forest. The bundled exporter emits flat `const` tables walked by a
> ~20-line interpreter and reproduces scikit-learn's `predict_proba` exactly.
> `micromlgen` remains in `requirements.txt` only for legacy comparison and can be
> dropped.

### 4.4 Firmware libraries

Declared in `platformio.ini` and fetched automatically on first build:

| Library | Version | Environments |
|---------|---------|--------------|
| `adafruit/Adafruit MPU6050` | `^2.2.6` | all |
| `adafruit/Adafruit Unified Sensor` | `^1.1.14` | all |
| `adafruit/Adafruit HMC5883 Unified` | `^1.2.3` | all |
| `knolleary/PubSubClient` | `^2.8` | `raw_stream` only |
| `blynkkk/Blynk` | `^1.3.2` | `ml_inference` only |

The Adafruit HMC5883 library only drives the genuine Honeywell part. The QST
clones (QMC5883L at `0x0D`, QMC5883P at `0x2C`) share the name but not the
register map, so `sensor_hub.cpp` carries a small register-level driver for each.
No extra library is needed for them.

### 4.5 Credentials

```bash
cp include/secrets.example.h include/secrets.h
# then edit include/secrets.h
```

`include/secrets.h` is git-ignored. The config headers include it automatically
and raise a **compile error** naming the missing macro if it is absent — there are
no silent placeholder fallbacks. If the file exists but the strings are still
empty, the firmware detects that at boot, logs it, and runs fully offline rather
than retrying a connection that cannot succeed.

For campus WiFi using an EAP handshake, set `WIFI_EAP_ENABLED 1` and fill in
`WIFI_EAP_USERNAME` / `WIFI_EAP_PASSWORD` (method `1` = PEAP-MSCHAPv2 covers
almost all universities). See [§11.3](#113-wifi-modes).

---

## 5. Machine Learning Pipeline

### 5.1 Dataset

[UMAFall](https://figshare.com/articles/dataset/UMA_ADL_FALL_Dataset/4214283)
(Casilari et al.) — 746 recordings from **19 subjects**, ~15 s each. Only the
**wrist** sensor (`Sensor ID == 3`) is used, matching the wearable's placement.

```bash
python dataset/process_dataset.py
```

This reads `dataset/raw/`, skips the 40 metadata lines, keeps wrist rows, aligns
the interleaved accelerometer / gyroscope / magnetometer streams onto a common
timestamp, and writes tidy
`[TimeStamp, Ax, Ay, Az, Gx, Gy, Gz, Mx, My, Mz]` files into
`dataset/processed/{ADL,Fall}/`.

### 5.2 Windowing

The wrist stream runs at **~20 Hz with jittery timestamps**, so windows are
defined in *milliseconds*, not sample counts:

| Parameter | Value |
|-----------|-------|
| Window length | 2.0 s |
| Overlap | 50 % (1.0 s hop) |
| Minimum samples | 10 |
| Result | **9 683 windows** (2 689 Fall / 6 994 ADL) |

### 5.3 The 53 features

All are single-pass time-domain statistics — no FFT, no filtering, no cross-axis
terms — so the identical computation is cheap on the MCU.

| Block | Count | Detail |
|-------|:-----:|--------|
| Per-axis statistics | **45** | 9 axes (`Ax…Mz`) × `{mean, std, min, max, var}` |
| Acceleration vector magnitude | **3** | `√(Ax²+Ay²+Az²)` → `{max, mean, std}` |
| Angular velocity vector magnitude | **3** | `√(Gx²+Gy²+Gz²)` → `{max, mean, std}` |
| Signal Magnitude Area (SMA) | **2** | `mean(|x|+|y|+|z|)` for accel and gyro |
| **Total** | **53** | |

`std` uses the population definition (`ddof = 0`) to match a naive embedded
implementation. `var` is kept alongside `std` despite being redundant by
construction. The firmware computes variance in **two passes** (mean first, then
squared deviations) because the single-pass `E[x²] − E[x]²` form loses most of its
significant digits in `float` when the mean dwarfs the deviation — exactly the
case for magnetometer axes with large hard-iron offsets.

The canonical ordering is emitted to `include/features_order.json` and mirrored in
`RandomForest.h`. **The firmware feature vector must follow it exactly.**

### 5.4 Training & evaluation

```bash
python ml_pipeline/run_pipeline.py            # extract features + LOSO benchmark
python ml_pipeline/run_pipeline.py --reuse-features
python ml_pipeline/train_evaluate.py --models random_forest svm_rbf
```

Evaluation uses **`LeaveOneGroupOut` grouped on subject ID** — 19 folds, one per
person. Windows from one recording are highly correlated and windows from one
subject share gait and posture signatures, so a random split would leak subject
identity and inflate every score.

### 5.5 Exporting the model

```bash
# default: 53-feature 9-DoF model → include/RandomForest.h
python ml_pipeline/export_model.py

# 38-feature accelerometer+gyroscope model (no magnetometer required)
python ml_pipeline/export_model.py --axes acc_gyro

# smaller forest for tighter flash budgets
python ml_pipeline/export_model.py --n-estimators 30 --max-depth 8

# report only, write nothing
python ml_pipeline/export_model.py --no-write
```

Outputs:

| File | Contents |
|------|----------|
| `include/RandomForest.h` | Flat `const` tables + `predictProba()`, `voteCount()`, `predict()`, self-test vectors |
| `include/features_order.json` | The 53-feature ordering, units, and window geometry |
| `ml_pipeline/outputs/threshold_sweep.csv` | LOSO out-of-fold metrics per decision threshold |

The generated API:

```cpp
float p        = RandomForest::predictProba(x);       // == sklearn predict_proba[:,1]
uint16_t votes = RandomForest::voteCount(x);          // trees favouring "fall"
bool fall      = RandomForest::predict(x, 0.30f);     // thresholded decision
bool ok        = RandomForest::selfTest();            // flashed tables vs sklearn
```

**The exporter refuses to write unless its own interpreter reproduces
scikit-learn's `predict_proba` across the entire dataset.** Measured agreement:
`max |diff| = 4.4e-16`.

### 5.6 Tuning `FALL_THRESHOLD`

Defined in `src/inference_main.cpp`:

```cpp
static constexpr float FALL_THRESHOLD = 0.30f;
```

Choose it from the **out-of-fold** sweep — never from training-set scores. Pooled
over all 19 LOSO folds (9 683 windows, 2 689 positive):

| Threshold | Sensitivity | Specificity | Precision | F1 | Missed falls | False alarms |
|:---------:|:-----------:|:-----------:|:---------:|:--:|:------------:|:------------:|
| 0.10 | 0.9725 | 0.2782 | 0.3413 | 0.5052 | 74 | 5048 |
| 0.15 | 0.9312 | 0.4919 | 0.4133 | 0.5725 | 185 | 3554 |
| 0.20 | 0.8977 | 0.6516 | 0.4976 | 0.6403 | 275 | 2437 |
| 0.25 | 0.8605 | 0.7538 | 0.5733 | 0.6882 | 375 | 1722 |
| **0.30** | **0.8353** | **0.8237** | 0.6456 | 0.7283 | **443** | **1233** |
| 0.35 | 0.8070 | 0.8705 | 0.7055 | 0.7528 | 519 | 906 |
| 0.40 | 0.7772 | 0.9066 | 0.7619 | 0.7695 | 599 | 653 |
| 0.50 | 0.7196 | 0.9477 | 0.8409 | **0.7756** | 754 | 366 |
| 0.60 | 0.6475 | 0.9683 | 0.8869 | 0.7485 | 948 | 222 |
| 0.70 | 0.5746 | 0.9784 | 0.9110 | 0.7047 | 1144 | 151 |
| 0.80 | 0.4805 | 0.9886 | 0.9417 | 0.6363 | 1397 | 80 |

**How to read this for a wearable.** Decisions are emitted once per second.
Specificity 0.8237 at the default threshold means roughly **one false alert every
six seconds of ordinary movement** — unusable with a buzzer strapped to a wrist,
even though the sensitivity looks attractive. Two levers fix that:

```cpp
static constexpr uint8_t CONFIRMATION_WINDOWS = 1;   // require N consecutive positives
static constexpr uint32_t RETRIGGER_COOLDOWN_MS = 6000;
```

Raising `CONFIRMATION_WINDOWS` to `2` or `3` trades ~1–2 s of detection latency for
a large reduction in nuisance alerts, and is usually a better move than pushing
the threshold up, because it suppresses isolated spikes while keeping the model's
sensitivity to sustained fall signatures. For field use, start at
`FALL_THRESHOLD = 0.35–0.40` with `CONFIRMATION_WINDOWS = 2`.

---

## 6. Firmware Environments: Build & Upload

Three environments share every module but compile **one** entry point each, so
exactly one `setup()`/`loop()` pair exists per binary:

```ini
[common]
build_src_filter = +<*.cpp> -<*_main.cpp>

[env:scanner]
build_src_filter = +<*.cpp> -<raw_stream_main.cpp> -<inference_main.cpp>
```

> `build_src_filter` is the PlatformIO 6 name for the option historically called
> `src_filter`. The old name still works but is deprecated and warns.

`default_envs = ml_inference` means a bare `pio run` targets the application.
**Always pass `-e` when you mean a specific target** — otherwise
`pio run -t upload` would flash all three in sequence.

### VS Code GUI

PlatformIO sidebar (alien icon) → **Project Tasks** → pick the environment →
**Build**, **Upload**, **Monitor**, or **Upload and Monitor**. The status-bar
environment switcher controls what the toolbar buttons act on.

### Step 1 — `[env:scanner]`: verify the hardware

```bash
pio run -e scanner -t upload -t monitor
```

Scans `0x01–0x7F`, verifies `WHO_AM_I`, distinguishes HMC5883L from a QMC5883L
clone, sweeps candidate SDA/SCL pairs if nothing answers, then pulses the buzzer
and motor individually so wiring can be confirmed by ear and by touch.

```
  0x68  MPU-6050                 (ID 0x68)
  0x0D  QMC5883L (clone)         (ID 0xFF)
Expected devices:
  MPU-6050 (0x68/0x69) : FOUND
  HMC5883L (0x1E)      : missing
  QMC5883L (0x0D)      : FOUND
Actuator test: buzzer only, then motor only, then both
```

Do not proceed until both sensors are found and both actuators respond.

### Step 2 — `[env:raw_stream]`: validate sampling and Adafruit IO

```bash
pio run -e raw_stream -t upload -t monitor
```

Streams one CSV line per sample in the **same units and column order as
`dataset/processed/`**, so a capture can be fed straight back into the Python
pipeline:

```
Timestamp_ms,Ax,Ay,Az,Gx,Gy,Gz,Mx,My,Mz
1240,-0.94727,-0.30640,0.16675,-4.1563,3.9375,-6.9609,29.833,-62.667,-71.667
```

Capture to a file (`--quiet` suppresses PlatformIO's banner so the file starts at
the CSV header):

```bash
pio device monitor -e raw_stream --quiet > capture.csv
```

Simultaneously publishes to Adafruit IO and subscribes to the dashboard's buzzer
toggle. Set `kEnableCloud = false` in `src/raw_stream_main.cpp` for purely local
logging.

### Step 3 — `[env:ml_inference]`: real-time edge inference

```bash
pio run -e ml_inference -t upload -t monitor      # or just: pio run -t upload -t monitor
```

```
 model     : 53 features, 60 trees, 23634 nodes
 window    : 100 samples @ 50 Hz (2.0 s), hop 50 samples
 threshold : 0.30, confirmation 1 window(s)
Model self-test: PASS (3 vectors, 412 us)
WiFi mode: WPA2-Enterprise (PEAP/MSCHAPv2)
w0     p=0.017 votes=1/60  adl      feat=1832us infer=387us |a|max=1.04g cloud
w1     p=0.412 votes=26/60 FALL?    feat=1829us infer=402us |a|max=3.71g cloud
*** FALL DETECTED  p=0.412  alerting for 4000 ms ***
```

Each window reports the probability, tree votes, feature-extraction and inference
latency from `micros()`, and whether telemetry is live.

---

## 7. Cloud Dashboard Setup

### 7.1 Adafruit IO (`[env:raw_stream]`)

**Feeds** — Feeds → *New Feed*. Names must match the firmware exactly:

| Feed key | Type | Direction | Widget |
|----------|------|-----------|--------|
| `accel-mag` | numeric | device → cloud, every 2.5 s | Line Chart |
| `motion-status` | numeric `0`/`1` | device → cloud, on change | Indicator |
| `buzzer-command` | numeric `0`/`1` | cloud → device | Toggle |

> **Naming discrepancy.** The firmware publishes to **`accel-mag`**. If you
> created a feed called `raw-accel-mag`, either rename the feed or change
> `AIO_FEED_ACCEL_MAG` in `include/adafruit_io_config.h`. Publishing auto-creates
> a missing feed, but **subscribing does not** — so `buzzer-command` must exist
> before the toggle will do anything.

Credentials: username is top-right on io.adafruit.com; the key is behind the
yellow key icon → *Active Key*. Put them in `include/secrets.h` as
`AIO_USERNAME` / `AIO_KEY`.

**Widgets** — Dashboards → *New Dashboard*:

| Widget | Feed | Settings |
|--------|------|----------|
| **Line Chart** | `accel-mag` | Y min `0`, Y max `4` (g); history 1 hour. Rest ≈ 1.0 g, walking 1.5–2.5 g, impacts higher |
| **Indicator** | `motion-status` | Condition `= 1`, on-colour green, off grey |
| **Toggle** | `buzzer-command` | On `1`, Off `0` → fires buzzer + motor via `AlarmManager::trigger()` |

### 7.2 Blynk IoT (`[env:ml_inference]`)

**Template** — Blynk Console → Developer Zone → Templates → *New Template*
(hardware **ESP32**, connection **WiFi**). Copy `BLYNK_TEMPLATE_ID`,
`BLYNK_TEMPLATE_NAME`, and the device's `BLYNK_AUTH_TOKEN` into
`include/secrets.h`.

**Datastreams** — types must match what the firmware writes:

| Pin | Name | Type | Range | Direction |
|:---:|------|------|:-----:|-----------|
| `V0` | Fall Probability | **Double** | 0 – 1 | device → cloud, 1 Hz |
| `V1` | Fall Alert | **Integer** | 0 – 1 | device → cloud, on change + 2 s heartbeat |
| `V2` | Reset / Test | **Integer** | 0 – 1 | cloud → device |

**Web dashboard widgets:**

| Widget | Datastream | Settings |
|--------|-----------|----------|
| **Chart / SuperChart** | `V0` | Fixed Y range 0–1; add a reference line at **0.30** to make the threshold visible against the trace |
| **LED** | `V1` | Red when `1` |
| **Button** | `V2` | Mode **Push**, On value `1` |

`V2` behaviour: dismisses an active alarm; if no alarm is active it runs a short
actuator self-test instead.

**Optional push notifications** — Template → Events → create an event with code
exactly `fall_detected`, severity Warning, notifications enabled. The firmware
calls `Blynk.logEvent("fall_detected")`. Without the event defined the call is
ignored and the local buzzer plus `V1` still work.

---

## 8. Troubleshooting & Common Pitfalls

### 8.1 `Arduino.h` / `<algorithm>` not found in the editor (build succeeds)

This is an **IntelliSense/clangd indexing problem, not a compile error**. Confirm
with `pio run -e ml_inference` — if that succeeds, the toolchain is fine.

Two distinct causes:

1. **Stale database.** `compile_commands.json` still references files that no
   longer exist. Regenerate it.
2. **Multi-environment blind spot.** Because each environment excludes the other
   entry points, a database generated for one environment contains **no entry** for
   the others. Generate for `ml_inference` and `raw_stream_main.cpp` has no include
   paths at all — so `Arduino.h`, then `<algorithm>`, fail to resolve.

Fix — merge all three environments into one database:

```bash
for env in scanner raw_stream ml_inference; do
  pio run -e $env -t compiledb
  cp compile_commands.json /tmp/db_$env.json
done
python - <<'PY'
import json
merged, seen = [], set()
for env in ("scanner", "raw_stream", "ml_inference"):
    for e in json.load(open(f"/tmp/db_{env}.json")):
        k = (e.get("directory", ""), e["file"])
        if k not in seen:
            seen.add(k); merged.append(e)
json.dump(merged, open("compile_commands.json", "w"), indent=2)
print(f"merged {len(merged)} entries")
PY
```

Then reload the index: **PlatformIO: Rebuild IntelliSense Index**, or
**clangd: Restart language server**, or **Developer: Reload Window**.

> Re-running `pio run -t compiledb` for a single environment **overwrites** the
> merged database and the squiggles return on the other two entry points.

### 8.2 Magnetometer answers at `0x0D` or `0x2C` instead of `0x1E`

Boards sold as "HMC5883L" have changed silicon three times. Each part has a
different address **and** a different register map, so this is not a matter of
patching one constant:

| Address | Part | Data registers | Config | Sensitivity @ 8 G |
|:-------:|------|:--------------:|--------|:-----------------:|
| `0x1E` | HMC5883L | `0x03–0x08` | Adafruit library | — |
| `0x0D` | QMC5883L | `0x00–0x05` | ctrl `0x09`/`0x0A`, set/reset `0x0B` | 3000 LSB/G → `µT = raw / 30` |
| `0x2C` | QMC5883P | `0x01–0x06` | ctrl1 `0x0A`, ctrl2 `0x0B` | 3750 LSB/G → `µT = raw / 37.5` |

All three are handled automatically. `sensor_hub` probes `0x1E` and verifies the
`'H' '4' '3'` ID registers, then `0x0D`, then `0x2C` verifying chip ID `0x80`, and
reports the winner:

```
  QMC5883P ready at 0x2C (8 G range, 100 Hz ODR, 3750 LSB/G)
```

On the QMC5883P, note that a soft reset (ctrl2 bit 7) clears the control
registers, so range must be written **before** mode — the driver does this in the
right order.

If a scan shows `0x1C`, that is a **QMC6310**, a different QST part that this
firmware does not yet drive. If **no** magnetometer address answers, check power
and pull-ups first, then re-export a 6-DoF model (`--axes acc_gyro`), since
`[env:ml_inference]` deliberately **refuses to run** a 53-feature model without a
magnetometer rather than feeding it zeros.

### 8.3 Adafruit IO rate limiting (30 data points/minute)

Publishing two feeds every 2.5 s is **48 points/min** and gets throttled. The
budget is counted per feed, not per publish call:

| Feed | Cadence | Points/min |
|------|---------|:----------:|
| `accel-mag` | every 2.5 s | 24 |
| `motion-status` | on change, ≥ 15 s apart | ≤ 4 |
| **Total** | | **≤ 28** |

That is why `motion-status` is edge-triggered rather than periodic. The firmware
also subscribes to `{user}/throttle` and `{user}/errors`, so a violation appears
as a log line instead of silent data loss:

```
# ADAFRUIT IO WARNING: ...
```

Adding a feed? Recompute the budget and raise `kPublishIntervalMs`.

### 8.4 No serial output on ESP32-C3

The LOLIN C3 Mini has no USB-UART bridge — the console runs over the SoC's
**native USB CDC**, which requires:

```ini
build_flags =
    -D ARDUINO_USB_MODE=1
    -D ARDUINO_USB_CDC_ON_BOOT=1
```

Both are already set. Additional notes:

- The port **re-enumerates after flashing**, so `-t upload -t monitor` is more
  reliable than starting the monitor separately.
- Every entry point waits up to 3 s for the host to attach before printing, so the
  boot banner is not lost.
- If an upload fails to start, hold **BOOT (GPIO9)** and tap **RESET** to force
  download mode.
- On Linux, ensure your user is in the serial group (`uucp` on Arch, `dialout` on
  Debian/Ubuntu) and that `99-platformio-udev.rules` is installed.

### 8.5 WiFi connects but MQTT/Blynk does not

| Symptom | Cause |
|---------|-------|
| `associating…` repeats, never `connected` | Wrong PSK, or an EAP network with `WIFI_EAP_ENABLED 0` |
| `connected, ip=…` then `mqtt connect failed, state=-2` | Captive portal, or outbound port 1883 blocked |
| Works on a phone hotspot, not on campus | Network policy, not firmware |

A phone hotspot is the fastest way to isolate firmware from network policy.

### 8.6 Flash region overflow

`[env:ml_inference]` uses **85 %** of the default 1.31 MB app partition. If you
add TLS, OTA, or a larger forest, switch the partition table:

```ini
board_build.partitions = min_spiffs.csv   ; ~1.9 MB app, keeps OTA
; or
board_build.partitions = huge_app.csv     ; ~3 MB app, no OTA
```

Alternatively shrink the model: `--n-estimators 30 --max-depth 8`.

### 8.7 Buzzer or motor runs continuously from power-up

Three different causes, in the order worth checking:

**1. Inverted polarity.** Cheap 3-pin breakouts are frequently active-LOW, so
driving the pin LOW for "off" turns them on permanently. Polarity is configured
per device, because a buzzer board and a motor driver are often opposite:

```cpp
constexpr bool kBuzzerActiveHigh = false;
constexpr bool kVibrationActiveHigh = false;
```

Flash `[env:scanner]`. It drives each actuator alone for ~0.9 s and prints the
configured polarity. If a device is silent during *its own* step but runs the rest
of the time, that device's flag is inverted.

**2. Floating pin during boot.** Before firmware runs, a GPIO is an undriven
input. An active-low module sees that as "on", so it sounds from power-up until
`setup()` executes — which includes the 3 s USB CDC wait. All three entry points
now call `AlarmManager::forceOff()` as their **first statement**, and it presets
the output latch *before* `pinMode(OUTPUT)` so there is no LOW glitch on the
transition. For a completely silent boot, add a 10 kΩ pull-up (active-low
modules) or pull-down (active-high) on each actuator input line — firmware cannot
control a pin before the CPU starts.

**3. Wiring.** If a device stays on with *either* polarity setting, firmware is
not the problem. Check that the module's `IN` pin is really on the GPIO rather
than tied to VCC, and that the motor is fed through its driver transistor instead
of directly off the 3V3 rail. `AlarmManager::update()` actively re-asserts the off
level every loop iteration whenever no alert is running, so a persistent "on" with
correct wiring is not possible from the firmware side.

### 8.8 Silencing an alarm

Two independent paths, both non-blocking:

| Path | Mechanism |
|------|-----------|
| **Physical button** | GPIO3 → GND, 40 ms debounce, handled in `AlarmManager::update()` |
| **Dashboard** | Blynk `V2` push button (`ml_inference`), or the Adafruit IO toggle (`raw_stream`) |

A button press stops both actuators immediately, clears the latched detection
state, restarts the re-trigger cooldown, and pushes `V1 = 0` to Blynk. In
`raw_stream` it also publishes `0` back to `buzzer-command` so the dashboard
switch stops showing an active command. Verify the wiring with `[env:scanner]`,
which prints `Button GPIO3: PRESSED` / `released` live.

### 8.9 Sensor reads fail intermittently

Long jumper wires plus 400 kHz I²C is a common culprit. Lower
`kI2cFrequencyHz` to `100000` in `board_config.h`, shorten the leads, and confirm
4.7 kΩ pull-ups exist on `SDA`/`SCL`. `[env:raw_stream]` reports cumulative failed
reads:

```
# warning: 12 failed sensor reads so far
```

---

## 9. Measured Performance

All figures are **Leave-One-Subject-Out**, pooled across 19 folds — i.e. measured
on subjects the model never trained on.

### 9.1 Model comparison (threshold 0.50)

| Model | Sensitivity | Specificity | F1 | Accuracy | Missed falls | False alarms |
|-------|:-----------:|:-----------:|:--:|:--------:|:------------:|:------------:|
| **Random Forest** | 0.7196 | **0.9477** | **0.7756** | **0.8843** | 754 | **366** |
| SVM (RBF) | 0.7605 | 0.9111 | 0.7636 | 0.8693 | 644 | 622 |
| Logistic Regression | 0.7646 | 0.7322 | 0.6213 | 0.7412 | 633 | 1873 |
| SVM (linear) | **0.7917** | 0.7070 | 0.6201 | 0.7306 | **560** | 2049 |

The linear models trade a few points of recall for roughly **5× the false
alarms** — the wrong trade for a wearable. Random Forest wins on F1 and is also
the only candidate that ports cleanly to flash as a compact table.

### 9.2 Aggregated confusion matrix — Random Forest

```
                     predicted ADL   predicted Fall
    true ADL              6628              366
    true Fall              754             1935
```

### 9.3 Feature-set comparison

| Axis set | Features | Sens @0.30 | Spec @0.30 | F1 @0.50 |
|----------|:--------:|:----------:|:----------:|:--------:|
| `all9` (default) | 53 | 0.8353 | 0.8237 | 0.7756 |
| `acc_gyro` | 38 | **0.8371** | **0.8336** | **0.7871** |

Dropping the magnetometer **improves** generalisation. See
[§12.1](#121-the-magnetometer-features-do-not-transfer).

### 9.4 On-device model

| Property | Value |
|----------|-------|
| Trees / max depth | 60 / 10 |
| Nodes / leaves | 23 634 / 11 847 |
| Const tables in flash | ~207.8 KiB |
| `predict_proba` agreement with scikit-learn | `4.4e-16` |

---

## 10. Resource Footprint

| Environment | RAM | Flash | Notes |
|-------------|:---:|:-----:|-------|
| `scanner` | 5.0 % (16 344 B) | 20.3 % (266 190 B) | No WiFi stack linked |
| `raw_stream` | 12.2 % (39 884 B) | 67.6 % (885 806 B) | + WiFi, EAP supplicant, PubSubClient |
| `ml_inference` | 13.7 % (44 764 B) | **85.0 %** (1 114 738 B) | + Blynk, 208 KiB forest |

Budget is against 320 KB SRAM and the default 1.31 MB app partition. The ESP32-C3
has **no hardware FPU**, so all `float` maths is software-emulated — the reason
hot paths use `float` rather than `double` and avoid transcendental functions.
Feature-extraction and inference latency are printed per window by
`[env:ml_inference]`.

---

## 11. Configuration Reference

### 11.1 `include/board_config.h`

| Constant | Default | Meaning |
|----------|:-------:|---------|
| `kI2cSdaPin` / `kI2cSclPin` | `8` / `9` | I²C pins |
| `kI2cFrequencyHz` | `400000` | Lower to `100000` for long wires |
| `kBuzzerPin` / `kVibrationPin` | `4` / `0` | Actuator GPIOs |
| `kBuzzerActiveHigh` | `false` | `true` if pin HIGH sounds the buzzer |
| `kVibrationActiveHigh` | `false` | `true` if pin HIGH runs the motor |
| `kDismissButtonPin` | `3` | Momentary button to silence the alarm |
| `kButtonActiveLow` | `true` | `INPUT_PULLUP`, pressed = LOW |
| `kButtonDebounceMs` | `40` | Debounce window |
| `kSampleRateHz` | `50` | Acquisition rate |
| `kWindowSamples` / `kHopSamples` | `100` / `50` | 2.0 s window, 50 % overlap |
| `kAccelLimitG` | `8.0` | Matches the training capture's saturation |
| `kGyroLimitDps` | `256.0` | Matches the training capture's saturation |

The MPU-6050 is configured to **±8 g / ±250 °/s** deliberately: the UMAFall
capture saturates at exactly 8.00 g and 256 °/s per axis, so matching the ranges
reproduces the same clipping instead of presenting the forest with values it never
saw during training.

### 11.2 Unit conventions

| Quantity | Training units | Driver output | Conversion in `sensor_hub.cpp` |
|----------|----------------|---------------|-------------------------------|
| Acceleration | **g** | m/s² | `÷ 9.80665` |
| Angular velocity | **°/s** | rad/s | `× 180/π` |
| Magnetic field | see §12.1 | µT | none |

### 11.3 WiFi modes

| `WIFI_EAP_ENABLED` | Mode | Fields used |
|:------------------:|------|-------------|
| `0` | WPA2-Personal | `WIFI_SSID`, `WIFI_PASSWORD` |
| `1` | WPA2-Enterprise | `WIFI_SSID`, `WIFI_EAP_IDENTITY`, `WIFI_EAP_USERNAME`, `WIFI_EAP_PASSWORD`, `WIFI_EAP_METHOD` |

`WIFI_EAP_METHOD`: `0` = TLS (needs a client certificate), `1` = PEAP-MSCHAPv2
(almost all campuses), `2` = TTLS-MSCHAPv2. Leave `WIFI_EAP_IDENTITY` empty to
reuse the username as the outer identity. Credentials are capped at **64 bytes**
by the supplicant; longer values are rejected with a log line.

---

## 12. Known Limitations

Documented honestly, because each one affects how much to trust the numbers.

### 12.1 The magnetometer features do not transfer

27 of the 53 features come from a magnetometer, and UMAFall's magnetometer channel
is **not calibrated to µT**: its median |M| is ~155 against an Earth field of
25–65 µT, with large per-axis hard-iron offsets. An HMC5883L reporting true µT
therefore lands in a different region of feature space than the model was trained
on. Measurement also shows the magnetometer is **not earning its place** — the
38-feature accelerometer+gyroscope model scores *better* under LOSO at every
threshold (§9.3), because magnetometer features largely encode room- and
session-specific orientation.

**Recommendation for deployment:** `python ml_pipeline/export_model.py --axes
acc_gyro`. The firmware adapts automatically through the header's feature-index
map; no C++ changes are needed.

### 12.2 Sample-rate mismatch: trained at ~20 Hz, deployed at 50 Hz

Training windows hold ~40 samples; the firmware's hold 100. Window *duration*
matches at 2.0 s and `mean`/`min`/`max`/SMA are rate-insensitive, but `std` and
`var` pick up more high-frequency content at 50 Hz. `kSampleRateHz` and
`kWindowSamples` are single constants if field testing shows drift.

### 12.3 Label granularity caps achievable sensitivity

Following the dataset convention, **every** window of a Fall recording is labelled
positive — but a fall trial is ~15 s of mostly ordinary movement around a
sub-second impact, so roughly two thirds of "positive" windows contain no fall
event. `--fall-label-mode impact` labels only windows within ±1 s of the peak
acceleration; under that stricter and more realistic definition the same forest
scores sensitivity 0.509 / specificity 0.973. Use `file` mode for comparability
with published UMAFall results, `impact` mode for an honest estimate of on-device
difficulty.

### 12.4 Six subjects contributed no falls

Subjects 1, 3, 9, 10, 12 and 13 performed ADLs only. Their folds contain no
positive samples, so sensitivity, precision and F1 are mathematically undefined
there. Those folds report `NaN` rather than a misleading `0.0`, fold averages are
NaN-aware, and the `n_folds_<metric>` columns in `loso_summary.csv` record how
many folds each average rests on (13 of 19 for recall-side metrics). Pooled
metrics from the summed confusion matrix are the headline numbers because fold
sizes range from 130 to 2 457 windows.

### 12.5 Security posture

- **MQTT is plaintext** on port 1883, so the Adafruit IO key crosses the network in
  the clear. Lab-grade, not production. Port 8883 with `WiFiClientSecure` and a CA
  bundle is the upgrade path.
- **EAP without a CA certificate** cannot verify the RADIUS server, so the device
  will hand campus credentials to any AP advertising the same SSID — an evil-twin
  risk with your real account. Set `WIFI_EAP_CA_PEM` if IT will provide the
  certificate.
- **Credentials live in flash in plaintext** and are readable via
  `esptool read_flash`. `include/secrets.h` is git-ignored, but do not lend the
  assembled device out.

### 12.6 Not a medical device

This is a university engineering project. It has been validated against a public
dataset, **not** against real falls by real at-risk users, and carries no
certification. Do not deploy it as anyone's actual safety net.

---

## 13. Verification Methodology

What has actually been tested, and how:

| Claim | Method | Result |
|-------|--------|--------|
| Firmware features == Python features | Host harness compiles the real `feature_extractor.cpp`, pushes 24 real dataset windows through the real `RingBuffer`, diffs all 53 outputs | max relative diff `3.2e-6` |
| Firmware probability == scikit-learn | Same harness, through the generated `RandomForest.h` | max diff `2.4e-7` |
| Exported tables == trained model | Exporter replays its interpreter over all 9 683 windows before writing | max diff `4.4e-16` |
| Flashed tables intact | 3 self-test vectors embedded in the header, checked at boot | `PASS` |
| No subject leakage | Asserted every LOSO fold is disjoint by subject *and* by source recording | 0 leaks |
| Every window scored once | Per-model fold test-set sizes sum to the dataset size | exact |
| All environments compile | `pio run -e scanner -e raw_stream -e ml_inference` | 3 × SUCCESS, no warnings from project sources |

Not verified: **anything requiring the physical device.** Sensor bring-up, I²C
pin discovery, actuator behaviour, WiFi association, MQTT/Blynk connectivity, and
on-device latency figures all need hardware in hand.

---

## 14. License & Credits

### License

Released under the **MIT License**. A `LICENSE` file is not yet present in this
repository — add one before publishing.

### Dataset

**UMAFall: A Multisensor Dataset for the Research on Automatic Fall Detection** —
Eduardo Casilari, José A. Santoyo-Ramón, José M. Cano-García. Universidad de
Málaga. Published on
[Figshare](https://figshare.com/articles/dataset/UMA_ADL_FALL_Dataset/4214283).
Please honour the dataset's own licence terms and cite the authors in any
derivative work.

### Third-party libraries

| Library | Author | License |
|---------|--------|---------|
| Adafruit MPU6050 / Unified Sensor / HMC5883 | Adafruit Industries | BSD / MIT |
| PubSubClient | Nick O'Leary | MIT |
| Blynk | Blynk Inc. | MIT |
| Arduino core for ESP32 | Espressif Systems | LGPL-2.1 |
| scikit-learn, pandas, NumPy | respective authors | BSD-3-Clause |

---

<div align="center">

**Mitifall** — edge inference first, cloud second, alarm always.

</div>
