# Mitifall

Wrist-worn fall detector on an ESP32-C3. The bracelet reads a 9-DoF IMU (and an
optional BMP280 barometer) at 50 Hz, detects falls, sounds an audio-haptic alarm
(buzzer and vibration motor), logs telemetry to internal flash SQLite storage,
and reports to an Adafruit IO dashboard over Wi-Fi.

---

## System Architecture

```
                                  MITIFALL BRACELET (ESP32-C3)
+-----------------------------------------------------------------------------------------------+
|                                                                                               |
|  [MPU-6050] (Acc + Gyro)  ---\                                                                |
|  [HMC5883L] (Magnetometer) ----> [SensorHub @ 50 Hz] -> [Ring Buffer: 100 samples (2.0 s)]   |
|  [BMP280]   (Barometer)   ---/                                |                               |
|                                                               v                               |
|                                                   +-----------------------+                   |
|                                                   |   Detection Paths     |                   |
|                                                   +-----------------------+                   |
|                                                   | 1. env:ml_inference   | (On-device RF)    |
|                                                   | 2. env:raw_stream     | (Rule cascade)    |
|                                                   | 3. env:dl_stream      | (USB -> Host DL)  |
|                                                   +-----------------------+                   |
|                                                               |                               |
|                                      +------------------------+------------------------+      |
|                                      | Fall Confirmed                                  |      |
|                                      v                                                 v      |
|                         [Actuators & AlarmManager]                           [Local Storage]  |
|                         - Active Buzzer (GPIO2)                              - LittleFS Flash |
|                         - Vibration Motor (GPIO1)                            - SQLite DB      |
|                         - Physical Dismiss Button (GPIO3)                    - FIFO Pruning   |
+--------------------------------------|-------------------------------------------------+------+
                                       |
                                       v 2.4 GHz Wi-Fi (MQTT)
                         +-----------------------------------+
                         |      Adafruit IO Cloud Broker     |
                         |  accel-mag · pressure · altitude  |
                         |  fall-detected · fall-log · LWT   |
                         +-----------------+-----------------+
                                           | WebSockets
                                           v
                         +-----------------------------------+
                         |       Web Telemetry Dashboard     |
                         |  - Real-time |a| chart            |
                         |  - Barometer metrics (h, dh, p)   |
                         |  - Live sensor log & CSV export   |
                         |  - Fall event audit trail         |
                         |  - Remote buzzer trigger/silence  |
                         +-----------------------------------+
```

---

## Hardware

| Part | Connection | Notes |
|------|------------|-------|
| **LOLIN C3 Mini** (ESP32-C3) | — | Single-core 32-bit RISC-V @ 160 MHz, 4 MB Flash, 2.4 GHz Wi-Fi |
| **MPU-6050** (accelerometer + gyroscope) | I²C: SDA GPIO8, SCL GPIO9 | 6-DoF, address `0x68` (AD0 to GND) |
| **HMC5883L / QMC5883L / QMC5883P** (magnetometer) | same I²C bus | 3-DoF, address `0x1E` (HMC) or `0x0D` (QMC) |
| **BMP280 / BME280** (optional barometer) | same I²C bus | Address `0x76` or `0x77`, altitude tracker |
| **Active buzzer** | GPIO2 | Active-High audio alert |
| **Vibration motor module** | GPIO1 | Driven via NPN transistor / MOSFET switch |
| **Dismiss button** | GPIO3 to GND | Active-Low with internal pull-up |

Pins, thresholds, and actuator polarities are configured in [`include/board_config.h`](include/board_config.h).

---

## Firmware Environments

| Environment | What it does | Cloud / Telemetry |
|-------------|--------------|-------------------|
| `scanner` | I²C bus scan, sensor verification, actuator test | — |
| `raw_stream` | Rule-based cascade fall detection on bracelet, local SQLite log | Adafruit IO (MQTT) |
| `ml_inference` | On-device Random Forest classifier (53 features, 2.0 s window), local SQLite log | Adafruit IO (MQTT) |
| `dl_stream` / `dl_inference` | Streams raw 50 Hz CSV over USB; fall detection executed by CNN-BiLSTM on host | Adafruit IO (MQTT) |

```bash
cp include/secrets.example.h include/secrets.h   # WiFi + Adafruit IO credentials
pio run -e ml_inference -t upload -t monitor    # or -e raw_stream / dl_stream
```

> [!NOTE]
> The ESP32-C3 only connects to **2.4 GHz** Wi-Fi networks. If Wi-Fi credentials are unset in `include/secrets.h`, the firmware gracefully falls back to offline operation.

---

## Python Setup

```bash
python -m venv venv
source venv/bin/activate            # fish: source venv/bin/activate.fish
pip install -r requirements.txt -r deep_learning/requirements.txt -r storage/requirements.txt
```

---

## Machine Learning (Random Forest)

Trained on the UMAFall wrist recordings (19 subjects), resampled to 50 Hz, evaluated with Leave-One-Subject-Out (LOSO) cross-validation.

- **Window Geometry**: 2.0 s window (100 samples @ 50 Hz) with 50% overlap (1.0 s / 50 samples hop).
- **53 Time-Domain Features**: 9 axes ($Ax, Ay, Az, Gx, Gy, Gz, Mx, My, Mz$) $\times$ 5 statistics (`mean`, `std`, `min`, `max`, `var`), magnitude $|a|$ (`max`, `mean`, `std`), magnitude $|\omega|$ (`max`, `mean`, `std`), and Signal Magnitude Area (`acc_sma`, `gyr_sma`).
- **Operating Threshold**: Tuned to `FALL_THRESHOLD = 0.30` to maximize fall sensitivity (83.7%).
- **TinyML Deployment**: Exported as flat-array C++ static tables in [`include/RandomForest.h`](include/RandomForest.h) with zero heap allocations, fast execution (<5 ms per window), and boot self-test vectors.
- **Connectivity**: Connects to 2.4 GHz Wi-Fi, syncs NTP time, publishes periodic telemetry and fall alerts to Adafruit IO, and records local LittleFS SQLite logs.

```bash
python dataset/process_dataset.py
python ml_pipeline/run_pipeline.py      # feature extraction + LOSO benchmark
python ml_pipeline/export_model.py      # -> generates include/RandomForest.h
pio run -e ml_inference -t upload -t monitor
```

---

## Deep Learning (CNN-BiLSTM)

Runs on the laptop or edge gateway, not on the bracelet. The bracelet flashes `env:dl_stream` to stream raw 9-DoF samples over USB serial, and raises the alarm when the host model reports a fall.

```bash
pio run -e dl_stream -t upload
python deep_learning/infer_stream.py --port /dev/ttyACM0
```

Useful options: `--threshold 0.7`, `--quiet`, `--int8`.
`python deep_learning/infer_stream.py --selftest` checks the streaming path against the batch evaluation. Retraining and export:

```bash
python deep_learning/train.py --arch cnn_bilstm --with-gyro
python deep_learning/export_onnx.py --int8
```

---

## Benchmark Results (LOSO, 9 683 windows)

Pooled Leave-One-Subject-Out (LOSO) cross-validation across all 19 UMAFall subjects (2 689 Fall / 6 994 ADL windows):

| Model | Accuracy | F1 Macro | F1 Fall | Sensitivity (Recall) | Specificity | ROC-AUC | Operating Notes |
|-------|:--------:|:--------:|:-------:|:--------------------:|:-----------:|:-------:|-----------------|
| **Random Forest (RF-53 @ 0.30)** | **0.842** | **0.816** | **0.747** | **0.837** | **0.844** | **0.906** | **Deployed firmware threshold (`ml_inference`)** |
| Random Forest (val-tuned threshold) | 0.877 | 0.846 | 0.778 | 0.777 | 0.915 | 0.906 | Youden's J optimal threshold (~0.38) |
| Random Forest (RF-53 @ 0.50 default) | 0.884 | 0.841 | 0.776 | 0.720 | 0.948 | 0.906 | Standard pipeline default (`train_evaluate.py`) |
| CNN-BiLSTM + gyro (9 ch.) | 0.880 | 0.851 | 0.784 | 0.783 | 0.918 | 0.928 | Host model (val-tuned threshold) |

> [!NOTE]
> **Threshold Trade-off (`0.50` vs `0.30`)**:
> - At the default threshold (`0.50`), Random Forest achieves high specificity ($94.8\%$) and $88.4\%$ accuracy, but misses 754 falls (sensitivity $72.0\%$).
> - For wearable patient safety, the deployed threshold is tuned to `0.30` (`FALL_THRESHOLD = 0.30f` in `src/inference_main.cpp`). This boosts sensitivity to **$83.7\%$** (cutting missed falls to 438) with an acceptable trade-off in specificity ($84.4\%$) and overall accuracy ($84.2\%$).
> - F1 Macro averages both ADL and Fall classes, while F1 Fall evaluates the positive class only. Detailed fold breakdowns are available in [`ml_pipeline/outputs/loso_report.md`](ml_pipeline/outputs/loso_report.md).

Full evaluation report: `deep_learning/outputs/evaluation/report.md` and `ml_pipeline/outputs/loso_report.md`.

---

## Telemetry Logging (Internal Flash SQLite)

The bracelet firmware (`ml_inference`, `raw_stream`, `dl_stream`) maintains an embedded SQLite database on internal SPI flash memory (LittleFS, ~1.9 MB via [`partitions_log.csv`](partitions_log.csv)):
- **Periodic Log**: One row every 2.5 s with UTC timestamp, mean $|a|$, pressure, altitude, and alarm state.
- **Event Log**: Instant rows on fall detection (`fall`, `fall_ml`, `fall_dl`) and button dismissal (`dismiss`).
- **Resilience**: Operates completely offline without Wi-Fi. Automatic FIFO pruning deletes the oldest 2 000 rows when storage usage exceeds 85%.

### Pulling Data to Host:
```bash
# Dump SQLite database from bracelet over USB serial:
python storage/pull_device.py --port /dev/ttyACM0            # add --clear to wipe bracelet DB

# Export to timestamped CSV file:
python storage/export_csv.py                                 # -> storage/exports/telemetry_<UTC>.csv
```

Serial commands supported by the bracelet: `DBDUMP` (stream table), `DBINFO` (row count & flash usage), and `DBCLEAR` (truncate table).

`storage/logger_service.py` can also record live Adafruit IO MQTT feeds into the same database schema while online.

---

## Web Dashboard

The repository includes a responsive HTML5 / JavaScript dashboard in [`dashboard/`](dashboard/).

```bash
python -m http.server 8000 -d dashboard
```

Open `http://localhost:8000` in your web browser and enter your Adafruit IO username and key (or toggle demo mode).

### Features:
- **Live Acceleration Chart**: Real-time $|a|$ magnitude plotted every 2.5 s with a 1.0g resting reference line.
- **Barometer Telemetry**: Cards for atmospheric pressure (hPa), relative altitude ($m$), and peak altitude differential ($\Delta h$).
- **Device Status Badge**: Shows `ONLINE` / `OFFLINE` status backed by MQTT Last Will and Testament (LWT).
- **Incident Audit Trail**: Real-time event log recording fall alerts and user dismissals with timestamps.
- **Sensor Log Table & CSV Export**: Displays historical telemetry rows in a tabular view with a one-click **Export CSV** download button (`mitifall-sensor-*.csv`).
- **Remote Buzzer Control**: Interactive buttons to manually trigger an emergency alert or silence an active buzzer from the dashboard.

---

## Adafruit IO Feeds

| Feed Name | Direction | Interval / Trigger | Description |
|-----------|:---------:|:------------------:|-------------|
| `accel-mag` | Bracelet $\rightarrow$ Cloud | Every 2.5 s | Average acceleration magnitude $\|a\|$ in $g$ ($1.0g$ = stationary) |
| `pressure` | Bracelet $\rightarrow$ Cloud | Rotated (~5 s) | Barometric pressure in hPa from BMP280 |
| `altitude` | Bracelet $\rightarrow$ Cloud | Rotated (~5 s) | Calibrated relative altitude in meters |
| `altitude-diff` | Bracelet $\rightarrow$ Cloud | Rotated (~5 s) | Peak altitude differential $\Delta h$ in meters |
| `fall-detected` | Bracelet $\rightarrow$ Cloud | On event / reset | `1` on fall confirmed; resets to `0` on dismissal or timeout |
| `fall-log` | Bracelet $\rightarrow$ Cloud | On event | Timestamped log string (e.g. `[11:02:15 WIB] FALL DETECTED (ML p=0.85)`) |
| `fall-tilt` | Bracelet $\rightarrow$ Cloud | On fall | Measured post-impact tilt angle in degrees |
| `device-status` | Bracelet $\leftrightarrow$ Cloud | Connect / LWT | `ONLINE` when bracelet connects; `OFFLINE` via broker Last Will |
| `buzzer-command` | Cloud $\rightarrow$ Bracelet | Interactive | `1` triggers alarm buzzer remotely; `0` silences buzzer |

The Adafruit IO free tier allows 10 feeds and 30 data points per minute. The firmware throttles telemetries to ~24 requests per minute to stay safely within quotas.

---

## Troubleshooting

| Problem | Root Cause | Solution |
|---------|------------|----------|
| Wi-Fi `reason 201` | SSID not found | ESP32-C3 requires a **2.4 GHz** Wi-Fi network. Check SSID spelling and router settings. |
| MQTT `connect failed, state=5` | Authentication failure | Verify `AIO_USERNAME` and `AIO_KEY` in `include/secrets.h`. |
| Motor always on or never on | Inverted transistor polarity | Flip `kVibrationActiveHigh` in `include/board_config.h` (set `true` or `false`). |
| Sensor `FATAL: no IMU` | I²C wiring / address issue | Check SDA (GPIO8) and SCL (GPIO9). Ensure MPU-6050 AD0 is wired to GND (address `0x68`). Flash `scanner` environment to verify. |
| Model self-test failed on boot | Float mismatch or outdated header | Re-export the model with `python ml_pipeline/export_model.py` and re-upload the firmware. |
| `pio device monitor` cannot be piped | TTY control characters | Use `python deep_learning/infer_stream.py --port /dev/ttyACM0` for piping or programmatic reads. |

---

## License

MIT License. Dataset: UMAFall (Casilari et al., 2017).
