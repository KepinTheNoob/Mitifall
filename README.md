# Mitifall

Wrist-worn fall detector on an ESP32-C3. The bracelet reads a 9-DoF IMU (and an
optional BMP280 barometer) at 50 Hz, detects falls, sounds a buzzer and a
vibration motor, and reports to Adafruit IO.

Detailed documentation (Indonesian):
[machine learning path](docs/machine_learning.md) ·
[deep learning path](docs/deep_learning.md).

## Hardware

| Part | Connection |
|------|------------|
| LOLIN C3 Mini (ESP32-C3) | — |
| MPU-6050 (accelerometer + gyroscope) | I²C, SDA GPIO8, SCL GPIO9 |
| HMC5883L / QMC5883L / QMC5883P (magnetometer) | same I²C bus |
| BMP280 / BME280 (optional) | same I²C bus, 0x76 or 0x77 |
| Active buzzer | GPIO2 |
| Vibration motor module (via transistor) | GPIO1 |
| Dismiss button (to GND) | GPIO3 |

Pins and actuator polarity are set in `include/board_config.h`.

## Firmware environments

| Environment | What it does | Cloud |
|-------------|--------------|-------|
| `scanner` | I²C scan, sensor check, actuator test | — |
| `raw_stream` | Threshold-based fall detection on the bracelet, telemetry | Adafruit IO (MQTT) |
| `dl_stream` | Streams CSV over USB; falls are detected **only** by the deep learning model on the host | Adafruit IO (MQTT) |
| `ml_inference` | Random Forest running on the bracelet, offline | — |

```bash
cp include/secrets.example.h include/secrets.h   # WiFi + Adafruit IO credentials
pio run -e raw_stream -t upload -t monitor
```

The ESP32-C3 only connects to **2.4 GHz** WiFi.

## Python setup

```bash
python -m venv venv
source venv/bin/activate            # fish: source venv/bin/activate.fish
pip install -r requirements.txt -r deep_learning/requirements.txt -r storage/requirements.txt
```

## Machine learning (Random Forest)

Trained on the UMAFall wrist recordings (19 subjects), 2 s windows, 53 features,
validated with Leave-One-Subject-Out.

```bash
python dataset/process_dataset.py
python ml_pipeline/run_pipeline.py      # features + LOSO benchmark
python ml_pipeline/export_model.py      # -> include/RandomForest.h
pio run -e ml_inference -t upload
```

## Deep learning (CNN-BiLSTM)

Runs on the laptop, not on the bracelet. The bracelet sends raw samples over USB
and raises the alarm when the model reports a fall.

```bash
pio run -e dl_stream -t upload
python deep_learning/infer_stream.py --port /dev/ttyACM0
```

Useful options: `--threshold 0.7`, `--quiet`, `--int8`.
`python deep_learning/infer_stream.py --selftest` checks the streaming path
against the batch evaluation. Retraining:

```bash
python deep_learning/train.py --arch cnn_bilstm --with-gyro
python deep_learning/export_onnx.py --int8
```

## Results (LOSO, 9 683 windows)

| Model | Accuracy | F1 macro | Sensitivity | Specificity | ROC-AUC |
|-------|:--------:|:--------:|:-----------:|:-----------:|:-------:|
| Random Forest | 0.877 | 0.846 | 0.777 | 0.915 | 0.906 |
| CNN-BiLSTM | 0.880 | 0.851 | 0.783 | 0.918 | 0.928 |

Full report: `deep_learning/outputs/evaluation/report.md`.

## Telemetry logging

The bracelet (`raw_stream`, `dl_stream`) keeps its own SQLite log on internal
flash (LittleFS, ~1.9 MB): one row every 2.5 s with the timestamp, mean |a|,
pressure, altitude and alarm state, plus a row on every fall and dismissal.
It works without WiFi; the oldest rows are removed when the storage is nearly
full. Timestamps are UTC once NTP has synced.

Copy it to the laptop over USB (rows already copied are skipped):

```bash
python storage/pull_device.py --port /dev/ttyACM0            # add --clear to empty the bracelet
python storage/export_csv.py                                 # -> storage/exports/telemetry_<UTC>.csv
```

`storage/logger_service.py` can also record the Adafruit IO feeds into the same
database while the laptop is online.

`motion_status` stays empty: the current firmware does not publish it.

## Dashboard

```bash
python -m http.server 8000 -d dashboard
```

Open http://localhost:8000 and log in with your Adafruit IO username and key,
or use demo mode.

## Adafruit IO feeds

`accel-mag`, `pressure`, `altitude`, `altitude-diff`, `fall-detected`,
`fall-tilt`, `fall-log`, `device-status`, `buzzer-command`.
The free plan allows 10 feeds and 30 data points per minute.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| WiFi `reason 201` | SSID not found: use a 2.4 GHz network, check the spelling |
| `mqtt connect failed, state=5` | Wrong Adafruit IO username or key |
| Motor always on or never on | Flip `kVibrationActiveHigh` in `include/board_config.h`; check wiring and the module's supply |
| `pio device monitor` cannot be piped | Use `infer_stream.py --port` instead |

## License

MIT. Dataset: UMAFall (Casilari et al., 2017).
