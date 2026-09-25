// =============================================================================
//  sensor_hub.h - 9-DoF acquisition in the units the model was trained on
// =============================================================================
//  Wraps the MPU-6050 (accel + gyro) and the magnetometer, which may be either a
//  genuine HMC5883L at 0x1E (Adafruit driver) or a QMC5883L clone at 0x0D
//  (register-level driver below, because the two parts are not register
//  compatible despite the similar name).
//
//  Output units match the UMAFall training data:
//    accelerometer  g       (Adafruit reports m/s^2 -> divided by 9.80665)
//    gyroscope      deg/s   (Adafruit reports rad/s -> multiplied by 180/pi)
//    magnetometer   uT
//
//  NOTE ON THE MAGNETOMETER: the training capture's magnetometer channel is not
//  calibrated to uT (its median |M| is ~155 against an Earth field of 25-65 uT
//  and it carries large hard-iron offsets). Magnetometer features therefore do
//  NOT transfer from the dataset to this hardware. Prefer the 6-DoF model
//  (`python ml_pipeline/export_model.py --axes acc_gyro`), which scored slightly
//  better under LOSO anyway. See the project notes for measurements.
// =============================================================================

#pragma once

#include <Arduino.h>
#include <Wire.h>

#include <cstdint>

#include "i2c_diagnostics.h"

namespace SensorHub {

/// Axis order is the pipeline's: Ax, Ay, Az, Gx, Gy, Gz, Mx, My, Mz.
enum AxisIndex : std::size_t {
    kAx = 0, kAy, kAz,
    kGx, kGy, kGz,
    kMx, kMy, kMz,
    kAxisCount
};

struct Sample9 {
    float v[kAxisCount] = {0};
    std::uint32_t timestampMs = 0;
};

enum class MagKind : std::uint8_t { None, Hmc5883L, Qmc5883L };

struct Status {
    bool imuReady = false;
    bool magReady = false;
    std::uint8_t imuAddress = 0;
    MagKind magKind = MagKind::None;
    int sdaPin = -1;
    int sclPin = -1;
};

/// Bring up the I2C bus and both sensors.
///
/// When `autoDetectPins` is set and nothing answers on the configured pins, the
/// candidate pin pairs are swept and the first working pair is adopted.
Status begin(Stream &log, bool autoDetectPins = true);

/// Latest status from the last begin() call.
const Status &status();

/// Read all nine channels. Returns false if the IMU read failed.
/// Magnetometer channels are left at 0 when no magnetometer is present.
bool read(Sample9 &sample);

const char *magKindName(MagKind kind);

}  // namespace SensorHub
