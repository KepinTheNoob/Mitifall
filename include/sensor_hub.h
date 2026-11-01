#pragma once

#include <Arduino.h>
#include <Wire.h>

#include <cstdint>

#include "i2c_diagnostics.h"

namespace SensorHub {

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

enum class MagKind : std::uint8_t { None, Hmc5883L, Qmc5883L, Qmc5883P };

struct Status {
    bool imuReady = false;
    bool magReady = false;
    std::uint8_t imuAddress = 0;
    MagKind magKind = MagKind::None;
    int sdaPin = -1;
    int sclPin = -1;
};

Status begin(Stream &log, bool autoDetectPins = true);

const Status &status();

bool read(Sample9 &sample);

const char *magKindName(MagKind kind);

}
