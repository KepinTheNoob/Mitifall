#pragma once

#include <Arduino.h>
#include <Wire.h>

#include <cstdint>

#include "baro_math.h"

namespace Barometer {

constexpr std::uint32_t kReadPeriodMs = 40;

enum class Kind : std::uint8_t { None, Bmp280, Bme280 };

struct Status {
    bool present = false;
    bool ready = false;
    Kind kind = Kind::None;
    std::uint8_t address = 0;
    std::uint8_t chipId = 0;
    bool calibrationValid = false;
    float firstPressureHpa = 0.0f;
    float firstTemperatureC = 0.0f;
};

Status begin(Stream &log, TwoWire &bus = Wire);

const Status &status();
bool ready();

bool service();

const BaroMath::AltitudeTracker &tracker();

float temperatureC();
float pressureHpa();

void rezero();

std::uint32_t readFailures();
std::uint32_t lastReadMicros();

const char *kindName(Kind kind);

}
