#pragma once

#include <Print.h>

#include <cstdint>

namespace DeviceLog {

struct Row {
    bool hasAccel = false;
    float accelMag = 0.0f;
    bool hasPressure = false;
    float pressureHpa = 0.0f;
    bool hasAltitude = false;
    float altitudeM = 0.0f;
    bool alarm = false;
    const char *event = "window";
};

bool begin(Print *log);
bool ready();
void add(const Row &row);
void flush();
std::uint32_t rowCount();
void dump(Print &out);
void clear();
void info(Print &out);

}
