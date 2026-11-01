#pragma once

#include <Arduino.h>
#include <Wire.h>

#include <cstdint>

namespace I2cDiagnostics {

constexpr std::size_t kMaxDevices = 16;

enum class DeviceKind : std::uint8_t {
    Unknown,
    Mpu6050,
    MpuVariant,
    Hmc5883L,
    Qmc5883L,
    Qmc5883P,
    Bmp280,
    Bme280,
    Bmp180,
};

const char *deviceKindName(DeviceKind kind);

struct DeviceInfo {
    std::uint8_t address = 0;
    DeviceKind kind = DeviceKind::Unknown;
    std::uint8_t idByte = 0;
};

struct ScanResult {
    DeviceInfo devices[kMaxDevices];
    std::size_t count = 0;

    bool contains(std::uint8_t address) const;
    const DeviceInfo *find(std::uint8_t address) const;
};

bool probeAddress(TwoWire &bus, std::uint8_t address);

bool readRegister(TwoWire &bus, std::uint8_t address, std::uint8_t reg, std::uint8_t &value);

ScanResult scanBus(TwoWire &bus);

DeviceKind identify(TwoWire &bus, std::uint8_t address, std::uint8_t &idByte);

void printScanResult(Stream &out, const ScanResult &result);

struct PinCandidate {
    int sda;
    int scl;
    std::size_t deviceCount;
};

extern const int kCandidatePins[];
extern const std::size_t kCandidatePinCount;

bool isPinUsable(int pin);

std::size_t sweepPins(Stream &out, PinCandidate *results, std::size_t maxResults,
                      bool stopAfterFirstHit = false);

}
