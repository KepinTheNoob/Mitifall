#pragma once

#include <cstdint>

namespace BoardConfig {

constexpr int kI2cSdaPin = 8;
constexpr int kI2cSclPin = 9;
constexpr std::uint32_t kI2cFrequencyHz = 400000;

constexpr std::uint32_t kI2cProbeFrequencyHz = 100000;

constexpr std::uint8_t kMpu6050AddrPrimary = 0x68;
constexpr std::uint8_t kMpu6050AddrSecondary = 0x69;

constexpr std::uint8_t kHmc5883Addr = 0x1E;
constexpr std::uint8_t kQmc5883lAddr = 0x0D;
constexpr std::uint8_t kQmc5883pAddr = 0x2C;

constexpr std::uint8_t kBmp280AddrPrimary = 0x76;
constexpr std::uint8_t kBmp280AddrSecondary = 0x77;

constexpr int kBuzzerPin = 2;
constexpr int kVibrationPin = 1;

constexpr bool kBuzzerActiveHigh = true;

constexpr bool kVibrationActiveHigh = true;

constexpr int kDismissButtonPin = 3;
constexpr bool kButtonActiveLow = true;
constexpr std::uint32_t kButtonDebounceMs = 40;

constexpr std::uint32_t kSampleRateHz = 50;
constexpr std::uint32_t kSamplePeriodMs = 1000 / kSampleRateHz;

constexpr std::size_t kWindowSamples = 2 * kSampleRateHz;
constexpr std::size_t kHopSamples = kWindowSamples / 2;

constexpr float kStandardGravity = 9.80665f;
constexpr float kRadToDeg = 57.2957795131f;

constexpr float kAccelLimitG = 8.0f;
constexpr float kGyroLimitDps = 256.0f;

}
