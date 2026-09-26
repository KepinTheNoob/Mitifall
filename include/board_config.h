// =============================================================================
//  board_config.h - pin map, timing and unit conventions shared by all targets
// =============================================================================
//  Target: LOLIN C3 Mini (ESP32-C3, RV32IMC - no hardware FPU, all float math
//  is software emulated, which is why the hot paths use float and not double).
// =============================================================================

#pragma once

#include <cstdint>

namespace BoardConfig {

// ---------------------------------------------------------------------------
// I2C bus
// ---------------------------------------------------------------------------
// LOLIN C3 Mini variant defaults are SDA = 8, SCL = 10. Override here if your
// wiring differs; scanner_main can sweep candidate pairs to find the real pins.
constexpr int kI2cSdaPin = 8;
constexpr int kI2cSclPin = 9;
constexpr std::uint32_t kI2cFrequencyHz = 400000;

// Slower clock used while probing, for reliability on long jumper wires.
constexpr std::uint32_t kI2cProbeFrequencyHz = 100000;

// Device addresses.
constexpr std::uint8_t kMpu6050AddrPrimary = 0x68;    // AD0 low
constexpr std::uint8_t kMpu6050AddrSecondary = 0x69;  // AD0 high

// Three magnetometer parts ship on boards sold as "HMC5883L", each with its own
// address AND its own register map. All three are probed, in this order.
constexpr std::uint8_t kHmc5883Addr = 0x1E;   // genuine Honeywell HMC5883L
constexpr std::uint8_t kQmc5883lAddr = 0x0D;  // QST QMC5883L clone
constexpr std::uint8_t kQmc5883pAddr = 0x2C;  // QST QMC5883P, the current part

// ---------------------------------------------------------------------------
// Actuators
// ---------------------------------------------------------------------------
constexpr int kBuzzerPin = 3;     // active buzzer: drive HIGH to sound
constexpr int kVibrationPin = 4;  // vibration motor driver input

// Set to false if your buzzer/motor module is active-low.
constexpr bool kActuatorsActiveHigh = true;

// ---------------------------------------------------------------------------
// Acquisition timing
// ---------------------------------------------------------------------------
constexpr std::uint32_t kSampleRateHz = 50;
constexpr std::uint32_t kSamplePeriodMs = 1000 / kSampleRateHz;  // 20 ms

// 2.0 s window with 50 % overlap, matching the Python pipeline geometry.
constexpr std::size_t kWindowSamples = 2 * kSampleRateHz;  // 100
constexpr std::size_t kHopSamples = kWindowSamples / 2;    // 50 -> 1.0 s hop

// ---------------------------------------------------------------------------
// Unit conventions - MUST match the training data
// ---------------------------------------------------------------------------
// UMAFall wrist units: accelerometer in g, gyroscope in deg/s, magnetometer in
// the capture's raw magnetometer scale. The Adafruit drivers report m/s^2 and
// rad/s, so sensor_hub converts on read.
constexpr float kStandardGravity = 9.80665f;      // m/s^2 per g
constexpr float kRadToDeg = 57.2957795131f;       // deg per rad

// The training capture saturates at these limits (verified against the dataset:
// per-axis |a| tops out at 8.00 g and |w| at 256 deg/s). The MPU6050 is
// configured to the matching ranges and readings are clamped, so an on-device
// impact cannot land outside the range the forest was trained on.
constexpr float kAccelLimitG = 8.0f;
constexpr float kGyroLimitDps = 256.0f;

}  // namespace BoardConfig
