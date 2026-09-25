// =============================================================================
//  i2c_diagnostics.h - bus scanning, device identification and pin sweeping
// =============================================================================

#pragma once

#include <Arduino.h>
#include <Wire.h>

#include <cstdint>

namespace I2cDiagnostics {

constexpr std::size_t kMaxDevices = 16;

/// What kind of chip answered at a given address.
enum class DeviceKind : std::uint8_t {
    Unknown,
    Mpu6050,        ///< WHO_AM_I = 0x68
    MpuVariant,     ///< MPU6500/9250/6555 family - register-compatible enough
    Hmc5883L,       ///< genuine Honeywell, ID regs spell "H43"
    Qmc5883L,       ///< QST clone at 0x0D
};

const char *deviceKindName(DeviceKind kind);

struct DeviceInfo {
    std::uint8_t address = 0;
    DeviceKind kind = DeviceKind::Unknown;
    std::uint8_t idByte = 0;  ///< WHO_AM_I / chip-ID value, for diagnostics
};

struct ScanResult {
    DeviceInfo devices[kMaxDevices];
    std::size_t count = 0;

    bool contains(std::uint8_t address) const;
    const DeviceInfo *find(std::uint8_t address) const;
};

/// True if a device ACKs its address.
bool probeAddress(TwoWire &bus, std::uint8_t address);

/// Read one 8-bit register. Returns false if the device did not respond.
bool readRegister(TwoWire &bus, std::uint8_t address, std::uint8_t reg, std::uint8_t &value);

/// Scan 0x01..0x7F and identify anything recognisable.
ScanResult scanBus(TwoWire &bus);

/// Identify the chip at a known address (WHO_AM_I / ID register checks).
DeviceKind identify(TwoWire &bus, std::uint8_t address, std::uint8_t &idByte);

void printScanResult(Stream &out, const ScanResult &result);

// ---------------------------------------------------------------------------
// Pin sweeping
// ---------------------------------------------------------------------------

struct PinCandidate {
    int sda;
    int scl;
    std::size_t deviceCount;
};

/// ESP32-C3 pins safe to try for I2C.
///
/// Excluded: GPIO11-17 (internal SPI flash - driving these bricks the boot),
/// GPIO18/19 (native USB D-/D+ used by the CDC console) and the two actuator
/// pins. GPIO2/8/9 are strapping pins and GPIO20/21 are UART0 RX/TX; they work
/// for I2C but are listed last because they have boot-time side effects.
extern const int kCandidatePins[];
extern const std::size_t kCandidatePinCount;

bool isPinUsable(int pin);

/// Try every ordered SDA/SCL pair from kCandidatePins and report where devices
/// answered. Stops early once `stopAfterFirstHit` pairs have been found.
/// Leaves the bus de-initialised; call Wire.begin() again afterwards.
std::size_t sweepPins(Stream &out, PinCandidate *results, std::size_t maxResults,
                      bool stopAfterFirstHit = false);

}  // namespace I2cDiagnostics
