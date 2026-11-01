#include "i2c_diagnostics.h"

#include "board_config.h"

namespace I2cDiagnostics {
namespace {

constexpr std::uint8_t kRegWhoAmI = 0x75;
constexpr std::uint8_t kWhoAmIMpu6050 = 0x68;

constexpr std::uint8_t kRegHmcIdA = 0x0A;
constexpr std::uint8_t kHmcIdA = 0x48;
constexpr std::uint8_t kHmcIdB = 0x34;
constexpr std::uint8_t kHmcIdC = 0x33;

constexpr std::uint8_t kRegQmcLChipId = 0x0D;
constexpr std::uint8_t kQmcLChipId = 0xFF;

constexpr std::uint8_t kRegQmcPChipId = 0x00;
constexpr std::uint8_t kQmcPChipId = 0x80;

constexpr std::uint8_t kRegBoschChipId = 0xD0;

bool isMpuVariant(std::uint8_t whoAmI) {

    return whoAmI == 0x70 || whoAmI == 0x71 || whoAmI == 0x73 || whoAmI == 0x7C;
}

}

const char *deviceKindName(DeviceKind kind) {
    switch (kind) {
        case DeviceKind::Mpu6050:
            return "MPU-6050";
        case DeviceKind::MpuVariant:
            return "MPU-6500/9250 variant";
        case DeviceKind::Hmc5883L:
            return "HMC5883L (genuine)";
        case DeviceKind::Qmc5883L:
            return "QMC5883L (QST clone)";
        case DeviceKind::Qmc5883P:
            return "QMC5883P (QST)";
        case DeviceKind::Bmp280:
            return "BMP280 (barometer)";
        case DeviceKind::Bme280:
            return "BME280 (barometer)";
        case DeviceKind::Bmp180:
            return "BMP180 (unsupported)";
        default:
            return "unknown device";
    }
}

bool ScanResult::contains(std::uint8_t address) const { return find(address) != nullptr; }

const DeviceInfo *ScanResult::find(std::uint8_t address) const {
    for (std::size_t i = 0; i < count; ++i) {
        if (devices[i].address == address) {
            return &devices[i];
        }
    }
    return nullptr;
}

bool probeAddress(TwoWire &bus, std::uint8_t address) {
    bus.beginTransmission(address);
    return bus.endTransmission() == 0;
}

bool readRegister(TwoWire &bus, std::uint8_t address, std::uint8_t reg, std::uint8_t &value) {
    bus.beginTransmission(address);
    bus.write(reg);
    if (bus.endTransmission(false) != 0) {
        return false;
    }
    if (bus.requestFrom(static_cast<int>(address), 1) != 1) {
        return false;
    }
    value = static_cast<std::uint8_t>(bus.read());
    return true;
}

DeviceKind identify(TwoWire &bus, std::uint8_t address, std::uint8_t &idByte) {
    idByte = 0;

    if (address == BoardConfig::kMpu6050AddrPrimary ||
        address == BoardConfig::kMpu6050AddrSecondary) {
        if (readRegister(bus, address, kRegWhoAmI, idByte)) {
            if (idByte == kWhoAmIMpu6050) {
                return DeviceKind::Mpu6050;
            }
            if (isMpuVariant(idByte)) {
                return DeviceKind::MpuVariant;
            }
        }
        return DeviceKind::Unknown;
    }

    if (address == BoardConfig::kHmc5883Addr) {
        std::uint8_t a = 0, b = 0, c = 0;
        const bool ok = readRegister(bus, address, kRegHmcIdA, a) &&
                        readRegister(bus, address, kRegHmcIdA + 1, b) &&
                        readRegister(bus, address, kRegHmcIdA + 2, c);
        idByte = a;
        if (ok && a == kHmcIdA && b == kHmcIdB && c == kHmcIdC) {
            return DeviceKind::Hmc5883L;
        }
        return DeviceKind::Unknown;
    }

    if (address == BoardConfig::kQmc5883lAddr) {
        readRegister(bus, address, kRegQmcLChipId, idByte);

        (void)kQmcLChipId;
        return DeviceKind::Qmc5883L;
    }

    if (address == BoardConfig::kQmc5883pAddr) {
        if (readRegister(bus, address, kRegQmcPChipId, idByte) && idByte == kQmcPChipId) {
            return DeviceKind::Qmc5883P;
        }
        return DeviceKind::Unknown;
    }

    if (address == BoardConfig::kBmp280AddrPrimary ||
        address == BoardConfig::kBmp280AddrSecondary) {
        if (!readRegister(bus, address, kRegBoschChipId, idByte)) {
            return DeviceKind::Unknown;
        }
        switch (idByte) {
            case 0x58:
            case 0x56:
            case 0x57:
                return DeviceKind::Bmp280;
            case 0x60:
                return DeviceKind::Bme280;
            case 0x55:
                return DeviceKind::Bmp180;
            default:
                return DeviceKind::Unknown;
        }
    }

    return DeviceKind::Unknown;
}

ScanResult scanBus(TwoWire &bus) {
    ScanResult result;
    for (std::uint8_t address = 0x01; address <= 0x7F; ++address) {
        if (!probeAddress(bus, address)) {
            continue;
        }
        if (result.count >= kMaxDevices) {
            break;
        }
        DeviceInfo &info = result.devices[result.count++];
        info.address = address;
        info.kind = identify(bus, address, info.idByte);
    }
    return result;
}

void printScanResult(Stream &out, const ScanResult &result) {
    if (result.count == 0) {
        out.println("  no devices responded");
        return;
    }
    for (std::size_t i = 0; i < result.count; ++i) {
        const DeviceInfo &info = result.devices[i];
        out.printf("  0x%02X  %-24s", info.address, deviceKindName(info.kind));
        if (info.idByte != 0) {
            out.printf(" (ID 0x%02X)", info.idByte);
        }
        out.println();
    }
}

const int kCandidatePins[] = {0, 1, 5, 6, 7, 8, 10, 2, 9, 20, 21};
const std::size_t kCandidatePinCount = sizeof(kCandidatePins) / sizeof(kCandidatePins[0]);

bool isPinUsable(int pin) {
    if (pin < 0 || pin > 21) {
        return false;
    }
    if (pin >= 11 && pin <= 17) {
        return false;
    }
    if (pin == 18 || pin == 19) {
        return false;
    }
    if (pin == BoardConfig::kBuzzerPin || pin == BoardConfig::kVibrationPin) {
        return false;
    }
    if (pin == BoardConfig::kDismissButtonPin) {
        return false;
    }
    return true;
}

std::size_t sweepPins(Stream &out, PinCandidate *results, std::size_t maxResults,
                      bool stopAfterFirstHit) {
    std::size_t found = 0;

    out.println("Sweeping candidate SDA/SCL pairs (excluding GPIO11-17, 18-19 and actuators)");

    for (std::size_t i = 0; i < kCandidatePinCount; ++i) {
        for (std::size_t j = 0; j < kCandidatePinCount; ++j) {
            if (i == j) {
                continue;
            }
            const int sda = kCandidatePins[i];
            const int scl = kCandidatePins[j];
            if (!isPinUsable(sda) || !isPinUsable(scl)) {
                continue;
            }

            Wire.end();

            pinMode(sda, INPUT_PULLUP);
            pinMode(scl, INPUT_PULLUP);
            if (!Wire.begin(sda, scl, BoardConfig::kI2cProbeFrequencyHz)) {
                continue;
            }
            Wire.setTimeOut(20);

            std::size_t devices = 0;
            for (std::uint8_t address = 0x01; address <= 0x7F; ++address) {
                if (probeAddress(Wire, address)) {
                    ++devices;
                }
            }

            if (devices > 0 && devices < 0x7F) {
                out.printf("  SDA=%2d SCL=%2d -> %u device(s)\n", sda, scl,
                           static_cast<unsigned>(devices));
                if (found < maxResults) {
                    results[found++] = PinCandidate{sda, scl, devices};
                }
                if (stopAfterFirstHit && found > 0) {
                    Wire.end();
                    return found;
                }
            }
        }
    }

    Wire.end();
    if (found == 0) {
        out.println("  no pin pair produced a response - check power, ground and pull-ups");
    }
    return found;
}

}
