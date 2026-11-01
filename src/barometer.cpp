#include "barometer.h"

#include "board_config.h"
#include "i2c_diagnostics.h"

namespace Barometer {
namespace {

constexpr std::uint8_t kRegCalibration = 0x88;
constexpr std::uint8_t kRegChipId = 0xD0;
constexpr std::uint8_t kRegReset = 0xE0;
constexpr std::uint8_t kRegStatus = 0xF3;
constexpr std::uint8_t kRegCtrlMeas = 0xF4;
constexpr std::uint8_t kRegConfig = 0xF5;
constexpr std::uint8_t kRegData = 0xF7;

constexpr std::uint8_t kChipBmp280 = 0x58;
constexpr std::uint8_t kChipBmp280SampleA = 0x56;
constexpr std::uint8_t kChipBmp280SampleB = 0x57;
constexpr std::uint8_t kChipBme280 = 0x60;
constexpr std::uint8_t kChipBmp180 = 0x55;

constexpr std::uint8_t kResetCommand = 0xB6;

constexpr std::uint8_t kConfigValue = (0b000 << 5) | (0b010 << 2);

constexpr std::uint8_t kCtrlMeasValue = (0b001 << 5) | (0b011 << 2) | 0b11;

TwoWire *gBus = &Wire;
Status gStatus;
BaroMath::Calibration gCal;
float gTemperatureC = 0.0f;
std::uint32_t gNextReadMs = 0;
std::uint32_t gFailures = 0;
std::uint32_t gConsecutiveFailures = 0;
std::uint32_t gLastReadUs = 0;

constexpr std::uint32_t kFailuresBeforeBackoff = 5;
constexpr std::uint32_t kBackoffMs = 1000;

bool readBytes(std::uint8_t reg, std::uint8_t *buf, std::size_t n) {
    gBus->beginTransmission(gStatus.address);
    gBus->write(reg);
    if (gBus->endTransmission(false) != 0) {
        return false;
    }
    if (gBus->requestFrom(static_cast<int>(gStatus.address), static_cast<int>(n)) !=
        static_cast<int>(n)) {
        return false;
    }
    for (std::size_t i = 0; i < n; ++i) {
        buf[i] = static_cast<std::uint8_t>(gBus->read());
    }
    return true;
}

BaroMath::AltitudeTracker &altitudeTracker() {
    static BaroMath::AltitudeTracker tracker;
    return tracker;
}

bool writeRegister(std::uint8_t reg, std::uint8_t value) {
    gBus->beginTransmission(gStatus.address);
    gBus->write(reg);
    gBus->write(value);
    return gBus->endTransmission() == 0;
}

bool readOnce(BaroMath::Reading &reading) {
    std::uint8_t raw[BaroMath::kDataBytes];
    if (!readBytes(kRegData, raw, sizeof(raw))) {
        return false;
    }
    std::int32_t adcP = 0, adcT = 0;
    BaroMath::unpackAdc(raw, adcP, adcT);
    return BaroMath::compensate(gCal, adcT, adcP, reading);
}

Kind kindFromChipId(std::uint8_t id) {
    switch (id) {
        case kChipBmp280:
        case kChipBmp280SampleA:
        case kChipBmp280SampleB:
            return Kind::Bmp280;
        case kChipBme280:
            return Kind::Bme280;
        default:
            return Kind::None;
    }
}

bool tryAddress(Stream &log, std::uint8_t address) {
    if (!I2cDiagnostics::probeAddress(*gBus, address)) {
        return false;
    }
    gStatus = Status{};
    gStatus.present = true;
    gStatus.address = address;

    std::uint8_t id = 0;
    if (!readBytes(kRegChipId, &id, 1)) {
        log.printf("  0x%02X answers but the chip-ID read failed\n", address);
        return false;
    }
    gStatus.chipId = id;
    gStatus.kind = kindFromChipId(id);
    if (gStatus.kind == Kind::None) {
        log.printf("  0x%02X chip ID 0x%02X is not a BMP280/BME280%s\n", address, id,
                   id == kChipBmp180 ? " (BMP180: different register map, unsupported)" : "");
        return false;
    }

    writeRegister(kRegReset, kResetCommand);
    delay(5);
    for (int i = 0; i < 20; ++i) {
        std::uint8_t st = 0;
        if (readBytes(kRegStatus, &st, 1) && (st & 0x01) == 0) {
            break;
        }
        delay(1);
    }

    std::uint8_t calibration[BaroMath::kCalibrationBytes];
    gStatus.calibrationValid = readBytes(kRegCalibration, calibration, sizeof(calibration)) &&
                               BaroMath::parseCalibration(calibration, gCal);
    if (!gStatus.calibrationValid) {
        log.printf("  0x%02X %s: calibration block invalid (blank or wrong chip)\n", address,
                   kindName(gStatus.kind));
        return false;
    }

    if (!writeRegister(kRegConfig, kConfigValue) || !writeRegister(kRegCtrlMeas, kCtrlMeasValue)) {
        log.printf("  0x%02X: configuration write failed\n", address);
        return false;
    }
    delay(30);

    BaroMath::Reading first;
    if (!readOnce(first)) {
        log.printf("  0x%02X %s: first reading out of range (300-1100 hPa, -40-85 C)\n", address,
                   kindName(gStatus.kind));
        return false;
    }
    gStatus.firstPressureHpa = first.pressurePa / 100.0f;
    gStatus.firstTemperatureC = first.temperatureC;
    gTemperatureC = first.temperatureC;
    gStatus.ready = true;
    return true;
}

}

const char *kindName(Kind kind) {
    switch (kind) {
        case Kind::Bmp280:
            return "BMP280";
        case Kind::Bme280:
            return "BME280";
        default:
            return "none";
    }
}

Status begin(Stream &log, TwoWire &bus) {
    gBus = &bus;
    gStatus = Status{};
    altitudeTracker().reset();
    gFailures = 0;
    gConsecutiveFailures = 0;

    const std::uint8_t addresses[] = {BoardConfig::kBmp280AddrPrimary,
                                      BoardConfig::kBmp280AddrSecondary};
    for (std::uint8_t address : addresses) {
        if (tryAddress(log, address)) {
            log.printf("  %s ready at 0x%02X (chip 0x%02X): %.2f hPa, %.1f C "
                       "(normal mode, P x4, T x1, IIR x4, read %lu Hz)\n",
                       kindName(gStatus.kind), gStatus.address, gStatus.chipId,
                       gStatus.firstPressureHpa, gStatus.firstTemperatureC,
                       static_cast<unsigned long>(1000 / kReadPeriodMs));
            gNextReadMs = millis();
            return gStatus;
        }
    }
    if (!gStatus.present) {
        log.println("  no barometer at 0x76/0x77 - pressure/altitude columns stay empty");
    }
    const Status failed = gStatus;
    gStatus = Status{};
    gStatus.present = failed.present;
    gStatus.address = failed.address;
    gStatus.chipId = failed.chipId;
    return gStatus;
}

const Status &status() { return gStatus; }

bool ready() { return gStatus.ready; }

bool service() {
    if (!gStatus.ready) {
        return false;
    }
    const std::uint32_t now = millis();
    if (static_cast<std::int32_t>(now - gNextReadMs) < 0) {
        return false;
    }

    gNextReadMs = (now - gNextReadMs > 4 * kReadPeriodMs) ? now + kReadPeriodMs
                                                          : gNextReadMs + kReadPeriodMs;

    const std::uint32_t started = micros();
    BaroMath::Reading reading;
    const bool ok = readOnce(reading);
    gLastReadUs = micros() - started;
    if (!ok) {
        ++gFailures;

        if (++gConsecutiveFailures >= kFailuresBeforeBackoff) {
            gNextReadMs = now + kBackoffMs;
        }
        return false;
    }
    gConsecutiveFailures = 0;
    gTemperatureC = reading.temperatureC;
    altitudeTracker().addSample(now, reading.pressurePa);
    return true;
}

const BaroMath::AltitudeTracker &tracker() { return altitudeTracker(); }

float temperatureC() { return gTemperatureC; }

float pressureHpa() { return altitudeTracker().pressurePa() / 100.0f; }

void rezero() { altitudeTracker().rezero(); }

std::uint32_t readFailures() { return gFailures; }

std::uint32_t lastReadMicros() { return gLastReadUs; }

}
