#include "sensor_hub.h"

#include <Adafruit_HMC5883_U.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#include "board_config.h"

namespace SensorHub {
namespace {

using namespace BoardConfig;

Adafruit_MPU6050 gMpu;
Adafruit_HMC5883_Unified gHmc(12345);
Status gStatus;

// --------------------------------------------------------------------------
// Minimal QMC5883L driver (the Adafruit HMC5883 library cannot drive it)
// --------------------------------------------------------------------------
namespace qmc {

constexpr std::uint8_t kRegData = 0x00;      // X_LSB, X_MSB, Y_LSB, Y_MSB, Z_LSB, Z_MSB
constexpr std::uint8_t kRegStatus = 0x06;    // bit0 DRDY
constexpr std::uint8_t kRegControl1 = 0x09;
constexpr std::uint8_t kRegControl2 = 0x0A;
constexpr std::uint8_t kRegSetReset = 0x0B;

// MODE = continuous (0x01), ODR = 100 Hz (0x08), RNG = 8 G (0x10), OSR = 512.
constexpr std::uint8_t kControl1Value = 0x01 | 0x08 | 0x10;

// 8 G range gives 3000 LSB/Gauss; 1 Gauss = 100 uT -> uT = raw / 30.
constexpr float kLsbToMicroTesla = 1.0f / 30.0f;

bool writeRegister(std::uint8_t reg, std::uint8_t value) {
    Wire.beginTransmission(kQmc5883Addr);
    Wire.write(reg);
    Wire.write(value);
    return Wire.endTransmission() == 0;
}

bool begin() {
    if (!I2cDiagnostics::probeAddress(Wire, kQmc5883Addr)) {
        return false;
    }
    // Soft reset, then the recommended SET/RESET period, then continuous mode.
    writeRegister(kRegControl2, 0x80);
    delay(10);
    if (!writeRegister(kRegSetReset, 0x01)) {
        return false;
    }
    return writeRegister(kRegControl1, kControl1Value);
}

bool read(float &mx, float &my, float &mz) {
    Wire.beginTransmission(kQmc5883Addr);
    Wire.write(kRegData);
    if (Wire.endTransmission(false) != 0) {
        return false;
    }
    if (Wire.requestFrom(static_cast<int>(kQmc5883Addr), 6) != 6) {
        return false;
    }
    // Little-endian, signed 16-bit per axis.
    const std::int16_t rawX = static_cast<std::int16_t>(Wire.read() | (Wire.read() << 8));
    const std::int16_t rawY = static_cast<std::int16_t>(Wire.read() | (Wire.read() << 8));
    const std::int16_t rawZ = static_cast<std::int16_t>(Wire.read() | (Wire.read() << 8));

    mx = static_cast<float>(rawX) * kLsbToMicroTesla;
    my = static_cast<float>(rawY) * kLsbToMicroTesla;
    mz = static_cast<float>(rawZ) * kLsbToMicroTesla;
    return true;
}

}  // namespace qmc

float clampf(float value, float limit) {
    if (value > limit) return limit;
    if (value < -limit) return -limit;
    return value;
}

bool startBus(int sda, int scl) {
    Wire.end();
    // setPins() before begin() matters: the Adafruit drivers call the no-argument
    // Wire.begin() internally, which would otherwise fall back to the variant
    // default pins and silently undo a custom wiring.
    Wire.setPins(sda, scl);
    if (!Wire.begin(sda, scl, kI2cFrequencyHz)) {
        return false;
    }
    Wire.setTimeOut(25);
    gStatus.sdaPin = sda;
    gStatus.sclPin = scl;
    return true;
}

bool beginImu(Stream &log) {
    const std::uint8_t addresses[] = {kMpu6050AddrPrimary, kMpu6050AddrSecondary};
    for (std::uint8_t address : addresses) {
        if (!I2cDiagnostics::probeAddress(Wire, address)) {
            continue;
        }
        if (!gMpu.begin(address, &Wire)) {
            continue;
        }
        // Ranges chosen to reproduce the training capture's saturation limits
        // (+/-8 g, +/-250 deg/s) and a 21 Hz DLPF, below the 25 Hz Nyquist
        // limit of the 50 Hz acquisition loop.
        gMpu.setAccelerometerRange(MPU6050_RANGE_8_G);
        gMpu.setGyroRange(MPU6050_RANGE_250_DEG);
        gMpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

        gStatus.imuAddress = address;
        gStatus.imuReady = true;
        log.printf("  MPU-6050 ready at 0x%02X (+/-8 g, +/-250 deg/s, 21 Hz DLPF)\n", address);
        return true;
    }
    log.println("  MPU-6050 NOT found at 0x68 or 0x69");
    return false;
}

bool beginMag(Stream &log) {
    if (I2cDiagnostics::probeAddress(Wire, kHmc5883Addr)) {
        std::uint8_t id = 0;
        const auto kind = I2cDiagnostics::identify(Wire, kHmc5883Addr, id);
        if (kind == I2cDiagnostics::DeviceKind::Hmc5883L && gHmc.begin()) {
            gStatus.magKind = MagKind::Hmc5883L;
            gStatus.magReady = true;
            log.println("  HMC5883L ready at 0x1E");
            return true;
        }
        log.printf("  device at 0x1E did not identify as HMC5883L (ID 0x%02X)\n", id);
    }

    if (qmc::begin()) {
        gStatus.magKind = MagKind::Qmc5883L;
        gStatus.magReady = true;
        log.println("  QMC5883L clone ready at 0x0D (8 G range, 100 Hz ODR)");
        return true;
    }

    log.println("  no magnetometer found at 0x1E or 0x0D");
    return false;
}

}  // namespace

const char *magKindName(MagKind kind) {
    switch (kind) {
        case MagKind::Hmc5883L:
            return "HMC5883L";
        case MagKind::Qmc5883L:
            return "QMC5883L";
        default:
            return "none";
    }
}

Status begin(Stream &log, bool autoDetectPins) {
    gStatus = Status{};

    log.printf("I2C: trying SDA=%d SCL=%d\n", kI2cSdaPin, kI2cSclPin);
    bool busUp = startBus(kI2cSdaPin, kI2cSclPin);

    if (busUp) {
        const auto scan = I2cDiagnostics::scanBus(Wire);
        if (scan.count == 0 && autoDetectPins) {
            log.println("I2C: nothing on the configured pins, sweeping candidates");
            I2cDiagnostics::PinCandidate hit{};
            if (I2cDiagnostics::sweepPins(log, &hit, 1, true) == 1) {
                log.printf("I2C: adopting SDA=%d SCL=%d\n", hit.sda, hit.scl);
                busUp = startBus(hit.sda, hit.scl);
            }
        } else {
            I2cDiagnostics::printScanResult(log, scan);
        }
    }

    if (!busUp) {
        log.println("I2C: bus initialisation failed");
        return gStatus;
    }

    beginImu(log);
    beginMag(log);
    return gStatus;
}

const Status &status() { return gStatus; }

bool read(Sample9 &sample) {
    sample.timestampMs = millis();

    if (!gStatus.imuReady) {
        return false;
    }

    sensors_event_t accel, gyro, temp;
    if (!gMpu.getEvent(&accel, &gyro, &temp)) {
        return false;
    }

    sample.v[kAx] = clampf(accel.acceleration.x / kStandardGravity, kAccelLimitG);
    sample.v[kAy] = clampf(accel.acceleration.y / kStandardGravity, kAccelLimitG);
    sample.v[kAz] = clampf(accel.acceleration.z / kStandardGravity, kAccelLimitG);

    sample.v[kGx] = clampf(gyro.gyro.x * kRadToDeg, kGyroLimitDps);
    sample.v[kGy] = clampf(gyro.gyro.y * kRadToDeg, kGyroLimitDps);
    sample.v[kGz] = clampf(gyro.gyro.z * kRadToDeg, kGyroLimitDps);

    if (gStatus.magKind == MagKind::Hmc5883L) {
        sensors_event_t magEvent;
        if (gHmc.getEvent(&magEvent)) {
            sample.v[kMx] = magEvent.magnetic.x;
            sample.v[kMy] = magEvent.magnetic.y;
            sample.v[kMz] = magEvent.magnetic.z;
        }
    } else if (gStatus.magKind == MagKind::Qmc5883L) {
        float mx = 0, my = 0, mz = 0;
        if (qmc::read(mx, my, mz)) {
            sample.v[kMx] = mx;
            sample.v[kMy] = my;
            sample.v[kMz] = mz;
        }
    }

    return true;
}

}  // namespace SensorHub
