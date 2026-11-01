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

bool readAxes6(std::uint8_t address, std::uint8_t firstDataReg, std::int16_t &x,
               std::int16_t &y, std::int16_t &z) {
    Wire.beginTransmission(address);
    Wire.write(firstDataReg);
    if (Wire.endTransmission(false) != 0) {
        return false;
    }
    if (Wire.requestFrom(static_cast<int>(address), 6) != 6) {
        return false;
    }

    const int xl = Wire.read(), xh = Wire.read();
    const int yl = Wire.read(), yh = Wire.read();
    const int zl = Wire.read(), zh = Wire.read();
    x = static_cast<std::int16_t>((xh << 8) | xl);
    y = static_cast<std::int16_t>((yh << 8) | yl);
    z = static_cast<std::int16_t>((zh << 8) | zl);
    return true;
}

bool writeRegister(std::uint8_t address, std::uint8_t reg, std::uint8_t value) {
    Wire.beginTransmission(address);
    Wire.write(reg);
    Wire.write(value);
    return Wire.endTransmission() == 0;
}

namespace qmc5883l {

constexpr std::uint8_t kRegData = 0x00;
constexpr std::uint8_t kRegControl1 = 0x09;
constexpr std::uint8_t kRegControl2 = 0x0A;
constexpr std::uint8_t kRegSetReset = 0x0B;

constexpr std::uint8_t kControl1Value = 0x01 | 0x08 | 0x10;

constexpr float kLsbToMicroTesla = 1.0f / 30.0f;

bool begin() {
    if (!I2cDiagnostics::probeAddress(Wire, kQmc5883lAddr)) {
        return false;
    }

    writeRegister(kQmc5883lAddr, kRegControl2, 0x80);
    delay(10);
    if (!writeRegister(kQmc5883lAddr, kRegSetReset, 0x01)) {
        return false;
    }
    return writeRegister(kQmc5883lAddr, kRegControl1, kControl1Value);
}

bool read(float &mx, float &my, float &mz) {
    std::int16_t rawX = 0, rawY = 0, rawZ = 0;
    if (!readAxes6(kQmc5883lAddr, kRegData, rawX, rawY, rawZ)) {
        return false;
    }
    mx = static_cast<float>(rawX) * kLsbToMicroTesla;
    my = static_cast<float>(rawY) * kLsbToMicroTesla;
    mz = static_cast<float>(rawZ) * kLsbToMicroTesla;
    return true;
}

}

namespace qmc5883p {

constexpr std::uint8_t kRegChipId = 0x00;
constexpr std::uint8_t kRegData = 0x01;
constexpr std::uint8_t kRegStatus = 0x09;
constexpr std::uint8_t kRegControl1 = 0x0A;
constexpr std::uint8_t kRegControl2 = 0x0B;

constexpr std::uint8_t kChipId = 0x80;

constexpr std::uint8_t kControl1Value = 0x03 | (0x02 << 2);

constexpr std::uint8_t kControl2Value = (0x02 << 2);

constexpr float kLsbToMicroTesla = 1.0f / 37.5f;

bool begin() {
    if (!I2cDiagnostics::probeAddress(Wire, kQmc5883pAddr)) {
        return false;
    }

    std::uint8_t id = 0;
    if (!I2cDiagnostics::readRegister(Wire, kQmc5883pAddr, kRegChipId, id) || id != kChipId) {
        return false;
    }

    writeRegister(kQmc5883pAddr, kRegControl2, 0x80);
    delay(50);

    if (!writeRegister(kQmc5883pAddr, kRegControl2, kControl2Value)) {
        return false;
    }
    return writeRegister(kQmc5883pAddr, kRegControl1, kControl1Value);
}

bool read(float &mx, float &my, float &mz) {
    std::int16_t rawX = 0, rawY = 0, rawZ = 0;
    if (!readAxes6(kQmc5883pAddr, kRegData, rawX, rawY, rawZ)) {
        return false;
    }
    mx = static_cast<float>(rawX) * kLsbToMicroTesla;
    my = static_cast<float>(rawY) * kLsbToMicroTesla;
    mz = static_cast<float>(rawZ) * kLsbToMicroTesla;
    return true;
}

}

float clampf(float value, float limit) {
    if (value > limit) return limit;
    if (value < -limit) return -limit;
    return value;
}

bool startBus(int sda, int scl) {
    Wire.end();

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

    if (qmc5883l::begin()) {
        gStatus.magKind = MagKind::Qmc5883L;
        gStatus.magReady = true;
        log.println("  QMC5883L ready at 0x0D (8 G range, 100 Hz ODR)");
        return true;
    }

    if (qmc5883p::begin()) {
        gStatus.magKind = MagKind::Qmc5883P;
        gStatus.magReady = true;
        log.println("  QMC5883P ready at 0x2C (8 G range, 100 Hz ODR, 3750 LSB/G)");
        return true;
    }

    log.println("  no magnetometer found at 0x1E, 0x0D or 0x2C");
    return false;
}

}

const char *magKindName(MagKind kind) {
    switch (kind) {
        case MagKind::Hmc5883L:
            return "HMC5883L";
        case MagKind::Qmc5883L:
            return "QMC5883L";
        case MagKind::Qmc5883P:
            return "QMC5883P";
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
        if (qmc5883l::read(mx, my, mz)) {
            sample.v[kMx] = mx;
            sample.v[kMy] = my;
            sample.v[kMz] = mz;
        }
    } else if (gStatus.magKind == MagKind::Qmc5883P) {
        float mx = 0, my = 0, mz = 0;
        if (qmc5883p::read(mx, my, mz)) {
            sample.v[kMx] = mx;
            sample.v[kMy] = my;
            sample.v[kMz] = mz;
        }
    }

    return true;
}

}
