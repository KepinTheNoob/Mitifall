#include <Arduino.h>
#include <Wire.h>

#include <cmath>

#include "alarm_manager.h"
#include "barometer.h"
#include "board_config.h"
#include "i2c_diagnostics.h"

namespace {

using namespace BoardConfig;

constexpr bool kMotorCycleTest = true;
constexpr std::uint32_t kMotorOnMs = 1000;
constexpr std::uint32_t kMotorOffMs = 2000;

constexpr std::uint32_t kBusRescanMs = 5000;
constexpr std::uint32_t kBaroReportMs = 1000;

constexpr std::uint32_t kBaroValidationDelayMs = 3000;
constexpr float kBaroNoiseLimitM = 0.10f;
constexpr float kBaroStillSpeedLimitMps = 0.25f;

void banner() {
    Serial.println();
    Serial.println("=====================================================");
    Serial.println(" Mitifall - I2C, barometer and actuator diagnostics");
    Serial.printf(" SDA=%d SCL=%d @ %lu Hz\n", kI2cSdaPin, kI2cSclPin,
                  static_cast<unsigned long>(kI2cFrequencyHz));
    Serial.printf(" buzzer GPIO%d (active %s) | motor GPIO%d (active %s) | button GPIO%d\n",
                  kBuzzerPin, kBuzzerActiveHigh ? "HIGH" : "LOW", kVibrationPin,
                  kVibrationActiveHigh ? "HIGH" : "LOW", kDismissButtonPin);
    Serial.println("=====================================================");
}

const char *barometerVerdict(const I2cDiagnostics::ScanResult &scan) {
    for (std::uint8_t address : {kBmp280AddrPrimary, kBmp280AddrSecondary}) {
        const I2cDiagnostics::DeviceInfo *info = scan.find(address);
        if (info == nullptr) {
            continue;
        }
        switch (info->kind) {
            case I2cDiagnostics::DeviceKind::Bmp280:
            case I2cDiagnostics::DeviceKind::Bme280:
                return "FOUND";
            case I2cDiagnostics::DeviceKind::Bmp180:
                return "BMP180 found - unsupported, needs a BMP280/BME280";
            default:
                return "answering, but the chip ID is not a BMP280/BME280";
        }
    }
    return "missing";
}

void reportExpectations(const I2cDiagnostics::ScanResult &scan) {
    const bool imu = scan.contains(kMpu6050AddrPrimary) || scan.contains(kMpu6050AddrSecondary);
    const bool hmc = scan.contains(kHmc5883Addr);
    const bool qmcL = scan.contains(kQmc5883lAddr);
    const bool qmcP = scan.contains(kQmc5883pAddr);
    const int magCount = (hmc ? 1 : 0) + (qmcL ? 1 : 0) + (qmcP ? 1 : 0);

    Serial.println("Expected devices:");
    Serial.printf("  MPU-6050 (0x68/0x69) : %s\n", imu ? "FOUND" : "missing");
    Serial.printf("  HMC5883L (0x1E)      : %s\n", hmc ? "FOUND" : "missing");
    Serial.printf("  QMC5883L (0x0D)      : %s\n", qmcL ? "FOUND" : "missing");
    Serial.printf("  QMC5883P (0x2C)      : %s\n", qmcP ? "FOUND" : "missing");
    Serial.printf("  BMP280   (0x76/0x77) : %s\n", barometerVerdict(scan));

    if (magCount == 0) {
        Serial.println("  -> no magnetometer: only the 6-DoF (acc_gyro) model can run");
        Serial.println("     python ml_pipeline/export_model.py --axes acc_gyro");
    }
    if (magCount > 1) {
        Serial.println("  -> more than one magnetometer address answered; check for a");
        Serial.println("     second board on the bus or an address conflict");
    }
}

void runActuatorTest() {
    Serial.println();
    Serial.println("Actuator test: buzzer only, then motor only, then both");
    Serial.flush();
    delay(200);

    struct Step {
        const char *label;
        bool buzzer;
        bool vibration;
    };
    static const Step steps[] = {
        {"buzzer", true, false},
        {"vibration motor", false, true},
        {"both", true, true},
    };

    for (const Step &step : steps) {
        AlarmManager::Pattern pattern;
        pattern.pulseOnMs = 150;
        pattern.pulseOffMs = 150;
        pattern.durationMs = 900;
        pattern.useBuzzer = step.buzzer;
        pattern.useVibration = step.vibration;

        Serial.printf("  -> Memulai uji: %s\n", step.label);
        Serial.flush();

        AlarmManager::trigger(pattern);

        while (AlarmManager::isAlerting()) {
            AlarmManager::update();
            delay(10);
        }

        delay(300);
    }

    Serial.println("Actuator test complete");
    Serial.flush();
}

void reportBarometer() {
    if (!Barometer::ready()) {
        return;
    }
    const BaroMath::AltitudeTracker &t = Barometer::tracker();
    if (!t.calibrated() || t.calibrating()) {
        Serial.printf("BARO calibrating altitude reference... %lu samples, keep the board still\n",
                      static_cast<unsigned long>(t.calibrationSamples()));
        return;
    }
    Serial.printf("BARO p=%.2f hPa T=%.1f C alt=%+.2f m dh(%.1fs)=%+.2f m vz=%+.2f m/s "
                  "noise=%.3f m read=%lu us fail=%lu\n",
                  Barometer::pressureHpa(), Barometer::temperatureC(), t.altitudeM(),
                  t.windowSpanS(), t.altitudeDiffM(), t.verticalSpeedMps(), t.noiseM(),
                  static_cast<unsigned long>(Barometer::lastReadMicros()),
                  static_cast<unsigned long>(Barometer::readFailures()));
}

void serviceBaroValidation() {
    static bool wasCalibrated = false;
    static bool reported = false;
    static std::uint32_t referenceSetMs = 0;
    static std::uint32_t failuresAtStart = 0;

    if (!Barometer::ready()) {
        return;
    }
    const BaroMath::AltitudeTracker &t = Barometer::tracker();
    const bool calibrated = t.calibrated() && !t.calibrating();
    if (calibrated && !wasCalibrated) {
        referenceSetMs = millis();
        failuresAtStart = Barometer::readFailures();
        reported = false;
        Serial.printf("BARO reference set: p0=%.2f hPa from %lu samples\n",
                      t.referencePa() / 100.0f,
                      static_cast<unsigned long>(t.calibrationSamples()));
    }
    wasCalibrated = calibrated;
    if (!calibrated || reported || millis() - referenceSetMs < kBaroValidationDelayMs) {
        return;
    }
    reported = true;

    const float noise = t.noiseM();
    const float speed = std::fabs(t.verticalSpeedMps());
    const std::uint32_t failures = Barometer::readFailures() - failuresAtStart;
    const bool noiseOk = noise < kBaroNoiseLimitM;
    const bool stillOk = speed < kBaroStillSpeedLimitMps;
    const bool readsOk = failures == 0;

    Serial.println("BARO validation (board held still):");
    Serial.printf("  altitude noise  %.3f m   (limit %.2f m)    %s\n", noise, kBaroNoiseLimitM,
                  noiseOk ? "PASS" : "WARN");
    Serial.printf("  vertical speed  %.3f m/s (limit %.2f m/s)  %s\n", speed,
                  kBaroStillSpeedLimitMps, stillOk ? "PASS" : "WARN");
    Serial.printf("  read failures   %lu                        %s\n",
                  static_cast<unsigned long>(failures), readsOk ? "PASS" : "WARN");
    Serial.printf("  read latency    %lu us per burst read\n",
                  static_cast<unsigned long>(Barometer::lastReadMicros()));
    if (!noiseOk || !stillOk) {
        Serial.println("  -> keep the board still, away from fans/air-con draughts and heat");
        Serial.println("     sources, then press the GPIO3 button to re-zero and re-check");
    }
}

void serviceMotorCycle() {
    static std::uint32_t nextStartMs = 0;
    static bool running = false;

    const std::uint32_t now = millis();
    if (running) {
        if (!AlarmManager::isAlerting()) {
            running = false;
            nextStartMs = now + kMotorOffMs;
            Serial.println("Motor OFF...");
        }
        return;
    }
    if (static_cast<std::int32_t>(now - nextStartMs) >= 0 && !AlarmManager::isAlerting()) {
        AlarmManager::Pattern pattern;
        pattern.pulseOnMs = kMotorOnMs;
        pattern.pulseOffMs = 0;
        pattern.durationMs = kMotorOnMs;
        pattern.useBuzzer = false;
        pattern.useVibration = true;
        AlarmManager::trigger(pattern);
        running = true;
        Serial.println("Motor ON (GETAR)...");
    }
}

}

void setup() {

    AlarmManager::forceOff();

    Serial.begin(115200);

    const std::uint32_t waitStart = millis();
    while (!Serial && millis() - waitStart < 3000) {
    }
    delay(200);

    AlarmManager::begin();
    banner();

    pinMode(kI2cSdaPin, INPUT_PULLUP);
    pinMode(kI2cSclPin, INPUT_PULLUP);
    Wire.begin(kI2cSdaPin, kI2cSclPin, kI2cFrequencyHz);
    Wire.setTimeOut(25);

    Serial.println("Scanning 0x01..0x7F on the configured pins:");
    const auto scan = I2cDiagnostics::scanBus(Wire);
    I2cDiagnostics::printScanResult(Serial, scan);
    reportExpectations(scan);

    if (scan.count == 0) {
        Serial.println();
        I2cDiagnostics::PinCandidate hits[8];
        const std::size_t found = I2cDiagnostics::sweepPins(Serial, hits, 8, false);
        if (found > 0) {
            Serial.println("Update kI2cSdaPin / kI2cSclPin in include/board_config.h:");
            for (std::size_t i = 0; i < found; ++i) {
                Serial.printf("  SDA=%d SCL=%d (%u devices)\n", hits[i].sda, hits[i].scl,
                              static_cast<unsigned>(hits[i].deviceCount));
            }
        }
        Wire.begin(kI2cSdaPin, kI2cSclPin, kI2cFrequencyHz);
    }

    Serial.println();
    Serial.println("Barometer:");
    const Barometer::Status baro = Barometer::begin(Serial, Wire);
    if (baro.ready) {
        Serial.println("  self-check: chip ID, factory calibration and first reading PASS");
        Serial.println("  keep the board still for ~5 s after the actuator test for validation");
    }

    runActuatorTest();

    Serial.println();
    Serial.println("Idle. Re-scanning every 5 s; press GPIO3 to re-zero the barometer.");
}

void loop() {
    static std::uint32_t lastScanMs = 0;
    static std::uint32_t lastBaroMs = 0;
    static bool lastButton = false;

    AlarmManager::update();
    Barometer::service();
    serviceBaroValidation();

    if (AlarmManager::buttonHeld() != lastButton) {
        lastButton = AlarmManager::buttonHeld();
        Serial.printf("Button GPIO%d: %s\n", kDismissButtonPin,
                      lastButton ? "PRESSED" : "released");
    }
    if (AlarmManager::consumeDismissPress()) {
        Serial.println("  -> dismiss press registered (would silence an active alarm)");
        if (Barometer::ready()) {
            Barometer::rezero();
            Serial.println("  -> BARO re-zeroing altitude: keep the board still for 2 s");
        }
    }

    if (millis() - lastScanMs >= kBusRescanMs) {
        lastScanMs = millis();
        const auto scan = I2cDiagnostics::scanBus(Wire);
        Serial.printf("[%lus] %u device(s) on the bus\n",
                      static_cast<unsigned long>(millis() / 1000),
                      static_cast<unsigned>(scan.count));
        I2cDiagnostics::printScanResult(Serial, scan);
    }

    if (millis() - lastBaroMs >= kBaroReportMs) {
        lastBaroMs = millis();
        reportBarometer();
    }

    if (kMotorCycleTest) {
        serviceMotorCycle();
    }
}
