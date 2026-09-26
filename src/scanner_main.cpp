// =============================================================================
//  scanner_main.cpp - bring-up diagnostics       [env:scanner]
// =============================================================================
//  pio run -e scanner -t upload && pio device monitor -e scanner
//
//  Scans the I2C bus on the configured pins, identifies the IMU and the
//  magnetometer (HMC5883L vs QMC5883L clone), sweeps candidate GPIO pairs if
//  nothing answers, then pulses the buzzer and the vibration motor so wiring can
//  be confirmed by ear and by touch.
// =============================================================================

#include <Arduino.h>
#include <Wire.h>

#include "alarm_manager.h"
#include "board_config.h"
#include "i2c_diagnostics.h"

namespace {

using namespace BoardConfig;

void banner() {
    Serial.println();
    Serial.println("=====================================================");
    Serial.println(" Mitifall - I2C and actuator diagnostics");
    Serial.printf(" SDA=%d SCL=%d @ %lu Hz\n", kI2cSdaPin, kI2cSclPin,
                  static_cast<unsigned long>(kI2cFrequencyHz));
    Serial.printf(" buzzer GPIO%d (active %s) | motor GPIO%d (active %s) | button GPIO%d\n",
                  kBuzzerPin, kBuzzerActiveHigh ? "HIGH" : "LOW", kVibrationPin,
                  kVibrationActiveHigh ? "HIGH" : "LOW", kDismissButtonPin);
    Serial.println("=====================================================");
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

    if (magCount == 0) {
        Serial.println("  -> no magnetometer: only the 6-DoF (acc_gyro) model can run");
        Serial.println("     python ml_pipeline/export_model.py --axes acc_gyro");
    }
    if (magCount > 1) {
        Serial.println("  -> more than one magnetometer address answered; check for a");
        Serial.println("     second board on the bus or an address conflict");
    }
}

/// Non-blocking actuator test: AlarmManager owns the timing, we just pump it.
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

        // Tambahkan delay minimal 10ms di dalam polling loop
        while (AlarmManager::isAlerting()) {
            AlarmManager::update();
            delay(10);
        }

        delay(300); // Jeda antar pengujian aktuator
    }

    Serial.println("Actuator test complete");
    Serial.flush();
}

}  // namespace

void setup() {
    // Silence the actuators before anything else, so a floating pin cannot leave
    // the buzzer sounding through boot and the USB CDC wait below.
    AlarmManager::forceOff();

    Serial.begin(115200);
    // Native USB CDC needs a moment before the host opens the port.
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

    runActuatorTest();

    Serial.println();
    Serial.println("Idle. Re-scanning every 5 s; reset to run the full sweep again.");
}
const int kMotorPin = BoardConfig::kVibrationPin;
void loop() {
    static std::uint32_t lastScanMs = 0;
    static bool lastButton = false;

    AlarmManager::update();

    // Live button feedback so the GPIO3 wiring can be verified by pressing it.
    if (AlarmManager::buttonHeld() != lastButton) {
        lastButton = AlarmManager::buttonHeld();
        Serial.printf("Button GPIO%d: %s\n", kDismissButtonPin,
                      lastButton ? "PRESSED" : "released");
    }
    if (AlarmManager::consumeDismissPress()) {
        Serial.println("  -> dismiss press registered (would silence an active alarm)");
    }

    if (millis() - lastScanMs >= 5000) {
        lastScanMs = millis();
        const auto scan = I2cDiagnostics::scanBus(Wire);
        Serial.printf("[%lus] %u device(s) on the bus\n",
                      static_cast<unsigned long>(millis() / 1000),
                      static_cast<unsigned>(scan.count));
        I2cDiagnostics::printScanResult(Serial, scan);
    }

    Serial.println("Motor ON (GETAR)...");
    digitalWrite(kMotorPin, HIGH);
    delay(1000); // Getar selama 1 detik

    Serial.println("Motor OFF...");
    digitalWrite(kMotorPin, LOW);
    delay(2000); // Berhenti 2 detik
}
