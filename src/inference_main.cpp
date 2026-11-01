#include <Arduino.h>

#include "RandomForest.h"
#include "alarm_manager.h"
#include "board_config.h"
#include "feature_extractor.h"
#include "sensor_hub.h"

namespace {

using namespace BoardConfig;

static constexpr float FALL_THRESHOLD = 0.30f;

static constexpr std::uint8_t CONFIRMATION_WINDOWS = 1;

static constexpr std::uint32_t RETRIGGER_COOLDOWN_MS = 6000;
static constexpr bool VERBOSE_EVERY_WINDOW = true;

FeatureExtractor::RingBuffer<kWindowSamples> gWindow;
float gPipelineFeatures[FeatureExtractor::kFeatureCount];
float gModelInput[RandomForest::kFeatureCount];

std::uint32_t gNextSampleMs = 0;
std::size_t gSamplesSinceInference = 0;
std::uint32_t gWindowIndex = 0;
std::uint32_t gLastAlertMs = 0;
std::uint8_t gConsecutivePositives = 0;
bool gMagRequired = false;

void reportModel() {
    Serial.println();
    Serial.println("=====================================================");
    Serial.println(" Mitifall - on-device fall detection (Random Forest)");
    Serial.printf(" model     : %u features, %u trees, %u nodes\n",
                  static_cast<unsigned>(RandomForest::kFeatureCount),
                  static_cast<unsigned>(RandomForest::kTreeCount),
                  static_cast<unsigned>(RandomForest::kNodeCount));
    Serial.printf(" window    : %u samples @ %lu Hz (%.1f s), hop %u samples\n",
                  static_cast<unsigned>(kWindowSamples),
                  static_cast<unsigned long>(kSampleRateHz),
                  static_cast<double>(kWindowSamples) / kSampleRateHz,
                  static_cast<unsigned>(kHopSamples));
    Serial.printf(" threshold : %.2f, confirmation %u window(s)\n",
                  static_cast<double>(FALL_THRESHOLD),
                  static_cast<unsigned>(CONFIRMATION_WINDOWS));
    Serial.println("=====================================================");
}

void runSelfTest() {
#if RF_INCLUDE_METADATA
    const std::uint32_t started = micros();
    const bool ok = RandomForest::selfTest();
    const std::uint32_t elapsed = micros() - started;
    Serial.printf("Model self-test: %s (%u vectors, %lu us)\n", ok ? "PASS" : "FAIL",
                  static_cast<unsigned>(RandomForest::kSelfTestCount),
                  static_cast<unsigned long>(elapsed));
    if (!ok) {
        Serial.println("  the flashed tables disagree with the trained model - re-export");
    }
#else
    Serial.println("Model self-test: skipped (RF_INCLUDE_METADATA = 0)");
#endif
}

void handleDetection(float probability) {
    const std::uint32_t now = millis();
    const bool cooling = (now - gLastAlertMs) < RETRIGGER_COOLDOWN_MS;

    if (AlarmManager::isAlerting() || cooling) {
        Serial.printf("  fall p=%.3f suppressed (%s)\n", static_cast<double>(probability),
                      AlarmManager::isAlerting() ? "already alerting" : "cooldown");
        return;
    }

    gLastAlertMs = now;
    AlarmManager::trigger();

    Serial.printf("*** FALL DETECTED  p=%.3f  alerting for %lu ms ***\n",
                  static_cast<double>(probability),
                  static_cast<unsigned long>(AlarmManager::defaultPattern().durationMs));
}

void runInference() {
    const std::uint32_t featureStart = micros();
    FeatureExtractor::compute(gWindow, gPipelineFeatures);
    const std::uint32_t featureMicros = micros() - featureStart;

    const std::uint32_t inferenceStart = micros();
    RandomForest::gather(gPipelineFeatures, gModelInput);
    const float probability = RandomForest::predictProba(gModelInput);
    const std::uint16_t votes = RandomForest::voteCount(gModelInput);
    const std::uint32_t inferenceMicros = micros() - inferenceStart;

    const bool positive = probability >= FALL_THRESHOLD;
    if (positive) {
        if (gConsecutivePositives < 255) {
            ++gConsecutivePositives;
        }
    } else {
        gConsecutivePositives = 0;
    }

    if (VERBOSE_EVERY_WINDOW) {
        Serial.printf("w%-5lu p=%.3f votes=%u/%u %-8s feat=%luus infer=%luus |a|max=%.2fg\n",
                      static_cast<unsigned long>(gWindowIndex), static_cast<double>(probability),
                      static_cast<unsigned>(votes),
                      static_cast<unsigned>(RandomForest::kTreeCount),
                      positive ? "FALL?" : "adl",
                      static_cast<unsigned long>(featureMicros),
                      static_cast<unsigned long>(inferenceMicros),
                      static_cast<double>(gPipelineFeatures[45]));
    }

    if (positive && gConsecutivePositives >= CONFIRMATION_WINDOWS) {
        handleDetection(probability);
        gConsecutivePositives = 0;
    }

    ++gWindowIndex;
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
    Serial.printf("Actuators off; dismiss button on GPIO%d%s\n", kDismissButtonPin,
                  AlarmManager::buttonHeld() ? " (currently HELD - check wiring)" : "");
    reportModel();
    runSelfTest();

    const auto status = SensorHub::begin(Serial);
    if (!status.imuReady) {
        Serial.println("FATAL: no IMU - flash the 'scanner' environment to debug wiring");
    }

    gMagRequired = RandomForest::kFeatureCount == FeatureExtractor::kFeatureCount;
    if (gMagRequired && !status.magReady) {
        Serial.println("FATAL: this model expects 9-DoF input but no magnetometer was found.");
        Serial.println("       Re-export a 6-DoF model:");
        Serial.println("       python ml_pipeline/export_model.py --axes acc_gyro");
    }

    gWindow.clear();
    gNextSampleMs = millis();
}

void loop() {
    AlarmManager::update();

    if (AlarmManager::consumeDismissPress()) {
        gConsecutivePositives = 0;
        gLastAlertMs = millis();
        Serial.println("Button GPIO3: alarm dismissed");
    }

    const std::uint32_t now = millis();
    if (static_cast<std::int32_t>(now - gNextSampleMs) < 0) {
        return;
    }

    if (now - gNextSampleMs > 5 * kSamplePeriodMs) {
        gNextSampleMs = now + kSamplePeriodMs;
    } else {
        gNextSampleMs += kSamplePeriodMs;
    }

    if (!SensorHub::status().imuReady || (gMagRequired && !SensorHub::status().magReady)) {
        return;
    }

    SensorHub::Sample9 sample;
    if (!SensorHub::read(sample)) {
        return;
    }
    gWindow.push(sample.v);

    if (!gWindow.full()) {
        return;
    }

    static bool firstWindowScored = false;
    if (!firstWindowScored) {
        firstWindowScored = true;
        gSamplesSinceInference = 0;
        runInference();
        return;
    }

    if (++gSamplesSinceInference >= kHopSamples) {
        gSamplesSinceInference = 0;
        runInference();
    }
}
