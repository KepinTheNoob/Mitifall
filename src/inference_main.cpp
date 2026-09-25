// =============================================================================
//  inference_main.cpp - on-device fall detection + Blynk IoT   [env:ml_inference]
// =============================================================================
//  pio run -e ml_inference -t upload -t monitor
//
//  Device pipeline, mirroring the Python training pipeline:
//    50 Hz acquisition -> 100-sample (2.0 s) ring buffer -> every 50 samples
//    (1.0 s hop, 50 % overlap) extract 53 features -> RandomForest probability
//    -> threshold -> non-blocking buzzer + vibration alert.
//
//  Blynk virtual pins:
//    V0  Double   fall probability 0.0 .. 1.0   -> SuperChart
//    V1  Integer  alarm status 0 / 1            -> LED indicator
//    V2  Integer  dashboard button              -> dismiss alarm / test actuators
//
//  Detection and the local alarm are never gated on connectivity: if WiFi or
//  Blynk is down the device still samples, infers, buzzes and vibrates.
// =============================================================================

#include <Arduino.h>

// blynk_config.h must precede BlynkSimpleEsp32.h - the library reads
// BLYNK_TEMPLATE_ID / BLYNK_TEMPLATE_NAME / BLYNK_PRINT at preprocessor time.
#include "blynk_config.h"
// clang-format off
#include <BlynkSimpleEsp32.h>
// clang-format on

#include "RandomForest.h"
#include "alarm_manager.h"
#include "board_config.h"
#include "feature_extractor.h"
#include "sensor_hub.h"
#include "wifi_link.h"

namespace {

using namespace BoardConfig;

// ---------------------------------------------------------------------------
// Operating point
// ---------------------------------------------------------------------------
// Chosen from the LOSO out-of-fold sweep in ml_pipeline/outputs/threshold_sweep.csv.
// For the 53-feature forest, pooled over all 19 subjects:
//   0.25 -> sensitivity 0.861  specificity 0.754
//   0.30 -> sensitivity 0.835  specificity 0.824   <- default
//   0.35 -> sensitivity 0.807  specificity 0.871
//   0.50 -> sensitivity 0.720  specificity 0.948
static constexpr float FALL_THRESHOLD = 0.30f;

// Specificity 0.824 at one decision per second means a nuisance alert every few
// seconds of ordinary movement. Requiring N consecutive windows over threshold
// trades detection latency for far fewer false alarms; 1 alerts on a single
// window, as the bare threshold rule does.
static constexpr std::uint8_t CONFIRMATION_WINDOWS = 1;

static constexpr std::uint32_t RETRIGGER_COOLDOWN_MS = 6000;
static constexpr bool VERBOSE_EVERY_WINDOW = true;

// Set false to run fully offline (no WiFi, no Blynk).
static constexpr bool ENABLE_CLOUD = true;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
FeatureExtractor::RingBuffer<kWindowSamples> gWindow;
float gPipelineFeatures[FeatureExtractor::kFeatureCount];
float gModelInput[RandomForest::kFeatureCount];

BlynkTimer gTimer;

std::uint32_t gNextSampleMs = 0;
std::size_t gSamplesSinceInference = 0;
std::uint32_t gWindowIndex = 0;
std::uint32_t gLastAlertMs = 0;
std::uint32_t gLastBlynkAttemptMs = 0;
std::uint8_t gConsecutivePositives = 0;
bool gMagRequired = false;

// Shared with the Blynk timer callbacks.
volatile float gLatestProbability = 0.0f;
volatile int gAlarmStatus = 0;
bool gStatusDirty = false;

// ENABLE_CLOUD AND credentials actually present in secrets.h.
bool gCloudActive = false;

bool cloudReady() { return gCloudActive && Blynk.connected(); }

void reportModel() {
    Serial.println();
    Serial.println("=====================================================");
    Serial.println(" Mitifall - on-device fall detection + Blynk IoT");
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
    Serial.printf(" blynk     : template %s, pins V0/V1/V2\n", BLYNK_TEMPLATE_ID);
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

// ---------------------------------------------------------------------------
// Blynk telemetry - driven by BlynkTimer, never from the sampling path
// ---------------------------------------------------------------------------

void pushProbability() {
    if (cloudReady()) {
        Blynk.virtualWrite(VPIN_FALL_PROBABILITY, gLatestProbability);
    }
}

void pushStatus() {
    if (cloudReady()) {
        Blynk.virtualWrite(VPIN_ALARM_STATUS, gAlarmStatus);
        gStatusDirty = false;
    }
}

void updateAlarmStatus(int status) {
    if (gAlarmStatus == status) {
        return;
    }
    gAlarmStatus = status;
    gStatusDirty = true;
    // Status changes are pushed immediately; the timer only provides a heartbeat.
    pushStatus();
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
    updateAlarmStatus(1);

    Serial.printf("*** FALL DETECTED  p=%.3f  alerting for %lu ms ***\n",
                  static_cast<double>(probability),
                  static_cast<unsigned long>(AlarmManager::defaultPattern().durationMs));

    if (cloudReady()) {
        Blynk.logEvent("fall_detected");
    }
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

    gLatestProbability = probability;

    const bool positive = probability >= FALL_THRESHOLD;
    if (positive) {
        if (gConsecutivePositives < 255) {
            ++gConsecutivePositives;
        }
    } else {
        gConsecutivePositives = 0;
    }

    if (VERBOSE_EVERY_WINDOW) {
        Serial.printf("w%-5lu p=%.3f votes=%u/%u %-8s feat=%luus infer=%luus |a|max=%.2fg %s\n",
                      static_cast<unsigned long>(gWindowIndex), static_cast<double>(probability),
                      static_cast<unsigned>(votes),
                      static_cast<unsigned>(RandomForest::kTreeCount),
                      positive ? "FALL?" : "adl",
                      static_cast<unsigned long>(featureMicros),
                      static_cast<unsigned long>(inferenceMicros),
                      static_cast<double>(gPipelineFeatures[45]),  // acc_mag_max
                      cloudReady() ? "cloud" : "local");
    }

    if (positive && gConsecutivePositives >= CONFIRMATION_WINDOWS) {
        handleDetection(probability);
        gConsecutivePositives = 0;
    }

    ++gWindowIndex;
}

/// Bounded, throttled Blynk connection maintenance.
void serviceBlynk() {
    if (!gCloudActive) {
        return;
    }

    WifiLink::update();
    if (!WifiLink::isConnected()) {
        return;
    }

    if (Blynk.connected()) {
        Blynk.run();
        return;
    }

    if (millis() - gLastBlynkAttemptMs < BlynkConfig::kConnectRetryIntervalMs) {
        return;
    }
    gLastBlynkAttemptMs = millis();

    // Bounded timeout: a dead server must not stall the acquisition loop.
    if (Blynk.connect(BlynkConfig::kConnectTimeoutMs)) {
        Serial.println("Blynk: connected");
        pushStatus();
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// V2: dashboard button - dismiss an active alarm, or test the actuators
// ---------------------------------------------------------------------------
BLYNK_WRITE(V2) {
    const int value = param.asInt();
    if (value == 0) {
        return;
    }

    if (AlarmManager::isAlerting()) {
        AlarmManager::cancel();
        updateAlarmStatus(0);
        gConsecutivePositives = 0;
        Serial.println("Blynk V2: alarm dismissed");
    } else {
        AlarmManager::Pattern test;
        test.pulseOnMs = 120;
        test.pulseOffMs = 120;
        test.durationMs = 900;
        AlarmManager::trigger(test);
        Serial.println("Blynk V2: actuator test");
    }
}

BLYNK_CONNECTED() {
    // Pull the button state and push current values so a freshly opened
    // dashboard is never showing stale data.
    Blynk.syncVirtual(VPIN_DISMISS_BUTTON);
    Blynk.virtualWrite(VPIN_FALL_PROBABILITY, gLatestProbability);
    Blynk.virtualWrite(VPIN_ALARM_STATUS, gAlarmStatus);
}

void setup() {
    Serial.begin(115200);
    const std::uint32_t waitStart = millis();
    while (!Serial && millis() - waitStart < 3000) {
    }
    delay(200);

    AlarmManager::begin();
    reportModel();
    runSelfTest();

    const auto status = SensorHub::begin(Serial);
    if (!status.imuReady) {
        Serial.println("FATAL: no IMU - flash the 'scanner' environment to debug wiring");
    }

    // A 53-feature model needs magnetometer data; feeding zeros would put every
    // window far outside the training distribution, so refuse instead.
    gMagRequired = RandomForest::kFeatureCount == FeatureExtractor::kFeatureCount;
    if (gMagRequired && !status.magReady) {
        Serial.println("FATAL: this model expects 9-DoF input but no magnetometer was found.");
        Serial.println("       Re-export a 6-DoF model:");
        Serial.println("       python ml_pipeline/export_model.py --axes acc_gyro");
    }

    gCloudActive = ENABLE_CLOUD && BlynkConfig::credentialsConfigured();

    if (ENABLE_CLOUD && !gCloudActive) {
        Serial.println("Blynk disabled: credentials are empty.");
        Serial.println("  cp include/secrets.example.h include/secrets.h  (then fill it in)");
        Serial.println("Fall detection, buzzer and vibration continue to work offline.");
    }

    if (gCloudActive) {
        // PSK vs WPA2-Enterprise is decided by include/secrets.h.
        WifiLink::beginFromSecrets(&Serial);
        Serial.printf("WiFi mode: %s\n", WifiLink::modeName());
        Blynk.config(BLYNK_AUTH_TOKEN);
        gTimer.setInterval(BlynkConfig::kProbabilityPushMs, pushProbability);
        gTimer.setInterval(BlynkConfig::kStatusPushMs, pushStatus);
    }

    gWindow.clear();
    gNextSampleMs = millis();
}

void loop() {
    // Actuator timing first: it must keep running even if a read or the network
    // stalls. The alarm is the safety-critical output.
    AlarmManager::update();

    if (gAlarmStatus == 1 && !AlarmManager::isAlerting()) {
        updateAlarmStatus(0);  // pattern finished on its own
    }

    const std::uint32_t now = millis();
    const bool sampleDue = static_cast<std::int32_t>(now - gNextSampleMs) >= 0;

    if (!sampleDue) {
        // Spare time between samples: this is where all network work happens.
        if (gCloudActive) {
            gTimer.run();
            serviceBlynk();
        }
        return;
    }

    // Fixed-cadence scheduling, with a resync if we fell badly behind so the
    // window never fills with a burst of back-to-back samples.
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
        return;  // wait for a complete 2.0 s window before the first inference
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
