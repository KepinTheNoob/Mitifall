// =============================================================================
//  raw_stream_main.cpp - 50 Hz 9-DoF logger + Adafruit IO telemetry
//                                                        [env:raw_stream]
// =============================================================================
//  pio run -e raw_stream -t upload -t monitor
//
//  Local: one CSV line per sample over USB CDC, same units and column order as
//  dataset/processed/, so a capture can go straight into the Python pipeline:
//
//    Timestamp_ms,Ax,Ay,Az,Gx,Gy,Gz,Mx,My,Mz
//
//  Capture with:  pio device monitor -e raw_stream --quiet > capture.csv
//
//  Cloud: Adafruit IO over plain MQTT (PubSubClient)
//    publish   {user}/feeds/accel-mag        mean |a| over the window   -> chart
//    publish   {user}/feeds/fall-detected    1 on detection, 0 on clear -> LED
//    subscribe {user}/feeds/buzzer-command   dashboard toggle           -> buzzer
//
//  Sampling is the priority: WiFi and MQTT work is only ever done between
//  samples, and the network never gates acquisition or CSV output.
// =============================================================================

#include <Arduino.h>
#include <PubSubClient.h>
#include <WiFi.h>

#include <cmath>

#include "adafruit_io_config.h"
#include "alarm_manager.h"
#include "board_config.h"
#include "sensor_hub.h"
#include "threshold_detector.h"
#include "wifi_link.h"

namespace {

using namespace BoardConfig;
using namespace AdafruitIoConfig;

// Set false for a pure CSV capture with no '#' preamble or status lines.
constexpr bool kPrintBanner = true;

// Set false to run purely local (no WiFi, no MQTT), e.g. for bench logging.
constexpr bool kEnableCloud = true;

// Rule-based fall detection. Independent of the Random Forest: this target
// contains no model, only the threshold cascade in threshold_detector.cpp.
constexpr bool kEnableFallDetection = true;

// Log every impact candidate and why it was rejected. Impacts are rare enough
// that this stays readable, and it is how you tune the thresholds on real motion.
constexpr bool kLogCandidates = true;

// Sensitivity preset applied at boot. Numbers are the LOSO-validated
// recording-level scores from the UMAFall wrist data (see README 9.4).
//
//   0  shipped default   impact 3.2 g, std 0.50  -> sens 0.856  spec 0.942
//   1  sensitive         impact 2.6 g, std 0.50  -> sens 0.933  spec 0.913
//   2  strict            impact 3.0 g, std 0.32  -> sens 0.827  spec 0.954
//   3  bench testing     impact 1.8 g, tilt 15 deg - fires easily by hand,
//                        NOT for real use: far too many false alarms
constexpr int kSensitivityPreset = 3;

ThresholdDetector::Detector gDetector;

// Diagnostics: how far the real motion gets through the cascade.
std::uint32_t gImpactCount = 0;
std::uint32_t gRejectCount = 0;
float gPeakAccelG = 0.0f;   // since the last status line
float gSessionPeakG = 0.0f;  // since boot

WiFiClient gTcp;
PubSubClient gMqtt(gTcp);

std::uint32_t gNextSampleMs = 0;
std::uint32_t gSampleCount = 0;
std::uint32_t gMissedReads = 0;

std::uint32_t gLastPublishMs = 0;
std::uint32_t gLastMqttAttemptMs = 0;
std::uint32_t gPublishCount = 0;

std::uint32_t gFallCount = 0;
bool gFallDetectedPublished = false;

// kEnableCloud AND credentials actually present in secrets.h.
bool gCloudActive = false;

// ---------------------------------------------------------------------------
// Rolling |a| statistics for the publish window (Welford, single pass)
// ---------------------------------------------------------------------------
struct MagnitudeStats {
    std::uint32_t count = 0;
    float mean = 0.0f;
    float m2 = 0.0f;
    float max = 0.0f;

    void add(float value) {
        ++count;
        const float delta = value - mean;
        mean += delta / static_cast<float>(count);
        m2 += delta * (value - mean);
        if (value > max) {
            max = value;
        }
    }

    float stddev() const {
        return count > 1 ? std::sqrt(m2 / static_cast<float>(count)) : 0.0f;
    }

    void reset() { *this = MagnitudeStats{}; }
};

MagnitudeStats gWindow;

// ---------------------------------------------------------------------------
// MQTT
// ---------------------------------------------------------------------------

void onMqttMessage(char *topic, std::uint8_t *payload, unsigned int length) {
    // Payloads are tiny ("ON"/"OFF"/"1"/"0"); bound the copy regardless.
    char text[16];
    const unsigned int copied = length < sizeof(text) - 1 ? length : sizeof(text) - 1;
    memcpy(text, payload, copied);
    text[copied] = '\0';

    if (kPrintBanner) {
        Serial.printf("# mqtt rx %s = %s\n", topic, text);
    }

    if (strcmp(topic, AIO_FEED_BUZZER_COMMAND) == 0) {
        const bool on = text[0] == '1' || text[0] == 'O' || text[0] == 'o' || text[0] == 't' ||
                        text[0] == 'T';
        if (on) {
            AlarmManager::trigger();
            if (kPrintBanner) {
                Serial.println("# dashboard triggered the buzzer");
            }
        } else {
            AlarmManager::cancel();
        }
        return;
    }

    if (strcmp(topic, AIO_TOPIC_THROTTLE) == 0 || strcmp(topic, AIO_TOPIC_ERRORS) == 0) {
        Serial.printf("# ADAFRUIT IO WARNING: %s\n", text);
    }
}

/// One bounded connection attempt. Returns true when the broker is up.
bool serviceMqtt() {
    if (gMqtt.connected()) {
        gMqtt.loop();
        return true;
    }

    if (!WifiLink::isConnected()) {
        return false;
    }
    if (millis() - gLastMqttAttemptMs < kMqttRetryIntervalMs) {
        return false;
    }
    gLastMqttAttemptMs = millis();

    // Client id must be unique per connection; the MAC keeps it stable per board.
    char clientId[32];
    snprintf(clientId, sizeof(clientId), "mitifall-%06llX",
             static_cast<unsigned long long>(ESP.getEfuseMac() & 0xFFFFFFULL));

    if (kPrintBanner) {
        Serial.printf("# mqtt connecting to %s:%d as %s\n", AIO_SERVER, AIO_PORT, clientId);
    }

    if (!gMqtt.connect(clientId, AIO_USERNAME, AIO_KEY)) {
        Serial.printf("# mqtt connect failed, state=%d (retry in %lu ms)\n", gMqtt.state(),
                      static_cast<unsigned long>(kMqttRetryIntervalMs));
        return false;
    }

    gMqtt.subscribe(AIO_FEED_BUZZER_COMMAND);
    gMqtt.subscribe(AIO_TOPIC_THROTTLE);
    gMqtt.subscribe(AIO_TOPIC_ERRORS);
    if (kPrintBanner) {
        Serial.println("# mqtt connected, subscribed to buzzer-command / throttle / errors");
    }
    return true;
}

void publishWindow() {
    if (gWindow.count == 0) {
        return;
    }

    const float meanMag = gWindow.mean;
    const float stdMag = gWindow.stddev();
    const bool moving = stdMag > kMotionStdThresholdG;

    if (kPrintBanner) {
        // |a|max vs the impact threshold is the first thing to check when nothing
        // is being detected: if peak never approaches impactG, stage 2 never fires.
        Serial.printf("# window n=%lu |a|mean=%.3f |a|max=%.3f |a|std=%.3f -> %s | "
                      "det=%s peak=%.2fg (need %.2fg) impacts=%lu rejected=%lu falls=%lu\n",
                      static_cast<unsigned long>(gWindow.count), meanMag, gWindow.max, stdMag,
                      moving ? "MOVING" : "static",
                      ThresholdDetector::Detector::stateName(gDetector.state()), gPeakAccelG,
                      gDetector.config().impactG, static_cast<unsigned long>(gImpactCount),
                      static_cast<unsigned long>(gRejectCount),
                      static_cast<unsigned long>(gFallCount));
        gPeakAccelG = 0.0f;
    }

    if (gMqtt.connected()) {
        // accel-mag is the only periodic feed; fall-detected is edge-triggered
        // from the detector. See the rate budget in adafruit_io_config.h.
        char payload[16];
        snprintf(payload, sizeof(payload), "%.4f", meanMag);
        if (gMqtt.publish(AIO_FEED_ACCEL_MAG, payload)) {
            ++gPublishCount;
        }
    }

    gWindow.reset();
}

// ---------------------------------------------------------------------------
// Rule-based fall detection
// ---------------------------------------------------------------------------

void publishFallDetected(int value) {
    if (!gMqtt.connected()) {
        return;
    }
    if (gMqtt.publish(AIO_FEED_FALL_DETECTED, value ? "1" : "0")) {
        ++gPublishCount;
    }
}

/// Apply the selected sensitivity preset to the detector.
void applySensitivityPreset(int preset) {
    auto &cfg = gDetector.config();
    switch (preset) {
        case 1:  // sensitive
            cfg.impactG = 2.6f;
            cfg.stillStdG = 0.50f;
            break;
        case 2:  // strict
            cfg.impactG = 3.0f;
            cfg.stillStdG = 0.32f;
            break;
        case 3:  // bench testing - deliberately trigger-happy
            cfg.impactG = 1.8f;
            cfg.stillStdG = 0.60f;
            cfg.orientationChangeDeg = 15.0f;
            break;
        default:  // 0 = header defaults, leave as-is
            break;
    }
}

/// Push a synthetic fall through a scratch detector to prove the cascade and the
/// configured thresholds work, independently of whatever the hardware is doing.
///
/// Rest at 1 g -> a hard impact spike -> rest in a tilted orientation. If this
/// fails, the thresholds are unreachable. If it passes but real movement never
/// triggers, the problem is the test motion, not the code.
void runDetectorSelfTest() {
    ThresholdDetector::Detector probe(gDetector.config());
    bool fired = false;
    std::uint32_t t = 0;

    auto feed = [&](float ax, float ay, float az, float gyro, std::uint32_t durationMs) {
        for (std::uint32_t elapsed = 0; elapsed < durationMs; elapsed += 20) {
            t += 20;
            if (probe.update(t, ax, ay, az, gyro, 0.0f, 0.0f) ==
                ThresholdDetector::Event::FallConfirmed) {
                fired = true;
            }
        }
    };

    feed(0.0f, 0.0f, 1.0f, 0.0f, 2000);   // upright and still
    feed(0.0f, 0.0f, 4.5f, 200.0f, 60);   // impact, 4.5 g with angular rate
    feed(0.95f, 0.0f, 0.1f, 0.0f, 2600);  // still again, tipped ~85 degrees

    Serial.printf("# detector self-test: %s (synthetic fall, tilt=%.0fdeg std=%.3fg)\n",
                  fired ? "PASS" : "FAIL", probe.lastOrientationChangeDeg(),
                  probe.lastStillStdG());
    if (!fired) {
        Serial.println("#   thresholds are unreachable - lower impactG or raise stillStdG");
    }
}

/// Feed one sample to the detector and act on whatever it decides.
void runFallDetection(const SensorHub::Sample9 &sample) {
    const std::uint32_t started = micros();
    const ThresholdDetector::Event event =
        gDetector.update(sample.timestampMs, sample.v[SensorHub::kAx], sample.v[SensorHub::kAy],
                         sample.v[SensorHub::kAz], sample.v[SensorHub::kGx],
                         sample.v[SensorHub::kGy], sample.v[SensorHub::kGz]);
    const std::uint32_t elapsed = micros() - started;

    switch (event) {
        case ThresholdDetector::Event::ImpactDetected:
            ++gImpactCount;
            if (kLogCandidates) {
                Serial.printf("# impact |a|=%.2fg -> waiting for stillness + tilt\n",
                              gDetector.lastImpactG());
            }
            break;

        case ThresholdDetector::Event::Rejected:
            ++gRejectCount;
            if (kLogCandidates) {
                Serial.printf("# candidate rejected (%s): |a|max=%.2fg std=%.3fg tilt=%.0fdeg "
                              "|w|peak=%.0fdps\n",
                              ThresholdDetector::Detector::rejectionName(
                                  gDetector.lastRejection()),
                              gDetector.lastImpactG(), gDetector.lastStillStdG(),
                              gDetector.lastOrientationChangeDeg(),
                              gDetector.lastGyroPeakDps());
            }
            break;

        case ThresholdDetector::Event::FallConfirmed: {
            ++gFallCount;
            // Shared default pattern (single 0.5 s burst), identical to the alert
            // [env:ml_inference] raises - see AlarmManager::Pattern.
            AlarmManager::trigger();
            gFallDetectedPublished = true;
            publishFallDetected(1);
            Serial.printf("*** FALL DETECTED (rule-based) #%lu  |a|max=%.2fg  std=%.3fg  "
                          "tilt=%.0fdeg  |w|peak=%.0fdps  detect=%luus ***\n",
                          static_cast<unsigned long>(gFallCount), gDetector.lastImpactG(),
                          gDetector.lastStillStdG(), gDetector.lastOrientationChangeDeg(),
                          gDetector.lastGyroPeakDps(), static_cast<unsigned long>(elapsed));
            break;
        }

        default:
            break;
    }
}

}  // namespace

void setup() {
    // Drive the actuators off before anything else: an undriven GPIO floats, and
    // a floating input on an active-low module reads as "on".
    AlarmManager::forceOff();

    Serial.begin(115200);
    const std::uint32_t waitStart = millis();
    while (!Serial && millis() - waitStart < 3000) {
    }
    delay(200);

    AlarmManager::begin();

    if (kPrintBanner) {
        Serial.println();
        Serial.println("# Mitifall raw 9-DoF stream + Adafruit IO");
        Serial.printf("# rate=%lu Hz period=%lu ms\n", static_cast<unsigned long>(kSampleRateHz),
                      static_cast<unsigned long>(kSamplePeriodMs));
        Serial.println("# units: accel g, gyro deg/s, mag uT");
    }

    const auto status = SensorHub::begin(Serial);
    if (!status.imuReady) {
        Serial.println("# FATAL: no IMU. Run the 'scanner' environment to debug wiring.");
    }
    if (kPrintBanner) {
        Serial.printf("# magnetometer: %s\n", SensorHub::magKindName(status.magKind));
    }

    if (kEnableFallDetection) {
        gDetector.reset();
        applySensitivityPreset(kSensitivityPreset);
        const auto &cfg = gDetector.config();
        Serial.println("# fall detection: rule-based cascade (no ML model)");
        Serial.printf("#   impact >= %.2f g, then still (std <= %.2f g) with tilt >= %.0f deg\n",
                      cfg.impactG, cfg.stillStdG, cfg.orientationChangeDeg);
        Serial.printf("#   assessed %lu-%lu ms after impact; decision ~%.1f s post-impact\n",
                      static_cast<unsigned long>(cfg.settleDelayMs),
                      static_cast<unsigned long>(cfg.settleDelayMs + cfg.stillnessWindowMs),
                      (cfg.settleDelayMs + cfg.stillnessWindowMs) / 1000.0);
        Serial.printf("#   preset %d; set kSensitivityPreset=3 in raw_stream_main.cpp for "
                      "easy bench triggering\n", kSensitivityPreset);
        runDetectorSelfTest();
    }

    gCloudActive = kEnableCloud && credentialsConfigured();

    if (kEnableCloud && !gCloudActive) {
        Serial.println("# Adafruit IO disabled: credentials are empty.");
        Serial.println("#   cp include/secrets.example.h include/secrets.h  (then fill it in)");
        Serial.println("# Local CSV streaming continues normally.");
    }

    if (gCloudActive) {
        // PSK vs WPA2-Enterprise is decided by include/secrets.h.
        WifiLink::beginFromSecrets(kPrintBanner ? &Serial : nullptr);
        if (kPrintBanner) {
            Serial.printf("# wifi mode: %s\n", WifiLink::modeName());
        }
        gMqtt.setServer(AIO_SERVER, AIO_PORT);
        gMqtt.setCallback(onMqttMessage);
        gMqtt.setKeepAlive(kMqttKeepAliveSec);
        gMqtt.setBufferSize(256);
        if (kPrintBanner) {
            Serial.printf("# publishing %s every %lu ms\n", AIO_FEED_ACCEL_MAG,
                          static_cast<unsigned long>(kPublishIntervalMs));
            Serial.printf("# publishing %s on detection (1) and on clear (0)\n",
                          AIO_FEED_FALL_DETECTED);
            Serial.printf("# subscribed to %s\n", AIO_FEED_BUZZER_COMMAND);
        }
    }

    Serial.println("Timestamp_ms,Ax,Ay,Az,Gx,Gy,Gz,Mx,My,Mz");
    gNextSampleMs = millis();
    gLastPublishMs = millis();
}

void loop() {
    AlarmManager::update();

    if (AlarmManager::consumeDismissPress()) {
        // update() already silenced the actuators. Push the switch back to off so
        // the dashboard stops showing an active buzzer command.
        if (kPrintBanner) {
            Serial.println("# button GPIO3: actuators silenced");
        }
        if (gMqtt.connected()) {
            gMqtt.publish(AIO_FEED_BUZZER_COMMAND, "0");
        }
        if (gFallDetectedPublished) {
            publishFallDetected(0);
            gFallDetectedPublished = false;
        }
    }

    // Clear the dashboard indicator once the alert pattern has finished.
    if (gFallDetectedPublished && !AlarmManager::isAlerting()) {
        publishFallDetected(0);
        gFallDetectedPublished = false;
    }

    const std::uint32_t now = millis();
    const bool sampleDue = static_cast<std::int32_t>(now - gNextSampleMs) >= 0;

    if (sampleDue) {
        // Fixed-cadence scheduling: advance by the period so jitter does not
        // accumulate. If we fell far behind (a host that stopped draining the
        // CDC endpoint), resync instead of firing a burst of catch-up samples.
        if (now - gNextSampleMs > 5 * kSamplePeriodMs) {
            gNextSampleMs = now + kSamplePeriodMs;
        } else {
            gNextSampleMs += kSamplePeriodMs;
        }

        SensorHub::Sample9 sample;
        if (SensorHub::read(sample)) {
            ++gSampleCount;

            const float ax = sample.v[SensorHub::kAx];
            const float ay = sample.v[SensorHub::kAy];
            const float az = sample.v[SensorHub::kAz];
            const float accelMag = std::sqrt(ax * ax + ay * ay + az * az);
            gWindow.add(accelMag);
            if (accelMag > gPeakAccelG) {
                gPeakAccelG = accelMag;
            }
            if (accelMag > gSessionPeakG) {
                gSessionPeakG = accelMag;
            }

            if (kEnableFallDetection) {
                runFallDetection(sample);
            }

            // Serial.printf("%lu,%.5f,%.5f,%.5f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f\n",
            //               static_cast<unsigned long>(sample.timestampMs), ax, ay, az,
            //               sample.v[SensorHub::kGx], sample.v[SensorHub::kGy],
            //               sample.v[SensorHub::kGz], sample.v[SensorHub::kMx],
            //               sample.v[SensorHub::kMy], sample.v[SensorHub::kMz]);
        } else {
            ++gMissedReads;
        }
        return;  // one unit of work per iteration; network waits for the gap
    }

    if (!gCloudActive) {
        return;
    }

    WifiLink::update();
    serviceMqtt();

    if (now - gLastPublishMs >= kPublishIntervalMs) {
        gLastPublishMs = now;
        publishWindow();
    }

    if (kPrintBanner && WifiLink::consumeJustConnected()) {
        Serial.printf("# wifi ip=%s rssi=%ld publishes=%lu missed_reads=%lu\n",
                      WifiLink::ipAddress().c_str(), static_cast<long>(WifiLink::rssi()),
                      static_cast<unsigned long>(gPublishCount),
                      static_cast<unsigned long>(gMissedReads));
    }
}
