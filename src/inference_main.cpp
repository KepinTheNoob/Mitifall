#include <Arduino.h>
#include <PubSubClient.h>
#include <WiFi.h>

#include <cmath>
#include <cstring>
#include <ctime>

#include "RandomForest.h"
#include "adafruit_io_config.h"
#include "alarm_manager.h"
#include "barometer.h"
#include "board_config.h"
#include "device_log.h"
#include "feature_extractor.h"
#include "sensor_hub.h"
#include "wifi_link.h"

namespace {

using namespace BoardConfig;
using namespace AdafruitIoConfig;

constexpr bool kPrintBanner = true;
constexpr bool kEnableCloud = true;
constexpr std::uint32_t kDeviceLogIntervalMs = 2500;

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

WiFiClient gTcp;
PubSubClient gMqtt(gTcp);

std::uint32_t gSampleCount = 0;
std::uint32_t gMissedReads = 0;

std::uint32_t gLastPublishMs = 0;
std::uint32_t gLastMqttAttemptMs = 0;
std::uint32_t gPublishCount = 0;

std::uint32_t gFallCount = 0;
bool gFallDetectedPublished = false;
bool gIncidentOpen = false;

bool gWillEnabled = true;
std::uint8_t gWillFailures = 0;

bool gTimeSyncStarted = false;
bool gCloudActive = false;

std::uint32_t gPublishSlot = 0;
std::uint8_t gBaroFeedIndex = 0;

float gAltDiffPeakM = 0.0f;
bool gAltDiffPeakValid = false;
float gStatusAltDiffPeakM = 0.0f;

bool baroAltitudeValid() {
    if (!Barometer::ready()) {
        return false;
    }
    const BaroMath::AltitudeTracker &t = Barometer::tracker();
    return t.calibrated() && !t.calibrating();
}

bool baroTelemetryActive() { return kPublishBarometer && Barometer::ready(); }

void onBaroSample() {
    if (!baroAltitudeValid()) {
        return;
    }
    const float dh = Barometer::tracker().altitudeDiffM();
    if (!gAltDiffPeakValid || std::fabs(dh) > std::fabs(gAltDiffPeakM)) {
        gAltDiffPeakM = dh;
        gAltDiffPeakValid = true;
    }
    if (std::fabs(dh) > std::fabs(gStatusAltDiffPeakM)) {
        gStatusAltDiffPeakM = dh;
    }
}

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

MagnitudeStats gWindowStats;
MagnitudeStats gAccelMagWindow;
MagnitudeStats gLogWindow;
std::uint32_t gLastLogMs = 0;

void logDeviceRow(const char *event) {
    DeviceLog::Row row;
    row.event = event;
    row.hasAccel = gLogWindow.count > 0;
    row.accelMag = gLogWindow.mean;
    row.alarm = AlarmManager::isAlerting();
    if (Barometer::ready() && Barometer::tracker().sampleCount() > 0) {
        row.hasPressure = true;
        row.pressureHpa = Barometer::pressureHpa();
        if (baroAltitudeValid()) {
            row.hasAltitude = true;
            row.altitudeM = Barometer::tracker().altitudeM();
        }
    }
    DeviceLog::add(row);
}

void formatTimestamp(char *out, std::size_t size) {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
    localtime_r(&now, &local);
    if (local.tm_year > (2020 - 1900)) {
        std::strftime(out, size, "%Y-%m-%d %H:%M:%S %Z", &local);
        return;
    }
    const unsigned long up = millis() / 1000UL;
    snprintf(out, size, "uptime %02lu:%02lu:%02lu", up / 3600UL, (up / 60UL) % 60UL, up % 60UL);
}

struct PendingEvent {
    const char *topic;
    char payload[64];
};

PendingEvent gEventQueue[kEventQueueCapacity];
std::size_t gEventHead = 0;
std::size_t gEventCount = 0;
std::uint32_t gEventsDropped = 0;

void queueEvent(const char *topic, const char *payload) {
    if (gEventCount == kEventQueueCapacity) {
        gEventHead = (gEventHead + 1) % kEventQueueCapacity;
        --gEventCount;
        ++gEventsDropped;
    }
    PendingEvent &slot = gEventQueue[(gEventHead + gEventCount) % kEventQueueCapacity];
    slot.topic = topic;
    snprintf(slot.payload, sizeof(slot.payload), "%s", payload);
    ++gEventCount;

    if (kPrintBanner) {
        Serial.printf("# event queued -> %s: %s%s\n", topic, payload,
                      gMqtt.connected() ? "" : " (MQTT down, will send on reconnect)");
    }
}

void flushEvents() {
    while (gEventCount > 0 && gMqtt.connected()) {
        const PendingEvent &next = gEventQueue[gEventHead];
        if (!gMqtt.publish(next.topic, next.payload)) {
            return;
        }
        ++gPublishCount;
        if (kPrintBanner) {
            Serial.printf("# event sent   -> %s: %s\n", next.topic, next.payload);
        }
        gEventHead = (gEventHead + 1) % kEventQueueCapacity;
        --gEventCount;
    }
}

void logIncident(const char *what) {
    char stamp[32];
    formatTimestamp(stamp, sizeof(stamp));
    char line[64];
    snprintf(line, sizeof(line), "[%s] %s", stamp, what);
    queueEvent(AIO_FEED_FALL_LOG, line);
}

void onMqttMessage(char *topic, std::uint8_t *payload, unsigned int length) {
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

    char clientId[32];
    snprintf(clientId, sizeof(clientId), "mitifall-%06llX",
             static_cast<unsigned long long>(ESP.getEfuseMac() & 0xFFFFFFULL));

    if (kPrintBanner) {
        Serial.printf("# mqtt connecting to %s:%d as %s\n", AIO_SERVER, AIO_PORT, clientId);
    }

    const bool probeWithoutWill = gWillEnabled && gWillFailures >= kWillFailuresBeforeFallback;
    const bool useWill = gWillEnabled && !probeWithoutWill;

    const bool connected =
        useWill ? gMqtt.connect(clientId, AIO_USERNAME, AIO_KEY, AIO_FEED_DEVICE_STATUS, kWillQos,
                                kWillRetain, kStatusOffline)
                : gMqtt.connect(clientId, AIO_USERNAME, AIO_KEY);

    if (!connected) {
        if (probeWithoutWill) {
            gWillFailures = 0;
        } else if (useWill) {
            ++gWillFailures;
        }
        Serial.printf("# mqtt connect failed, state=%d (retry in %lu ms)\n", gMqtt.state(),
                      static_cast<unsigned long>(kMqttRetryIntervalMs));
        return false;
    }

    if (probeWithoutWill) {
        gWillEnabled = false;
        Serial.println("# mqtt: broker rejected the Last Will; connected without it");
    }
    gWillFailures = 0;

    gMqtt.subscribe(AIO_TOPIC_THROTTLE);
    gMqtt.subscribe(AIO_TOPIC_ERRORS);
    gMqtt.subscribe(AIO_FEED_BUZZER_COMMAND);

    const bool announced = gMqtt.publish(AIO_FEED_DEVICE_STATUS, kStatusOnline, true);
    if (announced) {
        ++gPublishCount;
    }
    if (kPrintBanner) {
        Serial.printf("# mqtt connected (LWT %s), %s = %s%s\n",
                      useWill ? "registered" : "not registered", AIO_FEED_DEVICE_STATUS,
                      kStatusOnline, announced ? "" : " (publish FAILED)");
        Serial.println("# subscribed to throttle / errors / buzzer-command");
    }

    flushEvents();
    return true;
}

bool publishValue(const char *topic, const char *format, float value) {
    if (!gMqtt.connected()) {
        return false;
    }
    char payload[16];
    snprintf(payload, sizeof(payload), format, value);
    if (!gMqtt.publish(topic, payload)) {
        return false;
    }
    ++gPublishCount;
    return true;
}

void publishBarometerSlot() {
    const std::uint8_t feed = gBaroFeedIndex;
    gBaroFeedIndex = static_cast<std::uint8_t>((gBaroFeedIndex + 1) % 3);

    switch (feed) {
        case 0:
            if (Barometer::tracker().sampleCount() > 0) {
                publishValue(AIO_FEED_PRESSURE, "%.2f", Barometer::pressureHpa());
            }
            break;
        case 1:
            if (baroAltitudeValid()) {
                publishValue(AIO_FEED_ALTITUDE, "%.2f", Barometer::tracker().altitudeM());
            }
            break;
        default:
            if (gAltDiffPeakValid &&
                publishValue(AIO_FEED_ALTITUDE_DIFF, "%.2f", gAltDiffPeakM)) {
                gAltDiffPeakM = 0.0f;
                gAltDiffPeakValid = false;
            }
            break;
    }
}

void printStatusLine() {
    const float stdMag = gWindowStats.stddev();
    const bool moving = stdMag > kMotionStdThresholdG;

    Serial.printf("# window n=%lu |a|mean=%.3f |a|max=%.3f |a|std=%.3f -> %s | "
                  "infer_w=%lu falls=%lu",
                  static_cast<unsigned long>(gWindowStats.count), gWindowStats.mean, gWindowStats.max, stdMag,
                  moving ? "MOVING" : "static",
                  static_cast<unsigned long>(gWindowIndex),
                  static_cast<unsigned long>(gFallCount));

    if (Barometer::ready()) {
        const BaroMath::AltitudeTracker &t = Barometer::tracker();
        if (baroAltitudeValid()) {
            Serial.printf(" | baro p=%.2fhPa h=%+.2fm dh=%+.2fm (peak %+.2f) vz=%+.2fm/s",
                          Barometer::pressureHpa(), t.altitudeM(), t.altitudeDiffM(),
                          gStatusAltDiffPeakM, t.verticalSpeedMps());
        } else {
            Serial.printf(" | baro p=%.2fhPa calibrating", Barometer::pressureHpa());
        }
        gStatusAltDiffPeakM = 0.0f;
    }
    Serial.println();
}

void publishWindow() {
    if (gWindowStats.count == 0) {
        return;
    }

    if (kPrintBanner) {
        printStatusLine();
    }
    gWindowStats.reset();

    const bool accelSlot = !baroTelemetryActive() || (gPublishSlot % 2 == 0);
    ++gPublishSlot;

    if (accelSlot) {
        publishValue(AIO_FEED_ACCEL_MAG, "%.4f", gAccelMagWindow.mean);
        gAccelMagWindow.reset();
    } else {
        publishBarometerSlot();
    }
}

void publishFallDetected(int value) {
    if (!gMqtt.connected()) {
        return;
    }
    if (gMqtt.publish(AIO_FEED_FALL_DETECTED, value ? "1" : "0")) {
        ++gPublishCount;
    }
}

void pollHostCommands() {
    static char line[48];
    static std::size_t len = 0;
    while (Serial.available() > 0) {
        const char c = static_cast<char>(Serial.read());
        if (c != '\n' && c != '\r') {
            if (len + 1 < sizeof(line)) {
                line[len++] = c;
            }
            continue;
        }
        line[len] = '\0';
        if (std::strcmp(line, "DBDUMP") == 0) {
            DeviceLog::dump(Serial);
        } else if (std::strcmp(line, "DBINFO") == 0) {
            DeviceLog::info(Serial);
        } else if (std::strcmp(line, "DBCLEAR") == 0) {
            DeviceLog::clear();
            Serial.println("#DBCLEAR-DONE");
        }
        len = 0;
    }
}

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
    ++gFallCount;

    logDeviceRow("fall_ml");
    gFallDetectedPublished = true;
    publishFallDetected(1);

    char incidentMsg[64];
    snprintf(incidentMsg, sizeof(incidentMsg), "FALL DETECTED (ML p=%.2f)",
             static_cast<double>(probability));
    logIncident(incidentMsg);
    gIncidentOpen = true;

    Serial.printf("*** FALL DETECTED (machine learning) #%lu  p=%.3f  alerting for %lu ms ***\n",
                  static_cast<unsigned long>(gFallCount),
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

}  // namespace

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

    const Barometer::Status baro = Barometer::begin(Serial);
    if (kPrintBanner) {
        if (baro.ready) {
            Serial.println("# barometer: altitude reference = mean pressure of the first 2 s, "
                           "keep still; GPIO3 (no open incident) re-zeroes");
        } else {
            Serial.println("# barometer: none - Pressure/Altitude columns stay empty");
        }
    }

    DeviceLog::begin(&Serial);

    gCloudActive = kEnableCloud && credentialsConfigured();

    if (kEnableCloud && !gCloudActive) {
        Serial.println("# Adafruit IO disabled: credentials are empty.");
        Serial.println("#   cp include/secrets.example.h include/secrets.h  (then fill it in)");
        Serial.println("# Local inference continues normally.");
    }

    if (gCloudActive) {
        WifiLink::beginFromSecrets(kPrintBanner ? &Serial : nullptr);
        if (kPrintBanner) {
            Serial.printf("# wifi mode: %s\n", WifiLink::modeName());
        }
        gMqtt.setServer(AIO_SERVER, AIO_PORT);
        gMqtt.setCallback(onMqttMessage);
        gMqtt.setKeepAlive(kMqttKeepAliveSec);
        gMqtt.setBufferSize(256);
        if (kPrintBanner) {
            if (baroTelemetryActive()) {
                Serial.printf("# publishing %s every %lu ms\n", AIO_FEED_ACCEL_MAG,
                              static_cast<unsigned long>(2 * kPublishIntervalMs));
                Serial.printf("# publishing %s / %s / %s in turn, each every %lu ms\n",
                              AIO_FEED_PRESSURE, AIO_FEED_ALTITUDE, AIO_FEED_ALTITUDE_DIFF,
                              static_cast<unsigned long>(6 * kPublishIntervalMs));
            } else {
                Serial.printf("# publishing %s every %lu ms\n", AIO_FEED_ACCEL_MAG,
                              static_cast<unsigned long>(kPublishIntervalMs));
            }
            Serial.printf("# publishing %s on detection (1) and on clear (0)\n",
                          AIO_FEED_FALL_DETECTED);
            Serial.printf("# publishing %s on fall and on dismissal\n", AIO_FEED_FALL_LOG);
            Serial.printf("# publishing %s = %s on connect, LWT %s\n", AIO_FEED_DEVICE_STATUS,
                          kStatusOnline, kStatusOffline);
            Serial.printf("# subscribed to %s\n", AIO_FEED_BUZZER_COMMAND);
        }
    }

    gWindow.clear();
    gNextSampleMs = millis();
    gLastPublishMs = millis();
}

void loop() {
    AlarmManager::update();
    pollHostCommands();

    if (AlarmManager::consumeDismissPress()) {
        gConsecutivePositives = 0;
        gLastAlertMs = millis();
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

        if (gIncidentOpen) {
            logIncident("DISMISSED BY USER");
            logDeviceRow("dismiss");
            gIncidentOpen = false;
        } else if (Barometer::ready()) {
            Barometer::rezero();
            if (kPrintBanner) {
                Serial.println("# barometer: re-zeroing altitude, keep still for 2 s");
            }
        }
    }

    if (gFallDetectedPublished && !AlarmManager::isAlerting()) {
        publishFallDetected(0);
        gFallDetectedPublished = false;
    }

    const std::uint32_t now = millis();
    const bool sampleDue = static_cast<std::int32_t>(now - gNextSampleMs) >= 0;

    if (sampleDue) {
        if (now - gNextSampleMs > 5 * kSamplePeriodMs) {
            gNextSampleMs = now + kSamplePeriodMs;
        } else {
            gNextSampleMs += kSamplePeriodMs;
        }

        if (SensorHub::status().imuReady && (!gMagRequired || SensorHub::status().magReady)) {
            SensorHub::Sample9 sample;
            if (SensorHub::read(sample)) {
                ++gSampleCount;

                const float ax = sample.v[SensorHub::kAx];
                const float ay = sample.v[SensorHub::kAy];
                const float az = sample.v[SensorHub::kAz];
                const float accelMag = std::sqrt(ax * ax + ay * ay + az * az);
                gWindowStats.add(accelMag);
                gAccelMagWindow.add(accelMag);
                gLogWindow.add(accelMag);

                gWindow.push(sample.v);

                if (gWindow.full()) {
                    static bool firstWindowScored = false;
                    if (!firstWindowScored) {
                        firstWindowScored = true;
                        gSamplesSinceInference = 0;
                        runInference();
                    } else if (++gSamplesSinceInference >= kHopSamples) {
                        gSamplesSinceInference = 0;
                        runInference();
                    }
                }
            } else {
                ++gMissedReads;
            }
        }
        return;
    }

    if (Barometer::service()) {
        onBaroSample();
    }

    if (DeviceLog::ready() && now - gLastLogMs >= kDeviceLogIntervalMs) {
        gLastLogMs = now;
        logDeviceRow("window");
        gLogWindow.reset();
    }

    if (!gCloudActive) {
        if (kPrintBanner && now - gLastPublishMs >= kPublishIntervalMs) {
            gLastPublishMs = now;
            if (gWindowStats.count > 0) {
                printStatusLine();
                gWindowStats.reset();
            }
            gAccelMagWindow.reset();
        }
        return;
    }

    WifiLink::update();
    serviceMqtt();
    flushEvents();

    if (now - gLastPublishMs >= kPublishIntervalMs) {
        gLastPublishMs = now;
        publishWindow();
    }

    const bool justConnected = WifiLink::consumeJustConnected();

    if (justConnected && !gTimeSyncStarted) {
        configTzTime(kTimezonePosix, kNtpServer1, kNtpServer2);
        gTimeSyncStarted = true;
        if (kPrintBanner) {
            Serial.printf("# time: SNTP started (%s), fall-log uses uptime until synced\n",
                          kTimezonePosix);
        }
    }

    if (kPrintBanner && justConnected) {
        Serial.printf("# wifi ip=%s rssi=%ld publishes=%lu missed_reads=%lu\n",
                      WifiLink::ipAddress().c_str(), static_cast<long>(WifiLink::rssi()),
                      static_cast<unsigned long>(gPublishCount),
                      static_cast<unsigned long>(gMissedReads));
    }
}
