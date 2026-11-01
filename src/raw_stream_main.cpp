#include <Arduino.h>
#include <PubSubClient.h>
#include <WiFi.h>

#include <cmath>
#include <cstring>
#include <ctime>

#include "adafruit_io_config.h"
#include "alarm_manager.h"
#include "barometer.h"
#include "board_config.h"
#include "device_log.h"
#include "sensor_hub.h"
#include "threshold_detector.h"
#include "wifi_link.h"

namespace {

using namespace BoardConfig;
using namespace AdafruitIoConfig;

constexpr bool kPrintBanner = true;

#ifdef MITIFALL_DL_STREAM
constexpr bool kDlStream = true;
#else
constexpr bool kDlStream = false;
#endif

constexpr bool kStreamCsvInRawStream = false;
constexpr bool kStreamCsv = kDlStream || kStreamCsvInRawStream;

constexpr bool kEnableCloud = true;

constexpr bool kEnableFallDetection = !kDlStream;

constexpr std::uint32_t kDeviceLogIntervalMs = 2500;

constexpr const char *kCsvHeader =
    "Timestamp_ms,Ax,Ay,Az,Gx,Gy,Gz,Mx,My,Mz,Pressure_hPa,Altitude_m,AltitudeDiff_m";

constexpr bool kLogCandidates = true;

constexpr int kSensitivityPreset = 3;

ThresholdDetector::Detector gDetector;

std::uint32_t gImpactCount = 0;
std::uint32_t gRejectCount = 0;
float gPeakAccelG = 0.0f;
float gSessionPeakG = 0.0f;

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

struct PendingEvent {
    const char *topic;
    char payload[64];
};

PendingEvent gEventQueue[kEventQueueCapacity];
std::size_t gEventHead = 0;
std::size_t gEventCount = 0;
std::uint32_t gEventsDropped = 0;

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

void reportFallTilt(float tiltDeg) {
    char value[16];
    snprintf(value, sizeof(value), "%.1f", tiltDeg);
    queueEvent(AIO_FEED_FALL_TILT, value);
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

void printStatusLine() {
    const float stdMag = gWindow.stddev();
    const bool moving = stdMag > kMotionStdThresholdG;

    Serial.printf("# window n=%lu |a|mean=%.3f |a|max=%.3f |a|std=%.3f -> %s | "
                  "det=%s peak=%.2fg (need %.2fg) impacts=%lu rejected=%lu falls=%lu",
                  static_cast<unsigned long>(gWindow.count), gWindow.mean, gWindow.max, stdMag,
                  moving ? "MOVING" : "static",
                  ThresholdDetector::Detector::stateName(gDetector.state()), gPeakAccelG,
                  gDetector.config().impactG, static_cast<unsigned long>(gImpactCount),
                  static_cast<unsigned long>(gRejectCount),
                  static_cast<unsigned long>(gFallCount));
    gPeakAccelG = 0.0f;

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

void publishWindow() {
    if (gWindow.count == 0) {
        return;
    }

    if (kPrintBanner) {
        printStatusLine();
    }
    gWindow.reset();

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

void applySensitivityPreset(int preset) {
    auto &cfg = gDetector.config();
    switch (preset) {
        case 1:
            cfg.impactG = 2.6f;
            cfg.stillStdG = 0.50f;
            break;
        case 2:
            cfg.impactG = 3.0f;
            cfg.stillStdG = 0.32f;
            break;
        case 3:
            cfg.impactG = 1.8f;
            cfg.stillStdG = 0.60f;
            cfg.orientationChangeDeg = 15.0f;
            break;
        default:
            break;
    }
}

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

    feed(0.0f, 0.0f, 1.0f, 0.0f, 2000);
    feed(0.0f, 0.0f, 4.5f, 200.0f, 60);
    feed(0.95f, 0.0f, 0.1f, 0.0f, 2600);

    Serial.printf("# detector self-test: %s (synthetic fall, tilt=%.0fdeg std=%.3fg)\n",
                  fired ? "PASS" : "FAIL", probe.lastOrientationChangeDeg(),
                  probe.lastStillStdG());
    if (!fired) {
        Serial.println("#   thresholds are unreachable - lower impactG or raise stillStdG");
    }
}

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

            AlarmManager::trigger();
            logDeviceRow("fall");
            gFallDetectedPublished = true;
            publishFallDetected(1);

            reportFallTilt(gDetector.lastOrientationChangeDeg());
            logIncident("FALL DETECTED");
            gIncidentOpen = true;

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

void raiseHostFall(const char *details) {
    ++gFallCount;
    AlarmManager::trigger();
    logDeviceRow("fall_dl");
    gFallDetectedPublished = true;
    publishFallDetected(1);
    logIncident("FALL DETECTED (DL)");
    gIncidentOpen = true;
    Serial.printf("*** FALL DETECTED (deep learning, host) #%lu %s ***\n",
                  static_cast<unsigned long>(gFallCount), details);
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
        if (kDlStream && std::strncmp(line, "FALL", 4) == 0) {
            raiseHostFall(line + 4);
        } else if (std::strcmp(line, "HEADER") == 0) {
            Serial.println(kCsvHeader);
        } else if (std::strcmp(line, "DBDUMP") == 0) {
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

}

void setup() {

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
            Serial.printf("# publishing %s once per fall (tilt, deg)\n", AIO_FEED_FALL_TILT);
            Serial.printf("# publishing %s on fall and on dismissal\n", AIO_FEED_FALL_LOG);
            Serial.printf("# publishing %s = %s on connect, LWT %s\n", AIO_FEED_DEVICE_STATUS,
                          kStatusOnline, kStatusOffline);
            Serial.printf("# subscribed to %s\n", AIO_FEED_BUZZER_COMMAND);
        }
    }

    if (kDlStream) {
        Serial.println("# fall detection: deep learning model on the host ONLY "
                       "(python deep_learning/infer_stream.py --port ...)");
        Serial.println("#   rule-based detector disabled; no host = no fall detection");
    }
    if (kStreamCsv) {
        Serial.println(kCsvHeader);
    } else if (kPrintBanner) {
        Serial.println("# CSV rows off (kStreamCsv = false in raw_stream_main.cpp)");
    }
    gNextSampleMs = millis();
    gLastPublishMs = millis();
}

void loop() {
    AlarmManager::update();
    pollHostCommands();

    if (AlarmManager::consumeDismissPress()) {

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

        SensorHub::Sample9 sample;
        if (SensorHub::read(sample)) {
            ++gSampleCount;

            const float ax = sample.v[SensorHub::kAx];
            const float ay = sample.v[SensorHub::kAy];
            const float az = sample.v[SensorHub::kAz];
            const float accelMag = std::sqrt(ax * ax + ay * ay + az * az);
            gWindow.add(accelMag);
            gAccelMagWindow.add(accelMag);
            gLogWindow.add(accelMag);
            if (accelMag > gPeakAccelG) {
                gPeakAccelG = accelMag;
            }
            if (accelMag > gSessionPeakG) {
                gSessionPeakG = accelMag;
            }

            if (kEnableFallDetection) {
                runFallDetection(sample);
            }

            if (kStreamCsv) {

                char baro[48] = ",,,";
                if (Barometer::ready() && Barometer::tracker().sampleCount() > 0) {
                    if (baroAltitudeValid()) {
                        const BaroMath::AltitudeTracker &t = Barometer::tracker();
                        snprintf(baro, sizeof(baro), ",%.3f,%.3f,%.3f", Barometer::pressureHpa(),
                                 t.altitudeM(), t.altitudeDiffM());
                    } else {
                        snprintf(baro, sizeof(baro), ",%.3f,,", Barometer::pressureHpa());
                    }
                }
                Serial.printf("%lu,%.5f,%.5f,%.5f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f%s\n",
                              static_cast<unsigned long>(sample.timestampMs), ax, ay, az,
                              sample.v[SensorHub::kGx], sample.v[SensorHub::kGy],
                              sample.v[SensorHub::kGz], sample.v[SensorHub::kMx],
                              sample.v[SensorHub::kMy], sample.v[SensorHub::kMz], baro);
            }
        } else {
            ++gMissedReads;
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
            if (gWindow.count > 0) {
                printStatusLine();
                gWindow.reset();
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
