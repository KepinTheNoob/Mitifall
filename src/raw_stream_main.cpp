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
//    publish   {user}/feeds/motion-status    1 moving / 0 static        -> LED
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
#include "wifi_link.h"

namespace {

using namespace BoardConfig;
using namespace AdafruitIoConfig;

// Set false for a pure CSV capture with no '#' preamble or status lines.
constexpr bool kPrintBanner = true;

// Set false to run purely local (no WiFi, no MQTT), e.g. for bench logging.
constexpr bool kEnableCloud = true;

WiFiClient gTcp;
PubSubClient gMqtt(gTcp);

std::uint32_t gNextSampleMs = 0;
std::uint32_t gSampleCount = 0;
std::uint32_t gMissedReads = 0;

std::uint32_t gLastPublishMs = 0;
std::uint32_t gLastMotionPublishMs = 0;
std::uint32_t gLastMqttAttemptMs = 0;
std::uint32_t gPublishCount = 0;
int gLastMotionState = -1;  // -1 = never published

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
        Serial.printf("# window n=%lu |a|mean=%.3f |a|max=%.3f |a|std=%.3f -> %s\n",
                      static_cast<unsigned long>(gWindow.count), meanMag, gWindow.max, stdMag,
                      moving ? "MOVING" : "static");
    }

    if (gMqtt.connected()) {
        char payload[16];

        snprintf(payload, sizeof(payload), "%.4f", meanMag);
        if (gMqtt.publish(AIO_FEED_ACCEL_MAG, payload)) {
            ++gPublishCount;
        }

        // Edge-triggered, rate-limited: see the budget note in the config header.
        const int state = moving ? 1 : 0;
        const bool changed = state != gLastMotionState;
        const bool spaced = millis() - gLastMotionPublishMs >= kMotionMinIntervalMs;
        if (changed && (spaced || gLastMotionState < 0)) {
            snprintf(payload, sizeof(payload), "%d", state);
            if (gMqtt.publish(AIO_FEED_MOTION_STATUS, payload)) {
                gLastMotionState = state;
                gLastMotionPublishMs = millis();
                ++gPublishCount;
            }
        }
    }

    gWindow.reset();
}

}  // namespace

void setup() {
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
        }
    }

    Serial.println("Timestamp_ms,Ax,Ay,Az,Gx,Gy,Gz,Mx,My,Mz");
    gNextSampleMs = millis();
    gLastPublishMs = millis();
}

void loop() {
    AlarmManager::update();

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
            gWindow.add(std::sqrt(ax * ax + ay * ay + az * az));

            Serial.printf("%lu,%.5f,%.5f,%.5f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f\n",
                          static_cast<unsigned long>(sample.timestampMs), ax, ay, az,
                          sample.v[SensorHub::kGx], sample.v[SensorHub::kGy],
                          sample.v[SensorHub::kGz], sample.v[SensorHub::kMx],
                          sample.v[SensorHub::kMy], sample.v[SensorHub::kMz]);
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
