#pragma once

#include <cstddef>
#include <cstdint>

#if __has_include("secrets.h")
#include "secrets.h"
#endif

#ifndef WIFI_SSID
#error "WIFI_SSID undefined. Run: cp include/secrets.example.h include/secrets.h -- then fill it in (or pass the value as a -D build flag)."
#endif

#ifndef WIFI_PASSWORD
#error "WIFI_PASSWORD undefined. See include/secrets.example.h."
#endif

#ifndef AIO_USERNAME
#error "AIO_USERNAME undefined. See include/secrets.example.h."
#endif

#ifndef AIO_KEY
#error "AIO_KEY undefined. See include/secrets.example.h."
#endif

#ifndef AIO_SERVER
#define AIO_SERVER "io.adafruit.com"
#endif

#ifndef AIO_PORT
#define AIO_PORT 1883
#endif

#define AIO_FEED_ACCEL_MAG AIO_USERNAME "/feeds/accel-mag"
#define AIO_FEED_FALL_DETECTED AIO_USERNAME "/feeds/fall-detected"
#define AIO_FEED_BUZZER_COMMAND AIO_USERNAME "/feeds/buzzer-command"

#define AIO_FEED_PRESSURE AIO_USERNAME "/feeds/pressure"
#define AIO_FEED_ALTITUDE AIO_USERNAME "/feeds/altitude"
#define AIO_FEED_ALTITUDE_DIFF AIO_USERNAME "/feeds/altitude-diff"

#define AIO_FEED_FALL_TILT AIO_USERNAME "/feeds/fall-tilt"
#define AIO_FEED_FALL_LOG AIO_USERNAME "/feeds/fall-log"
#define AIO_FEED_DEVICE_STATUS AIO_USERNAME "/feeds/device-status"

#define AIO_TOPIC_THROTTLE AIO_USERNAME "/throttle"
#define AIO_TOPIC_ERRORS AIO_USERNAME "/errors"

namespace AdafruitIoConfig {

inline bool credentialsConfigured() {
    return WIFI_SSID[0] != '\0' && AIO_USERNAME[0] != '\0' && AIO_KEY[0] != '\0';
}

constexpr std::uint32_t kPublishIntervalMs = 2500;

constexpr bool kPublishBarometer = true;

constexpr std::uint8_t kWillQos = 1;
constexpr bool kWillRetain = true;
constexpr char kStatusOnline[] = "ONLINE";
constexpr char kStatusOffline[] = "OFFLINE";

constexpr std::uint8_t kWillFailuresBeforeFallback = 2;

constexpr char kTimezonePosix[] = "WIB-7";
constexpr char kNtpServer1[] = "pool.ntp.org";
constexpr char kNtpServer2[] = "time.google.com";

constexpr std::size_t kEventQueueCapacity = 8;

constexpr std::uint32_t kMqttRetryIntervalMs = 5000;
constexpr std::uint16_t kMqttKeepAliveSec = 60;

constexpr float kMotionStdThresholdG = 0.10f;

}
