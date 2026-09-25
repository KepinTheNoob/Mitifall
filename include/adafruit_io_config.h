// =============================================================================
//  adafruit_io_config.h - Adafruit IO MQTT settings for [env:raw_stream]
// =============================================================================
//  NO CREDENTIALS IN THIS FILE. It is tracked in git and holds only non-secret
//  configuration: broker host/port, feed topic composition, publish cadence and
//  the motion threshold.
//
//  Credentials come from include/secrets.h (git-ignored) or from -D build flags:
//      cp include/secrets.example.h include/secrets.h   # then fill it in
//
//  Required macros: WIFI_SSID, WIFI_PASSWORD, AIO_USERNAME, AIO_KEY.
// =============================================================================

#pragma once

#include <cstdint>

#if __has_include("secrets.h")
#include "secrets.h"
#endif

// ---------------------------------------------------------------------------
// Required credentials - fail at compile time with a usable message rather than
// falling back to placeholders that silently never connect.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Broker (not secret)
// ---------------------------------------------------------------------------
#ifndef AIO_SERVER
#define AIO_SERVER "io.adafruit.com"
#endif

// 1883 = plain MQTT. 8883 would need WiFiClientSecure plus a CA bundle; plain
// MQTT keeps the footprint small, but the AIO key crosses the network in the
// clear, so treat this as lab-grade rather than production.
#ifndef AIO_PORT
#define AIO_PORT 1883
#endif

// ---------------------------------------------------------------------------
// Feed topics - composed from AIO_USERNAME, so they follow whatever account
// secrets.h configures.
// ---------------------------------------------------------------------------
#define AIO_FEED_ACCEL_MAG AIO_USERNAME "/feeds/accel-mag"
#define AIO_FEED_MOTION_STATUS AIO_USERNAME "/feeds/motion-status"
#define AIO_FEED_BUZZER_COMMAND AIO_USERNAME "/feeds/buzzer-command"

// Adafruit IO publishes rate-limit warnings here; subscribing turns a silent
// throttle into a visible log line.
#define AIO_TOPIC_THROTTLE AIO_USERNAME "/throttle"
#define AIO_TOPIC_ERRORS AIO_USERNAME "/errors"

namespace AdafruitIoConfig {

/// True when secrets.h has actually been filled in.
///
/// An empty SSID or key is unambiguous - nobody has a zero-length SSID - so the
/// firmware can skip cloud setup and run offline instead of retrying a
/// connection that cannot succeed.
inline bool credentialsConfigured() {
    return WIFI_SSID[0] != '\0' && AIO_USERNAME[0] != '\0' && AIO_KEY[0] != '\0';
}

// ---------------------------------------------------------------------------
// Rate budget
// ---------------------------------------------------------------------------
// The free tier allows 30 data points per minute across ALL feeds, so the
// budget has to be counted per feed, not per publish call:
//
//   accel-mag     every 2.5 s            -> 24 points/min
//   motion-status on change, >=15 s apart ->  4 points/min worst case
//                                          ---------------
//                                            28 points/min
//
// Publishing both feeds every 2.5 s would be 48 points/min and would get
// throttled, which is why motion-status is edge-triggered instead of periodic.
constexpr std::uint32_t kPublishIntervalMs = 2500;
constexpr std::uint32_t kMotionMinIntervalMs = 15000;

// MQTT reconnect backoff (non-blocking; the sampling loop keeps running).
constexpr std::uint32_t kMqttRetryIntervalMs = 5000;
constexpr std::uint16_t kMqttKeepAliveSec = 60;

// ---------------------------------------------------------------------------
// Motion indicator
// ---------------------------------------------------------------------------
// "Moving" is the standard deviation of |a| over the publish window exceeding
// this cut. Measured on the UMAFall wrist data, a 0.10 g cut flags 80 % of
// dynamic-activity windows (walking, jogging, stairs) and 49 % of quiet ones
// (lying down, phone call, sitting). The overlap is inherent: wrist-worn
// "quiet" activities still swing the arm. Treat this as a coarse activity
// light, not a classifier.
constexpr float kMotionStdThresholdG = 0.10f;

}  // namespace AdafruitIoConfig
