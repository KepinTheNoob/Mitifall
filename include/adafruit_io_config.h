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

#include <cstddef>
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
#define AIO_FEED_FALL_DETECTED AIO_USERNAME "/feeds/fall-detected"
#define AIO_FEED_BUZZER_COMMAND AIO_USERNAME "/feeds/buzzer-command"

// Event-driven feeds. None of these is published on a timer.
#define AIO_FEED_FALL_TILT AIO_USERNAME "/feeds/fall-tilt"          // posture change, deg
#define AIO_FEED_FALL_LOG AIO_USERNAME "/feeds/fall-log"            // timestamped incidents
#define AIO_FEED_DEVICE_STATUS AIO_USERNAME "/feeds/device-status"  // ONLINE / OFFLINE

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
//   accel-mag      every 2.5 s          -> 24 points/min
//   fall-detected  on detection only    ->  2 per fall (1, then 0)
//   fall-tilt      on detection only    ->  1 per fall
//   fall-log       on detection/dismiss ->  1 per fall + 1 per dismissal
//   device-status  on (re)connect only  ->  1 per MQTT connection
//                                        ---------------
//                                          24 points/min + ~5 per incident
//
// Only accel-mag is periodic. A second feed published on the same 2.5 s cadence
// would take the total to 48 points/min and get throttled, which is why every
// other feed is edge-triggered. A fall costs 4 points and a dismissal 2 (fall-log
// plus the existing buzzer-command reset), leaving room for one incident a minute.
constexpr std::uint32_t kPublishIntervalMs = 2500;

// ---------------------------------------------------------------------------
// device-status lifecycle
// ---------------------------------------------------------------------------
// The Last Will is registered on every CONNECT: QoS 1, retained, "OFFLINE".
//
// PLATFORM LIMITATION: the Adafruit IO broker does not support Last Will and
// does not honour the retain flag (Adafruit IO FAQ / MQTT API docs). The will is
// therefore never published by io.adafruit.com and the feed keeps showing
// "ONLINE" after the device dies. The will is still registered so the firmware
// behaves correctly on a standards-compliant broker; for offline detection on
// Adafruit IO, use a feed notification on accel-mag (see README).
constexpr std::uint8_t kWillQos = 1;
constexpr bool kWillRetain = true;
constexpr char kStatusOnline[] = "ONLINE";
constexpr char kStatusOffline[] = "OFFLINE";

// If a CONNECT carrying a will keeps failing while WiFi is up, try one plain
// CONNECT. If that succeeds, the broker is refusing the will and it is dropped
// for the rest of the session - so the LWT can never take MQTT down entirely.
constexpr std::uint8_t kWillFailuresBeforeFallback = 2;

// ---------------------------------------------------------------------------
// fall-log timestamps
// ---------------------------------------------------------------------------
// Wall-clock time comes from SNTP, started once WiFi first connects; it never
// blocks. Until the clock is set, entries carry device uptime instead.
// POSIX TZ string: WIB = UTC+7 (Western Indonesia Time).
constexpr char kTimezonePosix[] = "WIB-7";
constexpr char kNtpServer1[] = "pool.ntp.org";
constexpr char kNtpServer2[] = "time.google.com";

// Events raised while MQTT is down are held and published on reconnect, stamped
// with the time they happened. Oldest entries are dropped if this fills.
constexpr std::size_t kEventQueueCapacity = 8;

// MQTT reconnect backoff (non-blocking; the sampling loop keeps running).
constexpr std::uint32_t kMqttRetryIntervalMs = 5000;
constexpr std::uint16_t kMqttKeepAliveSec = 60;

// ---------------------------------------------------------------------------
// Local moving/static hint
// ---------------------------------------------------------------------------
// Not published any more - used only for the "-> MOVING / static" annotation on
// the serial status line, as a sanity check that the accelerometer is alive.
//
// "Moving" is the standard deviation of |a| over the publish window exceeding
// this cut. Measured on the UMAFall wrist data, a 0.10 g cut flags 80 % of
// dynamic-activity windows (walking, jogging, stairs) and 49 % of quiet ones
// (lying down, phone call, sitting), so it was never a good enough signal to be
// worth a dashboard widget.
constexpr float kMotionStdThresholdG = 0.10f;

}  // namespace AdafruitIoConfig
