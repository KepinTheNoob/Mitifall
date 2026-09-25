// =============================================================================
//  blynk_config.h - Blynk IoT settings for [env:ml_inference]
// =============================================================================
//  NO CREDENTIALS IN THIS FILE. It is tracked in git and holds only non-secret
//  configuration: the virtual pin map, push cadence and connection timeouts.
//
//  Credentials come from include/secrets.h (git-ignored) or from -D build flags:
//      cp include/secrets.example.h include/secrets.h   # then fill it in
//
//  Required macros: WIFI_SSID, WIFI_PASSWORD, BLYNK_TEMPLATE_ID,
//  BLYNK_TEMPLATE_NAME, BLYNK_AUTH_TOKEN.
//
//  MUST be included before <BlynkSimpleEsp32.h>: the Blynk library reads
//  BLYNK_TEMPLATE_ID / BLYNK_TEMPLATE_NAME at preprocessor time.
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

#ifndef BLYNK_TEMPLATE_ID
#error "BLYNK_TEMPLATE_ID undefined. See include/secrets.example.h."
#endif

#ifndef BLYNK_TEMPLATE_NAME
#error "BLYNK_TEMPLATE_NAME undefined. See include/secrets.example.h."
#endif

#ifndef BLYNK_AUTH_TOKEN
#error "BLYNK_AUTH_TOKEN undefined. See include/secrets.example.h."
#endif

// Route the library's diagnostics to the USB CDC console (not secret).
#ifndef BLYNK_PRINT
#define BLYNK_PRINT Serial
#endif

// ---------------------------------------------------------------------------
// Virtual pin map (must match the datastreams configured in Blynk Console)
// ---------------------------------------------------------------------------
//   V0  Double   fall probability 0.0 .. 1.0   -> SuperChart / line chart
//   V1  Integer  alarm status 0 or 1           -> LED / status indicator
//   V2  Integer  button from the dashboard     -> dismiss alarm / test actuators
//
// BLYNK_WRITE() needs the bare token, so the handler in inference_main.cpp is
// written as BLYNK_WRITE(V2); these aliases are for the virtualWrite calls.
#define VPIN_FALL_PROBABILITY V0
#define VPIN_ALARM_STATUS V1
#define VPIN_DISMISS_BUTTON V2

namespace BlynkConfig {

/// True when secrets.h has actually been filled in.
///
/// An empty SSID or token is unambiguous, so the firmware can skip cloud setup
/// and run offline instead of retrying a connection that cannot succeed.
inline bool credentialsConfigured() {
    return WIFI_SSID[0] != '\0' && BLYNK_AUTH_TOKEN[0] != '\0' && BLYNK_TEMPLATE_ID[0] != '\0';
}

// Telemetry cadence. BlynkTimer drives these, so the 50 Hz acquisition loop is
// never blocked waiting on the network.
constexpr std::uint32_t kProbabilityPushMs = 1000;  // one point per inference hop
constexpr std::uint32_t kStatusPushMs = 2000;       // heartbeat for V1

// Non-blocking connection management. Blynk.connect() is only called from the
// throttled reconnect path with a bounded timeout, never from the sample path.
constexpr std::uint32_t kConnectRetryIntervalMs = 5000;
constexpr std::uint32_t kConnectTimeoutMs = 1500;

}  // namespace BlynkConfig
