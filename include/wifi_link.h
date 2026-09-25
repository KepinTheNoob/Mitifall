// =============================================================================
//  wifi_link.h - non-blocking WiFi connection manager (WPA2-PSK + Enterprise)
// =============================================================================
//  Shared by both cloud targets. Deliberately knows nothing about MQTT or Blynk
//  so it can compile in every environment without pulling in their libraries.
//
//  The contract that matters for a safety device: begin() never blocks waiting
//  for an association, and update() returns immediately. Losing WiFi must never
//  stop sampling, inference, or the local alarm.
//
//  Two association modes:
//    * WPA2-Personal   ssid + pre-shared key (home / hotspot)
//    * WPA2-Enterprise ssid + EAP identity/username/password (campus, eduroam)
//
//  Pick the mode once in include/secrets.h and call beginFromSecrets().
// =============================================================================

#pragma once

#include <Arduino.h>

#include <cstdint>

namespace WifiLink {

// EAP outer methods, matching the Arduino wpa2_auth_method_t values. Use these
// numbers for WIFI_EAP_METHOD in secrets.h.
constexpr int kEapTls = 0;   ///< certificate based; needs client cert + key
constexpr int kEapPeap = 1;  ///< PEAP-MSCHAPv2 - what most campuses use
constexpr int kEapTtls = 2;  ///< TTLS, phase 2 forced to MSCHAPv2

struct EapCredentials {
    /// Outer identity sent in the clear. If null/empty, `username` is reused.
    const char *identity = nullptr;
    const char *username = nullptr;
    const char *password = nullptr;
    int method = kEapPeap;

    /// PEM of the RADIUS server's CA. Optional, but without it the supplicant
    /// cannot verify the authentication server - see the note in the .cpp.
    const char *caPem = nullptr;

    /// Only for kEapTls.
    const char *clientCertPem = nullptr;
    const char *clientKeyPem = nullptr;
};

struct Options {
    std::uint32_t retryIntervalMs = 8000;  ///< gap between reconnect attempts
    bool disableSleep = true;              ///< keep latency low for MQTT/Blynk
};

/// WPA2-Personal. Returns immediately; poll with update()/isConnected().
void begin(const char *ssid, const char *password, Stream *log = nullptr,
           const Options &options = Options{});

/// WPA2-Enterprise. Credential strings must outlive the call (string literals
/// from secrets.h do). Returns immediately.
void beginEnterprise(const char *ssid, const EapCredentials &credentials,
                     Stream *log = nullptr, const Options &options = Options{});

/// Start using whatever include/secrets.h defines, choosing Enterprise when
/// WIFI_EAP_ENABLED is 1 and PSK otherwise. Safe to call in a build with no
/// credentials compiled in: it logs and does nothing.
void beginFromSecrets(Stream *log = nullptr, const Options &options = Options{});

/// Drive reconnection. Call every loop iteration. Never blocks.
void update();

bool isConnected();

/// True once begin*() has been called with usable credentials.
bool isEnabled();

/// Empty string until connected.
String ipAddress();

std::int32_t rssi();

/// Transitions to true exactly once per new connection, for one-shot logging.
bool consumeJustConnected();

std::uint32_t reconnectCount();

/// "WPA2-PSK", "WPA2-Enterprise (PEAP)", ... for status output.
const char *modeName();

}  // namespace WifiLink
