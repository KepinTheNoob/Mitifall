// =============================================================================
//  wifi_link.h - non-blocking WPA2-PSK connection manager
// =============================================================================
//  Shared by both cloud targets. Deliberately knows nothing about MQTT or Blynk
//  so it can compile in every environment without pulling in their libraries.
//
//  The contract that matters for a safety device: begin() never blocks waiting
//  for an association, and update() returns immediately. Losing WiFi must never
//  stop sampling, inference, or the local alarm.
//
//  Only WPA2-Personal (a single pre-shared key) is supported: home routers and
//  phone hotspots. Networks needing a username and password per user
//  (WPA2-Enterprise / EAP, typical on campus) are not handled.
// =============================================================================

#pragma once

#include <Arduino.h>

#include <cstdint>

namespace WifiLink {

struct Options {
    std::uint32_t retryIntervalMs = 8000;  ///< gap between reconnect attempts
    bool disableSleep = true;              ///< keep latency low for MQTT/Blynk

    /// Scan once during begin() and report what is visible for the target SSID:
    /// channel, signal strength and the encryption in use. Blocks for ~2 s, which
    /// is harmless inside setup().
    bool scanOnBegin = true;

    /// Print the driver's disconnect reason code on every failed attempt. This is
    /// what distinguishes "wrong password" from "SSID not found".
    bool logDisconnectReason = true;
};

/// Start associating. Returns immediately; poll with update()/isConnected().
/// `password` may be empty for an open network.
void begin(const char *ssid, const char *password, Stream *log = nullptr,
           const Options &options = Options{});

/// Start using WIFI_SSID / WIFI_PASSWORD from include/secrets.h. Safe to call in
/// a build with no credentials compiled in: it logs and does nothing.
void beginFromSecrets(Stream *log = nullptr, const Options &options = Options{});

/// Drive reconnection. Call every loop iteration. Never blocks.
void update();

bool isConnected();

/// True once begin*() has been called with a usable SSID.
bool isEnabled();

/// Empty string until connected.
String ipAddress();

std::int32_t rssi();

/// Transitions to true exactly once per new connection, for one-shot logging.
bool consumeJustConnected();

std::uint32_t reconnectCount();

/// "WPA2-PSK" or "disabled", for status output.
const char *modeName();

/// Most recent driver disconnect reason code (0 = none seen yet).
std::uint8_t lastDisconnectReason();

/// Human-readable form of a disconnect reason code.
const char *disconnectReasonName(std::uint8_t reason);

/// Scan and print the networks matching `ssid` (blocking, ~2 s). Exposed so it
/// can be re-run from a diagnostic command.
void logVisibleNetworks(Stream &out, const char *ssid);

}  // namespace WifiLink
