#include "wifi_link.h"

#include <WiFi.h>

#if __has_include("secrets.h")
#include "secrets.h"
#endif

namespace WifiLink {
namespace {

enum class Mode : std::uint8_t { Disabled, Psk };

const char *gSsid = nullptr;
const char *gPassword = nullptr;
Mode gMode = Mode::Disabled;

Stream *gLog = nullptr;
Options gOptions;

bool gWasConnected = false;
bool gJustConnected = false;
std::uint32_t gLastAttemptMs = 0;
std::uint32_t gReconnects = 0;
std::uint32_t gAttempts = 0;

volatile std::uint8_t gLastReason = 0;
volatile bool gReasonPending = false;
bool gEventHooked = false;

// One failure line per attempt: the driver emits several disconnect events per
// association attempt and the log becomes unreadable otherwise.
std::uint32_t gReportedFailureForAttempt = 0;

// Strongest matching AP seen during the startup scan. Associating directly to a
// known BSSID and channel skips the full-channel scan, which is markedly more
// reliable in a congested band.
bool gHaveBssid = false;
std::uint8_t gBssid[6] = {0};
std::int32_t gChannel = 0;

bool empty(const char *text) { return text == nullptr || text[0] == '\0'; }

/// True when the configured network is open (no passphrase).
bool isOpenNetwork() { return empty(gPassword); }

void onWifiEvent(arduino_event_id_t event, arduino_event_info_t info) {
    if (event == ARDUINO_EVENT_WIFI_STA_DISCONNECTED) {
        gLastReason = info.wifi_sta_disconnected.reason;
        gReasonPending = true;
    }
}

const char *authModeName(wifi_auth_mode_t mode) {
    switch (mode) {
        case WIFI_AUTH_OPEN:
            return "OPEN (no encryption)";
        case WIFI_AUTH_WEP:
            return "WEP";
        case WIFI_AUTH_WPA_PSK:
            return "WPA-PSK";
        case WIFI_AUTH_WPA2_PSK:
            return "WPA2-PSK";
        case WIFI_AUTH_WPA_WPA2_PSK:
            return "WPA/WPA2-PSK";
        case WIFI_AUTH_WPA2_ENTERPRISE:
            return "WPA2-Enterprise (EAP) - NOT supported by this firmware";
        case WIFI_AUTH_WPA3_PSK:
            return "WPA3-PSK";
        case WIFI_AUTH_WPA2_WPA3_PSK:
            return "WPA2/WPA3-PSK";
        default:
            return "unknown";
    }
}

void prepareRadio() {
    WiFi.persistent(false);
    WiFi.mode(WIFI_STA);
    if (!gEventHooked) {
        WiFi.onEvent(onWifiEvent);
        gEventHooked = true;
    }
    // Let update() own all retry timing. With the driver's own auto-reconnect
    // enabled as well, the two fight: it re-associates underneath us while we are
    // issuing begin(), which shows up as a burst of AUTH_EXPIRE (reason 2) events
    // for a single logical attempt.
    WiFi.setAutoReconnect(false);
    if (gOptions.disableSleep) {
        // Modem sleep adds up to ~100 ms of latency to every publish; the power
        // saving is not worth it while streaming telemetry.
        WiFi.setSleep(false);
    }
}

void attempt() {
    gLastAttemptMs = millis();
    ++gAttempts;
    if (gMode != Mode::Psk) {
        return;
    }
    WiFi.disconnect(false, false);
    WiFi.begin(gSsid, gPassword);
    WiFi.setTxPower(WIFI_POWER_8_5dBm);
    if (gLog != nullptr) {
        gLog->printf("WiFi: associating with \"%s\" (WPA2-PSK, attempt %lu)\n", gSsid,
                     static_cast<unsigned long>(gAttempts));
    }
}

}  // namespace

const char *modeName() { return gMode == Mode::Psk ? "WPA2-PSK" : "disabled"; }

std::uint8_t lastDisconnectReason() { return gLastReason; }

const char *disconnectReasonName(std::uint8_t reason) {
    // Values from esp_wifi_types.h (wifi_err_reason_t). Only the ones that
    // actually show up during bring-up are named individually.
    switch (reason) {
        case 1:
            return "unspecified";
        case 2:
            return "previous auth no longer valid - wrong password?";
        case 4:
            return "disassociated due to inactivity";
        case 8:
            return "deauth, station leaving";
        case 15:
            return "4-way handshake timeout - WRONG PASSWORD";
        case 23:
            return "802.1X auth failed - this network needs EAP, not a shared key";
        case 200:
            return "beacon timeout - AP out of range";
        case 201:
            return "no AP found - SSID not visible on 2.4 GHz";
        case 202:
            return "auth failed";
        case 203:
            return "assoc failed";
        case 204:
            return "handshake timeout";
        case 205:
            return "connection failed";
        default:
            return "see wifi_err_reason_t in esp_wifi_types.h";
    }
}

void logVisibleNetworks(Stream &out, const char *ssid) {
    out.printf("WiFi: scanning for \"%s\"...\n", ssid == nullptr ? "" : ssid);
    const int found = WiFi.scanNetworks();
    if (found <= 0) {
        out.println("WiFi:   no networks visible at all - check the antenna/band");
        WiFi.scanDelete();
        return;
    }

    int matches = 0;
    for (int i = 0; i < found; ++i) {
        if (ssid != nullptr && WiFi.SSID(i) != ssid) {
            continue;
        }
        ++matches;
        out.printf("WiFi:   match ch=%d rssi=%d dBm auth=%s\n", WiFi.channel(i), WiFi.RSSI(i),
                   authModeName(WiFi.encryptionType(i)));
    }
    out.printf("WiFi:   %d network(s) visible, %d matching the configured SSID\n", found, matches);
    if (matches == 0) {
        out.println("WiFi:   the SSID is NOT in range on 2.4 GHz. The ESP32-C3 has no 5 GHz");
        out.println("WiFi:   radio, so a 5 GHz-only AP is invisible to it.");
    }
    WiFi.scanDelete();
}

void begin(const char *ssid, const char *password, Stream *log, const Options &options) {
    gLog = log;
    gOptions = options;

    if (empty(ssid)) {
        gMode = Mode::Disabled;
        if (gLog != nullptr) {
            gLog->println("WiFi: no SSID configured, staying offline");
        }
        return;
    }

    gSsid = ssid;
    gPassword = password;
    gMode = Mode::Psk;

    prepareRadio();
    if (gOptions.scanOnBegin && gLog != nullptr) {
        logVisibleNetworks(*gLog, gSsid);
    }
    attempt();
}

void beginFromSecrets(Stream *log, const Options &options) {
#if defined(WIFI_SSID)
    begin(WIFI_SSID, WIFI_PASSWORD, log, options);
#else
    (void)options;
    gMode = Mode::Disabled;
    if (log != nullptr) {
        log->println("WiFi: no credentials compiled in (include/secrets.h missing)");
    }
#endif
}

void update() {
    if (gMode == Mode::Disabled) {
        return;
    }

    const bool connected = WiFi.status() == WL_CONNECTED;

    if (connected && !gWasConnected) {
        gJustConnected = true;
        ++gReconnects;
        if (gLog != nullptr) {
            gLog->printf("WiFi: connected, ip=%s rssi=%ld dBm\n", WiFi.localIP().toString().c_str(),
                         static_cast<long>(WiFi.RSSI()));
        }
    } else if (!connected && gWasConnected && gLog != nullptr) {
        gLog->println("WiFi: connection lost");
    }
    gWasConnected = connected;

    // Surface the driver's reason code: this is what separates "wrong password"
    // (15) from "SSID not found" (201).
    if (gReasonPending) {
        gReasonPending = false;
        const std::uint8_t reason = gLastReason;
        if (gOptions.logDisconnectReason && gLog != nullptr && !connected) {
            gLog->printf("WiFi: attempt %lu failed, reason %u (%s)\n",
                         static_cast<unsigned long>(gAttempts), static_cast<unsigned>(reason),
                         disconnectReasonName(reason));
        }
    }

    if (!connected && millis() - gLastAttemptMs >= gOptions.retryIntervalMs) {
        attempt();
    }
}

bool isConnected() { return gMode != Mode::Disabled && WiFi.status() == WL_CONNECTED; }

bool isEnabled() { return gMode != Mode::Disabled; }

String ipAddress() { return isConnected() ? WiFi.localIP().toString() : String(); }

std::int32_t rssi() { return isConnected() ? WiFi.RSSI() : 0; }

bool consumeJustConnected() {
    const bool value = gJustConnected;
    gJustConnected = false;
    return value;
}

std::uint32_t reconnectCount() { return gReconnects; }

}  // namespace WifiLink
