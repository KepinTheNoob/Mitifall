#include "wifi_link.h"

#include <WiFi.h>

#include <cstring>

#include "esp_wifi.h"

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

// Radio region. Indonesia, like most of the world outside North America, allows
// 2.4 GHz channels 1-13, and phone hotspots frequently auto-select 12 or 13. The
// policy is MANUAL so the setting cannot be narrowed by country info advertised
// by some other AP.
constexpr char kCountryCode[] = "ID";
constexpr std::uint8_t kFirstChannel = 1;
constexpr std::uint8_t kChannelCount = 13;

// Every Nth consecutive "no AP found" (reason 201) triggers a background rescan,
// so the log shows what the radio can hear right now instead of only at boot.
constexpr std::uint32_t kRescanEveryNotFound = 3;
constexpr std::uint32_t kRescanTimeoutMs = 8000;

std::uint32_t gConsecutiveNotFound = 0;
bool gRescanRunning = false;
std::uint32_t gRescanStartedMs = 0;

bool empty(const char *text) { return text == nullptr || text[0] == '\0'; }

/// Same SSID ignoring case and leading/trailing whitespace - the usual reason a
/// network that is clearly "there" never matches.
bool nearMatch(const String &seen, const char *wanted) {
    String a = seen;
    String b = wanted == nullptr ? "" : wanted;
    a.trim();
    b.trim();
    return a.length() > 0 && a.equalsIgnoreCase(b);
}

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

    // Channels 1-13, fixed. Must run after WiFi.mode() has started the driver.
    wifi_country_t country = {};
    std::strncpy(country.cc, kCountryCode, sizeof(country.cc));
    country.schan = kFirstChannel;
    country.nchan = kChannelCount;
    country.policy = WIFI_COUNTRY_POLICY_MANUAL;
    const esp_err_t countryErr = esp_wifi_set_country(&country);

    // Known workaround for the ESP32-C3 SuperMini / LOLIN C3 Mini antenna layout:
    // at full TX power these boards often fail to scan or associate. Applied here
    // so the startup scan uses it too, not only the connection attempts.
    WiFi.setTxPower(WIFI_POWER_8_5dBm);

    if (gLog != nullptr) {
        gLog->printf("WiFi: radio country=%s ch%u-%u%s, tx power 8.5 dBm\n", kCountryCode,
                     static_cast<unsigned>(kFirstChannel),
                     static_cast<unsigned>(kFirstChannel + kChannelCount - 1),
                     countryErr == ESP_OK ? "" : " (set_country FAILED)");
    }

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

/// Print every network from a completed scan and return how many match `ssid`.
/// The full list is the point: it separates "wrong name" from "can't hear it".
static int reportScanResults(Stream &out, const char *ssid, int found) {
    if (found <= 0) {
        out.println("WiFi:   nothing heard at all - antenna, power supply or RF problem");
        return 0;
    }

    int matches = 0;
    int hidden = 0;
    bool near = false;
    for (int i = 0; i < found; ++i) {
        const String name = WiFi.SSID(i);
        const bool exact = ssid != nullptr && name == ssid;
        if (name.length() == 0) {
            ++hidden;
        }
        if (exact) {
            ++matches;
        } else if (nearMatch(name, ssid)) {
            near = true;
        }
        out.printf("WiFi:   %s \"%s\" ch=%d rssi=%d dBm %s\n", exact ? "MATCH" : "     ",
                   name.length() ? name.c_str() : "<hidden>", WiFi.channel(i), WiFi.RSSI(i),
                   authModeName(WiFi.encryptionType(i)));
    }

    out.printf("WiFi:   %d network(s) heard, %d matching \"%s\"", found, matches,
               ssid == nullptr ? "" : ssid);
    if (hidden > 0) {
        out.printf(", %d hidden", hidden);
    }
    out.println();

    if (matches == 0 && near) {
        out.println("WiFi:   an SSID differs only in case or spaces - copy it into");
        out.println("WiFi:   WIFI_SSID in include/secrets.h exactly as listed above");
    }
    return matches;
}

void logVisibleNetworks(Stream &out, const char *ssid) {
    out.printf("WiFi: scanning for \"%s\" (active, all channels)...\n",
               ssid == nullptr ? "" : ssid);
    int matches = reportScanResults(out, ssid, WiFi.scanNetworks(false, true));
    WiFi.scanDelete();

    if (matches == 0) {
        // Phone hotspots can be slow to answer probe requests; a passive scan
        // with a long dwell listens for beacons instead.
        out.println("WiFi: not found, retrying with a passive 500 ms/channel scan...");
        matches = reportScanResults(out, ssid, WiFi.scanNetworks(false, true, true, 500));
        WiFi.scanDelete();
    }

    if (matches == 0) {
        out.println("WiFi:   still not heard. Check, in order: hotspot still on and on");
        out.println("WiFi:   2.4 GHz; board within ~1 m of the phone; the phone lists the");
        out.println("WiFi:   exact same name; nothing touching the chip antenna.");
    }
}

/// Start a non-blocking rescan; update() collects the results.
static void startBackgroundRescan() {
    if (gRescanRunning) {
        return;
    }
    if (WiFi.scanNetworks(true, true) == WIFI_SCAN_FAILED) {
        return;
    }
    gRescanRunning = true;
    gRescanStartedMs = millis();
    if (gLog != nullptr) {
        gLog->println("WiFi: rescanning in the background to see what is audible...");
    }
}

/// Poll the background rescan. Returns true while it is still running, so the
/// caller holds off on new association attempts (the two would collide).
static bool serviceBackgroundRescan() {
    if (!gRescanRunning) {
        return false;
    }
    const int16_t state = WiFi.scanComplete();
    if (state == WIFI_SCAN_RUNNING) {
        if (millis() - gRescanStartedMs < kRescanTimeoutMs) {
            return true;
        }
        WiFi.scanDelete();
        gRescanRunning = false;
        return false;
    }
    if (gLog != nullptr) {
        reportScanResults(*gLog, gSsid, state);
    }
    WiFi.scanDelete();
    gRescanRunning = false;
    return false;
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

        // Repeated "no AP found": look again, in the background, so the log shows
        // whether the AP ever becomes audible (hotspot waking up, board moved).
        if (!connected && reason == 201) {
            ++gConsecutiveNotFound;
            if (gConsecutiveNotFound % kRescanEveryNotFound == 0) {
                startBackgroundRescan();
            }
        } else {
            gConsecutiveNotFound = 0;
        }
    }

    if (connected) {
        gConsecutiveNotFound = 0;
    }

    // A scan and an association attempt cannot run at the same time.
    if (serviceBackgroundRescan()) {
        return;
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
