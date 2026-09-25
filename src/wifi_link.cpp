#include "wifi_link.h"

#include <WiFi.h>

// ESP-IDF 4.x enterprise supplicant API. Present in this core (the Arduino
// WiFiSTA.cpp includes the same header); used here for the TTLS phase-2 method
// and the clock check, which the Arduino wrapper does not expose.
#include "esp_wpa2.h"

#if __has_include("secrets.h")
#include "secrets.h"
#endif

namespace WifiLink {
namespace {

enum class Mode : std::uint8_t { Disabled, Psk, Enterprise };

const char *gSsid = nullptr;
const char *gPassword = nullptr;
EapCredentials gEap;
Mode gMode = Mode::Disabled;

Stream *gLog = nullptr;
Options gOptions;

bool gWasConnected = false;
bool gJustConnected = false;
std::uint32_t gLastAttemptMs = 0;
std::uint32_t gReconnects = 0;

bool empty(const char *text) { return text == nullptr || text[0] == '\0'; }

void prepareRadio() {
    WiFi.persistent(false);
    WiFi.mode(WIFI_STA);
    WiFi.setAutoReconnect(true);
    if (gOptions.disableSleep) {
        // Modem sleep adds up to ~100 ms of latency to every publish; the power
        // saving is not worth it while streaming telemetry.
        WiFi.setSleep(false);
    }
}

void attemptPsk() {
    WiFi.disconnect(false, false);
    WiFi.begin(gSsid, gPassword);
}

void attemptEnterprise() {
    WiFi.disconnect(false, false);

    // The Arduino enterprise begin() calls strlen() on the identity without a
    // null check, so an outer identity is always supplied; most campus setups
    // accept the full username as the outer identity.
    const char *identity = empty(gEap.identity) ? gEap.username : gEap.identity;

    // With no CA PEM the supplicant cannot validate the server certificate, so
    // certificate expiry checking is meaningless and the board has no wall clock
    // at boot anyway. Disabling the time check avoids a spurious failure.
    if (empty(gEap.caPem)) {
        esp_wifi_sta_wpa2_ent_set_disable_time_check(true);
    }

    // The Arduino wrapper ignores `method` for phase 2, so TTLS needs this set
    // explicitly or the handshake stalls after the outer tunnel comes up.
    if (gEap.method == kEapTtls) {
        esp_wifi_sta_wpa2_ent_set_ttls_phase2_method(ESP_EAP_TTLS_PHASE2_MSCHAPV2);
    }

    // Sets identity/username/password, installs certificates when provided,
    // calls esp_wifi_sta_wpa2_ent_enable() and then associates.
    WiFi.begin(gSsid, static_cast<wpa2_auth_method_t>(gEap.method), identity, gEap.username,
               gEap.password, gEap.caPem, gEap.clientCertPem, gEap.clientKeyPem);
}

void attempt() {
    gLastAttemptMs = millis();
    switch (gMode) {
        case Mode::Psk:
            attemptPsk();
            break;
        case Mode::Enterprise:
            attemptEnterprise();
            break;
        default:
            return;
    }
    if (gLog != nullptr) {
        gLog->printf("WiFi: associating with \"%s\" (%s)\n", gSsid, modeName());
    }
}

}  // namespace

const char *modeName() {
    switch (gMode) {
        case Mode::Psk:
            return "WPA2-PSK";
        case Mode::Enterprise:
            switch (gEap.method) {
                case kEapTls:
                    return "WPA2-Enterprise (TLS)";
                case kEapTtls:
                    return "WPA2-Enterprise (TTLS/MSCHAPv2)";
                default:
                    return "WPA2-Enterprise (PEAP/MSCHAPv2)";
            }
        default:
            return "disabled";
    }
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

    // Must come after WiFi.mode() has initialised the driver: leaving enterprise
    // mode enabled from a previous session would break a plain PSK association.
    esp_wifi_sta_wpa2_ent_disable();

    attempt();
}

void beginEnterprise(const char *ssid, const EapCredentials &credentials, Stream *log,
                     const Options &options) {
    gLog = log;
    gOptions = options;

    if (empty(ssid)) {
        gMode = Mode::Disabled;
        if (gLog != nullptr) {
            gLog->println("WiFi: no SSID configured, staying offline");
        }
        return;
    }
    if (empty(credentials.username) || empty(credentials.password)) {
        gMode = Mode::Disabled;
        if (gLog != nullptr) {
            gLog->println("WiFi: enterprise mode needs WIFI_EAP_USERNAME and "
                          "WIFI_EAP_PASSWORD in include/secrets.h, staying offline");
        }
        return;
    }

    // The supplicant caps these at 64 bytes and silently fails beyond it.
    if (strlen(credentials.username) > 64 || strlen(credentials.password) > 64 ||
        (credentials.identity != nullptr && strlen(credentials.identity) > 64)) {
        gMode = Mode::Disabled;
        if (gLog != nullptr) {
            gLog->println("WiFi: EAP identity/username/password must be <= 64 chars");
        }
        return;
    }

    gSsid = ssid;
    gEap = credentials;
    gMode = Mode::Enterprise;

    prepareRadio();
    attempt();
}

void beginFromSecrets(Stream *log, const Options &options) {
#if defined(WIFI_SSID)
#if defined(WIFI_EAP_ENABLED) && (WIFI_EAP_ENABLED)
    EapCredentials eap;
#if defined(WIFI_EAP_IDENTITY)
    eap.identity = WIFI_EAP_IDENTITY;
#endif
#if defined(WIFI_EAP_USERNAME)
    eap.username = WIFI_EAP_USERNAME;
#endif
#if defined(WIFI_EAP_PASSWORD)
    eap.password = WIFI_EAP_PASSWORD;
#endif
#if defined(WIFI_EAP_METHOD)
    eap.method = WIFI_EAP_METHOD;
#endif
#if defined(WIFI_EAP_CA_PEM)
    eap.caPem = WIFI_EAP_CA_PEM;
#endif
    beginEnterprise(WIFI_SSID, eap, log, options);
#else
    begin(WIFI_SSID, WIFI_PASSWORD, log, options);
#endif
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
