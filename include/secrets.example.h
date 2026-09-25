// =============================================================================
//  secrets.example.h - template for include/secrets.h
// =============================================================================
//  This file is TRACKED IN GIT and must never hold real credentials.
//
//  Setup:
//      cp include/secrets.example.h include/secrets.h
//      # then fill in the values in include/secrets.h
//
//  include/secrets.h is listed in .gitignore, so your keys stay local. The
//  config headers (adafruit_io_config.h, blynk_config.h) include it
//  automatically and raise a compile error if a required macro is missing.
//
//  CI alternative: leave secrets.h absent and pass the values as build flags
//  in platformio.ini instead, e.g.
//      build_flags = -D WIFI_SSID=\"my-network\" -D AIO_KEY=\"aio_xxx\"
// =============================================================================

#pragma once

// ---------------------------------------------------------------------------
// WiFi - used by [env:raw_stream] and [env:ml_inference]
// ---------------------------------------------------------------------------
#define WIFI_SSID ""

// Only used when WIFI_EAP_ENABLED is 0 (home network / phone hotspot).
#define WIFI_PASSWORD ""

// ---------------------------------------------------------------------------
// WPA2-Enterprise (campus / eduroam style EAP handshake)
// ---------------------------------------------------------------------------
// 1 for a university network that asks for a username and password, 0 for a
// normal network with a single shared password.
#define WIFI_EAP_ENABLED 0

// 0 = TLS (needs client certificate), 1 = PEAP-MSCHAPv2 (almost always this),
// 2 = TTLS-MSCHAPv2.
#define WIFI_EAP_METHOD 1

// Outer identity, sent unencrypted. Leave "" to reuse WIFI_EAP_USERNAME.
#define WIFI_EAP_IDENTITY ""
#define WIFI_EAP_USERNAME ""
#define WIFI_EAP_PASSWORD ""

// Optional: PEM of the RADIUS server's CA, as a string literal.
// #define WIFI_EAP_CA_PEM "-----BEGIN CERTIFICATE-----\n...\n-----END CERTIFICATE-----\n"

// ---------------------------------------------------------------------------
// Adafruit IO - [env:raw_stream]
// ---------------------------------------------------------------------------
// Username: shown top-right in io.adafruit.com.
// Key: the yellow key icon -> "Active Key", begins with "aio_".
#define AIO_USERNAME ""
#define AIO_KEY ""

// ---------------------------------------------------------------------------
// Blynk IoT - [env:ml_inference]
// ---------------------------------------------------------------------------
// From Blynk Console -> Developer Zone -> Templates. The auth token is issued
// per device (Devices -> your device -> Device Info).
#define BLYNK_TEMPLATE_ID ""
#define BLYNK_TEMPLATE_NAME "Mitifall"
#define BLYNK_AUTH_TOKEN ""
