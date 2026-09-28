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
// WPA2-Personal only: one SSID, one shared password (home router or phone
// hotspot). Networks that ask for a per-user login are not supported.
#define WIFI_SSID ""
#define WIFI_PASSWORD ""

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
