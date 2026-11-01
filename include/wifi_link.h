#pragma once

#include <Arduino.h>

#include <cstdint>

namespace WifiLink {

struct Options {
    std::uint32_t retryIntervalMs = 8000;
    bool disableSleep = true;

    bool scanOnBegin = true;

    bool logDisconnectReason = true;
};

void begin(const char *ssid, const char *password, Stream *log = nullptr,
           const Options &options = Options{});

void beginFromSecrets(Stream *log = nullptr, const Options &options = Options{});

void update();

bool isConnected();

bool isEnabled();

String ipAddress();

std::int32_t rssi();

bool consumeJustConnected();

std::uint32_t reconnectCount();

const char *modeName();

std::uint8_t lastDisconnectReason();

const char *disconnectReasonName(std::uint8_t reason);

void logVisibleNetworks(Stream &out, const char *ssid);

}
