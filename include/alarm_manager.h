#pragma once

#include <Arduino.h>

#include <cstdint>

namespace AlarmManager {

struct Pattern {
    std::uint32_t pulseOnMs = 2000;
    std::uint32_t pulseOffMs = 0;
    std::uint32_t durationMs = 2000;
    bool useBuzzer = true;
    bool useVibration = true;
};

enum class State : std::uint8_t { Idle, Alerting };

void forceOff();

void begin();

void trigger();

void trigger(const Pattern &pattern);

void cancel();

void update();

State state();
bool isAlerting();

bool consumeDismissPress();

bool buttonHeld();

Pattern &defaultPattern();

}
