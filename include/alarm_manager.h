// =============================================================================
//  alarm_manager.h - non-blocking buzzer + vibration motor alert
// =============================================================================
//  Everything is driven from update(), which must be called from the main loop.
//  No delay() anywhere: the 50 Hz acquisition loop must never be stalled.
// =============================================================================

#pragma once

#include <Arduino.h>

#include <cstdint>

namespace AlarmManager {

struct Pattern {
    std::uint32_t pulseOnMs = 180;    ///< actuators energised
    std::uint32_t pulseOffMs = 120;   ///< gap between pulses
    std::uint32_t durationMs = 4000;  ///< total alert length
    bool useBuzzer = true;
    bool useVibration = true;
};

enum class State : std::uint8_t { Idle, Alerting };

/// Configure the GPIOs and leave both actuators off.
void begin();

/// Start (or restart, extending the alert) the configured pattern.
void trigger();

/// Start a one-off custom pattern, e.g. a short wiring test.
void trigger(const Pattern &pattern);

/// Stop immediately and de-energise both actuators.
void cancel();

/// Advance the state machine. Call every loop iteration.
void update();

State state();
bool isAlerting();

/// Default pattern used by trigger(); edit fields before begin() if desired.
Pattern &defaultPattern();

}  // namespace AlarmManager
