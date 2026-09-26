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

/// Drive both actuators to their off level as early as possible.
///
/// Safe to call before Serial is up, and worth doing as the very first statement
/// in setup(): until a pin is driven it floats, and a floating input on an
/// active-low module reads as "on". Called automatically by begin().
void forceOff();

/// Configure the GPIOs and the dismiss button, and leave both actuators off.
void begin();

/// Start (or restart, extending the alert) the configured pattern.
void trigger();

/// Start a one-off custom pattern, e.g. a short wiring test.
void trigger(const Pattern &pattern);

/// Stop immediately and de-energise both actuators.
void cancel();

/// Advance the state machine and poll the dismiss button. Call every loop
/// iteration. Never blocks, never calls delay().
void update();

State state();
bool isAlerting();

/// True once per debounced press of the dismiss button, then cleared.
///
/// update() already cancels the alarm on a press; this is for the caller to do
/// the rest (clear latched state, notify the dashboard, log).
bool consumeDismissPress();

/// Current debounced button state, for wiring diagnostics.
bool buttonHeld();

/// Default pattern used by trigger(); edit fields before begin() if desired.
Pattern &defaultPattern();

}  // namespace AlarmManager
