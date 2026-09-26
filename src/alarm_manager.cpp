#include "alarm_manager.h"

#include "board_config.h"

namespace AlarmManager {
namespace {

Pattern gDefaultPattern;
Pattern gActivePattern;
State gState = State::Idle;

std::uint32_t gAlertStartedMs = 0;
std::uint32_t gPhaseStartedMs = 0;
bool gPulseHigh = false;

// Debounced dismiss button.
bool gButtonStable = false;      // true = pressed
bool gButtonLastRaw = false;
std::uint32_t gButtonChangedMs = 0;
bool gDismissPending = false;

/// Logic level that leaves a device OFF.
inline int idleLevel(bool activeHigh) { return activeHigh ? LOW : HIGH; }

inline void writeDevice(int pin, bool activeHigh, bool on) {
    const bool level = activeHigh ? on : !on;
    digitalWrite(pin, level ? HIGH : LOW);
}

void setOutputs(bool on) {
    writeDevice(BoardConfig::kBuzzerPin, BoardConfig::kBuzzerActiveHigh,
                on && gActivePattern.useBuzzer);
    writeDevice(BoardConfig::kVibrationPin, BoardConfig::kVibrationActiveHigh,
                on && gActivePattern.useVibration);
}

bool readButtonRaw() {
    const int level = digitalRead(BoardConfig::kDismissButtonPin);
    return BoardConfig::kButtonActiveLow ? (level == LOW) : (level == HIGH);
}

void pollButton() {
    const bool raw = readButtonRaw();
    const std::uint32_t now = millis();

    if (raw != gButtonLastRaw) {
        gButtonLastRaw = raw;
        gButtonChangedMs = now;
        return;
    }

    if (raw == gButtonStable) {
        return;
    }
    if (now - gButtonChangedMs < BoardConfig::kButtonDebounceMs) {
        return;  // still settling
    }

    gButtonStable = raw;
    if (gButtonStable) {  // press edge
        gDismissPending = true;
        cancel();
    }
}

}  // namespace

void forceOff() {
    // Preset the output latch BEFORE switching the pin to OUTPUT. Doing it the
    // other way round drives the reset-default LOW for a few microseconds, which
    // an active-low module hears as a click.
    digitalWrite(BoardConfig::kBuzzerPin, idleLevel(BoardConfig::kBuzzerActiveHigh));
    digitalWrite(BoardConfig::kVibrationPin, idleLevel(BoardConfig::kVibrationActiveHigh));

    pinMode(BoardConfig::kBuzzerPin, OUTPUT);
    pinMode(BoardConfig::kVibrationPin, OUTPUT);

    // Re-assert now that the driver is actually enabled.
    digitalWrite(BoardConfig::kBuzzerPin, idleLevel(BoardConfig::kBuzzerActiveHigh));
    digitalWrite(BoardConfig::kVibrationPin, idleLevel(BoardConfig::kVibrationActiveHigh));

    gState = State::Idle;
    gPulseHigh = false;
}

void begin() {
    gActivePattern = gDefaultPattern;
    forceOff();

    pinMode(BoardConfig::kDismissButtonPin,
            BoardConfig::kButtonActiveLow ? INPUT_PULLUP : INPUT_PULLDOWN);

    // Seed the debouncer with the current state so holding the button at boot is
    // not reported as a fresh press.
    gButtonLastRaw = readButtonRaw();
    gButtonStable = gButtonLastRaw;
    gButtonChangedMs = millis();
    gDismissPending = false;
}

Pattern &defaultPattern() { return gDefaultPattern; }

void trigger() { trigger(gDefaultPattern); }

void trigger(const Pattern &pattern) {
    gActivePattern = pattern;
    const std::uint32_t now = millis();
    gAlertStartedMs = now;
    gPhaseStartedMs = now;
    gPulseHigh = true;
    gState = State::Alerting;
    setOutputs(true);
}

void cancel() {
    gState = State::Idle;
    gPulseHigh = false;
    setOutputs(false);
}

void update() {
    pollButton();

    if (gState != State::Alerting) {
        // Defensive: keep both lines actively driven to their off level whenever
        // no alert is running, so no glitch or stray write can leave the buzzer
        // or motor latched on.
        digitalWrite(BoardConfig::kBuzzerPin, idleLevel(BoardConfig::kBuzzerActiveHigh));
        digitalWrite(BoardConfig::kVibrationPin, idleLevel(BoardConfig::kVibrationActiveHigh));
        return;
    }

    const std::uint32_t now = millis();

    // Unsigned subtraction keeps this correct across the 49-day millis() wrap.
    if (now - gAlertStartedMs >= gActivePattern.durationMs) {
        cancel();
        return;
    }

    const std::uint32_t phaseLength =
        gPulseHigh ? gActivePattern.pulseOnMs : gActivePattern.pulseOffMs;
    if (now - gPhaseStartedMs >= phaseLength) {
        gPulseHigh = !gPulseHigh;
        gPhaseStartedMs = now;
        setOutputs(gPulseHigh);
    }
}

State state() { return gState; }

bool isAlerting() { return gState == State::Alerting; }

bool consumeDismissPress() {
    const bool pressed = gDismissPending;
    gDismissPending = false;
    return pressed;
}

bool buttonHeld() { return gButtonStable; }

}  // namespace AlarmManager
