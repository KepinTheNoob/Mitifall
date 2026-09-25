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

inline void writeActuator(int pin, bool on) {
    const bool level = BoardConfig::kActuatorsActiveHigh ? on : !on;
    digitalWrite(pin, level ? HIGH : LOW);
}

void setOutputs(bool on) {
    writeActuator(BoardConfig::kBuzzerPin, on && gActivePattern.useBuzzer);
    writeActuator(BoardConfig::kVibrationPin, on && gActivePattern.useVibration);
}

}  // namespace

void begin() {
    pinMode(BoardConfig::kBuzzerPin, OUTPUT);
    pinMode(BoardConfig::kVibrationPin, OUTPUT);
    gActivePattern = gDefaultPattern;
    setOutputs(false);
    gState = State::Idle;
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
    if (gState != State::Alerting) {
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

}  // namespace AlarmManager
