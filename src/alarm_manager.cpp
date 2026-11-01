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

bool gButtonStable = false;
bool gButtonLastRaw = false;
std::uint32_t gButtonChangedMs = 0;
bool gDismissPending = false;

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
        return;
    }

    gButtonStable = raw;
    if (gButtonStable) {

        if (gState == State::Alerting) {
            gDismissPending = true;
        }
        cancel();
    }
}

}

void forceOff() {

    digitalWrite(BoardConfig::kBuzzerPin, idleLevel(BoardConfig::kBuzzerActiveHigh));
    digitalWrite(BoardConfig::kVibrationPin, idleLevel(BoardConfig::kVibrationActiveHigh));

    pinMode(BoardConfig::kBuzzerPin, OUTPUT);
    pinMode(BoardConfig::kVibrationPin, OUTPUT);

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

        digitalWrite(BoardConfig::kBuzzerPin, idleLevel(BoardConfig::kBuzzerActiveHigh));
        digitalWrite(BoardConfig::kVibrationPin, idleLevel(BoardConfig::kVibrationActiveHigh));
        return;
    }

    const std::uint32_t now = millis();

    if (now - gAlertStartedMs >= gActivePattern.durationMs) {
        cancel();
        return;
    }

    if (gActivePattern.pulseOffMs == 0) {
        if (now - gAlertStartedMs >= gActivePattern.pulseOnMs) {
            cancel();
        }
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

}
