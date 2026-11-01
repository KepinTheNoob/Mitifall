#pragma once

#include <cstdint>

namespace ThresholdDetector {

struct Config {

    float freeFallG = 0.60f;
    std::uint32_t freeFallToImpactMs = 800;
    bool requireFreeFall = false;

    float impactG = 3.20f;

    std::uint32_t settleDelayMs = 900;
    std::uint32_t stillnessWindowMs = 1400;
    float stillStdG = 0.50f;
    float orientationChangeDeg = 28.0f;
    bool requireOrientationChange = true;

    float gyroPeakDps = 90.0f;

    std::uint32_t cooldownMs = 5000;
    std::uint32_t impactTimeoutMs = 3500;
};

enum class State : std::uint8_t {
    Idle,
    FreeFall,
    Settling,
    Assessing,
    Cooldown,
};

enum class Event : std::uint8_t {
    None,
    FreeFallDetected,
    ImpactDetected,
    FallConfirmed,
    Rejected,
};

enum class Rejection : std::uint8_t { None, NotStill, NoOrientationChange, Timeout };

class Detector {
   public:
    explicit Detector(const Config &config = Config{}) : config_(config) {}

    void reset();

    Event update(std::uint32_t timestampMs, float ax, float ay, float az, float gx, float gy,
                 float gz);

    State state() const { return state_; }
    Config &config() { return config_; }
    const Config &config() const { return config_; }

    float lastImpactG() const { return lastImpactG_; }
    float lastGyroPeakDps() const { return lastGyroPeakDps_; }
    float lastStillStdG() const { return lastStillStdG_; }
    float lastOrientationChangeDeg() const { return lastTiltDeg_; }
    Rejection lastRejection() const { return lastRejection_; }
    std::uint32_t fallCount() const { return fallCount_; }

    static const char *stateName(State state);
    static const char *rejectionName(Rejection rejection);

   private:
    void enterIdle();
    void beginAssessment(std::uint32_t timestampMs);
    Event finishAssessment();

    Config config_;
    State state_ = State::Idle;

    float refX_ = 0.0f, refY_ = 0.0f, refZ_ = 0.0f;
    bool refValid_ = false;
    std::uint32_t lastSampleMs_ = 0;

    std::uint32_t freeFallStartMs_ = 0;
    std::uint32_t impactMs_ = 0;
    std::uint32_t stateEnteredMs_ = 0;

    std::uint32_t stillCount_ = 0;
    float stillMean_ = 0.0f;
    float stillM2_ = 0.0f;
    float postX_ = 0.0f, postY_ = 0.0f, postZ_ = 0.0f;

    float lastImpactG_ = 0.0f;
    float lastGyroPeakDps_ = 0.0f;
    float lastStillStdG_ = 0.0f;
    float lastTiltDeg_ = 0.0f;
    Rejection lastRejection_ = Rejection::None;
    std::uint32_t fallCount_ = 0;
};

}
