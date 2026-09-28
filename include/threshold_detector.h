// =============================================================================
//  threshold_detector.h - rule-based fall detection, no machine learning
// =============================================================================
//  Used by [env:raw_stream]. Independent of RandomForest.h and of the 53-feature
//  extractor, so this target carries no model at all.
//
//  Classic three-stage cascade from the fall-detection literature, driven as a
//  state machine one sample at a time:
//
//    1. FREE FALL (optional)  |a| dips toward 0 g as the body accelerates down
//    2. IMPACT                |a| spikes when the body hits the floor
//    3. POST-IMPACT           the body then goes still, in a changed orientation
//                             and/or after a large angular-rate excursion
//
//  Requiring all three is what separates a fall from sitting down hard or from
//  setting the wrist down: an impact alone happens constantly in normal life.
//
//  All timing is in milliseconds and every decision is driven by the sample
//  timestamps, so the logic is sample-rate agnostic: it behaves the same on the
//  ~20 Hz UMAFall recordings it was tuned against and at the firmware's 50 Hz.
//
//  Deliberately free of Arduino headers so it can be compiled and evaluated
//  against the dataset on a host machine.
// =============================================================================

#pragma once

#include <cstdint>

namespace ThresholdDetector {

/// Tunable thresholds.
///
/// Defaults were selected by a 240-point grid search over the UMAFall wrist
/// recordings, validated Leave-One-Subject-Out (configuration chosen on 18
/// subjects, scored on the held-out one):
///
///   pooled LOSO   sensitivity 0.856   specificity 0.942   F1 0.854
///   in-sample     sensitivity 0.880   specificity 0.952   F1 0.878
///
/// Recording-level: "did this 15 s trial raise an alarm". 178 of 208 fall trials
/// triggered; 31 of 538 ADL trials triggered falsely.
///
/// Alternative operating points measured on the same data:
///   more sensitive   impactG 2.6, stillStdG 0.50  -> sens 0.933  spec 0.913
///   fewer alarms     impactG 3.0, stillStdG 0.32  -> sens 0.827  spec 0.954
struct Config {
    // --- stage 1: free fall (optional evidence, not required by default) ---
    float freeFallG = 0.60f;            ///< |a| below this counts as free fall
    std::uint32_t freeFallToImpactMs = 800;  ///< impact must follow within this
    bool requireFreeFall = false;       ///< true = no impact accepted without it

    // --- stage 2: impact ---
    float impactG = 3.20f;              ///< |a| at or above this is an impact

    // --- stage 3: post-impact confirmation ---
    std::uint32_t settleDelayMs = 900;     ///< ignore the chaotic period after impact
    std::uint32_t stillnessWindowMs = 1400;  ///< assess stillness over this span
    float stillStdG = 0.50f;               ///< |a| std below this counts as still
    float orientationChangeDeg = 28.0f;    ///< tilt change across the impact
    bool requireOrientationChange = true;  ///< true = tilt only; false = tilt OR gyro spike

    /// Peak |w| after impact. Only consulted when requireOrientationChange is
    /// false; with the default rule it is tracked for diagnostics but does not
    /// affect the decision, so there is no point tuning it.
    float gyroPeakDps = 90.0f;

    // --- housekeeping ---
    std::uint32_t cooldownMs = 5000;    ///< suppress re-triggering after a fall
    std::uint32_t impactTimeoutMs = 3500;  ///< give up on an unconfirmed impact
};

enum class State : std::uint8_t {
    Idle,       ///< watching for free fall or impact
    FreeFall,   ///< low-g detected, waiting for the impact
    Settling,   ///< impact seen, waiting out settleDelayMs
    Assessing,  ///< measuring stillness and orientation
    Cooldown,   ///< a fall was reported, ignoring new impacts
};

/// Returned by update() once per state transition of interest.
enum class Event : std::uint8_t {
    None,
    FreeFallDetected,
    ImpactDetected,
    FallConfirmed,  ///< all stages passed - this is the alarm trigger
    Rejected,       ///< impact seen but post-impact checks failed
};

/// Why an impact was rejected, for tuning and serial diagnostics.
enum class Rejection : std::uint8_t { None, NotStill, NoOrientationChange, Timeout };

class Detector {
   public:
    explicit Detector(const Config &config = Config{}) : config_(config) {}

    void reset();

    /// Feed one sample. Accelerations in g, angular rates in deg/s, timestamp in
    /// milliseconds. Call at the acquisition rate; returns an Event when the
    /// state machine reaches a decision.
    Event update(std::uint32_t timestampMs, float ax, float ay, float az, float gx, float gy,
                 float gz);

    State state() const { return state_; }
    Config &config() { return config_; }
    const Config &config() const { return config_; }

    // --- diagnostics for the most recent candidate ---
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

    // Slow-moving gravity estimate, used as the pre-impact orientation reference.
    // Only updated while Idle, so it is not polluted by the fall itself.
    float refX_ = 0.0f, refY_ = 0.0f, refZ_ = 0.0f;
    bool refValid_ = false;
    std::uint32_t lastSampleMs_ = 0;

    std::uint32_t freeFallStartMs_ = 0;
    std::uint32_t impactMs_ = 0;
    std::uint32_t stateEnteredMs_ = 0;

    // Stillness accumulators (Welford) over the assessment window.
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

}  // namespace ThresholdDetector
