#include "threshold_detector.h"

#include <cmath>

namespace ThresholdDetector {
namespace {

constexpr float kRadToDeg = 57.2957795131f;

// Time constant of the gravity-direction reference. Long enough to ignore normal
// arm motion, short enough to follow a genuine posture change within a second or
// two of ordinary activity.
constexpr float kReferenceTauMs = 1500.0f;

inline float magnitude3(float x, float y, float z) { return std::sqrt(x * x + y * y + z * z); }

/// Angle between two vectors, in degrees. Returns 0 if either is degenerate.
float angleBetweenDeg(float ax, float ay, float az, float bx, float by, float bz) {
    const float na = magnitude3(ax, ay, az);
    const float nb = magnitude3(bx, by, bz);
    if (na < 1e-4f || nb < 1e-4f) {
        return 0.0f;
    }
    float cosine = (ax * bx + ay * by + az * bz) / (na * nb);
    if (cosine > 1.0f) cosine = 1.0f;
    if (cosine < -1.0f) cosine = -1.0f;
    return std::acos(cosine) * kRadToDeg;
}

}  // namespace

const char *Detector::stateName(State state) {
    switch (state) {
        case State::FreeFall:
            return "free-fall";
        case State::Settling:
            return "settling";
        case State::Assessing:
            return "assessing";
        case State::Cooldown:
            return "cooldown";
        default:
            return "idle";
    }
}

const char *Detector::rejectionName(Rejection rejection) {
    switch (rejection) {
        case Rejection::NotStill:
            return "still moving after impact";
        case Rejection::NoOrientationChange:
            return "no orientation change or angular spike";
        case Rejection::Timeout:
            return "impact not confirmed in time";
        default:
            return "none";
    }
}

void Detector::reset() {
    state_ = State::Idle;
    refValid_ = false;
    lastSampleMs_ = 0;
    freeFallStartMs_ = 0;
    impactMs_ = 0;
    stateEnteredMs_ = 0;
    stillCount_ = 0;
    stillMean_ = 0.0f;
    stillM2_ = 0.0f;
    lastRejection_ = Rejection::None;
}

void Detector::enterIdle() {
    state_ = State::Idle;
    stillCount_ = 0;
    stillMean_ = 0.0f;
    stillM2_ = 0.0f;
}

void Detector::beginAssessment(std::uint32_t timestampMs) {
    state_ = State::Assessing;
    stateEnteredMs_ = timestampMs;
    stillCount_ = 0;
    stillMean_ = 0.0f;
    stillM2_ = 0.0f;
    postX_ = postY_ = postZ_ = 0.0f;
}

Event Detector::finishAssessment() {
    // Population standard deviation of |a| across the assessment window.
    lastStillStdG_ =
        stillCount_ > 1 ? std::sqrt(stillM2_ / static_cast<float>(stillCount_)) : 0.0f;

    const float inv = stillCount_ > 0 ? 1.0f / static_cast<float>(stillCount_) : 0.0f;
    const float px = postX_ * inv, py = postY_ * inv, pz = postZ_ * inv;
    lastTiltDeg_ = refValid_ ? angleBetweenDeg(refX_, refY_, refZ_, px, py, pz) : 0.0f;

    const bool still = lastStillStdG_ <= config_.stillStdG;
    const bool tilted = lastTiltDeg_ >= config_.orientationChangeDeg;
    const bool spun = lastGyroPeakDps_ >= config_.gyroPeakDps;

    if (!still) {
        lastRejection_ = Rejection::NotStill;
        enterIdle();
        return Event::Rejected;
    }

    // A fall reorients the body, spins it on the way down, or both. Requiring the
    // orientation change alone is stricter; the default accepts either, because
    // some wrist falls end with the arm back near its original attitude.
    const bool posture = config_.requireOrientationChange ? tilted : (tilted || spun);
    if (!posture) {
        lastRejection_ = Rejection::NoOrientationChange;
        enterIdle();
        return Event::Rejected;
    }

    lastRejection_ = Rejection::None;
    ++fallCount_;
    state_ = State::Cooldown;
    stateEnteredMs_ = impactMs_;
    return Event::FallConfirmed;
}

Event Detector::update(std::uint32_t timestampMs, float ax, float ay, float az, float gx,
                       float gy, float gz) {
    const float accelMag = magnitude3(ax, ay, az);
    const float gyroMag = magnitude3(gx, gy, gz);

    const std::uint32_t dt = lastSampleMs_ == 0 ? 0 : timestampMs - lastSampleMs_;
    lastSampleMs_ = timestampMs;

    switch (state_) {
        case State::Cooldown:
            if (timestampMs - stateEnteredMs_ >= config_.cooldownMs) {
                enterIdle();
            }
            return Event::None;

        case State::Idle: {
            // Track the gravity direction with a first-order filter, but only
            // while idle: once a candidate starts, this must stay frozen at the
            // pre-impact posture for the comparison to mean anything.
            if (!refValid_) {
                refX_ = ax;
                refY_ = ay;
                refZ_ = az;
                refValid_ = true;
            } else if (dt > 0) {
                float alpha = static_cast<float>(dt) / kReferenceTauMs;
                if (alpha > 1.0f) alpha = 1.0f;
                refX_ += alpha * (ax - refX_);
                refY_ += alpha * (ay - refY_);
                refZ_ += alpha * (az - refZ_);
            }

            if (accelMag >= config_.impactG) {
                if (config_.requireFreeFall) {
                    return Event::None;  // impact without the preceding low-g dip
                }
                impactMs_ = timestampMs;
                lastImpactG_ = accelMag;
                lastGyroPeakDps_ = gyroMag;
                state_ = State::Settling;
                stateEnteredMs_ = timestampMs;
                return Event::ImpactDetected;
            }

            if (accelMag <= config_.freeFallG) {
                freeFallStartMs_ = timestampMs;
                state_ = State::FreeFall;
                stateEnteredMs_ = timestampMs;
                return Event::FreeFallDetected;
            }
            return Event::None;
        }

        case State::FreeFall: {
            if (accelMag >= config_.impactG) {
                impactMs_ = timestampMs;
                lastImpactG_ = accelMag;
                lastGyroPeakDps_ = gyroMag;
                state_ = State::Settling;
                stateEnteredMs_ = timestampMs;
                return Event::ImpactDetected;
            }
            if (timestampMs - freeFallStartMs_ >= config_.freeFallToImpactMs) {
                enterIdle();
            }
            return Event::None;
        }

        case State::Settling: {
            // Still inside the chaotic period right after the impact: all we do
            // is track the peak angular rate.
            if (gyroMag > lastGyroPeakDps_) {
                lastGyroPeakDps_ = gyroMag;
            }
            if (accelMag > lastImpactG_) {
                lastImpactG_ = accelMag;
            }
            if (timestampMs - impactMs_ >= config_.settleDelayMs) {
                beginAssessment(timestampMs);
            } else if (timestampMs - impactMs_ >= config_.impactTimeoutMs) {
                lastRejection_ = Rejection::Timeout;
                enterIdle();
                return Event::Rejected;
            }
            return Event::None;
        }

        case State::Assessing: {
            if (gyroMag > lastGyroPeakDps_) {
                lastGyroPeakDps_ = gyroMag;
            }

            // Welford accumulation of |a|, plus the mean accel vector for the
            // post-impact orientation.
            ++stillCount_;
            const float delta = accelMag - stillMean_;
            stillMean_ += delta / static_cast<float>(stillCount_);
            stillM2_ += delta * (accelMag - stillMean_);
            postX_ += ax;
            postY_ += ay;
            postZ_ += az;

            if (timestampMs - stateEnteredMs_ >= config_.stillnessWindowMs) {
                return finishAssessment();
            }
            if (timestampMs - impactMs_ >= config_.impactTimeoutMs) {
                // Ran out of time but we have samples; judge on what we have.
                return finishAssessment();
            }
            return Event::None;
        }
    }

    return Event::None;
}

}  // namespace ThresholdDetector
