#include "baro_math.h"

#include <cmath>

namespace BaroMath {
namespace {

inline std::uint16_t u16le(const std::uint8_t *p) {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

inline std::int16_t s16le(const std::uint8_t *p) { return static_cast<std::int16_t>(u16le(p)); }

}

bool parseCalibration(const std::uint8_t raw[kCalibrationBytes], Calibration &out) {
    bool allZero = true, allOnes = true;
    for (std::size_t i = 0; i < kCalibrationBytes; ++i) {
        allZero = allZero && raw[i] == 0x00;
        allOnes = allOnes && raw[i] == 0xFF;
    }
    out.T1 = u16le(raw + 0);
    out.T2 = s16le(raw + 2);
    out.T3 = s16le(raw + 4);
    out.P1 = u16le(raw + 6);
    out.P2 = s16le(raw + 8);
    out.P3 = s16le(raw + 10);
    out.P4 = s16le(raw + 12);
    out.P5 = s16le(raw + 14);
    out.P6 = s16le(raw + 16);
    out.P7 = s16le(raw + 18);
    out.P8 = s16le(raw + 20);
    out.P9 = s16le(raw + 22);
    return !allZero && !allOnes && out.T1 != 0 && out.P1 != 0;
}

void unpackAdc(const std::uint8_t raw[kDataBytes], std::int32_t &adcP, std::int32_t &adcT) {
    adcP = (static_cast<std::int32_t>(raw[0]) << 12) | (static_cast<std::int32_t>(raw[1]) << 4) |
           (raw[2] >> 4);
    adcT = (static_cast<std::int32_t>(raw[3]) << 12) | (static_cast<std::int32_t>(raw[4]) << 4) |
           (raw[5] >> 4);
}

bool compensate(const Calibration &c, std::int32_t adcT, std::int32_t adcP, Reading &out) {
    if (adcT == kAdcSkipped || adcP == kAdcSkipped) {
        return false;
    }

    const std::int32_t t1 = static_cast<std::int32_t>(c.T1);
    const std::int32_t var1t = (((adcT >> 3) - t1 * 2) * static_cast<std::int32_t>(c.T2)) >> 11;
    const std::int32_t d = (adcT >> 4) - t1;
    const std::int32_t var2t = (((d * d) >> 12) * static_cast<std::int32_t>(c.T3)) >> 14;
    const std::int32_t tFine = var1t + var2t;
    const std::int32_t centiDegC = (tFine * 5 + 128) >> 8;

    std::int64_t var1 = static_cast<std::int64_t>(tFine) - 128000;
    std::int64_t var2 = var1 * var1 * static_cast<std::int64_t>(c.P6);
    var2 = var2 + (var1 * static_cast<std::int64_t>(c.P5)) * 131072;
    var2 = var2 + static_cast<std::int64_t>(c.P4) * 34359738368LL;
    var1 = ((var1 * var1 * static_cast<std::int64_t>(c.P3)) >> 8) +
           (var1 * static_cast<std::int64_t>(c.P2)) * 4096;
    var1 = ((static_cast<std::int64_t>(1) << 47) + var1) * static_cast<std::int64_t>(c.P1) >> 33;
    if (var1 == 0) {
        return false;
    }
    std::int64_t p = 1048576 - adcP;
    p = (((p << 31) - var2) * 3125) / var1;
    var1 = (static_cast<std::int64_t>(c.P9) * (p >> 13) * (p >> 13)) >> 25;
    var2 = (static_cast<std::int64_t>(c.P8) * p) >> 19;
    p = ((p + var1 + var2) >> 8) + static_cast<std::int64_t>(c.P7) * 16;

    out.temperatureC = static_cast<float>(centiDegC) / 100.0f;
    out.pressurePa = static_cast<float>(p) / 256.0f;
    return out.pressurePa >= kMinPressurePa && out.pressurePa <= kMaxPressurePa &&
           out.temperatureC >= kMinTemperatureC && out.temperatureC <= kMaxTemperatureC;
}

float altitudeM(float pressurePa, float referencePa) {
    if (referencePa <= 0.0f || pressurePa <= 0.0f) {
        return 0.0f;
    }
    return 44330.0f * (1.0f - std::pow(pressurePa / referencePa, 0.1902949f));
}

AltitudeTracker::AltitudeTracker(const Config &cfg) : cfg_(cfg) { reset(); }

void AltitudeTracker::reset() {
    haveFilter_ = false;
    pFilt_ = 0.0f;
    lastMs_ = 0;
    samples_ = 0;
    calibrated_ = false;
    p0_ = 0.0f;
    rezero();
}

void AltitudeTracker::rezero() {
    calibrating_ = true;
    calSum_ = 0.0;
    calCount_ = 0;
    head_ = 0;
    size_ = 0;
    alt_ = diff_ = vz_ = span_ = 0.0f;
}

void AltitudeTracker::addSample(std::uint32_t tMs, float pressurePa) {
    ++samples_;

    if (!haveFilter_) {
        pFilt_ = pressurePa;
        haveFilter_ = true;
    } else {
        const float dt = static_cast<float>(tMs - lastMs_) / 1000.0f;
        const float alpha = dt > 0.0f ? dt / (cfg_.filterTauS + dt) : 1.0f;
        pFilt_ += alpha * (pressurePa - pFilt_);
    }
    lastMs_ = tMs;

    if (calibrating_) {
        if (calCount_ == 0) {
            calStartMs_ = tMs;
        }
        calSum_ += pressurePa;
        ++calCount_;
        if (tMs - calStartMs_ >= cfg_.calibrationMs) {
            p0_ = static_cast<float>(calSum_ / calCount_);
            calibrating_ = false;
            calibrated_ = true;
            pFilt_ = p0_;
        }
        return;
    }

    alt_ = BaroMath::altitudeM(pFilt_, p0_);

    if (size_ == kHistory) {
        head_ = (head_ + 1) % kHistory;
        --size_;
    }
    const std::size_t idx = (head_ + size_) % kHistory;
    histT_[idx] = tMs;
    histH_[idx] = alt_;
    ++size_;

    while (size_ >= 2) {
        const std::size_t second = (head_ + 1) % kHistory;
        if (tMs - histT_[second] >= cfg_.windowMs) {
            head_ = second;
            --size_;
        } else {
            break;
        }
    }

    const std::uint32_t spanMs = tMs - histT_[head_];
    span_ = static_cast<float>(spanMs) / 1000.0f;
    diff_ = alt_ - histH_[head_];
    vz_ = spanMs > 0 ? diff_ / span_ : 0.0f;
}

float AltitudeTracker::noiseM() const {
    if (size_ < 2) {
        return 0.0f;
    }
    double mean = 0.0;
    for (std::size_t i = 0; i < size_; ++i) {
        mean += histH_[(head_ + i) % kHistory];
    }
    mean /= static_cast<double>(size_);
    double m2 = 0.0;
    for (std::size_t i = 0; i < size_; ++i) {
        const double d = histH_[(head_ + i) % kHistory] - mean;
        m2 += d * d;
    }
    return static_cast<float>(std::sqrt(m2 / static_cast<double>(size_)));
}

}
