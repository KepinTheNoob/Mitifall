#pragma once

#include <cstddef>
#include <cstdint>

namespace BaroMath {

struct Calibration {
    std::uint16_t T1 = 0;
    std::int16_t T2 = 0, T3 = 0;
    std::uint16_t P1 = 0;
    std::int16_t P2 = 0, P3 = 0, P4 = 0, P5 = 0, P6 = 0, P7 = 0, P8 = 0, P9 = 0;
};

constexpr std::size_t kCalibrationBytes = 24;
constexpr std::size_t kDataBytes = 6;

constexpr std::int32_t kAdcSkipped = 0x80000;

constexpr float kMinPressurePa = 30000.0f;
constexpr float kMaxPressurePa = 110000.0f;
constexpr float kMinTemperatureC = -40.0f;
constexpr float kMaxTemperatureC = 85.0f;

bool parseCalibration(const std::uint8_t raw[kCalibrationBytes], Calibration &out);

void unpackAdc(const std::uint8_t raw[kDataBytes], std::int32_t &adcP, std::int32_t &adcT);

struct Reading {
    float temperatureC = 0.0f;
    float pressurePa = 0.0f;
};

bool compensate(const Calibration &cal, std::int32_t adcT, std::int32_t adcP, Reading &out);

float altitudeM(float pressurePa, float referencePa);

class AltitudeTracker {
   public:
    struct Config {
        float filterTauS = 0.25f;
        std::uint32_t windowMs = 1000;
        std::uint32_t calibrationMs = 2000;
    };

    AltitudeTracker() : AltitudeTracker(Config{}) {}
    explicit AltitudeTracker(const Config &cfg);

    void reset();
    void rezero();

    void addSample(std::uint32_t tMs, float pressurePa);

    bool calibrated() const { return calibrated_; }
    bool calibrating() const { return calibrating_; }
    float referencePa() const { return p0_; }
    float pressurePa() const { return pFilt_; }
    float altitudeM() const { return alt_; }
    float altitudeDiffM() const { return diff_; }
    float verticalSpeedMps() const { return vz_; }
    float windowSpanS() const { return span_; }
    float noiseM() const;
    std::uint32_t sampleCount() const { return samples_; }
    std::uint32_t calibrationSamples() const { return calCount_; }
    const Config &config() const { return cfg_; }

   private:
    static constexpr std::size_t kHistory = 128;

    Config cfg_;

    bool calibrated_ = false;
    bool calibrating_ = true;
    std::uint32_t calStartMs_ = 0;
    double calSum_ = 0.0;
    std::uint32_t calCount_ = 0;
    float p0_ = 0.0f;

    bool haveFilter_ = false;
    float pFilt_ = 0.0f;
    std::uint32_t lastMs_ = 0;

    std::uint32_t histT_[kHistory] = {};
    float histH_[kHistory] = {};
    std::size_t head_ = 0;
    std::size_t size_ = 0;

    float alt_ = 0.0f;
    float diff_ = 0.0f;
    float vz_ = 0.0f;
    float span_ = 0.0f;
    std::uint32_t samples_ = 0;
};

}
