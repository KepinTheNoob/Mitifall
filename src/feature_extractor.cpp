#include "feature_extractor.h"

#include <cmath>

namespace FeatureExtractor {
namespace {

constexpr std::size_t kAccBase = 0;  // Ax, Ay, Az
constexpr std::size_t kGyrBase = 3;  // Gx, Gy, Gz

const char *const kFeatureNames[kFeatureCount] = {
    "Ax_mean", "Ax_std", "Ax_min", "Ax_max", "Ax_var",
    "Ay_mean", "Ay_std", "Ay_min", "Ay_max", "Ay_var",
    "Az_mean", "Az_std", "Az_min", "Az_max", "Az_var",
    "Gx_mean", "Gx_std", "Gx_min", "Gx_max", "Gx_var",
    "Gy_mean", "Gy_std", "Gy_min", "Gy_max", "Gy_var",
    "Gz_mean", "Gz_std", "Gz_min", "Gz_max", "Gz_var",
    "Mx_mean", "Mx_std", "Mx_min", "Mx_max", "Mx_var",
    "My_mean", "My_std", "My_min", "My_max", "My_var",
    "Mz_mean", "Mz_std", "Mz_min", "Mz_max", "Mz_var",
    "acc_mag_max", "acc_mag_mean", "acc_mag_std",
    "gyr_mag_max", "gyr_mag_mean", "gyr_mag_std",
    "acc_sma", "gyr_sma",
};

inline float magnitude3(const float *sample, std::size_t base) {
    const float x = sample[base + 0];
    const float y = sample[base + 1];
    const float z = sample[base + 2];
    return std::sqrt(x * x + y * y + z * z);
}

inline float absSum3(const float *sample, std::size_t base) {
    return std::fabs(sample[base + 0]) + std::fabs(sample[base + 1]) +
           std::fabs(sample[base + 2]);
}

}  // namespace

const char *featureName(std::size_t index) {
    return index < kFeatureCount ? kFeatureNames[index] : "";
}

void compute(const float *(*fetch)(const void *source, std::size_t index),
             const void *source, std::size_t sampleCount, float *features) {
    if (sampleCount == 0) {
        for (std::size_t i = 0; i < kFeatureCount; ++i) {
            features[i] = 0.0f;
        }
        return;
    }

    const float inverseCount = 1.0f / static_cast<float>(sampleCount);

    float sum[kAxisCount] = {0};
    float minimum[kAxisCount];
    float maximum[kAxisCount];

    float accMagSum = 0.0f, accMagMax = 0.0f, accAbsSum = 0.0f;
    float gyrMagSum = 0.0f, gyrMagMax = 0.0f, gyrAbsSum = 0.0f;

    // Pass 1: sums, extrema and the magnitude/SMA accumulators.
    for (std::size_t i = 0; i < sampleCount; ++i) {
        const float *sample = fetch(source, i);

        for (std::size_t axis = 0; axis < kAxisCount; ++axis) {
            const float value = sample[axis];
            sum[axis] += value;
            if (i == 0 || value < minimum[axis]) {
                minimum[axis] = value;
            }
            if (i == 0 || value > maximum[axis]) {
                maximum[axis] = value;
            }
        }

        const float accMag = magnitude3(sample, kAccBase);
        accMagSum += accMag;
        if (accMag > accMagMax) {
            accMagMax = accMag;
        }
        accAbsSum += absSum3(sample, kAccBase);

        const float gyrMag = magnitude3(sample, kGyrBase);
        gyrMagSum += gyrMag;
        if (gyrMag > gyrMagMax) {
            gyrMagMax = gyrMag;
        }
        gyrAbsSum += absSum3(sample, kGyrBase);
    }

    float mean[kAxisCount];
    for (std::size_t axis = 0; axis < kAxisCount; ++axis) {
        mean[axis] = sum[axis] * inverseCount;
    }
    const float accMagMean = accMagSum * inverseCount;
    const float gyrMagMean = gyrMagSum * inverseCount;

    // Pass 2: squared deviations about the mean. A second pass costs one more
    // sweep of 100 samples but avoids the catastrophic cancellation that
    // E[x^2] - E[x]^2 suffers in float when the mean dwarfs the deviation -
    // exactly the case for magnetometer axes with large hard-iron offsets.
    float squaredDeviation[kAxisCount] = {0};
    float accMagSquaredDeviation = 0.0f;
    float gyrMagSquaredDeviation = 0.0f;

    for (std::size_t i = 0; i < sampleCount; ++i) {
        const float *sample = fetch(source, i);

        for (std::size_t axis = 0; axis < kAxisCount; ++axis) {
            const float deviation = sample[axis] - mean[axis];
            squaredDeviation[axis] += deviation * deviation;
        }

        const float accDeviation = magnitude3(sample, kAccBase) - accMagMean;
        accMagSquaredDeviation += accDeviation * accDeviation;

        const float gyrDeviation = magnitude3(sample, kGyrBase) - gyrMagMean;
        gyrMagSquaredDeviation += gyrDeviation * gyrDeviation;
    }

    // Emit in the pipeline's order. Population variance (ddof = 0), matching
    // numpy's default and the training-time feature definition.
    std::size_t out = 0;
    for (std::size_t axis = 0; axis < kAxisCount; ++axis) {
        const float variance = squaredDeviation[axis] * inverseCount;
        features[out++] = mean[axis];
        features[out++] = std::sqrt(variance);
        features[out++] = minimum[axis];
        features[out++] = maximum[axis];
        features[out++] = variance;
    }

    features[out++] = accMagMax;
    features[out++] = accMagMean;
    features[out++] = std::sqrt(accMagSquaredDeviation * inverseCount);

    features[out++] = gyrMagMax;
    features[out++] = gyrMagMean;
    features[out++] = std::sqrt(gyrMagSquaredDeviation * inverseCount);

    features[out++] = accAbsSum * inverseCount;  // acc_sma
    features[out++] = gyrAbsSum * inverseCount;  // gyr_sma
}

}  // namespace FeatureExtractor
