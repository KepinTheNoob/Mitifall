// =============================================================================
//  feature_extractor.h - ring buffer + the 53-feature block used by the model
// =============================================================================
//  Mirrors ml_pipeline/feature_extraction.py exactly. The emitted order is the
//  contract in include/features_order.json:
//
//    [ 0..44] per axis (Ax, Ay, Az, Gx, Gy, Gz, Mx, My, Mz), five stats each,
//             in the order mean, std, min, max, var
//    [45..47] acceleration vector magnitude: max, mean, std
//    [48..50] angular velocity vector magnitude: max, mean, std
//    [51]     accelerometer signal magnitude area
//    [52]     gyroscope signal magnitude area
//
//  Deliberately free of Arduino headers so the exact same code can be compiled
//  and diffed against the Python implementation on a host machine.
// =============================================================================

#pragma once

#include <cstddef>
#include <cstdint>

namespace FeatureExtractor {

constexpr std::size_t kAxisCount = 9;
constexpr std::size_t kStatsPerAxis = 5;  // mean, std, min, max, var
constexpr std::size_t kFeatureCount = kAxisCount * kStatsPerAxis + 3 + 3 + 2;  // 53

static_assert(kFeatureCount == 53, "the exported model expects 53 pipeline features");

/// Fixed-capacity circular buffer of 9-axis samples.
///
/// Capacity is the window length; once full, the oldest sample is overwritten,
/// so the buffer always holds the most recent `capacity()` samples.
template <std::size_t Capacity>
class RingBuffer {
   public:
    static_assert(Capacity > 1, "a window needs at least two samples");

    void clear() {
        head_ = 0;
        size_ = 0;
    }

    void push(const float (&sample)[kAxisCount]) {
        float *slot = data_[head_];
        for (std::size_t axis = 0; axis < kAxisCount; ++axis) {
            slot[axis] = sample[axis];
        }
        head_ = (head_ + 1) % Capacity;
        if (size_ < Capacity) {
            ++size_;
        }
    }

    /// Oldest-to-newest access; index 0 is the oldest retained sample.
    const float *at(std::size_t index) const {
        const std::size_t start = (head_ + Capacity - size_) % Capacity;
        return data_[(start + index) % Capacity];
    }

    std::size_t size() const { return size_; }
    bool full() const { return size_ == Capacity; }
    static constexpr std::size_t capacity() { return Capacity; }

   private:
    float data_[Capacity][kAxisCount] = {};
    std::size_t head_ = 0;
    std::size_t size_ = 0;
};

/// Compute the 53 features from `sampleCount` samples supplied oldest-first.
///
/// `fetch` returns a pointer to the nine axis values of sample `i`; this keeps
/// the maths independent of how the caller stores its samples.
void compute(const float *(*fetch)(const void *source, std::size_t index),
             const void *source, std::size_t sampleCount, float *features);

/// Convenience overload for a RingBuffer.
template <std::size_t Capacity>
void compute(const RingBuffer<Capacity> &buffer, float *features) {
    compute(
        [](const void *source, std::size_t index) -> const float * {
            return static_cast<const RingBuffer<Capacity> *>(source)->at(index);
        },
        &buffer, buffer.size(), features);
}

/// Feature names in emission order (diagnostics only).
const char *featureName(std::size_t index);

}  // namespace FeatureExtractor
