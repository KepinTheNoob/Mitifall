#pragma once

#include <cstddef>
#include <cstdint>

namespace FeatureExtractor {

constexpr std::size_t kAxisCount = 9;
constexpr std::size_t kStatsPerAxis = 5;
constexpr std::size_t kFeatureCount = kAxisCount * kStatsPerAxis + 3 + 3 + 2;

static_assert(kFeatureCount == 53, "the exported model expects 53 pipeline features");

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

void compute(const float *(*fetch)(const void *source, std::size_t index),
             const void *source, std::size_t sampleCount, float *features);

template <std::size_t Capacity>
void compute(const RingBuffer<Capacity> &buffer, float *features) {
    compute(
        [](const void *source, std::size_t index) -> const float * {
            return static_cast<const RingBuffer<Capacity> *>(source)->at(index);
        },
        &buffer, buffer.size(), features);
}

const char *featureName(std::size_t index);

}
