#include "c4a0/rng.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace c4a0 {
namespace {

std::uint64_t next_u64(Pcg32& rng) noexcept {
  return (static_cast<std::uint64_t>(rng.next_u32()) << 32U) |
         static_cast<std::uint64_t>(rng.next_u32());
}

std::uint64_t bounded_u64(Pcg32& rng, std::uint64_t bound) {
  if (bound == 0) {
    throw std::invalid_argument("random bound must be positive");
  }
  const auto threshold = (std::uint64_t{0} - bound) % bound;
  for (;;) {
    const auto value = next_u64(rng);
    if (value >= threshold) {
      return value % bound;
    }
  }
}

}  // namespace

Pcg32::Pcg32(std::uint64_t seed, std::uint64_t sequence) noexcept
    : increment_((sequence << 1U) | 1U) {
  static_cast<void>(next_u32());
  state_ += seed;
  static_cast<void>(next_u32());
}

std::uint32_t Pcg32::next_u32() noexcept {
  const auto old_state = state_;
  state_ = old_state * 6364136223846793005ULL + increment_;
  const auto xorshifted =
      static_cast<std::uint32_t>(((old_state >> 18U) ^ old_state) >> 27U);
  const auto rotation = static_cast<std::uint32_t>(old_state >> 59U);
  return (xorshifted >> rotation) |
         (xorshifted << ((std::uint32_t{0} - rotation) & 31U));
}

float Pcg32::uniform_float() noexcept {
  constexpr float kScale = 1.0F / 16777216.0F;
  return static_cast<float>(next_u32() >> 8U) * kScale;
}

std::size_t sample_policy(const Policy& policy, Pcg32& rng) {
  double total = 0.0;
  std::size_t last_positive = 0;
  bool has_positive = false;
  for (std::size_t col = 0; col < policy.size(); ++col) {
    const auto weight = policy[col];
    if (!std::isfinite(weight) || weight < 0.0F) {
      throw std::invalid_argument(
          "policy sampling weights must be finite and non-negative");
    }
    total += static_cast<double>(weight);
    if (weight > 0.0F) {
      has_positive = true;
      last_positive = col;
    }
  }
  if (!has_positive || !std::isfinite(total)) {
    throw std::invalid_argument(
        "policy sampling requires at least one positive weight");
  }

  const auto target = static_cast<double>(rng.uniform_float()) * total;
  double cumulative = 0.0;
  for (std::size_t col = 0; col < policy.size(); ++col) {
    cumulative += static_cast<double>(policy[col]);
    if (target < cumulative) {
      return col;
    }
  }
  return last_positive;
}

void deterministic_shuffle(std::vector<std::size_t>& indices, std::uint64_t seed) {
  Pcg32 rng(seed);
  for (std::size_t end = indices.size(); end > 1; --end) {
    const auto selected =
        static_cast<std::size_t>(bounded_u64(rng, static_cast<std::uint64_t>(end)));
    const auto last = end - 1;
    const auto temporary = indices[last];
    indices[last] = indices[selected];
    indices[selected] = temporary;
  }
}

}  // namespace c4a0
