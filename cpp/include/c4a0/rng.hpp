#pragma once

#include <cstdint>
#include <vector>

#include "c4a0/types.hpp"

namespace c4a0 {

class Pcg32 {
 public:
  explicit Pcg32(std::uint64_t seed, std::uint64_t sequence = 1) noexcept;
  [[nodiscard]] std::uint32_t next_u32() noexcept;
  [[nodiscard]] float uniform_float() noexcept;

 private:
  std::uint64_t state_{};
  std::uint64_t increment_{};
};

[[nodiscard]] std::size_t sample_policy(const Policy& policy, Pcg32& rng);
void deterministic_shuffle(std::vector<std::size_t>& indices, std::uint64_t seed);

}  // namespace c4a0
