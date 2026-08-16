#pragma once

#include <array>
#include <filesystem>
#include <memory>
#include <utility>
#include <vector>

#include "c4a0/position.hpp"

namespace c4a0 {

using Solution = std::array<std::int16_t, kCols>;

[[nodiscard]] float score_policy(const Solution& solution, const Policy& policy);

class CachingSolver {
 public:
  CachingSolver(std::filesystem::path solver_path, std::filesystem::path book_path,
                std::filesystem::path cache_path);
  ~CachingSolver();
  CachingSolver(CachingSolver&&) noexcept;
  CachingSolver& operator=(CachingSolver&&) noexcept;
  CachingSolver(const CachingSolver&) = delete;
  CachingSolver& operator=(const CachingSolver&) = delete;

  [[nodiscard]] std::vector<float> score_policies(
      const std::vector<std::pair<Position, Policy>>& positions_and_policies);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace c4a0
