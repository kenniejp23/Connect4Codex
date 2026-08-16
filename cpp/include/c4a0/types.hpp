#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace c4a0 {

inline constexpr std::size_t kRows = 6;
inline constexpr std::size_t kCols = 7;
inline constexpr std::size_t kBufferChannels = 2;
inline constexpr std::size_t kBufferLength = kBufferChannels * kRows * kCols;

using Move = std::size_t;
using ModelId = std::uint64_t;
using Policy = std::array<float, kCols>;
using QValue = float;

class Position;

struct EvalPosResult {
  Policy policy{};
  QValue q_penalty{};
  QValue q_no_penalty{};
};

class Evaluator {
 public:
  virtual ~Evaluator() = default;
  virtual std::vector<EvalPosResult> evaluate(
      ModelId model_id, const std::vector<Position>& positions) = 0;
};

struct GameMetadata {
  std::uint64_t game_id{};
  ModelId player0_id{};
  ModelId player1_id{};

  bool operator==(const GameMetadata&) const = default;
};

}  // namespace c4a0
