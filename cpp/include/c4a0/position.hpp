#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "c4a0/types.hpp"

namespace c4a0 {

enum class CellValue : std::uint8_t { kOpponent = 0, kPlayer = 1 };
enum class TerminalState : std::uint8_t { kPlayerWin, kOpponentWin, kDraw };

class Position {
 public:
  constexpr Position() = default;

  static Position from_bits(std::uint64_t mask, std::uint64_t value);
  static Position from_moves(const std::vector<Move>& moves);
  static Position from_string(std::string_view board);

  [[nodiscard]] std::uint64_t mask() const noexcept { return mask_; }
  [[nodiscard]] std::uint64_t value() const noexcept { return value_; }
  [[nodiscard]] std::optional<Position> make_move(Move col) const;
  [[nodiscard]] std::optional<CellValue> get(std::size_t row, std::size_t col) const;
  [[nodiscard]] std::size_t ply() const noexcept;
  [[nodiscard]] Position inverted() const noexcept;
  [[nodiscard]] std::optional<TerminalState> terminal_state() const noexcept;
  [[nodiscard]] std::optional<std::pair<QValue, QValue>> terminal_values(
      float c_ply_penalty) const noexcept;
  [[nodiscard]] std::array<bool, kCols> legal_moves() const noexcept;
  void mask_policy(Policy& policy_logits) const;
  [[nodiscard]] Position flip_horizontal() const;
  [[nodiscard]] std::vector<Move> to_moves() const;
  [[nodiscard]] std::array<float, kBufferLength> to_buffer() const noexcept;
  [[nodiscard]] std::string to_string() const;

  bool operator==(const Position&) const = default;

 private:
  static constexpr std::uint64_t bit(std::size_t row, std::size_t col) noexcept {
    return std::uint64_t{1} << (row * kCols + col);
  }
  void set_piece(std::size_t row, std::size_t col,
                 std::optional<CellValue> piece) noexcept;
  [[nodiscard]] bool player_has_won() const noexcept;
  [[nodiscard]] bool to_moves_recursive(Position temporary,
                                        std::vector<Move>& removals) const;

  std::uint64_t mask_{};
  std::uint64_t value_{};
};

struct PositionHash {
  std::size_t operator()(const Position& position) const noexcept;
};

struct Sample {
  Position pos;
  Policy policy{};
  QValue q_penalty{};
  QValue q_no_penalty{};

  [[nodiscard]] Sample flip_horizontal() const;
  bool operator==(const Sample&) const = default;
};

struct GameResult {
  GameMetadata metadata;
  std::vector<Sample> samples;

  [[nodiscard]] float player0_score() const;
  bool operator==(const GameResult&) const = default;
};

class PlayGamesResult {
 public:
  std::vector<GameResult> results;

  [[nodiscard]] PlayGamesResult combined(const PlayGamesResult& other) const;
  [[nodiscard]] std::pair<std::vector<Sample>, std::vector<Sample>> split_train_test(
      float train_fraction, std::uint64_t seed) const;
  [[nodiscard]] std::size_t unique_positions() const;
};

}  // namespace c4a0

template <>
struct std::hash<c4a0::Position> {
  std::size_t operator()(const c4a0::Position& position) const noexcept {
    return c4a0::PositionHash{}(position);
  }
};
