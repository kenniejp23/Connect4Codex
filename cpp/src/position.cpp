#include "c4a0/position.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "c4a0/rng.hpp"

namespace c4a0 {
namespace {

constexpr std::uint64_t board_bit(std::size_t row, std::size_t col) noexcept {
  return std::uint64_t{1} << (row * kCols + col);
}

consteval std::array<std::uint64_t, 69> make_win_masks() {
  std::array<std::uint64_t, 69> masks{};
  std::size_t index = 0;

  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t col = 0; col <= kCols - 4; ++col) {
      masks[index++] = board_bit(row, col) | board_bit(row, col + 1) |
                       board_bit(row, col + 2) | board_bit(row, col + 3);
    }
  }
  for (std::size_t col = 0; col < kCols; ++col) {
    for (std::size_t row = 0; row <= kRows - 4; ++row) {
      masks[index++] = board_bit(row, col) | board_bit(row + 1, col) |
                       board_bit(row + 2, col) | board_bit(row + 3, col);
    }
  }
  for (std::size_t row = 0; row <= kRows - 4; ++row) {
    for (std::size_t col = 0; col <= kCols - 4; ++col) {
      masks[index++] = board_bit(row, col) | board_bit(row + 1, col + 1) |
                       board_bit(row + 2, col + 2) | board_bit(row + 3, col + 3);
    }
  }
  for (std::size_t row = 3; row < kRows; ++row) {
    for (std::size_t col = 0; col <= kCols - 4; ++col) {
      masks[index++] = board_bit(row, col) | board_bit(row - 1, col + 1) |
                       board_bit(row - 2, col + 2) | board_bit(row - 3, col + 3);
    }
  }

  if (index != masks.size()) {
    throw "incorrect number of Connect Four win masks";
  }
  return masks;
}

inline constexpr auto kWinMasks = make_win_masks();
inline constexpr std::uint64_t kBoardMask = (std::uint64_t{1} << (kRows * kCols)) - 1;
inline constexpr std::string_view kPlayerToken = "\360\237\224\264";
inline constexpr std::string_view kOpponentToken = "\360\237\224\265";
inline constexpr std::string_view kEmptyToken = "\342\232\253";

std::vector<std::string_view> split_board_lines(std::string_view board) {
  std::vector<std::string_view> lines;
  std::size_t start = 0;
  while (start < board.size()) {
    const auto newline = board.find('\n', start);
    auto line =
        board.substr(start, newline == std::string_view::npos ? std::string_view::npos
                                                              : newline - start);
    if (!line.empty() && line.back() == '\r') {
      line.remove_suffix(1);
    }
    lines.push_back(line);
    if (newline == std::string_view::npos) {
      break;
    }
    start = newline + 1;
  }
  return lines;
}

std::array<std::optional<CellValue>, kCols> parse_board_line(std::string_view line) {
  std::array<std::optional<CellValue>, kCols> cells{};
  std::size_t offset = 0;
  for (std::size_t col = 0; col < kCols; ++col) {
    if (line.substr(offset).starts_with(kPlayerToken)) {
      cells[col] = CellValue::kPlayer;
      offset += kPlayerToken.size();
    } else if (line.substr(offset).starts_with(kOpponentToken)) {
      cells[col] = CellValue::kOpponent;
      offset += kOpponentToken.size();
    } else if (line.substr(offset).starts_with(kEmptyToken)) {
      cells[col] = std::nullopt;
      offset += kEmptyToken.size();
    } else {
      throw std::invalid_argument(
          "position rows must contain exactly seven Connect Four tokens");
    }
  }
  if (offset != line.size()) {
    throw std::invalid_argument(
        "position rows must contain exactly seven Connect Four tokens");
  }
  return cells;
}

std::uint64_t mix_hash(std::uint64_t value) noexcept {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

}  // namespace

Position Position::from_bits(std::uint64_t mask, std::uint64_t value) {
  if ((mask & ~kBoardMask) != 0) {
    throw std::invalid_argument("position mask contains bits outside the board");
  }
  if ((value & ~mask) != 0) {
    throw std::invalid_argument("position value contains bits outside its mask");
  }
  Position position;
  position.mask_ = mask;
  position.value_ = value;
  return position;
}

Position Position::from_moves(const std::vector<Move>& moves) {
  Position position;
  for (const Move move : moves) {
    const auto next = position.make_move(move);
    if (!next.has_value()) {
      throw std::invalid_argument("invalid move sequence");
    }
    position = *next;
  }
  return position;
}

Position Position::from_string(std::string_view board) {
  const auto lines = split_board_lines(board);
  if (lines.size() != kRows) {
    throw std::invalid_argument("position must contain exactly six rows");
  }

  Position position;
  for (std::size_t display_row = 0; display_row < kRows; ++display_row) {
    const auto cells = parse_board_line(lines[display_row]);
    const auto row = kRows - 1 - display_row;
    for (std::size_t col = 0; col < kCols; ++col) {
      position.set_piece(row, col, cells[col]);
    }
  }
  return position;
}

std::optional<Position> Position::make_move(Move col) const {
  if (col >= kCols) {
    return std::nullopt;
  }
  for (std::size_t row = 0; row < kRows; ++row) {
    if ((mask_ & bit(row, col)) == 0) {
      Position result = *this;
      result.set_piece(row, col, CellValue::kPlayer);
      return result.inverted();
    }
  }
  return std::nullopt;
}

std::optional<CellValue> Position::get(std::size_t row, std::size_t col) const {
  if (row >= kRows || col >= kCols) {
    return std::nullopt;
  }
  const auto cell = bit(row, col);
  if ((mask_ & cell) == 0) {
    return std::nullopt;
  }
  return (value_ & cell) == 0 ? CellValue::kOpponent : CellValue::kPlayer;
}

std::size_t Position::ply() const noexcept {
  return static_cast<std::size_t>(std::popcount(mask_));
}

Position Position::inverted() const noexcept {
  Position result = *this;
  result.value_ = (~result.value_) & result.mask_;
  return result;
}

std::optional<TerminalState> Position::terminal_state() const noexcept {
  if (player_has_won()) {
    return TerminalState::kPlayerWin;
  }
  if (inverted().player_has_won()) {
    return TerminalState::kOpponentWin;
  }
  if (ply() == kRows * kCols) {
    return TerminalState::kDraw;
  }
  return std::nullopt;
}

std::optional<std::pair<QValue, QValue>> Position::terminal_values(
    float c_ply_penalty) const noexcept {
  const auto terminal = terminal_state();
  if (!terminal.has_value()) {
    return std::nullopt;
  }
  const auto penalty = c_ply_penalty * static_cast<float>(ply());
  switch (*terminal) {
    case TerminalState::kPlayerWin:
      return std::pair<QValue, QValue>{1.0F - penalty, 1.0F};
    case TerminalState::kOpponentWin:
      return std::pair<QValue, QValue>{-1.0F + penalty, -1.0F};
    case TerminalState::kDraw:
      return std::pair<QValue, QValue>{0.0F, 0.0F};
  }
  return std::nullopt;
}

std::array<bool, kCols> Position::legal_moves() const noexcept {
  std::array<bool, kCols> legal{};
  for (std::size_t col = 0; col < kCols; ++col) {
    legal[col] = (mask_ & bit(kRows - 1, col)) == 0;
  }
  return legal;
}

void Position::mask_policy(Policy& policy_logits) const {
  const auto legal = legal_moves();
  for (std::size_t col = 0; col < kCols; ++col) {
    if (!legal[col]) {
      policy_logits[col] = -std::numeric_limits<float>::infinity();
    }
  }
}

Position Position::flip_horizontal() const {
  Position result;
  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t col = 0; col < kCols; ++col) {
      result.set_piece(row, kCols - 1 - col, get(row, col));
    }
  }
  return result;
}

std::vector<Move> Position::to_moves() const {
  std::vector<Move> removals;
  if (!to_moves_recursive(*this, removals)) {
    throw std::logic_error("position cannot be reconstructed as a legal move sequence");
  }
  std::reverse(removals.begin(), removals.end());
  return removals;
}

std::array<float, kBufferLength> Position::to_buffer() const noexcept {
  std::array<float, kBufferLength> buffer{};
  constexpr auto channel_length = kRows * kCols;
  for (std::size_t channel = 0; channel < kBufferChannels; ++channel) {
    for (std::size_t row = 0; row < kRows; ++row) {
      for (std::size_t col = 0; col < kCols; ++col) {
        const auto piece = get(row, col);
        const bool occupied = (channel == 0 && piece == CellValue::kPlayer) ||
                              (channel == 1 && piece == CellValue::kOpponent);
        buffer[channel * channel_length + row * kCols + col] = occupied ? 1.0F : 0.0F;
      }
    }
  }
  return buffer;
}

std::string Position::to_string() const {
  std::string result;
  result.reserve(kRows * kCols * kPlayerToken.size() + kRows - 1);
  for (std::size_t display_row = 0; display_row < kRows; ++display_row) {
    const auto row = kRows - 1 - display_row;
    for (std::size_t col = 0; col < kCols; ++col) {
      const auto piece = get(row, col);
      if (piece == CellValue::kPlayer) {
        result.append(kPlayerToken);
      } else if (piece == CellValue::kOpponent) {
        result.append(kOpponentToken);
      } else {
        result.append(kEmptyToken);
      }
    }
    if (display_row + 1 != kRows) {
      result.push_back('\n');
    }
  }
  return result;
}

void Position::set_piece(std::size_t row, std::size_t col,
                         std::optional<CellValue> piece) noexcept {
  const auto cell = bit(row, col);
  if (!piece.has_value()) {
    mask_ &= ~cell;
    value_ &= ~cell;
  } else if (*piece == CellValue::kOpponent) {
    mask_ |= cell;
    value_ &= ~cell;
  } else {
    mask_ |= cell;
    value_ |= cell;
  }
}

bool Position::player_has_won() const noexcept {
  const auto player_pieces = mask_ & value_;
  return std::ranges::any_of(kWinMasks, [player_pieces](std::uint64_t win) {
    return (player_pieces & win) == win;
  });
}

bool Position::to_moves_recursive(Position temporary,
                                  std::vector<Move>& removals) const {
  if (temporary.ply() == 0) {
    return true;
  }

  const bool removing_player0 = (ply() % 2 == 0) != (temporary.ply() % 2 == 0);
  for (std::size_t col = 0; col < kCols; ++col) {
    for (std::size_t row = kRows; row-- > 0;) {
      const auto original_piece = get(row, col);
      const auto temporary_piece = temporary.get(row, col);
      const auto expected =
          removing_player0 ? CellValue::kPlayer : CellValue::kOpponent;
      if (original_piece == expected && temporary_piece == expected) {
        auto next = temporary;
        next.set_piece(row, col, std::nullopt);
        removals.push_back(col);
        if (to_moves_recursive(next, removals)) {
          return true;
        }
        removals.pop_back();
        break;
      }
      if (!temporary_piece.has_value()) {
        continue;
      }
      break;
    }
  }
  return false;
}

std::size_t PositionHash::operator()(const Position& position) const noexcept {
  const auto mask_hash = mix_hash(position.mask());
  const auto value_hash = mix_hash(position.value() ^ 0xd6e8feb86659fd93ULL);
  return static_cast<std::size_t>(mask_hash ^ std::rotl(value_hash, 29));
}

Sample Sample::flip_horizontal() const {
  Sample result{.pos = pos.flip_horizontal(),
                .policy = {},
                .q_penalty = q_penalty,
                .q_no_penalty = q_no_penalty};
  for (std::size_t col = 0; col < kCols; ++col) {
    result.policy[col] = policy[kCols - 1 - col];
  }
  return result;
}

float GameResult::player0_score() const {
  for (const auto& sample : samples) {
    const auto terminal = sample.pos.terminal_state();
    if (!terminal.has_value()) {
      continue;
    }
    float score = 0.5F;
    if (*terminal == TerminalState::kPlayerWin) {
      score = 1.0F;
    } else if (*terminal == TerminalState::kOpponentWin) {
      score = 0.0F;
    }
    return sample.pos.ply() % 2 == 1 ? 1.0F - score : score;
  }
  throw std::logic_error("player0_score called on an unfinished game");
}

PlayGamesResult PlayGamesResult::combined(const PlayGamesResult& other) const {
  PlayGamesResult result;
  result.results.reserve(results.size() + other.results.size());
  result.results.insert(result.results.end(), results.begin(), results.end());
  result.results.insert(result.results.end(), other.results.begin(),
                        other.results.end());
  return result;
}

std::pair<std::vector<Sample>, std::vector<Sample>> PlayGamesResult::split_train_test(
    float train_fraction, std::uint64_t seed) const {
  if (!std::isfinite(train_fraction) || train_fraction < 0.0F ||
      train_fraction > 1.0F) {
    throw std::invalid_argument("train_fraction must be in [0, 1]");
  }

  std::vector<std::size_t> indices(results.size());
  for (std::size_t index = 0; index < indices.size(); ++index) {
    indices[index] = index;
  }
  deterministic_shuffle(indices, seed);
  const auto train_count = static_cast<std::size_t>(
      std::round(static_cast<float>(results.size()) * train_fraction));

  std::vector<Sample> train;
  std::vector<Sample> test;
  for (std::size_t shuffled_index = 0; shuffled_index < indices.size();
       ++shuffled_index) {
    const auto& samples = results[indices[shuffled_index]].samples;
    auto& destination = shuffled_index < train_count ? train : test;
    destination.insert(destination.end(), samples.begin(), samples.end());
  }
  return {std::move(train), std::move(test)};
}

std::size_t PlayGamesResult::unique_positions() const {
  std::unordered_set<Position, PositionHash> positions;
  for (const auto& result : results) {
    for (const auto& sample : result.samples) {
      positions.insert(sample.pos);
    }
  }
  return positions.size();
}

}  // namespace c4a0
