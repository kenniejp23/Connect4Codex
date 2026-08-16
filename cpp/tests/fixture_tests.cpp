#include <algorithm>
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "c4a0/mcts.hpp"
#include "c4a0/position.hpp"

namespace c4a0 {
namespace {

using Catch::Approx;
using Json = nlohmann::json;

Json read_fixture(const std::string& filename) {
  std::ifstream stream(std::string{C4A0_FIXTURE_DIR} + '/' + filename);
  if (!stream) {
    throw std::runtime_error("could not open fixture: " + filename);
  }
  return Json::parse(stream);
}

Policy read_policy(const Json& values) {
  REQUIRE(values.size() == kCols);
  Policy policy{};
  for (std::size_t col = 0; col < kCols; ++col) {
    policy[col] = values.at(col).get<float>();
  }
  return policy;
}

std::optional<TerminalState> read_terminal(const Json& value) {
  if (value.is_null()) {
    return std::nullopt;
  }
  const auto name = value.get<std::string>();
  if (name == "player_win") {
    return TerminalState::kPlayerWin;
  }
  if (name == "opponent_win") {
    return TerminalState::kOpponentWin;
  }
  if (name == "draw") {
    return TerminalState::kDraw;
  }
  throw std::runtime_error("unknown terminal fixture value: " + name);
}

}  // namespace

TEST_CASE("language-neutral position parity fixtures remain exact") {
  const auto fixture = read_fixture("position_parity.json");
  REQUIRE(fixture.at("format") == "c4a0.position-fixtures");
  REQUIRE(fixture.at("version") == 1);

  for (const auto& item : fixture.at("positions")) {
    CAPTURE(item);
    std::vector<Move> moves;
    for (const auto& move : item.at("moves")) {
      moves.push_back(move.get<Move>());
    }

    const auto position = Position::from_moves(moves);
    CHECK(position.mask() == item.at("mask").get<std::uint64_t>());
    CHECK(position.value() == item.at("value").get<std::uint64_t>());
    CHECK(position.terminal_state() == read_terminal(item.at("terminal")));

    const auto expected_legal = item.at("legal_moves").get<std::array<bool, kCols>>();
    CHECK(position.legal_moves() == expected_legal);
    CHECK(Position::from_bits(position.mask(), position.value()) == position);
    CHECK(Position::from_moves(position.to_moves()) == position);
  }
}

TEST_CASE("historical MCTS regression vectors remain stable") {
  const auto fixture = read_fixture("mcts_regressions.json");
  REQUIRE(fixture.at("format") == "c4a0.mcts-regressions");
  REQUIRE(fixture.at("version") == 1);

  for (const auto& values : fixture.at("softmax_logits")) {
    CAPTURE(values);
    const auto policy = softmax(read_policy(values));
    CHECK(std::accumulate(policy.begin(), policy.end(), 0.0F) ==
          Approx(1.0F).margin(1.0e-6F));
    CHECK(policy[2] == 0.0F);
    for (std::size_t col = 0; col < kCols; ++col) {
      if (col != 2) {
        CHECK(policy[col] == Approx(1.0F / 6.0F).margin(1.0e-6F));
      }
    }
  }

  for (const auto& values : fixture.at("temperature_policies")) {
    CAPTURE(values);
    const auto policy = read_policy(values);
    CHECK(apply_temperature(policy, 1.0F) == policy);

    const auto sum = std::accumulate(policy.begin(), policy.end(), 0.0F);
    if (sum == 0.0F) {
      // This all-zero historical shrink case intentionally preserves the old
      // implementation's uniform-input no-op behavior at every temperature.
      CHECK(apply_temperature(policy, 0.0F) == policy);
      CHECK(apply_temperature(policy, 2.0F) == policy);
      continue;
    }

    const auto cold = apply_temperature(policy, 0.0F);
    CHECK(std::accumulate(cold.begin(), cold.end(), 0.0F) ==
          Approx(1.0F).margin(1.0e-6F));
    const auto original_max = *std::max_element(policy.begin(), policy.end());
    for (std::size_t col = 0; col < kCols; ++col) {
      CHECK((cold[col] > 0.0F) == (policy[col] == original_max));
    }

    const auto hot = apply_temperature(policy, 2.0F);
    CHECK(std::accumulate(hot.begin(), hot.end(), 0.0F) ==
          Approx(1.0F).margin(1.0e-5F));
    CHECK(std::all_of(hot.begin(), hot.end(), [](float value) {
      return std::isfinite(value) && value >= 0.0F;
    }));
  }
}

}  // namespace c4a0
