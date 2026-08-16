#include <algorithm>
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "c4a0/mcts.hpp"
#include "c4a0/position.hpp"
#include "c4a0/rng.hpp"

namespace c4a0 {
namespace {

using Catch::Approx;

constexpr float kExploration = 4.0F;
constexpr float kPlyPenalty = 0.01F;

void run_mcts(MctsGame& game, std::size_t iterations, Policy logits = {}) {
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    game.receive_evaluation(logits, 0.0F, 0.0F, kExploration, kPlyPenalty);
  }
}

void check_policy(const Policy& policy) {
  CHECK(std::accumulate(policy.begin(), policy.end(), 0.0F) ==
        Approx(1.0F).margin(1.0e-5F));
  for (const auto probability : policy) {
    CHECK(probability >= 0.0F);
    CHECK(probability <= 1.0F);
  }
}

}  // namespace

TEST_CASE("position moves use the side-to-play representation and strict bounds") {
  Position position;
  for (std::size_t row = 0; row < kRows; ++row) {
    const auto moved = position.make_move(0);
    REQUIRE(moved.has_value());
    position = *moved;
    CHECK(position.get(row, 0) == CellValue::kOpponent);
  }

  CHECK_FALSE(position.make_move(0).has_value());
  CHECK_FALSE(position.make_move(kCols).has_value());
  CHECK_FALSE(position.get(kRows, 0).has_value());
  CHECK_FALSE(position.get(0, kCols).has_value());
  CHECK_THROWS_AS(Position::from_moves({0, 0, 0, 0, 0, 0, 0}), std::invalid_argument);
  CHECK_THROWS_AS(Position::from_bits(std::uint64_t{1} << 42U, 0),
                  std::invalid_argument);
  CHECK_THROWS_AS(Position::from_bits(0, 1), std::invalid_argument);
}

TEST_CASE("position detects every win direction and draw") {
  const auto horizontal = Position::from_string(
      "⚫⚫⚫⚫⚫⚫⚫\n"
      "⚫⚫⚫⚫⚫⚫⚫\n"
      "⚫⚫⚫⚫⚫⚫⚫\n"
      "⚫⚫⚫⚫⚫⚫⚫\n"
      "⚫⚫⚫⚫⚫⚫⚫\n"
      "🔴🔴🔴🔴⚫⚫⚫");
  const auto vertical = Position::from_string(
      "⚫⚫⚫⚫⚫⚫⚫\n"
      "⚫⚫⚫⚫⚫⚫⚫\n"
      "🔴⚫⚫⚫⚫⚫⚫\n"
      "🔴⚫⚫⚫⚫⚫⚫\n"
      "🔴⚫⚫⚫⚫⚫⚫\n"
      "🔴⚫⚫⚫⚫⚫⚫");
  const auto ascending = Position::from_string(
      "⚫⚫⚫⚫⚫⚫⚫\n"
      "⚫⚫⚫⚫⚫⚫⚫\n"
      "⚫⚫⚫🔴⚫⚫⚫\n"
      "⚫⚫🔴⚫⚫⚫⚫\n"
      "⚫🔴⚫⚫⚫⚫⚫\n"
      "🔴⚫⚫⚫⚫⚫⚫");
  const auto descending = Position::from_string(
      "⚫⚫⚫⚫⚫⚫⚫\n"
      "⚫⚫⚫⚫⚫⚫⚫\n"
      "🔴⚫⚫⚫⚫⚫⚫\n"
      "⚫🔴⚫⚫⚫⚫⚫\n"
      "⚫⚫🔴⚫⚫⚫⚫\n"
      "⚫⚫⚫🔴⚫⚫⚫");

  CHECK(horizontal.terminal_state() == TerminalState::kPlayerWin);
  CHECK(vertical.terminal_state() == TerminalState::kPlayerWin);
  CHECK(ascending.terminal_state() == TerminalState::kPlayerWin);
  CHECK(descending.terminal_state() == TerminalState::kPlayerWin);
  CHECK(horizontal.inverted().terminal_state() == TerminalState::kOpponentWin);

  const auto draw = Position::from_moves({0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1,
                                          2, 3, 4, 5, 5, 4, 3, 2, 1, 0, 5, 4, 3, 2,
                                          1, 0, 5, 4, 3, 2, 1, 0, 6, 6, 6, 6, 6, 6});
  CHECK(draw.terminal_state() == TerminalState::kDraw);
  CHECK(draw.terminal_values(kPlyPenalty) == std::pair<QValue, QValue>{0.0F, 0.0F});
}

TEST_CASE("position string, move, reflection, and tensor representations round trip") {
  Pcg32 rng(0x5eedULL);
  for (std::size_t game_index = 0; game_index < 128; ++game_index) {
    Position position;
    for (std::size_t attempt = 0; attempt < 100; ++attempt) {
      if (position.terminal_state().has_value()) {
        break;
      }
      const auto move = static_cast<Move>(rng.next_u32() % kCols);
      if (const auto next = position.make_move(move); next.has_value()) {
        position = *next;
      }
    }

    CHECK(Position::from_string(position.to_string()) == position);
    CHECK(Position::from_moves(position.to_moves()) == position);
    CHECK(position.flip_horizontal().flip_horizontal() == position);

    const auto buffer = position.to_buffer();
    CHECK(std::count(buffer.begin(), buffer.end(), 1.0F) ==
          static_cast<std::ptrdiff_t>(position.ply()));
    for (std::size_t row = 0; row < kRows; ++row) {
      for (std::size_t col = 0; col < kCols; ++col) {
        const auto index = row * kCols + col;
        CHECK(buffer[index] + buffer[kRows * kCols + index] <= 1.0F);
      }
    }
  }

  CHECK_THROWS_AS(Position::from_string("⚫⚫⚫⚫⚫⚫⚫"), std::invalid_argument);
}

TEST_CASE("policy masking, softmax, and temperature are stable") {
  auto position = Position::from_moves({0, 0, 0, 0, 0, 0});
  Policy logits{1000.0F, 999.0F, 998.0F, 997.0F, 996.0F, 995.0F, 994.0F};
  position.mask_policy(logits);
  CHECK(logits[0] == -std::numeric_limits<float>::infinity());
  const auto policy = softmax(logits);
  check_policy(policy);
  CHECK(policy[0] == 0.0F);

  CHECK(apply_temperature(policy, 1.0F) == policy);
  const auto hot = apply_temperature(policy, 2.0F);
  check_policy(hot);
  CHECK(hot[1] < policy[1]);

  const Policy ties{0.4F, 0.4F, 0.2F, 0.0F, 0.0F, 0.0F, 0.0F};
  const auto cold = apply_temperature(ties, 0.0F);
  CHECK(cold == Policy{0.5F, 0.5F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});

  const auto negative_infinity = -std::numeric_limits<float>::infinity();
  CHECK_THROWS_AS(
      softmax(Policy{negative_infinity, negative_infinity, negative_infinity,
                     negative_infinity, negative_infinity, negative_infinity,
                     negative_infinity}),
      std::invalid_argument);
  CHECK_THROWS_AS(softmax(Policy{0.0F, 0.0F, std::numeric_limits<float>::quiet_NaN(),
                                 0.0F, 0.0F, 0.0F, 0.0F}),
                  std::invalid_argument);
  CHECK_THROWS_AS(apply_temperature(policy, -1.0F), std::invalid_argument);
}

TEST_CASE("PCG and categorical sampling have a portable deterministic contract") {
  Pcg32 rng(42);
  const std::array<std::uint32_t, 5> expected{1307692281U, 3850602322U, 1491967504U,
                                              4091771729U, 3882238836U};
  for (const auto value : expected) {
    CHECK(rng.next_u32() == value);
  }

  Pcg32 sample_rng(42);
  CHECK(sample_policy(Policy{0.0F, 0.1F, 0.2F, 0.3F, 0.4F, 0.0F, 0.0F}, sample_rng) ==
        3);
  CHECK_THROWS_AS(sample_policy(Policy{}, sample_rng), std::invalid_argument);

  std::vector<std::size_t> first{0, 1, 2, 3, 4, 5, 6, 7};
  auto second = first;
  deterministic_shuffle(first, 1337);
  deterministic_shuffle(second, 1337);
  CHECK(first == second);
  CHECK(first != std::vector<std::size_t>{0, 1, 2, 3, 4, 5, 6, 7});
}

TEST_CASE("MCTS preserves reference visit, value, and model-routing semantics") {
  MctsGame game(Position{},
                GameMetadata{.game_id = 4, .player0_id = 11, .player1_id = 22});
  CHECK(game.leaf_model_id_to_play() == 11);
  CHECK(game.root_visit_count() == 0);
  CHECK(game.root_policy() == MctsGame::kUniformPolicy);

  run_mcts(game, 1);
  CHECK(game.root_visit_count() == 1);
  CHECK(game.leaf_position().ply() == 1);
  CHECK(game.leaf_model_id_to_play() == 22);

  run_mcts(game, 14);
  CHECK(game.root_visit_count() == 15);
  const auto policy = game.root_policy();
  check_policy(policy);
  for (const auto probability : policy) {
    CHECK(probability == Approx(1.0F / 7.0F).margin(1.0e-7F));
  }

  const auto selected = static_cast<Move>(
      std::distance(policy.begin(), std::max_element(policy.begin(), policy.end())));
  game.make_move(selected, kExploration);
  CHECK(game.root_position().ply() == 1);
  CHECK(game.root_visit_count() > 0);
  CHECK(game.undo());
  CHECK(game.root_position() == Position{});
  CHECK(game.root_visit_count() == 0);
  CHECK_FALSE(game.undo());
}

TEST_CASE("MCTS early uniform visits preserve lazy-edge behavior", "[mcts][lazy]") {
  MctsGame game;

  run_mcts(game, 1);
  CHECK(game.root_visit_count() == 1);
  CHECK(game.root_policy() == MctsGame::kUniformPolicy);
  CHECK(game.leaf_position() == Position::from_moves({6}));

  run_mcts(game, 1);
  CHECK(game.root_visit_count() == 2);
  CHECK(game.leaf_position() == Position::from_moves({5}));
  CHECK(game.root_policy() == Policy{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F});

  run_mcts(game, 1);
  CHECK(game.root_visit_count() == 3);
  CHECK(game.leaf_position() == Position::from_moves({4}));
  CHECK(game.root_policy() == Policy{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.5F, 0.5F});

  run_mcts(game, 4);
  CHECK(game.root_visit_count() == 7);
  CHECK(game.leaf_position() == Position::from_moves({0}));
  const auto before_every_child_is_visited = game.root_policy();
  CHECK(before_every_child_is_visited[0] == 0.0F);
  for (std::size_t col = 1; col < kCols; ++col) {
    CHECK(before_every_child_is_visited[col] == Approx(1.0F / 6.0F));
  }

  run_mcts(game, 1);
  CHECK(game.root_visit_count() == 8);
  CHECK(game.leaf_position() == Position::from_moves({6, 6}));
  const auto after_every_child_is_visited = game.root_policy();
  for (const auto probability : after_every_child_is_visited) {
    CHECK(probability == Approx(1.0F / 7.0F));
  }
}

TEST_CASE("MCTS promotes unvisited legal edges beside a full column", "[mcts][lazy]") {
  const auto position = Position::from_moves({0, 0, 0, 0, 0, 0});
  REQUIRE_FALSE(position.make_move(0).has_value());
  MctsGame game(position);

  run_mcts(game, 1);
  const auto expected_six = position.make_move(6);
  REQUIRE(expected_six.has_value());
  CHECK(game.leaf_position() == *expected_six);
  CHECK(game.root_policy() == MctsGame::kUniformPolicy);

  run_mcts(game, 1);
  const auto expected_five = position.make_move(5);
  REQUIRE(expected_five.has_value());
  CHECK(game.leaf_position() == *expected_five);
  const auto policy = game.root_policy();
  CHECK(policy[0] == 0.0F);
  CHECK(policy[6] == 1.0F);

  CHECK_THROWS_AS(game.make_move(0, kExploration), std::invalid_argument);
  CHECK(game.root_position() == position);
  CHECK(game.root_visit_count() == 2);

  const auto expected = position.make_move(1);
  REQUIRE(expected.has_value());
  game.make_move(1, kExploration);
  CHECK(game.root_position() == *expected);
  CHECK(game.root_visit_count() == 0);
  CHECK(game.root_policy() == MctsGame::kUniformPolicy);
}

TEST_CASE("MCTS promotion retains a visited lazy subtree", "[mcts][lazy]") {
  MctsGame game;
  run_mcts(game, 9);

  REQUIRE(game.root_visit_count() == 9);
  const auto parent_policy = game.root_policy();
  const auto retained_visits = static_cast<std::size_t>(
      std::lround(parent_policy[6] * static_cast<float>(game.root_visit_count() - 1)));
  REQUIRE(retained_visits == 2);

  game.make_move(6, kExploration);
  CHECK(game.root_position() == Position::from_moves({6}));
  CHECK(game.root_visit_count() == retained_visits);
  CHECK(game.root_policy() == Policy{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F});

  run_mcts(game, 1);
  CHECK(game.root_visit_count() == retained_visits + 1);
  CHECK(game.root_policy() == Policy{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.5F, 0.5F});
}

TEST_CASE("MCTS strongly prefers immediate winning moves") {
  const auto position = Position::from_string(
      "⚫⚫⚫⚫⚫⚫⚫\n"
      "⚫⚫⚫⚫⚫⚫⚫\n"
      "⚫⚫⚫⚫⚫⚫⚫\n"
      "⚫⚫⚫⚫⚫⚫⚫\n"
      "⚫🔵🔵🔵⚫⚫⚫\n"
      "⚫🔴🔴🔴⚫⚫⚫");
  MctsGame game(position);
  run_mcts(game, 10'000);

  const auto policy = game.root_policy();
  check_policy(policy);
  CHECK(policy[0] + policy[4] > 0.99F);
  CHECK(game.root_q_penalty() > 0.92F);
  CHECK(game.root_q_no_penalty() > 0.99F);
  CHECK(game.root_q_no_penalty() > game.root_q_penalty());
}

TEST_CASE("MCTS result records alternating targets and a terminal sample") {
  const auto near_win = Position::from_moves({0, 0, 1, 1, 2, 2});
  MctsGame game(near_win, GameMetadata{.game_id = 7, .player0_id = 2, .player1_id = 3});
  run_mcts(game, 100);
  game.make_move(3, kExploration);
  REQUIRE(game.root_position().terminal_state() == TerminalState::kOpponentWin);

  auto result = std::move(game).to_result(kPlyPenalty);
  REQUIRE(result.samples.size() == 2);
  CHECK(result.metadata.game_id == 7);
  CHECK(result.samples.front().pos == near_win);
  CHECK(result.samples.front().q_no_penalty == 1.0F);
  CHECK(result.samples.back().policy == MctsGame::kUniformPolicy);
  CHECK(result.samples.back().q_no_penalty == -1.0F);
  CHECK(result.player0_score() == 1.0F);
}

TEST_CASE("samples and aggregate results preserve games and split deterministically") {
  const Sample sample{.pos = Position::from_moves({0, 1, 2}),
                      .policy = {0.0F, 0.1F, 0.2F, 0.3F, 0.4F, 0.0F, 0.0F},
                      .q_penalty = 0.25F,
                      .q_no_penalty = 0.5F};
  const auto flipped = sample.flip_horizontal();
  CHECK(flipped.flip_horizontal() == sample);
  CHECK(flipped.policy == Policy{0.0F, 0.0F, 0.4F, 0.3F, 0.2F, 0.1F, 0.0F});

  PlayGamesResult first;
  PlayGamesResult second;
  for (std::uint64_t game_id = 0; game_id < 4; ++game_id) {
    first.results.push_back(GameResult{
        .metadata = {.game_id = game_id},
        .samples = {
            Sample{.pos = Position::from_moves({static_cast<Move>(game_id)})}}});
  }
  second.results.push_back(
      GameResult{.metadata = {.game_id = 10}, .samples = {sample}});

  const auto combined = first.combined(second);
  CHECK(combined.results.size() == 5);
  CHECK(first.results.size() == 4);
  CHECK(combined.unique_positions() == 5);

  const auto [train1, test1] = first.split_train_test(0.5F, 1337);
  const auto [train2, test2] = first.split_train_test(0.5F, 1337);
  CHECK(train1 == train2);
  CHECK(test1 == test2);
  CHECK(train1.size() == 2);
  CHECK(test1.size() == 2);
  CHECK_THROWS_AS(first.split_train_test(-0.1F, 0), std::invalid_argument);
  CHECK_THROWS_AS(first.split_train_test(std::numeric_limits<float>::quiet_NaN(), 0),
                  std::invalid_argument);
}

}  // namespace c4a0
