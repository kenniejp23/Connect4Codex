#include <algorithm>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "c4a0/interactive.hpp"
#include "c4a0/self_play.hpp"

namespace c4a0 {
namespace {

class RecordingUniformEvaluator final : public Evaluator {
 public:
  explicit RecordingUniformEvaluator(std::size_t maximum_batch_size)
      : maximum_batch_size_(maximum_batch_size) {}

  std::vector<EvalPosResult> evaluate(ModelId model_id,
                                      const std::vector<Position>& positions) override {
    std::unordered_set<Position> unique(positions.begin(), positions.end());
    {
      std::lock_guard lock(mutex_);
      calls_.emplace_back(model_id, positions.size());
      respected_batch_limit_ =
          respected_batch_limit_ && positions.size() <= maximum_batch_size_;
      batches_were_unique_ = batches_were_unique_ && unique.size() == positions.size();
    }
    return std::vector<EvalPosResult>(positions.size(),
                                      EvalPosResult{.policy = MctsGame::kUniformPolicy,
                                                    .q_penalty = 0.0F,
                                                    .q_no_penalty = 0.0F});
  }

  [[nodiscard]] std::vector<std::pair<ModelId, std::size_t>> calls() const {
    std::lock_guard lock(mutex_);
    return calls_;
  }

  [[nodiscard]] bool respected_batch_limit() const {
    std::lock_guard lock(mutex_);
    return respected_batch_limit_;
  }

  [[nodiscard]] bool batches_were_unique() const {
    std::lock_guard lock(mutex_);
    return batches_were_unique_;
  }

 private:
  const std::size_t maximum_batch_size_;
  mutable std::mutex mutex_;
  std::vector<std::pair<ModelId, std::size_t>> calls_;
  bool respected_batch_limit_{true};
  bool batches_were_unique_{true};
};

class CountingEvaluator final : public Evaluator {
 public:
  std::vector<EvalPosResult> evaluate(ModelId,
                                      const std::vector<Position>& positions) override {
    calls.fetch_add(1);
    return std::vector<EvalPosResult>(positions.size(),
                                      EvalPosResult{.policy = MctsGame::kUniformPolicy,
                                                    .q_penalty = 0.0F,
                                                    .q_no_penalty = 0.0F});
  }

  std::atomic<std::size_t> calls{};
};

class RightBiasedEvaluator final : public Evaluator {
 public:
  std::vector<EvalPosResult> evaluate(ModelId,
                                      const std::vector<Position>& positions) override {
    Policy policy{};
    policy[kCols - 1] = 20.0F;
    return std::vector<EvalPosResult>(
        positions.size(),
        EvalPosResult{.policy = policy, .q_penalty = 0.0F, .q_no_penalty = 0.0F});
  }
};

class ThrowingEvaluator final : public Evaluator {
 public:
  std::vector<EvalPosResult> evaluate(ModelId, const std::vector<Position>&) override {
    throw std::runtime_error("inference failed");
  }
};

class WrongSizeEvaluator final : public Evaluator {
 public:
  std::vector<EvalPosResult> evaluate(ModelId, const std::vector<Position>&) override {
    return {};
  }
};

Snapshot wait_until_search_finishes(InteractivePlay& play) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  Snapshot latest;
  do {
    latest = play.snapshot();
    if (!latest.background_running &&
        (latest.n_mcts_iterations >= latest.max_mcts_iterations ||
         latest.pos.terminal_state().has_value())) {
      return latest;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  } while (std::chrono::steady_clock::now() < deadline);
  throw std::runtime_error("interactive MCTS did not become idle");
}

}  // namespace

TEST_CASE("self-play returns immediately for an empty request list") {
  CountingEvaluator evaluator;
  CHECK(self_play(evaluator, {}, 0, 0, 0.0F, 0.0F).empty());
  CHECK(evaluator.calls.load() == 0);
}

TEST_CASE("self-play batches unique leaves and produces complete games") {
  constexpr std::size_t kMaximumBatch = 2;
  RecordingUniformEvaluator evaluator(kMaximumBatch);
  std::vector<GameMetadata> requests{
      {.game_id = 1, .player0_id = 3, .player1_id = 7},
      {.game_id = 2, .player0_id = 3, .player1_id = 7},
      {.game_id = 3, .player0_id = 3, .player1_id = 7},
  };

  const auto results = self_play(evaluator, requests, kMaximumBatch, 2, 1.0F, 0.01F);

  REQUIRE(results.size() == requests.size());
  CHECK(evaluator.respected_batch_limit());
  CHECK(evaluator.batches_were_unique());
  const auto calls = evaluator.calls();
  REQUIRE_FALSE(calls.empty());
  CHECK(calls.front().second == 1);  // The three initial leaves are deduplicated.
  std::set<ModelId> evaluated_models;
  for (const auto& [model_id, unused_size] : calls) {
    static_cast<void>(unused_size);
    evaluated_models.insert(model_id);
  }
  const std::set<ModelId> expected_models{3, 7};
  CHECK(evaluated_models == expected_models);

  std::vector<std::uint64_t> game_ids;
  for (const auto& result : results) {
    game_ids.push_back(result.metadata.game_id);
    REQUIRE(result.samples.size() >= 7);
    CHECK(std::count_if(
              result.samples.begin(), result.samples.end(),
              [](const Sample& sample) { return sample.pos == Position{}; }) == 1);
    const auto terminal_count = std::count_if(
        result.samples.begin(), result.samples.end(),
        [](const Sample& sample) { return sample.pos.terminal_state().has_value(); });
    CHECK(terminal_count == 1);
    CHECK((result.player0_score() == 0.0F || result.player0_score() == 0.5F ||
           result.player0_score() == 1.0F));
  }
  std::sort(game_ids.begin(), game_ids.end());
  const std::vector<std::uint64_t> expected_game_ids{1, 2, 3};
  CHECK(game_ids == expected_game_ids);
}

TEST_CASE("V2 self-play validates openings and becomes greedy at the cutoff") {
  RightBiasedEvaluator evaluator;
  const std::vector<Move> opening{0, 1, 0, 1, 2, 3, 2, 3};
  SelfPlayOptions options{.max_nn_batch_size = 2,
                          .n_mcts_iterations = 16,
                          .c_exploration = 1.0F,
                          .c_ply_penalty = 0.01F,
                          .root_dirichlet_alpha = 0.3F,
                          .root_dirichlet_epsilon = 0.0F,
                          .temperature_midpoint_ply = 8,
                          .temperature_cutoff_ply = 8,
                          .early_temperature = 1.0F,
                          .middle_temperature = 1.0F,
                          .late_temperature = 0.0F,
                          .seed = 9,
                          .worker_threads = 1};
  const auto results = self_play(
      evaluator,
      {GameRequest{GameMetadata{.game_id = 17, .player0_id = 3, .player1_id = 7},
                   opening}},
      options);

  REQUIRE(results.size() == 1);
  REQUIRE(results.front().samples.size() >= 2);
  CHECK(results.front().metadata.player0_id == 3);
  CHECK(results.front().metadata.player1_id == 7);
  CHECK(results.front().samples.front().pos == Position::from_moves(opening));
  const auto expected = Position::from_moves(opening).make_move(kCols - 1);
  REQUIRE(expected.has_value());
  CHECK(results.front().samples[1].pos == *expected);

  CHECK_THROWS_AS(
      self_play(evaluator,
                {GameRequest{GameMetadata{.game_id = 18}, {0, 1, 0, 1, 0, 1, 0}}},
                options),
      std::invalid_argument);
  CHECK_THROWS_AS(
      self_play(evaluator,
                {GameRequest{GameMetadata{.game_id = 19}, {0, 0, 0, 0, 0, 0, 0}}},
                options),
      std::invalid_argument);
}

TEST_CASE("self-play cancels every worker and propagates evaluator failures") {
  ThrowingEvaluator evaluator;
  const std::vector<GameMetadata> requests{
      {.game_id = 1, .player0_id = 0, .player1_id = 0},
      {.game_id = 2, .player0_id = 0, .player1_id = 0},
  };
  CHECK_THROWS_WITH(self_play(evaluator, requests, 2, 2, 1.0F, 0.01F),
                    "inference failed");
}

TEST_CASE("self-play rejects evaluator result-count mismatches") {
  WrongSizeEvaluator evaluator;
  CHECK_THROWS_WITH(
      self_play(evaluator, {{.game_id = 1, .player0_id = 0, .player1_id = 0}}, 1, 1,
                1.0F, 0.01F),
      "evaluator returned a different number of results than positions");
}

TEST_CASE("interactive search reaches its target and snapshots player zero") {
  RecordingUniformEvaluator evaluator(1);
  InteractivePlay play(evaluator, 3, 1.0F, 0.01F);

  auto snapshot = wait_until_search_finishes(play);
  CHECK(snapshot.n_mcts_iterations >= 3);
  CHECK(snapshot.pos == Position{});

  REQUIRE(play.make_move(0));
  snapshot = wait_until_search_finishes(play);
  CHECK(snapshot.pos == Position::from_moves({0}).inverted());
  CHECK(snapshot.pos.ply() == 1);

  play.increase_mcts_iterations(2);
  snapshot = wait_until_search_finishes(play);
  CHECK(snapshot.max_mcts_iterations == 5);
  CHECK(snapshot.n_mcts_iterations >= 5);

  REQUIRE(play.undo());
  snapshot = wait_until_search_finishes(play);
  CHECK(snapshot.pos == Position{});
  CHECK_FALSE(play.undo());
  CHECK_FALSE(play.make_move(kCols));
}

TEST_CASE("interactive play automatically selects a best move when search is ready") {
  RecordingUniformEvaluator evaluator(1);
  InteractivePlay play(evaluator, 3, 1.0F, 0.01F);

  const auto searched = wait_until_search_finishes(play);
  CHECK(searched.pos.ply() == 0);
  REQUIRE(play.make_best_move_if_ready());
  CHECK(play.snapshot().pos.ply() == 1);
}

TEST_CASE("interactive search preserves and rethrows background errors") {
  ThrowingEvaluator evaluator;
  InteractivePlay play(evaluator, 1, 1.0F, 0.01F);

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  bool saw_error = false;
  do {
    try {
      play.rethrow_background_error();
    } catch (const std::runtime_error& error) {
      CHECK(std::string(error.what()) == "inference failed");
      saw_error = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  } while (std::chrono::steady_clock::now() < deadline);
  CHECK(saw_error);
  CHECK_FALSE(play.snapshot().background_running);
  CHECK_THROWS_WITH(play.rethrow_background_error(), "inference failed");
}

TEST_CASE("interactive search repeatedly shuts down cleanly") {
  for (std::size_t iteration = 0; iteration < 10; ++iteration) {
    RecordingUniformEvaluator evaluator(1);
    InteractivePlay play(evaluator, 1, 1.0F, 0.01F);
    CHECK(wait_until_search_finishes(play).n_mcts_iterations >= 1);
  }
}

}  // namespace c4a0
