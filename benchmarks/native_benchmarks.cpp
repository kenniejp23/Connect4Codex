#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

#include "c4a0/mcts.hpp"
#include "c4a0/self_play.hpp"

namespace {

class UniformEvaluator final : public c4a0::Evaluator {
 public:
  std::vector<c4a0::EvalPosResult> evaluate(
      c4a0::ModelId, const std::vector<c4a0::Position>& positions) override {
    return std::vector<c4a0::EvalPosResult>(
        positions.size(),
        c4a0::EvalPosResult{.policy = {}, .q_penalty = 0.0F, .q_no_penalty = 0.0F});
  }
};

template <typename Function>
double timed_seconds(Function&& function) {
  const auto start = std::chrono::steady_clock::now();
  function();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

}  // namespace

int main(int argc, char** argv) {
  bool self_play_only = false;
  if (argc == 2 && std::string_view(argv[1]) == "--self-play-only") {
    self_play_only = true;
  } else if (argc != 1) {
    std::cerr << "usage: c4a0_benchmarks [--self-play-only]\n";
    return 2;
  }

  std::size_t accumulator = 0;
  double position_ops_per_second = 0.0;
  double mcts_iterations_per_second = 0.0;
  if (!self_play_only) {
    constexpr std::size_t kPositionIterations = 10'000'000;
    c4a0::Position position = c4a0::Position::from_moves({3, 2, 3, 2, 4});
    const double position_seconds = timed_seconds([&] {
      for (std::size_t i = 0; i < kPositionIterations; ++i) {
        accumulator += position.legal_moves()[i % c4a0::kCols] ? 1U : 0U;
        accumulator += position.terminal_state().has_value() ? 1U : 0U;
        position = position.flip_horizontal();
      }
    });
    position_ops_per_second =
        static_cast<double>(kPositionIterations) / position_seconds;

    // Chunking keeps peak memory bounded while still timing enough tree work
    // for a stable measurement. Tree destruction is intentionally included.
    constexpr std::size_t kMctsIterations = 1'000'000;
    constexpr std::size_t kMctsChunkIterations = 50'000;
    const double mcts_seconds = timed_seconds([&] {
      for (std::size_t completed = 0; completed < kMctsIterations;
           completed += kMctsChunkIterations) {
        c4a0::MctsGame game;
        for (std::size_t i = 0; i < kMctsChunkIterations; ++i) {
          game.receive_evaluation(c4a0::MctsGame::kUniformPolicy, 0.0F, 0.0F, 4.0F,
                                  0.01F);
        }
        accumulator += game.root_visit_count();
      }
    });
    mcts_iterations_per_second = static_cast<double>(kMctsIterations) / mcts_seconds;
  }

  constexpr std::size_t kSelfPlayGames = 512;
  UniformEvaluator evaluator;
  std::vector<c4a0::GameMetadata> requests;
  requests.reserve(kSelfPlayGames);
  for (std::uint64_t game_id = 0; game_id < kSelfPlayGames; ++game_id) {
    requests.push_back({.game_id = game_id, .player0_id = 0, .player1_id = 0});
  }
  const double self_play_seconds = timed_seconds([&] {
    const auto results = c4a0::self_play(evaluator, requests, 32, 64, 4.0F, 0.01F);
    accumulator += results.size();
  });

  std::cout << std::fixed << std::setprecision(3);
  if (!self_play_only) {
    std::cout << "position_ops_per_second=" << position_ops_per_second << '\n'
              << "mcts_iterations_per_second=" << mcts_iterations_per_second << '\n';
  }
  std::cout << "self_play_games_per_second="
            << static_cast<double>(kSelfPlayGames) / self_play_seconds << '\n'
            << "checksum=" << accumulator << '\n';
}
