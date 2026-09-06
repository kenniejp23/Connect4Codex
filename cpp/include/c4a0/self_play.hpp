#pragma once

#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include "c4a0/mcts.hpp"

namespace c4a0 {

struct GameRequest {
  GameMetadata metadata;
  std::vector<Move> opening_moves;

  GameRequest() = default;
  GameRequest(GameMetadata game_metadata, std::vector<Move> moves = {})
      : metadata(game_metadata), opening_moves(std::move(moves)) {}
};

struct SelfPlayOptions {
  std::size_t max_nn_batch_size{};
  std::size_t n_mcts_iterations{};
  float c_exploration{};
  float c_ply_penalty{};
  float root_dirichlet_alpha{};
  float root_dirichlet_epsilon{};
  std::size_t temperature_midpoint_ply{4};
  std::size_t temperature_cutoff_ply{8};
  float early_temperature{1.0F};
  float middle_temperature{1.0F};
  float late_temperature{};
  std::uint64_t seed{1337};
  std::size_t worker_threads{};
};

struct SelfPlayTelemetry {
  std::size_t completed_games{};
  std::size_t total_games{};
  std::size_t neural_evaluations{};
  std::size_t mcts_iterations{};
  std::size_t neural_queue_depth{};
  std::size_t mcts_queue_depth{};
  double elapsed_seconds{};
  bool final{};
};

using SelfPlayProgressCallback =
    std::function<void(std::size_t completed_games, std::size_t total_games)>;
using SelfPlayTelemetryCallback = std::function<void(const SelfPlayTelemetry&)>;
using CancellationCallback = std::function<bool()>;

[[nodiscard]] std::vector<GameResult> self_play(
    Evaluator& evaluator, std::vector<GameMetadata> requests,
    std::size_t max_nn_batch_size, std::size_t n_mcts_iterations, float c_exploration,
    float c_ply_penalty, SelfPlayProgressCallback progress = {},
    CancellationCallback cancelled = {});

[[nodiscard]] std::vector<GameResult> self_play(
    Evaluator& evaluator, std::vector<GameRequest> requests,
    const SelfPlayOptions& options, SelfPlayProgressCallback progress = {},
    CancellationCallback cancelled = {}, SelfPlayTelemetryCallback telemetry = {});

}  // namespace c4a0
