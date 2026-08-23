#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "c4a0/mcts.hpp"

namespace c4a0 {

using SelfPlayProgressCallback =
    std::function<void(std::size_t completed_games, std::size_t total_games)>;
using CancellationCallback = std::function<bool()>;

[[nodiscard]] std::vector<GameResult> self_play(
    Evaluator& evaluator, std::vector<GameMetadata> requests,
    std::size_t max_nn_batch_size, std::size_t n_mcts_iterations, float c_exploration,
    float c_ply_penalty, SelfPlayProgressCallback progress = {},
    CancellationCallback cancelled = {});

}  // namespace c4a0
