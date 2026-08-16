#pragma once

#include <cstddef>
#include <vector>

#include "c4a0/mcts.hpp"

namespace c4a0 {

[[nodiscard]] std::vector<GameResult> self_play(Evaluator& evaluator,
                                                std::vector<GameMetadata> requests,
                                                std::size_t max_nn_batch_size,
                                                std::size_t n_mcts_iterations,
                                                float c_exploration,
                                                float c_ply_penalty);

}  // namespace c4a0
