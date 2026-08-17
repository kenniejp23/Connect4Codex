#pragma once

#include "c4a0/interactive.hpp"

namespace c4a0 {

void run_tui(Evaluator& evaluator, std::size_t max_mcts_iterations, float c_exploration,
             float c_ply_penalty, bool auto_red = false, bool auto_blue = false);

}  // namespace c4a0
