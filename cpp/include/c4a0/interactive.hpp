#pragma once

#include <memory>

#include "c4a0/mcts.hpp"

namespace c4a0 {

struct Snapshot {
  Position pos;
  Policy policy{};
  QValue q_penalty{};
  QValue q_no_penalty{};
  std::size_t n_mcts_iterations{};
  std::size_t max_mcts_iterations{};
  float c_exploration{};
  float c_ply_penalty{};
  bool background_running{};
};

class InteractivePlay {
 public:
  InteractivePlay(Evaluator& evaluator, std::size_t max_mcts_iterations,
                  float c_exploration, float c_ply_penalty, Position position = {});
  ~InteractivePlay();
  InteractivePlay(const InteractivePlay&) = delete;
  InteractivePlay& operator=(const InteractivePlay&) = delete;

  [[nodiscard]] Snapshot snapshot() const;
  void increase_mcts_iterations(std::size_t count);
  [[nodiscard]] bool make_move(Move move);
  [[nodiscard]] bool make_random_move(float temperature);
  [[nodiscard]] bool make_best_move_if_ready();
  void reset();
  [[nodiscard]] bool undo();
  void rethrow_background_error();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace c4a0
