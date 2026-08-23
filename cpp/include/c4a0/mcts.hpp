#pragma once

#include <memory>
#include <vector>

#include "c4a0/position.hpp"

namespace c4a0 {

[[nodiscard]] Policy softmax(Policy policy_logits);
[[nodiscard]] Policy apply_temperature(const Policy& policy, float temperature);

class MctsGame {
 public:
  static constexpr Policy kUniformPolicy = {
      1.0F / static_cast<float>(kCols), 1.0F / static_cast<float>(kCols),
      1.0F / static_cast<float>(kCols), 1.0F / static_cast<float>(kCols),
      1.0F / static_cast<float>(kCols), 1.0F / static_cast<float>(kCols),
      1.0F / static_cast<float>(kCols)};

  explicit MctsGame(Position position = {}, GameMetadata metadata = {});
  ~MctsGame();
  MctsGame(MctsGame&&) noexcept;
  MctsGame& operator=(MctsGame&&) noexcept;
  MctsGame(const MctsGame&) = delete;
  MctsGame& operator=(const MctsGame&) = delete;

  [[nodiscard]] Position root_position() const;
  [[nodiscard]] Position leaf_position() const;
  [[nodiscard]] ModelId leaf_model_id_to_play() const;
  void receive_evaluation(Policy policy_logits, QValue q_penalty, QValue q_no_penalty,
                          float c_exploration, float c_ply_penalty);
  void make_move(Move move, float c_exploration);
  void make_random_move(float c_exploration, float temperature);
  void reset();
  [[nodiscard]] bool undo();
  [[nodiscard]] std::size_t root_visit_count() const;
  [[nodiscard]] Policy root_policy() const;
  [[nodiscard]] QValue root_q_penalty() const;
  [[nodiscard]] QValue root_q_no_penalty() const;
  [[nodiscard]] std::vector<Move> move_history() const;
  [[nodiscard]] GameResult to_result(float c_ply_penalty) &&;

 private:
  struct Node;
  struct RecordedMove;

  void expand_leaf(const Policy& policy);
  [[nodiscard]] Node* materialize_child(Node* parent, std::size_t col);
  void backpropagate(QValue q_penalty, QValue q_no_penalty);
  void select_leaf(float c_exploration);

  GameMetadata metadata_;
  std::unique_ptr<Node> root_;
  Node* leaf_{};
  std::vector<RecordedMove> moves_;
};

}  // namespace c4a0
