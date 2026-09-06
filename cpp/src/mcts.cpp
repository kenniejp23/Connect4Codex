#include "c4a0/mcts.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "c4a0/rng.hpp"

namespace c4a0 {
namespace {

constexpr float kNodeEpsilon = 1.0e-8F;

bool all_equal(const Policy& policy) noexcept {
  return std::all_of(policy.begin() + 1, policy.end(),
                     [&policy](float value) { return value == policy[0]; });
}

Policy mix_dirichlet_noise(const Position& position, Policy policy, float alpha,
                           float epsilon, std::uint64_t seed) {
  if (!std::isfinite(alpha) || alpha <= 0.0F || !std::isfinite(epsilon) ||
      epsilon <= 0.0F || epsilon > 1.0F) {
    throw std::invalid_argument("invalid root Dirichlet noise parameters");
  }
  const auto legal = position.legal_moves();
  Pcg32 rng(seed);
  std::gamma_distribution<float> gamma(alpha, 1.0F);
  Policy noise{};
  float noise_sum = 0.0F;
  for (std::size_t col = 0; col < kCols; ++col) {
    if (legal[col]) {
      noise[col] = gamma(rng);
      noise_sum += noise[col];
    }
  }
  if (!(noise_sum > 0.0F) || !std::isfinite(noise_sum)) {
    throw std::runtime_error("could not generate root Dirichlet noise");
  }
  for (std::size_t col = 0; col < kCols; ++col) {
    if (legal[col]) {
      noise[col] /= noise_sum;
      policy[col] = (1.0F - epsilon) * policy[col] + epsilon * noise[col];
    }
  }
  return policy;
}

}  // namespace

Policy softmax(Policy policy_logits) {
  float maximum = -std::numeric_limits<float>::infinity();
  for (const auto value : policy_logits) {
    if (std::isnan(value) || value == std::numeric_limits<float>::infinity()) {
      throw std::invalid_argument(
          "softmax policy cannot contain NaN or positive infinity");
    }
    maximum = std::max(maximum, value);
  }
  if (!std::isfinite(maximum)) {
    throw std::invalid_argument(
        "softmax policy must contain at least one finite value");
  }
  if (all_equal(policy_logits)) {
    // exp(0) is exactly 1 for every element, so the general path below would
    // produce this same distribution. Avoid seven transcendental calls for
    // uniform evaluators and symmetric early-network outputs.
    return MctsGame::kUniformPolicy;
  }

  float sum = 0.0F;
  for (auto& value : policy_logits) {
    value = std::exp(value - maximum);
    sum += value;
  }
  if (!(sum > 0.0F) || !std::isfinite(sum)) {
    throw std::invalid_argument("softmax policy could not be normalized");
  }
  for (auto& value : policy_logits) {
    value /= sum;
  }
  return policy_logits;
}

Policy apply_temperature(const Policy& policy, float temperature) {
  if (!std::isfinite(temperature) || temperature < 0.0F) {
    throw std::invalid_argument("temperature must be finite and non-negative");
  }
  for (const auto probability : policy) {
    if (!std::isfinite(probability) || probability < 0.0F) {
      throw std::invalid_argument(
          "policy probabilities must be finite and non-negative");
    }
  }

  if (temperature == 1.0F || all_equal(policy)) {
    return policy;
  }
  if (temperature == 0.0F) {
    const auto maximum = *std::max_element(policy.begin(), policy.end());
    Policy result{};
    float count = 0.0F;
    for (std::size_t col = 0; col < kCols; ++col) {
      if (policy[col] == maximum) {
        result[col] = 1.0F;
        count += 1.0F;
      }
    }
    for (auto& probability : result) {
      probability /= count;
    }
    return result;
  }

  Policy result{};
  float sum = 0.0F;
  const auto inverse_temperature = 1.0F / temperature;
  for (std::size_t col = 0; col < kCols; ++col) {
    result[col] = std::pow(policy[col], inverse_temperature);
    sum += result[col];
  }
  if (!(sum > 0.0F) || !std::isfinite(sum)) {
    throw std::invalid_argument("temperature-scaled policy could not be normalized");
  }
  for (auto& probability : result) {
    probability = std::clamp(probability / sum, 0.0F, 1.0F);
  }
  return result;
}

struct MctsGame::Node {
  explicit Node(Position node_position, Node* node_parent, float prior) noexcept
      : position(std::move(node_position)),
        parent(node_parent),
        initial_policy_value(prior) {}

  [[nodiscard]] QValue q_penalty() const noexcept {
    return q_sum_penalty / (static_cast<float>(visit_count) + 1.0F);
  }

  [[nodiscard]] QValue q_no_penalty() const noexcept {
    return q_sum_no_penalty / (static_cast<float>(visit_count) + 1.0F);
  }

  [[nodiscard]] QValue exploration(float parent_log_visits) const noexcept {
    const auto exploration_value =
        std::sqrt(parent_log_visits / (static_cast<float>(visit_count) + 1.0F));
    return exploration_value * (initial_policy_value + kNodeEpsilon);
  }

  [[nodiscard]] QValue uct(float c_exploration,
                           float parent_log_visits) const noexcept {
    return -q_penalty() + c_exploration * exploration(parent_log_visits);
  }

  [[nodiscard]] Policy policy() const noexcept {
    if (!expanded) {
      return MctsGame::kUniformPolicy;
    }
    Policy child_counts{};
    float count_sum = 0.0F;
    for (std::size_t col = 0; col < kCols; ++col) {
      if (children[col] != nullptr) {
        child_counts[col] = static_cast<float>(children[col]->visit_count);
        count_sum += child_counts[col];
      }
    }
    if (count_sum == 0.0F) {
      return MctsGame::kUniformPolicy;
    }
    for (auto& count : child_counts) {
      count /= count_sum;
    }
    return child_counts;
  }

  Position position;
  Node* parent{};
  std::size_t visit_count{};
  QValue q_sum_penalty{};
  QValue q_sum_no_penalty{};
  QValue initial_policy_value{};
  Policy child_priors{};
  std::uint8_t legal_children{};
  bool expanded{};
  std::array<std::unique_ptr<Node>, kCols> children{};
};

struct MctsGame::RecordedMove {
  Position position;
  Policy policy{};
  Move move{};
};

MctsGame::MctsGame(Position position, GameMetadata metadata)
    : metadata_(metadata),
      root_(std::make_unique<Node>(std::move(position), nullptr, 1.0F)),
      leaf_(root_.get()) {}

MctsGame::~MctsGame() = default;
MctsGame::MctsGame(MctsGame&&) noexcept = default;
MctsGame& MctsGame::operator=(MctsGame&&) noexcept = default;

Position MctsGame::root_position() const { return root_->position; }

Position MctsGame::leaf_position() const { return leaf_->position; }

ModelId MctsGame::leaf_model_id_to_play() const {
  return leaf_->position.ply() % 2 == 0 ? metadata_.player0_id : metadata_.player1_id;
}

void MctsGame::receive_evaluation(Policy policy_logits, QValue q_penalty,
                                  QValue q_no_penalty, float c_exploration,
                                  float c_ply_penalty, float root_dirichlet_alpha,
                                  float root_dirichlet_epsilon,
                                  std::uint64_t noise_seed) {
  const auto terminal = leaf_->position.terminal_values(c_ply_penalty);
  if (terminal.has_value()) {
    backpropagate(terminal->first, terminal->second);
    select_leaf(c_exploration);
    return;
  }

  leaf_->position.mask_policy(policy_logits);
  auto policy = softmax(policy_logits);
  if (leaf_ == root_.get() && root_dirichlet_epsilon > 0.0F) {
    policy =
        mix_dirichlet_noise(leaf_->position, policy, root_dirichlet_alpha,
                            root_dirichlet_epsilon, noise_seed ^ metadata_.game_id);
  }
  expand_leaf(policy);
  backpropagate(q_penalty, q_no_penalty);
  select_leaf(c_exploration);
}

void MctsGame::add_root_dirichlet_noise(float alpha, float epsilon, float c_exploration,
                                        std::uint64_t noise_seed) {
  if (epsilon == 0.0F || !root_->expanded ||
      root_->position.terminal_state().has_value()) {
    return;
  }
  root_->child_priors = mix_dirichlet_noise(root_->position, root_->child_priors, alpha,
                                            epsilon, noise_seed ^ metadata_.game_id);
  for (std::size_t col = 0; col < kCols; ++col) {
    if (root_->children[col] != nullptr) {
      root_->children[col]->initial_policy_value = root_->child_priors[col];
    }
  }
  select_leaf(c_exploration);
}

void MctsGame::expand_leaf(const Policy& policy) {
  // receive_evaluation() only expands after proving this leaf is non-terminal.
  // Repeating the 69-mask terminal scan here would duplicate that hot-path work.
  const auto legal = leaf_->position.legal_moves();
  for (std::size_t col = 0; col < kCols; ++col) {
    if (legal[col]) {
      leaf_->legal_children |= static_cast<std::uint8_t>(1U << col);
      leaf_->child_priors[col] = policy[col];
    }
  }
  leaf_->expanded = true;
}

MctsGame::Node* MctsGame::materialize_child(Node* parent, std::size_t col) {
  if (col >= kCols ||
      (parent->legal_children & static_cast<std::uint8_t>(1U << col)) == 0) {
    return nullptr;
  }
  auto& child = parent->children[col];
  if (child == nullptr) {
    const auto child_position = parent->position.make_move(col);
    if (!child_position.has_value()) {
      throw std::logic_error("legal move unexpectedly failed");
    }
    child = std::make_unique<Node>(*child_position, parent, parent->child_priors[col]);
  }
  return child.get();
}

void MctsGame::backpropagate(QValue q_penalty, QValue q_no_penalty) {
  for (auto* node = leaf_; node != nullptr; node = node->parent) {
    ++node->visit_count;
    node->q_sum_penalty += q_penalty;
    node->q_sum_no_penalty += q_no_penalty;
    q_penalty = -q_penalty;
    q_no_penalty = -q_no_penalty;
  }
}

void MctsGame::select_leaf(float c_exploration) {
  auto* node = root_.get();
  while (node->expanded) {
    // Every sibling has the same parent visit count. Computing this logarithm
    // once per tree level preserves the UCT formula while avoiding up to seven
    // identical transcendental calls on each selection step.
    const auto parent_log_visits = std::log(static_cast<float>(node->visit_count));
    Node* best = nullptr;
    std::size_t best_col{};
    float best_score = -std::numeric_limits<float>::infinity();
    bool selected = false;
    for (std::size_t col = 0; col < kCols; ++col) {
      if ((node->legal_children & static_cast<std::uint8_t>(1U << col)) == 0) {
        continue;
      }
      const auto& child = node->children[col];
      const auto score =
          child == nullptr ? c_exploration * (std::sqrt(parent_log_visits) *
                                              (node->child_priors[col] + kNodeEpsilon))
                           : child->uct(c_exploration, parent_log_visits);
      if (std::isnan(score)) {
        throw std::domain_error("MCTS UCT score became NaN");
      }
      // Preserve the reference engine's last-item-wins tie break.
      if (!selected || score >= best_score) {
        best = child.get();
        best_col = col;
        best_score = score;
        selected = true;
      }
    }
    if (!selected) {
      break;
    }
    if (best == nullptr) {
      best = materialize_child(node, best_col);
    }
    if (best == nullptr) {
      break;
    }
    node = best;
  }
  leaf_ = node;
}

void MctsGame::make_move(Move move, float c_exploration) {
  if (move >= kCols) {
    throw std::out_of_range("move column is outside the board");
  }
  if (!root_->expanded) {
    throw std::logic_error("root node has not been expanded");
  }
  if ((root_->legal_children & static_cast<std::uint8_t>(1U << move)) == 0) {
    throw std::invalid_argument("attempted to make an illegal move");
  }

  moves_.push_back(RecordedMove{
      .position = root_->position, .policy = root_->policy(), .move = move});
  static_cast<void>(materialize_child(root_.get(), move));
  auto promoted = std::move(root_->children[move]);
  promoted->parent = nullptr;
  root_ = std::move(promoted);
  select_leaf(c_exploration);
}

void MctsGame::make_random_move(float c_exploration, float temperature,
                                std::uint64_t seed) {
  const auto move_seed =
      metadata_.game_id *
          static_cast<std::uint64_t>(kRows * kCols + root_->position.ply()) ^
      seed;
  Pcg32 rng(move_seed);
  auto policy = root_->policy();
  const auto legal = root_->position.legal_moves();
  for (std::size_t col = 0; col < kCols; ++col) {
    if (!legal[col]) {
      policy[col] = 0.0F;
    }
  }
  const auto tempered = apply_temperature(policy, temperature);
  make_move(sample_policy(tempered, rng), c_exploration);
}

void MctsGame::reset() {
  while (undo()) {
  }
}

bool MctsGame::undo() {
  if (moves_.empty()) {
    return false;
  }
  const auto previous_position = moves_.back().position;
  moves_.pop_back();
  root_ = std::make_unique<Node>(previous_position, nullptr, 1.0F);
  leaf_ = root_.get();
  return true;
}

std::size_t MctsGame::root_visit_count() const { return root_->visit_count; }

Policy MctsGame::root_policy() const { return root_->policy(); }

QValue MctsGame::root_q_penalty() const { return root_->q_penalty(); }

QValue MctsGame::root_q_no_penalty() const { return root_->q_no_penalty(); }

std::vector<Move> MctsGame::move_history() const {
  std::vector<Move> result;
  result.reserve(moves_.size());
  for (const auto& move : moves_) {
    result.push_back(move.move);
  }
  return result;
}

GameResult MctsGame::to_result(float c_ply_penalty) && {
  const auto terminal = root_->position.terminal_values(c_ply_penalty);
  if (!terminal.has_value()) {
    throw std::logic_error("attempted to convert a non-terminal game to a result");
  }

  std::vector<Sample> samples;
  samples.reserve(moves_.size() + 1);
  for (std::size_t index = 0; index < moves_.size(); ++index) {
    const auto positive = ((moves_.size() - index) % 2) == 0;
    const auto sign = positive ? 1.0F : -1.0F;
    samples.push_back(Sample{.pos = moves_[index].position,
                             .policy = moves_[index].policy,
                             .q_penalty = sign * terminal->first,
                             .q_no_penalty = sign * terminal->second});
  }
  samples.push_back(Sample{.pos = root_->position,
                           .policy = kUniformPolicy,
                           .q_penalty = terminal->first,
                           .q_no_penalty = terminal->second});
  return GameResult{.metadata = metadata_, .samples = std::move(samples)};
}

}  // namespace c4a0
