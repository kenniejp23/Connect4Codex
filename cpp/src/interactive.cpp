#include "c4a0/interactive.hpp"

#include <cmath>
#include <condition_variable>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

namespace c4a0 {
namespace {

void validate_evaluation(const Position& position, const EvalPosResult& evaluation) {
  if (!std::isfinite(evaluation.q_penalty) || !std::isfinite(evaluation.q_no_penalty)) {
    throw std::invalid_argument("evaluator returned a non-finite Q value");
  }

  for (const float value : evaluation.policy) {
    if (std::isnan(value) || value == std::numeric_limits<float>::infinity()) {
      throw std::invalid_argument("evaluator policy contains NaN or positive infinity");
    }
  }

  Policy legal_policy = evaluation.policy;
  position.mask_policy(legal_policy);
  if (!position.terminal_state().has_value()) {
    bool has_finite_legal_move = false;
    for (const float value : legal_policy) {
      has_finite_legal_move = has_finite_legal_move || std::isfinite(value);
    }
    if (!has_finite_legal_move) {
      throw std::invalid_argument(
          "evaluator policy has no finite logit for a legal move");
    }
  }
}

}  // namespace

struct InteractivePlay::Impl {
  Impl(Evaluator& evaluator_value, std::size_t max_iterations, float exploration,
       float ply_penalty, Position position, GameMetadata metadata)
      : evaluator(evaluator_value),
        game(std::move(position), metadata),
        max_mcts_iterations(max_iterations),
        c_exploration(exploration),
        c_ply_penalty(ply_penalty) {
    if (max_mcts_iterations == 0) {
      throw std::invalid_argument("max_mcts_iterations must be positive");
    }
    if (!std::isfinite(c_exploration) || !std::isfinite(c_ply_penalty)) {
      throw std::invalid_argument("MCTS coefficients must be finite");
    }

    background_running = should_search_locked();
    background = std::jthread(
        [this](std::stop_token stop_token) { background_loop(stop_token); });
  }

  ~Impl() {
    background.request_stop();
    state_changed.notify_all();
    if (background.joinable()) {
      background.join();
    }
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  [[nodiscard]] bool should_search_locked() const {
    return !background_error && game.root_visit_count() < max_mcts_iterations &&
           !game.root_position().terminal_state().has_value();
  }

  void background_loop(std::stop_token stop_token) noexcept {
    std::unique_lock lock(mutex);
    while (!stop_token.stop_requested()) {
      if (!should_search_locked()) {
        background_running = false;
        state_changed.notify_all();
        state_changed.wait(lock, stop_token, [this, stop_token] {
          return stop_token.stop_requested() || should_search_locked();
        });
        if (stop_token.stop_requested()) {
          break;
        }
      }

      background_running = true;
      try {
        const Position leaf = game.leaf_position();
        const ModelId model_id = game.leaf_model_id_to_play();
        auto evaluations = evaluator.evaluate(model_id, std::vector<Position>{leaf});
        if (evaluations.size() != 1) {
          throw std::runtime_error(
              "evaluator returned a different number of results than "
              "positions");
        }
        validate_evaluation(leaf, evaluations.front());
        const auto& evaluation = evaluations.front();
        game.receive_evaluation(evaluation.policy, evaluation.q_penalty,
                                evaluation.q_no_penalty, c_exploration, c_ply_penalty);
      } catch (...) {
        background_error = std::current_exception();
        background_running = false;
        state_changed.notify_all();
        return;
      }

      if (!should_search_locked()) {
        background_running = false;
      }
      state_changed.notify_all();

      // Match the reference implementation's lock-per-tick behavior so snapshots and
      // user moves can interleave with a long search.
      lock.unlock();
      std::this_thread::yield();
      lock.lock();
    }

    background_running = false;
    state_changed.notify_all();
  }

  void throw_background_error_locked() const {
    if (background_error) {
      std::rethrow_exception(background_error);
    }
  }

  void signal_search_locked() {
    background_running = should_search_locked();
    state_changed.notify_all();
  }

  void wait_until_root_is_expanded_locked(std::unique_lock<std::mutex>& lock) {
    if (game.root_position().terminal_state().has_value() ||
        game.root_visit_count() != 0) {
      return;
    }
    state_changed.wait(lock, [this] {
      return background_error || game.root_visit_count() != 0 ||
             game.root_position().terminal_state().has_value();
    });
    throw_background_error_locked();
  }

  Evaluator& evaluator;
  mutable std::mutex mutex;
  std::condition_variable_any state_changed;
  MctsGame game;
  std::size_t max_mcts_iterations;
  float c_exploration;
  float c_ply_penalty;
  bool background_running{};
  std::exception_ptr background_error;
  std::jthread background;
};

InteractivePlay::InteractivePlay(Evaluator& evaluator, std::size_t max_mcts_iterations,
                                 float c_exploration, float c_ply_penalty,
                                 Position position, GameMetadata metadata)
    : impl_(std::make_unique<Impl>(evaluator, max_mcts_iterations, c_exploration,
                                   c_ply_penalty, std::move(position), metadata)) {}

InteractivePlay::~InteractivePlay() = default;

Snapshot InteractivePlay::snapshot() const {
  std::lock_guard lock(impl_->mutex);
  Position position = impl_->game.root_position();
  QValue q_penalty = impl_->game.root_q_penalty();
  QValue q_no_penalty = impl_->game.root_q_no_penalty();
  if ((position.ply() & 1U) != 0U) {
    position = position.inverted();
    q_penalty = -q_penalty;
    q_no_penalty = -q_no_penalty;
  }

  return Snapshot{
      .pos = position,
      .policy = impl_->game.root_policy(),
      .q_penalty = q_penalty,
      .q_no_penalty = q_no_penalty,
      .n_mcts_iterations = impl_->game.root_visit_count(),
      .max_mcts_iterations = impl_->max_mcts_iterations,
      .c_exploration = impl_->c_exploration,
      .c_ply_penalty = impl_->c_ply_penalty,
      .background_running = impl_->background_running,
      .moves = impl_->game.move_history(),
  };
}

void InteractivePlay::increase_mcts_iterations(std::size_t count) {
  std::lock_guard lock(impl_->mutex);
  impl_->throw_background_error_locked();
  if (count > std::numeric_limits<std::size_t>::max() - impl_->max_mcts_iterations) {
    throw std::overflow_error("max_mcts_iterations overflow");
  }
  impl_->max_mcts_iterations += count;
  impl_->signal_search_locked();
}

bool InteractivePlay::make_move(Move move) {
  std::unique_lock lock(impl_->mutex);
  impl_->throw_background_error_locked();
  const Position position = impl_->game.root_position();
  if (move >= kCols || position.terminal_state().has_value() ||
      !position.legal_moves()[move]) {
    return false;
  }

  impl_->wait_until_root_is_expanded_locked(lock);
  impl_->game.make_move(move, impl_->c_exploration);
  impl_->signal_search_locked();
  return true;
}

bool InteractivePlay::make_random_move(float temperature) {
  if (!std::isfinite(temperature) || temperature < 0.0F) {
    throw std::invalid_argument("temperature must be finite and non-negative");
  }

  std::unique_lock lock(impl_->mutex);
  impl_->throw_background_error_locked();
  if (impl_->game.root_position().terminal_state().has_value()) {
    return false;
  }

  impl_->wait_until_root_is_expanded_locked(lock);
  impl_->game.make_random_move(impl_->c_exploration, temperature);
  impl_->signal_search_locked();
  return true;
}

bool InteractivePlay::make_best_move_if_ready() {
  std::lock_guard lock(impl_->mutex);
  impl_->throw_background_error_locked();
  if (impl_->game.root_position().terminal_state().has_value() ||
      impl_->game.root_visit_count() < impl_->max_mcts_iterations) {
    return false;
  }

  impl_->game.make_random_move(impl_->c_exploration, 0.0F);
  impl_->signal_search_locked();
  return true;
}

void InteractivePlay::reset() {
  std::lock_guard lock(impl_->mutex);
  impl_->throw_background_error_locked();
  impl_->game.reset();
  impl_->signal_search_locked();
}

bool InteractivePlay::undo() {
  std::lock_guard lock(impl_->mutex);
  impl_->throw_background_error_locked();
  const bool undone = impl_->game.undo();
  if (undone) {
    impl_->signal_search_locked();
  }
  return undone;
}

void InteractivePlay::rethrow_background_error() {
  std::lock_guard lock(impl_->mutex);
  impl_->throw_background_error_locked();
}

}  // namespace c4a0
