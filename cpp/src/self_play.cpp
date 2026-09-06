#include "c4a0/self_play.hpp"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <exception>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace c4a0 {
namespace {

template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(std::size_t capacity) : capacity_(capacity), slots_(capacity) {
    if (capacity == 0) {
      throw std::invalid_argument("bounded queue capacity must be positive");
    }
  }

  BoundedQueue(const BoundedQueue&) = delete;
  BoundedQueue& operator=(const BoundedQueue&) = delete;

  bool push(T value, std::stop_token stop_token) {
    std::unique_lock lock(mutex_);
    not_full_.wait(lock, [this, stop_token] {
      return closed_ || stop_token.stop_requested() || size_ < capacity_;
    });
    if (closed_ || stop_token.stop_requested()) {
      return false;
    }

    slots_[tail_].emplace(std::move(value));
    tail_ = (tail_ + 1) % capacity_;
    ++size_;
    lock.unlock();
    not_empty_.notify_one();
    return true;
  }

  bool push_batch(std::vector<T>& values, std::stop_token stop_token) {
    if (values.empty()) {
      return true;
    }
    if (values.size() > capacity_) {
      throw std::invalid_argument("queue batch exceeds bounded queue capacity");
    }

    std::unique_lock lock(mutex_);
    not_full_.wait(lock, [this, &values, stop_token] {
      return closed_ || stop_token.stop_requested() ||
             capacity_ - size_ >= values.size();
    });
    if (closed_ || stop_token.stop_requested()) {
      return false;
    }

    for (auto& value : values) {
      slots_[tail_].emplace(std::move(value));
      tail_ = (tail_ + 1) % capacity_;
      ++size_;
    }
    values.clear();
    lock.unlock();
    not_empty_.notify_all();
    return true;
  }

  [[nodiscard]] std::optional<T> pop(std::stop_token stop_token) {
    std::unique_lock lock(mutex_);
    not_empty_.wait(lock, [this, stop_token] {
      return closed_ || stop_token.stop_requested() || size_ != 0;
    });
    if (stop_token.stop_requested() || size_ == 0) {
      return std::nullopt;
    }

    T value = std::move(*slots_[head_]);
    slots_[head_].reset();
    head_ = (head_ + 1) % capacity_;
    --size_;
    lock.unlock();
    not_full_.notify_one();
    return value;
  }

  void drain_available(std::vector<T>& destination) {
    std::size_t drained = 0;
    {
      std::lock_guard lock(mutex_);
      destination.reserve(destination.size() + size_);
      while (size_ != 0) {
        destination.push_back(std::move(*slots_[head_]));
        slots_[head_].reset();
        head_ = (head_ + 1) % capacity_;
        --size_;
        ++drained;
      }
    }
    if (drained != 0) {
      not_full_.notify_all();
    }
  }

  void close() noexcept {
    {
      std::lock_guard lock(mutex_);
      closed_ = true;
    }
    not_empty_.notify_all();
    not_full_.notify_all();
  }

  [[nodiscard]] std::size_t size() const noexcept {
    std::lock_guard lock(mutex_);
    return size_;
  }

 private:
  const std::size_t capacity_;
  mutable std::mutex mutex_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  std::vector<std::optional<T>> slots_;
  std::size_t head_{};
  std::size_t tail_{};
  std::size_t size_{};
  bool closed_{};
};

class Progress {
 public:
  explicit Progress(std::size_t total_games, SelfPlayTelemetryCallback callback)
      : enabled_(::isatty(STDERR_FILENO) != 0 || static_cast<bool>(callback)),
        total_games_(total_games),
        callback_(std::move(callback)),
        started_(std::chrono::steady_clock::now()),
        last_render_(started_) {}

  void neural_evaluations(std::size_t count) {
    if (!enabled_) {
      return;
    }
    const auto value = neural_evaluations_.fetch_add(count) + count;
    if ((value & 0x3fU) == 0U) {
      render(false);
    }
  }

  void mcts_iteration() {
    if (!enabled_) {
      return;
    }
    const auto value = mcts_iterations_.fetch_add(1) + 1;
    if ((value & 0xffU) == 0U) {
      render(false);
    }
  }

  void game_finished() {
    if (!enabled_) {
      return;
    }
    games_finished_.fetch_add(1);
    render(false, true);
  }

  void queue_depths(std::size_t neural, std::size_t mcts) noexcept {
    neural_queue_depth_.store(neural);
    mcts_queue_depth_.store(mcts);
  }

  void finish() noexcept {
    if (!enabled_) {
      return;
    }
    try {
      render(true, true);
    } catch (...) {
      // Progress reporting must never affect self-play.
    }
  }

 private:
  void render(bool final, bool force = false) {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard lock(render_mutex_);
    if (!final && !force && now - last_render_ < std::chrono::milliseconds(100)) {
      return;
    }
    last_render_ = now;
    const auto elapsed = std::chrono::duration<double>(now - started_).count();
    const SelfPlayTelemetry snapshot{
        .completed_games = games_finished_.load(),
        .total_games = total_games_,
        .neural_evaluations = neural_evaluations_.load(),
        .mcts_iterations = mcts_iterations_.load(),
        .neural_queue_depth = neural_queue_depth_.load(),
        .mcts_queue_depth = mcts_queue_depth_.load(),
        .elapsed_seconds = elapsed,
        .final = final,
    };
    if (::isatty(STDERR_FILENO) != 0) {
      std::cerr << '\r' << "self-play " << snapshot.completed_games << '/'
                << snapshot.total_games << " games, " << snapshot.neural_evaluations
                << " NN evals, " << snapshot.mcts_iterations
                << " MCTS iterations, q=" << snapshot.neural_queue_depth << '/'
                << snapshot.mcts_queue_depth << " ["
                << static_cast<std::size_t>(elapsed) << "s]";
      if (final) {
        std::cerr << '\n';
      }
      std::cerr.flush();
    }
    if (callback_) {
      callback_(snapshot);
    }
  }

  const bool enabled_;
  const std::size_t total_games_;
  SelfPlayTelemetryCallback callback_;
  const std::chrono::steady_clock::time_point started_;
  std::chrono::steady_clock::time_point last_render_;
  std::atomic<std::size_t> games_finished_{};
  std::atomic<std::size_t> neural_evaluations_{};
  std::atomic<std::size_t> mcts_iterations_{};
  std::atomic<std::size_t> neural_queue_depth_{};
  std::atomic<std::size_t> mcts_queue_depth_{};
  std::mutex render_mutex_;
};

struct MctsJob {
  MctsGame game;
  EvalPosResult evaluation;
};

[[nodiscard]] bool position_less(const Position& lhs, const Position& rhs) noexcept {
  if (lhs.mask() != rhs.mask()) {
    return lhs.mask() < rhs.mask();
  }
  return lhs.value() < rhs.value();
}

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
  if (std::none_of(legal_policy.begin(), legal_policy.end(),
                   [](float value) { return std::isfinite(value); }) &&
      !position.terminal_state().has_value()) {
    throw std::invalid_argument(
        "evaluator policy has no finite logit for a legal move");
  }
}

}  // namespace

std::vector<GameResult> self_play(Evaluator& evaluator,
                                  std::vector<GameMetadata> requests,
                                  std::size_t max_nn_batch_size,
                                  std::size_t n_mcts_iterations, float c_exploration,
                                  float c_ply_penalty,
                                  SelfPlayProgressCallback progress_callback,
                                  CancellationCallback cancelled_callback) {
  std::vector<GameRequest> game_requests;
  game_requests.reserve(requests.size());
  for (const auto& metadata : requests) {
    game_requests.emplace_back(metadata);
  }
  return self_play(evaluator, std::move(game_requests),
                   SelfPlayOptions{.max_nn_batch_size = max_nn_batch_size,
                                   .n_mcts_iterations = n_mcts_iterations,
                                   .c_exploration = c_exploration,
                                   .c_ply_penalty = c_ply_penalty,
                                   .root_dirichlet_alpha = 0.0F,
                                   .root_dirichlet_epsilon = 0.0F,
                                   .temperature_midpoint_ply = 4,
                                   .temperature_cutoff_ply = 8,
                                   .early_temperature = 4.0F,
                                   .middle_temperature = 2.0F,
                                   .late_temperature = 1.0F,
                                   .seed = 0,
                                   .worker_threads = 0},
                   std::move(progress_callback), std::move(cancelled_callback));
}

std::vector<GameResult> self_play(Evaluator& evaluator,
                                  std::vector<GameRequest> requests,
                                  const SelfPlayOptions& options,
                                  SelfPlayProgressCallback progress_callback,
                                  CancellationCallback cancelled_callback,
                                  SelfPlayTelemetryCallback telemetry_callback) {
  if (requests.empty()) {
    return {};
  }
  if (options.max_nn_batch_size == 0) {
    throw std::invalid_argument("max_nn_batch_size must be positive");
  }
  if (options.n_mcts_iterations == 0) {
    throw std::invalid_argument("n_mcts_iterations must be positive");
  }
  if (!std::isfinite(options.c_exploration) || !std::isfinite(options.c_ply_penalty) ||
      !std::isfinite(options.root_dirichlet_alpha) ||
      !std::isfinite(options.root_dirichlet_epsilon) ||
      !std::isfinite(options.early_temperature) ||
      !std::isfinite(options.middle_temperature) ||
      !std::isfinite(options.late_temperature) || options.root_dirichlet_alpha < 0.0F ||
      options.root_dirichlet_epsilon < 0.0F || options.root_dirichlet_epsilon > 1.0F ||
      options.early_temperature < 0.0F || options.middle_temperature < 0.0F ||
      options.late_temperature < 0.0F ||
      options.temperature_midpoint_ply > options.temperature_cutoff_ply ||
      (options.root_dirichlet_epsilon > 0.0F && options.root_dirichlet_alpha <= 0.0F)) {
    throw std::invalid_argument("MCTS coefficients must be finite");
  }
  const std::size_t game_count = requests.size();
  BoundedQueue<MctsGame> neural_queue(game_count);
  BoundedQueue<MctsJob> mcts_queue(game_count);
  std::vector<GameResult> results;
  results.reserve(game_count);
  std::mutex results_mutex;
  std::atomic<std::size_t> games_remaining{game_count};
  std::atomic<std::size_t> cancellation_checks{};
  std::stop_source cancellation;
  std::mutex error_mutex;
  std::exception_ptr first_error;
  Progress progress(game_count, std::move(telemetry_callback));

  const auto close_queues = [&] {
    neural_queue.close();
    mcts_queue.close();
  };
  const auto fail = [&](std::exception_ptr error) noexcept {
    try {
      std::lock_guard lock(error_mutex);
      if (!first_error) {
        first_error = std::move(error);
      }
    } catch (...) {
      // There is no useful recovery if acquiring an uncontended mutex fails.
    }
    cancellation.request_stop();
    close_queues();
  };

  const std::stop_token stop_token = cancellation.get_token();
  for (const auto& request : requests) {
    const auto position = Position::from_moves(request.opening_moves);
    if (position.terminal_state().has_value()) {
      throw std::invalid_argument("self-play opening must be non-terminal");
    }
    if (!neural_queue.push(MctsGame(position, request.metadata), stop_token)) {
      throw std::runtime_error("failed to enqueue initial self-play game");
    }
  }
  progress.queue_depths(neural_queue.size(), mcts_queue.size());

  std::jthread neural_thread;
  std::vector<std::jthread> mcts_threads;
  try {
    neural_thread = std::jthread([&](std::stop_token) {
      try {
        std::vector<MctsGame> pending_games;
        pending_games.reserve(game_count);

        while (!stop_token.stop_requested()) {
          if (cancelled_callback && cancelled_callback()) {
            throw std::runtime_error("self-play cancelled");
          }
          if (pending_games.empty()) {
            auto game = neural_queue.pop(stop_token);
            if (!game) {
              break;
            }
            pending_games.push_back(std::move(*game));
          }
          neural_queue.drain_available(pending_games);
          progress.queue_depths(neural_queue.size(), mcts_queue.size());

          std::map<ModelId, std::vector<Position>> positions_by_model;
          std::map<ModelId, std::unordered_set<Position>> seen_by_model;
          for (const auto& game : pending_games) {
            const ModelId model_id = game.leaf_model_id_to_play();
            const Position position = game.leaf_position();
            if (seen_by_model[model_id].insert(position).second) {
              positions_by_model[model_id].push_back(position);
            }
          }

          ModelId selected_model{};
          std::size_t selected_count{};
          bool selected{};
          for (const auto& [model_id, positions] : positions_by_model) {
            if (!selected || positions.size() > selected_count ||
                (positions.size() == selected_count && model_id > selected_model)) {
              selected = true;
              selected_model = model_id;
              selected_count = positions.size();
            }
          }
          if (!selected) {
            throw std::logic_error("self-play inference queue lost its games");
          }

          auto positions = positions_by_model.at(selected_model);
          std::sort(positions.begin(), positions.end(), position_less);
          if (positions.size() > options.max_nn_batch_size) {
            positions.resize(options.max_nn_batch_size);
          }

          auto evaluations = evaluator.evaluate(selected_model, positions);
          if (evaluations.size() != positions.size()) {
            throw std::runtime_error(
                "evaluator returned a different number of results than "
                "positions");
          }
          for (std::size_t index = 0; index < positions.size(); ++index) {
            validate_evaluation(positions[index], evaluations[index]);
          }
          progress.neural_evaluations(positions.size());

          std::unordered_map<Position, EvalPosResult> evaluation_by_position;
          evaluation_by_position.reserve(positions.size());
          for (std::size_t index = 0; index < positions.size(); ++index) {
            evaluation_by_position.emplace(positions[index], evaluations[index]);
          }

          auto games = std::move(pending_games);
          pending_games.clear();
          pending_games.reserve(game_count);
          std::vector<MctsJob> ready_jobs;
          ready_jobs.reserve(positions.size());
          for (auto& game : games) {
            const auto evaluation = evaluation_by_position.find(game.leaf_position());
            if (game.leaf_model_id_to_play() != selected_model ||
                evaluation == evaluation_by_position.end()) {
              pending_games.push_back(std::move(game));
              continue;
            }
            ready_jobs.push_back(MctsJob{std::move(game), evaluation->second});
          }
          if (!mcts_queue.push_batch(ready_jobs, stop_token)) {
            return;
          }
          progress.queue_depths(neural_queue.size(), mcts_queue.size());
        }
      } catch (...) {
        fail(std::current_exception());
      }
    });

    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    // MCTS work arrives in short bursts after each neural batch. More blocking
    // workers than hardware threads reduce queue turnaround without consuming
    // those cores continuously; cap at the number of games to avoid pointless
    // threads for small interactive and test workloads.
    const std::size_t automatic_workers = std::min<std::size_t>(
        game_count, std::max<std::size_t>(1, hardware_threads * 6U));
    const std::size_t worker_count =
        options.worker_threads > 0 ? options.worker_threads : automatic_workers;
    mcts_threads.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
      mcts_threads.emplace_back([&](std::stop_token) {
        try {
          while (!stop_token.stop_requested()) {
            if (cancelled_callback &&
                (cancellation_checks.fetch_add(1) & 0x3ffU) == 0U &&
                cancelled_callback()) {
              throw std::runtime_error("self-play cancelled");
            }
            auto job = mcts_queue.pop(stop_token);
            if (!job) {
              return;
            }
            progress.queue_depths(neural_queue.size(), mcts_queue.size());

            job->game.receive_evaluation(
                job->evaluation.policy, job->evaluation.q_penalty,
                job->evaluation.q_no_penalty, options.c_exploration,
                options.c_ply_penalty, options.root_dirichlet_alpha,
                options.root_dirichlet_epsilon,
                options.seed ^ job->game.root_position().mask() ^
                    (job->game.root_position().value() << 1U));
            progress.mcts_iteration();

            if (job->game.root_visit_count() < options.n_mcts_iterations) {
              if (!neural_queue.push(std::move(job->game), stop_token)) {
                return;
              }
              progress.queue_depths(neural_queue.size(), mcts_queue.size());
              continue;
            }

            const Position root_position = job->game.root_position();
            if (!root_position.terminal_state().has_value()) {
              const std::size_t ply = root_position.ply();
              const float temperature = ply < options.temperature_midpoint_ply
                                            ? options.early_temperature
                                            : (ply < options.temperature_cutoff_ply
                                                   ? options.middle_temperature
                                                   : options.late_temperature);
              job->game.make_random_move(options.c_exploration, temperature,
                                         options.seed);
              const auto next_root = job->game.root_position();
              job->game.add_root_dirichlet_noise(
                  options.root_dirichlet_alpha, options.root_dirichlet_epsilon,
                  options.c_exploration,
                  options.seed ^ next_root.mask() ^ (next_root.value() << 1U));
              if (!neural_queue.push(std::move(job->game), stop_token)) {
                return;
              }
              progress.queue_depths(neural_queue.size(), mcts_queue.size());
              continue;
            }

            GameResult result = std::move(job->game).to_result(options.c_ply_penalty);
            {
              std::lock_guard lock(results_mutex);
              results.push_back(std::move(result));
            }
            progress.game_finished();

            const std::size_t previous_remaining = games_remaining.fetch_sub(1);
            if (progress_callback) {
              progress_callback(game_count - previous_remaining + 1, game_count);
            }
            if (previous_remaining == 1) {
              close_queues();
              return;
            }
          }
        } catch (...) {
          fail(std::current_exception());
        }
      });
    }
  } catch (...) {
    fail(std::current_exception());
  }

  if (neural_thread.joinable()) {
    neural_thread.join();
  }
  for (auto& thread : mcts_threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  progress.finish();
  {
    std::lock_guard lock(error_mutex);
    if (first_error) {
      std::rethrow_exception(first_error);
    }
  }
  if (results.size() != game_count) {
    throw std::runtime_error("self-play stopped before every game completed");
  }
  return results;
}

}  // namespace c4a0
