#include "c4a0/interactive.hpp"
#include "c4a0/position.hpp"
#include "c4a0/self_play.hpp"
#include "c4a0/serialization.hpp"
#include "c4a0/solver.hpp"
#include "c4a0/tui.hpp"

// Keep strict shadow diagnostics for our binding code without imposing them
// on nanobind's public headers (nanobind 2.15 intentionally uses constructor
// parameter names that shadow members).
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#endif
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace c4a0 {
namespace {

using PositionBatch = nb::ndarray<nb::numpy, float, nb::shape<-1, 2, 6, 7>,
                                  nb::c_contig, nb::device::cpu>;
using PolicyBatch = nb::ndarray<nb::numpy, const float, nb::shape<-1, 7>, nb::c_contig,
                                nb::device::cpu>;
using ValueBatch =
    nb::ndarray<nb::numpy, const float, nb::shape<-1>, nb::c_contig, nb::device::cpu>;

template <typename Array, typename... Shape>
Array owned_array(std::vector<float> values, Shape... shape) {
  auto* data = new float[values.size()];
  std::copy(values.begin(), values.end(), data);
  nb::capsule owner(data,
                    [](void* ptr) noexcept { delete[] static_cast<float*>(ptr); });
  return Array(data, {static_cast<std::size_t>(shape)...}, owner);
}

class PythonEvaluator final : public Evaluator {
 public:
  explicit PythonEvaluator(nb::object callback, bool repeatable_errors = false)
      : callback_(std::move(callback)), repeatable_errors_(repeatable_errors) {}

  std::vector<EvalPosResult> evaluate(ModelId model_id,
                                      const std::vector<Position>& positions) override {
    nb::gil_scoped_acquire acquire;
    std::vector<float> buffer;
    buffer.reserve(positions.size() * kBufferLength);
    for (const auto& position : positions) {
      const auto position_buffer = position.to_buffer();
      buffer.insert(buffer.end(), position_buffer.begin(), position_buffer.end());
    }

    auto batch = owned_array<PositionBatch>(std::move(buffer), positions.size(),
                                            kBufferChannels, kRows, kCols);
    nb::object raw_result;
    try {
      raw_result = callback_(model_id, batch);
    } catch (const nb::python_error& error) {
      // python_error can only be restored once. Interactive errors are reported
      // repeatedly, so capture diagnostics while holding the GIL instead.
      if (repeatable_errors_) {
        throw std::runtime_error(error.what());
      }
      throw;
    }
    if (!nb::isinstance<nb::tuple>(raw_result)) {
      throw std::invalid_argument(
          "evaluation callback must return a three-element tuple");
    }
    nb::tuple result = nb::cast<nb::tuple>(raw_result);
    if (result.size() != 3) {
      throw std::invalid_argument(
          "evaluation callback must return a three-element tuple");
    }

    // Conversion is deliberately disabled: the callback contract requires
    // NumPy float32, C-contiguous arrays with the exact declared shapes.
    PolicyBatch policies;
    ValueBatch q_penalty;
    ValueBatch q_no_penalty;
    if (!nb::try_cast(result[0], policies, false) ||
        !nb::try_cast(result[1], q_penalty, false) ||
        !nb::try_cast(result[2], q_no_penalty, false)) {
      throw std::invalid_argument(
          "evaluation callback outputs must be C-contiguous NumPy float32 "
          "arrays with shapes [B, 7], [B], and [B]");
    }
    const auto batch_size = positions.size();
    if (policies.shape(0) != batch_size || q_penalty.shape(0) != batch_size ||
        q_no_penalty.shape(0) != batch_size) {
      throw std::invalid_argument(
          "evaluation callback output batch dimensions do not match input");
    }

    std::vector<EvalPosResult> evaluations;
    evaluations.reserve(batch_size);
    for (std::size_t row = 0; row < batch_size; ++row) {
      EvalPosResult evaluation;
      bool has_usable_legal_logit = false;
      const auto legal_moves = positions[row].legal_moves();
      for (std::size_t col = 0; col < kCols; ++col) {
        const float logit = policies(row, col);
        if (std::isnan(logit) || logit == std::numeric_limits<float>::infinity()) {
          throw std::invalid_argument(
              "evaluation callback policy contains NaN or positive infinity");
        }
        has_usable_legal_logit =
            has_usable_legal_logit || (legal_moves[col] && std::isfinite(logit));
        evaluation.policy[col] = logit;
      }
      evaluation.q_penalty = q_penalty(row);
      evaluation.q_no_penalty = q_no_penalty(row);
      if ((!positions[row].terminal_state().has_value() && !has_usable_legal_logit) ||
          !std::isfinite(evaluation.q_penalty) ||
          !std::isfinite(evaluation.q_no_penalty)) {
        throw std::invalid_argument(
            "evaluation callback returned invalid policy or value data");
      }
      evaluations.push_back(evaluation);
    }
    return evaluations;
  }

 private:
  nb::object callback_;
  bool repeatable_errors_;
};

struct PythonGameSnapshot {
  std::vector<std::vector<std::uint8_t>> board;
  std::array<bool, kCols> legal_moves{};
  std::string side_to_move;
  std::string terminal_state;
  std::vector<std::pair<std::size_t, std::size_t>> winning_cells;
  std::vector<Move> move_history;
  Policy policy{};
  QValue q_penalty{};
  QValue q_no_penalty{};
  std::size_t n_mcts_iterations{};
  std::size_t max_mcts_iterations{};
  float c_exploration{};
  float c_ply_penalty{};
  bool background_running{};
};

[[nodiscard]] std::vector<std::pair<std::size_t, std::size_t>> winning_cells(
    const std::vector<std::vector<std::uint8_t>>& board, std::uint8_t winner) {
  constexpr std::array<std::pair<int, int>, 4> directions = {
      std::pair{0, 1}, std::pair{1, 0}, std::pair{1, 1}, std::pair{1, -1}};
  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t col = 0; col < kCols; ++col) {
      if (board[row][col] != winner) {
        continue;
      }
      for (const auto& [row_step, col_step] : directions) {
        std::vector<std::pair<std::size_t, std::size_t>> cells;
        cells.reserve(4);
        for (int offset = 0; offset < 4; ++offset) {
          const int candidate_row = static_cast<int>(row) + row_step * offset;
          const int candidate_col = static_cast<int>(col) + col_step * offset;
          if (candidate_row < 0 || candidate_row >= static_cast<int>(kRows) ||
              candidate_col < 0 || candidate_col >= static_cast<int>(kCols) ||
              board[static_cast<std::size_t>(candidate_row)]
                   [static_cast<std::size_t>(candidate_col)] != winner) {
            cells.clear();
            break;
          }
          cells.emplace_back(static_cast<std::size_t>(candidate_row),
                             static_cast<std::size_t>(candidate_col));
        }
        if (cells.size() == 4) {
          return cells;
        }
      }
    }
  }
  return {};
}

[[nodiscard]] PythonGameSnapshot to_python_snapshot(const Snapshot& snapshot) {
  PythonGameSnapshot result;
  result.board.assign(kRows, std::vector<std::uint8_t>(kCols, 0));
  for (std::size_t visible_row = 0; visible_row < kRows; ++visible_row) {
    const std::size_t position_row = kRows - visible_row - 1;
    for (std::size_t col = 0; col < kCols; ++col) {
      const auto cell = snapshot.pos.get(position_row, col);
      if (cell == CellValue::kPlayer) {
        result.board[visible_row][col] = 1;
      } else if (cell == CellValue::kOpponent) {
        result.board[visible_row][col] = 2;
      }
    }
  }

  result.legal_moves = snapshot.pos.legal_moves();
  result.side_to_move = snapshot.pos.ply() % 2 == 0 ? "red" : "gold";
  const auto terminal = snapshot.pos.terminal_state();
  if (!terminal.has_value()) {
    result.terminal_state = "ongoing";
  } else if (*terminal == TerminalState::kDraw) {
    result.terminal_state = "draw";
  } else {
    result.terminal_state = snapshot.pos.ply() % 2 == 0 ? "gold_win" : "red_win";
    result.winning_cells = winning_cells(
        result.board,
        result.terminal_state == "red_win" ? std::uint8_t{1} : std::uint8_t{2});
  }
  result.move_history = snapshot.moves;
  result.policy = snapshot.policy;
  result.q_penalty = snapshot.q_penalty;
  result.q_no_penalty = snapshot.q_no_penalty;
  result.n_mcts_iterations = snapshot.n_mcts_iterations;
  result.max_mcts_iterations = snapshot.max_mcts_iterations;
  result.c_exploration = snapshot.c_exploration;
  result.c_ply_penalty = snapshot.c_ply_penalty;
  result.background_running = snapshot.background_running;
  return result;
}

class PythonInteractivePlay {
 public:
  PythonInteractivePlay(nb::object callback, std::size_t max_mcts_iterations,
                        float c_exploration, float c_ply_penalty, ModelId red_model_id,
                        ModelId gold_model_id)
      : evaluator_(std::make_unique<PythonEvaluator>(std::move(callback), true)),
        game_(std::make_unique<InteractivePlay>(
            *evaluator_, max_mcts_iterations, c_exploration, c_ply_penalty, Position{},
            GameMetadata{.game_id = 0,
                         .player0_id = red_model_id,
                         .player1_id = gold_model_id})) {}

  ~PythonInteractivePlay() {
    if (game_) {
      nb::gil_scoped_release release;
      game_.reset();
    }
  }

  PythonInteractivePlay(const PythonInteractivePlay&) = delete;
  PythonInteractivePlay& operator=(const PythonInteractivePlay&) = delete;

  [[nodiscard]] PythonGameSnapshot snapshot() {
    require_open();
    game_->rethrow_background_error();
    return to_python_snapshot(game_->snapshot());
  }

  [[nodiscard]] bool make_move(Move move) {
    require_open();
    return game_->make_move(move);
  }

  [[nodiscard]] bool make_best_move() {
    require_open();
    return game_->make_random_move(0.0F);
  }

  [[nodiscard]] bool make_best_move_if_ready() {
    require_open();
    return game_->make_best_move_if_ready();
  }

  [[nodiscard]] bool make_random_move(float temperature) {
    require_open();
    return game_->make_random_move(temperature);
  }

  void increase_mcts_iterations(std::size_t count) {
    require_open();
    game_->increase_mcts_iterations(count);
  }

  [[nodiscard]] bool undo() {
    require_open();
    return game_->undo();
  }

  void reset() {
    require_open();
    game_->reset();
  }

  void retry_evaluation() {
    require_open();
    game_->retry_evaluation();
  }

  void close() { game_.reset(); }

 private:
  void require_open() const {
    if (!game_) {
      throw std::runtime_error("interactive game is closed");
    }
  }

  std::unique_ptr<PythonEvaluator> evaluator_;
  std::unique_ptr<InteractivePlay> game_;
};

nb::tuple sample_to_numpy(const Sample& sample) {
  const auto position_buffer = sample.pos.to_buffer();
  auto position =
      owned_array<nb::ndarray<nb::numpy, float, nb::shape<2, 6, 7>, nb::c_contig>>(
          std::vector<float>(position_buffer.begin(), position_buffer.end()),
          kBufferChannels, kRows, kCols);
  auto policy = owned_array<nb::ndarray<nb::numpy, float, nb::shape<7>, nb::c_contig>>(
      std::vector<float>(sample.policy.begin(), sample.policy.end()), kCols);
  auto q_penalty =
      owned_array<nb::ndarray<nb::numpy, float, nb::ndim<0>, nb::c_contig>>(
          std::vector<float>{sample.q_penalty});
  auto q_no_penalty =
      owned_array<nb::ndarray<nb::numpy, float, nb::ndim<0>, nb::c_contig>>(
          std::vector<float>{sample.q_no_penalty});
  return nb::make_tuple(nb::cast(position), nb::cast(policy), nb::cast(q_penalty),
                        nb::cast(q_no_penalty));
}

nb::bytes result_to_bytes(const PlayGamesResult& result) {
  const auto bytes = to_cbor(result);
  return nb::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

PlayGamesResult result_from_bytes(const nb::bytes& bytes) {
  const auto* begin = reinterpret_cast<const std::uint8_t*>(bytes.c_str());
  return from_cbor(std::span<const std::uint8_t>(begin, bytes.size()));
}

void validate_engine_arguments(std::size_t max_batch_size, std::size_t iterations,
                               float c_exploration, float c_ply_penalty) {
  if (max_batch_size == 0) {
    throw std::invalid_argument("max_nn_batch_size must be greater than zero");
  }
  if (iterations == 0) {
    throw std::invalid_argument("n_mcts_iterations must be greater than zero");
  }
  if (!std::isfinite(c_exploration) || c_exploration < 0.0F) {
    throw std::invalid_argument("c_exploration must be finite and non-negative");
  }
  if (!std::isfinite(c_ply_penalty) || c_ply_penalty < 0.0F) {
    throw std::invalid_argument("c_ply_penalty must be finite and non-negative");
  }
}

}  // namespace
}  // namespace c4a0

NB_MODULE(_native, module) {
  using namespace c4a0;

  module.doc() = "C++20 Connect Four AlphaZero engine";
  module.attr("N_ROWS") = kRows;
  module.attr("N_COLS") = kCols;
  module.attr("BUF_N_CHANNELS") = kBufferChannels;

  auto game_snapshot =
      nb::class_<PythonGameSnapshot>(module, "GameSnapshot")
          .def_ro("board", &PythonGameSnapshot::board)
          .def_ro("legal_moves", &PythonGameSnapshot::legal_moves)
          .def_ro("side_to_move", &PythonGameSnapshot::side_to_move)
          .def_ro("terminal_state", &PythonGameSnapshot::terminal_state)
          .def_ro("winning_cells", &PythonGameSnapshot::winning_cells)
          .def_ro("move_history", &PythonGameSnapshot::move_history)
          .def_ro("policy", &PythonGameSnapshot::policy)
          .def_ro("q_penalty", &PythonGameSnapshot::q_penalty)
          .def_ro("q_no_penalty", &PythonGameSnapshot::q_no_penalty)
          .def_ro("n_mcts_iterations", &PythonGameSnapshot::n_mcts_iterations)
          .def_ro("max_mcts_iterations", &PythonGameSnapshot::max_mcts_iterations)
          .def_ro("c_exploration", &PythonGameSnapshot::c_exploration)
          .def_ro("c_ply_penalty", &PythonGameSnapshot::c_ply_penalty)
          .def_ro("background_running", &PythonGameSnapshot::background_running);
  game_snapshot.attr("__module__") = "c4a0_cpp";

  auto interactive_play =
      nb::class_<PythonInteractivePlay>(module, "InteractivePlay")
          .def(nb::init<nb::object, std::size_t, float, float, ModelId, ModelId>(),
               "py_eval_pos_cb"_a, "max_mcts_iters"_a, "c_exploration"_a,
               "c_ply_penalty"_a, "red_model_id"_a = 0, "gold_model_id"_a = 0)
          .def("snapshot", &PythonInteractivePlay::snapshot,
               nb::call_guard<nb::gil_scoped_release>())
          .def("make_move", &PythonInteractivePlay::make_move, "column"_a,
               nb::call_guard<nb::gil_scoped_release>())
          .def("make_best_move", &PythonInteractivePlay::make_best_move,
               nb::call_guard<nb::gil_scoped_release>())
          .def("make_best_move_if_ready",
               &PythonInteractivePlay::make_best_move_if_ready,
               nb::call_guard<nb::gil_scoped_release>())
          .def("make_random_move", &PythonInteractivePlay::make_random_move,
               "temperature"_a = 1.0F, nb::call_guard<nb::gil_scoped_release>())
          .def("increase_mcts_iterations",
               &PythonInteractivePlay::increase_mcts_iterations, "count"_a,
               nb::call_guard<nb::gil_scoped_release>())
          .def("undo", &PythonInteractivePlay::undo,
               nb::call_guard<nb::gil_scoped_release>())
          .def("retry_evaluation", &PythonInteractivePlay::retry_evaluation,
               nb::call_guard<nb::gil_scoped_release>())
          .def("reset", &PythonInteractivePlay::reset,
               nb::call_guard<nb::gil_scoped_release>())
          .def("close", &PythonInteractivePlay::close,
               nb::call_guard<nb::gil_scoped_release>());
  interactive_play.attr("__module__") = "c4a0_cpp";

  auto metadata = nb::class_<GameMetadata>(module, "GameMetadata")
                      .def(nb::init<std::uint64_t, ModelId, ModelId>(), "game_id"_a,
                           "player0_id"_a, "player1_id"_a)
                      .def_ro("game_id", &GameMetadata::game_id)
                      .def_ro("player0_id", &GameMetadata::player0_id)
                      .def_ro("player1_id", &GameMetadata::player1_id);
  metadata.attr("__module__") = "c4a0_cpp";

  auto game_request = nb::class_<GameRequest>(module, "GameRequest")
                          .def(nb::init<GameMetadata, std::vector<Move>>(),
                               "metadata"_a, "opening_moves"_a = std::vector<Move>{})
                          .def_ro("metadata", &GameRequest::metadata)
                          .def_ro("opening_moves", &GameRequest::opening_moves);
  game_request.attr("__module__") = "c4a0_cpp";

  auto self_play_options =
      nb::class_<SelfPlayOptions>(module, "SelfPlayOptions")
          .def(nb::init<>())
          .def_rw("max_nn_batch_size", &SelfPlayOptions::max_nn_batch_size)
          .def_rw("n_mcts_iterations", &SelfPlayOptions::n_mcts_iterations)
          .def_rw("c_exploration", &SelfPlayOptions::c_exploration)
          .def_rw("c_ply_penalty", &SelfPlayOptions::c_ply_penalty)
          .def_rw("root_dirichlet_alpha", &SelfPlayOptions::root_dirichlet_alpha)
          .def_rw("root_dirichlet_epsilon", &SelfPlayOptions::root_dirichlet_epsilon)
          .def_rw("temperature_midpoint_ply",
                  &SelfPlayOptions::temperature_midpoint_ply)
          .def_rw("temperature_cutoff_ply", &SelfPlayOptions::temperature_cutoff_ply)
          .def_rw("early_temperature", &SelfPlayOptions::early_temperature)
          .def_rw("middle_temperature", &SelfPlayOptions::middle_temperature)
          .def_rw("late_temperature", &SelfPlayOptions::late_temperature)
          .def_rw("seed", &SelfPlayOptions::seed)
          .def_rw("worker_threads", &SelfPlayOptions::worker_threads);
  self_play_options.attr("__module__") = "c4a0_cpp";

  auto sample =
      nb::class_<Sample>(module, "Sample")
          .def("flip_h", &Sample::flip_horizontal)
          .def("to_numpy", &sample_to_numpy)
          .def_prop_ro("ply", [](const Sample& value) { return value.pos.ply(); })
          .def("pos_str", [](const Sample& value) { return value.pos.to_string(); });
  sample.attr("__module__") = "c4a0_cpp";

  auto game_result =
      nb::class_<GameResult>(module, "GameResult")
          .def_prop_ro("metadata",
                       [](const GameResult& value) { return value.metadata; })
          .def_prop_ro("samples", [](const GameResult& value) { return value.samples; })
          .def("player0_score", &GameResult::player0_score);
  game_result.attr("__module__") = "c4a0_cpp";

  auto play_result =
      nb::class_<PlayGamesResult>(module, "PlayGamesResult")
          .def(nb::init<>())
          .def_prop_ro("results",
                       [](const PlayGamesResult& value) { return value.results; })
          .def("__add__", &PlayGamesResult::combined)
          .def("split_games", &PlayGamesResult::split_games, "chunk_size"_a)
          .def("split_train_test", &PlayGamesResult::split_train_test, "train_frac"_a,
               "seed"_a)
          .def("unique_positions", &PlayGamesResult::unique_positions)
          .def("to_cbor", &result_to_bytes)
          .def_static("from_cbor", &result_from_bytes)
          .def("__getstate__", &result_to_bytes)
          .def("__setstate__",
               [](PlayGamesResult& self, const nb::bytes& state) {
                 // During unpickling nanobind provides raw, uninitialized
                 // storage. Construct in place just like a custom __init__.
                 new (&self) PlayGamesResult(result_from_bytes(state));
               })
          .def(
              "score_policies",
              [](const PlayGamesResult& self, const std::string& solver_path,
                 const std::string& book_path, const std::string& cache_path) {
                CachingSolver solver(solver_path, book_path, cache_path);
                std::vector<std::pair<Position, Policy>> values;
                for (const auto& result : self.results) {
                  for (const auto& item : result.samples) {
                    if (!item.pos.terminal_state().has_value()) {
                      values.emplace_back(item.pos, item.policy);
                    }
                  }
                }
                if (values.empty()) {
                  throw std::invalid_argument(
                      "cannot score a result with no non-terminal samples");
                }
                const auto scores = solver.score_policies(values);
                float sum = 0.0F;
                for (const float score : scores) {
                  sum += score;
                }
                return sum / static_cast<float>(scores.size());
              },
              "solver_path"_a, "solver_book_path"_a, "solution_cache_path"_a);
  play_result.attr("__module__") = "c4a0_cpp";

  module.def(
      "play_games",
      [](const std::vector<GameMetadata>& requests, std::size_t max_nn_batch_size,
         std::size_t n_mcts_iterations, float c_exploration, float c_ply_penalty,
         nb::object callback, nb::object progress_callback,
         nb::object cancelled_callback) {
        if (requests.empty()) {
          return PlayGamesResult{};
        }
        validate_engine_arguments(max_nn_batch_size, n_mcts_iterations, c_exploration,
                                  c_ply_penalty);
        PythonEvaluator evaluator(std::move(callback));
        SelfPlayProgressCallback progress;
        if (!progress_callback.is_none()) {
          progress = [callback = std::move(progress_callback)](std::size_t completed,
                                                               std::size_t total) {
            nb::gil_scoped_acquire acquire;
            callback(completed, total);
          };
        }
        CancellationCallback cancelled;
        if (!cancelled_callback.is_none()) {
          cancelled = [callback = std::move(cancelled_callback)] {
            nb::gil_scoped_acquire acquire;
            return nb::cast<bool>(callback());
          };
        }
        std::vector<GameResult> results;
        {
          nb::gil_scoped_release release;
          results = self_play(evaluator, requests, max_nn_batch_size, n_mcts_iterations,
                              c_exploration, c_ply_penalty, std::move(progress),
                              std::move(cancelled));
        }
        return PlayGamesResult{.results = std::move(results)};
      },
      "reqs"_a, "max_nn_batch_size"_a, "n_mcts_iterations"_a, "c_exploration"_a,
      "c_ply_penalty"_a, "py_eval_pos_cb"_a, "progress_callback"_a = nb::none(),
      "cancelled_callback"_a = nb::none());

  module.def(
      "play_games_v2",
      [](const std::vector<GameRequest>& requests, const SelfPlayOptions& options,
         nb::object callback, nb::object progress_callback,
         nb::object cancelled_callback, nb::object telemetry_callback) {
        if (requests.empty()) {
          return PlayGamesResult{};
        }
        PythonEvaluator evaluator(std::move(callback));
        SelfPlayProgressCallback progress;
        if (!progress_callback.is_none()) {
          progress = [callback = std::move(progress_callback)](std::size_t completed,
                                                               std::size_t total) {
            nb::gil_scoped_acquire acquire;
            callback(completed, total);
          };
        }
        CancellationCallback cancelled;
        if (!cancelled_callback.is_none()) {
          cancelled = [callback = std::move(cancelled_callback)] {
            nb::gil_scoped_acquire acquire;
            return nb::cast<bool>(callback());
          };
        }
        SelfPlayTelemetryCallback telemetry;
        if (!telemetry_callback.is_none()) {
          telemetry = [callback = std::move(telemetry_callback)](
                          const SelfPlayTelemetry& value) {
            nb::gil_scoped_acquire acquire;
            nb::dict snapshot;
            snapshot["completed_games"] = value.completed_games;
            snapshot["total_games"] = value.total_games;
            snapshot["neural_evaluations"] = value.neural_evaluations;
            snapshot["mcts_iterations"] = value.mcts_iterations;
            snapshot["neural_queue_depth"] = value.neural_queue_depth;
            snapshot["mcts_queue_depth"] = value.mcts_queue_depth;
            snapshot["elapsed_seconds"] = value.elapsed_seconds;
            snapshot["final"] = value.final;
            callback(snapshot);
          };
        }
        std::vector<GameResult> results;
        {
          nb::gil_scoped_release release;
          results = self_play(evaluator, requests, options, std::move(progress),
                              std::move(cancelled), std::move(telemetry));
        }
        return PlayGamesResult{.results = std::move(results)};
      },
      "requests"_a, "options"_a, "py_eval_pos_cb"_a, "progress_callback"_a = nb::none(),
      "cancelled_callback"_a = nb::none(), "telemetry_callback"_a = nb::none());

  module.def(
      "run_tui",
      [](nb::object callback, std::size_t max_mcts_iterations, float c_exploration,
         float c_ply_penalty, bool auto_red, bool auto_blue) {
        validate_engine_arguments(1, max_mcts_iterations, c_exploration, c_ply_penalty);
        PythonEvaluator evaluator(std::move(callback));
        nb::gil_scoped_release release;
        run_tui(evaluator, max_mcts_iterations, c_exploration, c_ply_penalty, auto_red,
                auto_blue);
      },
      "py_eval_pos_cb"_a, "max_mcts_iters"_a, "c_exploration"_a, "c_ply_penalty"_a,
      "auto_red"_a = false, "auto_blue"_a = false);
}
