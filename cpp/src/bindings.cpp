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
  explicit PythonEvaluator(nb::object callback) : callback_(std::move(callback)) {}

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
    nb::object raw_result = callback_(model_id, batch);
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

  auto metadata = nb::class_<GameMetadata>(module, "GameMetadata")
                      .def(nb::init<std::uint64_t, ModelId, ModelId>(), "game_id"_a,
                           "player0_id"_a, "player1_id"_a)
                      .def_ro("game_id", &GameMetadata::game_id)
                      .def_ro("player0_id", &GameMetadata::player0_id)
                      .def_ro("player1_id", &GameMetadata::player1_id);
  metadata.attr("__module__") = "c4a0_cpp";

  auto sample =
      nb::class_<Sample>(module, "Sample")
          .def("flip_h", &Sample::flip_horizontal)
          .def("to_numpy", &sample_to_numpy)
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
         nb::object callback) {
        if (requests.empty()) {
          return PlayGamesResult{};
        }
        validate_engine_arguments(max_nn_batch_size, n_mcts_iterations, c_exploration,
                                  c_ply_penalty);
        PythonEvaluator evaluator(std::move(callback));
        std::vector<GameResult> results;
        {
          nb::gil_scoped_release release;
          results = self_play(evaluator, requests, max_nn_batch_size, n_mcts_iterations,
                              c_exploration, c_ply_penalty);
        }
        return PlayGamesResult{.results = std::move(results)};
      },
      "reqs"_a, "max_nn_batch_size"_a, "n_mcts_iterations"_a, "c_exploration"_a,
      "c_ply_penalty"_a, "py_eval_pos_cb"_a);

  module.def(
      "run_tui",
      [](nb::object callback, std::size_t max_mcts_iterations, float c_exploration,
         float c_ply_penalty) {
        validate_engine_arguments(1, max_mcts_iterations, c_exploration, c_ply_penalty);
        PythonEvaluator evaluator(std::move(callback));
        nb::gil_scoped_release release;
        run_tui(evaluator, max_mcts_iterations, c_exploration, c_ply_penalty);
      },
      "py_eval_pos_cb"_a, "max_mcts_iters"_a, "c_exploration"_a, "c_ply_penalty"_a);
}
