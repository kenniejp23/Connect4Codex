#include "c4a0/tui.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace c4a0 {
namespace {

using ftxui::bgcolor;
using ftxui::bold;
using ftxui::border;
using ftxui::center;
using ftxui::Color;
using ftxui::color;
using ftxui::Element;
using ftxui::EQUAL;
using ftxui::Event;
using ftxui::filler;
using ftxui::flex;
using ftxui::gauge;
using ftxui::hbox;
using ftxui::HEIGHT;
using ftxui::ScreenInteractive;
using ftxui::separator;
using ftxui::size;
using ftxui::text;
using ftxui::vbox;
using ftxui::WIDTH;
using ftxui::window;

[[nodiscard]] std::string format_fixed(float value, int precision) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

[[nodiscard]] float normalized_q(float value) noexcept {
  if (!std::isfinite(value)) {
    return 0.5F;
  }
  return std::clamp((value + 1.0F) / 2.0F, 0.0F, 1.0F);
}

[[nodiscard]] Element cell(std::optional<CellValue> value) {
  Element token = text("   ") | size(WIDTH, EQUAL, 5) | size(HEIGHT, EQUAL, 3) | center;
  if (value == CellValue::kPlayer) {
    token = token | bgcolor(Color::Red);
  } else if (value == CellValue::kOpponent) {
    token = token | bgcolor(Color::Blue);
  } else {
    token = token | bgcolor(Color::Black);
  }
  return token | border;
}

[[nodiscard]] Element board(const Position& position) {
  std::vector<Element> rows;
  rows.reserve(kRows + 1);
  for (std::size_t visible_row = 0; visible_row < kRows; ++visible_row) {
    std::vector<Element> columns;
    columns.reserve(kCols);
    const std::size_t position_row = kRows - visible_row - 1;
    for (std::size_t column = 0; column < kCols; ++column) {
      columns.push_back(cell(position.get(position_row, column)));
    }
    rows.push_back(hbox(std::move(columns)));
  }

  std::vector<Element> labels;
  labels.reserve(kCols);
  for (std::size_t column = 0; column < kCols; ++column) {
    labels.push_back(text(std::to_string(column + 1)) | size(WIDTH, EQUAL, 5) | center |
                     bold);
  }
  rows.push_back(hbox(std::move(labels)));
  return vbox(std::move(rows));
}

struct TurnStatus {
  std::string text;
  Color color;
};

[[nodiscard]] TurnStatus turn_status(const Position& position) {
  const auto terminal = position.terminal_state();
  if (terminal == TerminalState::kDraw) {
    return {"Draw", Color::GrayDark};
  }
  if (terminal.has_value()) {
    // A legal terminal position was produced by the player who made the most
    // recent move. Snapshots are always oriented to Player 0, so parity maps
    // that move back to the stable red/blue player colours.
    return position.ply() % 2 == 0 ? TurnStatus{"Blue won", Color::Blue}
                                   : TurnStatus{"Red won", Color::Red};
  }
  return position.ply() % 2 == 0 ? TurnStatus{"Red to play", Color::Red}
                                 : TurnStatus{"Blue to play", Color::Blue};
}

[[nodiscard]] Element evaluation_bar(std::string label, float value,
                                     std::string displayed_value) {
  const Color side_color = value >= 0.0F ? Color::Red : Color::Blue;
  return vbox({
      hbox({text(std::move(label)) | bold, filler(),
            text(std::move(displayed_value)) | color(side_color) | bold}),
      gauge(normalized_q(value)) | color(side_color),
  });
}

[[nodiscard]] Element evaluations(const Snapshot& snapshot) {
  return window(
      text(" Evaluations ") | bold,
      vbox({
          evaluation_bar("Eval", snapshot.q_penalty,
                         format_fixed(snapshot.q_penalty, 2)),
          text(""),
          evaluation_bar("Win %", snapshot.q_no_penalty,
                         format_fixed(snapshot.q_no_penalty * 100.0F, 0) + "%"),
          filler(),
          text("Positive values favour Red") | color(Color::GrayDark),
          text("Negative values favour Blue") | color(Color::GrayDark),
      }));
}

[[nodiscard]] Element policy_chart(const Snapshot& snapshot) {
  std::vector<Element> rows;
  rows.reserve(kCols + 2);
  const std::string progress = std::to_string(snapshot.n_mcts_iterations) + "/" +
                               std::to_string(snapshot.max_mcts_iterations);
  rows.push_back(hbox({
      text(snapshot.background_running ? "MCTS running" : "MCTS stopped") |
          color(snapshot.background_running ? Color::Green : Color::Red) | bold,
      filler(),
      text(progress) | bold,
  }));
  rows.push_back(separator());

  for (std::size_t column = 0; column < kCols; ++column) {
    const float probability = std::isfinite(snapshot.policy[column])
                                  ? std::clamp(snapshot.policy[column], 0.0F, 1.0F)
                                  : 0.0F;
    rows.push_back(hbox({
        text(std::to_string(column + 1)) | size(WIDTH, EQUAL, 3) | bold,
        gauge(probability) | color(Color::Yellow) | flex,
        text(" " + format_fixed(probability, 2)) | size(WIDTH, EQUAL, 6) |
            color(Color::Green) | bold,
    }));
  }
  return window(text(" Policy ") | bold, vbox(std::move(rows)));
}

[[nodiscard]] Element instruction(std::string key, std::string description) {
  return hbox({text("<" + std::move(key) + ">") | color(Color::Blue) | bold |
                   size(WIDTH, EQUAL, 7),
               text(std::move(description))});
}

[[nodiscard]] Element instructions() {
  return window(text(" Instructions ") | bold,
                hbox({
                    vbox({
                        instruction("1-7", "Play move"),
                        instruction("B", "Play best move"),
                        instruction("R", "Play random move"),
                        instruction("M", "Add 100 MCTS iterations"),
                    }) | flex,
                    separator(),
                    vbox({
                        instruction("T", "Add 1 MCTS iteration"),
                        instruction("U", "Undo last move"),
                        instruction("N", "New game"),
                        instruction("Q", "Quit"),
                    }) | flex,
                }));
}

[[nodiscard]] Element render_app(const Snapshot& snapshot) {
  const TurnStatus status = turn_status(snapshot.pos);
  const Element game =
      window(text(" Game - " + status.text + " ") | color(status.color) | bold,
             board(snapshot.pos) | center);

  const std::string parameters =
      "Exploration " + format_fixed(snapshot.c_exploration, 2) + "  |  Ply penalty " +
      format_fixed(snapshot.c_ply_penalty, 3);
  return vbox({
             text(" c4a0 - Connect Four AlphaZero ") | center | bold,
             hbox({game | flex, evaluations(snapshot) | size(WIDTH, EQUAL, 32)}),
             policy_chart(snapshot) | flex,
             instructions(),
             text(parameters) | center | color(Color::White),
         }) |
         border;
}

[[nodiscard]] bool is_character(const Event& event, char lower, char upper) {
  return event == Event::Character(lower) || event == Event::Character(upper);
}

[[nodiscard]] bool handle_event(const Event& event, InteractivePlay& game,
                                ScreenInteractive& screen) {
  if (is_character(event, 'q', 'Q')) {
    screen.ExitLoopClosure()();
    return true;
  }
  if (is_character(event, 'b', 'B')) {
    static_cast<void>(game.make_random_move(0.0F));
    return true;
  }
  if (is_character(event, 'r', 'R')) {
    static_cast<void>(game.make_random_move(1.0F));
    return true;
  }
  if (is_character(event, 'm', 'M')) {
    game.increase_mcts_iterations(100);
    return true;
  }
  if (is_character(event, 't', 'T')) {
    game.increase_mcts_iterations(1);
    return true;
  }
  if (is_character(event, 'u', 'U')) {
    static_cast<void>(game.undo());
    return true;
  }
  if (is_character(event, 'n', 'N')) {
    game.reset();
    return true;
  }
  for (std::size_t column = 0; column < kCols; ++column) {
    if (event == Event::Character(static_cast<char>('1' + column))) {
      static_cast<void>(game.make_move(column));
      return true;
    }
  }
  return event == Event::Custom;
}

}  // namespace

void run_tui(Evaluator& evaluator, std::size_t max_mcts_iterations, float c_exploration,
             float c_ply_penalty) {
  InteractivePlay game(evaluator, max_mcts_iterations, c_exploration, c_ply_penalty);
  auto screen = ScreenInteractive::Fullscreen();

  auto renderer = ftxui::Renderer([&game] {
    game.rethrow_background_error();
    return render_app(game.snapshot());
  });
  auto component = ftxui::CatchEvent(renderer, [&game, &screen](const Event& event) {
    game.rethrow_background_error();
    return handle_event(event, game, screen);
  });

  // FTXUI redraws in response to events. A small custom-event heartbeat keeps
  // the live MCTS counters, policies and any background exception visible
  // without busy-waiting. jthread guarantees the ticker stops before the
  // ScreenInteractive object restores terminal state.
  std::jthread refresh_thread([&screen](const std::stop_token& stop_token) {
    while (!stop_token.stop_requested()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (!stop_token.stop_requested()) {
        screen.PostEvent(Event::Custom);
      }
    }
  });

  screen.Loop(component);
  refresh_thread.request_stop();
  game.rethrow_background_error();
}

}  // namespace c4a0
