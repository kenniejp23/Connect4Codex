#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <termios.h>
#include <unistd.h>

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "c4a0/tui.hpp"

namespace c4a0 {
namespace {

class TestEvaluator final : public Evaluator {
 public:
  explicit TestEvaluator(bool should_throw) : should_throw_(should_throw) {}

  std::vector<EvalPosResult> evaluate(ModelId /*model_id*/,
                                      const std::vector<Position>& positions) override {
    if (should_throw_) {
      throw std::runtime_error("deliberate TUI evaluator failure");
    }
    std::vector<EvalPosResult> results(positions.size());
    for (auto& result : results) {
      result.policy.fill(0.0F);
    }
    return results;
  }

 private:
  bool should_throw_{};
};

class Descriptor {
 public:
  explicit Descriptor(int value = -1) : value_(value) {}
  ~Descriptor() {
    if (value_ >= 0) {
      ::close(value_);
    }
  }
  Descriptor(const Descriptor&) = delete;
  Descriptor& operator=(const Descriptor&) = delete;
  [[nodiscard]] int get() const noexcept { return value_; }
  [[nodiscard]] int release() noexcept {
    const int value = value_;
    value_ = -1;
    return value;
  }

 private:
  int value_;
};

struct TuiRunResult {
  int exit_status{};
  std::string output;
  tcflag_t initial_local_flags{};
  tcflag_t restored_local_flags{};
};

[[noreturn]] void child_exit(int status) { ::_exit(status); }

[[nodiscard]] TuiRunResult run_in_pseudo_terminal(bool evaluator_throws) {
  Descriptor master(::posix_openpt(O_RDWR | O_NOCTTY | O_NONBLOCK));
  if (master.get() < 0 || ::grantpt(master.get()) != 0 ||
      ::unlockpt(master.get()) != 0) {
    throw std::runtime_error("failed to open pseudo-terminal: " +
                             std::string(std::strerror(errno)));
  }
  const char* slave_name_raw = ::ptsname(master.get());
  if (slave_name_raw == nullptr) {
    throw std::runtime_error("failed to find pseudo-terminal slave");
  }
  const std::string slave_name(slave_name_raw);
  Descriptor observer(::open(slave_name.c_str(), O_RDWR | O_NOCTTY));
  if (observer.get() < 0) {
    throw std::runtime_error("failed to open pseudo-terminal slave");
  }

  winsize terminal_size{};
  terminal_size.ws_row = 45;
  terminal_size.ws_col = 120;
  if (::ioctl(master.get(), TIOCSWINSZ, &terminal_size) != 0) {
    throw std::runtime_error("failed to size pseudo-terminal");
  }

  termios initial{};
  if (::tcgetattr(observer.get(), &initial) != 0) {
    throw std::runtime_error("failed to read initial terminal attributes");
  }

  const pid_t child = ::fork();
  if (child < 0) {
    throw std::runtime_error("failed to fork TUI smoke test");
  }
  if (child == 0) {
    ::close(observer.release());
    ::close(master.release());
    if (::setsid() < 0) {
      child_exit(80);
    }
    const int slave = ::open(slave_name.c_str(), O_RDWR);
    if (slave < 0 || ::ioctl(slave, TIOCSCTTY, 0) != 0 ||
        ::dup2(slave, STDIN_FILENO) < 0 || ::dup2(slave, STDOUT_FILENO) < 0 ||
        ::dup2(slave, STDERR_FILENO) < 0) {
      child_exit(81);
    }
    if (slave > STDERR_FILENO) {
      ::close(slave);
    }
    ::setenv("TERM", "xterm-256color", 1);

    TestEvaluator evaluator(evaluator_throws);
    try {
      run_tui(evaluator, 1, 4.0F, 0.01F);
      child_exit(evaluator_throws ? 82 : 0);
    } catch (const std::runtime_error&) {
      child_exit(evaluator_throws ? 0 : 83);
    } catch (...) {
      child_exit(84);
    }
  }

  std::string output;
  bool sent_quit = evaluator_throws;
  int status = 0;
  bool exited = false;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
  while (std::chrono::steady_clock::now() < deadline) {
    pollfd descriptor{master.get(), POLLIN, 0};
    const int poll_result = ::poll(&descriptor, 1, 50);
    if (poll_result > 0 && (descriptor.revents & POLLIN) != 0) {
      std::array<char, 4096> buffer{};
      while (true) {
        const ssize_t count = ::read(master.get(), buffer.data(), buffer.size());
        if (count > 0) {
          output.append(buffer.data(), static_cast<std::size_t>(count));
          continue;
        }
        if (count < 0 && errno == EINTR) {
          continue;
        }
        break;
      }
    }

    if (!sent_quit && !output.empty()) {
      const char quit = 'q';
      if (::write(master.get(), &quit, 1) == 1) {
        sent_quit = true;
      }
    }

    const pid_t wait_result = ::waitpid(child, &status, WNOHANG);
    if (wait_result == child) {
      exited = true;
      break;
    }
    if (wait_result < 0 && errno != EINTR) {
      break;
    }
  }

  if (!exited) {
    ::kill(child, SIGKILL);
    while (::waitpid(child, &status, 0) < 0 && errno == EINTR) {
    }
  }

  termios restored{};
  if (::tcgetattr(observer.get(), &restored) != 0) {
    throw std::runtime_error("failed to read restored terminal attributes");
  }

  return TuiRunResult{
      .exit_status = exited ? status : -1,
      .output = std::move(output),
      .initial_local_flags = initial.c_lflag,
      .restored_local_flags = restored.c_lflag,
  };
}

void check_clean_exit_and_terminal_restore(const TuiRunResult& result) {
  REQUIRE(result.exit_status != -1);
  REQUIRE(WIFEXITED(result.exit_status));
  CHECK(WEXITSTATUS(result.exit_status) == 0);
  constexpr tcflag_t kInteractiveFlags = ICANON | ECHO;
  CHECK((result.restored_local_flags & kInteractiveFlags) ==
        (result.initial_local_flags & kInteractiveFlags));
}

TEST_CASE("TUI accepts Q in a pseudo-terminal and restores terminal mode", "[tui]") {
  const TuiRunResult result = run_in_pseudo_terminal(false);
  check_clean_exit_and_terminal_restore(result);
  CHECK(result.output.find("c4a0") != std::string::npos);
  CHECK(result.output.find("Policy") != std::string::npos);
  CHECK(result.output.find("Instructions") != std::string::npos);
}

TEST_CASE("TUI restores terminal mode when background evaluation throws", "[tui]") {
  const TuiRunResult result = run_in_pseudo_terminal(true);
  check_clean_exit_and_terminal_restore(result);
}

}  // namespace
}  // namespace c4a0
