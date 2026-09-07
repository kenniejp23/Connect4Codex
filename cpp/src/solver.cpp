#include "c4a0/solver.hpp"

#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <sqlite3.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

extern char** environ;

namespace c4a0 {
namespace {

constexpr std::string_view kCacheFormat = "c4a0-solver-cache";
constexpr int kCacheVersion = 1;
constexpr std::size_t kSolverChunkSize = 100;
constexpr std::size_t kEncodedSolutionSize = kCols * sizeof(std::int16_t);
constexpr std::size_t kMaximumSubprocessOutput = 4U * 1024U * 1024U;

[[noreturn]] void throw_system_error(std::string_view operation, int error) {
  throw std::runtime_error(std::string(operation) + ": " + std::strerror(error));
}

[[noreturn]] void throw_sqlite_error(sqlite3* database, std::string_view operation,
                                     int result) {
  const char* message =
      database == nullptr ? sqlite3_errstr(result) : sqlite3_errmsg(database);
  throw std::runtime_error(std::string(operation) + ": " + message +
                           " (SQLite result " + std::to_string(result) + ")");
}

void execute_sql(sqlite3* database, const char* sql, std::string_view operation) {
  char* error_message = nullptr;
  const int result = sqlite3_exec(database, sql, nullptr, nullptr, &error_message);
  if (result == SQLITE_OK) {
    return;
  }

  std::string message = std::string(operation) + ": ";
  if (error_message != nullptr) {
    message += error_message;
    sqlite3_free(error_message);
  } else {
    message += sqlite3_errmsg(database);
  }
  throw std::runtime_error(message + " (SQLite result " + std::to_string(result) + ")");
}

class Statement {
 public:
  Statement(sqlite3* database, const char* sql, std::string_view operation)
      : database_(database) {
    const int result = sqlite3_prepare_v2(database_, sql, -1, &statement_, nullptr);
    if (result != SQLITE_OK) {
      throw_sqlite_error(database_, operation, result);
    }
  }

  ~Statement() {
    if (statement_ != nullptr) {
      sqlite3_finalize(statement_);
    }
  }

  Statement(const Statement&) = delete;
  Statement& operator=(const Statement&) = delete;

  [[nodiscard]] sqlite3_stmt* get() const noexcept { return statement_; }

 private:
  sqlite3* database_{};
  sqlite3_stmt* statement_{};
};

void bind_position(sqlite3* database, sqlite3_stmt* statement,
                   const Position& position) {
  int result =
      sqlite3_bind_int64(statement, 1, static_cast<sqlite3_int64>(position.mask()));
  if (result != SQLITE_OK) {
    throw_sqlite_error(database, "failed to bind position mask", result);
  }
  result =
      sqlite3_bind_int64(statement, 2, static_cast<sqlite3_int64>(position.value()));
  if (result != SQLITE_OK) {
    throw_sqlite_error(database, "failed to bind position value", result);
  }
}

[[nodiscard]] std::array<std::uint8_t, kEncodedSolutionSize> encode_solution(
    const Solution& solution) noexcept {
  std::array<std::uint8_t, kEncodedSolutionSize> encoded{};
  for (std::size_t index = 0; index < solution.size(); ++index) {
    const auto bits = static_cast<std::uint16_t>(solution[index]);
    encoded[index * 2] = static_cast<std::uint8_t>(bits & 0xffU);
    encoded[index * 2 + 1] = static_cast<std::uint8_t>(bits >> 8U);
  }
  return encoded;
}

[[nodiscard]] Solution decode_solution(const void* data, int size) {
  if (data == nullptr || size != static_cast<int>(kEncodedSolutionSize)) {
    throw std::runtime_error(
        "solver cache contains a corrupt solution record: expected " +
        std::to_string(kEncodedSolutionSize) + " bytes, got " + std::to_string(size));
  }

  const auto* encoded = static_cast<const std::uint8_t*>(data);
  Solution solution{};
  for (std::size_t index = 0; index < solution.size(); ++index) {
    const auto bits = static_cast<std::uint16_t>(encoded[index * 2]) |
                      (static_cast<std::uint16_t>(encoded[index * 2 + 1]) << 8U);
    solution[index] = static_cast<std::int16_t>(bits);
  }
  return solution;
}

class Transaction {
 public:
  explicit Transaction(sqlite3* database) : database_(database) {
    execute_sql(database_, "BEGIN IMMEDIATE", "failed to begin transaction");
  }

  ~Transaction() {
    if (!finished_) {
      sqlite3_exec(database_, "ROLLBACK", nullptr, nullptr, nullptr);
    }
  }

  Transaction(const Transaction&) = delete;
  Transaction& operator=(const Transaction&) = delete;

  void commit() {
    execute_sql(database_, "COMMIT", "failed to commit transaction");
    finished_ = true;
  }

 private:
  sqlite3* database_{};
  bool finished_{};
};

class FileDescriptor {
 public:
  FileDescriptor() = default;
  explicit FileDescriptor(int descriptor) : descriptor_(descriptor) {}
  ~FileDescriptor() { reset(); }

  FileDescriptor(FileDescriptor&& other) noexcept : descriptor_(other.release()) {}
  FileDescriptor& operator=(FileDescriptor&& other) noexcept {
    if (this != &other) {
      reset(other.release());
    }
    return *this;
  }

  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;

  [[nodiscard]] int get() const noexcept { return descriptor_; }
  [[nodiscard]] explicit operator bool() const noexcept { return descriptor_ >= 0; }

  [[nodiscard]] int release() noexcept { return std::exchange(descriptor_, -1); }

  void reset(int descriptor = -1) noexcept {
    if (descriptor_ >= 0) {
      int result = 0;
      do {
        result = ::close(descriptor_);
      } while (result < 0 && errno == EINTR);
    }
    descriptor_ = descriptor;
  }

 private:
  int descriptor_{-1};
};

struct Pipe {
  FileDescriptor read;
  FileDescriptor write;
};

[[nodiscard]] Pipe make_pipe() {
  std::array<int, 2> descriptors{};
  if (::pipe2(descriptors.data(), O_CLOEXEC) != 0) {
    throw_system_error("failed to create subprocess pipe", errno);
  }
  return Pipe{FileDescriptor(descriptors[0]), FileDescriptor(descriptors[1])};
}

class SpawnFileActions {
 public:
  SpawnFileActions() {
    const int result = posix_spawn_file_actions_init(&actions_);
    if (result != 0) {
      throw_system_error("failed to initialize posix_spawn file actions", result);
    }
    initialized_ = true;
  }

  ~SpawnFileActions() {
    if (initialized_) {
      posix_spawn_file_actions_destroy(&actions_);
    }
  }

  SpawnFileActions(const SpawnFileActions&) = delete;
  SpawnFileActions& operator=(const SpawnFileActions&) = delete;

  [[nodiscard]] posix_spawn_file_actions_t* get() noexcept { return &actions_; }

  void duplicate(int descriptor, int target) {
    const int result = posix_spawn_file_actions_adddup2(&actions_, descriptor, target);
    if (result != 0) {
      throw_system_error("failed to configure posix_spawn pipe", result);
    }
  }

  void close(int descriptor) {
    const int result = posix_spawn_file_actions_addclose(&actions_, descriptor);
    if (result != 0) {
      throw_system_error("failed to configure posix_spawn descriptor close", result);
    }
  }

 private:
  posix_spawn_file_actions_t actions_{};
  bool initialized_{};
};

using SolverDeadline = std::chrono::steady_clock::time_point;

SolverDeadline solver_deadline() {
  double seconds = 60.0;
  if (const char* setting = std::getenv("C4A0_SOLVER_TIMEOUT_SECONDS")) {
    char* end = nullptr;
    seconds = std::strtod(setting, &end);
    if (end == setting || *end != '\0' || !std::isfinite(seconds) || seconds <= 0 ||
        seconds > 86400) {
      throw std::invalid_argument("C4A0_SOLVER_TIMEOUT_SECONDS must be in (0, 86400]");
    }
  }
  return std::chrono::steady_clock::now() +
         std::chrono::milliseconds(static_cast<long long>(seconds * 1000));
}

void check_solver_deadline(SolverDeadline deadline) {
  if (std::chrono::steady_clock::now() >= deadline) {
    throw std::runtime_error("solver subprocess exceeded its deadline");
  }
}

class ChildProcess {
 public:
  explicit ChildProcess(pid_t process) : process_(process) {}
  ~ChildProcess() {
    if (!waited_ && process_ > 0) {
      ::kill(process_, SIGKILL);
      int status = 0;
      while (::waitpid(process_, &status, 0) < 0 && errno == EINTR) {
      }
    }
  }

  ChildProcess(const ChildProcess&) = delete;
  ChildProcess& operator=(const ChildProcess&) = delete;

  [[nodiscard]] int wait(SolverDeadline deadline) {
    int status = 0;
    pid_t result = 0;
    do {
      check_solver_deadline(deadline);
      result = ::waitpid(process_, &status, WNOHANG);
      if (result == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    } while (result == 0 || (result < 0 && errno == EINTR));
    if (result < 0) {
      throw_system_error("failed to wait for solver subprocess", errno);
    }
    waited_ = true;
    return status;
  }

 private:
  pid_t process_{};
  bool waited_{};
};

// A pipe write to a subprocess that has already exited must be reported as an
// exception, not terminate the host process. Blocking SIGPIPE for just the
// calling thread avoids changing process-wide signal handling.
class ScopedSigpipeBlock {
 public:
  ScopedSigpipeBlock() {
    sigemptyset(&signal_set_);
    sigaddset(&signal_set_, SIGPIPE);

    sigset_t pending{};
    if (sigpending(&pending) == 0) {
      pending_before_ = sigismember(&pending, SIGPIPE) == 1;
    }

    const int result = pthread_sigmask(SIG_BLOCK, &signal_set_, &old_mask_);
    if (result != 0) {
      throw_system_error("failed to block SIGPIPE", result);
    }
    active_ = true;
    signal_was_blocked_ = sigismember(&old_mask_, SIGPIPE) == 1;
  }

  ~ScopedSigpipeBlock() {
    if (!active_) {
      return;
    }

    if (!signal_was_blocked_ && !pending_before_) {
      sigset_t pending{};
      if (sigpending(&pending) == 0 && sigismember(&pending, SIGPIPE) == 1) {
        timespec timeout{};
        while (sigtimedwait(&signal_set_, nullptr, &timeout) < 0 && errno == EINTR) {
        }
      }
    }
    pthread_sigmask(SIG_SETMASK, &old_mask_, nullptr);
  }

  ScopedSigpipeBlock(const ScopedSigpipeBlock&) = delete;
  ScopedSigpipeBlock& operator=(const ScopedSigpipeBlock&) = delete;

 private:
  sigset_t signal_set_{};
  sigset_t old_mask_{};
  bool active_{};
  bool signal_was_blocked_{};
  bool pending_before_{};
};

[[nodiscard]] std::optional<std::string> write_all(int descriptor,
                                                   std::string_view input,
                                                   SolverDeadline deadline) {
  ScopedSigpipeBlock signal_block;
  std::size_t offset = 0;
  while (offset < input.size()) {
    check_solver_deadline(deadline);
    const ssize_t written =
        ::write(descriptor, input.data() + offset, input.size() - offset);
    if (written > 0) {
      offset += static_cast<std::size_t>(written);
      continue;
    }
    if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      pollfd writable{descriptor, POLLOUT, 0};
      ::poll(&writable, 1, 50);
      continue;
    }
    if (written < 0 && errno == EINTR) {
      continue;
    }
    const int error = written == 0 ? EIO : errno;
    return "failed to write solver subprocess input: " +
           std::string(std::strerror(error));
  }
  return std::nullopt;
}

void set_nonblocking(int descriptor) {
  const int flags = ::fcntl(descriptor, F_GETFL);
  if (flags < 0 || ::fcntl(descriptor, F_SETFL, flags | O_NONBLOCK) < 0) {
    throw_system_error("failed to configure solver output pipe", errno);
  }
}

void read_available(FileDescriptor& descriptor, std::string& destination,
                    std::string_view stream_name) {
  std::array<char, 4096> buffer{};
  while (descriptor) {
    const ssize_t count = ::read(descriptor.get(), buffer.data(), buffer.size());
    if (count > 0) {
      if (destination.size() + static_cast<std::size_t>(count) >
          kMaximumSubprocessOutput) {
        throw std::runtime_error("solver " + std::string(stream_name) +
                                 " exceeded the output limit");
      }
      destination.append(buffer.data(), static_cast<std::size_t>(count));
      continue;
    }
    if (count == 0) {
      descriptor.reset();
      return;
    }
    if (errno == EINTR) {
      continue;
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return;
    }
    throw_system_error("failed to read solver " + std::string(stream_name), errno);
  }
}

struct CapturedOutput {
  std::string standard_output;
  std::string standard_error;
};

[[nodiscard]] CapturedOutput read_process_output(FileDescriptor standard_output,
                                                 FileDescriptor standard_error,
                                                 SolverDeadline deadline) {
  set_nonblocking(standard_output.get());
  set_nonblocking(standard_error.get());

  CapturedOutput captured;
  while (standard_output || standard_error) {
    std::array<pollfd, 2> descriptors{{
        {standard_output ? standard_output.get() : -1,
         static_cast<short>(POLLIN | POLLHUP | POLLERR), 0},
        {standard_error ? standard_error.get() : -1,
         static_cast<short>(POLLIN | POLLHUP | POLLERR), 0},
    }};

    int result = 0;
    do {
      check_solver_deadline(deadline);
      result = ::poll(descriptors.data(), descriptors.size(), 50);
    } while (result < 0 && errno == EINTR);
    if (result < 0) {
      throw_system_error("failed to poll solver output", errno);
    }

    if (descriptors[0].revents != 0) {
      read_available(standard_output, captured.standard_output, "stdout");
    }
    if (descriptors[1].revents != 0) {
      read_available(standard_error, captured.standard_error, "stderr");
    }
  }
  return captured;
}

[[nodiscard]] bool all_whitespace(std::string_view text) noexcept {
  return std::all_of(text.begin(), text.end(), [](char character) {
    return character == ' ' || character == '\t' || character == '\r';
  });
}

[[nodiscard]] std::vector<std::string_view> split_whitespace(std::string_view line) {
  std::vector<std::string_view> tokens;
  std::size_t offset = 0;
  while (offset < line.size()) {
    while (offset < line.size() &&
           (line[offset] == ' ' || line[offset] == '\t' || line[offset] == '\r')) {
      ++offset;
    }
    const std::size_t start = offset;
    while (offset < line.size() && line[offset] != ' ' && line[offset] != '\t' &&
           line[offset] != '\r') {
      ++offset;
    }
    if (start != offset) {
      tokens.emplace_back(line.substr(start, offset - start));
    }
  }
  return tokens;
}

[[nodiscard]] std::int16_t parse_score(std::string_view token,
                                       std::size_t line_number) {
  int value = 0;
  const auto [end, error] =
      std::from_chars(token.data(), token.data() + token.size(), value);
  if (error != std::errc{} || end != token.data() + token.size() ||
      value < std::numeric_limits<std::int16_t>::min() ||
      value > std::numeric_limits<std::int16_t>::max()) {
    throw std::runtime_error("invalid solver score '" + std::string(token) +
                             "' on output line " + std::to_string(line_number));
  }
  return static_cast<std::int16_t>(value);
}

[[nodiscard]] std::vector<Solution> parse_solver_output(
    std::string_view output, const std::vector<std::string>& move_sequences) {
  std::vector<std::string_view> lines;
  std::size_t start = 0;
  while (start <= output.size()) {
    const std::size_t newline = output.find('\n', start);
    const std::size_t end = newline == std::string_view::npos ? output.size() : newline;
    const std::string_view line = output.substr(start, end - start);
    if (!line.empty() && !all_whitespace(line)) {
      lines.push_back(line);
    }
    if (newline == std::string_view::npos) {
      break;
    }
    start = newline + 1;
  }

  if (lines.size() != move_sequences.size()) {
    throw std::runtime_error("solver returned " + std::to_string(lines.size()) +
                             " non-empty output lines for " +
                             std::to_string(move_sequences.size()) +
                             " requested positions");
  }

  std::vector<Solution> solutions;
  solutions.reserve(lines.size());
  for (std::size_t line_index = 0; line_index < lines.size(); ++line_index) {
    const auto tokens = split_whitespace(lines[line_index]);
    std::size_t score_offset = 0;
    if (tokens.size() == kCols + 1) {
      if (tokens.front() != move_sequences[line_index]) {
        throw std::runtime_error("solver echoed move sequence '" +
                                 std::string(tokens.front()) + "' on output line " +
                                 std::to_string(line_index + 1) + ", expected '" +
                                 move_sequences[line_index] + "'");
      }
      score_offset = 1;
    } else if (tokens.size() == kCols && move_sequences[line_index].empty()) {
      score_offset = 0;
    } else {
      throw std::runtime_error(
          "solver output line " + std::to_string(line_index + 1) + " contains " +
          std::to_string(tokens.size()) +
          " fields; expected an echoed move sequence and seven scores");
    }

    Solution solution{};
    for (std::size_t column = 0; column < kCols; ++column) {
      solution[column] = parse_score(tokens[score_offset + column], line_index + 1);
    }
    solutions.push_back(solution);
  }
  return solutions;
}

[[nodiscard]] std::string position_move_sequence(const Position& position) {
  std::string sequence;
  const auto moves = position.to_moves();
  sequence.reserve(moves.size());
  for (const Move move : moves) {
    if (move >= kCols) {
      throw std::runtime_error(
          "position produced an invalid move while preparing solver input");
    }
    sequence.push_back(static_cast<char>('1' + move));
  }
  return sequence;
}

[[nodiscard]] std::string describe_stderr(std::string_view standard_error) {
  while (!standard_error.empty() &&
         (standard_error.back() == '\n' || standard_error.back() == '\r' ||
          standard_error.back() == ' ' || standard_error.back() == '\t')) {
    standard_error.remove_suffix(1);
  }
  if (standard_error.empty()) {
    return {};
  }
  constexpr std::size_t kDisplayedErrorLimit = 4096;
  if (standard_error.size() > kDisplayedErrorLimit) {
    standard_error = standard_error.substr(0, kDisplayedErrorLimit);
  }
  return "; stderr: " + std::string(standard_error);
}

[[nodiscard]] std::vector<Solution> run_solver_subprocess(
    const std::filesystem::path& solver_path, const std::filesystem::path& book_path,
    const std::vector<Position>& positions) {
  if (positions.empty()) {
    return {};
  }
  if (::access(solver_path.c_str(), X_OK) != 0) {
    throw_system_error(
        "solver executable is unavailable at '" + solver_path.string() + "'", errno);
  }
  if (::access(book_path.c_str(), R_OK) != 0) {
    throw_system_error("solver book is unavailable at '" + book_path.string() + "'",
                       errno);
  }

  std::vector<std::string> move_sequences;
  move_sequences.reserve(positions.size());
  std::string input;
  for (const Position& position : positions) {
    move_sequences.push_back(position_move_sequence(position));
    input += move_sequences.back();
    input.push_back('\n');
  }

  const auto deadline = solver_deadline();
  Pipe child_input = make_pipe();
  Pipe child_output = make_pipe();
  Pipe child_error = make_pipe();

  SpawnFileActions actions;
  actions.duplicate(child_input.read.get(), STDIN_FILENO);
  actions.duplicate(child_output.write.get(), STDOUT_FILENO);
  actions.duplicate(child_error.write.get(), STDERR_FILENO);
  actions.close(child_input.read.get());
  actions.close(child_input.write.get());
  actions.close(child_output.read.get());
  actions.close(child_output.write.get());
  actions.close(child_error.read.get());
  actions.close(child_error.write.get());

  std::string solver_argument = solver_path.string();
  std::string book_argument = book_path.string();
  std::array<char*, 5> arguments{solver_argument.data(), const_cast<char*>("-b"),
                                 book_argument.data(), const_cast<char*>("-a"),
                                 nullptr};

  pid_t process = 0;
  const int spawn_result = posix_spawn(&process, solver_argument.c_str(), actions.get(),
                                       nullptr, arguments.data(), environ);
  if (spawn_result != 0) {
    throw_system_error(
        "failed to start solver subprocess '" + solver_path.string() + "'",
        spawn_result);
  }
  ChildProcess child(process);

  child_input.read.reset();
  child_output.write.reset();
  child_error.write.reset();

  set_nonblocking(child_input.write.get());
  const auto write_error = write_all(child_input.write.get(), input, deadline);
  child_input.write.reset();

  CapturedOutput output = read_process_output(std::move(child_output.read),
                                              std::move(child_error.read), deadline);
  const int status = child.wait(deadline);
  if (write_error.has_value()) {
    throw std::runtime_error(*write_error + describe_stderr(output.standard_error));
  }
  if (WIFSIGNALED(status)) {
    throw std::runtime_error("solver subprocess terminated by signal " +
                             std::to_string(WTERMSIG(status)) +
                             describe_stderr(output.standard_error));
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    const std::string exit_description =
        WIFEXITED(status) ? std::to_string(WEXITSTATUS(status)) : "unknown";
    throw std::runtime_error("solver subprocess exited with status " +
                             exit_description + describe_stderr(output.standard_error));
  }

  return parse_solver_output(output.standard_output, move_sequences);
}

}  // namespace

float score_policy(const Solution& solution, const Policy& policy) {
  if (std::any_of(policy.begin(), policy.end(),
                  [](float probability) { return !std::isfinite(probability); })) {
    throw std::invalid_argument("policy values must all be finite");
  }

  const auto best_solution = std::max_element(solution.begin(), solution.end());
  const auto selected = std::max_element(policy.begin(), policy.end());
  const std::size_t selected_index =
      static_cast<std::size_t>(selected - policy.begin());
  if (solution[selected_index] == *best_solution) {
    return 1.0F;
  }
  return solution[selected_index] > 0 ? 0.5F : 0.0F;
}

struct CachingSolver::Impl {
  Impl(std::filesystem::path solver, std::filesystem::path book,
       std::filesystem::path cache)
      : solver_path(std::move(solver)), book_path(std::move(book)) {
    if (cache.empty()) {
      throw std::invalid_argument("solver cache path must not be empty");
    }
    std::error_code filesystem_error;
    if (std::filesystem::is_directory(cache, filesystem_error)) {
      throw std::invalid_argument("solver cache path is a directory: '" +
                                  cache.string() + "'");
    }

    sqlite3* opened = nullptr;
    const int open_result = sqlite3_open_v2(
        cache.c_str(), &opened,
        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX, nullptr);
    if (open_result != SQLITE_OK) {
      const std::string message =
          opened == nullptr ? sqlite3_errstr(open_result) : sqlite3_errmsg(opened);
      if (opened != nullptr) {
        sqlite3_close(opened);
      }
      throw std::runtime_error("failed to open solver cache '" + cache.string() +
                               "': " + message);
    }
    database = opened;

    try {
      sqlite3_extended_result_codes(database, 1);
      const int timeout_result = sqlite3_busy_timeout(database, 5000);
      if (timeout_result != SQLITE_OK) {
        throw_sqlite_error(database, "failed to configure cache timeout",
                           timeout_result);
      }
      initialize_schema();
    } catch (...) {
      sqlite3_close(database);
      database = nullptr;
      throw;
    }
  }

  ~Impl() {
    if (database != nullptr) {
      sqlite3_close_v2(database);
    }
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  void initialize_schema() {
    Transaction transaction(database);
    execute_sql(database,
                "CREATE TABLE IF NOT EXISTS cache_metadata ("
                "format TEXT PRIMARY KEY NOT NULL, "
                "version INTEGER NOT NULL CHECK(version > 0)) WITHOUT ROWID",
                "failed to create solver cache metadata");

    Statement metadata(database, "SELECT format, version FROM cache_metadata",
                       "failed to inspect solver cache metadata");
    int result = sqlite3_step(metadata.get());
    if (result == SQLITE_DONE) {
      Statement insert_metadata(
          database, "INSERT INTO cache_metadata(format, version) VALUES(?1, ?2)",
          "failed to prepare solver cache metadata");
      result = sqlite3_bind_text(insert_metadata.get(), 1, kCacheFormat.data(),
                                 static_cast<int>(kCacheFormat.size()), SQLITE_STATIC);
      if (result != SQLITE_OK) {
        throw_sqlite_error(database, "failed to bind cache format", result);
      }
      result = sqlite3_bind_int(insert_metadata.get(), 2, kCacheVersion);
      if (result != SQLITE_OK) {
        throw_sqlite_error(database, "failed to bind cache version", result);
      }
      result = sqlite3_step(insert_metadata.get());
      if (result != SQLITE_DONE) {
        throw_sqlite_error(database, "failed to store solver cache metadata", result);
      }
    } else if (result == SQLITE_ROW) {
      const auto* format =
          reinterpret_cast<const char*>(sqlite3_column_text(metadata.get(), 0));
      const int version = sqlite3_column_int(metadata.get(), 1);
      const std::string actual_format = format == nullptr ? "" : format;
      if (actual_format != kCacheFormat || version != kCacheVersion) {
        throw std::runtime_error(
            "unsupported solver cache format/version: '" + actual_format +
            "' version " + std::to_string(version) + "; expected '" +
            std::string(kCacheFormat) + "' version " + std::to_string(kCacheVersion));
      }
      result = sqlite3_step(metadata.get());
      if (result == SQLITE_ROW) {
        throw std::runtime_error("solver cache contains multiple metadata records");
      }
      if (result != SQLITE_DONE) {
        throw_sqlite_error(database, "failed to read solver cache metadata", result);
      }
    } else {
      throw_sqlite_error(database, "failed to read solver cache metadata", result);
    }

    execute_sql(database,
                "CREATE TABLE IF NOT EXISTS solutions ("
                "mask INTEGER NOT NULL, value INTEGER NOT NULL, scores BLOB NOT NULL "
                "CHECK(typeof(scores) = 'blob' AND length(scores) = 14), "
                "PRIMARY KEY(mask, value)) WITHOUT ROWID",
                "failed to create solver cache records");
    transaction.commit();
  }

  [[nodiscard]] std::optional<Solution> get(const Position& position) {
    Statement query(database,
                    "SELECT scores FROM solutions "
                    "WHERE mask = ?1 AND value = ?2",
                    "failed to prepare solver cache lookup");
    bind_position(database, query.get(), position);
    const int result = sqlite3_step(query.get());
    if (result == SQLITE_DONE) {
      return std::nullopt;
    }
    if (result != SQLITE_ROW) {
      throw_sqlite_error(database, "failed to read solver cache", result);
    }
    return decode_solution(sqlite3_column_blob(query.get(), 0),
                           sqlite3_column_bytes(query.get(), 0));
  }

  void put_all(const std::vector<Position>& positions,
               const std::vector<Solution>& solutions) {
    if (positions.size() != solutions.size()) {
      throw std::logic_error("internal solver cache result count mismatch");
    }
    if (positions.empty()) {
      return;
    }

    Transaction transaction(database);
    Statement insert(database,
                     "INSERT INTO solutions(mask, value, scores) VALUES(?1, ?2, ?3) "
                     "ON CONFLICT(mask, value) DO NOTHING",
                     "failed to prepare solver cache insert");
    for (std::size_t index = 0; index < positions.size(); ++index) {
      bind_position(database, insert.get(), positions[index]);
      const auto encoded = encode_solution(solutions[index]);
      int result =
          sqlite3_bind_blob(insert.get(), 3, encoded.data(),
                            static_cast<int>(encoded.size()), SQLITE_TRANSIENT);
      if (result != SQLITE_OK) {
        throw_sqlite_error(database, "failed to bind solver solution", result);
      }
      result = sqlite3_step(insert.get());
      if (result != SQLITE_DONE) {
        throw_sqlite_error(database, "failed to insert solver solution", result);
      }
      result = sqlite3_reset(insert.get());
      if (result != SQLITE_OK) {
        throw_sqlite_error(database, "failed to reset solver cache insert", result);
      }
      result = sqlite3_clear_bindings(insert.get());
      if (result != SQLITE_OK) {
        throw_sqlite_error(database, "failed to clear solver cache insert bindings",
                           result);
      }
    }
    transaction.commit();
  }

  [[nodiscard]] std::vector<float> score(
      const std::vector<std::pair<Position, Policy>>& positions_and_policies) {
    std::lock_guard lock(database_mutex);
    if (positions_and_policies.empty()) {
      return {};
    }

    std::vector<Position> missing_positions;
    std::unordered_set<Position, PositionHash> seen_missing;
    for (const auto& [position, policy] : positions_and_policies) {
      static_cast<void>(policy);
      if (!get(position).has_value() && seen_missing.insert(position).second) {
        missing_positions.push_back(position);
      }
    }

    std::vector<Solution> missing_solutions;
    missing_solutions.reserve(missing_positions.size());
    for (std::size_t offset = 0; offset < missing_positions.size();
         offset += kSolverChunkSize) {
      const std::size_t end =
          std::min(offset + kSolverChunkSize, missing_positions.size());
      const std::vector<Position> chunk(
          missing_positions.begin() + static_cast<std::ptrdiff_t>(offset),
          missing_positions.begin() + static_cast<std::ptrdiff_t>(end));
      auto chunk_solutions = run_solver_subprocess(solver_path, book_path, chunk);
      missing_solutions.insert(missing_solutions.end(),
                               std::make_move_iterator(chunk_solutions.begin()),
                               std::make_move_iterator(chunk_solutions.end()));
    }
    put_all(missing_positions, missing_solutions);

    std::vector<float> scores;
    scores.reserve(positions_and_policies.size());
    for (const auto& [position, policy] : positions_and_policies) {
      const auto solution = get(position);
      if (!solution.has_value()) {
        throw std::runtime_error("solver cache lookup failed after storing a solution");
      }
      scores.push_back(score_policy(*solution, policy));
    }
    return scores;
  }

  std::filesystem::path solver_path;
  std::filesystem::path book_path;
  sqlite3* database{};
  std::mutex database_mutex;
};

CachingSolver::CachingSolver(std::filesystem::path solver_path,
                             std::filesystem::path book_path,
                             std::filesystem::path cache_path)
    : impl_(std::make_unique<Impl>(std::move(solver_path), std::move(book_path),
                                   std::move(cache_path))) {}

CachingSolver::~CachingSolver() = default;
CachingSolver::CachingSolver(CachingSolver&&) noexcept = default;
CachingSolver& CachingSolver::operator=(CachingSolver&&) noexcept = default;

std::vector<float> CachingSolver::score_policies(
    const std::vector<std::pair<Position, Policy>>& positions_and_policies) {
  if (impl_ == nullptr) {
    throw std::logic_error("cannot use a moved-from CachingSolver");
  }
  return impl_->score(positions_and_policies);
}

}  // namespace c4a0
