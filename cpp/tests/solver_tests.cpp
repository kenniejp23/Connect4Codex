#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "c4a0/solver.hpp"

namespace c4a0 {
namespace {

using Catch::Matchers::ContainsSubstring;

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    static std::atomic<unsigned long> sequence{};
    path_ = std::filesystem::temp_directory_path() /
            ("c4a0-solver-test-" + std::to_string(::getpid()) + "-" +
             std::to_string(sequence.fetch_add(1)));
    std::filesystem::create_directories(path_);
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
};

void write_text(const std::filesystem::path& path, std::string_view contents) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  REQUIRE(stream.good());
  stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  stream.close();
  REQUIRE(stream.good());
}

void write_executable(const std::filesystem::path& path, std::string_view contents) {
  write_text(path, contents);
  REQUIRE(::chmod(path.c_str(), S_IRUSR | S_IWUSR | S_IXUSR) == 0);
}

[[nodiscard]] std::size_t line_count(const std::filesystem::path& path) {
  std::ifstream stream(path);
  std::size_t count = 0;
  std::string line;
  while (std::getline(stream, line)) {
    ++count;
  }
  return count;
}

[[nodiscard]] Policy one_hot(std::size_t column) {
  Policy policy{};
  policy[column] = 1.0F;
  return policy;
}

[[nodiscard]] std::string successful_solver_script(
    const std::filesystem::path& invocation_log,
    const std::filesystem::path& request_log) {
  return "#!/bin/sh\n"
         "if [ \"$1\" != \"-b\" ] || [ \"$3\" != \"-a\" ] || "
         "[ ! -r \"$2\" ]; then\n"
         "  echo 'invalid arguments' >&2\n"
         "  exit 64\n"
         "fi\n"
         "printf 'invoked\\n' >> '" +
         invocation_log.string() +
         "'\n"
         "while IFS= read -r moves; do\n"
         "  printf '%s\\n' \"$moves\" >> '" +
         request_log.string() +
         "'\n"
         "  if [ -n \"$moves\" ]; then printf '%s ' \"$moves\"; fi\n"
         "  printf '0 5 3 4 1 -1 -2\\n'\n"
         "done\n";
}

[[nodiscard]] std::vector<Position> distinct_positions(std::size_t count) {
  std::vector<Position> positions{Position{}};
  for (std::size_t cursor = 0; cursor < positions.size() && positions.size() < count;
       ++cursor) {
    if (positions[cursor].terminal_state().has_value()) {
      continue;
    }
    for (std::size_t column = 0; column < kCols && positions.size() < count; ++column) {
      const auto next = positions[cursor].make_move(column);
      if (!next.has_value()) {
        continue;
      }
      bool duplicate = false;
      for (const Position& existing : positions) {
        if (existing == *next) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) {
        positions.push_back(*next);
      }
    }
  }
  REQUIRE(positions.size() == count);
  return positions;
}

TEST_CASE("solver policy scoring preserves reference semantics", "[solver]") {
  const Solution solution{0, 5, 3, 4, 1, -1, -2};

  CHECK(score_policy(solution, one_hot(1)) == 1.0F);
  CHECK(score_policy(solution, one_hot(2)) == 0.5F);
  CHECK(score_policy(solution, one_hot(5)) == 0.0F);

  SECTION("the first policy maximum wins a tie") {
    Policy policy{};
    policy[1] = 0.7F;
    policy[2] = 0.7F;
    CHECK(score_policy(solution, policy) == 1.0F);
  }

  SECTION("all-negative solutions still award their best move") {
    const Solution losing{-4, -3, -2, -1, -2, -3, -4};
    CHECK(score_policy(losing, one_hot(3)) == 1.0F);
    CHECK(score_policy(losing, one_hot(2)) == 0.0F);
  }

  SECTION("non-finite policies are rejected") {
    Policy policy = one_hot(0);
    policy[4] = std::numeric_limits<float>::quiet_NaN();
    CHECK_THROWS_WITH(score_policy(solution, policy), ContainsSubstring("finite"));
  }
}

TEST_CASE(
    "solver deduplicates, persists, and serves cache hits without the "
    "executable",
    "[solver][cache]") {
  TemporaryDirectory temporary;
  const auto executable = temporary.path() / "fake-solver";
  const auto book = temporary.path() / "book";
  const auto database = temporary.path() / "solutions.db";
  const auto invocation_log = temporary.path() / "invocations.log";
  const auto request_log = temporary.path() / "requests.log";
  write_text(book, "fake book");
  write_executable(executable, successful_solver_script(invocation_log, request_log));

  const Position initial;
  const Position after_move = Position::from_moves({3});
  const std::vector<std::pair<Position, Policy>> requests{
      {initial, one_hot(1)},
      {after_move, one_hot(2)},
      {initial, one_hot(5)},
  };

  {
    CachingSolver solver(executable, book, database);
    const auto scores = solver.score_policies(requests);
    REQUIRE(scores.size() == 3);
    CHECK(scores[0] == 1.0F);
    CHECK(scores[1] == 0.5F);
    CHECK(scores[2] == 0.0F);
  }
  CHECK(line_count(invocation_log) == 1);
  CHECK(line_count(request_log) == 2);

  std::filesystem::remove(executable);
  std::filesystem::remove(book);
  CachingSolver cache_only(executable, book, database);
  CHECK(cache_only.score_policies(requests) == std::vector<float>{1.0F, 0.5F, 0.0F});
  CHECK(line_count(invocation_log) == 1);
}

TEST_CASE("solver chunks more than one hundred unique positions", "[solver][cache]") {
  TemporaryDirectory temporary;
  const auto executable = temporary.path() / "fake-solver";
  const auto book = temporary.path() / "book";
  const auto database = temporary.path() / "solutions.db";
  const auto invocation_log = temporary.path() / "invocations.log";
  const auto request_log = temporary.path() / "requests.log";
  write_text(book, "fake book");
  write_executable(executable, successful_solver_script(invocation_log, request_log));

  std::vector<std::pair<Position, Policy>> requests;
  for (const Position& position : distinct_positions(101)) {
    requests.emplace_back(position, one_hot(1));
  }
  requests.push_back(requests.front());

  CachingSolver solver(executable, book, database);
  const auto scores = solver.score_policies(requests);
  CHECK(scores.size() == requests.size());
  CHECK(line_count(invocation_log) == 2);
  CHECK(line_count(request_log) == 101);
}

TEST_CASE("solver validates subprocess output and exit status", "[solver]") {
  TemporaryDirectory temporary;
  const auto book = temporary.path() / "book";
  write_text(book, "fake book");

  SECTION("malformed fields") {
    const auto executable = temporary.path() / "malformed-solver";
    write_executable(executable,
                     "#!/bin/sh\n"
                     "IFS= read -r moves\n"
                     "printf '%s 0 1 nope 3 4 5 6\\n' \"$moves\"\n");
    CachingSolver solver(executable, book, temporary.path() / "malformed.db");
    CHECK_THROWS_WITH(solver.score_policies({{Position::from_moves({3}), one_hot(0)}}),
                      ContainsSubstring("invalid solver score"));
  }

  SECTION("wrong echoed move sequence") {
    const auto executable = temporary.path() / "wrong-echo-solver";
    write_executable(executable,
                     "#!/bin/sh\n"
                     "IFS= read -r moves\n"
                     "printf '777 0 1 2 3 4 5 6\\n'\n");
    CachingSolver solver(executable, book, temporary.path() / "wrong-echo.db");
    CHECK_THROWS_WITH(solver.score_policies({{Position::from_moves({3}), one_hot(0)}}),
                      ContainsSubstring("expected"));
  }

  SECTION("non-zero exit") {
    const auto executable = temporary.path() / "failing-solver";
    write_executable(executable,
                     "#!/bin/sh\n"
                     "echo 'deliberate failure' >&2\n"
                     "exit 23\n");
    CachingSolver solver(executable, book, temporary.path() / "failure.db");
    CHECK_THROWS_WITH(solver.score_policies({{Position{}, one_hot(0)}}),
                      ContainsSubstring("status 23"));
  }
}

TEST_CASE("solver reports path and cache errors", "[solver][cache]") {
  TemporaryDirectory temporary;
  const auto missing_solver = temporary.path() / "missing-solver";
  const auto missing_book = temporary.path() / "missing-book";

  SECTION("empty requests do not require external solver files") {
    CachingSolver solver(missing_solver, missing_book, temporary.path() / "empty.db");
    CHECK(solver.score_policies({}).empty());
  }

  SECTION("missing executable") {
    write_text(missing_book, "fake book");
    CachingSolver solver(missing_solver, missing_book,
                         temporary.path() / "missing-executable.db");
    CHECK_THROWS_WITH(solver.score_policies({{Position{}, one_hot(0)}}),
                      ContainsSubstring("executable is unavailable"));
  }

  SECTION("missing book") {
    const auto executable = temporary.path() / "solver";
    write_executable(executable, "#!/bin/sh\nexit 0\n");
    CachingSolver solver(executable, missing_book,
                         temporary.path() / "missing-book.db");
    CHECK_THROWS_WITH(solver.score_policies({{Position{}, one_hot(0)}}),
                      ContainsSubstring("book is unavailable"));
  }

  SECTION("a directory cannot be opened as the cache file") {
    CHECK_THROWS_WITH(CachingSolver(missing_solver, missing_book, temporary.path()),
                      ContainsSubstring("directory"));
  }

  SECTION("a non-SQLite file is rejected") {
    const auto corrupt = temporary.path() / "corrupt.db";
    write_text(corrupt, "this is not a SQLite database");
    CHECK_THROWS_WITH(CachingSolver(missing_solver, missing_book, corrupt),
                      ContainsSubstring("not a database"));
  }
}

}  // namespace
}  // namespace c4a0
