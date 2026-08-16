cmake_minimum_required(VERSION 3.28)

project(c4a0 LANGUAGES C CXX)

include(CTest)
include(CheckIPOSupported)
include(FetchContent)

option(C4A0_BUILD_PYTHON "Build the c4a0_cpp Python extension" ON)
option(C4A0_BUILD_TESTS "Build native tests" ${BUILD_TESTING})
option(C4A0_BUILD_BENCHMARKS "Build native benchmarks" OFF)
option(C4A0_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(C4A0_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(C4A0_ENABLE_TSAN "Enable ThreadSanitizer" OFF)

if(C4A0_ENABLE_TSAN AND (C4A0_ENABLE_ASAN OR C4A0_ENABLE_UBSAN))
  message(FATAL_ERROR "ThreadSanitizer must be enabled on its own")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The engine is deliberately split into small translation units. Release LTO
# lets hot MCTS calls inline Position operations across those boundaries, which
# matches the whole-crate optimization available to the reference build.
set(C4A0_RELEASE_IPO OFF)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  check_ipo_supported(RESULT C4A0_IPO_SUPPORTED OUTPUT C4A0_IPO_ERROR)
  if(NOT C4A0_IPO_SUPPORTED)
    message(FATAL_ERROR "Release interprocedural optimization is unavailable: ${C4A0_IPO_ERROR}")
  endif()
  set(C4A0_RELEASE_IPO ON)
endif()

set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG d34771cafc87b358ba421faca28facc7f8080174
  GIT_SHALLOW TRUE
)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlohmann_json)
# nlohmann/json is a pinned third-party header dependency. Treat its headers
# as system headers so project-wide -Werror does not turn diagnostics in that
# dependency into c4a0 build failures.
set_property(
  TARGET nlohmann_json
  APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
  "$<TARGET_PROPERTY:nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>"
)

set(FTXUI_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FTXUI_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  ftxui
  GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI.git
  GIT_TAG f921fad208912747c17d129a8ef75ec7624b6eec
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(ftxui)

FetchContent_Declare(
  sqlite_amalgamation
  URL https://www.sqlite.org/2026/sqlite-amalgamation-3530400.zip
  URL_HASH SHA3_256=628a44cfe82c66aed1ccbbe85a562d2e33ebe64b3288981ed76285612227934e
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(sqlite_amalgamation)
add_library(c4a0_sqlite STATIC "${sqlite_amalgamation_SOURCE_DIR}/sqlite3.c")
target_include_directories(c4a0_sqlite PUBLIC "${sqlite_amalgamation_SOURCE_DIR}")
target_compile_definitions(c4a0_sqlite PRIVATE
  SQLITE_DQS=0
  SQLITE_DEFAULT_MEMSTATUS=0
  SQLITE_OMIT_DEPRECATED
  SQLITE_THREADSAFE=1
)

find_package(Threads REQUIRED)

add_library(c4a0_core STATIC
  cpp/src/interactive.cpp
  cpp/src/mcts.cpp
  cpp/src/position.cpp
  cpp/src/rng.cpp
  cpp/src/self_play.cpp
  cpp/src/serialization.cpp
  cpp/src/solver.cpp
  cpp/src/tui.cpp
)
target_include_directories(c4a0_core PUBLIC cpp/include)
target_link_libraries(c4a0_core
  PUBLIC Threads::Threads
  PRIVATE
    c4a0_sqlite
    nlohmann_json::nlohmann_json
    ftxui::component
    ftxui::dom
    ftxui::screen
)
target_compile_features(c4a0_core PUBLIC cxx_std_20)
set_property(TARGET c4a0_core PROPERTY INTERPROCEDURAL_OPTIMIZATION ${C4A0_RELEASE_IPO})

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(c4a0_core PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
  )
endif()

set(C4A0_SANITIZER_FLAGS "")
if(C4A0_ENABLE_ASAN)
  list(APPEND C4A0_SANITIZER_FLAGS -fsanitize=address -fno-omit-frame-pointer)
endif()
if(C4A0_ENABLE_UBSAN)
  list(APPEND C4A0_SANITIZER_FLAGS -fsanitize=undefined -fno-omit-frame-pointer)
endif()
if(C4A0_ENABLE_TSAN)
  list(APPEND C4A0_SANITIZER_FLAGS -fsanitize=thread -fno-omit-frame-pointer)
endif()
if(C4A0_SANITIZER_FLAGS)
  target_compile_options(c4a0_core PUBLIC ${C4A0_SANITIZER_FLAGS})
  target_link_options(c4a0_core PUBLIC ${C4A0_SANITIZER_FLAGS})
endif()

if(C4A0_BUILD_PYTHON)
  find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_native NB_STATIC cpp/src/bindings.cpp)
  target_link_libraries(_native PRIVATE c4a0_core)
  target_compile_features(_native PRIVATE cxx_std_20)
  set_property(TARGET _native PROPERTY INTERPROCEDURAL_OPTIMIZATION ${C4A0_RELEASE_IPO})
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(_native PRIVATE
      -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
    )
  endif()
  install(TARGETS _native LIBRARY DESTINATION c4a0_cpp)
endif()

if(C4A0_BUILD_TESTS)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG 644821ce28cb25d7992a4d0375b1d83214392592
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(Catch2)
  list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
  include(Catch)

  add_executable(c4a0_cpp_tests
    cpp/tests/core_tests.cpp
    cpp/tests/fixture_tests.cpp
    cpp/tests/runtime_tests.cpp
    cpp/tests/solver_tests.cpp
    cpp/tests/tui_tests.cpp
  )
  target_link_libraries(c4a0_cpp_tests PRIVATE
    c4a0_core
    Catch2::Catch2WithMain
    nlohmann_json::nlohmann_json
  )
  target_compile_definitions(c4a0_cpp_tests PRIVATE
    C4A0_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
  )
  target_compile_features(c4a0_cpp_tests PRIVATE cxx_std_20)
  set_property(TARGET c4a0_cpp_tests PROPERTY INTERPROCEDURAL_OPTIMIZATION ${C4A0_RELEASE_IPO})
  catch_discover_tests(c4a0_cpp_tests)
endif()

if(C4A0_BUILD_BENCHMARKS)
  add_executable(c4a0_benchmarks benchmarks/native_benchmarks.cpp)
  target_link_libraries(c4a0_benchmarks PRIVATE c4a0_core)
  set_property(TARGET c4a0_benchmarks PROPERTY INTERPROCEDURAL_OPTIMIZATION ${C4A0_RELEASE_IPO})
endif()
