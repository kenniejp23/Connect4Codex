#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "c4a0/position.hpp"

namespace c4a0 {

[[nodiscard]] std::vector<std::uint8_t> to_cbor(const PlayGamesResult& value);
[[nodiscard]] PlayGamesResult from_cbor(std::span<const std::uint8_t> bytes);

}  // namespace c4a0
