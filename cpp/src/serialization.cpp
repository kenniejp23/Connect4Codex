#include "c4a0/serialization.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace c4a0 {
namespace {

using Json = nlohmann::json;

constexpr std::string_view kFormat = "c4a0_cpp.play_games";
constexpr std::uint32_t kVersion = 1;

Json sample_to_json(const Sample& sample) {
  return Json{{"mask", sample.pos.mask()},
              {"value", sample.pos.value()},
              {"policy", sample.policy},
              {"q_penalty", sample.q_penalty},
              {"q_no_penalty", sample.q_no_penalty}};
}

Sample sample_from_json(const Json& value) {
  Sample sample;
  sample.pos = Position::from_bits(value.at("mask").get<std::uint64_t>(),
                                   value.at("value").get<std::uint64_t>());
  const auto policy = value.at("policy").get<std::vector<float>>();
  if (policy.size() != kCols) {
    throw std::invalid_argument("CBOR sample policy must have seven values");
  }
  std::copy(policy.begin(), policy.end(), sample.policy.begin());
  sample.q_penalty = value.at("q_penalty").get<float>();
  sample.q_no_penalty = value.at("q_no_penalty").get<float>();
  return sample;
}

}  // namespace

std::vector<std::uint8_t> to_cbor(const PlayGamesResult& value) {
  Json root{{"format", kFormat}, {"version", kVersion}};
  auto& json_results = root["results"] = Json::array();
  for (const auto& result : value.results) {
    Json json_result{{"metadata",
                      {{"game_id", result.metadata.game_id},
                       {"player0_id", result.metadata.player0_id},
                       {"player1_id", result.metadata.player1_id}}},
                     {"samples", Json::array()}};
    for (const auto& sample : result.samples) {
      json_result["samples"].push_back(sample_to_json(sample));
    }
    json_results.push_back(std::move(json_result));
  }
  return Json::to_cbor(root);
}

PlayGamesResult from_cbor(std::span<const std::uint8_t> bytes) {
  try {
    const auto root = Json::from_cbor(bytes.begin(), bytes.end(), true, true);
    if (root.at("format").get<std::string>() != kFormat) {
      throw std::invalid_argument("unsupported PlayGamesResult CBOR format");
    }
    if (root.at("version").get<std::uint32_t>() != kVersion) {
      throw std::invalid_argument("unsupported PlayGamesResult CBOR version");
    }

    PlayGamesResult value;
    for (const auto& json_result : root.at("results")) {
      const auto& metadata = json_result.at("metadata");
      GameResult result{
          .metadata =
              GameMetadata{.game_id = metadata.at("game_id").get<std::uint64_t>(),
                           .player0_id = metadata.at("player0_id").get<ModelId>(),
                           .player1_id = metadata.at("player1_id").get<ModelId>()},
          .samples = {}};
      for (const auto& sample : json_result.at("samples")) {
        result.samples.push_back(sample_from_json(sample));
      }
      value.results.push_back(std::move(result));
    }
    return value;
  } catch (const std::invalid_argument&) {
    throw;
  } catch (const std::exception& error) {
    throw std::invalid_argument(std::string("invalid PlayGamesResult CBOR: ") +
                                error.what());
  }
}

}  // namespace c4a0
