// Decodes the public status progress union without forcing image steps into token-specific fields.
// Keeping this wire boundary explicit prevents a newly added server activity from invalidating the
// complete menu status document and hiding otherwise valid model telemetry.

import Foundation

extension SupervisorStatusDocument {
  struct Progress: Codable, Equatable {
    enum Unit: Equatable {
      case tokens
      case steps
    }

    let phase: String
    let completedUnitCount: UInt32
    let totalUnitCount: UInt32
    let elapsedMilliseconds: UInt64
    let unit: Unit

    enum CodingKeys: String, CodingKey {
      case phase
      case processedTokens = "processed_tokens"
      case totalTokens = "total_tokens"
      case completedSteps = "completed_steps"
      case totalSteps = "total_steps"
      case elapsedMilliseconds = "elapsed_ms"
    }

    init(from decoder: Decoder) throws {
      let container = try decoder.container(keyedBy: CodingKeys.self)
      phase = try container.decode(String.self, forKey: .phase)
      elapsedMilliseconds = try container.decode(UInt64.self, forKey: .elapsedMilliseconds)

      let processedTokens = try container.decodeIfPresent(UInt32.self, forKey: .processedTokens)
      let totalTokens = try container.decodeIfPresent(UInt32.self, forKey: .totalTokens)
      let completedSteps = try container.decodeIfPresent(UInt32.self, forKey: .completedSteps)
      let totalSteps = try container.decodeIfPresent(UInt32.self, forKey: .totalSteps)

      switch (processedTokens, totalTokens, completedSteps, totalSteps) {
      case let (.some(completedTokenCount), .some(totalTokenCount), .none, .none):
        completedUnitCount = completedTokenCount
        totalUnitCount = totalTokenCount
        unit = .tokens
      case let (.none, .none, .some(completedStepCount), .some(totalStepCount)):
        completedUnitCount = completedStepCount
        totalUnitCount = totalStepCount
        unit = .steps
      default:
        throw DecodingError.dataCorruptedError(
          forKey: .phase,
          in: container,
          debugDescription: "Progress must contain exactly one complete token or step count pair"
        )
      }
    }

    func encode(to encoder: Encoder) throws {
      var container = encoder.container(keyedBy: CodingKeys.self)
      try container.encode(phase, forKey: .phase)
      try container.encode(elapsedMilliseconds, forKey: .elapsedMilliseconds)
      switch unit {
      case .tokens:
        try container.encode(completedUnitCount, forKey: .processedTokens)
        try container.encode(totalUnitCount, forKey: .totalTokens)
      case .steps:
        try container.encode(completedUnitCount, forKey: .completedSteps)
        try container.encode(totalUnitCount, forKey: .totalSteps)
      }
    }

    var completionPercentageTitle: String {
      guard totalUnitCount > 0 else { return "0%" }
      let boundedCompletedUnitCount = min(completedUnitCount, totalUnitCount)
      let completionPercentage =
        Double(boundedCompletedUnitCount) / Double(totalUnitCount) * 100
      return "\(Int(completionPercentage.rounded(.down)))%"
    }

    var hasCompletedDenoising: Bool {
      phase == "denoising" && totalUnitCount > 0 && completedUnitCount >= totalUnitCount
    }

    var imagePhaseTitle: String {
      switch phase {
      case "preparing": "Preparing image"
      case "encoding_prompt": "Encoding prompt"
      case "denoising": "Generating image"
      case "decoding": "Decoding image"
      case "encoding_image": "Encoding image"
      default: "Generating image"
      }
    }

    var shortImagePhaseTitle: String {
      switch phase {
      case "preparing": "Preparing"
      case "encoding_prompt": "Prompt"
      case "denoising": "Generating"
      case "decoding": "Decoding"
      case "encoding_image": "Encoding"
      default: "Active"
      }
    }
  }
}
