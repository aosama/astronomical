import Foundation

/// Generations remain optional because unavailable or pre-acknowledgement status cannot safely
/// claim that persisted configuration is effective.
struct ConfigurationStatusDocument: Codable, Equatable {
  let configuredGeneration: String?
  let resolvedGeneration: String?
  let effectiveGeneration: String?
  let isEffective: Bool
  let restartRequired: Bool

  enum CodingKeys: String, CodingKey {
    case configuredGeneration = "configured_generation"
    case resolvedGeneration = "resolved_generation"
    case effectiveGeneration = "effective_generation"
    case isEffective = "is_effective"
    case restartRequired = "restart_required"
  }
}
