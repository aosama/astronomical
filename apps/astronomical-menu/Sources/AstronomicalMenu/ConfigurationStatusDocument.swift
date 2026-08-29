import Foundation

/// Generations remain optional because unavailable or pre-acknowledgement status cannot safely
/// claim that persisted configuration is effective.
struct ConfigurationStatusDocument: Codable, Equatable {
  let configuredGeneration: String?
  let resolvedGeneration: String?
  let effectiveGeneration: String?
  let isEffective: Bool
  let restartRequired: Bool
  let modelDiscoveryDiagnostics: [ModelDiscoveryDiagnosticDocument]?

  enum CodingKeys: String, CodingKey {
    case configuredGeneration = "configured_generation"
    case resolvedGeneration = "resolved_generation"
    case effectiveGeneration = "effective_generation"
    case isEffective = "is_effective"
    case restartRequired = "restart_required"
    case modelDiscoveryDiagnostics = "model_discovery_diagnostics"
  }
}

struct ModelDiscoveryDiagnosticDocument: Codable, Equatable {
  let code: String
  let modelID: String
  let configuredRootNumbers: [Int]

  var message: String {
    let entries = configuredRootNumbers.map(String.init).joined(separator: ", ")
    if code == "unavailable_model_directory" {
      return "model_directories entry \(entries) is missing or unreadable. Other models remain available."
    }
    return "Model \(modelID) appears in model_directories entries \(entries). Remove one duplicate root."
  }

  enum CodingKeys: String, CodingKey {
    case code
    case modelID = "model_id"
    case configuredRootNumbers = "configured_root_numbers"
  }
}
