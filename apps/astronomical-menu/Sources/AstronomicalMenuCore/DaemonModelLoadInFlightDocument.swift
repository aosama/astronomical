// Decodes the daemon's report that a model load is under way.

import Foundation

extension DaemonStatusDocument {
  struct ModelLoadInFlight: Codable, Equatable {
    let modelIdentifier: String?

    enum CodingKeys: String, CodingKey {
      case modelIdentifier = "model_id"
    }
  }
}
