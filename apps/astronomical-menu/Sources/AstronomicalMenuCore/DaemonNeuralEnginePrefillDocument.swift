// Decodes what the Neural Engine is doing for the loaded model. The state stays a string so a state
// added by a newer daemon shows nothing instead of invalidating the whole status document.

import Foundation

extension DaemonStatusDocument {
  struct NeuralEnginePrefill: Codable, Equatable {
    let state: String
    let completedProgramCount: UInt32?
    let totalProgramCount: UInt32?
    let estimatedRemainingMilliseconds: UInt64?
    let reason: String?

    enum CodingKeys: String, CodingKey {
      case state, reason
      case completedProgramCount = "completed_program_count"
      case totalProgramCount = "total_program_count"
      case estimatedRemainingMilliseconds = "estimated_remaining_ms"
    }
  }
}
