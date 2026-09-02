/// Concrete sparse-expert ownership reported by the active worker.
struct ExpertResidencySnapshot: Codable, Equatable {
  let totalLayerCount: UInt32
  let residentExpertCount: UInt32
  let residentExpertPayloadBytes: UInt64

  enum CodingKeys: String, CodingKey {
    case totalLayerCount = "total_layer_count"
    case residentExpertCount = "resident_expert_count"
    case residentExpertPayloadBytes = "resident_expert_payload_bytes"
  }
}
