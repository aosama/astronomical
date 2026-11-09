/// Concrete sparse-expert ownership reported by the active worker, and for a
/// dense model on the streaming engine, where its decoder layers live.
struct ExpertResidencySnapshot: Codable, Equatable {
  let totalLayerCount: UInt32
  let residentExpertCount: UInt32
  let residentExpertPayloadBytes: UInt64
  /// Absent for a sparse model and from older daemons.
  let denseLayers: DenseLayerResidency?

  enum CodingKeys: String, CodingKey {
    case totalLayerCount = "total_layer_count"
    case residentExpertCount = "resident_expert_count"
    case residentExpertPayloadBytes = "resident_expert_payload_bytes"
    case denseLayers = "dense_layers"
  }
}

/// A dense model's decoder layers: how many stay in RAM, and the bytes the
/// rest read from the SSD for every forward that runs them (issue #1985).
struct DenseLayerResidency: Codable, Equatable {
  let decoderLayerCount: UInt32
  let keptLayerCount: UInt32
  let streamedLayerPayloadBytes: UInt64

  enum CodingKeys: String, CodingKey {
    case decoderLayerCount = "decoder_layer_count"
    case keptLayerCount = "kept_layer_count"
    case streamedLayerPayloadBytes = "streamed_layer_payload_bytes"
  }

  /// Whether a forward reads any of the model's decoder layers from the SSD.
  var streamsFromTheSSD: Bool { streamedLayerPayloadBytes > 0 }
}
