/// Concrete sparse-expert ownership reported by the active worker.
///
/// The execution mode describes which owner serves experts, while these counts
/// describe whether that owner currently has every expert layer in memory.
struct ExpertResidencySnapshot: Codable, Equatable {
  let totalLayerCount: UInt32
  let completeLayerCount: UInt32
  let completeLayerPayloadBytes: UInt64
  let partialLayerCount: UInt32
  let partialLayerPayloadBytes: UInt64

  enum CodingKeys: String, CodingKey {
    case totalLayerCount = "total_layer_count"
    case completeLayerCount = "complete_layer_count"
    case completeLayerPayloadBytes = "complete_layer_payload_bytes"
    case partialLayerCount = "partial_layer_count"
    case partialLayerPayloadBytes = "partial_layer_payload_bytes"
  }

  /// True when no sparse layer needs a source read for its missing experts.
  var retainsEveryLayerCompletely: Bool {
    totalLayerCount > 0
      && completeLayerCount == totalLayerCount
      && partialLayerCount == 0
  }
}
