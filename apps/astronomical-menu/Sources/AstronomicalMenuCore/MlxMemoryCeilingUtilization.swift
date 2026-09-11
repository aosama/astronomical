import Foundation

/// Engine-published split of unused MLX headroom (issue #509).
///
/// The worker composes this from the same RAM-budget owner that admits
/// forwards, so the menu must not invent a second arithmetic. Missing on
/// older snapshots; the popover then keeps a single undifferentiated
/// headroom row.
struct MlxMemoryCeilingUtilization: Codable, Equatable {
  let unusedHeadroomBytes: UInt64
  let reservedModelCoreSlackBytes: UInt64
  let reservedContextGrowthBytes: UInt64
  let reservedActivationAndWorkspaceBytes: UInt64
  let unseatedExpertEntitlementBytes: UInt64
  let speculativeDraftPayloadBytes: UInt64
  let unexplainedHeadroomBytes: UInt64
  let ownerOverrunBytes: UInt64

  enum CodingKeys: String, CodingKey {
    case unusedHeadroomBytes = "unused_headroom_bytes"
    case reservedModelCoreSlackBytes = "reserved_model_core_slack_bytes"
    case reservedContextGrowthBytes = "reserved_context_growth_bytes"
    case reservedActivationAndWorkspaceBytes = "reserved_activation_and_workspace_bytes"
    case unseatedExpertEntitlementBytes = "unseated_expert_entitlement_bytes"
    case speculativeDraftPayloadBytes = "speculative_draft_payload_bytes"
    case unexplainedHeadroomBytes = "unexplained_headroom_bytes"
    case ownerOverrunBytes = "owner_overrun_bytes"
  }
}

/// Headroom owners scaled onto the popover's remaining-capacity width.
///
/// Occupied MLX bytes stay in the existing ownership bar. This split only
/// paints the unused remainder, so the headline active/ceiling number is
/// unchanged. Integer scaling keeps the colored segments summing to
/// `availableByteCount` instead of drifting from the occupancy bar.
struct MlxHeadroomSplit: Equatable {
  let unusedBudgetByteCount: UInt64
  let reservedContextGrowthByteCount: UInt64
  let reservedActivationByteCount: UInt64
  let reservedModelCoreSlackByteCount: UInt64
  let unexplainedHeadroomByteCount: UInt64
  let remainderByteCount: UInt64
  let ownerOverrunByteCount: UInt64
  let enginePublishedTheSplit: Bool

  static func undifferentiated(availableByteCount: UInt64) -> MlxHeadroomSplit {
    MlxHeadroomSplit(
      unusedBudgetByteCount: 0,
      reservedContextGrowthByteCount: 0,
      reservedActivationByteCount: 0,
      reservedModelCoreSlackByteCount: 0,
      unexplainedHeadroomByteCount: 0,
      remainderByteCount: availableByteCount,
      ownerOverrunByteCount: 0,
      enginePublishedTheSplit: false
    )
  }

  static func from(
    utilization: MlxMemoryCeilingUtilization?,
    availableByteCount: UInt64
  ) -> MlxHeadroomSplit {
    guard let utilization else {
      return .undifferentiated(availableByteCount: availableByteCount)
    }
    let weights = [
      utilization.unseatedExpertEntitlementBytes,
      utilization.reservedContextGrowthBytes,
      utilization.reservedActivationAndWorkspaceBytes,
      utilization.reservedModelCoreSlackBytes,
      utilization.unexplainedHeadroomBytes,
    ]
    let allocated = allocateProportions(weights, onto: availableByteCount)
    let allocatedSum = allocated.reduce(0, +)
    return MlxHeadroomSplit(
      unusedBudgetByteCount: allocated[0],
      reservedContextGrowthByteCount: allocated[1],
      reservedActivationByteCount: allocated[2],
      reservedModelCoreSlackByteCount: allocated[3],
      unexplainedHeadroomByteCount: allocated[4],
      remainderByteCount: availableByteCount.saturatingSubtracting(allocatedSum),
      ownerOverrunByteCount: utilization.ownerOverrunBytes,
      enginePublishedTheSplit: true
    )
  }

  var paintedByteCount: UInt64 {
    unusedBudgetByteCount
      + reservedContextGrowthByteCount
      + reservedActivationByteCount
      + reservedModelCoreSlackByteCount
      + unexplainedHeadroomByteCount
      + remainderByteCount
  }
}

/// Distributes `total` across `weights` with integer proportions.
///
/// Remainder bytes go to the last positive weight so the painted bar never
/// undershoots the unused-capacity width by rounding.
func allocateProportions(_ weights: [UInt64], onto total: UInt64) -> [UInt64] {
  let weightSum = weights.reduce(0, +)
  guard weightSum > 0, total > 0 else {
    return Array(repeating: 0, count: weights.count)
  }
  var allocated = weights.map { weight in
    UInt64((Double(weight) / Double(weightSum)) * Double(total))
  }
  let allocatedSum = allocated.reduce(0, +)
  let roundingRemainder = total.saturatingSubtracting(allocatedSum)
  if roundingRemainder > 0, let lastPositiveIndex = allocated.lastIndex(where: { $0 > 0 }) {
    allocated[lastPositiveIndex] += roundingRemainder
  } else if roundingRemainder > 0, !allocated.isEmpty {
    allocated[allocated.count - 1] += roundingRemainder
  }
  return allocated
}
