import Foundation

/// Decode-handoff seating after an atomic complete-owner demote.
///
/// The planner may name complete layers that leftover RAM can hold. Decode
/// never takes the complete-layer stream path, so those indexes are not
/// loaded by later generated tokens. This decision is the only answer to
/// "which complete layers must be seated before the first decode token."
/// Family code enacts it; it must not invent a second policy of skipping
/// the load.
public enum DecodeSeating {

    /**
     * Names the complete-layer promotion targets that are not already retained.
     *
     * An empty result means decode may proceed without a seating pass.
     *
     * - Parameter plan: The phase's expert residency plan.
     * - Returns: Layer indexes requiring a seating read before decode.
     */
    public static func completeLayerIndexesRequiredBeforeDecode(
        plan: ExpertResidencyPlan
    ) -> [Int] {
        var requiredLayerIndexes: [Int] = []
        requiredLayerIndexes.reserveCapacity(plan.completeLayerTargets.count)
        for layerIndex: Int in plan.completeLayerTargets {
            guard layerIndex < plan.layerTargets.count else {
                continue
            }
            if plan.layerTargets[layerIndex] == .promoteCompleteOnMandatoryRead {
                requiredLayerIndexes.append(layerIndex)
            }
        }
        return requiredLayerIndexes
    }
}
