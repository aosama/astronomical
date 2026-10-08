import Foundation

/// Chooses previous-token experts that fit leftover slots without
/// eviction, port of the Rust `plan_previous_token_prefetch`.
///
/// After a decode token streams its routed experts, leftover expert RAM may
/// keep that exact set for the next token. Candidates are considered in
/// caller order. Already-resident experts are skipped, and a zero-payload
/// expert is dropped rather than planned as an empty retain. A layer with
/// no free slot drops every remaining miss for that layer rather than
/// displacing a demanded page; this owner never names an eviction victim.
public enum PreviousTokenPrefetchPlanner {

    public static func planPreviousTokenPrefetch(
        candidates: [PreviousTokenPrefetchCandidate],
        layerCapacities: [PreviousTokenPrefetchLayerCapacity]
    ) -> PreviousTokenPrefetchPlan {
        var remainingFreeSlotsByLayer: [(layerIndex: Int, remainingFreeSlots: Int)] =
            layerCapacities.map { ($0.layerIndex, $0.freeSlotCount) }
        var expertsToRetain: [PreviousTokenPrefetchRetention] = []
        var skippedAlreadyResidentCount: UInt64 = 0
        var droppedForCapacityCount: UInt64 = 0
        for candidate in candidates {
            if candidate.isAlreadyResident {
                skippedAlreadyResidentCount += 1
                continue
            }
            if candidate.payloadBytes == 0 {
                droppedForCapacityCount += 1
                continue
            }
            guard let slotIndex = remainingFreeSlotsByLayer.firstIndex(where: {
                $0.layerIndex == candidate.layerIndex
            }) else {
                droppedForCapacityCount += 1
                continue
            }
            if remainingFreeSlotsByLayer[slotIndex].remainingFreeSlots == 0 {
                droppedForCapacityCount += 1
                continue
            }
            remainingFreeSlotsByLayer[slotIndex].remainingFreeSlots -= 1
            expertsToRetain.append(
                PreviousTokenPrefetchRetention(
                    layerIndex: candidate.layerIndex,
                    expertId: candidate.expertId))
        }
        return PreviousTokenPrefetchPlan(
            expertsToRetain: expertsToRetain,
            skippedAlreadyResidentCount: skippedAlreadyResidentCount,
            droppedForCapacityCount: droppedForCapacityCount)
    }
}
