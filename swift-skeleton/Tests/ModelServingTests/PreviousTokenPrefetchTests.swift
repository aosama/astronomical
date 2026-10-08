import Foundation

import Testing

import ModelServing

/// Behavior coverage for previous-token prefetch admission: leftover
/// slots only, never eviction, port of
/// crates/model-serving/tests/hermetic/previous_token_prefetch.rs.
@Suite
final class PreviousTokenPrefetchTests {

    private static func candidate(
        layerIndex: Int,
        expertId: Int,
        isAlreadyResident: Bool
    ) -> PreviousTokenPrefetchCandidate {
        return PreviousTokenPrefetchCandidate(
            layerIndex: layerIndex,
            expertId: expertId,
            payloadBytes: 8,
            isAlreadyResident: isAlreadyResident)
    }

    private static func layerCapacity(
        layerIndex: Int,
        freeSlotCount: Int
    ) -> PreviousTokenPrefetchLayerCapacity {
        return PreviousTokenPrefetchLayerCapacity(
            layerIndex: layerIndex,
            freeSlotCount: freeSlotCount)
    }

    @Test
    func shouldDropEveryMissWhenTheLayerHasNoFreeSlot() {
        let plan = PreviousTokenPrefetchPlanner.planPreviousTokenPrefetch(
            candidates: [
                Self.candidate(layerIndex: 0, expertId: 1, isAlreadyResident: false),
                Self.candidate(layerIndex: 0, expertId: 2, isAlreadyResident: false),
            ],
            layerCapacities: [Self.layerCapacity(layerIndex: 0, freeSlotCount: 0)])

        #expect(
            plan.expertsToRetain.isEmpty,
            "a full layer must not displace a demanded page to keep a previous-token expert")
        #expect(plan.droppedForCapacityCount == 2)
        #expect(plan.skippedAlreadyResidentCount == 0)
    }

    @Test
    func shouldFillFreeSlotsThenDropTheOverflowInsteadOfEvicting() {
        let plan = PreviousTokenPrefetchPlanner.planPreviousTokenPrefetch(
            candidates: [
                Self.candidate(layerIndex: 0, expertId: 1, isAlreadyResident: false),
                Self.candidate(layerIndex: 0, expertId: 2, isAlreadyResident: false),
                Self.candidate(layerIndex: 0, expertId: 3, isAlreadyResident: false),
            ],
            layerCapacities: [Self.layerCapacity(layerIndex: 0, freeSlotCount: 2)])

        #expect(
            plan.expertsToRetain
                == [PreviousTokenPrefetchRetention(layerIndex: 0, expertId: 1),
                    PreviousTokenPrefetchRetention(layerIndex: 0, expertId: 2)])
        #expect(plan.droppedForCapacityCount == 1)
    }

    @Test
    func shouldSkipExpertsTheWarmTableAlreadyHolds() {
        let plan = PreviousTokenPrefetchPlanner.planPreviousTokenPrefetch(
            candidates: [
                Self.candidate(layerIndex: 1, expertId: 4, isAlreadyResident: true),
                Self.candidate(layerIndex: 1, expertId: 5, isAlreadyResident: false),
            ],
            layerCapacities: [Self.layerCapacity(layerIndex: 1, freeSlotCount: 1)])

        #expect(
            plan.expertsToRetain == [PreviousTokenPrefetchRetention(layerIndex: 1, expertId: 5)])
        #expect(plan.skippedAlreadyResidentCount == 1)
        #expect(plan.droppedForCapacityCount == 0)
    }

    @Test
    func shouldNotBorrowFreeSlotsFromAnotherLayer() {
        let plan = PreviousTokenPrefetchPlanner.planPreviousTokenPrefetch(
            candidates: [
                Self.candidate(layerIndex: 0, expertId: 1, isAlreadyResident: false),
                Self.candidate(layerIndex: 1, expertId: 2, isAlreadyResident: false),
            ],
            layerCapacities: [
                Self.layerCapacity(layerIndex: 0, freeSlotCount: 0),
                Self.layerCapacity(layerIndex: 1, freeSlotCount: 1),
            ])

        #expect(
            plan.expertsToRetain == [PreviousTokenPrefetchRetention(layerIndex: 1, expertId: 2)])
        #expect(plan.droppedForCapacityCount == 1)
    }

    @Test
    func shouldDropAZeroPayloadExpertRatherThanPlanningAnEmptyRetain() {
        let plan = PreviousTokenPrefetchPlanner.planPreviousTokenPrefetch(
            candidates: [
                PreviousTokenPrefetchCandidate(
                    layerIndex: 0,
                    expertId: 9,
                    payloadBytes: 0,
                    isAlreadyResident: false),
            ],
            layerCapacities: [Self.layerCapacity(layerIndex: 0, freeSlotCount: 4)])

        #expect(plan.expertsToRetain.isEmpty)
        #expect(plan.droppedForCapacityCount == 1)
    }
}
