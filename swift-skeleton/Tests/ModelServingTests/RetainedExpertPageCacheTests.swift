import Foundation

import Testing

import ModelServing

/// Fictional page payload used only for byte accounting.
private struct FakeExpertPage: ExpertWeightPage, Equatable {
    var payloadBytes: UInt64

    func residentPayloadByteCount() -> UInt64 {
        return self.payloadBytes
    }
}

/// Hermetic journeys over retained expert page ownership, port of
/// crates/model-serving/tests/hermetic/retained_expert_page_cache.rs:
/// complete and partial pages coexist as explicit classes, a live ceiling
/// rejects commits while preserving the previous owner, complete mandatory
/// reads promote over partial owners, strict-superset routed pages replace
/// their owner while differing route sets are preserved, reclamation
/// releases partials before complete layers, request-pressure caps resume
/// without reloads, demand evidence ignores invalid identifiers and scales
/// by last-chunk token density, and zero or overflowing payload accounting
/// is rejected without mutating ownership.
@Suite
final class RetainedExpertPageCacheTests {

    private static func commitPartialPage(
        _ retainedPages: RetainedExpertPageCache<FakeExpertPage>,
        at layerIndex: Int,
        expertIds: [Int],
        payloadBytes: UInt64
    ) throws -> RetainedExpertLayerCommitOutcome {
        let commit: RetainedExpertLayerCommit<FakeExpertPage> =
            try retainedPages.commitMaterializedRoutedPage(
                at: layerIndex,
                expertCapacity: 4,
                expertIds: expertIds,
                expertPage: FakeExpertPage(payloadBytes: payloadBytes))
        return commit.outcome
    }

    private static func commitCompletePage(
        _ retainedPages: RetainedExpertPageCache<FakeExpertPage>,
        at layerIndex: Int,
        payloadBytes: UInt64
    ) throws -> RetainedExpertLayerCommitOutcome {
        let commit: RetainedExpertLayerCommit<FakeExpertPage> =
            try retainedPages.commitMaterializedCompleteLayer(
                at: layerIndex,
                expertCapacity: 4,
                expertPage: FakeExpertPage(payloadBytes: payloadBytes))
        return commit.outcome
    }

    @Test
    func should_commit_complete_and_partial_pages_with_explicit_classes() throws {
        let retainedPages: RetainedExpertPageCache<FakeExpertPage> = RetainedExpertPageCache(layerCount: 2)
        _ = retainedPages.updateMaximumResidentPayloadBytes(to: 60)

        if case .committed = try RetainedExpertPageCacheTests.commitCompletePage(
            retainedPages, at: 0, payloadBytes: 40) {} else {
            Issue.record("complete page commit should be accepted")
        }
        if case .committed = try RetainedExpertPageCacheTests.commitPartialPage(
            retainedPages, at: 1, expertIds: [1, 3], payloadBytes: 20) {} else {
            Issue.record("partial page commit should be accepted")
        }

        let statistics: RetainedExpertPageStatistics = retainedPages.statistics()
        #expect(statistics.entryCount == 2)
        #expect(statistics.residentPayloadByteCount == 60)
        #expect(statistics.completeLayerCount == 1)
        #expect(statistics.partialLayerCount == 1)
    }

    @Test
    func should_reject_a_commit_above_the_ceiling_while_preserving_the_previous_owner() throws {
        let retainedPages: RetainedExpertPageCache<FakeExpertPage> = RetainedExpertPageCache(layerCount: 1)
        _ = retainedPages.updateMaximumResidentPayloadBytes(to: 30)
        if case .committed = try RetainedExpertPageCacheTests.commitPartialPage(
            retainedPages, at: 0, expertIds: [0, 1], payloadBytes: 20) {} else {
            Issue.record("initial partial page commit should be accepted")
        }

        let rejectedOutcome: RetainedExpertLayerCommitOutcome =
            try RetainedExpertPageCacheTests.commitCompletePage(retainedPages, at: 0, payloadBytes: 40)
        #expect(rejectedOutcome == .rejectedByCurrentCeiling)
        #expect(retainedPages.statistics().residentPayloadByteCount == 20)
        #expect(retainedPages.statistics().partialLayerCount == 1)
    }

    @Test
    func should_replace_a_partial_owner_with_a_complete_mandatory_read() throws {
        let retainedPages: RetainedExpertPageCache<FakeExpertPage> = RetainedExpertPageCache(layerCount: 1)
        _ = retainedPages.updateMaximumResidentPayloadBytes(to: 40)
        _ = try RetainedExpertPageCacheTests.commitPartialPage(
            retainedPages, at: 0, expertIds: [0, 2], payloadBytes: 20)

        let promotedOutcome: RetainedExpertLayerCommitOutcome =
            try RetainedExpertPageCacheTests.commitCompletePage(retainedPages, at: 0, payloadBytes: 40)
        let expectedDelta: RetainedExpertLayerCommitDelta = RetainedExpertLayerCommitDelta(
            releasedPayloadBytes: 20,
            committedPayloadBytes: 40)
        #expect(promotedOutcome == .committed(expectedDelta))

        let statistics: RetainedExpertPageStatistics = retainedPages.statistics()
        #expect(statistics.entryCount == 1)
        #expect(statistics.completeLayerCount == 1)
        #expect(statistics.partialLayerCount == 0)
        #expect(statistics.mandatoryReadPromotionCount == 1)
    }

    @Test
    func should_preserve_a_useful_partial_owner_when_a_proposed_route_set_differs() throws {
        let retainedPages: RetainedExpertPageCache<FakeExpertPage> = RetainedExpertPageCache(layerCount: 1)
        _ = retainedPages.updateMaximumResidentPayloadBytes(to: 40)
        _ = try RetainedExpertPageCacheTests.commitPartialPage(
            retainedPages, at: 0, expertIds: [0, 1], payloadBytes: 20)

        let preservedOutcome: RetainedExpertLayerCommitOutcome =
            try RetainedExpertPageCacheTests.commitPartialPage(
                retainedPages, at: 0, expertIds: [2, 3], payloadBytes: 20)
        #expect(preservedOutcome == .preservedExisting)
        #expect(retainedPages.topologySnapshot(expertCapacity: 4)[0].retainedExpertIds == [0, 1])
    }

    @Test
    func should_replace_a_partial_owner_with_a_strict_superset_routed_page() throws {
        let retainedPages: RetainedExpertPageCache<FakeExpertPage> = RetainedExpertPageCache(layerCount: 1)
        _ = retainedPages.updateMaximumResidentPayloadBytes(to: 40)
        _ = try RetainedExpertPageCacheTests.commitPartialPage(
            retainedPages, at: 0, expertIds: [0, 1], payloadBytes: 20)

        if case .committed = try RetainedExpertPageCacheTests.commitPartialPage(
            retainedPages, at: 0, expertIds: [0, 1, 2], payloadBytes: 30) {} else {
            Issue.record("strict superset routed page should replace its owner")
        }
        #expect(retainedPages.topologySnapshot(expertCapacity: 4)[0].retainedExpertIds == [0, 1, 2])
        #expect(retainedPages.statistics().residentPayloadByteCount == 30)
    }

    @Test
    func should_reclaim_multiple_partial_pages_before_one_complete_page() throws {
        let retainedPages: RetainedExpertPageCache<FakeExpertPage> = RetainedExpertPageCache(layerCount: 3)
        _ = retainedPages.updateMaximumResidentPayloadBytes(to: 80)
        _ = try RetainedExpertPageCacheTests.commitCompletePage(retainedPages, at: 0, payloadBytes: 40)
        _ = try RetainedExpertPageCacheTests.commitPartialPage(
            retainedPages, at: 1, expertIds: [0, 1], payloadBytes: 20)
        _ = try RetainedExpertPageCacheTests.commitPartialPage(
            retainedPages, at: 2, expertIds: [2, 3], payloadBytes: 20)

        let reclamation: RetainedExpertReclamation =
            retainedPages.reclaimForRequestPressure(requiredPayloadBytes: 30)
        let statistics: RetainedExpertPageStatistics = retainedPages.statistics()
        #expect(reclamation.releasedPartialLayerCount == 2)
        #expect(reclamation.releasedPartialPayloadBytes == 40)
        #expect(reclamation.releasedCompleteLayerCount == 0)
        #expect(statistics.entryCount == 1)
        #expect(statistics.residentPayloadByteCount == 40)
        #expect(retainedPages.retainedLayer(at: 0) != nil)
    }

    @Test
    func should_resume_a_request_pressure_cap_without_loading_any_page() throws {
        let retainedPages: RetainedExpertPageCache<FakeExpertPage> = RetainedExpertPageCache(layerCount: 2)
        _ = retainedPages.updateMaximumResidentPayloadBytes(to: 80)
        _ = try RetainedExpertPageCacheTests.commitCompletePage(retainedPages, at: 0, payloadBytes: 40)
        _ = try RetainedExpertPageCacheTests.commitCompletePage(retainedPages, at: 1, payloadBytes: 40)

        #expect(retainedPages.limitForRequestPressure(reclamationTargetBytes: 40))
        #expect(retainedPages.resumeAfterRequestPressure())
        let statistics: RetainedExpertPageStatistics = retainedPages.statistics()
        #expect(statistics.maximumResidentPayloadByteCount == 80)
        #expect(statistics.diskPageLoadCount == 0)
    }

    @Test
    func should_apply_an_absolute_forward_pressure_cap_and_allow_safe_growth_to_it() throws {
        let retainedPages: RetainedExpertPageCache<FakeExpertPage> = RetainedExpertPageCache(layerCount: 3)
        _ = retainedPages.updateMaximumResidentPayloadBytes(to: 120)
        _ = try RetainedExpertPageCacheTests.commitCompletePage(retainedPages, at: 0, payloadBytes: 40)
        _ = try RetainedExpertPageCacheTests.commitCompletePage(retainedPages, at: 1, payloadBytes: 40)

        #expect(retainedPages.limitForRequestPressureToMaximum(pressureMaximumResidentPayloadBytes: 50))
        #expect(retainedPages.statistics().residentPayloadByteCount == 40)
        let growthOutcome: RetainedExpertLayerCommitOutcome =
            try RetainedExpertPageCacheTests.commitPartialPage(
                retainedPages, at: 2, expertIds: [0], payloadBytes: 10)
        let expectedDelta: RetainedExpertLayerCommitDelta = RetainedExpertLayerCommitDelta(
            releasedPayloadBytes: 0,
            committedPayloadBytes: 10)
        #expect(growthOutcome == .committed(expectedDelta))
        #expect(retainedPages.statistics().residentPayloadByteCount == 50)
        #expect(retainedPages.canCommitMaterializedPage(at: 2, candidatePayloadBytes: 20) == false)
    }

    @Test
    func should_apply_a_lower_normal_ceiling_with_partial_first_reclamation() throws {
        let retainedPages: RetainedExpertPageCache<FakeExpertPage> = RetainedExpertPageCache(layerCount: 3)
        _ = retainedPages.updateMaximumResidentPayloadBytes(to: 80)
        _ = try RetainedExpertPageCacheTests.commitCompletePage(retainedPages, at: 0, payloadBytes: 40)
        _ = try RetainedExpertPageCacheTests.commitPartialPage(
            retainedPages, at: 1, expertIds: [0, 1], payloadBytes: 20)
        _ = try RetainedExpertPageCacheTests.commitPartialPage(
            retainedPages, at: 2, expertIds: [2, 3], payloadBytes: 20)

        let reclamation: RetainedExpertReclamation =
            retainedPages.updateMaximumResidentPayloadBytes(to: 50)

        let statistics: RetainedExpertPageStatistics = retainedPages.statistics()
        #expect(reclamation.releasedPartialLayerCount == 2)
        #expect(reclamation.releasedPartialPayloadBytes == 40)
        #expect(reclamation.releasedCompleteLayerCount == 0)
        #expect(statistics.completeLayerCount == 1)
        #expect(statistics.partialLayerCount == 0)
        #expect(statistics.partialLayerEvictionCount == 2)
    }

    @Test
    func should_ignore_invalid_route_identifiers_without_corrupting_demand() throws {
        let retainedPages: RetainedExpertPageCache<FakeExpertPage> = RetainedExpertPageCache(layerCount: 1)
        _ = retainedPages.updateMaximumResidentPayloadBytes(to: 10)
        retainedPages.recordExpertDemand(layerIndex: 0, expertCapacity: 2, selectedExpertIds: [1, 9, 1])
        _ = try RetainedExpertPageCacheTests.commitPartialPage(
            retainedPages, at: 0, expertIds: [1], payloadBytes: 10)

        #expect(retainedPages.topologySnapshot(expertCapacity: 2)[0].coveredWeightedDemand == 2)
    }

    @Test
    func should_start_a_fresh_demand_window_after_topology_planning() throws {
        let retainedPages: RetainedExpertPageCache<FakeExpertPage> = RetainedExpertPageCache(layerCount: 1)
        _ = retainedPages.updateMaximumResidentPayloadBytes(to: 20)
        retainedPages.recordExpertDemand(layerIndex: 0, expertCapacity: 2, selectedExpertIds: [1, 1, 1])
        retainedPages.clearExpertDemand()
        retainedPages.recordExpertDemand(layerIndex: 0, expertCapacity: 2, selectedExpertIds: [0])
        _ = try RetainedExpertPageCacheTests.commitPartialPage(
            retainedPages, at: 0, expertIds: [0, 1], payloadBytes: 20)

        #expect(retainedPages.topologySnapshot(expertCapacity: 2)[0].coveredWeightedDemand == 1)
    }

    @Test
    func should_scale_last_prefill_chunk_demand_by_earlier_token_density() {
        #expect(ExpertDemandWeighting.lastPrefillChunkDemandWeight(
            earlierPrefillTokenCount: 6,
            lastPrefillChunkTokenCount: 3) == 2)
        #expect(ExpertDemandWeighting.lastPrefillChunkDemandWeight(
            earlierPrefillTokenCount: 5,
            lastPrefillChunkTokenCount: 3) == 1)
        #expect(ExpertDemandWeighting.lastPrefillChunkDemandWeight(
            earlierPrefillTokenCount: 0,
            lastPrefillChunkTokenCount: 8) == 1)
        #expect(ExpertDemandWeighting.lastPrefillChunkDemandWeight(
            earlierPrefillTokenCount: 8,
            lastPrefillChunkTokenCount: 0) == 1)
        #expect(ExpertDemandWeighting.lastPrefillChunkDemandWeight(
            earlierPrefillTokenCount: UInt64.max,
            lastPrefillChunkTokenCount: 1) == UInt64.max)
    }

    @Test
    func should_prefer_last_chunk_routes_when_weighted_demand_exceeds_earlier_frequency() throws {
        let retainedPages: RetainedExpertPageCache<FakeExpertPage> = RetainedExpertPageCache(layerCount: 2)
        _ = retainedPages.updateMaximumResidentPayloadBytes(to: 20)
        retainedPages.setDemandAssignmentWeight(1)
        retainedPages.recordExpertDemand(layerIndex: 0, expertCapacity: 2, selectedExpertIds: [0, 0, 0, 0, 0])
        retainedPages.setDemandAssignmentWeight(2)
        retainedPages.recordExpertDemand(layerIndex: 1, expertCapacity: 2, selectedExpertIds: [1, 1, 1])
        _ = try RetainedExpertPageCacheTests.commitPartialPage(
            retainedPages, at: 0, expertIds: [0], payloadBytes: 10)
        _ = try RetainedExpertPageCacheTests.commitPartialPage(
            retainedPages, at: 1, expertIds: [1], payloadBytes: 10)

        // Five earlier routes lose to three last-chunk routes counted twice.
        let topology: [CurrentExpertLayerResidency] = retainedPages.topologySnapshot(expertCapacity: 2)
        #expect(topology[0].coveredWeightedDemand == 5)
        #expect(topology[1].coveredWeightedDemand == 6)
    }

    @Test
    func should_keep_raw_frequency_ranking_when_last_chunk_weight_is_one() throws {
        let retainedPages: RetainedExpertPageCache<FakeExpertPage> = RetainedExpertPageCache(layerCount: 2)
        _ = retainedPages.updateMaximumResidentPayloadBytes(to: 20)
        retainedPages.setDemandAssignmentWeight(1)
        retainedPages.recordExpertDemand(layerIndex: 0, expertCapacity: 2, selectedExpertIds: [0, 0, 0, 0, 0])
        retainedPages.recordExpertDemand(layerIndex: 1, expertCapacity: 2, selectedExpertIds: [1, 1, 1])
        _ = try RetainedExpertPageCacheTests.commitPartialPage(
            retainedPages, at: 0, expertIds: [0], payloadBytes: 10)
        _ = try RetainedExpertPageCacheTests.commitPartialPage(
            retainedPages, at: 1, expertIds: [1], payloadBytes: 10)

        let topology: [CurrentExpertLayerResidency] = retainedPages.topologySnapshot(expertCapacity: 2)
        #expect(topology[0].coveredWeightedDemand == 5)
        #expect(topology[1].coveredWeightedDemand == 3)
    }

    @Test
    func should_treat_a_zero_demand_weight_as_one_assignment() throws {
        let retainedPages: RetainedExpertPageCache<FakeExpertPage> = RetainedExpertPageCache(layerCount: 2)
        _ = retainedPages.updateMaximumResidentPayloadBytes(to: 20)
        retainedPages.setDemandAssignmentWeight(0)
        retainedPages.recordExpertDemand(layerIndex: 0, expertCapacity: 2, selectedExpertIds: [0])
        retainedPages.recordExpertDemand(layerIndex: 1, expertCapacity: 2, selectedExpertIds: [1, 1])
        _ = try RetainedExpertPageCacheTests.commitPartialPage(
            retainedPages, at: 0, expertIds: [0], payloadBytes: 10)
        _ = try RetainedExpertPageCacheTests.commitPartialPage(
            retainedPages, at: 1, expertIds: [1], payloadBytes: 10)

        // A zero weight must not discard assignments; two routes still beat one.
        let topology: [CurrentExpertLayerResidency] = retainedPages.topologySnapshot(expertCapacity: 2)
        #expect(topology[0].coveredWeightedDemand == 1)
        #expect(topology[1].coveredWeightedDemand == 2)
    }

    @Test
    func should_reject_zero_or_overflowing_payload_accounting_without_mutating_ownership() throws {
        let retainedPages: RetainedExpertPageCache<FakeExpertPage> = RetainedExpertPageCache(layerCount: 3)
        _ = retainedPages.updateMaximumResidentPayloadBytes(to: UInt64.max)

        #expect(retainedPages.canCommitMaterializedPage(at: 0, candidatePayloadBytes: 0) == false)
        #expect(throws: RetainedExpertLayerCommitError.zeroPayload(layerIndex: 0)) {
            _ = try retainedPages.commitMaterializedCompleteLayer(
                at: 0,
                expertCapacity: 4,
                expertPage: FakeExpertPage(payloadBytes: 0))
        }
        _ = try RetainedExpertPageCacheTests.commitCompletePage(
            retainedPages, at: 1, payloadBytes: UInt64.max)
        #expect(retainedPages.canCommitMaterializedPage(at: 2, candidatePayloadBytes: 1) == false)
        #expect(throws: RetainedExpertLayerCommitError.payloadByteCountOverflow(layerIndex: 2)) {
            _ = try retainedPages.commitMaterializedCompleteLayer(
                at: 2,
                expertCapacity: 4,
                expertPage: FakeExpertPage(payloadBytes: 1))
        }
        #expect(retainedPages.statistics().residentPayloadByteCount == UInt64.max)
        #expect(retainedPages.retainedLayer(at: 2) == nil)
    }
}
