import Foundation

import Testing

import ModelServing

/// Fictional per-expert payload used only for byte accounting.
private struct FakeExpertWeight: ResidentExpertWeight, Equatable {
    var payloadBytes: UInt64
}

/// Hermetic journeys over the decode expert cache, port of
/// crates/model-serving/tests/hermetic/decode_expert_cache.rs: the
/// irreducible unit is one expert, a repeated token route is a complete
/// hit against resident experts, tight ceilings evict the coldest experts
/// instead of dropping a whole layer, and ceiling enforcement protects the
/// current decode route.
@Suite
final class DecodeExpertCacheTests {

    private static func admitExperts(
        _ decodeCache: DecodeExpertCache<FakeExpertWeight>,
        slot: Int,
        expertIds: [Int]
    ) -> Void {
        for expertId: Int in expertIds {
            decodeCache.admit(slot: slot, expertId: expertId, weight: FakeExpertWeight(payloadBytes: 8))
        }
    }

    @Test
    func should_report_no_missing_experts_when_the_route_is_already_resident() {
        let decodeCache: DecodeExpertCache<FakeExpertWeight> = DecodeExpertCache(sparseLayerCount: 4)
        DecodeExpertCacheTests.admitExperts(decodeCache, slot: 2, expertIds: [2, 5, 7])
        #expect(decodeCache.containsEvery(slot: 2, routedIds: [2, 5, 7]))
        let missingExpertIds: [Int] = decodeCache.missing(slot: 2, routedIds: [2, 5])
        #expect(missingExpertIds.isEmpty)
    }

    @Test
    func should_evict_the_coldest_experts_instead_of_dropping_the_whole_layer() {
        let decodeCache: DecodeExpertCache<FakeExpertWeight> = DecodeExpertCache(sparseLayerCount: 1)
        DecodeExpertCacheTests.admitExperts(decodeCache, slot: 0, expertIds: [0, 1, 2, 3, 4, 5])
        decodeCache.recordDemand(slot: 0, routedIds: [0, 1, 2, 3, 4, 5])
        decodeCache.recordDemand(slot: 0, routedIds: [0, 1, 2, 3])
        decodeCache.setCeiling(40)
        let remainingExpertIds: [Int] = decodeCache.residentIds(slot: 0)
        #expect(remainingExpertIds.count < 6)
        #expect(remainingExpertIds.isEmpty == false)
        #expect(remainingExpertIds.contains(0) && remainingExpertIds.contains(1))
    }

    @Test
    func should_keep_the_current_route_when_enforcing_the_decode_ceiling() {
        let decodeCache: DecodeExpertCache<FakeExpertWeight> = DecodeExpertCache(sparseLayerCount: 1)
        DecodeExpertCacheTests.admitExperts(decodeCache, slot: 0, expertIds: [0, 1, 2, 3])
        decodeCache.recordDemand(slot: 0, routedIds: [0, 1])
        let vacatedSlots: [Int] = decodeCache.enforceCeiling(protected: [(0, 2), (0, 3)])
        let remainingExpertIds: [Int] = decodeCache.residentIds(slot: 0)
        #expect(vacatedSlots == [0])
        #expect(remainingExpertIds.contains(2) && remainingExpertIds.contains(3))
        #expect(remainingExpertIds.contains(0) == false && remainingExpertIds.contains(1) == false)
    }
}
