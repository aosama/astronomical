import Foundation

/**
 * Family-neutral decode expert cache. The irreducible unit of ownership is
 * one expert id in one sparse-layer slot: bytes and pages are accounting
 * and I/O batching, never ownership. A repeated token route is therefore a
 * complete hit against resident experts, and eviction removes the coldest
 * experts — never a whole layer. Ceilings arrive from the composed memory
 * budget, and paging timings feed the switchable performance attribution
 * log on the serving critical path.
 *
 * - Parameters:
 *   - Weight: The opaque per-expert payload a model family materializes.
 */
public final class DecodeExpertCache<Weight: ResidentExpertWeight> {

    /// One resident set per sparse layer slot.
    private var layers: [ResidentExpertSet<Weight>]

    /// Route counts that rank eviction; demand never causes I/O.
    private var ledger: DecodeDemandLedger

    /// Payload-byte ceiling enforced after every admission batch.
    private var ceilingBytes: UInt64

    /// Total experts evicted since construction.
    private var evictionCount: UInt64

    /// Expert weights read from disk since construction.
    private var diskExpertLoadCount: UInt64

    /// Disk batches (grouped expert reads) since construction.
    private var diskBatchLoadCount: UInt64

    public init(sparseLayerCount: Int) {
        self.layers = (0..<sparseLayerCount).map({ (_: Int) -> ResidentExpertSet<Weight> in
            return ResidentExpertSet<Weight>()
        })
        self.ledger = DecodeDemandLedger(sparseLayerCount: sparseLayerCount)
        self.ceilingBytes = 0
        self.evictionCount = 0
        self.diskExpertLoadCount = 0
        self.diskBatchLoadCount = 0
    }

    /// Sparse layer slots this cache holds.
    public var sparseLayerCount: Int {
        return self.layers.count
    }

    /// Total experts evicted since construction.
    public var observedEvictionCount: UInt64 {
        return self.evictionCount
    }

    /// Expert weights read from disk since construction.
    public var observedDiskExpertLoadCount: UInt64 {
        return self.diskExpertLoadCount
    }

    /// Disk batches since construction.
    public var observedDiskBatchLoadCount: UInt64 {
        return self.diskBatchLoadCount
    }

    /// Current payload-byte ceiling.
    public var observedCeilingBytes: UInt64 {
        return self.ceilingBytes
    }

    /// Routed experts the slot must still load; an unknown slot misses
    /// everything, in route order.
    public func missing(slot: Int, routedIds: [Int]) -> [Int] {
        guard slot >= 0 && slot < self.layers.count else {
            return routedIds
        }
        return self.layers[slot].missing(routedIds: routedIds)
    }

    /// Whether every routed expert is already resident in the slot.
    public func containsEvery(slot: Int, routedIds: [Int]) -> Bool {
        guard slot >= 0 && slot < self.layers.count else {
            return false
        }
        return self.layers[slot].containsEvery(routedIds: routedIds)
    }

    /// Counts one route assignment per expert id; demand never causes I/O.
    public func recordDemand(slot: Int, routedIds: [Int]) {
        self.ledger.record(slot: slot, routedIds: routedIds)
    }

    /// Installs one resident expert; unknown slots are ignored.
    public func admit(slot: Int, expertId: Int, weight: Weight) {
        guard slot >= 0 && slot < self.layers.count else {
            return
        }
        self.layers[slot].admit(expertId: expertId, weight: weight)
    }

    /// Records disk I/O counts for telemetry attribution.
    public func recordDiskLoad(expertCount: Int, batchCount: Int) {
        self.diskExpertLoadCount =
            SaturatingArithmetic.add(self.diskExpertLoadCount, UInt64(expertCount))
        self.diskBatchLoadCount =
            SaturatingArithmetic.add(self.diskBatchLoadCount, UInt64(batchCount))
    }

    /// Saturating payload bytes owned across every slot.
    public func totalPayloadBytes() -> UInt64 {
        var payloadByteTotal: UInt64 = 0
        for layer: ResidentExpertSet<Weight> in self.layers {
            payloadByteTotal = SaturatingArithmetic.add(payloadByteTotal, layer.payloadBytes())
        }
        return payloadByteTotal
    }

    /// Total resident experts across every slot.
    public func residentExpertCount() -> Int {
        var expertTotal: Int = 0
        for layer: ResidentExpertSet<Weight> in self.layers {
            expertTotal += layer.expertCount()
        }
        return expertTotal
    }

    /// Resident expert ids of one slot in ascending order.
    public func residentIds(slot: Int) -> [Int] {
        guard slot >= 0 && slot < self.layers.count else {
            return []
        }
        return self.layers[slot].residentIds()
    }

    /// Publishes a new ceiling and immediately enforces it.
    public func setCeiling(_ ceilingBytes: UInt64) {
        self.ceilingBytes = ceilingBytes
        _ = self.enforceCeiling(protected: [])
    }

    /**
     * Evicts unprotected cold experts until the cache fits the ceiling.
     *
     * - Parameters:
     *   - protected: `(slot, expertId)` pairs the current forward just
     *     routed; they survive this pass.
     * - Returns: The sparse-layer slots that actually lost experts, so
     *   gather views for untouched layers can stay.
     */
    public func enforceCeiling(protected: [(Int, Int)]) -> [Int] {
        let totalPayloadBytes: UInt64 = self.totalPayloadBytes()
        if totalPayloadBytes <= self.ceilingBytes {
            return []
        }
        let bytesToFree: UInt64 =
            SaturatingArithmetic.subtract(totalPayloadBytes, self.ceilingBytes)
        let eviction: DecodeExpertEvictionOutcome = DecodeExpertEviction.evictColdestExperts(
            layers: self.layers,
            ledger: self.ledger,
            bytesToFree: bytesToFree,
            protected: protected)
        self.evictionCount =
            SaturatingArithmetic.add(self.evictionCount, UInt64(eviction.evictedExpertCount))
        return eviction.vacatedSlots
    }
}
