import Foundation

/// One sparse layer's resident experts; presence in the dictionary is
/// residency for that layer. Identifiers stay sorted on read so eviction
/// and telemetry observe a deterministic order.
internal final class ResidentExpertSet<Weight: ResidentExpertWeight> {

    /// Resident experts keyed by expert id.
    private var experts: [Int: Weight]

    internal init() {
        self.experts = [:]
    }

    /// Whether every routed expert is already resident here.
    internal func containsEvery(routedIds: [Int]) -> Bool {
        return routedIds.allSatisfy({ (expertId: Int) -> Bool in
            return self.experts[expertId] != nil
        })
    }

    /// Routed experts this layer must still load, in route order.
    internal func missing(routedIds: [Int]) -> [Int] {
        return routedIds.filter({ (expertId: Int) -> Bool in
            return self.experts[expertId] == nil
        })
    }

    /// Saturating payload bytes owned by this layer.
    internal func payloadBytes() -> UInt64 {
        var payloadByteTotal: UInt64 = 0
        for residentWeight: Weight in self.experts.values {
            payloadByteTotal = SaturatingArithmetic.add(payloadByteTotal, residentWeight.payloadBytes)
        }
        return payloadByteTotal
    }

    /// Number of resident experts.
    internal func expertCount() -> Int {
        return self.experts.count
    }

    /// Installs or replaces one resident expert.
    internal func admit(expertId: Int, weight: Weight) {
        self.experts[expertId] = weight
    }

    /// Removes one resident expert, returning its weight when present.
    internal func evict(expertId: Int) -> Weight? {
        return self.experts.removeValue(forKey: expertId)
    }

    /// Payload bytes of one resident expert; nil when absent.
    internal func payloadBytesFor(expertId: Int) -> UInt64? {
        return self.experts[expertId]?.payloadBytes
    }

    /// Resident expert ids in ascending order.
    internal func residentIds() -> [Int] {
        return self.experts.keys.sorted()
    }
}
