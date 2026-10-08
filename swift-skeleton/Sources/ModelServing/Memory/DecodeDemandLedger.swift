import Foundation

/// Per-slot, per-expert route counts that rank eviction and never cause I/O.
internal final class DecodeDemandLedger {

    /// Demand counts keyed by sparse slot, then expert id.
    private var countsBySlot: [[UInt64]]

    internal init(sparseLayerCount: Int) {
        self.countsBySlot = Array(repeating: [], count: sparseLayerCount)
    }

    /// Counts one route assignment per expert id; unknown slots are ignored.
    internal func record(slot: Int, routedIds: [Int]) {
        guard slot >= 0 && slot < self.countsBySlot.count else {
            return
        }
        var slotCounts: [UInt64] = self.countsBySlot[slot]
        for expertId: Int in routedIds where expertId >= 0 {
            if slotCounts.count <= expertId {
                slotCounts.append(contentsOf: Array(repeating: 0, count: expertId + 1 - slotCounts.count))
            }
            slotCounts[expertId] = SaturatingArithmetic.add(slotCounts[expertId], 1)
        }
        self.countsBySlot[slot] = slotCounts
    }

    /// Observed route count for one expert in one slot; zero when unseen.
    internal func demand(slot: Int, expertId: Int) -> UInt64 {
        guard slot >= 0 && slot < self.countsBySlot.count else {
            return 0
        }
        let slotCounts: [UInt64] = self.countsBySlot[slot]
        guard expertId >= 0 && expertId < slotCounts.count else {
            return 0
        }
        return slotCounts[expertId]
    }
}
