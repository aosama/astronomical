import Foundation

/// Sparse expert selection: validation and permutation math the paging
/// decorator owns around the upstream mlx-swift-lm gathered projections
/// (`gatherSort`/`scatterUnsort` and `SwitchGLU` carry the reduction
/// kernels themselves, so no expert reduction math is reimplemented
/// here). The contracts this module must keep feeding are the expert
/// paging attribution timings (a first-class critical path) and
/// config-driven structural checks for expert counts and group sizes.
public enum SparseExperts {

    /**
     * Inverts a complete assignment permutation: given the argsort order
     * that maps sorted slots back to their original positions, returns
     * the order that restores original positions from sorted slots.
     *
     * - Parameter sortedOrder: The permutation to invert, where
     *   `sortedOrder[sortedSlot]` names the original slot.
     * - Returns: The inverse permutation.
     * - Throws: `SparseExpertsError.invalidAssignmentGeometry` when the
     *   input holds a duplicate or out-of-range slot.
     */
    public static func invertAssignmentOrder(sortedOrder: [UInt32]) throws -> [UInt32] {
        let slotCount: Int = sortedOrder.count
        var inverseOrder: [UInt32] = Array(repeating: 0, count: slotCount)
        var seenSlots: Set<Int> = Set()
        for (sortedSlot, originalSlot): (Int, UInt32) in sortedOrder.enumerated() {
            let originalSlotIndex: Int = Int(originalSlot)
            if originalSlotIndex < 0 || originalSlotIndex >= slotCount {
                throw SparseExpertsError.invalidAssignmentGeometry(
                    description: "assignment permutation contains an out-of-range slot")
            }
            if seenSlots.insert(originalSlotIndex).inserted == false {
                throw SparseExpertsError.invalidAssignmentGeometry(
                    description: "assignment permutation contains a duplicate slot")
            }
            inverseOrder[originalSlotIndex] = UInt32(sortedSlot)
        }
        return inverseOrder
    }
}
