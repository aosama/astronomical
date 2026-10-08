import Foundation

/// One internally consistent observation at an allocation boundary.
///
/// Decides active-memory fit independently from allocator-cache ownership.
/// Callers remain responsible for synchronizing streams, clearing allocator
/// storage, and performing allocations.
public struct AllocationAdmissionObservation: Equatable, Hashable, Sendable {

    /// Live MLX active bytes before the pending allocation.
    public let activeMemoryBytes: UInt64

    /// Reclaimable allocator storage, excluded from active-limit enforcement.
    public let allocatorCacheBytes: UInt64

    /// Exact byte count of the allocation about to be created.
    public let pendingAllocationBytes: UInt64

    /// Worker-resolved stable MLX active-memory ceiling.
    public let activeMemoryCeilingBytes: UInt64

    public init(
        activeMemoryBytes: UInt64,
        allocatorCacheBytes: UInt64,
        pendingAllocationBytes: UInt64,
        activeMemoryCeilingBytes: UInt64
    ) {
        self.activeMemoryBytes = activeMemoryBytes
        self.allocatorCacheBytes = allocatorCacheBytes
        self.pendingAllocationBytes = pendingAllocationBytes
        self.activeMemoryCeilingBytes = activeMemoryCeilingBytes
    }

    /// Decides active-memory fit independently from allocator-cache ownership.
    public func decide() -> AllocationAdmissionDecision {
        let (projectedActiveMemoryBytes, projectionOverflowed) =
            activeMemoryBytes.addingReportingOverflow(pendingAllocationBytes)
        if projectionOverflowed {
            return .reject(boundary: .allocationProjection, shortfallBytes: UInt64.max)
        }
        if projectedActiveMemoryBytes > activeMemoryCeilingBytes {
            return .reject(
                boundary: .allocationProjection,
                shortfallBytes: projectedActiveMemoryBytes - activeMemoryCeilingBytes)
        }
        let (totalMemoryAfterAllocationBytes, totalOverflowed) =
            projectedActiveMemoryBytes.addingReportingOverflow(allocatorCacheBytes)
        let totalExceedsCeiling: Bool
        if totalOverflowed {
            totalExceedsCeiling = true
        } else {
            totalExceedsCeiling = totalMemoryAfterAllocationBytes > activeMemoryCeilingBytes
        }
        if allocatorCacheBytes > 0 && totalExceedsCeiling {
            return .clearAllocatorCacheThenAdmit
        }
        return .admit
    }
}
