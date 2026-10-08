import Foundation

/// Typed policy result for one pending MLX allocation.
public enum AllocationAdmissionDecision: Equatable, Hashable, Sendable {

    /// Active and total memory both fit without cleanup.
    case admit

    /// Active ownership fits, but reclaiming allocator storage avoids total pressure.
    case clearAllocatorCacheThenAdmit

    /// Active ownership cannot fit even if allocator storage is cleared.
    case reject(boundary: MemoryBoundary, shortfallBytes: UInt64)
}
