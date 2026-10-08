import Foundation

/**
 * Allocator-capacity evidence carried by a recoverable active-memory
 * pressure failure, port of the Rust
 * `Qwen3_5ExecutionError::active_memory_limit_exceeded_evidence`
 * evidence tuple.
 */
public struct Qwen35ActiveMemoryLimitEvidence: Equatable, Sendable {

    /// Live MLX active bytes when the ceiling rejected the allocation.
    public let activeMemoryBytes: Int

    /// Exact byte count of the rejected allocation.
    public let attemptedAllocationBytes: Int

    /// Configured active-memory ceiling the allocation was measured against.
    public let allowedActiveMemoryBytes: Int

    public init(
        activeMemoryBytes: Int,
        attemptedAllocationBytes: Int,
        allowedActiveMemoryBytes: Int
    ) {
        self.activeMemoryBytes = activeMemoryBytes
        self.attemptedAllocationBytes = attemptedAllocationBytes
        self.allowedActiveMemoryBytes = allowedActiveMemoryBytes
    }
}
