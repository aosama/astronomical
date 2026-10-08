import Foundation

import RuntimeIntegration

/**
 * A cause-preserving failure while loading or executing the native
 * Qwen3.5 model, port of the Rust `Qwen3_5ExecutionError`. Cases whose
 * payload types have not landed yet arrive with their owning tracks;
 * this surface carries the runtime channels the execution owners
 * classify for recovery.
 */
public enum Qwen35ExecutionError: Error, Equatable, Sendable {

    /// Direct MLX execution failed.
    case runtime(MlxRuntimeError)

    /// Expert paging failed while loading or streaming sparse layers.
    case expertPaging(ExpertPagingError)

    /**
     * Returns allocator-capacity evidence regardless of whether MLX failed
     * in ordinary model execution or in expert paging, so both channels
     * enter the same checkpoint, reclamation, and chunk-reduction
     * recovery path.
     *
     * - Returns: The active, attempted, and allowed byte evidence when the
     *   failure is recoverable active-memory pressure; otherwise `nil`.
     */
    public func activeMemoryLimitExceededEvidence() -> Qwen35ActiveMemoryLimitEvidence? {
        if case .expertPaging(.memoryBudget(.rejected(
            _, _, _,
            let rejectedActiveMemoryBytes,
            let rejectedPendingAllocationBytes,
            let rejectedActiveMemoryCeilingBytes))) = self
        {
            return Qwen35ActiveMemoryLimitEvidence(
                activeMemoryBytes: Int(clamping: rejectedActiveMemoryBytes),
                attemptedAllocationBytes: Int(clamping: rejectedPendingAllocationBytes),
                allowedActiveMemoryBytes: Int(clamping: rejectedActiveMemoryCeilingBytes))
        }
        guard let mlxRuntimeError = underlyingMlxRuntimeError() else {
            return nil
        }
        if case .activeMemoryLimitExceeded(
            let activeMemoryBytes,
            let attemptedAllocationBytes,
            let allowedActiveMemoryBytes) = mlxRuntimeError
        {
            return Qwen35ActiveMemoryLimitEvidence(
                activeMemoryBytes: activeMemoryBytes,
                attemptedAllocationBytes: attemptedAllocationBytes,
                allowedActiveMemoryBytes: allowedActiveMemoryBytes)
        }
        return nil
    }

    private func underlyingMlxRuntimeError() -> MlxRuntimeError? {
        switch self {
        case .runtime(let mlxRuntimeError):
            return mlxRuntimeError
        case .expertPaging(.nativeRuntime(let mlxRuntimeError)):
            return mlxRuntimeError
        case .expertPaging:
            return nil
        }
    }
}
