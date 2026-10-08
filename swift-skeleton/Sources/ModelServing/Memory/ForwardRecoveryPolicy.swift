import Foundation

/// Pure recovery arithmetic feeding one forward-retry decision.
///
/// These formulas know nothing about MLX, model identities, or hardware.
/// Runtime code supplies measured byte counts; the formulas decide only how
/// much elastic expert retention must yield and whether an unchanged retry
/// is authorized. Reclamation execution lives elsewhere; this policy only
/// computes the retry path's numbers.
public enum ForwardRecoveryPolicy {

    /**
     * Reconstructs the workspace required when one forward allocation fails.
     *
     * The failed allocation is additional to the transient arrays that were
     * already active at the failure boundary. Taking only the larger value
     * underestimates the retry. The observed high-water remains a reusable
     * lower bound.
     *
     * - Parameters:
     *   - stableActiveMemoryBytes: Stable active bytes after restoring the
     *     request checkpoint.
     *   - activeMemoryBytesAtFailure: Active bytes observed at the failed
     *     lazy-allocation boundary.
     *   - attemptedAllocationBytes: Exact allocation size rejected by the
     *     accelerator, or a conservative substitute.
     *   - observedTransientHighWaterBytes: Reusable transient lower bound
     *     learned from completed forwards.
     * - Returns: The fixed workspace the unchanged retry must fit.
     */
    public static func fixedWorkspaceBytes(
        stableActiveMemoryBytes: Int,
        activeMemoryBytesAtFailure: Int,
        attemptedAllocationBytes: Int,
        observedTransientHighWaterBytes: Int
    ) -> Int {
        let activeTransientMemoryBytes: Int = SaturatingArithmetic.subtractInt(
            activeMemoryBytesAtFailure, stableActiveMemoryBytes)
        let failedForwardWorkspaceBytes: Int =
            activeTransientMemoryBytes + attemptedAllocationBytes
        if failedForwardWorkspaceBytes > observedTransientHighWaterBytes {
            return failedForwardWorkspaceBytes
        }
        return observedTransientHighWaterBytes
    }

    /**
     * Expert bytes that must yield so a **fixed** forward size can fit the
     * ceiling.
     *
     * Chunk size is an input, not a free variable. Non-expert memory (model
     * core, restored context, and other owners) is treated as fixed for this
     * decision. Retained experts are the elastic category:
     *
     * `nonElastic = currentActive - retainedExperts`
     * `maxExpertsKeep = ceiling - nonElastic - fixedForwardWorkspace`
     * `reclaim = retainedExperts - max(0, maxExpertsKeep)`
     *
     * If even zero experts cannot leave room for the fixed workspace, the
     * full retained payload is required for reclamation and the caller must
     * reject when that still cannot satisfy the forward.
     *
     * - Parameters:
     *   - currentActiveMemoryBytes: Active bytes including retained experts.
     *   - retainedExpertPayloadBytes: Elastic expert payload currently held.
     *   - memoryCeilingBytes: Ceiling the retried forward must fit.
     *   - fixedForwardWorkspaceBytes: Workspace the retried forward needs.
     * - Returns: The expert bytes that must be reclaimed.
     */
    public static func requiredReclamationBytes(
        currentActiveMemoryBytes: Int,
        retainedExpertPayloadBytes: Int,
        memoryCeilingBytes: Int,
        fixedForwardWorkspaceBytes: Int
    ) -> Int {
        let nonElasticActiveMemoryBytes: Int = SaturatingArithmetic.subtractInt(
            currentActiveMemoryBytes, retainedExpertPayloadBytes)
        let ceilingAfterNonElasticBytes: Int = SaturatingArithmetic.subtractInt(
            memoryCeilingBytes, nonElasticActiveMemoryBytes)
        let maximumExpertPayloadBytesAfterForward: Int = SaturatingArithmetic.subtractInt(
            ceilingAfterNonElasticBytes, fixedForwardWorkspaceBytes)
        return SaturatingArithmetic.subtractInt(
            retainedExpertPayloadBytes, maximumExpertPayloadBytesAfterForward)
    }

    /**
     * Returns whether one unchanged fixed forward should retry after expert
     * eviction.
     *
     * Native cache residency is authoritative at this ownership boundary. An
     * accelerator active-memory sample can remain unchanged until an
     * immutable execution snapshot releases the evicted page array, even
     * though cache policy has made enough capacity available for the
     * restored request to retry.
     *
     * - Parameters:
     *   - hasAlreadyRetriedAfterReclamation: Prevents an unbounded retry loop.
     *   - retainedExpertPayloadBytesBeforeReclamation: Elastic payload before
     *     the executor reclaimed.
     *   - retainedExpertPayloadBytesAfterReclamation: Elastic payload after
     *     the executor reclaimed.
     *   - expertReclamationTargetBytes: The reclamation the retry requires.
     *   - sparseExpertsArePaged: A paged retry re-reads the layer just freed
     *     to satisfy the ceiling, promotes it again, and hits the same
     *     boundary after a long SSD stall; shrink the chunk instead.
     * - Returns: Whether the unchanged retry is authorized.
     */
    public static func retryIsAuthorized(
        hasAlreadyRetriedAfterReclamation: Bool,
        retainedExpertPayloadBytesBeforeReclamation: UInt64,
        retainedExpertPayloadBytesAfterReclamation: UInt64,
        expertReclamationTargetBytes: Int,
        sparseExpertsArePaged: Bool
    ) -> Bool {
        if sparseExpertsArePaged {
            return false
        }
        let releasedExpertPayloadBytes: UInt64 = SaturatingArithmetic.subtract(
            retainedExpertPayloadBytesBeforeReclamation,
            retainedExpertPayloadBytesAfterReclamation)
        let reclamationTargetBytes: UInt64 = UInt64(clamping: expertReclamationTargetBytes)
        return !hasAlreadyRetriedAfterReclamation
            && expertReclamationTargetBytes > 0
            && releasedExpertPayloadBytes >= reclamationTargetBytes
    }
}
