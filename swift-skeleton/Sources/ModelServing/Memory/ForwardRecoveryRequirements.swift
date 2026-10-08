import Foundation

/// Evidence available after checkpoint restoration and expert reclamation.
///
/// Execution families supply the failure's boundary and shortfall; this
/// value decides whether the failed forward may be retried unchanged after
/// expert reclamation, or the request must fail. Reclamation execution
/// lives elsewhere; this type only decides the retry path.
public struct ForwardRecoveryRequirements: Equatable, Hashable, Sendable {

    /// Stable active bytes after restoring the request checkpoint.
    public var stableActiveMemoryBytes: Int

    /// Active bytes observed at the failed lazy-allocation boundary.
    public var activeMemoryBytesAtFailure: Int

    /// Exact allocation size rejected by the accelerator, or a conservative substitute.
    public var attemptedAllocationBytes: Int

    /// Reusable transient lower bound learned from completed forwards.
    public var observedTransientHighWaterBytes: Int

    /// Elastic expert payload before the executor performs reclamation.
    public var retainedExpertPayloadBytesBeforeReclamation: Int

    /// Elastic expert payload after the executor performs reclamation.
    public var retainedExpertPayloadBytesAfterReclamation: Int

    /// Ceiling against which the unchanged retry must fit.
    public var activeMemoryCeilingBytes: Int

    /// Prevents a capacity failure from entering an unbounded retry loop.
    public var hasAlreadyRetriedAfterReclamation: Bool

    /// Paged experts must not retry the same chunk after reclaiming a layer:
    /// the retry re-reads the layer just freed, promotes it again, and hits
    /// the same boundary after a long SSD stall. Shrink the chunk instead.
    public var sparseExpertsArePaged: Bool

    /**
     * Creates the recovery evidence for one failed forward.
     *
     * - Parameters:
     *   - stableActiveMemoryBytes: Stable active bytes after checkpoint restore.
     *   - activeMemoryBytesAtFailure: Active bytes at the failed allocation.
     *   - attemptedAllocationBytes: Rejected allocation size (or substitute).
     *   - observedTransientHighWaterBytes: Reusable transient lower bound.
     *   - retainedExpertPayloadBytesBeforeReclamation: Elastic payload before reclamation.
     *   - retainedExpertPayloadBytesAfterReclamation: Elastic payload after reclamation.
     *   - activeMemoryCeilingBytes: Ceiling the retry must fit.
     *   - hasAlreadyRetriedAfterReclamation: Whether one retry already ran.
     *   - sparseExpertsArePaged: Whether experts are SSD-paged for this request.
     */
    public init(
        stableActiveMemoryBytes: Int,
        activeMemoryBytesAtFailure: Int,
        attemptedAllocationBytes: Int,
        observedTransientHighWaterBytes: Int,
        retainedExpertPayloadBytesBeforeReclamation: Int,
        retainedExpertPayloadBytesAfterReclamation: Int,
        activeMemoryCeilingBytes: Int,
        hasAlreadyRetriedAfterReclamation: Bool,
        sparseExpertsArePaged: Bool
    ) {
        self.stableActiveMemoryBytes = stableActiveMemoryBytes
        self.activeMemoryBytesAtFailure = activeMemoryBytesAtFailure
        self.attemptedAllocationBytes = attemptedAllocationBytes
        self.observedTransientHighWaterBytes = observedTransientHighWaterBytes
        self.retainedExpertPayloadBytesBeforeReclamation =
            retainedExpertPayloadBytesBeforeReclamation
        self.retainedExpertPayloadBytesAfterReclamation =
            retainedExpertPayloadBytesAfterReclamation
        self.activeMemoryCeilingBytes = activeMemoryCeilingBytes
        self.hasAlreadyRetriedAfterReclamation = hasAlreadyRetriedAfterReclamation
        self.sparseExpertsArePaged = sparseExpertsArePaged
    }

    /**
     * Decides whether the unchanged forward may retry and with what workspace.
     *
     * - Returns: The retry authorization with its complete calculation evidence.
     */
    public func decide() -> ForwardRecoveryDecision {
        let fixedForwardWorkspaceBytes: Int = ForwardRecoveryPolicy.fixedWorkspaceBytes(
            stableActiveMemoryBytes: stableActiveMemoryBytes,
            activeMemoryBytesAtFailure: activeMemoryBytesAtFailure,
            attemptedAllocationBytes: attemptedAllocationBytes,
            observedTransientHighWaterBytes: observedTransientHighWaterBytes)
        let requiredReclamationBytes: Int = ForwardRecoveryPolicy.requiredReclamationBytes(
            currentActiveMemoryBytes: stableActiveMemoryBytes,
            retainedExpertPayloadBytes: retainedExpertPayloadBytesBeforeReclamation,
            memoryCeilingBytes: activeMemoryCeilingBytes,
            fixedForwardWorkspaceBytes: fixedForwardWorkspaceBytes)
        let shouldRetry: Bool = ForwardRecoveryPolicy.retryIsAuthorized(
            hasAlreadyRetriedAfterReclamation: hasAlreadyRetriedAfterReclamation,
            retainedExpertPayloadBytesBeforeReclamation: UInt64(
                clamping: retainedExpertPayloadBytesBeforeReclamation),
            retainedExpertPayloadBytesAfterReclamation: UInt64(
                clamping: retainedExpertPayloadBytesAfterReclamation),
            expertReclamationTargetBytes: requiredReclamationBytes,
            sparseExpertsArePaged: sparseExpertsArePaged)
        if shouldRetry {
            return .retry(
                fixedForwardWorkspaceBytes: fixedForwardWorkspaceBytes,
                requiredReclamationBytes: requiredReclamationBytes)
        }
        return .reject(
            boundary: .allocationProjection,
            fixedForwardWorkspaceBytes: fixedForwardWorkspaceBytes,
            requiredReclamationBytes: requiredReclamationBytes)
    }
}
