import Foundation

/// Pure checked arithmetic for expert-memory ownership transitions.
///
/// Deliberately knows nothing about MLX, model identities, tensor layouts, or
/// hardware: runtime code supplies measured byte counts and these formulas
/// decide only whether those counts are internally consistent and how much
/// elastic expert retention must yield.
public enum ExpertMemoryAdmission {

    /// Projects active memory after replacing paged experts with complete
    /// experts. `currentActiveMemoryBytes` already includes the retained
    /// paged payload; complete residency replaces that owner rather than
    /// coexisting with it, so the projection is
    /// `current - retainedPaged + complete`.
    public static func projectedActiveMemoryAfterCompleteExpertReplacement(
        currentActiveMemoryBytes: UInt64,
        retainedPagedExpertPayloadBytes: UInt64,
        completeExpertPayloadBytes: UInt64
    ) throws -> UInt64 {
        let withoutRetainedPayload =
            currentActiveMemoryBytes.subtractingReportingOverflow(retainedPagedExpertPayloadBytes)
        if withoutRetainedPayload.overflow {
            throw ExpertMemoryAdmissionError.retainedExpertPayloadExceedsActiveMemory
        }
        let projectedActiveMemoryBytes =
            withoutRetainedPayload.partialValue.addingReportingOverflow(completeExpertPayloadBytes)
        if projectedActiveMemoryBytes.overflow {
            throw ExpertMemoryAdmissionError.completeResidencyProjectionOverflow
        }
        return projectedActiveMemoryBytes.partialValue
    }

    /// Activation headroom required before complete expert residency may
    /// promote. Complete residency only budgets static expert and non-expert
    /// payload, but serving still needs temporary activations, key-value
    /// growth, and workspace. Prefer the observed transient high-water from
    /// completed forwards; otherwise reserve one complete layer. Withholding
    /// a tenth of the whole expert payload instead forced SSD streaming for
    /// users who had already given enough RAM to seat the model.
    public static func requiredCompleteResidencyActivationHeadroomBytes(
        startupActivationFloorBytes: UInt64,
        observedTransientHighWaterBytes: UInt64
    ) -> UInt64 {
        if observedTransientHighWaterBytes > startupActivationFloorBytes {
            return observedTransientHighWaterBytes
        }
        return startupActivationFloorBytes
    }

    /// Returns whether static residency plus activation headroom exceeds the
    /// ceiling. Overflow means the projection already cannot fit any finite
    /// ceiling.
    public static func completeResidencyExceedsCeilingWithActivationHeadroom(
        projectedResidentActiveMemoryBytes: UInt64,
        stableMemoryCeilingBytes: UInt64,
        requiredActivationHeadroomBytes: UInt64
    ) -> Bool {
        let projectedWithHeadroom = projectedResidentActiveMemoryBytes
            .addingReportingOverflow(requiredActivationHeadroomBytes)
        if projectedWithHeadroom.overflow {
            return true
        }
        return projectedWithHeadroom.partialValue > stableMemoryCeilingBytes
    }

    /// Expert bytes that must yield so a fixed forward size can fit the
    /// ceiling. Chunk size is an input, not a free variable: non-expert
    /// memory (model core, restored context, and other owners) is treated as
    /// fixed, and retained experts are the elastic category. If even zero
    /// experts cannot leave room for the fixed workspace, the full retained
    /// payload is required and the caller must reject when that still cannot
    /// satisfy the forward.
    public static func expertReclamationBytesToFitFixedForward(
        currentActiveMemoryBytes: Int,
        retainedExpertPayloadBytes: Int,
        memoryCeilingBytes: Int,
        fixedForwardWorkspaceBytes: Int
    ) -> Int {
        let nonElasticActiveMemoryBytes: Int = SaturatingArithmetic.subtractInt(
            currentActiveMemoryBytes,
            retainedExpertPayloadBytes)
        let maximumExpertPayloadBytesAfterForward: Int = SaturatingArithmetic.subtractInt(
            SaturatingArithmetic.subtractInt(memoryCeilingBytes, nonElasticActiveMemoryBytes),
            fixedForwardWorkspaceBytes)
        return SaturatingArithmetic.subtractInt(
            retainedExpertPayloadBytes,
            maximumExpertPayloadBytesAfterForward)
    }

    /// Reconstructs the workspace required when one forward allocation fails.
    /// The failed allocation is additional to the transient arrays already
    /// active at the failure boundary; taking only the larger value would
    /// underestimate the retry. The observed high-water remains a reusable
    /// lower bound.
    public static func fixedForwardWorkspaceAfterAllocationFailure(
        stableActiveMemoryBytes: Int,
        activeMemoryBytesAtFailure: Int,
        attemptedAllocationBytes: Int,
        observedTransientHighWaterBytes: Int
    ) -> Int {
        let activeTransientMemoryBytes: Int = SaturatingArithmetic.subtractInt(
            activeMemoryBytesAtFailure,
            stableActiveMemoryBytes)
        let failedForwardWorkspaceBytes: Int = activeTransientMemoryBytes + attemptedAllocationBytes
        if failedForwardWorkspaceBytes > observedTransientHighWaterBytes {
            return failedForwardWorkspaceBytes
        }
        return observedTransientHighWaterBytes
    }

    /// Returns whether one unchanged fixed forward should retry after expert
    /// eviction. Native cache residency is authoritative at this ownership
    /// boundary: an MLX active-memory sample can remain unchanged until an
    /// immutable execution snapshot releases the evicted page array, even
    /// though cache policy has already made enough capacity available for
    /// the restored request to retry.
    public static func shouldRetryFixedForwardAfterExpertReclamation(
        hasAlreadyRetriedAfterReclamation: Bool,
        retainedExpertPayloadBytesBeforeReclamation: UInt64,
        retainedExpertPayloadBytesAfterReclamation: UInt64,
        expertReclamationTargetBytes: Int
    ) -> Bool {
        let releasedExpertPayloadBytes: UInt64 = SaturatingArithmetic.subtract(
            retainedExpertPayloadBytesBeforeReclamation,
            retainedExpertPayloadBytesAfterReclamation)
        let reclamationTargetBytes: UInt64 = UInt64(clamping: expertReclamationTargetBytes)
        return hasAlreadyRetriedAfterReclamation == false
            && expertReclamationTargetBytes > 0
            && releasedExpertPayloadBytes >= reclamationTargetBytes
    }

    /// Chooses the next paged-expert reclaim action for one forward
    /// admission. `previousPassReleasedPages` is nil before the first pass.
    /// A pass that released no pages must stop: repeating it cannot change
    /// the projection.
    public static func nextPagedExpertReclamationStep(
        fitsStableAndPeakLimits: Bool,
        reclamationPlan: ExpertReclamationPlan,
        previousPassReleasedPages: Bool?
    ) -> PagedExpertReclamationStep {
        if fitsStableAndPeakLimits {
            return .admit
        }
        if previousPassReleasedPages == false
            || reclamationPlan.canSatisfyEveryMemoryBoundary == false
            || reclamationPlan.reclamationTargetBytes == 0 {
            return .reject
        }
        return .reclaim(targetBytes: reclamationPlan.reclamationTargetBytes)
    }
}
