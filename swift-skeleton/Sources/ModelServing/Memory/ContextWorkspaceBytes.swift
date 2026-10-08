import Foundation

/// Exact persistent and temporary workspace byte accounting for requests.
///
/// Cache restore and full request KV are exclusive phases: restore finishes
/// before generation grows the output cache, so exclusive phase peaks are
/// taken, never summed. Prefill layer-weight heuristics and SSD stream slots
/// are already inside the seated active snapshot and are ignored while
/// complete experts stay resident.
public enum ContextWorkspaceBytes {

    /// Context-cache reconstruction temporarily owns both loaded and
    /// concatenated state; restored tokens are the ones that may already exist
    /// as prompt-cache blocks. Future output-budget tokens have no blocks and
    /// must not be multiplied in.
    public static func persistentContextRestoreWorkspaceBytes(
        contextMemoryReservationBytesPerToken: Int,
        restoredContextTokenCount: Int
    ) -> Int? {
        let (restoreWorkspaceBytes, multiplyOverflowed) = contextMemoryReservationBytesPerToken
            .multipliedReportingOverflow(by: restoredContextTokenCount)
        if multiplyOverflowed {
            return nil
        }
        return restoreWorkspaceBytes
    }

    /// Peak active memory while complete experts are already seated inside
    /// `currentActiveMemoryBytes`.
    public static func seatedCompleteExpertRequestPeakActiveMemoryBytes(
        currentActiveMemoryBytes: Int,
        contextGrowthBytes: Int,
        restoreOverlapWorkspaceBytes: Int,
        publicationWorkspaceBytes: Int
    ) -> Int? {
        let (restorePhaseBytes, restoreOverflowed) = currentActiveMemoryBytes
            .addingReportingOverflow(restoreOverlapWorkspaceBytes)
        if restoreOverflowed {
            return nil
        }
        let (restorePeakBytes, restorePeakOverflowed) = restorePhaseBytes
            .addingReportingOverflow(publicationWorkspaceBytes)
        if restorePeakOverflowed {
            return nil
        }
        let (servingPhaseBytes, servingOverflowed) = currentActiveMemoryBytes
            .addingReportingOverflow(contextGrowthBytes)
        if servingOverflowed {
            return nil
        }
        let (servingPeakBytes, servingPeakOverflowed) = servingPhaseBytes
            .addingReportingOverflow(publicationWorkspaceBytes)
        if servingPeakOverflowed {
            return nil
        }
        return max(restorePeakBytes, servingPeakBytes)
    }

    /// Temporary workspace charged against a request: seated peak extras, or
    /// paging extras.
    public static func requestContextTemporaryWorkspaceBytes(
        completeExpertsAreResident: Bool,
        contextGrowthBytes: Int,
        restoreOverlapWorkspaceBytes: Int,
        publicationWorkspaceBytes: Int,
        pagedPrefillActivationWorkspaceBytes: Int,
        pagedCompleteLayerScratchBytes: Int
    ) -> Int? {
        if completeExpertsAreResident {
            return seatedCompleteExpertRequestTemporaryWorkspaceBytes(
                contextGrowthBytes: contextGrowthBytes,
                restoreOverlapWorkspaceBytes: restoreOverlapWorkspaceBytes,
                publicationWorkspaceBytes: publicationWorkspaceBytes)
        }
        let (withRestoreBytes, restoreOverflowed) = publicationWorkspaceBytes
            .addingReportingOverflow(restoreOverlapWorkspaceBytes)
        if restoreOverflowed {
            return nil
        }
        let (withActivationBytes, activationOverflowed) = withRestoreBytes
            .addingReportingOverflow(pagedPrefillActivationWorkspaceBytes)
        if activationOverflowed {
            return nil
        }
        let (withScratchBytes, scratchOverflowed) = withActivationBytes
            .addingReportingOverflow(pagedCompleteLayerScratchBytes)
        if scratchOverflowed {
            return nil
        }
        return withScratchBytes
    }

    /// Temporary workspace that makes `current + contextGrowth + temporary`
    /// equal the seated peak: when request KV already covers prompt restore,
    /// this is publication workspace only.
    public static func seatedCompleteExpertRequestTemporaryWorkspaceBytes(
        contextGrowthBytes: Int,
        restoreOverlapWorkspaceBytes: Int,
        publicationWorkspaceBytes: Int
    ) -> Int? {
        let exclusiveRestoreBeyondContextGrowthBytes: Int = SaturatingArithmetic.subtractInt(
            restoreOverlapWorkspaceBytes,
            contextGrowthBytes)
        let (temporaryWorkspaceBytes, additionOverflowed) = publicationWorkspaceBytes
            .addingReportingOverflow(exclusiveRestoreBeyondContextGrowthBytes)
        if additionOverflowed {
            return nil
        }
        return temporaryWorkspaceBytes
    }

    /// Combines independently owned persistent growth categories.
    public static func combinedPersistentGrowthBytes(
        targetPersistentStateGrowthBytes: Int,
        additionalPersistentStateGrowthBytes: Int
    ) -> Int? {
        let (combinedGrowthBytes, additionOverflowed) = targetPersistentStateGrowthBytes
            .addingReportingOverflow(additionalPersistentStateGrowthBytes)
        if additionOverflowed {
            return nil
        }
        return combinedGrowthBytes
    }

    /// Smallest safe idle ceiling after reclaiming elastic expert payload.
    public static func safeMinimumActiveMemoryCeilingBytes(
        currentIdleActiveMemoryBytes: UInt64,
        evictableRetainedExpertPayloadBytes: UInt64,
        maximumExpertPageReserveBytes: UInt64
    ) -> UInt64 {
        let idleCeilingBytes: UInt64 = SaturatingArithmetic.subtract(
            currentIdleActiveMemoryBytes,
            evictableRetainedExpertPayloadBytes)
        return SaturatingArithmetic.add(idleCeilingBytes, maximumExpertPageReserveBytes)
    }
}
