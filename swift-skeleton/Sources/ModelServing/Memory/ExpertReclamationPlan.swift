import Foundation

/// Exact expert-retention reclamation required by one request operation.
public struct ExpertReclamationPlan: Equatable, Sendable {

    /// Largest deficit across every mandatory memory boundary. Reclaiming
    /// one retained expert byte lowers stable, peak, and recovery
    /// projections by one byte, so the smallest reclamation that satisfies
    /// all boundaries is the maximum of their individual deficits.
    public let requiredReclamationBytes: Int

    /// The reclamation goal; never exceeds the currently retained payload.
    public let reclamationTargetBytes: Int

    /// Deficit that remains even after every retained expert byte yields.
    public let unresolvedShortfallBytes: Int

    /// Whether every mandatory boundary fits once the target is released.
    public var canSatisfyEveryMemoryBoundary: Bool {
        return self.unresolvedShortfallBytes == 0
    }

    /// Computes the plan from projected ownership against each ceiling.
    public static func forProjectedMemory(
        stableProjectedBytes: Int,
        peakProjectedBytes: Int,
        recoveryProjectedBytes: Int,
        stableMemoryCeilingBytes: Int,
        transientMemoryCeilingBytes: Int,
        retainedExpertPayloadBytes: Int
    ) -> ExpertReclamationPlan {
        let stableDeficitBytes: Int = SaturatingArithmetic.subtractInt(
            stableProjectedBytes,
            stableMemoryCeilingBytes)
        let peakDeficitBytes: Int = SaturatingArithmetic.subtractInt(
            peakProjectedBytes,
            transientMemoryCeilingBytes)
        let recoveryDeficitBytes: Int = SaturatingArithmetic.subtractInt(
            recoveryProjectedBytes,
            transientMemoryCeilingBytes)
        let requiredReclamationBytes: Int = max(
            stableDeficitBytes,
            max(peakDeficitBytes, recoveryDeficitBytes))
        let reclamationTargetBytes: Int = min(requiredReclamationBytes, retainedExpertPayloadBytes)
        let unresolvedShortfallBytes: Int = SaturatingArithmetic.subtractInt(
            requiredReclamationBytes,
            reclamationTargetBytes)
        return ExpertReclamationPlan(
            requiredReclamationBytes: requiredReclamationBytes,
            reclamationTargetBytes: reclamationTargetBytes,
            unresolvedShortfallBytes: unresolvedShortfallBytes)
    }

    public init(
        requiredReclamationBytes: Int,
        reclamationTargetBytes: Int,
        unresolvedShortfallBytes: Int
    ) {
        self.requiredReclamationBytes = requiredReclamationBytes
        self.reclamationTargetBytes = reclamationTargetBytes
        self.unresolvedShortfallBytes = unresolvedShortfallBytes
    }
}
