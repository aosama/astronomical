import Foundation

/// Exact context and workspace categories supplied by execution owners.
///
/// Families measure the live MLX active bytes and the exact decoder-state
/// growth; `decide()` returns the shared `MemoryAdmissionDecision` without
/// mutating expert ownership. The workspace byte functions in
/// `ContextWorkspaceBytes` are the exact persistent/temporary accounting that
/// prefill, persistent-cache restore, and complete-resident seating charge.
public struct ContextAdmissionRequirements: Equatable, Hashable, Sendable {

    /// Live MLX active bytes sampled before admitting context growth.
    public let currentActiveMemoryBytes: Int

    /// Exact persistent decoder-state growth required by the operation.
    public let contextGrowthBytes: Int

    /// Largest expert page that may coexist with the context operation.
    public let expertPageReservationBytes: Int

    /// Explicit temporary owner, such as prompt-cache reconstruction workspace.
    public let temporaryWorkspaceBytes: Int

    /// Elastic paged-expert payload available for reclamation.
    public let retainedExpertPayloadBytes: Int

    /// Stable MLX active-memory ceiling resolved for the worker.
    public let activeMemoryCeilingBytes: Int

    /// Whether the current expert owner is indivisible and must demote first.
    public let completeExpertsAreResident: Bool

    public init(
        currentActiveMemoryBytes: Int,
        contextGrowthBytes: Int,
        expertPageReservationBytes: Int,
        temporaryWorkspaceBytes: Int,
        retainedExpertPayloadBytes: Int,
        activeMemoryCeilingBytes: Int,
        completeExpertsAreResident: Bool
    ) {
        self.currentActiveMemoryBytes = currentActiveMemoryBytes
        self.contextGrowthBytes = contextGrowthBytes
        self.expertPageReservationBytes = expertPageReservationBytes
        self.temporaryWorkspaceBytes = temporaryWorkspaceBytes
        self.retainedExpertPayloadBytes = retainedExpertPayloadBytes
        self.activeMemoryCeilingBytes = activeMemoryCeilingBytes
        self.completeExpertsAreResident = completeExpertsAreResident
    }

    /// Returns the complete active-memory projection or `nil` on overflow.
    public func projectedActiveMemoryBytes() -> Int? {
        let (withContextGrowthBytes, growthOverflowed) = currentActiveMemoryBytes
            .addingReportingOverflow(contextGrowthBytes)
        if growthOverflowed {
            return nil
        }
        let (withPageReservationBytes, pageOverflowed) = withContextGrowthBytes
            .addingReportingOverflow(expertPageReservationBytes)
        if pageOverflowed {
            return nil
        }
        let (withWorkspaceBytes, workspaceOverflowed) = withPageReservationBytes
            .addingReportingOverflow(temporaryWorkspaceBytes)
        if workspaceOverflowed {
            return nil
        }
        return withWorkspaceBytes
    }

    /// Decides admission without mutating expert ownership.
    public func decide() -> MemoryAdmissionDecision {
        guard let projectedActiveMemoryBytes: Int = projectedActiveMemoryBytes() else {
            return .reject(boundary: .stableActiveCeiling, shortfallBytes: UInt64.max)
        }
        if projectedActiveMemoryBytes <= activeMemoryCeilingBytes {
            return .admit
        }
        if completeExpertsAreResident {
            return .demoteCompleteResidency(reassessAfterDemotion: true)
        }
        let requiredBytes: Int = SaturatingArithmetic.subtractInt(
            projectedActiveMemoryBytes,
            activeMemoryCeilingBytes)
        if requiredBytes <= retainedExpertPayloadBytes {
            return .reclaim(requiredBytes: UInt64(clamping: requiredBytes))
        }
        let shortfallBytes: Int = SaturatingArithmetic.subtractInt(
            requiredBytes,
            retainedExpertPayloadBytes)
        return .reject(
            boundary: .stableActiveCeiling,
            shortfallBytes: UInt64(clamping: shortfallBytes))
    }
}
