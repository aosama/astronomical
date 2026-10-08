import Foundation

/// Ownership evidence used before changing a process-wide MLX ceiling.
///
/// Raising and lowering are not symmetric: raising lets MLX accept capacity
/// before the budget is published, lowering reclaims before MLX enforces the
/// smaller limit. The family enacts the returned decision against the MLX
/// native limit.
public struct MemoryCeilingChangeRequirements: Equatable, Hashable, Sendable {

    /// Ceiling currently installed in the MLX runtime and memory policy owners.
    public let currentCeilingBytes: UInt64

    /// User-requested replacement ceiling.
    public let requestedCeilingBytes: UInt64

    /// Smallest ceiling that preserves non-evictable model and page ownership.
    public let minimumSafeCeilingBytes: UInt64

    /// Live active bytes sampled before planning the transition.
    public let currentActiveMemoryBytes: UInt64

    /// Elastic paged-expert bytes that a lower ceiling may reclaim.
    public let retainedPagedExpertPayloadBytes: UInt64

    /// Whether expert ownership must first transition from complete to paged.
    public let completeExpertsAreResident: Bool

    /// Request workspace that must remain available beside complete experts.
    public let completeResidencyRequiredHeadroomBytes: UInt64

    public init(
        currentCeilingBytes: UInt64,
        requestedCeilingBytes: UInt64,
        minimumSafeCeilingBytes: UInt64,
        currentActiveMemoryBytes: UInt64,
        retainedPagedExpertPayloadBytes: UInt64,
        completeExpertsAreResident: Bool,
        completeResidencyRequiredHeadroomBytes: UInt64
    ) {
        self.currentCeilingBytes = currentCeilingBytes
        self.requestedCeilingBytes = requestedCeilingBytes
        self.minimumSafeCeilingBytes = minimumSafeCeilingBytes
        self.currentActiveMemoryBytes = currentActiveMemoryBytes
        self.retainedPagedExpertPayloadBytes = retainedPagedExpertPayloadBytes
        self.completeExpertsAreResident = completeExpertsAreResident
        self.completeResidencyRequiredHeadroomBytes = completeResidencyRequiredHeadroomBytes
    }

    /// Plans the raise, lower, or refusal for the requested ceiling.
    public func decide() -> MemoryCeilingChangeDecision {
        if requestedCeilingBytes < minimumSafeCeilingBytes {
            return .reject(
                boundary: .liveCeilingMinimum,
                shortfallBytes: minimumSafeCeilingBytes - requestedCeilingBytes)
        }
        if requestedCeilingBytes == currentCeilingBytes {
            return .unchanged
        }
        if requestedCeilingBytes > currentCeilingBytes {
            return .raise(mayAttemptCompleteResidency: true)
        }
        let requiredReclamationBytes: UInt64 = SaturatingArithmetic.subtract(
            currentActiveMemoryBytes,
            requestedCeilingBytes)
        let completeResidencyProjectionBytes: UInt64 = SaturatingArithmetic.add(
            currentActiveMemoryBytes,
            completeResidencyRequiredHeadroomBytes)
        let mustDemoteCompleteResidency: Bool =
            completeExpertsAreResident
            && completeResidencyProjectionBytes > requestedCeilingBytes
        let retainedPagedExpertReclamationBytes: UInt64
        if completeExpertsAreResident {
            retainedPagedExpertReclamationBytes = 0
        } else if requiredReclamationBytes < retainedPagedExpertPayloadBytes {
            retainedPagedExpertReclamationBytes = requiredReclamationBytes
        } else {
            retainedPagedExpertReclamationBytes = retainedPagedExpertPayloadBytes
        }
        return .lower(
            mustDemoteCompleteResidency: mustDemoteCompleteResidency,
            retainedPagedExpertReclamationBytes: retainedPagedExpertReclamationBytes)
    }
}
