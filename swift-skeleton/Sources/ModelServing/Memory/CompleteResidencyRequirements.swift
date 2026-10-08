import Foundation

/// Admission question: may complete expert residency be promoted now?
///
/// Complete experts replace paged retention instead of coexisting with it.
/// Families supply the current active bytes, the retained paged payload
/// being replaced, and the exact complete payload; `decide()` projects the
/// resulting active memory plus activation headroom against the ceiling.
public struct CompleteResidencyRequirements: Sendable {

    /// Live active bytes while paged expert ownership still exists.
    public let currentActiveMemoryBytes: UInt64

    /// Paged payload replaced, rather than coexisting, with complete experts.
    public let retainedPagedExpertPayloadBytes: UInt64

    /// Exact complete expert payload derived from validated artifact geometry.
    public let completeExpertPayloadBytes: UInt64

    /// Context, activation, and additional fixed headroom required after
    /// promotion.
    public let requiredHeadroomBytes: UInt64

    /// Stable MLX active-memory ceiling.
    public let activeMemoryCeilingBytes: UInt64

    public init(
        currentActiveMemoryBytes: UInt64,
        retainedPagedExpertPayloadBytes: UInt64,
        completeExpertPayloadBytes: UInt64,
        requiredHeadroomBytes: UInt64,
        activeMemoryCeilingBytes: UInt64
    ) {
        self.currentActiveMemoryBytes = currentActiveMemoryBytes
        self.retainedPagedExpertPayloadBytes = retainedPagedExpertPayloadBytes
        self.completeExpertPayloadBytes = completeExpertPayloadBytes
        self.requiredHeadroomBytes = requiredHeadroomBytes
        self.activeMemoryCeilingBytes = activeMemoryCeilingBytes
    }

    /// Projects the replacement and admits or names the blocking boundary.
    public func decide() -> CompleteResidencyDecision {
        let projectedActiveMemoryBytes: UInt64
        do {
            projectedActiveMemoryBytes = try ExpertMemoryAdmission
                .projectedActiveMemoryAfterCompleteExpertReplacement(
                    currentActiveMemoryBytes: self.currentActiveMemoryBytes,
                    retainedPagedExpertPayloadBytes: self.retainedPagedExpertPayloadBytes,
                    completeExpertPayloadBytes: self.completeExpertPayloadBytes)
        } catch let admissionError as ExpertMemoryAdmissionError {
            return .rejectInvalidObservation(error: admissionError)
        } catch {
            return .rejectInvalidObservation(
                error: .completeResidencyProjectionOverflow)
        }
        if ExpertMemoryAdmission.completeResidencyExceedsCeilingWithActivationHeadroom(
            projectedResidentActiveMemoryBytes: projectedActiveMemoryBytes,
            stableMemoryCeilingBytes: self.activeMemoryCeilingBytes,
            requiredActivationHeadroomBytes: self.requiredHeadroomBytes) {
            let projectedWithHeadroomBytes: UInt64 = SaturatingArithmetic.add(
                projectedActiveMemoryBytes,
                self.requiredHeadroomBytes)
            return .doesNotFit(
                boundary: .completeResidency,
                shortfallBytes: SaturatingArithmetic.subtract(
                    projectedWithHeadroomBytes,
                    self.activeMemoryCeilingBytes),
                projectedActiveMemoryBytes: projectedActiveMemoryBytes,
                requiredHeadroomBytes: self.requiredHeadroomBytes)
        }
        return .admit(
            projectedActiveMemoryBytes: projectedActiveMemoryBytes,
            requiredHeadroomBytes: self.requiredHeadroomBytes)
    }
}
