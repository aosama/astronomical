import Foundation

/// Typed complete-residency policy result; execution performs no fit
/// arithmetic of its own.
public enum CompleteResidencyDecision: Equatable, Sendable {

    /// Complete experts may promote at the projected active size.
    case admit(projectedActiveMemoryBytes: UInt64, requiredHeadroomBytes: UInt64)

    /// Complete experts do not fit behind the named boundary.
    case doesNotFit(
        boundary: MemoryBoundary,
        shortfallBytes: UInt64,
        projectedActiveMemoryBytes: UInt64,
        requiredHeadroomBytes: UInt64)

    /// The supplied observation was internally inconsistent.
    case rejectInvalidObservation(error: ExpertMemoryAdmissionError)
}
