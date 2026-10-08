import Foundation

/// Invalid byte accounting that must fail closed before changing ownership.
public enum ExpertMemoryAdmissionError: Error, Equatable, Sendable {

    /// Retained expert payload exceeds current active memory.
    case retainedExpertPayloadExceedsActiveMemory

    /// Complete expert residency projection overflowed.
    case completeResidencyProjectionOverflow
}
