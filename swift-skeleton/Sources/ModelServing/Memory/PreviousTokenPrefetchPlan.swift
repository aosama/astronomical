import Foundation

/// Admission outcome for one previous-token prefetch attempt, port of the
/// Rust `PreviousTokenPrefetchPlan`.
public struct PreviousTokenPrefetchPlan: Equatable, Sendable {

    public let expertsToRetain: [PreviousTokenPrefetchRetention]

    public let skippedAlreadyResidentCount: UInt64

    public let droppedForCapacityCount: UInt64

    public init(
        expertsToRetain: [PreviousTokenPrefetchRetention],
        skippedAlreadyResidentCount: UInt64,
        droppedForCapacityCount: UInt64
    ) {
        self.expertsToRetain = expertsToRetain
        self.skippedAlreadyResidentCount = skippedAlreadyResidentCount
        self.droppedForCapacityCount = droppedForCapacityCount
    }
}
