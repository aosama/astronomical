import Foundation

/// Disjoint assignment positions for one retained page and its route
/// misses. Port of the Rust `ExpertPageRoutePartition` record: the paging
/// decorator executes the retained positions against the resident page
/// and streams only the missing experts, so no expert runs twice.
public struct ExpertRoutePartition: Equatable, Sendable {

    /// Route-order positions whose expert already sits in the page.
    public var retainedAssignmentPositions: [Int]

    /// Sorted, deduplicated expert ids the page already holds.
    public var retainedExpertIds: [Int]

    /// Route-order positions whose expert the page is missing.
    public var missingAssignmentPositions: [Int]

    /// Sorted, deduplicated expert ids the page must stream in.
    public var missingExpertIds: [Int]

    public init(
        retainedAssignmentPositions: [Int],
        retainedExpertIds: [Int],
        missingAssignmentPositions: [Int],
        missingExpertIds: [Int]
    ) {
        self.retainedAssignmentPositions = retainedAssignmentPositions
        self.retainedExpertIds = retainedExpertIds
        self.missingAssignmentPositions = missingAssignmentPositions
        self.missingExpertIds = missingExpertIds
    }
}
