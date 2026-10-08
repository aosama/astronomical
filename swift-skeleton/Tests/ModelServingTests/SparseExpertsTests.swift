import Foundation

import Testing

import ModelServing

/// Hermetic journeys over the sparse expert selection shell, port of
/// crates/model-serving/tests/hermetic/sparse_experts.rs: the assignment
/// permutation inversion the paging decorator relies on must invert a
/// complete permutation, accept the empty set, and reject duplicates and
/// out-of-range slots with the geometry error.
@Suite
final class SparseExpertsTests {

    @Test
    func should_invert_a_complete_assignment_permutation() throws {
        // sortedOrder[sortedSlot] = originalSlot for the characterization
        // example: selected indices [5, 0, 4, 1, 3, 2] argsort to
        // [1, 3, 5, 4, 2, 0].
        let sortedOrder: [UInt32] = [1, 3, 5, 4, 2, 0]

        let inverseOrder: [UInt32] = try SparseExperts.invertAssignmentOrder(sortedOrder: sortedOrder)

        #expect(inverseOrder == [5, 0, 4, 1, 3, 2])
    }

    @Test
    func should_invert_an_empty_assignment_set() throws {
        let inverseOrder: [UInt32] = try SparseExperts.invertAssignmentOrder(sortedOrder: [])

        #expect(inverseOrder.isEmpty)
    }

    @Test
    func should_reject_a_duplicate_or_out_of_range_permutation() {
        #expect(throws: SparseExpertsError.invalidAssignmentGeometry(
            description: "assignment permutation contains a duplicate slot")) {
            _ = try SparseExperts.invertAssignmentOrder(sortedOrder: [0, 0])
        }
        #expect(throws: SparseExpertsError.invalidAssignmentGeometry(
            description: "assignment permutation contains an out-of-range slot")) {
            _ = try SparseExperts.invertAssignmentOrder(sortedOrder: [0, 2])
        }
    }
}
