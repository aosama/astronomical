import Foundation

import Testing

import ModelServing

/// Hermetic ceiling journeys, port of
/// crates/model-serving/tests/hermetic/complete_residency_headroom_boundary.rs:
/// the paging ceiling exists only where expert payload leaves no room for the
/// required startup activation headroom.
@Suite
final class CompleteResidencyHeadroomBoundaryTests {

    private func sparseGeometry() -> MlxRamBudgetModelGeometry {
        return MlxRamBudgetModelGeometry(
            modelCorePayloadBytes: 100,
            completeExpertPayloadBytes: 1_000,
            largestCompleteExpertLayerBytes: 50,
            largestRoutedExpertPageBytes: 5,
            sequenceStateBytesPerToken: 0)
    }

    @Test
    func should_return_no_paging_ceiling_when_the_artifact_has_no_headroom_gap() {
        let geometryWithoutExperts = MlxRamBudgetModelGeometry(
            modelCorePayloadBytes: 100,
            completeExpertPayloadBytes: 0,
            largestCompleteExpertLayerBytes: 50,
            largestRoutedExpertPageBytes: 5,
            sequenceStateBytesPerToken: 0)

        #expect(CompleteResidencyHeadroomBoundary(
            fromModelGeometry: geometryWithoutExperts,
            requiredHeadroomBytes: 40).pagingCeilingBytes() == nil)
        #expect(CompleteResidencyHeadroomBoundary(
            fromModelGeometry: sparseGeometry(),
            requiredHeadroomBytes: 0).pagingCeilingBytes() == nil)
    }

    @Test
    func should_place_the_paging_ceiling_where_static_weights_fit_and_headroom_does_not() {
        let requiredHeadroomBytes: UInt64 = 40
        let boundary = CompleteResidencyHeadroomBoundary(
            fromModelGeometry: sparseGeometry(),
            requiredHeadroomBytes: requiredHeadroomBytes)

        let pagingCeilingBytes: UInt64 = boundary.pagingCeilingBytes()!

        #expect(pagingCeilingBytes == 1_039)

        let rejected = CompleteResidencyRequirements(
            currentActiveMemoryBytes: 100,
            retainedPagedExpertPayloadBytes: 0,
            completeExpertPayloadBytes: 1_000,
            requiredHeadroomBytes: requiredHeadroomBytes,
            activeMemoryCeilingBytes: pagingCeilingBytes).decide()

        guard case .doesNotFit(let rejectionBoundary, _, _, _) = rejected else {
            Issue.record("expected doesNotFit, got \(rejected)")
            return
        }
        #expect(rejectionBoundary == .completeResidency)

        let admitted = CompleteResidencyRequirements(
            currentActiveMemoryBytes: 100,
            retainedPagedExpertPayloadBytes: 0,
            completeExpertPayloadBytes: 1_000,
            requiredHeadroomBytes: requiredHeadroomBytes,
            activeMemoryCeilingBytes: 100 + 1_000 + requiredHeadroomBytes).decide()

        guard case .admit = admitted else {
            Issue.record("expected admit, got \(admitted)")
            return
        }
    }
}
