import Foundation

import Testing

import ModelServing

/// Hermetic admission arithmetic journeys, port of
/// crates/model-serving/tests/hermetic/expert_memory_admission.rs: complete
/// experts replace paged retention, headroom is reserved before promotion,
/// and the reclaim loop stops exactly when a pass releases nothing.
@Suite
final class ExpertMemoryAdmissionTests {

    @Test
    func should_project_complete_residency_as_replacement() throws {
        #expect(try ExpertMemoryAdmission.projectedActiveMemoryAfterCompleteExpertReplacement(
            currentActiveMemoryBytes: 30,
            retainedPagedExpertPayloadBytes: 20,
            completeExpertPayloadBytes: 25) == 35)

        #expect(throws: ExpertMemoryAdmissionError.retainedExpertPayloadExceedsActiveMemory) {
            _ = try ExpertMemoryAdmission.projectedActiveMemoryAfterCompleteExpertReplacement(
                currentActiveMemoryBytes: 9,
                retainedPagedExpertPayloadBytes: 10,
                completeExpertPayloadBytes: 1)
        }
        #expect(throws: ExpertMemoryAdmissionError.completeResidencyProjectionOverflow) {
            _ = try ExpertMemoryAdmission.projectedActiveMemoryAfterCompleteExpertReplacement(
                currentActiveMemoryBytes: UInt64.max,
                retainedPagedExpertPayloadBytes: 0,
                completeExpertPayloadBytes: 1)
        }
    }

    @Test
    func should_reserve_activation_headroom_before_complete_residency() {
        #expect(ExpertMemoryAdmission.requiredCompleteResidencyActivationHeadroomBytes(
            startupActivationFloorBytes: 900_000_000,
            observedTransientHighWaterBytes: 0) == 900_000_000)
        #expect(ExpertMemoryAdmission.requiredCompleteResidencyActivationHeadroomBytes(
            startupActivationFloorBytes: 900_000_000,
            observedTransientHighWaterBytes: 5_000_000_000) == 5_000_000_000)
        #expect(ExpertMemoryAdmission.completeResidencyExceedsCeilingWithActivationHeadroom(
            projectedResidentActiveMemoryBytes: 38_000_000_000,
            stableMemoryCeilingBytes: 39_000_000_000,
            requiredActivationHeadroomBytes: 3_600_000_000))
    }

    @Test
    func should_reclaim_only_the_expert_bytes_needed_by_a_fixed_forward() {
        #expect(ExpertMemoryAdmission.expertReclamationBytesToFitFixedForward(
            currentActiveMemoryBytes: 30_000_000_000,
            retainedExpertPayloadBytes: 25_000_000_000,
            memoryCeilingBytes: 32_000_000_000,
            fixedForwardWorkspaceBytes: 4_000_000_000) == 2_000_000_000)
        #expect(ExpertMemoryAdmission.expertReclamationBytesToFitFixedForward(
            currentActiveMemoryBytes: 20_000_000_000,
            retainedExpertPayloadBytes: 10_000_000_000,
            memoryCeilingBytes: 32_000_000_000,
            fixedForwardWorkspaceBytes: 4_000_000_000) == 0)
    }

    @Test
    func should_include_active_transients_in_failed_forward_workspace() {
        #expect(ExpertMemoryAdmission.fixedForwardWorkspaceAfterAllocationFailure(
            stableActiveMemoryBytes: 36_000_000_000,
            activeMemoryBytesAtFailure: 38_000_000_000,
            attemptedAllocationBytes: 800_000_000,
            observedTransientHighWaterBytes: 1_000_000_000) == 2_800_000_000)
    }

    @Test
    func should_retry_once_only_after_the_reclamation_target_was_released() {
        #expect(ExpertMemoryAdmission.shouldRetryFixedForwardAfterExpertReclamation(
            hasAlreadyRetriedAfterReclamation: false,
            retainedExpertPayloadBytesBeforeReclamation: 1_000,
            retainedExpertPayloadBytesAfterReclamation: 700,
            expertReclamationTargetBytes: 300))
        #expect(!ExpertMemoryAdmission.shouldRetryFixedForwardAfterExpertReclamation(
            hasAlreadyRetriedAfterReclamation: true,
            retainedExpertPayloadBytesBeforeReclamation: 1_000,
            retainedExpertPayloadBytesAfterReclamation: 700,
            expertReclamationTargetBytes: 300))
        #expect(!ExpertMemoryAdmission.shouldRetryFixedForwardAfterExpertReclamation(
            hasAlreadyRetriedAfterReclamation: false,
            retainedExpertPayloadBytesBeforeReclamation: 1_000,
            retainedExpertPayloadBytesAfterReclamation: 800,
            expertReclamationTargetBytes: 300))
    }

    @Test
    func should_choose_the_largest_memory_boundary_deficit() {
        let plan: ExpertReclamationPlan = ExpertReclamationPlan.forProjectedMemory(
            stableProjectedBytes: 110,
            peakProjectedBytes: 130,
            recoveryProjectedBytes: 125,
            stableMemoryCeilingBytes: 100,
            transientMemoryCeilingBytes: 120,
            retainedExpertPayloadBytes: 15)

        #expect(plan.requiredReclamationBytes == 10)
        #expect(plan.reclamationTargetBytes == 10)
        #expect(plan.unresolvedShortfallBytes == 0)
        #expect(plan.canSatisfyEveryMemoryBoundary)
    }

    @Test
    func should_report_unresolved_reclamation_shortfall() {
        let plan: ExpertReclamationPlan = ExpertReclamationPlan.forProjectedMemory(
            stableProjectedBytes: 140,
            peakProjectedBytes: 150,
            recoveryProjectedBytes: 145,
            stableMemoryCeilingBytes: 100,
            transientMemoryCeilingBytes: 120,
            retainedExpertPayloadBytes: 20)

        #expect(plan.requiredReclamationBytes == 40)
        #expect(plan.reclamationTargetBytes == 20)
        #expect(plan.unresolvedShortfallBytes == 20)
        #expect(!plan.canSatisfyEveryMemoryBoundary)
    }

    @Test
    func should_keep_reclaiming_paged_experts_while_peak_still_misses_and_payload_remains() {
        let remainingDeficit: ExpertReclamationPlan = ExpertReclamationPlan.forProjectedMemory(
            stableProjectedBytes: 90,
            peakProjectedBytes: 130,
            recoveryProjectedBytes: 130,
            stableMemoryCeilingBytes: 100,
            transientMemoryCeilingBytes: 120,
            retainedExpertPayloadBytes: 50)

        #expect(ExpertMemoryAdmission.nextPagedExpertReclamationStep(
            fitsStableAndPeakLimits: false,
            reclamationPlan: remainingDeficit,
            previousPassReleasedPages: true) == .reclaim(targetBytes: 10))
    }

    @Test
    func should_stop_paged_expert_reclamation_when_a_pass_releases_nothing() {
        let remainingDeficit: ExpertReclamationPlan = ExpertReclamationPlan.forProjectedMemory(
            stableProjectedBytes: 90,
            peakProjectedBytes: 130,
            recoveryProjectedBytes: 130,
            stableMemoryCeilingBytes: 100,
            transientMemoryCeilingBytes: 120,
            retainedExpertPayloadBytes: 50)

        #expect(ExpertMemoryAdmission.nextPagedExpertReclamationStep(
            fitsStableAndPeakLimits: false,
            reclamationPlan: remainingDeficit,
            previousPassReleasedPages: false) == .reject)
    }

    @Test
    func should_admit_once_stable_and_peak_fit_after_paged_expert_reclamation() {
        let fitting: ExpertReclamationPlan = ExpertReclamationPlan.forProjectedMemory(
            stableProjectedBytes: 90,
            peakProjectedBytes: 110,
            recoveryProjectedBytes: 110,
            stableMemoryCeilingBytes: 100,
            transientMemoryCeilingBytes: 120,
            retainedExpertPayloadBytes: 50)

        #expect(ExpertMemoryAdmission.nextPagedExpertReclamationStep(
            fitsStableAndPeakLimits: true,
            reclamationPlan: fitting,
            previousPassReleasedPages: true) == .admit)
    }
}
