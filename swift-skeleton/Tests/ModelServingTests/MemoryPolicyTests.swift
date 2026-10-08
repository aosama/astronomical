import Foundation

import Testing

import IpcProtocol
import ModelServing

/// Hermetic memory-policy journeys, port of
/// crates/model-serving/tests/hermetic/memory_policy.rs: allocation fit,
/// ceiling changes, context admission, forward recovery, and expert-ownership
/// classification all decide from exact byte evidence alone.
@Suite
final class MemoryPolicyTests {

    @Test
    func should_identify_the_allocation_boundary_and_required_shortfall() {
        let observation: AllocationAdmissionObservation = AllocationAdmissionObservation(
            activeMemoryBytes: 900,
            allocatorCacheBytes: 0,
            pendingAllocationBytes: 125,
            activeMemoryCeilingBytes: 1_000)

        #expect(observation.decide() == .reject(
            boundary: .allocationProjection,
            shortfallBytes: 25))
    }

    @Test
    func should_plan_demotion_before_installing_a_lower_ceiling() {
        let decision: MemoryCeilingChangeDecision = MemoryCeilingChangeRequirements(
            currentCeilingBytes: 25_000,
            requestedCeilingBytes: 23_000,
            minimumSafeCeilingBytes: 10_000,
            currentActiveMemoryBytes: 24_000,
            retainedPagedExpertPayloadBytes: 20_000,
            completeExpertsAreResident: true,
            completeResidencyRequiredHeadroomBytes: 2_000).decide()

        #expect(decision == .lower(
            mustDemoteCompleteResidency: true,
            retainedPagedExpertReclamationBytes: 0))
    }

    @Test
    func should_demote_complete_residency_when_only_the_idle_snapshot_fits() {
        let decision: MemoryCeilingChangeDecision = MemoryCeilingChangeRequirements(
            currentCeilingBytes: 38_000,
            requestedCeilingBytes: 30_000,
            minimumSafeCeilingBytes: 10_000,
            currentActiveMemoryBytes: 29_000,
            retainedPagedExpertPayloadBytes: 25_000,
            completeExpertsAreResident: true,
            completeResidencyRequiredHeadroomBytes: 2_000).decide()

        #expect(decision == .lower(
            mustDemoteCompleteResidency: true,
            retainedPagedExpertReclamationBytes: 0))
    }

    @Test
    func should_keep_complete_residency_when_idle_memory_and_serving_headroom_fit() {
        let decision: MemoryCeilingChangeDecision = MemoryCeilingChangeRequirements(
            currentCeilingBytes: 38_000,
            requestedCeilingBytes: 32_000,
            minimumSafeCeilingBytes: 10_000,
            currentActiveMemoryBytes: 29_000,
            retainedPagedExpertPayloadBytes: 25_000,
            completeExpertsAreResident: true,
            completeResidencyRequiredHeadroomBytes: 2_000).decide()

        #expect(decision == .lower(
            mustDemoteCompleteResidency: false,
            retainedPagedExpertReclamationBytes: 0))
    }

    @Test
    func should_preserve_paged_reclamation_when_complete_residency_headroom_is_irrelevant() {
        let decision: MemoryCeilingChangeDecision = MemoryCeilingChangeRequirements(
            currentCeilingBytes: 38_000,
            requestedCeilingBytes: 30_000,
            minimumSafeCeilingBytes: 10_000,
            currentActiveMemoryBytes: 33_000,
            retainedPagedExpertPayloadBytes: 2_000,
            completeExpertsAreResident: false,
            completeResidencyRequiredHeadroomBytes: UInt64.max).decide()

        #expect(decision == .lower(
            mustDemoteCompleteResidency: false,
            retainedPagedExpertReclamationBytes: 2_000))
    }

    @Test
    func should_not_request_demotion_for_raised_or_unchanged_ceilings() {
        let raiseRequirements: MemoryCeilingChangeRequirements = MemoryCeilingChangeRequirements(
            currentCeilingBytes: 30_000,
            requestedCeilingBytes: 38_000,
            minimumSafeCeilingBytes: 10_000,
            currentActiveMemoryBytes: 29_000,
            retainedPagedExpertPayloadBytes: 25_000,
            completeExpertsAreResident: true,
            completeResidencyRequiredHeadroomBytes: UInt64.max)

        #expect(raiseRequirements.decide() == .raise(mayAttemptCompleteResidency: true))

        let unchangedRequirements: MemoryCeilingChangeRequirements =
            MemoryCeilingChangeRequirements(
                currentCeilingBytes: 30_000,
                requestedCeilingBytes: raiseRequirements.currentCeilingBytes,
                minimumSafeCeilingBytes: 10_000,
                currentActiveMemoryBytes: 29_000,
                retainedPagedExpertPayloadBytes: 25_000,
                completeExpertsAreResident: true,
                completeResidencyRequiredHeadroomBytes: UInt64.max)

        #expect(unchangedRequirements.decide() == .unchanged)
    }

    @Test
    func should_request_allocator_cleanup_without_confusing_cache_with_active_memory() {
        let observation: AllocationAdmissionObservation = AllocationAdmissionObservation(
            activeMemoryBytes: 800,
            allocatorCacheBytes: 300,
            pendingAllocationBytes: 100,
            activeMemoryCeilingBytes: 1_000)

        #expect(observation.decide() == .clearAllocatorCacheThenAdmit)
    }

    @Test
    func should_decide_complete_residency_from_one_replacement_aware_policy() {
        let requirements: CompleteResidencyRequirements = CompleteResidencyRequirements(
            currentActiveMemoryBytes: 2_000,
            retainedPagedExpertPayloadBytes: 600,
            completeExpertPayloadBytes: 1_000,
            requiredHeadroomBytes: 500,
            activeMemoryCeilingBytes: 3_000)

        #expect(requirements.decide() == .admit(
            projectedActiveMemoryBytes: 2_400,
            requiredHeadroomBytes: 500))
    }

    @Test
    func should_authorize_only_one_retry_after_required_experts_were_reclaimed() {
        let requirements: ForwardRecoveryRequirements = ForwardRecoveryRequirements(
            stableActiveMemoryBytes: 900,
            activeMemoryBytesAtFailure: 950,
            attemptedAllocationBytes: 100,
            observedTransientHighWaterBytes: 25,
            retainedExpertPayloadBytesBeforeReclamation: 200,
            retainedExpertPayloadBytesAfterReclamation: 100,
            activeMemoryCeilingBytes: 1_000,
            hasAlreadyRetriedAfterReclamation: false,
            sparseExpertsArePaged: false)

        #expect(requirements.decide() == .retry(
            fixedForwardWorkspaceBytes: 150,
            requiredReclamationBytes: 50))

        let alreadyRetriedRequirements: ForwardRecoveryRequirements =
            ForwardRecoveryRequirements(
                stableActiveMemoryBytes: 900,
                activeMemoryBytesAtFailure: 950,
                attemptedAllocationBytes: 100,
                observedTransientHighWaterBytes: 25,
                retainedExpertPayloadBytesBeforeReclamation: 200,
                retainedExpertPayloadBytesAfterReclamation: 100,
                activeMemoryCeilingBytes: 1_000,
                hasAlreadyRetriedAfterReclamation: true,
                sparseExpertsArePaged: false)
        var decidedToReject: Bool = false
        if case .reject = alreadyRetriedRequirements.decide() {
            decidedToReject = true
        }

        #expect(decidedToReject)
    }

    @Test
    func should_admit_decode_when_only_a_temporary_cache_restore_workspace_forced_demotion() {
        // A fitting resident model plus a large prompt-cache restore workspace
        // can exceed the ceiling even though decode itself still fits. Demote
        // for the restore, then admit once that workspace is gone.
        let restoreAdmission: MemoryAdmissionDecision = ContextAdmissionRequirements(
            currentActiveMemoryBytes: 26_500_000_000,
            contextGrowthBytes: 1_700_000_000,
            expertPageReservationBytes: 0,
            temporaryWorkspaceBytes: 8_100_000_000,
            retainedExpertPayloadBytes: 21_600_000_000,
            activeMemoryCeilingBytes: 35_000_000_000,
            completeExpertsAreResident: true).decide()
        let decodeAdmission: MemoryAdmissionDecision = ContextAdmissionRequirements(
            currentActiveMemoryBytes: 26_900_000_000,
            contextGrowthBytes: 400_000_000,
            expertPageReservationBytes: 0,
            temporaryWorkspaceBytes: 0,
            retainedExpertPayloadBytes: 21_600_000_000,
            activeMemoryCeilingBytes: 35_000_000_000,
            completeExpertsAreResident: true).decide()

        #expect(restoreAdmission == .demoteCompleteResidency(reassessAfterDemotion: true))
        #expect(decodeAdmission == .admit)
    }

    @Test
    func should_keep_complete_residency_when_cache_restore_overlap_excludes_output_budget() throws {
        // Live agent turn on resident sparse MoE at a 35 GB ceiling: 15,399
        // prompt tokens, 65,535 max output, cache on, 21.59 GB experts already
        // seated. Charging restore overlap on prompt+output demoted; charging
        // it on prompt tokens only keeps complete residency.
        let contextMemoryReservationBytesPerToken: Int = 20_480
        let promptTokenCount: Int = 15_399
        let maximumOutputTokenCount: Int = 65_535
        let currentActiveMemoryBytes: Int = 26_488_988_932
        let activeMemoryCeilingBytes: Int = 35_000_000_000
        let directPublicationWorkspaceBytes: Int = 3_407_872
        let prefillActivationWorkspaceBytes: Int = 4_831_838_208
        let completeLayerScratchBytes: Int = 1_610_612_736
        let retainedExpertPayloadBytes: Int = 21_592_276_992
        let totalContextTokenCount: Int = promptTokenCount + maximumOutputTokenCount
        let contextGrowthBytes: Int = try #require(
            ContextWorkspaceBytes.persistentContextRestoreWorkspaceBytes(
                contextMemoryReservationBytesPerToken: contextMemoryReservationBytesPerToken,
                restoredContextTokenCount: totalContextTokenCount))
        let restoreOverlapIncludingOutputBudgetBytes: Int = try #require(
            ContextWorkspaceBytes.persistentContextRestoreWorkspaceBytes(
                contextMemoryReservationBytesPerToken: contextMemoryReservationBytesPerToken,
                restoredContextTokenCount: totalContextTokenCount))
        let restoreOverlapPromptTokensOnlyBytes: Int = try #require(
            ContextWorkspaceBytes.persistentContextRestoreWorkspaceBytes(
                contextMemoryReservationBytesPerToken: contextMemoryReservationBytesPerToken,
                restoredContextTokenCount: promptTokenCount))
        let outputBudgetTemporaryWorkspaceBytes: Int =
            directPublicationWorkspaceBytes
            + restoreOverlapIncludingOutputBudgetBytes
            + prefillActivationWorkspaceBytes
            + completeLayerScratchBytes
        let promptOnlyTemporaryWorkspaceBytes: Int =
            directPublicationWorkspaceBytes
            + restoreOverlapPromptTokensOnlyBytes
            + prefillActivationWorkspaceBytes
            + completeLayerScratchBytes
        let admissionWithOutputBudgetInRestore: MemoryAdmissionDecision =
            ContextAdmissionRequirements(
                currentActiveMemoryBytes: currentActiveMemoryBytes,
                contextGrowthBytes: contextGrowthBytes,
                expertPageReservationBytes: 0,
                temporaryWorkspaceBytes: outputBudgetTemporaryWorkspaceBytes,
                retainedExpertPayloadBytes: retainedExpertPayloadBytes,
                activeMemoryCeilingBytes: activeMemoryCeilingBytes,
                completeExpertsAreResident: true).decide()
        let admissionWithPromptOnlyRestore: MemoryAdmissionDecision =
            ContextAdmissionRequirements(
                currentActiveMemoryBytes: currentActiveMemoryBytes,
                contextGrowthBytes: contextGrowthBytes,
                expertPageReservationBytes: 0,
                temporaryWorkspaceBytes: promptOnlyTemporaryWorkspaceBytes,
                retainedExpertPayloadBytes: retainedExpertPayloadBytes,
                activeMemoryCeilingBytes: activeMemoryCeilingBytes,
                completeExpertsAreResident: true).decide()

        #expect(admissionWithOutputBudgetInRestore == .demoteCompleteResidency(
            reassessAfterDemotion: true))
        #expect(admissionWithPromptOnlyRestore == .admit)
    }

    @Test
    func should_take_the_larger_exclusive_request_phase_instead_of_summing_them() throws {
        // Live 33 GB turn: restore 0.32 GB and serving KV 1.67 GB do not
        // coexist.
        let currentActiveMemoryBytes: Int = 26_488_988_934
        let contextGrowthBytes: Int = 1_665_556_480
        let restoreOverlapWorkspaceBytes: Int = 323_399_680
        let publicationWorkspaceBytes: Int = 3_407_872
        let stackedExclusivePeaksBytes: Int = currentActiveMemoryBytes
            + contextGrowthBytes
            + restoreOverlapWorkspaceBytes
            + publicationWorkspaceBytes
            + 4_831_838_208
            + 1_610_612_736
        let concurrentPeakBytes: Int = try #require(
            ContextWorkspaceBytes.seatedCompleteExpertRequestPeakActiveMemoryBytes(
                currentActiveMemoryBytes: currentActiveMemoryBytes,
                contextGrowthBytes: contextGrowthBytes,
                restoreOverlapWorkspaceBytes: restoreOverlapWorkspaceBytes,
                publicationWorkspaceBytes: publicationWorkspaceBytes))
        let expectedSeatedPeakBytes: Int = currentActiveMemoryBytes
            + contextGrowthBytes
            + publicationWorkspaceBytes

        #expect(concurrentPeakBytes == expectedSeatedPeakBytes)
        #expect(concurrentPeakBytes <= 33_000_000_000)
        #expect(stackedExclusivePeaksBytes > 33_000_000_000)
    }

    @Test
    func should_admit_seated_complete_experts_at_a_33_gb_ceiling_without_stacked_layer_weight_headroom() throws {
        let currentActiveMemoryBytes: Int = 26_488_988_934
        let contextGrowthBytes: Int = 1_665_556_480
        let restoreOverlapWorkspaceBytes: Int = 323_399_680
        let publicationWorkspaceBytes: Int = 3_407_872
        let retainedExpertPayloadBytes: Int = 21_592_276_992
        let activeMemoryCeilingBytes: Int = 33_000_000_000
        let temporaryWorkspaceBytes: Int = try #require(
            ContextWorkspaceBytes.seatedCompleteExpertRequestTemporaryWorkspaceBytes(
                contextGrowthBytes: contextGrowthBytes,
                restoreOverlapWorkspaceBytes: restoreOverlapWorkspaceBytes,
                publicationWorkspaceBytes: publicationWorkspaceBytes))

        #expect(temporaryWorkspaceBytes == publicationWorkspaceBytes)
        #expect(ContextAdmissionRequirements(
            currentActiveMemoryBytes: currentActiveMemoryBytes,
            contextGrowthBytes: contextGrowthBytes,
            expertPageReservationBytes: 0,
            temporaryWorkspaceBytes: temporaryWorkspaceBytes,
            retainedExpertPayloadBytes: retainedExpertPayloadBytes,
            activeMemoryCeilingBytes: activeMemoryCeilingBytes,
            completeExpertsAreResident: true).decide() == .admit)
    }

    @Test
    func should_admit_complete_residency_without_prefill_three_layer_weight_headroom() {
        let projectedResidentActiveMemoryBytes: UInt64 = 28_193_105_290
        let activeMemoryCeilingBytes: UInt64 = 33_000_000_000
        let gateUpFusionTransientPayloadBytes: UInt64 = 1_073_741_824
        let prefillThreeLayerWeightHeadroomBytes: UInt64 = 4_831_838_208

        #expect(!ExpertMemoryAdmission.completeResidencyExceedsCeilingWithActivationHeadroom(
            projectedResidentActiveMemoryBytes: projectedResidentActiveMemoryBytes,
            stableMemoryCeilingBytes: activeMemoryCeilingBytes,
            requiredActivationHeadroomBytes: gateUpFusionTransientPayloadBytes))
        #expect(ExpertMemoryAdmission.completeResidencyExceedsCeilingWithActivationHeadroom(
            projectedResidentActiveMemoryBytes: projectedResidentActiveMemoryBytes,
            stableMemoryCeilingBytes: activeMemoryCeilingBytes,
            requiredActivationHeadroomBytes: prefillThreeLayerWeightHeadroomBytes))
    }

    @Test
    func should_classify_complete_owner_as_resident_and_empty_pager_cache_as_paged() {
        #expect(ExpertMemoryModeClassification.classify(
            completeSparseOwnerIsInstalled: true,
            sparseExpertPagingIsConfigured: true,
            retainedPagedExpertPayloadBytes: 0) == ExpertMemoryMode.resident)
        #expect(ExpertMemoryModeClassification.classify(
            completeSparseOwnerIsInstalled: false,
            sparseExpertPagingIsConfigured: false,
            retainedPagedExpertPayloadBytes: 0) == ExpertMemoryMode.resident)
        #expect(ExpertMemoryModeClassification.classify(
            completeSparseOwnerIsInstalled: false,
            sparseExpertPagingIsConfigured: true,
            retainedPagedExpertPayloadBytes: 0) == ExpertMemoryMode.paged)
        #expect(ExpertMemoryModeClassification.classify(
            completeSparseOwnerIsInstalled: false,
            sparseExpertPagingIsConfigured: true,
            retainedPagedExpertPayloadBytes: 15_854_469_120) == ExpertMemoryMode.hybrid)
    }

    @Test
    func should_ignore_layer_weight_workspace_when_complete_experts_are_already_seated() throws {
        let contextGrowthBytes: Int = 1_665_556_480
        let restoreOverlapWorkspaceBytes: Int = 323_399_680
        let publicationWorkspaceBytes: Int = 3_407_872
        let pagedPrefillActivationWorkspaceBytes: Int = 4_831_838_208
        let pagedCompleteLayerScratchBytes: Int = 1_610_612_736
        let seatedTemporaryWorkspaceBytes: Int = try #require(
            ContextWorkspaceBytes.requestContextTemporaryWorkspaceBytes(
                completeExpertsAreResident: true,
                contextGrowthBytes: contextGrowthBytes,
                restoreOverlapWorkspaceBytes: restoreOverlapWorkspaceBytes,
                publicationWorkspaceBytes: publicationWorkspaceBytes,
                pagedPrefillActivationWorkspaceBytes: pagedPrefillActivationWorkspaceBytes,
                pagedCompleteLayerScratchBytes: pagedCompleteLayerScratchBytes))
        let pagedTemporaryWorkspaceBytes: Int = try #require(
            ContextWorkspaceBytes.requestContextTemporaryWorkspaceBytes(
                completeExpertsAreResident: false,
                contextGrowthBytes: contextGrowthBytes,
                restoreOverlapWorkspaceBytes: restoreOverlapWorkspaceBytes,
                publicationWorkspaceBytes: publicationWorkspaceBytes,
                pagedPrefillActivationWorkspaceBytes: pagedPrefillActivationWorkspaceBytes,
                pagedCompleteLayerScratchBytes: pagedCompleteLayerScratchBytes))

        #expect(seatedTemporaryWorkspaceBytes == 3_407_872)
        #expect(pagedTemporaryWorkspaceBytes == publicationWorkspaceBytes
            + restoreOverlapWorkspaceBytes
            + pagedPrefillActivationWorkspaceBytes
            + pagedCompleteLayerScratchBytes)
    }
}
