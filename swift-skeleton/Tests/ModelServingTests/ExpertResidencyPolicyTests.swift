import Foundation

import Testing

import ModelServing

/// Family-neutral expert residency policy journeys, port of
/// crates/model-serving/tests/hermetic/expert_residency_policy.rs: the
/// active plan is the phase's will for a layer, prefill keeps complete
/// layers it already seats, learned budget tightening never evicts pages a
/// request already paid for, and a paged forward must not retry the same
/// chunk after reclaiming a layer.
@Suite
final class ExpertResidencyPolicyTests {

    private static func uniformGeometry(_ layerCount: Int) -> [ExpertLayerGeometry] {
        var layerGeometries: [ExpertLayerGeometry] = []
        layerGeometries.reserveCapacity(layerCount)
        for layerIndex: Int in 0..<layerCount {
            layerGeometries.append(ExpertLayerGeometry(
                layerIndex: layerIndex,
                completeLayerPayloadBytes: 40,
                expertPayloadBytes: 10,
                expertCapacity: 4,
                expertsPerToken: 2))
        }
        return layerGeometries
    }

    private static func planError(
        phase: MemoryPhase,
        retainedExpertCeilingBytes: UInt64,
        layerGeometries: [ExpertLayerGeometry],
        currentResidencies: [CurrentExpertLayerResidency]
    ) -> ExpertResidencyPlanError? {
        do {
            _ = try ExpertResidencyPlanning.planExpertResidency(
                phase: phase,
                retainedExpertCeilingBytes: retainedExpertCeilingBytes,
                layerGeometries: layerGeometries,
                currentResidencies: currentResidencies)
            return nil
        } catch let planError as ExpertResidencyPlanError {
            return planError
        } catch {
            return nil
        }
    }

    @Test
    func should_keep_complete_layers_during_prefill_even_when_the_plan_names_a_release() {
        let layerGeometries: [ExpertLayerGeometry] = ExpertResidencyPolicyTests.uniformGeometry(3)
        let seatedCompleteLayer: [CurrentExpertLayerResidency] = [CurrentExpertLayerResidency(
            layerIndex: 2,
            pageClass: .stableCompleteLayer,
            retainedExpertIds: [0, 1, 2, 3],
            payloadBytes: 40,
            coveredWeightedDemand: 0)]
        let residencyPlan: ExpertResidencyPlan = try! ExpertResidencyPlanning.planExpertResidency(
            phase: .prefill,
            retainedExpertCeilingBytes: 60,
            layerGeometries: layerGeometries,
            currentResidencies: seatedCompleteLayer)
        #expect(residencyPlan.layerTargets[2] == .releaseCompleteForExactDeficit)
        #expect(!ExpertReleasePolicy.shouldEnactPlannedExpertRelease(
            phase: .prefill, target: residencyPlan.layerTargets[2]))
        #expect(!ExpertReleasePolicy.shouldEnactPlannedExpertRelease(
            phase: .generationPreparation, target: residencyPlan.layerTargets[2]))
        #expect(ExpertReleasePolicy.shouldEnactPlannedExpertRelease(
            phase: .idle, target: residencyPlan.layerTargets[2]))
    }

    @Test
    func should_seat_complete_layers_on_mandatory_prefill_reads() {
        #expect(ExpertPageCommitPolicy.shouldCommitMandatoryCompleteLayer(
            routeTokenCount: 2_048,
            productionDefaultPaging: true,
            residencyTarget: .promoteCompleteOnMandatoryRead))
        #expect(ExpertPageCommitPolicy.shouldCommitMandatoryCompleteLayer(
            routeTokenCount: 1,
            productionDefaultPaging: true,
            residencyTarget: .promoteCompleteOnMandatoryRead))
        #expect(!ExpertPageCommitPolicy.shouldCommitMandatoryCompleteLayer(
            routeTokenCount: 2_048,
            productionDefaultPaging: true,
            residencyTarget: .streamOperationLocal))
        // The active plan is the phase's will for the layer: a layer the plan
        // streams operation-local keeps streaming, and a layer the plan is
        // releasing is not refilled behind the plan's back.
        #expect(!ExpertPageCommitPolicy.shouldCommitMandatoryRoutedPage(
            routeTokenCount: 2_048,
            productionDefaultPaging: true,
            residencyTarget: .streamOperationLocal,
            layerHasNoRetainedPage: false))
        #expect(!ExpertPageCommitPolicy.shouldCommitMandatoryRoutedPage(
            routeTokenCount: 1,
            productionDefaultPaging: true,
            residencyTarget: .releasePartial,
            layerHasNoRetainedPage: false))
        #expect(!ExpertPageCommitPolicy.shouldCommitMandatoryRoutedPage(
            routeTokenCount: 1,
            productionDefaultPaging: true,
            residencyTarget: .releaseCompleteForExactDeficit,
            layerHasNoRetainedPage: false))
        #expect(ExpertPageCommitPolicy.shouldCommitMandatoryRoutedPage(
            routeTokenCount: 1,
            productionDefaultPaging: true,
            residencyTarget: .admitPartialOnMandatoryRouteRead,
            layerHasNoRetainedPage: true))
        #expect(ExpertPageCommitPolicy.shouldCommitMandatoryRoutedPage(
            routeTokenCount: 1,
            productionDefaultPaging: true,
            residencyTarget: .preservePartial,
            layerHasNoRetainedPage: false))
        #expect(ExpertPageCommitPolicy.shouldCommitMandatoryRoutedPage(
            routeTokenCount: 1,
            productionDefaultPaging: true,
            residencyTarget: nil,
            layerHasNoRetainedPage: false))
        #expect(!ExpertPageCommitPolicy.shouldCommitMandatoryRoutedPage(
            routeTokenCount: 2_048,
            productionDefaultPaging: false,
            residencyTarget: .streamOperationLocal,
            layerHasNoRetainedPage: true))
        #expect(!ExpertPageCommitPolicy.shouldCommitMandatoryRoutedPage(
            routeTokenCount: 1,
            productionDefaultPaging: false,
            residencyTarget: nil,
            layerHasNoRetainedPage: false))
    }

    @Test
    func should_cap_hot_expert_warm_tables_at_the_layer_expert_count() {
        // Issue #514: leftover decode entitlement is the economic cap, applied
        // by the caller. This policy is only the structural cap.
        #expect(ExpertPageCommitPolicy.hotExpertWarmSlotCount(expertCapacity: 512) == 512)
        #expect(ExpertPageCommitPolicy.hotExpertWarmSlotCount(expertCapacity: 4) == 4)
        #expect(ExpertPageCommitPolicy.hotExpertWarmSlotCount(expertCapacity: 6) == 6)
    }

    @Test
    func should_refuse_the_same_prefill_chunk_retry_when_sparse_experts_are_paged() {
        let residentRetry: ForwardRecoveryRequirements = ForwardRecoveryRequirements(
            stableActiveMemoryBytes: 900,
            activeMemoryBytesAtFailure: 950,
            attemptedAllocationBytes: 100,
            observedTransientHighWaterBytes: 25,
            retainedExpertPayloadBytesBeforeReclamation: 200,
            retainedExpertPayloadBytesAfterReclamation: 100,
            activeMemoryCeilingBytes: 1_000,
            hasAlreadyRetriedAfterReclamation: false,
            sparseExpertsArePaged: false)
        let pagedRetry: ForwardRecoveryRequirements = ForwardRecoveryRequirements(
            stableActiveMemoryBytes: 900,
            activeMemoryBytesAtFailure: 950,
            attemptedAllocationBytes: 100,
            observedTransientHighWaterBytes: 25,
            retainedExpertPayloadBytesBeforeReclamation: 200,
            retainedExpertPayloadBytesAfterReclamation: 100,
            activeMemoryCeilingBytes: 1_000,
            hasAlreadyRetriedAfterReclamation: false,
            sparseExpertsArePaged: true)

        var residentIsRetry: Bool = false
        if case .retry = residentRetry.decide() {
            residentIsRetry = true
        }
        #expect(residentIsRetry)
        var pagedIsReject: Bool = false
        if case .reject = pagedRetry.decide() {
            pagedIsReject = true
        }
        #expect(pagedIsReject)
    }

    @Test
    func should_keep_already_seated_complete_layers_when_leftover_budget_tightens() {
        let seatedCompleteLayerPayloadBytes: UInt64 = 80
        let tighterLeftoverExpertBudgetBytes: UInt64 = 70
        let richerLeftoverExpertBudgetBytes: UInt64 = 90
        #expect(RequestStableResidencyPlanning
            .retainedCompleteLayerCeilingAfterPrefillBudgetRefresh(
                leftoverExpertBudgetBytes: tighterLeftoverExpertBudgetBytes,
                currentCompleteLayerPayloadBytes: seatedCompleteLayerPayloadBytes)
            == seatedCompleteLayerPayloadBytes)
        #expect(RequestStableResidencyPlanning
            .retainedCompleteLayerCeilingAfterPrefillBudgetRefresh(
                leftoverExpertBudgetBytes: richerLeftoverExpertBudgetBytes,
                currentCompleteLayerPayloadBytes: seatedCompleteLayerPayloadBytes)
            == richerLeftoverExpertBudgetBytes)
        #expect(RequestStableResidencyPlanning
            .retainedCompleteLayerCeilingAfterPrefillBudgetRefresh(
                leftoverExpertBudgetBytes: 0, currentCompleteLayerPayloadBytes: 0) == 0)
    }

    @Test
    func should_keep_all_resident_expert_pages_when_learned_budget_tightens_between_requests() {
        let residentCompleteAndWarmTablePayloadBytes: UInt64 = 100
        let tighterLeftoverExpertBudgetBytes: UInt64 = 80
        let richerLeftoverExpertBudgetBytes: UInt64 = 120
        #expect(RequestStableResidencyPlanning.retainedResidentCeilingAfterBudgetRefresh(
            leftoverExpertBudgetBytes: tighterLeftoverExpertBudgetBytes,
            currentResidentPayloadBytes: residentCompleteAndWarmTablePayloadBytes)
            == residentCompleteAndWarmTablePayloadBytes)
        #expect(RequestStableResidencyPlanning.retainedResidentCeilingAfterBudgetRefresh(
            leftoverExpertBudgetBytes: richerLeftoverExpertBudgetBytes,
            currentResidentPayloadBytes: residentCompleteAndWarmTablePayloadBytes)
            == richerLeftoverExpertBudgetBytes)
        #expect(RequestStableResidencyPlanning.retainedResidentCeilingAfterBudgetRefresh(
            leftoverExpertBudgetBytes: 0, currentResidentPayloadBytes: 0) == 0)
    }

    @Test
    func should_plan_prefill_with_the_floored_ceiling_when_seated_layers_exceed_leftover() {
        let layerGeometries: [ExpertLayerGeometry] = ExpertResidencyPolicyTests.uniformGeometry(3)
        let seatedCompleteLayers: [CurrentExpertLayerResidency] = [
            CurrentExpertLayerResidency(
                layerIndex: 0,
                pageClass: .stableCompleteLayer,
                retainedExpertIds: [0, 1, 2, 3],
                payloadBytes: 40,
                coveredWeightedDemand: 0),
            CurrentExpertLayerResidency(
                layerIndex: 1,
                pageClass: .stableCompleteLayer,
                retainedExpertIds: [0, 1, 2, 3],
                payloadBytes: 40,
                coveredWeightedDemand: 0),
        ]
        let leftoverExpertBudgetBytes: UInt64 = 70
        let seatedCompleteLayerPayloadBytes: UInt64 = 80
        #expect(ExpertResidencyPolicyTests.planError(
            phase: .prefill,
            retainedExpertCeilingBytes: leftoverExpertBudgetBytes,
            layerGeometries: layerGeometries,
            currentResidencies: seatedCompleteLayers)
            == .currentResidencyExceedsCeiling)
        let flooredCeilingBytes: UInt64 = RequestStableResidencyPlanning
            .retainedCompleteLayerCeilingAfterPrefillBudgetRefresh(
                leftoverExpertBudgetBytes: leftoverExpertBudgetBytes,
                currentCompleteLayerPayloadBytes: seatedCompleteLayerPayloadBytes)
        #expect(try! ExpertResidencyPlanning.planExpertResidency(
            phase: .prefill,
            retainedExpertCeilingBytes: flooredCeilingBytes,
            layerGeometries: layerGeometries,
            currentResidencies: seatedCompleteLayers).layerTargets.count == 3)
        #expect(try! ExpertResidencyPlanning.planExpertResidency(
            phase: .decode,
            retainedExpertCeilingBytes: flooredCeilingBytes,
            layerGeometries: layerGeometries,
            currentResidencies: seatedCompleteLayers).layerTargets.count == 3)
        #expect(try! ExpertResidencyPlanning.planExpertResidency(
            phase: .generationPreparation,
            retainedExpertCeilingBytes: flooredCeilingBytes,
            layerGeometries: layerGeometries,
            currentResidencies: seatedCompleteLayers).layerTargets.count == 3)
    }

    @Test
    func should_keep_the_opening_prefill_pin_set_when_later_leftover_wants_more_layers() {
        let layerGeometries: [ExpertLayerGeometry] = ExpertResidencyPolicyTests.uniformGeometry(3)
        let openingCandidate: ExpertResidencyPlan = try! ExpertResidencyPlanning
            .planExpertResidency(
                phase: .prefill,
                retainedExpertCeilingBytes: 100,
                layerGeometries: layerGeometries,
                currentResidencies: [])
        let openedPublication: (requestResidency: RequestExpertResidency?,
            plan: ExpertResidencyPlan) =
            RequestStableResidencyPlanning.publishRequestStableResidencyPlan(
                phase: .prefill,
                existingRequestResidency: nil,
                candidatePlan: openingCandidate,
                currentResidencies: [],
                releasedCompletePayloadBytes: 0,
                layerGeometries: layerGeometries)
        let openedResidency: RequestExpertResidency = try! #require(
            openedPublication.requestResidency)
        #expect(openedResidency.layerRole(0) == .pinnedComplete)
        #expect(openedResidency.layerRole(2) == .streamed)
        #expect(openedPublication.plan.layerTargets[2] == .streamOperationLocal)

        let richerCandidate: ExpertResidencyPlan = try! ExpertResidencyPlanning
            .planExpertResidency(
                phase: .prefill,
                retainedExpertCeilingBytes: 120,
                layerGeometries: layerGeometries,
                currentResidencies: [])
        let continuedPublication: (requestResidency: RequestExpertResidency?,
            plan: ExpertResidencyPlan) =
            RequestStableResidencyPlanning.publishRequestStableResidencyPlan(
                phase: .prefill,
                existingRequestResidency: openedResidency,
                candidatePlan: richerCandidate,
                currentResidencies: [],
                releasedCompletePayloadBytes: 0,
                layerGeometries: layerGeometries)

        #expect(continuedPublication.plan.completeLayerTargets == [0, 1])
        #expect(continuedPublication.plan.layerTargets[2] == .streamOperationLocal)
    }

    @Test
    func should_keep_opening_prefill_pins_when_later_leftover_is_tighter_without_capacity_failure() {
        let layerGeometries: [ExpertLayerGeometry] = ExpertResidencyPolicyTests.uniformGeometry(3)
        let openingCandidate: ExpertResidencyPlan = try! ExpertResidencyPlanning
            .planExpertResidency(
                phase: .prefill,
                retainedExpertCeilingBytes: 100,
                layerGeometries: layerGeometries,
                currentResidencies: [])
        let openedResidency: RequestExpertResidency = RequestExpertResidency.openPrefill(
            candidatePlan: openingCandidate)
        #expect(openedResidency.pinnedCompleteLayerIndexes() == [0, 1])

        let tighterCandidate: ExpertResidencyPlan = try! ExpertResidencyPlanning
            .planExpertResidency(
                phase: .prefill,
                retainedExpertCeilingBytes: 70,
                layerGeometries: layerGeometries,
                currentResidencies: [])
        let continuedPublication: (requestResidency: RequestExpertResidency?,
            plan: ExpertResidencyPlan) =
            RequestStableResidencyPlanning.publishRequestStableResidencyPlan(
                phase: .prefill,
                existingRequestResidency: openedResidency,
                candidatePlan: tighterCandidate,
                currentResidencies: [],
                releasedCompletePayloadBytes: 0,
                layerGeometries: layerGeometries)
        let continuedResidency: RequestExpertResidency = try! #require(
            continuedPublication.requestResidency)

        #expect(continuedResidency.pinnedCompleteLayerIndexes() == [0, 1])
        #expect(continuedPublication.plan.completeLayerTargets == [0, 1])
        #expect(continuedPublication.plan.layerTargets[1] == .promoteCompleteOnMandatoryRead)
    }

    @Test
    func should_not_re_pin_a_layer_after_a_prefill_capacity_failure() {
        let layerGeometries: [ExpertLayerGeometry] = ExpertResidencyPolicyTests.uniformGeometry(3)
        let openingCandidate: ExpertResidencyPlan = try! ExpertResidencyPlanning
            .planExpertResidency(
                phase: .prefill,
                retainedExpertCeilingBytes: 100,
                layerGeometries: layerGeometries,
                currentResidencies: [])
        let openedResidency: RequestExpertResidency = RequestExpertResidency.openPrefill(
            candidatePlan: openingCandidate)
        let shrunkResidency: RequestExpertResidency = openedResidency
            .shrinkAfterCapacityFailure(
                requiredReclamationBytes: 40,
                layerGeometries: layerGeometries)
        #expect(shrunkResidency.layerRole(1) == .streamed)

        let richerCandidate: ExpertResidencyPlan = try! ExpertResidencyPlanning
            .planExpertResidency(
                phase: .prefill,
                retainedExpertCeilingBytes: 120,
                layerGeometries: layerGeometries,
                currentResidencies: [])
        let publishedPublication: (requestResidency: RequestExpertResidency?,
            plan: ExpertResidencyPlan) =
            RequestStableResidencyPlanning.publishRequestStableResidencyPlan(
                phase: .prefill,
                existingRequestResidency: shrunkResidency,
                candidatePlan: richerCandidate,
                currentResidencies: [],
                releasedCompletePayloadBytes: 0,
                layerGeometries: layerGeometries)

        #expect(publishedPublication.plan.completeLayerTargets == [0])
        #expect(publishedPublication.plan.layerTargets[1] == .streamOperationLocal)
    }

    @Test
    func should_keep_prefill_routed_pages_when_generation_handoff_replans() {
        let layerGeometries: [ExpertLayerGeometry] = ExpertResidencyPolicyTests.uniformGeometry(3)
        let prefillRoutedPages: [CurrentExpertLayerResidency] = [
            CurrentExpertLayerResidency(
                layerIndex: 0,
                pageClass: .elasticRoutedExperts,
                retainedExpertIds: [0, 1],
                payloadBytes: 20,
                coveredWeightedDemand: 8),
            CurrentExpertLayerResidency(
                layerIndex: 1,
                pageClass: .elasticRoutedExperts,
                retainedExpertIds: [0, 2],
                payloadBytes: 20,
                coveredWeightedDemand: 6),
            CurrentExpertLayerResidency(
                layerIndex: 2,
                pageClass: .elasticRoutedExperts,
                retainedExpertIds: [1, 3],
                payloadBytes: 20,
                coveredWeightedDemand: 4),
        ]
        let generationCandidate: ExpertResidencyPlan = try! ExpertResidencyPlanning
            .planExpertResidency(
                phase: .generationPreparation,
                retainedExpertCeilingBytes: 120,
                layerGeometries: layerGeometries,
                currentResidencies: prefillRoutedPages)
        let generationPublication: (requestResidency: RequestExpertResidency?,
            plan: ExpertResidencyPlan) =
            RequestStableResidencyPlanning.publishRequestStableResidencyPlan(
                phase: .generationPreparation,
                existingRequestResidency: nil,
                candidatePlan: generationCandidate,
                currentResidencies: prefillRoutedPages,
                releasedCompletePayloadBytes: 0,
                layerGeometries: layerGeometries)

        #expect(generationPublication.requestResidency == nil)
        #expect(generationPublication.plan.completeLayerTargets.isEmpty)
        var everyLayerPreservesPartial: Bool = true
        for layerTarget: ExpertLayerResidencyTarget in generationPublication.plan.layerTargets {
            if layerTarget != .preservePartial {
                everyLayerPreservesPartial = false
            }
        }
        #expect(everyLayerPreservesPartial)
    }
}
