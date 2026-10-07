import Foundation

import Testing

import ModelServing

/// Phase-aware expert residency planner journeys, port of
/// crates/model-serving/tests/hermetic/phase_aware_expert_residency.rs:
/// complete foundations win the composed ceiling, generation preserves what
/// prefill already streamed, release order is deterministic, and invalid
/// geometry fails closed.
@Suite
final class PhaseAwareExpertResidencyTests {

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

    private static func partialResidency(
        _ layerIndex: Int,
        _ retainedExpertIds: [Int],
        _ coveredWeightedDemand: UInt64
    ) -> CurrentExpertLayerResidency {
        return CurrentExpertLayerResidency(
            layerIndex: layerIndex,
            pageClass: .elasticRoutedExperts,
            retainedExpertIds: retainedExpertIds,
            payloadBytes: UInt64(retainedExpertIds.count) * 10,
            coveredWeightedDemand: coveredWeightedDemand)
    }

    private static func completeResidency(_ layerIndex: Int) -> CurrentExpertLayerResidency {
        return CurrentExpertLayerResidency(
            layerIndex: layerIndex,
            pageClass: .stableCompleteLayer,
            retainedExpertIds: [0, 1, 2, 3],
            payloadBytes: 40,
            coveredWeightedDemand: 0)
    }

    @Test
    func should_target_every_layer_complete_when_the_composed_budget_fits_the_model() throws {
        let residencyPlan: ExpertResidencyPlan = try ExpertResidencyPlanning.planExpertResidency(
            phase: .prefill,
            retainedExpertCeilingBytes: 120,
            layerGeometries: PhaseAwareExpertResidencyTests.uniformGeometry(3),
            currentResidencies: [])

        #expect(residencyPlan.completeLayerTargets == [0, 1, 2])
        var everyLayerPromotesComplete: Bool = true
        for layerTarget: ExpertLayerResidencyTarget in residencyPlan.layerTargets {
            if layerTarget != .promoteCompleteOnMandatoryRead {
                everyLayerPromotesComplete = false
            }
        }
        #expect(everyLayerPromotesComplete)
        #expect(residencyPlan.maximumNewRetainedBytes == 120)
    }

    @Test
    func should_preserve_existing_complete_layers_before_selecting_new_targets() throws {
        let residencyPlan: ExpertResidencyPlan = try ExpertResidencyPlanning.planExpertResidency(
            phase: .generationPreparation,
            retainedExpertCeilingBytes: 80,
            layerGeometries: PhaseAwareExpertResidencyTests.uniformGeometry(3),
            currentResidencies: [PhaseAwareExpertResidencyTests.completeResidency(1)])

        #expect(residencyPlan.completeLayerTargets == [1])
        #expect(residencyPlan.layerTargets[1] == .preserveComplete)
        #expect(residencyPlan.layerTargets[0] == .admitPartialOnMandatoryRouteRead)
        #expect(residencyPlan.reservedRoutedOverlayBytes == 0)
    }

    @Test
    func should_reserve_one_model_derived_routed_page_for_each_incomplete_layer() throws {
        let residencyPlan: ExpertResidencyPlan = try ExpertResidencyPlanning.planExpertResidency(
            phase: .decode,
            retainedExpertCeilingBytes: 80,
            layerGeometries: PhaseAwareExpertResidencyTests.uniformGeometry(3),
            currentResidencies: [PhaseAwareExpertResidencyTests.completeResidency(0)])

        #expect(residencyPlan.completeLayerTargets == [0])
        #expect(residencyPlan.layerTargets[1] == .admitPartialOnMandatoryRouteRead)
        #expect(residencyPlan.reservedRoutedOverlayBytes == 0)
    }

    @Test
    func should_select_additional_complete_layers_by_incremental_payload_then_layer_index() throws {
        let layerGeometries: [ExpertLayerGeometry] = [
            ExpertLayerGeometry(
                layerIndex: 0,
                completeLayerPayloadBytes: 40,
                expertPayloadBytes: 10,
                expertCapacity: 4,
                expertsPerToken: 2),
            ExpertLayerGeometry(
                layerIndex: 1,
                completeLayerPayloadBytes: 80,
                expertPayloadBytes: 20,
                expertCapacity: 4,
                expertsPerToken: 2),
        ]
        let residencyPlan: ExpertResidencyPlan = try ExpertResidencyPlanning.planExpertResidency(
            phase: .prefill,
            retainedExpertCeilingBytes: 80,
            layerGeometries: layerGeometries,
            currentResidencies: [])

        #expect(residencyPlan.completeLayerTargets == [0])
    }

    @Test
    func should_use_low_budget_partial_mode_when_routed_floors_do_not_fit() throws {
        let residencyPlan: ExpertResidencyPlan = try ExpertResidencyPlanning.planExpertResidency(
            phase: .decode,
            retainedExpertCeilingBytes: 10,
            layerGeometries: PhaseAwareExpertResidencyTests.uniformGeometry(3),
            currentResidencies: [PhaseAwareExpertResidencyTests.partialResidency(1, [2], 4)])

        #expect(residencyPlan.layerTargets == [
            .admitPartialOnMandatoryRouteRead,
            .preservePartial,
            .admitPartialOnMandatoryRouteRead,
        ])
    }

    @Test
    func should_preserve_a_fitting_partial_page_without_exact_set_equality() throws {
        let currentPage: CurrentExpertLayerResidency =
            PhaseAwareExpertResidencyTests.partialResidency(1, [0, 3], 8)
        let residencyPlan: ExpertResidencyPlan = try ExpertResidencyPlanning.planExpertResidency(
            phase: .generationPreparation,
            retainedExpertCeilingBytes: 60,
            layerGeometries: PhaseAwareExpertResidencyTests.uniformGeometry(3),
            currentResidencies: [currentPage])

        #expect(residencyPlan.layerTargets[1] == .preservePartial)
        #expect(residencyPlan.expectedPreservedBytes == 20)
    }

    @Test
    func should_not_plan_eager_io_for_an_empty_partial_layer_without_route_evidence() throws {
        let residencyPlan: ExpertResidencyPlan = try ExpertResidencyPlanning.planExpertResidency(
            phase: .decode,
            retainedExpertCeilingBytes: 60,
            layerGeometries: PhaseAwareExpertResidencyTests.uniformGeometry(3),
            currentResidencies: [])

        var everyLayerAdmitsPartial: Bool = true
        for layerTarget: ExpertLayerResidencyTarget in residencyPlan.layerTargets {
            if layerTarget != .admitPartialOnMandatoryRouteRead {
                everyLayerAdmitsPartial = false
            }
        }
        #expect(everyLayerAdmitsPartial)
        #expect(residencyPlan.completeLayerTargets.isEmpty)
    }

    @Test
    func should_seat_complete_layers_after_empty_demotion_when_leftover_budget_fits_them() throws {
        let residencyPlan: ExpertResidencyPlan = try ExpertResidencyPlanning.planExpertResidency(
            phase: .generationPreparation,
            retainedExpertCeilingBytes: 80,
            layerGeometries: PhaseAwareExpertResidencyTests.uniformGeometry(3),
            currentResidencies: [])

        #expect(!residencyPlan.completeLayerTargets.isEmpty)
        var someLayerPromotesComplete: Bool = false
        for layerTarget: ExpertLayerResidencyTarget in residencyPlan.layerTargets {
            if layerTarget == .promoteCompleteOnMandatoryRead {
                someLayerPromotesComplete = true
            }
        }
        #expect(someLayerPromotesComplete)
    }

    /// Issue #339: a Prefill plan republished after a mid-prefill demotion must
    /// name the complete layers its ceiling admits, and the request-stable
    /// contract must keep naming them. That is what lets the prefill stream
    /// hand each freshly streamed complete layer to retained ownership instead
    /// of dropping it and making decode seating read the identical payload
    /// again.
    @Test
    func should_name_every_admissible_complete_layer_after_a_prefill_demotion() throws {
        let layerGeometries: [ExpertLayerGeometry] =
            PhaseAwareExpertResidencyTests.uniformGeometry(3)
        var completeModelPayloadBytes: UInt64 = 0
        for layerGeometry: ExpertLayerGeometry in layerGeometries {
            completeModelPayloadBytes += layerGeometry.completeLayerPayloadBytes
        }
        let candidatePlan: ExpertResidencyPlan = try ExpertResidencyPlanning.planExpertResidency(
            phase: .prefill,
            retainedExpertCeilingBytes: completeModelPayloadBytes,
            layerGeometries: layerGeometries,
            currentResidencies: [])

        #expect(candidatePlan.completeLayerTargets == [0, 1, 2])
        var everyCandidateTargetPromotes: Bool = true
        for layerTarget: ExpertLayerResidencyTarget in candidatePlan.layerTargets {
            if layerTarget != .promoteCompleteOnMandatoryRead {
                everyCandidateTargetPromotes = false
            }
        }
        #expect(everyCandidateTargetPromotes)

        let openedPublication: (requestResidency: RequestExpertResidency?,
            plan: ExpertResidencyPlan) =
            RequestStableResidencyPlanning.publishRequestStableResidencyPlan(
                phase: .prefill,
                existingRequestResidency: nil,
                candidatePlan: candidatePlan,
                currentResidencies: [],
                releasedCompletePayloadBytes: 0,
                layerGeometries: layerGeometries)
        let openedResidency: RequestExpertResidency = try #require(
            openedPublication.requestResidency)

        #expect(openedResidency.pinnedCompleteLayerIndexes() == [0, 1, 2])
        var everyOpenedTargetPromotes: Bool = true
        for layerTarget: ExpertLayerResidencyTarget in openedPublication.plan.layerTargets {
            if layerTarget != .promoteCompleteOnMandatoryRead {
                everyOpenedTargetPromotes = false
            }
        }
        #expect(everyOpenedTargetPromotes)
        var everyOpenedTargetCommits: Bool = true
        for layerTarget: ExpertLayerResidencyTarget in openedPublication.plan.layerTargets {
            if !ExpertPageCommitPolicy.shouldCommitMandatoryCompleteLayer(
                routeTokenCount: 2_048,
                productionDefaultPaging: true,
                residencyTarget: layerTarget) {
                everyOpenedTargetCommits = false
            }
        }
        #expect(everyOpenedTargetCommits)
    }

    @Test
    func should_keep_most_complete_layers_when_leftover_is_slightly_under_the_full_model() throws {
        var layerGeometries: [ExpertLayerGeometry] = []
        layerGeometries.reserveCapacity(40)
        for layerIndex: Int in 0..<40 {
            let completeLayerPayloadBytes: UInt64 =
                (layerIndex == 0 || layerIndex == 1 || layerIndex == 39)
                    ? 1_610_612_736 : 452_984_832
            layerGeometries.append(ExpertLayerGeometry(
                layerIndex: layerIndex,
                completeLayerPayloadBytes: completeLayerPayloadBytes,
                expertPayloadBytes: completeLayerPayloadBytes / 128,
                expertCapacity: 128,
                expertsPerToken: 8))
        }
        let leftoverExpertBudgetBytes: UInt64 = 21_051_596_626
        let residencyPlan: ExpertResidencyPlan = try ExpertResidencyPlanning.planExpertResidency(
            phase: .generationPreparation,
            retainedExpertCeilingBytes: leftoverExpertBudgetBytes,
            layerGeometries: layerGeometries,
            currentResidencies: [])

        #expect(residencyPlan.completeLayerTargets.count >= 30)
        #expect(residencyPlan.completeLayerTargets.count < 40)
    }

    @Test
    func should_require_unseated_complete_layers_to_be_loaded_before_decode() throws {
        let residencyPlan: ExpertResidencyPlan = try ExpertResidencyPlanning.planExpertResidency(
            phase: .generationPreparation,
            retainedExpertCeilingBytes: 80,
            layerGeometries: PhaseAwareExpertResidencyTests.uniformGeometry(3),
            currentResidencies: [])
        let layerIndexes: [Int] = DecodeSeating.completeLayerIndexesRequiredBeforeDecode(
            plan: residencyPlan)

        #expect(layerIndexes == residencyPlan.completeLayerTargets)
        #expect(!layerIndexes.isEmpty)
    }

    @Test
    func should_not_require_already_preserved_complete_layers_to_be_loaded_again() throws {
        let residencyPlan: ExpertResidencyPlan = try ExpertResidencyPlanning.planExpertResidency(
            phase: .generationPreparation,
            retainedExpertCeilingBytes: 80,
            layerGeometries: PhaseAwareExpertResidencyTests.uniformGeometry(3),
            currentResidencies: [PhaseAwareExpertResidencyTests.completeResidency(1)])
        let layerIndexes: [Int] = DecodeSeating.completeLayerIndexesRequiredBeforeDecode(
            plan: residencyPlan)

        #expect(!layerIndexes.contains(1))
    }

    @Test
    func should_release_low_coverage_partial_pages_before_any_complete_layer() throws {
        let currentResidencies: [CurrentExpertLayerResidency] = [
            PhaseAwareExpertResidencyTests.partialResidency(0, [0], 10),
            PhaseAwareExpertResidencyTests.partialResidency(1, [1], 1),
            PhaseAwareExpertResidencyTests.completeResidency(2),
        ]
        let residencyPlan: ExpertResidencyPlan = try ExpertResidencyPlanning.planExpertResidency(
            phase: .idle,
            retainedExpertCeilingBytes: 60,
            layerGeometries: PhaseAwareExpertResidencyTests.uniformGeometry(3),
            currentResidencies: currentResidencies)

        #expect(residencyPlan.deterministicReleaseOrder == [1, 0, 2])
    }

    @Test
    func should_release_complete_layers_only_for_the_remaining_exact_deficit() throws {
        let residencyPlan: ExpertResidencyPlan = try ExpertResidencyPlanning.planExpertResidency(
            phase: .prefill,
            retainedExpertCeilingBytes: 60,
            layerGeometries: PhaseAwareExpertResidencyTests.uniformGeometry(3),
            currentResidencies: [PhaseAwareExpertResidencyTests.completeResidency(2)])

        #expect(residencyPlan.completeLayerTargets.isEmpty)
        #expect(residencyPlan.layerTargets[2] == .releaseCompleteForExactDeficit)
    }

    @Test
    func should_fail_closed_for_invalid_geometry_residency_and_overflow() throws {
        let duplicateResidencies: [CurrentExpertLayerResidency] = [
            PhaseAwareExpertResidencyTests.completeResidency(0),
            PhaseAwareExpertResidencyTests.completeResidency(0),
        ]
        let duplicateLayerError: ExpertResidencyPlanError? = PhaseAwareExpertResidencyTests
            .planError(
                phase: .prefill,
                retainedExpertCeilingBytes: 80,
                layerGeometries: PhaseAwareExpertResidencyTests.uniformGeometry(2),
                currentResidencies: duplicateResidencies)
        var duplicateLayerErrorMatches: Bool = false
        if case .duplicateOrUnorderedCurrentLayer = duplicateLayerError {
            duplicateLayerErrorMatches = true
        }
        #expect(duplicateLayerErrorMatches)

        var zeroLayerGeometries: [ExpertLayerGeometry] =
            PhaseAwareExpertResidencyTests.uniformGeometry(1)
        zeroLayerGeometries[0] = ExpertLayerGeometry(
            layerIndex: 0,
            completeLayerPayloadBytes: 40,
            expertPayloadBytes: 10,
            expertCapacity: 0,
            expertsPerToken: 2)
        #expect(PhaseAwareExpertResidencyTests.planError(
            phase: .prefill,
            retainedExpertCeilingBytes: 0,
            layerGeometries: zeroLayerGeometries,
            currentResidencies: [])
            == .zeroGeometry(layerIndex: 0))

        let overflowingGeometry: [ExpertLayerGeometry] = [ExpertLayerGeometry(
            layerIndex: 0,
            completeLayerPayloadBytes: UInt64.max,
            expertPayloadBytes: UInt64.max,
            expertCapacity: 2,
            expertsPerToken: 1)]
        #expect(PhaseAwareExpertResidencyTests.planError(
            phase: .prefill,
            retainedExpertCeilingBytes: UInt64.max,
            layerGeometries: overflowingGeometry,
            currentResidencies: [])
            == .byteCountOverflow)
    }

    @Test
    func should_keep_every_planned_owner_and_reservation_within_the_retained_budget() throws {
        let layerGeometries: [ExpertLayerGeometry] =
            PhaseAwareExpertResidencyTests.uniformGeometry(4)
        let residencyPlan: ExpertResidencyPlan = try ExpertResidencyPlanning.planExpertResidency(
            phase: .decode,
            retainedExpertCeilingBytes: 100,
            layerGeometries: layerGeometries,
            currentResidencies: [PhaseAwareExpertResidencyTests.partialResidency(3, [0, 2], 12)])
        let completeTargetBytes: UInt64 = UInt64(residencyPlan.completeLayerTargets.count) * 40

        #expect(completeTargetBytes + residencyPlan.reservedRoutedOverlayBytes
            <= residencyPlan.retainedExpertCeilingBytes)
    }

    /// Attempts one plan and returns its typed error, or nil when planning
    /// succeeded; keeps the fail-closed assertions readable.
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
}
