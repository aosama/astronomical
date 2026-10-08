import Foundation

import Testing

import ModelServing

/// Hermetic MLX RAM budget journeys, port of
/// crates/model-serving/tests/hermetic/mlx_ram_budget.rs: the owner composes
/// retained-expert budget from the ceiling minus non-overlapping fixed owners,
/// learns conservative high-water evidence that never shrinks, and resolves
/// the activation promise from the planned operation rather than the prompt.
@Suite
final class MlxRamBudgetTests {

    private func fableClassGeometry() -> MlxRamBudgetModelGeometry {
        MlxRamBudgetModelGeometry(
            modelCorePayloadBytes: 2_360_000_000,
            completeExpertPayloadBytes: 36_238_786_560,
            largestCompleteExpertLayerBytes: 905_969_664,
            largestRoutedExpertPageBytes: 28_311_552,
            sequenceStateBytesPerToken: 0)
    }

    private func unmeasuredPrefillActivationHeadroomBytes(
        _ geometry: MlxRamBudgetModelGeometry
    ) -> UInt64 {
        geometry.largestCompleteExpertLayerBytes * 3
    }

    /// A geometry whose three-layer static floor is small enough that learned
    /// activation evidence dominates the promise, making context-bucket
    /// resolution observable (issue #623 follow-up).
    private func smallLayerGeometry() -> MlxRamBudgetModelGeometry {
        MlxRamBudgetModelGeometry(
            modelCorePayloadBytes: 2_360_000_000,
            completeExpertPayloadBytes: 36_238_786_560,
            largestCompleteExpertLayerBytes: 10_000_000,
            largestRoutedExpertPageBytes: 28_311_552,
            sequenceStateBytesPerToken: 0)
    }

    @Test
    func should_bootstrap_context_window_reserve_at_one_gigabyte_before_measurements() throws {
        let mlxRamBudget = try MlxRamBudget(
            mlxActiveMemoryCeilingBytes: 39_000_000_000,
            modelGeometry: fableClassGeometry())

        #expect(MlxRamBudget.defaultBootstrapContextWindowReserveBytes == 1_000_000_000)
        #expect(mlxRamBudget.contextWindowReserveBytes(0) == 1_000_000_000)
        #expect(mlxRamBudget.contextWindowReserveBytes(4_096) == 1_000_000_000)
        #expect(!mlxRamBudget.hasContextWindowMeasurement)
        #expect(mlxRamBudget.activationHeadroomBytes(.prefill, 4_096)
            == unmeasuredPrefillActivationHeadroomBytes(fableClassGeometry()))
    }

    @Test
    func should_compose_retained_expert_budget_from_ceiling_minus_fixed_owners() throws {
        let mlxRamBudget = try MlxRamBudget(
            mlxActiveMemoryCeilingBytes: 39_000_000_000,
            modelGeometry: fableClassGeometry())

        let plannedBudget = mlxRamBudget.plan(
            phase: .prefill, contextTokenCount: 4_096, operationTokenCount: 4_096,
            otherFixedBytes: 0)

        let expectedActivationHeadroomBytes =
            unmeasuredPrefillActivationHeadroomBytes(fableClassGeometry())
        let expectedFixedOwnerBytes: UInt64 = 2_360_000_000 + 1_000_000_000
            + expectedActivationHeadroomBytes + 905_969_664
        #expect(plannedBudget.contextWindowReserveBytes == 1_000_000_000)
        #expect(plannedBudget.activationHeadroomBytes == expectedActivationHeadroomBytes)
        #expect(plannedBudget.completeLayerStreamSlotBytes == 905_969_664)
        #expect(plannedBudget.retainedExpertBudgetBytes
            == 39_000_000_000 - expectedFixedOwnerBytes)
    }

    @Test
    func should_raise_context_window_reserve_from_measurements_and_never_under_shoot() throws {
        let mlxRamBudget = try MlxRamBudget(
            mlxActiveMemoryCeilingBytes: 39_000_000_000,
            modelGeometry: fableClassGeometry())

        mlxRamBudget.recordMeasurement(MlxRamBudgetMeasurement(
            phase: .prefill,
            contextTokenCount: 2_048,
            measuredContextAndActivationBytes: 1_500_000_000,
            observedActivationHeadroomBytes: 400_000_000,
            exactTemporaryWorkspaceBytes: 0))
        mlxRamBudget.recordMeasurement(MlxRamBudgetMeasurement(
            phase: .prefill,
            contextTokenCount: 4_096,
            measuredContextAndActivationBytes: 2_200_000_000,
            observedActivationHeadroomBytes: 700_000_000,
            exactTemporaryWorkspaceBytes: 0))

        #expect(mlxRamBudget.hasContextWindowMeasurement)
        let contextWindowReserveFor2048 = mlxRamBudget.contextWindowReserveBytes(2_048)
        let contextWindowReserveFor4096 = mlxRamBudget.contextWindowReserveBytes(4_096)
        #expect(contextWindowReserveFor2048 >= 1_100_000_000)
        #expect(contextWindowReserveFor4096 >= contextWindowReserveFor2048)
        #expect(contextWindowReserveFor4096 >= 1_500_000_000)

        let plannedBudget = mlxRamBudget.plan(
            phase: .prefill, contextTokenCount: 4_096, operationTokenCount: 4_096,
            otherFixedBytes: 0)
        #expect(plannedBudget.activationHeadroomBytes
            == max(unmeasuredPrefillActivationHeadroomBytes(fableClassGeometry()), 700_000_000))
        // Experts must not be budgeted into the learned context-window /
        // activation reserve.
        let fixedNonExpertBytes: UInt64 = plannedBudget.modelCorePayloadBytes
            + plannedBudget.contextWindowReserveBytes
            + plannedBudget.activationHeadroomBytes
            + plannedBudget.completeLayerStreamSlotBytes
        #expect(plannedBudget.retainedExpertBudgetBytes
            == plannedBudget.mlxActiveMemoryCeilingBytes - fixedNonExpertBytes)
    }

    @Test
    func should_not_charge_transient_workspace_as_both_context_and_activation() throws {
        let mlxRamBudget = try MlxRamBudget(
            mlxActiveMemoryCeilingBytes: 23_000_000_000,
            modelGeometry: fableClassGeometry())
        mlxRamBudget.recordMeasurement(MlxRamBudgetMeasurement(
            phase: .prefill,
            contextTokenCount: 7_000,
            measuredContextAndActivationBytes: 2_100_000_000,
            observedActivationHeadroomBytes: 2_000_000_000,
            exactTemporaryWorkspaceBytes: 100_000_000))

        let plannedBudget = mlxRamBudget.plan(
            phase: .prefill, contextTokenCount: 7_000, operationTokenCount: 7_000,
            otherFixedBytes: 0)

        #expect(plannedBudget.contextWindowReserveBytes == 1_000_000_000)
        #expect(plannedBudget.activationHeadroomBytes
            == max(unmeasuredPrefillActivationHeadroomBytes(fableClassGeometry()), 2_000_000_000))
    }

    @Test
    func should_not_charge_exact_temporary_workspace_as_persistent_context() throws {
        let mlxRamBudget = try MlxRamBudget(
            mlxActiveMemoryCeilingBytes: 23_000_000_000,
            modelGeometry: fableClassGeometry())
        mlxRamBudget.recordMeasurement(MlxRamBudgetMeasurement(
            phase: .prefill,
            contextTokenCount: 7_000,
            measuredContextAndActivationBytes: 3_000_000_000,
            observedActivationHeadroomBytes: 500_000_000,
            exactTemporaryWorkspaceBytes: 500_000_000))

        #expect(mlxRamBudget.contextWindowReserveBytes(7_000) == 2_064_000_000)
    }

    @Test
    func should_protect_the_first_decode_with_prefill_activation_evidence() throws {
        let mlxRamBudget = try MlxRamBudget(
            mlxActiveMemoryCeilingBytes: 23_000_000_000,
            modelGeometry: fableClassGeometry())
        mlxRamBudget.recordMeasurement(MlxRamBudgetMeasurement(
            phase: .prefill,
            contextTokenCount: 7_000,
            measuredContextAndActivationBytes: 2_000_000_000,
            observedActivationHeadroomBytes: 1_900_000_000,
            exactTemporaryWorkspaceBytes: 0))

        let firstDecodePlan = mlxRamBudget.plan(
            phase: .decode, contextTokenCount: 7_000, operationTokenCount: 1,
            otherFixedBytes: 0)

        // Before decode has its own evidence, retaining experts into
        // prefill-proven transient space would force immediate reclamation on
        // the first token.
        #expect(firstDecodePlan.activationHeadroomBytes == 1_900_000_000)
    }

    @Test
    func should_use_decode_activation_evidence_after_decode_is_observed() throws {
        let mlxRamBudget = try MlxRamBudget(
            mlxActiveMemoryCeilingBytes: 23_000_000_000,
            modelGeometry: fableClassGeometry())
        mlxRamBudget.recordMeasurement(MlxRamBudgetMeasurement(
            phase: .prefill,
            contextTokenCount: 7_000,
            measuredContextAndActivationBytes: 2_000_000_000,
            observedActivationHeadroomBytes: 1_900_000_000,
            exactTemporaryWorkspaceBytes: 0))
        mlxRamBudget.recordMeasurement(MlxRamBudgetMeasurement(
            phase: .decode,
            contextTokenCount: 7_000,
            measuredContextAndActivationBytes: 500_000_000,
            observedActivationHeadroomBytes: 400_000_000,
            exactTemporaryWorkspaceBytes: 0))

        let learnedDecodePlan = mlxRamBudget.plan(
            phase: .decode, contextTokenCount: 7_000, operationTokenCount: 1,
            otherFixedBytes: 0)

        #expect(learnedDecodePlan.activationHeadroomBytes == 400_000_000)
    }

    @Test
    func should_limit_retained_experts_to_leave_the_exact_admitted_prefill_reserve() throws {
        let mlxRamBudget = try MlxRamBudget(
            mlxActiveMemoryCeilingBytes: 38_000_000_000,
            modelGeometry: fableClassGeometry())
        let currentActiveMemoryBytes: UInt64 = 35_879_800_070
        let currentRetainedExpertPayloadBytes: UInt64 = 33_520_877_568
        // Production evidence showed 1.554 GB of context/activation growth
        // followed by one exact 905,969,664-byte complete-layer allocation.
        let admittedForwardReserveBytes: UInt64 = 2_460_246_024

        let retainedExpertBudgetBytes =
            mlxRamBudget.retainedExpertBudgetForAdmittedForward(
                currentActiveMemoryBytes: currentActiveMemoryBytes,
                currentRetainedExpertPayloadBytes: currentRetainedExpertPayloadBytes,
                admittedForwardReserveBytes: admittedForwardReserveBytes)

        #expect(retainedExpertBudgetBytes == 33_180_831_474)
        #expect(retainedExpertBudgetBytes < currentRetainedExpertPayloadBytes)
        let nonExpertActiveBytes: UInt64 =
            currentActiveMemoryBytes - currentRetainedExpertPayloadBytes
        let composedTotalBytes: UInt64 = nonExpertActiveBytes
            + retainedExpertBudgetBytes + admittedForwardReserveBytes
        #expect(composedTotalBytes == 38_000_000_000)
    }

    @Test
    func should_project_unmeasured_suffix_tokens_on_top_of_learned_context_reserve() throws {
        let geometry = MlxRamBudgetModelGeometry(
            modelCorePayloadBytes: fableClassGeometry().modelCorePayloadBytes,
            completeExpertPayloadBytes: fableClassGeometry().completeExpertPayloadBytes,
            largestCompleteExpertLayerBytes: fableClassGeometry().largestCompleteExpertLayerBytes,
            largestRoutedExpertPageBytes: fableClassGeometry().largestRoutedExpertPageBytes,
            sequenceStateBytesPerToken: 20_000)
        let mlxRamBudget = try MlxRamBudget(
            mlxActiveMemoryCeilingBytes: 39_000_000_000,
            modelGeometry: geometry)
        mlxRamBudget.recordMeasurement(MlxRamBudgetMeasurement(
            phase: .prefill,
            contextTokenCount: 10_000,
            measuredContextAndActivationBytes: 400_000_000,
            observedActivationHeadroomBytes: 100_000_000,
            exactTemporaryWorkspaceBytes: 0))

        let reservedForMeasuredBucket = mlxRamBudget.contextWindowReserveBytes(10_000)
        let reservedForUnmeasuredSuffix = mlxRamBudget.contextWindowReserveBytes(26_000)

        #expect(reservedForMeasuredBucket == 1_000_000_000)
        let highestMeasuredTokenCount: UInt64 = (10_000 / 1_024 + 1) * 1_024
        let unmeasuredTokenCount: UInt64 = 26_000 - highestMeasuredTokenCount
        let expectedUnmeasuredSuffixBytes: UInt64 = unmeasuredTokenCount * 20_000
        #expect(reservedForUnmeasuredSuffix
            == reservedForMeasuredBucket + expectedUnmeasuredSuffixBytes)
    }

    @Test
    func should_exclude_newly_retained_experts_from_context_and_activation_learning() {
        #expect(MlxRamBudget.measuredNonExpertForwardGrowthBytes(
            activeMemoryBytesBeforeGrowth: 2_358_922_508,
            peakMemoryBytesDuringGrowth: 36_867_500_000,
            retainedExpertPayloadBytesBeforeGrowth: 0,
            retainedExpertPayloadBytesAfterGrowth: 33_520_877_568) == 987_699_924)
        #expect(MlxRamBudget.measuredNonExpertForwardGrowthBytes(
            activeMemoryBytesBeforeGrowth: 3_000,
            peakMemoryBytesDuringGrowth: 4_500,
            retainedExpertPayloadBytesBeforeGrowth: 2_000,
            retainedExpertPayloadBytesAfterGrowth: 2_000) == 1_500)
    }

    // Issue #623 follow-up: the prefill activation promise was one grow-only
    // session high-water pooled across every context size, so one
    // large-context request's workspace sized the promise shown to users and
    // the retention ceiling offered to paged experts for every later request.
    // The promise is now resolved per planned operation (issue #644 sharpened
    // the scope: the planned token count is the operation's own size, never
    // the prompt length): highest evidence at or below the operation bucket
    // within the measured span, proportionally projected beyond it, with the
    // static three-layer floor still bounding everything.

    @Test
    func should_resolve_the_prefill_activation_promise_from_the_planned_operation_bucket()
        throws {
        let mlxRamBudget = try MlxRamBudget(
            mlxActiveMemoryCeilingBytes: 39_000_000_000,
            modelGeometry: smallLayerGeometry())
        mlxRamBudget.recordMeasurement(MlxRamBudgetMeasurement(
            phase: .prefill,
            contextTokenCount: 2_048,
            measuredContextAndActivationBytes: 1_500_000_000,
            observedActivationHeadroomBytes: 400_000_000,
            exactTemporaryWorkspaceBytes: 0))
        mlxRamBudget.recordMeasurement(MlxRamBudgetMeasurement(
            phase: .prefill,
            contextTokenCount: 4_096,
            measuredContextAndActivationBytes: 2_200_000_000,
            observedActivationHeadroomBytes: 700_000_000,
            exactTemporaryWorkspaceBytes: 0))

        let planFor2048 = mlxRamBudget.plan(
            phase: .prefill, contextTokenCount: 2_048, operationTokenCount: 2_048,
            otherFixedBytes: 0)
        let planFor4096 = mlxRamBudget.plan(
            phase: .prefill, contextTokenCount: 4_096, operationTokenCount: 4_096,
            otherFixedBytes: 0)

        #expect(planFor2048.activationHeadroomBytes == 400_000_000)
        #expect(planFor4096.activationHeadroomBytes == 700_000_000)
    }

    @Test
    func should_project_the_prefill_activation_promise_beyond_the_highest_measured_operation()
        throws {
        let mlxRamBudget = try MlxRamBudget(
            mlxActiveMemoryCeilingBytes: 39_000_000_000,
            modelGeometry: smallLayerGeometry())
        mlxRamBudget.recordMeasurement(MlxRamBudgetMeasurement(
            phase: .prefill,
            contextTokenCount: 4_096,
            measuredContextAndActivationBytes: 2_200_000_000,
            observedActivationHeadroomBytes: 700_000_000,
            exactTemporaryWorkspaceBytes: 0))

        // Beyond the highest measured operation token count (5,120 for bucket
        // 4) the highest evidence scales proportionally:
        // 700 MB × 8,192 / 5,120 = 1,120 MB. This is the honest linear prior
        // for operations larger than anything measured; chunked prefill never
        // reaches it because every chunk is bounded by the configured chunk
        // size.
        let planFor8192 = mlxRamBudget.plan(
            phase: .prefill, contextTokenCount: 8_192, operationTokenCount: 8_192,
            otherFixedBytes: 0)

        #expect(planFor8192.activationHeadroomBytes == 1_120_000_000)
    }

    @Test
    func should_keep_the_prefill_activation_promise_independent_of_the_prompt_context()
        throws {
        // Issue #644 regression: the activation reserve is operation-scoped.
        // The production failure planned a 50,400-token prompt's activation
        // from a per-chunk observation, multiplying it ~49x into a reserve
        // several times the ceiling. The prompt length must size the context
        // reserve only; the activation reserve must not move with it.
        let geometry = MlxRamBudgetModelGeometry(
            modelCorePayloadBytes: smallLayerGeometry().modelCorePayloadBytes,
            completeExpertPayloadBytes: smallLayerGeometry().completeExpertPayloadBytes,
            largestCompleteExpertLayerBytes: smallLayerGeometry().largestCompleteExpertLayerBytes,
            largestRoutedExpertPageBytes: smallLayerGeometry().largestRoutedExpertPageBytes,
            sequenceStateBytesPerToken: 24_000)
        let mlxRamBudget = try MlxRamBudget(
            mlxActiveMemoryCeilingBytes: 39_000_000_000,
            modelGeometry: geometry)
        mlxRamBudget.recordMeasurement(MlxRamBudgetMeasurement(
            phase: .prefill,
            contextTokenCount: 2_048,
            measuredContextAndActivationBytes: 1_500_000_000,
            observedActivationHeadroomBytes: 400_000_000,
            exactTemporaryWorkspaceBytes: 0))

        let planForChunk = mlxRamBudget.plan(
            phase: .prefill, contextTokenCount: 50_400, operationTokenCount: 2_048,
            otherFixedBytes: 0)
        let planForSmall = mlxRamBudget.plan(
            phase: .prefill, contextTokenCount: 2_048, operationTokenCount: 2_048,
            otherFixedBytes: 0)

        #expect(planForChunk.activationHeadroomBytes == planForSmall.activationHeadroomBytes)
        #expect(planForChunk.contextWindowReserveBytes
            > planForSmall.contextWindowReserveBytes)
    }

    @Test
    func should_keep_the_first_decode_protected_by_the_largest_prefill_evidence() throws {
        let mlxRamBudget = try MlxRamBudget(
            mlxActiveMemoryCeilingBytes: 39_000_000_000,
            modelGeometry: smallLayerGeometry())
        mlxRamBudget.recordMeasurement(MlxRamBudgetMeasurement(
            phase: .prefill,
            contextTokenCount: 2_048,
            measuredContextAndActivationBytes: 1_500_000_000,
            observedActivationHeadroomBytes: 400_000_000,
            exactTemporaryWorkspaceBytes: 0))
        mlxRamBudget.recordMeasurement(MlxRamBudgetMeasurement(
            phase: .prefill,
            contextTokenCount: 4_096,
            measuredContextAndActivationBytes: 2_200_000_000,
            observedActivationHeadroomBytes: 700_000_000,
            exactTemporaryWorkspaceBytes: 0))

        // Before decode has its own evidence, the largest prefill observation
        // — not the smaller bucket the next request happens to plan — protects
        // the first decode from warm fill occupying transient space.
        let firstDecodePlan = mlxRamBudget.plan(
            phase: .decode, contextTokenCount: 2_048, operationTokenCount: 1,
            otherFixedBytes: 0)

        #expect(firstDecodePlan.activationHeadroomBytes == 700_000_000)
    }

    // Issue #690: the request-admission path composes the generation-context
    // workspace reservation from activation_headroom_bytes(Prefill,
    // total_context_tokens) — bypassing the operation-scoped plan() that the
    // #644 fix corrected. One successful chunked prefill teaches the budget a
    // chunk-sized activation observation; the next request scales that
    // observation proportionally by total context and admission rejects
    // everything with a paper reserve several times the ceiling (measured
    // 143 GB against a 39 GB ceiling in production). The admission-side
    // composition must resolve the activation reserve at the operation scope
    // exactly like plan() does.
    @Test
    func should_keep_the_admission_workspace_activation_reserve_independent_of_the_prompt_context()
        throws {
        // The learned evidence is chunk-shaped (2,048-token prefill chunks),
        // and the geometry's small layer floor lets learned evidence dominate
        // so the scaling defect is observable. Real per-token state bytes make
        // the context reserve distinguishable across the two context sizes.
        let geometry = MlxRamBudgetModelGeometry(
            modelCorePayloadBytes: smallLayerGeometry().modelCorePayloadBytes,
            completeExpertPayloadBytes: smallLayerGeometry().completeExpertPayloadBytes,
            largestCompleteExpertLayerBytes: smallLayerGeometry().largestCompleteExpertLayerBytes,
            largestRoutedExpertPageBytes: smallLayerGeometry().largestRoutedExpertPageBytes,
            sequenceStateBytesPerToken: 24_000)
        let mlxRamBudget = try MlxRamBudget(
            mlxActiveMemoryCeilingBytes: 39_000_000_000,
            modelGeometry: geometry)
        mlxRamBudget.recordMeasurement(MlxRamBudgetMeasurement(
            phase: .prefill,
            contextTokenCount: 2_048,
            measuredContextAndActivationBytes: 1_500_000_000,
            observedActivationHeadroomBytes: 400_000_000,
            exactTemporaryWorkspaceBytes: 0))

        // A follow-up turn of the same conversation: total context grew by a
        // few hundred tokens, prompt is 18k, and the admission path composes
        // the workspace from the planned chunk operation bound (2,048-token
        // chunks), matching the real admission call site.
        let admissionWorkspace = mlxRamBudget.contextAdmissionWorkspaceSnapshot(
            totalContextTokens: 38_814,
            plannedPrefillOperationTokenCount: 2_048,
            restoreOverlapWorkspaceBytes: 0,
            directPublicationWorkspaceBytes: 0)
        let smallRequestWorkspace = mlxRamBudget.contextAdmissionWorkspaceSnapshot(
            totalContextTokens: 2_048,
            plannedPrefillOperationTokenCount: 2_048,
            restoreOverlapWorkspaceBytes: 0,
            directPublicationWorkspaceBytes: 0)

        #expect(admissionWorkspace.activationHeadroomBytes
            == smallRequestWorkspace.activationHeadroomBytes)
        // The activation reserve must never exceed the active-memory ceiling:
        // measured evidence keeps its pinned dominance over the static floor,
        // but no projection may manufacture a reserve the ceiling could never
        // grant.
        #expect(admissionWorkspace.activationHeadroomBytes
            <= admissionWorkspace.mlxActiveMemoryCeilingBytes)
        // The context reserve must still grow with the prompt context.
        #expect(admissionWorkspace.contextWindowReserveBytes
            > smallRequestWorkspace.contextWindowReserveBytes)
    }

    // Issue #690: no projection — measured or scaled — may manufacture a
    // reserve the ceiling could never grant. Learned evidence keeps its pinned
    // dominance over the static floor, but the activation reserve is
    // structurally capped at the active-memory ceiling.
    @Test
    func should_cap_the_activation_reserve_at_the_active_memory_ceiling() throws {
        let ceilingBytes: UInt64 = 39_000_000_000
        let mlxRamBudget = try MlxRamBudget(
            mlxActiveMemoryCeilingBytes: ceilingBytes,
            modelGeometry: smallLayerGeometry())
        mlxRamBudget.recordMeasurement(MlxRamBudgetMeasurement(
            phase: .prefill,
            contextTokenCount: 2_048,
            measuredContextAndActivationBytes: 1_500_000_000,
            observedActivationHeadroomBytes: ceilingBytes * 10,
            exactTemporaryWorkspaceBytes: 0))

        let plan = mlxRamBudget.plan(
            phase: .prefill, contextTokenCount: 8_192, operationTokenCount: 8_192,
            otherFixedBytes: 0)

        #expect(plan.activationHeadroomBytes == ceilingBytes)
    }

    @Test
    func should_exclude_mandatory_expert_page_streaming_from_the_learned_context_growth() {
        // Issue #691: the composed context/activation learning sees a peak
        // that includes mandatory streamed page traffic. With a zero resident
        // delta the whole stream must leave the residual; the pre-fix formula
        // would charge it to context and activation.
        let learnedGrowthBytes =
            MlxRamBudget.measuredNonExpertForwardGrowthBytesExcludingExpertPageStreaming(
                activeMemoryBytesBeforeGrowth: 1_000,
                peakMemoryBytesDuringGrowth: 1_600,
                retainedExpertPayloadBytesBeforeGrowth: 500,
                retainedExpertPayloadBytesAfterGrowth: 500,
                promotedExpertPageStreamBytes: 400)

        #expect(learnedGrowthBytes == 200)
    }

    @Test
    func should_charge_retained_expert_growth_as_one_beyond_baseline_owner_while_streaming() {
        // Retained (300) and streamed (500) promotion overlap: the peak's
        // expert component beyond the pre-growth baseline is the larger of the
        // two, so the residual keeps only the genuine non-expert growth.
        let learnedGrowthBytes =
            MlxRamBudget.measuredNonExpertForwardGrowthBytesExcludingExpertPageStreaming(
                activeMemoryBytesBeforeGrowth: 1_000,
                peakMemoryBytesDuringGrowth: 1_700,
                retainedExpertPayloadBytesBeforeGrowth: 1_000,
                retainedExpertPayloadBytesAfterGrowth: 1_300,
                promotedExpertPageStreamBytes: 500)

        #expect(learnedGrowthBytes == 200)
    }

    @Test
    func should_match_the_stream_free_contract_when_no_expert_page_stream_occurred() {
        // Identical measurements to the pre-#691 formula when zero pages
        // streamed: the pinned #623/#644 contracts must keep holding byte for
        // byte.
        let learnedGrowthBytes =
            MlxRamBudget.measuredNonExpertForwardGrowthBytesExcludingExpertPageStreaming(
                activeMemoryBytesBeforeGrowth: 1_000,
                peakMemoryBytesDuringGrowth: 2_600,
                retainedExpertPayloadBytesBeforeGrowth: 400,
                retainedExpertPayloadBytesAfterGrowth: 900,
                promotedExpertPageStreamBytes: 0)
        let baselineGrowthBytes = MlxRamBudget.measuredNonExpertForwardGrowthBytes(
            activeMemoryBytesBeforeGrowth: 1_000,
            peakMemoryBytesDuringGrowth: 2_600,
            retainedExpertPayloadBytesBeforeGrowth: 400,
            retainedExpertPayloadBytesAfterGrowth: 900)

        #expect(learnedGrowthBytes == baselineGrowthBytes)
        #expect(learnedGrowthBytes == 1_100)
    }

    @Test
    func should_saturate_context_growth_to_zero_when_stream_evidence_dominates() {
        // Churn can make stream evidence exceed the whole recorded window; the
        // budget must clamp instead of wrapping (fail safe, recoverable by
        // design).
        let learnedGrowthBytes =
            MlxRamBudget.measuredNonExpertForwardGrowthBytesExcludingExpertPageStreaming(
                activeMemoryBytesBeforeGrowth: 1_000,
                peakMemoryBytesDuringGrowth: 1_300,
                retainedExpertPayloadBytesBeforeGrowth: 500,
                retainedExpertPayloadBytesAfterGrowth: 400,
                promotedExpertPageStreamBytes: 900)

        #expect(learnedGrowthBytes == 0)
    }
}
