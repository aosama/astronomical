import Foundation

import Testing

import ModelServing

/// Completed-observation and live-limit contracts for adaptive RAM growth,
/// port of crates/model-serving/tests/hermetic/adaptive_ram_growth_observations.rs:
/// these tests are separate from projection admission so each test owner stays
/// readable; this file explains how completed forwards teach reusable transient
/// evidence, while AdaptiveRamGrowthGuardTests covers projection boundaries.
@Suite
final class AdaptiveRamGrowthObservationsTests {

    private static let DEFAULT_DECODE_CONTEXT: AdaptiveRamGrowthContext =
        AdaptiveRamGrowthContext.decode(forwardTokenCount: 1, sparseExpertsArePaged: false)
    private static let DEFAULT_PREFILL_CONTEXT: AdaptiveRamGrowthContext =
        AdaptiveRamGrowthContext.prefill(
            forwardTokenCount: 128,
            promptPositionContextBucket: 0,
            hasVisualEmbeddings: false,
            sparseExpertsArePaged: false)

    @Test
    func should_keep_prefill_and_decode_transient_high_water_values_independent() throws {
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 10_000)

        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            Self.DEFAULT_PREFILL_CONTEXT,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 1_000,
            activeMemoryBytesAfterGrowth: 2_000,
            peakMemoryBytesDuringGrowth: 10_000,
            exactTemporaryWorkspaceBytes: 0)
        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            Self.DEFAULT_DECODE_CONTEXT,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 2_000,
            activeMemoryBytesAfterGrowth: 2_100,
            peakMemoryBytesDuringGrowth: 2_600,
            exactTemporaryWorkspaceBytes: 0)

        #expect(adaptiveRamGrowthGuard.observedTransientHighWaterBytes(
            memoryPhase: MemoryPhase.prefill) == 8_000)
        #expect(adaptiveRamGrowthGuard.observedTransientHighWaterBytes(
            memoryPhase: MemoryPhase.decode) == 500)
    }

    @Test
    func should_cap_decode_warming_with_decode_evidence_instead_of_the_all_phase_maximum() throws {
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 10_000)

        // One huge prefill teaches a large transient window; decode teaches a
        // small one. Issue #512: decode warming must reserve against decode's own
        // workspace, not stay suppressed by prefill's spent transient forever.
        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            Self.DEFAULT_PREFILL_CONTEXT,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 1_000,
            activeMemoryBytesAfterGrowth: 2_000,
            peakMemoryBytesDuringGrowth: 10_000,
            exactTemporaryWorkspaceBytes: 0)
        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            Self.DEFAULT_DECODE_CONTEXT,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 2_000,
            activeMemoryBytesAfterGrowth: 2_100,
            peakMemoryBytesDuringGrowth: 2_600,
            exactTemporaryWorkspaceBytes: 0)

        let decodeCeilingBytes: Int = adaptiveRamGrowthGuard.hotExpertRetentionCeilingBytes(
            memoryPhase: MemoryPhase.decode,
            currentActiveMemoryBytes: 2_100,
            currentRetainedPayloadBytes: 0,
            routedExpertPageReservationBytes: 0)
        let prefillCeilingBytes: Int = adaptiveRamGrowthGuard.hotExpertRetentionCeilingBytes(
            memoryPhase: MemoryPhase.prefill,
            currentActiveMemoryBytes: 2_100,
            currentRetainedPayloadBytes: 0,
            routedExpertPageReservationBytes: 0)

        // Ceiling = retained + (ceiling + 1% allowance - active - phase reserve).
        // Decode reserve is 500 bytes; the all-phase maximum is 8_000 bytes.
        #expect(decodeCeilingBytes == 10_000 + 100 - 2_100 - 500)
        #expect(prefillCeilingBytes == 10_000 + 100 - 2_100 - 8_000)
        #expect(
            decodeCeilingBytes > prefillCeilingBytes,
            "decode warming must claim more than a prefill-sized reserve allows")
    }

    @Test
    func should_fall_back_to_the_all_phase_maximum_before_the_phase_has_evidence() throws {
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 10_000)

        // Only prefill has observed anything. The first decode warming steps must
        // stay conservative until decode teaches its own workspace.
        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            Self.DEFAULT_PREFILL_CONTEXT,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 1_000,
            activeMemoryBytesAfterGrowth: 2_000,
            peakMemoryBytesDuringGrowth: 10_000,
            exactTemporaryWorkspaceBytes: 0)

        let decodeCeilingBytes: Int = adaptiveRamGrowthGuard.hotExpertRetentionCeilingBytes(
            memoryPhase: MemoryPhase.decode,
            currentActiveMemoryBytes: 2_000,
            currentRetainedPayloadBytes: 0,
            routedExpertPageReservationBytes: 0)

        #expect(decodeCeilingBytes == 10_000 + 100 - 2_000 - 8_000)
    }

    @Test
    func should_record_a_completed_zero_transient_prefill_observation() throws {
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 10_000)

        #expect(!adaptiveRamGrowthGuard.hasCompletedGrowthObservation(
            memoryPhase: MemoryPhase.prefill))
        #expect(!adaptiveRamGrowthGuard.hasCompletedGrowthObservation(
            memoryPhase: MemoryPhase.decode))

        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            Self.DEFAULT_PREFILL_CONTEXT,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 2_000,
            activeMemoryBytesAfterGrowth: 2_000,
            peakMemoryBytesDuringGrowth: 2_000,
            exactTemporaryWorkspaceBytes: 0)

        #expect(
            adaptiveRamGrowthGuard.hasCompletedGrowthObservation(memoryPhase: MemoryPhase.prefill),
            "a completed prefill must count as observed even when it used no transient bytes")
        #expect(
            !adaptiveRamGrowthGuard.hasCompletedGrowthObservation(
                memoryPhase: MemoryPhase.decode),
            "prefill evidence must not mark decode as observed")
        #expect(adaptiveRamGrowthGuard.observedTransientHighWaterBytes(
            memoryPhase: MemoryPhase.prefill) == 0)
    }

    @Test
    func should_preserve_adaptive_high_water_observations_when_the_limit_changes() throws {
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 10_000)
        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            Self.DEFAULT_DECODE_CONTEXT,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 4_000,
            activeMemoryBytesAfterGrowth: 5_000,
            peakMemoryBytesDuringGrowth: 6_000,
            exactTemporaryWorkspaceBytes: 0)

        try adaptiveRamGrowthGuard.updateActiveMemoryCeilingBytes(8_000)

        let updatedProjection: AdaptiveRamGrowthProjection = try adaptiveRamGrowthGuard
            .projectGrowthForContext(
                Self.DEFAULT_DECODE_CONTEXT,
                currentActiveMemoryBytes: 6_000,
                exactPersistentGrowthBytes: 500,
                routedExpertPageReservationBytes: 0,
                exactTemporaryWorkspaceBytes: 0)
        #expect(adaptiveRamGrowthGuard.observedTransientHighWaterBytes(
            memoryPhase: MemoryPhase.decode) == 1_000)
        #expect(updatedProjection.activeMemoryCeilingBytes == 8_000)
        #expect(updatedProjection.allowedActiveMemoryBytes == 8_080)
    }

    @Test
    func should_project_exact_temporary_workspace_without_double_counting_learned_residual_growth()
        throws {
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 2_000)
        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            Self.DEFAULT_PREFILL_CONTEXT,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 400,
            activeMemoryBytesAfterGrowth: 500,
            peakMemoryBytesDuringGrowth: 800,
            exactTemporaryWorkspaceBytes: 200)

        let projection: AdaptiveRamGrowthProjection = try adaptiveRamGrowthGuard
            .projectGrowthForContext(
                Self.DEFAULT_PREFILL_CONTEXT,
                currentActiveMemoryBytes: 500,
                exactPersistentGrowthBytes: 100,
                routedExpertPageReservationBytes: 0,
                exactTemporaryWorkspaceBytes: 200)

        #expect(projection.exactTemporaryWorkspaceBytes == 200)
        #expect(projection.observedTransientHighWaterBytes == 100)
        #expect(projection.stableProjectedBytes == 600)
        #expect(projection.peakProjectedBytes == 900)
        #expect(projection.recoveryProjectedBytes == 1_200)
        #expect(
            projection.forwardReserveBytes == 400,
            "the residency planner must reserve every byte between the admitted active baseline and expected peak boundary")
    }

    @Test
    func should_not_subtract_stable_expert_growth_twice_from_prefill_headroom() throws {
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 40_000)

        // Active memory grows by 10,000 stable expert bytes. Peak is another 3,000
        // bytes above the final active sample, so the reusable transient window is
        // exactly 3,000 bytes. The stable post-forward sample already excludes the
        // expert growth from this difference.
        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            Self.DEFAULT_PREFILL_CONTEXT,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 20_000,
            activeMemoryBytesAfterGrowth: 30_000,
            peakMemoryBytesDuringGrowth: 33_000,
            exactTemporaryWorkspaceBytes: 0)

        #expect(adaptiveRamGrowthGuard.observedTransientHighWaterBytes(
            memoryPhase: MemoryPhase.prefill) == 3_000)
    }

    @Test
    func should_reserve_a_routed_expert_page_alongside_lazy_persistent_growth_after_a_live_limit_reduction()
        throws {
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 28_000_000_000)

        let projection: AdaptiveRamGrowthProjection = try adaptiveRamGrowthGuard
            .projectGrowthForContext(
                AdaptiveRamGrowthContext.decode(forwardTokenCount: 1, sparseExpertsArePaged: true),
                currentActiveMemoryBytes: 27_806_577_158,
                exactPersistentGrowthBytes: 192_061_440,
                routedExpertPageReservationBytes: 70_778_880,
                exactTemporaryWorkspaceBytes: 0)

        #expect(projection.routedExpertPageReservationBytes == 70_778_880)
        #expect(projection.stableProjectedBytes == 28_069_417_478)
        #expect(projection.operationReclamationRequiredBytes == 69_417_478)
    }

    @Test
    func should_exclude_streamed_and_evicted_expert_pages_from_the_learned_prefill_transient()
        throws {
        // Issue #691: paged prefill promotes expert pages that appear in the MLX
        // peak but are evicted before completion (net resident delta zero). The
        // subtract chain must charge them to expert ownership, not activation:
        // baseline 1,000 stays stable, the forward peaks with 200 transient
        // activation bytes plus 400 streamed-and-evicted page bytes, workspace 0.
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 20_000)

        adaptiveRamGrowthGuard.recordCompletedGrowthExcludingExpertPageStream(
            Self.DEFAULT_PREFILL_CONTEXT,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 1_000,
            activeMemoryBytesAfterGrowth: 1_000,
            peakMemoryBytesDuringGrowth: 1_600,
            exactTemporaryWorkspaceBytes: 0,
            retainedExpertPayloadGrowthBytes: 0,
            promotedExpertPageStreamBytes: 400)

        #expect(adaptiveRamGrowthGuard.observedTransientHighWaterBytes(
            memoryPhase: MemoryPhase.prefill) == 200)
    }

    @Test
    func should_exclude_only_the_stream_bytes_beyond_the_retained_expert_delta() throws {
        // One forward promotes 500 page bytes and retains 300 of them: the retained
        // delta already sits inside the post-forward active baseline, so only the
        // 200 evicted stream bytes leave the residual. peak = 1,300 stable
        // (1,000 + 300 retained) + 200 activation + 200 evicted stream.
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 20_000)

        adaptiveRamGrowthGuard.recordCompletedGrowthExcludingExpertPageStream(
            Self.DEFAULT_PREFILL_CONTEXT,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 1_000,
            activeMemoryBytesAfterGrowth: 1_300,
            peakMemoryBytesDuringGrowth: 1_700,
            exactTemporaryWorkspaceBytes: 0,
            retainedExpertPayloadGrowthBytes: 300,
            promotedExpertPageStreamBytes: 500)

        #expect(adaptiveRamGrowthGuard.observedTransientHighWaterBytes(
            memoryPhase: MemoryPhase.prefill) == 200)
    }

    @Test
    func should_never_subtract_retained_pages_twice_from_the_learned_transient() throws {
        // Fully retained promotion with zero eviction: peak - stable already
        // excludes the retained pages through the post-forward active sample, so
        // the stream evidence must cost the residual nothing (issue #691 guard:
        // erasing real activation headroom would let the next forward overfill
        // retention). peak 1,650 = stable 1,000 + 400 retained = 1,400, plus
        // 250 activation, plus 0 evicted.
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 20_000)

        adaptiveRamGrowthGuard.recordCompletedGrowthExcludingExpertPageStream(
            Self.DEFAULT_PREFILL_CONTEXT,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 1_000,
            activeMemoryBytesAfterGrowth: 1_400,
            peakMemoryBytesDuringGrowth: 1_650,
            exactTemporaryWorkspaceBytes: 0,
            retainedExpertPayloadGrowthBytes: 400,
            promotedExpertPageStreamBytes: 400)

        #expect(adaptiveRamGrowthGuard.observedTransientHighWaterBytes(
            memoryPhase: MemoryPhase.prefill) == 250)
    }

    @Test
    func should_keep_the_older_learning_contract_unchanged_without_expert_page_stream_evidence()
        throws {
        // The pre-#691 method delegates to the same core with zero stream bytes, so
        // every existing observation shape is preserved verbatim (#623/#644 pins).
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 10_000)

        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            Self.DEFAULT_PREFILL_CONTEXT,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 1_000,
            activeMemoryBytesAfterGrowth: 2_000,
            peakMemoryBytesDuringGrowth: 10_000,
            exactTemporaryWorkspaceBytes: 0)

        #expect(adaptiveRamGrowthGuard.observedTransientHighWaterBytes(
            memoryPhase: MemoryPhase.prefill) == 8_000)
    }

    @Test
    func should_saturate_to_zero_when_stream_evidence_exceeds_the_whole_residual() throws {
        // Expert churn plus streaming can leave sampled stream bytes larger than
        // the observed peak window (allocator timing differences across events).
        // Under-projection is recoverable by design; the learned high-water must
        // clamp instead of wrapping.
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 20_000)

        adaptiveRamGrowthGuard.recordCompletedGrowthExcludingExpertPageStream(
            Self.DEFAULT_PREFILL_CONTEXT,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 1_000,
            activeMemoryBytesAfterGrowth: 1_000,
            peakMemoryBytesDuringGrowth: 1_300,
            exactTemporaryWorkspaceBytes: 100,
            retainedExpertPayloadGrowthBytes: 0,
            promotedExpertPageStreamBytes: 900)

        #expect(adaptiveRamGrowthGuard.observedTransientHighWaterBytes(
            memoryPhase: MemoryPhase.prefill) == 0)
    }
}
