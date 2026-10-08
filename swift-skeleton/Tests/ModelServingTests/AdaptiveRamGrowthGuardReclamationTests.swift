import Foundation

import Testing

import ModelServing

/// Reclamation and evidence-scaling portfolio for adaptive RAM growth:
/// active-memory release, recovery-reserve shortfalls, and unobserved-shape
/// scaling, port of crates/model-serving/tests/hermetic/adaptive_ram_growth_guard.rs.
@Suite
final class AdaptiveRamGrowthGuardReclamationTests {

    private static let DEFAULT_DECODE_CONTEXT: AdaptiveRamGrowthContext =
        AdaptiveRamGrowthContext.decode(forwardTokenCount: 1, sparseExpertsArePaged: false)

    @Test
    func should_not_underflow_when_active_memory_falls_without_a_new_allocator_peak() throws {
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 1_000)
        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            Self.DEFAULT_DECODE_CONTEXT,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 700,
            activeMemoryBytesAfterGrowth: 500,
            peakMemoryBytesDuringGrowth: 600,
            exactTemporaryWorkspaceBytes: 0)

        let projection: AdaptiveRamGrowthProjection = try adaptiveRamGrowthGuard
            .projectGrowthForContext(
                Self.DEFAULT_DECODE_CONTEXT,
                currentActiveMemoryBytes: 800,
                exactPersistentGrowthBytes: 200,
                routedExpertPageReservationBytes: 0,
                exactTemporaryWorkspaceBytes: 0)

        #expect(projection.observedTransientHighWaterBytes == 0)
        #expect(projection.fitsStableAndPeakLimits)
    }

    @Test
    func should_admit_growth_without_expert_reclamation_when_only_the_recovery_reserve_is_short()
        throws {
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 1_000)
        // Learn a 150-byte transient window.
        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            Self.DEFAULT_DECODE_CONTEXT,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 400,
            activeMemoryBytesAfterGrowth: 500,
            peakMemoryBytesDuringGrowth: 650,
            exactTemporaryWorkspaceBytes: 0)

        let projection: AdaptiveRamGrowthProjection = try adaptiveRamGrowthGuard
            .projectGrowthForContext(
                Self.DEFAULT_DECODE_CONTEXT,
                currentActiveMemoryBytes: 600,
                exactPersistentGrowthBytes: 150,
                routedExpertPageReservationBytes: 0,
                exactTemporaryWorkspaceBytes: 0)

        #expect(projection.currentActiveMemoryBytes == 600)
        #expect(projection.exactPersistentGrowthBytes == 150)
        #expect(projection.observedTransientHighWaterBytes == 150)
        #expect(projection.peakProjectedBytes == 900)
        #expect(projection.recoveryProjectedBytes == 1_050)
        #expect(projection.activeMemoryCeilingBytes == 1_000)
        #expect(projection.operationReclamationRequiredBytes == 0)
        #expect(projection.recoveryReserveShortfallBytes == 40)
        #expect(
            projection.expertRetentionReclamationPlan(retainedExpertPayloadBytes: 1_000)
                .reclamationTargetBytes == 0,
            "recovery-only headroom must not force expert reclamation before a typed allocation failure")
        #expect(projection.fitsStableAndPeakLimits)
        #expect(!projection.hasFullRecoveryReserve)
    }

    @Test
    func should_report_only_the_measured_peak_shortfall_as_required_reclamation() throws {
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 1_000)
        // Learn a 200-byte transient window.
        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            Self.DEFAULT_DECODE_CONTEXT,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 400,
            activeMemoryBytesAfterGrowth: 500,
            peakMemoryBytesDuringGrowth: 700,
            exactTemporaryWorkspaceBytes: 0)

        let projection: AdaptiveRamGrowthProjection = try adaptiveRamGrowthGuard
            .projectGrowthForContext(
                Self.DEFAULT_DECODE_CONTEXT,
                currentActiveMemoryBytes: 700,
                exactPersistentGrowthBytes: 150,
                routedExpertPageReservationBytes: 0,
                exactTemporaryWorkspaceBytes: 0)

        // peak: 700 + 150 + 200 = 1_050, deficit against P=1,010 is 40.
        // recovery: 1_050 + 200 = 1_250, shortfall against P=1,010 is 240.
        #expect(projection.operationReclamationRequiredBytes == 40)
        #expect(projection.recoveryReserveShortfallBytes == 240)
        #expect(
            projection.expertRetentionReclamationPlan(retainedExpertPayloadBytes: 1_000)
                .reclamationTargetBytes == 40,
            "preemptive reclamation should cover the measured peak shortfall, not the larger diagnostic recovery shortfall")
        #expect(!projection.fitsStableAndPeakLimits)
        #expect(!projection.hasFullRecoveryReserve)
    }

    @Test
    func should_reject_a_recovery_projection_overflow() throws {
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: Int.max)
        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            Self.DEFAULT_DECODE_CONTEXT,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 0,
            activeMemoryBytesAfterGrowth: 0,
            peakMemoryBytesDuringGrowth: Int.max,
            exactTemporaryWorkspaceBytes: 0)

        #expect(throws: AdaptiveRamGrowthGuardError.memoryProjectionOverflow) {
            try adaptiveRamGrowthGuard.projectGrowthForContext(
                Self.DEFAULT_DECODE_CONTEXT,
                currentActiveMemoryBytes: 0,
                exactPersistentGrowthBytes: 0,
                routedExpertPageReservationBytes: 0,
                exactTemporaryWorkspaceBytes: 0)
        }
    }

    // Issue #623: one large shape's transient high-water was charged unchanged to
    // every later forward of every other shape, demoting a fully resident expert
    // owner whose own chunk needed a fraction of the reserved bytes. The admission
    // reserve must therefore be shaped by the forward being admitted, with the
    // global maximum kept only for a phase that has no evidence at all.
    @Test
    func should_scale_an_unobserved_shape_reserve_by_its_own_token_count() throws {
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 1_000_000)
        let largePagedPrefillContext: AdaptiveRamGrowthContext = AdaptiveRamGrowthContext.prefill(
            forwardTokenCount: 8_192,
            promptPositionContextBucket: 0,
            hasVisualEmbeddings: false,
            sparseExpertsArePaged: true)
        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            largePagedPrefillContext,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 0,
            activeMemoryBytesAfterGrowth: 0,
            peakMemoryBytesDuringGrowth: 8_000,
            exactTemporaryWorkspaceBytes: 0)

        let smallResidentPrefillContext: AdaptiveRamGrowthContext =
            AdaptiveRamGrowthContext.prefill(
                forwardTokenCount: 2_048,
                promptPositionContextBucket: 0,
                hasVisualEmbeddings: false,
                sparseExpertsArePaged: false)
        let projection: AdaptiveRamGrowthProjection = try adaptiveRamGrowthGuard
            .projectGrowthForContext(
                smallResidentPrefillContext,
                currentActiveMemoryBytes: 900_000,
                exactPersistentGrowthBytes: 10_000,
                routedExpertPageReservationBytes: 0,
                exactTemporaryWorkspaceBytes: 0)

        // 8,000 bytes observed at 8,192 tokens scale to 2,000 bytes at 2,048 tokens
        // (ceiling division), not the borrowed 8,000-byte absolute window.
        #expect(projection.observedTransientHighWaterBytes == 2_000)
        #expect(projection.transientReserveSource == .phaseScaled)
    }

    @Test
    func should_prefer_exact_context_evidence_over_the_scaled_phase_estimate() throws {
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 1_000_000)
        let largePagedPrefillContext: AdaptiveRamGrowthContext = AdaptiveRamGrowthContext.prefill(
            forwardTokenCount: 8_192,
            promptPositionContextBucket: 0,
            hasVisualEmbeddings: false,
            sparseExpertsArePaged: true)
        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            largePagedPrefillContext,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 0,
            activeMemoryBytesAfterGrowth: 0,
            peakMemoryBytesDuringGrowth: 8_000,
            exactTemporaryWorkspaceBytes: 0)
        let exactResidentPrefillContext: AdaptiveRamGrowthContext =
            AdaptiveRamGrowthContext.prefill(
                forwardTokenCount: 2_048,
                promptPositionContextBucket: 3,
                hasVisualEmbeddings: false,
                sparseExpertsArePaged: false)
        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            exactResidentPrefillContext,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 0,
            activeMemoryBytesAfterGrowth: 0,
            peakMemoryBytesDuringGrowth: 1_500,
            exactTemporaryWorkspaceBytes: 0)

        let projection: AdaptiveRamGrowthProjection = try adaptiveRamGrowthGuard
            .projectGrowthForContext(
                exactResidentPrefillContext,
                currentActiveMemoryBytes: 900_000,
                exactPersistentGrowthBytes: 10_000,
                routedExpertPageReservationBytes: 0,
                exactTemporaryWorkspaceBytes: 0)

        #expect(projection.observedTransientHighWaterBytes == 1_500)
        #expect(projection.transientReserveSource == .exactContext)
    }

    @Test
    func should_use_the_largest_scaled_phase_observation_for_an_unobserved_shape() throws {
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 1_000_000)
        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            AdaptiveRamGrowthContext.prefill(
                forwardTokenCount: 8_192,
                promptPositionContextBucket: 0,
                hasVisualEmbeddings: false,
                sparseExpertsArePaged: true),
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 0,
            activeMemoryBytesAfterGrowth: 0,
            peakMemoryBytesDuringGrowth: 8_000,
            exactTemporaryWorkspaceBytes: 0)
        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            AdaptiveRamGrowthContext.prefill(
                forwardTokenCount: 1_024,
                promptPositionContextBucket: 9,
                hasVisualEmbeddings: false,
                sparseExpertsArePaged: false),
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 0,
            activeMemoryBytesAfterGrowth: 0,
            peakMemoryBytesDuringGrowth: 3_000,
            exactTemporaryWorkspaceBytes: 0)

        // 8,000 × 2,048/8,192 = 2,000; 3,000 × 2,048/1,024 = 6,000. The largest
        // scaled observation is the conservative proportional estimate.
        let projection: AdaptiveRamGrowthProjection = try adaptiveRamGrowthGuard
            .projectGrowthForContext(
                AdaptiveRamGrowthContext.prefill(
                    forwardTokenCount: 2_048,
                    promptPositionContextBucket: 0,
                    hasVisualEmbeddings: false,
                    sparseExpertsArePaged: false),
                currentActiveMemoryBytes: 900_000,
                exactPersistentGrowthBytes: 10_000,
                routedExpertPageReservationBytes: 0,
                exactTemporaryWorkspaceBytes: 0)

        #expect(projection.observedTransientHighWaterBytes == 6_000)
        #expect(projection.transientReserveSource == .phaseScaled)
    }

    @Test
    func should_keep_the_global_maximum_for_a_phase_without_any_evidence() throws {
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 1_000_000)
        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            Self.DEFAULT_DECODE_CONTEXT,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 0,
            activeMemoryBytesAfterGrowth: 0,
            peakMemoryBytesDuringGrowth: 500,
            exactTemporaryWorkspaceBytes: 0)

        // Decode evidence exists but prefill has none. The first prefill reserves
        // the global maximum; its completion records prefill evidence and every
        // later prefill reserves proportionally.
        let projection: AdaptiveRamGrowthProjection = try adaptiveRamGrowthGuard
            .projectGrowthForContext(
                AdaptiveRamGrowthContext.prefill(
                    forwardTokenCount: 2_048,
                    promptPositionContextBucket: 0,
                    hasVisualEmbeddings: false,
                    sparseExpertsArePaged: false),
                currentActiveMemoryBytes: 900_000,
                exactPersistentGrowthBytes: 10_000,
                routedExpertPageReservationBytes: 0,
                exactTemporaryWorkspaceBytes: 0)

        #expect(projection.observedTransientHighWaterBytes == 500)
        #expect(projection.transientReserveSource == .globalMaximum)
    }
}
