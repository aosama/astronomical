import Foundation

import Testing

import ModelServing

/// Verbalized-sampling portfolio for adaptive RAM growth:
/// ordinary fitting growth (0.55), rising transient pressure (0.20), exact-limit
/// boundary (0.10), arithmetic overflow (0.08), lower later spike (0.05), and
/// active-memory release between observations (0.02). Probabilities are
/// test-design estimates rather than measured production frequencies,
/// port of crates/model-serving/tests/hermetic/adaptive_ram_growth_guard.rs.
@Suite
final class AdaptiveRamGrowthGuardTests {

    private static let DEFAULT_DECODE_CONTEXT: AdaptiveRamGrowthContext =
        AdaptiveRamGrowthContext.decode(forwardTokenCount: 1, sparseExpertsArePaged: false)

    @Test
    func should_reject_a_zero_active_memory_limit() {
        #expect(throws: AdaptiveRamGrowthGuardError.invalidActiveMemoryCeiling) {
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 0)
        }
    }

    @Test
    func should_allow_unobserved_growth_when_exact_persistent_bytes_fit_the_limit() throws {
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 1_000)

        let projection: AdaptiveRamGrowthProjection = try adaptiveRamGrowthGuard
            .projectGrowthForContext(
                Self.DEFAULT_DECODE_CONTEXT,
                currentActiveMemoryBytes: 700,
                exactPersistentGrowthBytes: 300,
                routedExpertPageReservationBytes: 0,
                exactTemporaryWorkspaceBytes: 0)

        #expect(projection.fitsStableAndPeakLimits)
    }

    @Test
    func should_apply_transient_learning_across_unseen_contexts_for_admission() throws {
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 1_000)
        let observedPrefillContext: AdaptiveRamGrowthContext = AdaptiveRamGrowthContext.prefill(
            forwardTokenCount: 128,
            promptPositionContextBucket: 0,
            hasVisualEmbeddings: false,
            sparseExpertsArePaged: true)
        let differentPrefillContext: AdaptiveRamGrowthContext = AdaptiveRamGrowthContext.prefill(
            forwardTokenCount: 256,
            promptPositionContextBucket: 0,
            hasVisualEmbeddings: false,
            sparseExpertsArePaged: true)

        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            observedPrefillContext,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 400,
            activeMemoryBytesAfterGrowth: 500,
            peakMemoryBytesDuringGrowth: 700,
            exactTemporaryWorkspaceBytes: 0)

        let observedContextProjection: AdaptiveRamGrowthProjection = try adaptiveRamGrowthGuard
            .projectGrowthForContext(
                observedPrefillContext,
                currentActiveMemoryBytes: 700,
                exactPersistentGrowthBytes: 100,
                routedExpertPageReservationBytes: 0,
                exactTemporaryWorkspaceBytes: 0)
        let differentContextProjection: AdaptiveRamGrowthProjection = try adaptiveRamGrowthGuard
            .projectGrowthForContext(
                differentPrefillContext,
                currentActiveMemoryBytes: 700,
                exactPersistentGrowthBytes: 100,
                routedExpertPageReservationBytes: 0,
                exactTemporaryWorkspaceBytes: 0)

        #expect(observedContextProjection.observedTransientHighWaterBytes == 200)
        // Issue #623: an unseen shape reserves its own token-proportional share of
        // the observed window (200 bytes at 128 tokens → 400 bytes at 256 tokens),
        // not the observed shape's absolute window.
        #expect(differentContextProjection.observedTransientHighWaterBytes == 400)
        #expect(observedContextProjection.stableProjectedBytes == 800)
        #expect(observedContextProjection.peakProjectedBytes == 1_000)
        #expect(observedContextProjection.allowedActiveMemoryBytes == 1_010)
    }

    @Test
    func should_keep_exact_context_evidence_separate_from_phase_admission_maximum() throws {
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 1_000)
        let smallerContext: AdaptiveRamGrowthContext = AdaptiveRamGrowthContext.prefill(
            forwardTokenCount: 128,
            promptPositionContextBucket: 0,
            hasVisualEmbeddings: false,
            sparseExpertsArePaged: true)
        let largerContext: AdaptiveRamGrowthContext = AdaptiveRamGrowthContext.prefill(
            forwardTokenCount: 256,
            promptPositionContextBucket: 0,
            hasVisualEmbeddings: false,
            sparseExpertsArePaged: true)
        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            smallerContext,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 100,
            activeMemoryBytesAfterGrowth: 100,
            peakMemoryBytesDuringGrowth: 150,
            exactTemporaryWorkspaceBytes: 0)
        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            largerContext,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 100,
            activeMemoryBytesAfterGrowth: 100,
            peakMemoryBytesDuringGrowth: 300,
            exactTemporaryWorkspaceBytes: 0)

        #expect(adaptiveRamGrowthGuard.observedTransientHighWaterBytesForContext(
            smallerContext) == 50)
        #expect(adaptiveRamGrowthGuard.observedTransientHighWaterBytes(
            memoryPhase: MemoryPhase.prefill) == 200)
    }

    @Test
    func should_accept_stable_memory_at_c_and_reject_one_byte_above_c() throws {
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 1_000)
        let adaptiveRamGrowthContext: AdaptiveRamGrowthContext =
            AdaptiveRamGrowthContext.decode(forwardTokenCount: 1, sparseExpertsArePaged: false)

        let fittingProjection: AdaptiveRamGrowthProjection = try adaptiveRamGrowthGuard
            .projectGrowthForContext(
                adaptiveRamGrowthContext,
                currentActiveMemoryBytes: 700,
                exactPersistentGrowthBytes: 300,
                routedExpertPageReservationBytes: 0,
                exactTemporaryWorkspaceBytes: 0)
        let exceedingProjection: AdaptiveRamGrowthProjection = try adaptiveRamGrowthGuard
            .projectGrowthForContext(
                adaptiveRamGrowthContext,
                currentActiveMemoryBytes: 700,
                exactPersistentGrowthBytes: 301,
                routedExpertPageReservationBytes: 0,
                exactTemporaryWorkspaceBytes: 0)

        #expect(fittingProjection.stableProjectedBytes == 1_000)
        #expect(fittingProjection.fitsStableAndPeakLimits)
        #expect(exceedingProjection.stableProjectedBytes == 1_001)
        #expect(exceedingProjection.operationReclamationRequiredBytes == 1)
        #expect(!exceedingProjection.fitsStableAndPeakLimits)
    }

    @Test
    func should_accept_peak_memory_at_p_and_reject_one_byte_above_p() throws {
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 1_000)
        let fittingPeakContext: AdaptiveRamGrowthContext =
            AdaptiveRamGrowthContext.decode(forwardTokenCount: 1, sparseExpertsArePaged: false)
        let exceedingPeakContext: AdaptiveRamGrowthContext =
            AdaptiveRamGrowthContext.decode(forwardTokenCount: 2, sparseExpertsArePaged: false)
        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            fittingPeakContext,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 0,
            activeMemoryBytesAfterGrowth: 0,
            peakMemoryBytesDuringGrowth: 10,
            exactTemporaryWorkspaceBytes: 0)
        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            exceedingPeakContext,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 0,
            activeMemoryBytesAfterGrowth: 0,
            peakMemoryBytesDuringGrowth: 11,
            exactTemporaryWorkspaceBytes: 0)

        let fittingProjection: AdaptiveRamGrowthProjection = try adaptiveRamGrowthGuard
            .projectGrowthForContext(
                fittingPeakContext,
                currentActiveMemoryBytes: 900,
                exactPersistentGrowthBytes: 100,
                routedExpertPageReservationBytes: 0,
                exactTemporaryWorkspaceBytes: 0)
        let exceedingProjection: AdaptiveRamGrowthProjection = try adaptiveRamGrowthGuard
            .projectGrowthForContext(
                exceedingPeakContext,
                currentActiveMemoryBytes: 900,
                exactPersistentGrowthBytes: 101,
                routedExpertPageReservationBytes: 0,
                exactTemporaryWorkspaceBytes: 0)

        // Issue #623: each context reserves its own exact evidence (10 and 11),
        // not the global maximum. The fitting context therefore peaks exactly at
        // P = C + 1 percent and is admitted; one byte more requires reclamation.
        #expect(fittingProjection.peakProjectedBytes == 1_010)
        #expect(fittingProjection.fitsStableAndPeakLimits)
        #expect(exceedingProjection.peakProjectedBytes == 1_012)
        #expect(exceedingProjection.operationReclamationRequiredBytes == 2)
        #expect(!exceedingProjection.fitsStableAndPeakLimits)
    }

    @Test
    func should_reserve_transient_headroom_for_unseen_prefill_contexts() throws {
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 1_000)
        let observedPrefillContext: AdaptiveRamGrowthContext = AdaptiveRamGrowthContext.prefill(
            forwardTokenCount: 128,
            promptPositionContextBucket: 7,
            hasVisualEmbeddings: true,
            sparseExpertsArePaged: true)
        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            observedPrefillContext,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 0,
            activeMemoryBytesAfterGrowth: 0,
            peakMemoryBytesDuringGrowth: 200,
            exactTemporaryWorkspaceBytes: 0)

        let independentPrefillContexts: [AdaptiveRamGrowthContext] = [
            AdaptiveRamGrowthContext.prefill(
                forwardTokenCount: 128,
                promptPositionContextBucket: 8,
                hasVisualEmbeddings: true,
                sparseExpertsArePaged: true),
            AdaptiveRamGrowthContext.prefill(
                forwardTokenCount: 128,
                promptPositionContextBucket: 7,
                hasVisualEmbeddings: false,
                sparseExpertsArePaged: true),
            AdaptiveRamGrowthContext.prefill(
                forwardTokenCount: 128,
                promptPositionContextBucket: 7,
                hasVisualEmbeddings: true,
                sparseExpertsArePaged: false),
        ]
        for independentPrefillContext: AdaptiveRamGrowthContext in independentPrefillContexts {
            let projection: AdaptiveRamGrowthProjection = try adaptiveRamGrowthGuard
                .projectGrowthForContext(
                    independentPrefillContext,
                    currentActiveMemoryBytes: 500,
                    exactPersistentGrowthBytes: 100,
                    routedExpertPageReservationBytes: 0,
                    exactTemporaryWorkspaceBytes: 0)
            #expect(projection.observedTransientHighWaterBytes == 200)
        }
    }

    @Test
    func should_not_retain_a_final_partial_prefill_tail_as_reusable_evidence() throws {
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 1_000)
        let partialTailContext: AdaptiveRamGrowthContext = AdaptiveRamGrowthContext.prefill(
            forwardTokenCount: 37,
            promptPositionContextBucket: 0,
            hasVisualEmbeddings: false,
            sparseExpertsArePaged: true)
        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            partialTailContext,
            shouldRetainObservation: false,
            activeMemoryBytesBeforeGrowth: 0,
            activeMemoryBytesAfterGrowth: 0,
            peakMemoryBytesDuringGrowth: 500,
            exactTemporaryWorkspaceBytes: 0)

        let projection: AdaptiveRamGrowthProjection = try adaptiveRamGrowthGuard
            .projectGrowthForContext(
                partialTailContext,
                currentActiveMemoryBytes: 400,
                exactPersistentGrowthBytes: 100,
                routedExpertPageReservationBytes: 0,
                exactTemporaryWorkspaceBytes: 0)

        #expect(projection.observedTransientHighWaterBytes == 0)
    }

    @Test
    func should_reject_growth_when_the_peak_projection_exceeds_the_limit() throws {
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 1_000)
        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            Self.DEFAULT_DECODE_CONTEXT,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 400,
            activeMemoryBytesAfterGrowth: 500,
            peakMemoryBytesDuringGrowth: 650,
            exactTemporaryWorkspaceBytes: 0)

        // peak: 800 + 100 + 150 = 1_050 > P=1_010
        let projection: AdaptiveRamGrowthProjection = try adaptiveRamGrowthGuard
            .projectGrowthForContext(
                Self.DEFAULT_DECODE_CONTEXT,
                currentActiveMemoryBytes: 800,
                exactPersistentGrowthBytes: 100,
                routedExpertPageReservationBytes: 0,
                exactTemporaryWorkspaceBytes: 0)

        #expect(projection.peakProjectedBytes == 1_050)
        #expect(projection.operationReclamationRequiredBytes == 40)
        #expect(!projection.fitsStableAndPeakLimits)
    }

    @Test
    func should_reject_an_overflowing_memory_projection() throws {
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: Int.max)

        #expect(throws: AdaptiveRamGrowthGuardError.memoryProjectionOverflow) {
            try adaptiveRamGrowthGuard.projectGrowthForContext(
                Self.DEFAULT_DECODE_CONTEXT,
                currentActiveMemoryBytes: Int.max,
                exactPersistentGrowthBytes: 1,
                routedExpertPageReservationBytes: 0,
                exactTemporaryWorkspaceBytes: 0)
        }
    }

    @Test
    func should_preserve_the_highest_transient_observation_after_a_lower_spike() throws {
        let adaptiveRamGrowthGuard: AdaptiveRamGrowthGuard =
            try AdaptiveRamGrowthGuard(activeMemoryCeilingBytes: 1_000)
        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            Self.DEFAULT_DECODE_CONTEXT,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 400,
            activeMemoryBytesAfterGrowth: 500,
            peakMemoryBytesDuringGrowth: 700,
            exactTemporaryWorkspaceBytes: 0)
        adaptiveRamGrowthGuard.recordCompletedGrowthForContext(
            Self.DEFAULT_DECODE_CONTEXT,
            shouldRetainObservation: true,
            activeMemoryBytesBeforeGrowth: 500,
            activeMemoryBytesAfterGrowth: 550,
            peakMemoryBytesDuringGrowth: 600,
            exactTemporaryWorkspaceBytes: 0)

        // peak: 701 + 100 + 200 = 1_001 <= P=1_010
        let projection: AdaptiveRamGrowthProjection = try adaptiveRamGrowthGuard
            .projectGrowthForContext(
                Self.DEFAULT_DECODE_CONTEXT,
                currentActiveMemoryBytes: 701,
                exactPersistentGrowthBytes: 100,
                routedExpertPageReservationBytes: 0,
                exactTemporaryWorkspaceBytes: 0)

        #expect(projection.observedTransientHighWaterBytes == 200)
        #expect(projection.peakProjectedBytes == 1_001)
        #expect(projection.operationReclamationRequiredBytes == 0)
        #expect(projection.fitsStableAndPeakLimits)
    }
}
