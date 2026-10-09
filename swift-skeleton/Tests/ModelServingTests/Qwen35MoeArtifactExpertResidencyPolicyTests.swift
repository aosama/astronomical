import Foundation;

import Testing;

import JourneyCategories;

import ModelServing;

/**
 * Pure residency-decision journeys for the MoE artifact runtime: the
 * machine-adaptive choice between binding every routed expert resident and
 * paging experts from storage. The policy is CPU-only arithmetic over
 * payload bytes and the MLX memory ceiling, so these journeys run parallel
 * with no GPU evaluation.
 */
@Suite(.tags(.hermeticJourney))
final class Qwen35MoeArtifactExpertResidencyPolicyTests {

    @Test(.timeLimit(.minutes(1)))
    func should_choose_resident_when_core_expert_and_headroom_fit_the_ceiling() throws {
        let residency: Qwen35MoeArtifactExpertResidency = Qwen35MoeArtifactExpertResidencyPolicy
            .decide(
                residentPayloadBytes: 4_000_000_000,
                expertPayloadBytes: 20_000_000_000,
                contextWindowReserveBytes: 1_000_000_000,
                activationHeadroomBytes: 3_000_000_000,
                largestGateUpFusionTransientBytes: 400_000_000,
                mlxMemoryCeilingBytes: 40_000_000_000);
        #expect(residency == .resident);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_choose_paged_when_the_expert_payload_exceeds_the_ceiling() throws {
        let residency: Qwen35MoeArtifactExpertResidency = Qwen35MoeArtifactExpertResidencyPolicy
            .decide(
                residentPayloadBytes: 4_000_000_000,
                expertPayloadBytes: 44_000_000_000,
                contextWindowReserveBytes: 0,
                activationHeadroomBytes: 3_000_000_000,
                largestGateUpFusionTransientBytes: 400_000_000,
                mlxMemoryCeilingBytes: 40_000_000_000);
        #expect(residency == .paged);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_choose_paged_when_the_complete_layer_headroom_does_not_fit() throws {
        // Payload plus core fits exactly, but the fusion headroom
        // (the resident gate/up fusion transient for one layer) does not,
        // so the resident bind would race the ceiling on the first layer.
        let residency: Qwen35MoeArtifactExpertResidency = Qwen35MoeArtifactExpertResidencyPolicy
            .decide(
                residentPayloadBytes: 4_000_000_000,
                expertPayloadBytes: 34_500_000_000,
                contextWindowReserveBytes: 1_000_000_000,
                activationHeadroomBytes: 0,
                largestGateUpFusionTransientBytes: 600_000_000,
                mlxMemoryCeilingBytes: 40_000_000_000);
        #expect(residency == .paged);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_choose_paged_when_the_ceiling_is_unknown() throws {
        // A machine that reports no MLX ceiling cannot prove a resident
        // bind fits, so the policy fails toward paging.
        let residency: Qwen35MoeArtifactExpertResidency = Qwen35MoeArtifactExpertResidencyPolicy
            .decide(
                residentPayloadBytes: 4_000_000_000,
                expertPayloadBytes: 20_000_000_000,
                contextWindowReserveBytes: 1_000_000_000,
                activationHeadroomBytes: 3_000_000_000,
                largestGateUpFusionTransientBytes: 400_000_000,
                mlxMemoryCeilingBytes: 0);
        #expect(residency == .paged);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_choose_paged_when_model_fits_but_context_reserve_does_not() throws {
        let residency: Qwen35MoeArtifactExpertResidency = Qwen35MoeArtifactExpertResidencyPolicy
            .decide(
                residentPayloadBytes: 4_000_000_000,
                expertPayloadBytes: 33_000_000_000,
                contextWindowReserveBytes: 3_000_000_000,
                activationHeadroomBytes: 0,
                largestGateUpFusionTransientBytes: 400_000_000,
                mlxMemoryCeilingBytes: 40_000_000_000);
        #expect(residency == .paged);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_round_full_attention_context_reserve_to_the_cache_growth_slab() throws {
        let projectedContextReserveBytes: UInt64? = Qwen35MoeArtifactExpertResidencyPolicy
            .fullAttentionSequenceStateBytes(
                fullAttentionLayerCount: 1,
                keyValueHeadCount: 2,
                headDimension: 128,
                bytesPerElement: 2,
                maximumPositionCount: 257);
        #expect(projectedContextReserveBytes == 524_288);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_cap_artifact_context_reserve_at_the_worker_context_limit() throws {
        let workerContextReserveBytes: UInt64? =
            Qwen35MoeArtifactExpertResidencyPolicy.contextWindowReserveBytes(
                fullAttentionLayerCount: 8,
                keyValueHeadCount: 2,
                headDimension: 256,
                bytesPerElement: 2,
                artifactMaximumPositionCount: 262_144,
                maximumContextTokenCount: 24_576);
        let artifactMaximumContextReserveBytes: UInt64? =
            Qwen35MoeArtifactExpertResidencyPolicy.contextWindowReserveBytes(
                fullAttentionLayerCount: 8,
                keyValueHeadCount: 2,
                headDimension: 256,
                bytesPerElement: 2,
                artifactMaximumPositionCount: 262_144,
                maximumContextTokenCount: 262_144);
        let boundedReserveBytes: UInt64 = try #require(workerContextReserveBytes);
        let artifactReserveBytes: UInt64 = try #require(artifactMaximumContextReserveBytes);
        #expect(boundedReserveBytes == MlxRamBudgetDefaults.BOOTSTRAP_CONTEXT_WINDOW_RESERVE_BYTES);
        #expect(boundedReserveBytes < artifactReserveBytes);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_fail_closed_when_full_attention_context_geometry_overflows() throws {
        let projectedContextReserveBytes: UInt64? = Qwen35MoeArtifactExpertResidencyPolicy
            .fullAttentionSequenceStateBytes(
                fullAttentionLayerCount: UInt64.max,
                keyValueHeadCount: UInt64.max,
                headDimension: UInt64.max,
                bytesPerElement: 4,
                maximumPositionCount: UInt64.max);
        #expect(projectedContextReserveBytes == nil);
    }
}
