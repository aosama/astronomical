import Foundation;

/**
 * The machine-adaptive expert-residency choice for one MoE artifact load:
 * resident keeps every routed expert in wired memory, paged binds none and
 * serves routed experts from storage through the page seam.
 */
public enum Qwen35MoeArtifactExpertResidency: Equatable {
    case resident;
    case paged;
}

/**
 * Pure residency arithmetic shared by every caller: an artifact serves
 * resident exactly when the resident core payload, the complete expert
 * payload, context reserve, and the largest resident gate/up fusion
 * transient fit the machine's MLX memory ceiling. Every input is structural
 * byte arithmetic — no machine assumption is embedded, and an unknown
 * ceiling fails toward paging.
 */
public enum Qwen35MoeArtifactExpertResidencyPolicy {

    public static let FULL_ATTENTION_CACHE_GROWTH_TOKEN_COUNT: UInt64 = 256;

    /// Reserves full-attention KV state up to the worker's effective context
    /// limit, capped by the positions the artifact supports. The worker limit
    /// bounds executable cache growth even when the artifact advertises a
    /// much larger theoretical position range.
    public static func contextWindowReserveBytes(
        fullAttentionLayerCount: UInt64,
        keyValueHeadCount: UInt64,
        headDimension: UInt64,
        bytesPerElement: UInt64,
        artifactMaximumPositionCount: UInt64,
        maximumContextTokenCount: UInt32
    ) -> UInt64? {
        let effectivePositionCount: UInt64 = min(
            UInt64(maximumContextTokenCount), artifactMaximumPositionCount);
        guard let projectedSequenceStateBytes: UInt64 = Self.fullAttentionSequenceStateBytes(
            fullAttentionLayerCount: fullAttentionLayerCount,
            keyValueHeadCount: keyValueHeadCount,
            headDimension: headDimension,
            bytesPerElement: bytesPerElement,
            maximumPositionCount: effectivePositionCount) else {
            return nil;
        }
        return max(
            MlxRamBudgetDefaults.BOOTSTRAP_CONTEXT_WINDOW_RESERVE_BYTES,
            projectedSequenceStateBytes);
    }

    /// Projects full-attention key/value storage from model geometry and the
    /// actual cache element width, rounding positions to the upstream cache
    /// slab size. `nil` indicates invalid geometry or arithmetic overflow.
    public static func fullAttentionSequenceStateBytes(
        fullAttentionLayerCount: UInt64,
        keyValueHeadCount: UInt64,
        headDimension: UInt64,
        bytesPerElement: UInt64,
        maximumPositionCount: UInt64
    ) -> UInt64? {
        guard fullAttentionLayerCount > 0,
            keyValueHeadCount > 0,
            headDimension > 0,
            bytesPerElement > 0,
            maximumPositionCount > 0 else {
            return nil;
        }
        let slabTokenCount: UInt64 = Self.FULL_ATTENTION_CACHE_GROWTH_TOKEN_COUNT;
        let (roundedNumerator, roundingOverflow) = maximumPositionCount
            .addingReportingOverflow(slabTokenCount - 1);
        guard !roundingOverflow else {
            return nil;
        }
        let roundedTokenCount: UInt64 = (roundedNumerator / slabTokenCount) * slabTokenCount;
        let geometryFactors: Array<UInt64> = [
            fullAttentionLayerCount,
            keyValueHeadCount,
            headDimension,
            2,
            bytesPerElement,
            roundedTokenCount,
        ];
        var projectedSequenceStateBytes: UInt64 = 1;
        for geometryFactor: UInt64 in geometryFactors {
            let (projectedBytes, multiplicationOverflow) = projectedSequenceStateBytes
                .multipliedReportingOverflow(by: geometryFactor);
            guard !multiplicationOverflow else {
                return nil;
            }
            projectedSequenceStateBytes = projectedBytes;
        }
        return projectedSequenceStateBytes;
    }

    /// Decides the residency for one artifact from its payload byte facts
    /// and the machine's MLX memory ceiling in bytes.
    public static func decide(
        residentPayloadBytes: UInt64,
        expertPayloadBytes: UInt64,
        contextWindowReserveBytes: UInt64,
        activationHeadroomBytes: UInt64,
        largestGateUpFusionTransientBytes: UInt64,
        mlxMemoryCeilingBytes: UInt64
    ) -> Qwen35MoeArtifactExpertResidency {
        if mlxMemoryCeilingBytes == 0 {
            return .paged;
        }
        let (residentPlusExpert, residentOverflow) = residentPayloadBytes
            .addingReportingOverflow(expertPayloadBytes);
        let (payloadAndContext, contextOverflow) = residentPlusExpert
            .addingReportingOverflow(contextWindowReserveBytes);
        let (payloadContextAndActivation, activationOverflow) = payloadAndContext
            .addingReportingOverflow(activationHeadroomBytes);
        let (totalDemand, totalOverflow) = payloadContextAndActivation
            .addingReportingOverflow(largestGateUpFusionTransientBytes);
        if residentOverflow || contextOverflow || activationOverflow || totalOverflow {
            return .paged;
        }
        return totalDemand <= mlxMemoryCeilingBytes ? .resident : .paged;
    }
}
