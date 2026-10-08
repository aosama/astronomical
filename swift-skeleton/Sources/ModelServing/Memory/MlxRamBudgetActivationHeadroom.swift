import Foundation

/// Activation-headroom resolution for the MLX RAM budget
/// (port of `budget/ram_activation.rs`).
///
/// The activation reserve belongs to one planned operation (one forward), so
/// it resolves from the operation's own token count and learned per-bucket
/// evidence — never from the prompt or total context length. Sizing it from
/// the context length multiplied a chunk-shaped observation into a reserve
/// several times the ceiling and rejected every later request (issue #690).
extension MlxRamBudget {

    /// Activation headroom for one planned operation.
    ///
    /// Prefill resolves the learned evidence for the planned operation size
    /// (highest at-or-below bucket, proportionally projected beyond the
    /// highest measured bucket) so one large request's workspace does not size
    /// every later promise. The static three-layer floor still bounds every
    /// prefill promise. Decode evidence stays a scalar high-water: one-token
    /// writes are activation-cheap and phase-independent.
    ///
    /// The token count is the **operation's own size** (one forward's token
    /// count), never the prompt or total context length.
    public func activationHeadroomBytes(
        _ phase: MemoryPhase, _ operationTokenCount: UInt64
    ) -> UInt64 {
        // A reserve above the ceiling can never be admitted, so a projection
        // beyond it is meaningless paper. Cap learned evidence at the ceiling:
        // measured observations keep their pinned dominance over the static
        // floor, but no projection — measured or scaled — may manufacture a
        // reserve the ceiling could never grant (issue #690).
        min(
            phaseActivationHeadroomBytes(phase, operationTokenCount),
            mlxActiveMemoryCeilingBytes)
    }

    private func phaseActivationHeadroomBytes(
        _ phase: MemoryPhase, _ operationTokenCount: UInt64
    ) -> UInt64 {
        switch phase {
        case .prefill:
            // One complete layer is enough to stream a single page. A seated
            // 38.6 GB model under a 40 GB ceiling still needs room for a
            // multi-token prefill working set; three layers is the measured
            // first-chunk overshoot on that shape. Keep the learned high-water
            // when it is larger.
            return ExpertMemoryAdmission.requiredCompleteResidencyActivationHeadroomBytes(
                startupActivationFloorBytes: SaturatingArithmetic.multiply(
                    modelGeometry.largestCompleteExpertLayerBytes, 3),
                observedTransientHighWaterBytes: learnedPrefillActivationHeadroomBytes(
                    operationTokenCount))
        case .generationPreparation, .decode:
            // GenerationPreparation budgets like decode: token writing is
            // activation-cheap.
            if hasDecodeActivationMeasurement {
                return decodeActivationHeadroomBytes
            }
            if hasPrefillActivationMeasurement {
                return ExpertMemoryAdmission.requiredCompleteResidencyActivationHeadroomBytes(
                    startupActivationFloorBytes: modelGeometry.largestCompleteExpertLayerBytes,
                    observedTransientHighWaterBytes: MlxRamBudget.highestValue(
                        in: prefillActivationHighWaterByTokenBucket))
            }
            // Decode follows prefill in the user journey. Until one decode
            // completes, the one-layer floor is the only live evidence
            // preventing warm fill from occupying transient space that the
            // first token immediately needs back.
            return ExpertMemoryAdmission.requiredCompleteResidencyActivationHeadroomBytes(
                startupActivationFloorBytes: modelGeometry.largestCompleteExpertLayerBytes,
                observedTransientHighWaterBytes: 0)
        case .idle:
            return 0
        }
    }

    /// Learned prefill activation evidence resolved for the planned operation.
    ///
    /// Within the measured span the highest at-or-bucket observation wins
    /// (activation grows with the attended context, so a smaller observation
    /// must not override a larger known lower bucket). Beyond the highest
    /// measured token count the highest evidence scales proportionally by
    /// token count, mirroring the context-window reserve's projection rule.
    private func learnedPrefillActivationHeadroomBytes(_ operationTokenCount: UInt64) -> UInt64 {
        guard
            let highestBucketHighWaterBytes = MlxRamBudget.valueAtHighestBucket(
                in: prefillActivationHighWaterByTokenBucket),
            let highestMeasuredBucket = MlxRamBudget.highestMeasuredBucket(
                in: prefillActivationHighWaterByTokenBucket)
        else {
            return 0
        }
        let highestMeasuredTokenCount = SaturatingArithmetic.multiply(
            SaturatingArithmetic.add(highestMeasuredBucket, 1),
            MlxRamBudget.contextTokenBucketWidth)
        let atOrBelowHighWaterBytes = MlxRamBudget.highestValueAtOrBelowBucket(
            MlxRamBudget.contextTokenBucket(operationTokenCount),
            in: prefillActivationHighWaterByTokenBucket)
        if operationTokenCount > highestMeasuredTokenCount {
            // Proportional projection beyond measured evidence; the ceiling
            // division keeps the projection conservative.
            let scaledHighWaterBytes = MlxRamBudget.scaleBytesProportionallyToTokenCount(
                learnedBytes: highestBucketHighWaterBytes,
                learnedTokenCount: highestMeasuredTokenCount,
                targetTokenCount: operationTokenCount)
            return max(atOrBelowHighWaterBytes, scaledHighWaterBytes)
        }
        return atOrBelowHighWaterBytes
    }
}
