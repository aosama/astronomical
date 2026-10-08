import Foundation

/// Single-source owner for the MLX RAM policy split across streaming and
/// expert retention (port of `budget/ram.rs`).
///
/// ```text
/// retained_expert_budget_bytes =
///     mlx_active_memory_ceiling_bytes
///     - model_core_payload_bytes
///     - context_window_reserve_bytes
///     - activation_headroom_bytes
///     - complete_layer_stream_slot_bytes
///     - other_fixed_bytes
/// ```
///
/// Categories are intentionally non-overlapping: model core, context reserve,
/// activation headroom, one complete-layer stream slot, other fixed owners,
/// then retained experts receive only what every required owner left behind.
/// An idle status snapshot can make context and streaming space appear unused;
/// that does not make the bytes available to expert retention because the next
/// request needs those categories concurrently with retained experts.
///
/// Arithmetic saturates while composing a plan: saturation intentionally fails
/// closed by reducing the leftover expert budget to zero instead of wrapping to
/// a large value. Learning is high-water-only; one cheaper request cannot erase
/// evidence required by a larger future request.
public final class MlxRamBudget {

    /// Bootstrap context-window reserve before any live measurement exists
    /// (1 GB SI).
    public static let defaultBootstrapContextWindowReserveBytes: UInt64 = 1_000_000_000

    /// Coarse token buckets keep context-growth evidence comparable across
    /// memory policies.
    internal static let contextTokenBucketWidth: UInt64 = 1_024

    /// Extra safety margin applied on top of measured context-window need.
    /// It absorbs measurement jitter and small untracked MLX bookkeeping and is
    /// added before high-water comparison so every stored bucket is directly
    /// usable as a future reserve.
    private static let contextWindowMeasurementSafetyBufferBytes: UInt64 = 64_000_000

    /// User/machine-resolved hard production ceiling; never a model-specific constant.
    public private(set) var mlxActiveMemoryCeilingBytes: UInt64

    /// Startup-validated payload geometry for the currently loaded model.
    public private(set) var modelGeometry: MlxRamBudgetModelGeometry

    /// Conservative floor used until at least one real context measurement exists.
    internal let bootstrapContextWindowReserveBytes: UInt64

    /// High-water request workspace keyed by 1,024-token context buckets.
    internal private(set) var measuredContextWindowHighWaterByTokenBucket: [UInt64: UInt64] = [:]

    /// Phase-specific activation evidence. Decode keeps one scalar high-water;
    /// prefill evidence is keyed by 1,024-token context buckets so one large
    /// request's workspace does not size every later promise. Values can grow
    /// but never shrink.
    internal private(set) var prefillActivationHighWaterByTokenBucket: [UInt64: UInt64] = [:]

    internal private(set) var decodeActivationHeadroomBytes: UInt64 = 0

    /// Distinguishes an unobserved prefill phase from a valid zero-byte sample.
    internal private(set) var hasPrefillActivationMeasurement: Bool = false

    /// Distinguishes an unobserved decode phase from a valid zero-byte sample.
    internal private(set) var hasDecodeActivationMeasurement: Bool = false

    /// Distinguishes "no evidence" from a valid measured value of zero.
    public private(set) var hasContextWindowMeasurement: Bool = false

    /// Creates the owner with the 1 GB SI bootstrap context-window reserve.
    public convenience init(
        mlxActiveMemoryCeilingBytes: UInt64,
        modelGeometry: MlxRamBudgetModelGeometry
    ) throws {
        try self.init(
            mlxActiveMemoryCeilingBytes: mlxActiveMemoryCeilingBytes,
            modelGeometry: modelGeometry,
            bootstrapContextWindowReserveBytes: MlxRamBudget.defaultBootstrapContextWindowReserveBytes)
    }

    public init(
        mlxActiveMemoryCeilingBytes: UInt64,
        modelGeometry: MlxRamBudgetModelGeometry,
        bootstrapContextWindowReserveBytes: UInt64
    ) throws {
        if mlxActiveMemoryCeilingBytes == 0 {
            throw MlxRamBudgetError.invalidCeiling
        }
        self.mlxActiveMemoryCeilingBytes = mlxActiveMemoryCeilingBytes
        self.modelGeometry = modelGeometry
        self.bootstrapContextWindowReserveBytes = bootstrapContextWindowReserveBytes
    }

    /// Keeps learned model/workload evidence across live ceiling changes: the
    /// same bytes are re-composed against the new ceiling on the next plan.
    public func updateMlxActiveMemoryCeilingBytes(_ newCeilingBytes: UInt64) throws {
        if newCeilingBytes == 0 {
            throw MlxRamBudgetError.invalidCeiling
        }
        mlxActiveMemoryCeilingBytes = newCeilingBytes
    }

    /// The owner follows the loaded model. Callers must provide geometry from
    /// validated artifacts, not infer payload sizes from model names.
    public func updateModelGeometry(_ newGeometry: MlxRamBudgetModelGeometry) {
        modelGeometry = newGeometry
    }

    /// Context-window reserve for `contextTokenCount` tokens.
    ///
    /// Before measurements: the larger of the 1 GB SI bootstrap and the
    /// geometry-projected sequence-state payload. After measurements:
    /// conservative high-water by token bucket, plus a geometry projection for
    /// any tokens beyond the highest measured bucket.
    public func contextWindowReserveBytes(_ contextTokenCount: UInt64) -> UInt64 {
        let projectedSequenceStateBytes = SaturatingArithmetic.multiply(
            contextTokenCount,
            modelGeometry.sequenceStateBytesPerToken)
        if !hasContextWindowMeasurement {
            return max(bootstrapContextWindowReserveBytes, projectedSequenceStateBytes)
        }
        let tokenBucket = MlxRamBudget.contextTokenBucket(contextTokenCount)
        // Context memory normally grows with position, so the highest evidence
        // at or below the requested bucket wins: a smaller earlier observation
        // must not override a larger known lower-bucket high-water.
        let atOrBelowBucketBytes = MlxRamBudget.highestValueAtOrBelowBucket(
            tokenBucket, in: measuredContextWindowHighWaterByTokenBucket)
        // If this process has measured only larger requests, reuse the smallest
        // larger measurement rather than pretending the new smaller bucket has
        // no cost. This is conservative until direct evidence for the bucket exists.
        let largerBucketFloorBytes = MlxRamBudget.lowestValueAtOrAboveBucket(
            tokenBucket, in: measuredContextWindowHighWaterByTokenBucket)
        let learnedReserveBytes = atOrBelowBucketBytes > 0
            ? atOrBelowBucketBytes
            : largerBucketFloorBytes
        let measuredOrBootstrapReserveBytes = max(
            learnedReserveBytes, bootstrapContextWindowReserveBytes)
        guard let highestMeasuredBucket = MlxRamBudget.highestMeasuredBucket(
            in: measuredContextWindowHighWaterByTokenBucket)
        else {
            return measuredOrBootstrapReserveBytes
        }
        let highestMeasuredTokenCount = SaturatingArithmetic.multiply(
            SaturatingArithmetic.add(highestMeasuredBucket, 1),
            MlxRamBudget.contextTokenBucketWidth)
        if contextTokenCount > highestMeasuredTokenCount {
            let unmeasuredTokenCount = SaturatingArithmetic.subtract(
                contextTokenCount, highestMeasuredTokenCount)
            return SaturatingArithmetic.add(
                measuredOrBootstrapReserveBytes,
                SaturatingArithmetic.multiply(
                    unmeasuredTokenCount, modelGeometry.sequenceStateBytesPerToken))
        }
        return max(measuredOrBootstrapReserveBytes, projectedSequenceStateBytes)
    }

    /// Records one live observation and never lowers prior high-water evidence.
    public func recordMeasurement(_ measurement: MlxRamBudgetMeasurement) {
        hasContextWindowMeasurement = true
        let contextBucket = MlxRamBudget.contextTokenBucket(measurement.contextTokenCount)
        // The forward sample includes both persistent context and transient
        // activation bytes. Activation has its own separately composed owner,
        // so subtract that evidence before learning context or the same
        // workspace is reserved twice. Saturation fails safe when independently
        // sampled transient evidence is larger because of allocator timing.
        let measuredPersistentContextBytes = SaturatingArithmetic.subtract(
            SaturatingArithmetic.subtract(
                measurement.measuredContextAndActivationBytes,
                measurement.observedActivationHeadroomBytes),
            measurement.exactTemporaryWorkspaceBytes)
        let measuredWithBufferBytes = SaturatingArithmetic.add(
            measuredPersistentContextBytes,
            MlxRamBudget.contextWindowMeasurementSafetyBufferBytes)
        let existingContextHighWater =
            measuredContextWindowHighWaterByTokenBucket[contextBucket] ?? 0
        measuredContextWindowHighWaterByTokenBucket[contextBucket] = max(
            existingContextHighWater, measuredWithBufferBytes)

        switch measurement.phase {
        case .prefill:
            hasPrefillActivationMeasurement = true
            let existingActivationHighWater =
                prefillActivationHighWaterByTokenBucket[contextBucket] ?? 0
            prefillActivationHighWaterByTokenBucket[contextBucket] = max(
                existingActivationHighWater, measurement.observedActivationHeadroomBytes)
        case .generationPreparation, .decode:
            // GenerationPreparation evidence belongs to the decode window: it
            // observes the same activation shape that generation will repeat.
            hasDecodeActivationMeasurement = true
            decodeActivationHeadroomBytes = max(
                decodeActivationHeadroomBytes, measurement.observedActivationHeadroomBytes)
        case .idle:
            // Idle has no activation operation to learn from; context evidence
            // above is still valid if a caller deliberately records an idle sample.
            break
        }
    }

    /// Composes the workspace reservation exactly as the request-admission path
    /// does before it checks the active-memory ceiling.
    ///
    /// Admission-side mirror of `plan(phase:contextTokenCount:…)`: the
    /// admission path composes the same owners (context growth, restore
    /// overlap, publication workspace, prefill activation) but historically
    /// computed the activation component from the total context token count,
    /// reintroducing the #644 scaling defect `plan` had already fixed. Exposing
    /// the composition gives the hermetic suite a direct contract on the
    /// numbers admission actually charges, so the two paths cannot silently
    /// diverge again.
    public func contextAdmissionWorkspaceSnapshot(
        totalContextTokens: UInt64,
        plannedPrefillOperationTokenCount: UInt64,
        restoreOverlapWorkspaceBytes: UInt64,
        directPublicationWorkspaceBytes: UInt64
    ) -> MlxRamBudgetSnapshot {
        let contextWindowReserve = contextWindowReserveBytes(totalContextTokens)
        let activationHeadroom = activationHeadroomBytes(
            .prefill, plannedPrefillOperationTokenCount)
        let completeLayerStreamSlotBytes = modelGeometry.largestCompleteExpertLayerBytes
        let fixedNonExpertBytes = SaturatingArithmetic.add(
            SaturatingArithmetic.add(
                SaturatingArithmetic.add(
                    SaturatingArithmetic.add(
                        modelGeometry.modelCorePayloadBytes,
                        contextWindowReserve),
                    activationHeadroom),
                completeLayerStreamSlotBytes),
            SaturatingArithmetic.add(
                restoreOverlapWorkspaceBytes, directPublicationWorkspaceBytes))
        let otherFixedBytes = SaturatingArithmetic.add(
            restoreOverlapWorkspaceBytes, directPublicationWorkspaceBytes)
        return MlxRamBudgetSnapshot(
            mlxActiveMemoryCeilingBytes: mlxActiveMemoryCeilingBytes,
            modelCorePayloadBytes: modelGeometry.modelCorePayloadBytes,
            contextWindowReserveBytes: contextWindowReserve,
            activationHeadroomBytes: activationHeadroom,
            completeLayerStreamSlotBytes: completeLayerStreamSlotBytes,
            otherFixedBytes: otherFixedBytes,
            retainedExpertBudgetBytes: SaturatingArithmetic.subtract(
                mlxActiveMemoryCeilingBytes, fixedNonExpertBytes))
    }

    /// Composes the full budget for one planned operation.
    ///
    /// The two token counts describe different scopes and must not be conflated
    /// (issue #644): `contextTokenCount` sizes the context-window reserve,
    /// which genuinely grows with the request's prompt, while
    /// `operationTokenCount` sizes the activation reserve, which belongs to the
    /// single forward being planned. A chunked prefill's activation workspace
    /// is a function of the chunk size — measured active memory stays flat
    /// across a 50K-token chunked prefill — so planning activation against the
    /// prompt length multiplied a per-chunk observation into a reserve several
    /// times the ceiling.
    public func plan(
        phase: MemoryPhase,
        contextTokenCount: UInt64,
        operationTokenCount: UInt64,
        otherFixedBytes: UInt64
    ) -> MlxRamBudgetSnapshot {
        // Idle refill happens after request arrays are released, so it needs no
        // live context reserve. Prefill/decode plans protect the next
        // operation's context even if the current allocator snapshot is lower.
        let contextWindowReserve: UInt64
        if phase == .idle {
            contextWindowReserve = 0
        } else {
            contextWindowReserve = contextWindowReserveBytes(contextTokenCount)
        }
        let activationHeadroom = activationHeadroomBytes(phase, operationTokenCount)
        // GenerationPreparation budgets like decode: token writing needs the
        // routed-page slot, not a complete-layer slot.
        let completeLayerStreamSlotBytes: UInt64
        switch phase {
        case .prefill:
            completeLayerStreamSlotBytes = modelGeometry.largestCompleteExpertLayerBytes
        case .generationPreparation, .decode:
            completeLayerStreamSlotBytes = modelGeometry.largestRoutedExpertPageBytes
        case .idle:
            completeLayerStreamSlotBytes = 0
        }
        let fixedNonExpertBytes = SaturatingArithmetic.add(
            SaturatingArithmetic.add(
                SaturatingArithmetic.add(
                    SaturatingArithmetic.add(
                        modelGeometry.modelCorePayloadBytes,
                        contextWindowReserve),
                    activationHeadroom),
                completeLayerStreamSlotBytes),
            otherFixedBytes)
        return MlxRamBudgetSnapshot(
            mlxActiveMemoryCeilingBytes: mlxActiveMemoryCeilingBytes,
            modelCorePayloadBytes: modelGeometry.modelCorePayloadBytes,
            contextWindowReserveBytes: contextWindowReserve,
            activationHeadroomBytes: activationHeadroom,
            completeLayerStreamSlotBytes: completeLayerStreamSlotBytes,
            otherFixedBytes: otherFixedBytes,
            retainedExpertBudgetBytes: SaturatingArithmetic.subtract(
                mlxActiveMemoryCeilingBytes, fixedNonExpertBytes))
    }

    /// Refines retained ownership against one concrete admitted forward.
    ///
    /// `currentActiveMemoryBytes` already includes current retained experts,
    /// while `admittedForwardReserveBytes` includes exact persistent growth,
    /// one phase-correct expert page, and expected transient work. Adding the
    /// remaining strict-ceiling capacity to current retention therefore yields
    /// the largest expert payload that may coexist with that forward.
    public func retainedExpertBudgetForAdmittedForward(
        currentActiveMemoryBytes: UInt64,
        currentRetainedExpertPayloadBytes: UInt64,
        admittedForwardReserveBytes: UInt64
    ) -> UInt64 {
        let projectedActiveMemoryBytes = SaturatingArithmetic.add(
            currentActiveMemoryBytes, admittedForwardReserveBytes)
        if projectedActiveMemoryBytes <= mlxActiveMemoryCeilingBytes {
            let remainingCeilingCapacity = SaturatingArithmetic.subtract(
                mlxActiveMemoryCeilingBytes, projectedActiveMemoryBytes)
            return SaturatingArithmetic.add(
                currentRetainedExpertPayloadBytes, remainingCeilingCapacity)
        }
        let ceilingExcessBytes = SaturatingArithmetic.subtract(
            projectedActiveMemoryBytes, mlxActiveMemoryCeilingBytes)
        return SaturatingArithmetic.subtract(
            currentRetainedExpertPayloadBytes, ceilingExcessBytes)
    }

    /// Reclamation needed so a fixed forward workspace fits the ceiling.
    public func expertReclamationBytesForFixedForward(
        currentActiveMemoryBytes: UInt64,
        retainedExpertPayloadBytes: UInt64,
        fixedForwardWorkspaceBytes: UInt64
    ) -> Int {
        ExpertMemoryAdmission.expertReclamationBytesToFitFixedForward(
            currentActiveMemoryBytes: Int(clamping: currentActiveMemoryBytes),
            retainedExpertPayloadBytes: Int(clamping: retainedExpertPayloadBytes),
            memoryCeilingBytes: Int(clamping: mlxActiveMemoryCeilingBytes),
            fixedForwardWorkspaceBytes: Int(clamping: fixedForwardWorkspaceBytes))
    }

    /// Separates request workspace from expert ownership acquired during a
    /// forward. The MLX peak includes both categories; charging newly retained
    /// experts to context learning would reserve those bytes again and evict
    /// the topology that produced the measurement.
    public static func measuredNonExpertForwardGrowthBytes(
        activeMemoryBytesBeforeGrowth: UInt64,
        peakMemoryBytesDuringGrowth: UInt64,
        retainedExpertPayloadBytesBeforeGrowth: UInt64,
        retainedExpertPayloadBytesAfterGrowth: UInt64
    ) -> UInt64 {
        let newlyRetainedExpertPayloadBytes = SaturatingArithmetic.subtract(
            retainedExpertPayloadBytesAfterGrowth,
            retainedExpertPayloadBytesBeforeGrowth)
        return SaturatingArithmetic.subtract(
            SaturatingArithmetic.subtract(
                peakMemoryBytesDuringGrowth, activeMemoryBytesBeforeGrowth),
            newlyRetainedExpertPayloadBytes)
    }

    /// Excludes mandatory expert-page streaming from a forward's learned growth.
    ///
    /// Paged prefill promotes expert pages whose payload appears in the MLX
    /// peak but is evicted before completion, so the resident-payload delta
    /// misses it and `measuredNonExpertForwardGrowthBytes` would falsely charge
    /// the stream to context and activation (issue #691). The peak's expert
    /// component beyond the pre-growth baseline is the larger of the retention
    /// delta and the promoted stream: the retention delta counts what stayed
    /// resident, the stream counts everything promoted, and promotion transfers
    /// bytes between the two without creating more. Saturating arithmetic fails
    /// safe when sampled evidence disagrees.
    public static func measuredNonExpertForwardGrowthBytesExcludingExpertPageStreaming(
        activeMemoryBytesBeforeGrowth: UInt64,
        peakMemoryBytesDuringGrowth: UInt64,
        retainedExpertPayloadBytesBeforeGrowth: UInt64,
        retainedExpertPayloadBytesAfterGrowth: UInt64,
        promotedExpertPageStreamBytes: UInt64
    ) -> UInt64 {
        let retainedExpertPayloadGrowthBytes = SaturatingArithmetic.subtract(
            retainedExpertPayloadBytesAfterGrowth,
            retainedExpertPayloadBytesBeforeGrowth)
        let expertBytesBeyondPreGrowthBaseline = max(
            retainedExpertPayloadGrowthBytes, promotedExpertPageStreamBytes)
        return SaturatingArithmetic.subtract(
            SaturatingArithmetic.subtract(
                peakMemoryBytesDuringGrowth, activeMemoryBytesBeforeGrowth),
            expertBytesBeyondPreGrowthBaseline)
    }

    // MARK: - Ordered evidence-bucket helpers

    /// Shared coarse bucket keeps context-growth evidence comparable across
    /// memory policies.
    internal static func contextTokenBucket(_ contextTokenCount: UInt64) -> UInt64 {
        contextTokenCount / contextTokenBucketWidth
    }

    /// Scales one learned byte quantity proportionally between token counts
    /// with ceiling division, so projections beyond measured evidence stay
    /// conservative.
    internal static func scaleBytesProportionallyToTokenCount(
        learnedBytes: UInt64,
        learnedTokenCount: UInt64,
        targetTokenCount: UInt64
    ) -> UInt64 {
        if learnedTokenCount == 0 || targetTokenCount == 0 {
            return 0
        }
        let scaledBytes = (UInt128(learnedBytes) * UInt128(targetTokenCount)
            + UInt128(learnedTokenCount) - 1) / UInt128(learnedTokenCount)
        if scaledBytes > UInt128(UInt64.max) {
            return UInt64.max
        }
        return UInt64(scaledBytes)
    }

    /// Highest measured value among buckets at or below `tokenBucket`
    /// (mirrors `BTreeMap::range(..=bucket).max()` with an empty-map default of 0).
    internal static func highestValueAtOrBelowBucket(
        _ tokenBucket: UInt64, in evidence: [UInt64: UInt64]
    ) -> UInt64 {
        var highestValue: UInt64 = 0
        for (measuredBucket, measuredBytes) in evidence where measuredBucket <= tokenBucket {
            if measuredBytes > highestValue {
                highestValue = measuredBytes
            }
        }
        return highestValue
    }

    /// Lowest measured value among buckets at or above `tokenBucket`
    /// (mirrors `BTreeMap::range(bucket..).min()` with an empty-map default of 0).
    internal static func lowestValueAtOrAboveBucket(
        _ tokenBucket: UInt64, in evidence: [UInt64: UInt64]
    ) -> UInt64 {
        var lowestValue: UInt64 = UInt64.max
        for (measuredBucket, measuredBytes) in evidence where measuredBucket >= tokenBucket {
            if measuredBytes < lowestValue {
                lowestValue = measuredBytes
            }
        }
        return lowestValue == UInt64.max ? 0 : lowestValue
    }

    /// The highest measured bucket, or nil when nothing was measured yet.
    internal static func highestMeasuredBucket(in evidence: [UInt64: UInt64]) -> UInt64? {
        evidence.keys.max()
    }

    /// The value recorded at the highest measured bucket, or nil when empty.
    internal static func valueAtHighestBucket(in evidence: [UInt64: UInt64]) -> UInt64? {
        guard let highestBucket = evidence.keys.max() else {
            return nil
        }
        return evidence[highestBucket]
    }

    /// The largest recorded value across every bucket, or 0 when empty.
    internal static func highestValue(in evidence: [UInt64: UInt64]) -> UInt64 {
        evidence.values.max() ?? 0
    }
}
