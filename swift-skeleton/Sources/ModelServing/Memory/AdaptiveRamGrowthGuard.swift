import Foundation

/**
 * Forward-specific admission and transient-memory learning
 * (port of `budget/adaptive_growth.rs`).
 *
 * The guard answers a narrower question than `MlxRamBudget`: can one concrete
 * forward grow from the *current* MLX state without crossing stable or
 * expected peak limits? It uses exact persistent growth supplied by
 * decoder-state owners, one routed expert-page reservation, explicit
 * temporary workspace, and a transient reserve resolved from completed
 * forwards at the highest specificity available: the exact execution context
 * first, then a token-proportional estimate from the same phase, then the
 * global maximum for a phase with no evidence at all (issue #623).
 *
 * Three projections are retained for diagnostics:
 *
 * - `stable`: current active + persistent growth + routed page;
 * - `peak`: stable + explicit workspace + the resolved transient reserve;
 * - `recovery`: peak + one equal transient window.
 *
 * Stable and peak are admission boundaries. Recovery is deliberately
 * diagnostic. Preemptively evicting experts for a recovery-only shortfall
 * caused avoidable SSD paging on requests whose actual expected peak fitted.
 * If a real allocation still fails, the caller restores the request
 * checkpoint, reclaims the exact expert deficit, and retries the unchanged
 * forward once.
 */
public final class AdaptiveRamGrowthGuard {

    /// Stable ceiling C. A one-percent allowance P is derived for transient peak.
    private var activeMemoryCeilingBytes: Int

    /// High-water transient bytes keyed by execution shape and ownership mode.
    private var observedTransientHighWaterBytesByContext: [AdaptiveRamGrowthContext: Int] = [:]

    /**
     * Creates a guard for one machine-derived MLX active-memory limit.
     *
     * - Parameter activeMemoryCeilingBytes: Machine-derived stable ceiling C.
     * - Returns: A guard with no learned evidence yet.
     * - Throws: `AdaptiveRamGrowthGuardError.invalidActiveMemoryCeiling` when
     *   the limit is zero.
     */
    public init(activeMemoryCeilingBytes: Int) throws {
        if (activeMemoryCeilingBytes == 0) {
            throw AdaptiveRamGrowthGuardError.invalidActiveMemoryCeiling
        }
        self.activeMemoryCeilingBytes = activeMemoryCeilingBytes
        self.observedTransientHighWaterBytesByContext = [:]
    }

    /**
     * Replaces the active limit while retaining all exact-context measurements.
     *
     * - Parameter activeMemoryCeilingBytes: The new machine-derived ceiling.
     * - Throws: `AdaptiveRamGrowthGuardError.invalidActiveMemoryCeiling` when
     *   the limit is zero.
     */
    public func updateActiveMemoryCeilingBytes(_ activeMemoryCeilingBytes: Int) throws {
        if (activeMemoryCeilingBytes == 0) {
            throw AdaptiveRamGrowthGuardError.invalidActiveMemoryCeiling
        }
        self.activeMemoryCeilingBytes = activeMemoryCeilingBytes
    }

    /**
     * Returns whether this phase has at least one retained exact-context
     * observation.
     *
     * - Parameter memoryPhase: The request lifecycle phase to inspect.
     * - Returns: True when the phase owns at least one completed observation.
     */
    public func hasCompletedGrowthObservation(memoryPhase: MemoryPhase) -> Bool {
        let phaseObservationContexts: [AdaptiveRamGrowthContext] = self
            .observedTransientHighWaterBytesByContext
            .keys
            .filter({ (observedContext: AdaptiveRamGrowthContext) in
                observedContext.memoryPhase == memoryPhase
            })
        return !phaseObservationContexts.isEmpty
    }

    /**
     * Returns the phase maximum for phase-specific telemetry.
     *
     * - Parameter memoryPhase: The request lifecycle phase to inspect.
     * - Returns: The largest observed high-water in the phase, or zero.
     */
    public func observedTransientHighWaterBytes(memoryPhase: MemoryPhase) -> Int {
        let phaseHighWaterBytes: [Int] = self.observedTransientHighWaterBytesByContext
            .filter({ (observationEntry: (key: AdaptiveRamGrowthContext, value: Int)) in
                observationEntry.key.memoryPhase == memoryPhase
            })
            .map({ (observationEntry: (key: AdaptiveRamGrowthContext, value: Int)) -> Int in
                observationEntry.value
            })
        return phaseHighWaterBytes.max() ?? 0
    }

    /**
     * Returns transient evidence for this exact execution context.
     *
     * - Parameter adaptiveRamGrowthContext: The exact execution shape to look up.
     * - Returns: The context's high-water, or zero when it was never observed.
     */
    public func observedTransientHighWaterBytesForContext(
        _ adaptiveRamGrowthContext: AdaptiveRamGrowthContext
    ) -> Int {
        return self.observedTransientHighWaterBytesByContext[adaptiveRamGrowthContext] ?? 0
    }

    /**
     * Returns the largest transient window observed across all completed
     * forward phases. Callers without a concrete forward context use this
     * conservative value; forward admission resolves a shape-fitted reserve
     * through `admissionTransientReserveForContext`.
     *
     * - Returns: The all-phase maximum high-water, or zero when nothing was
     *   observed yet.
     */
    public func admissionTransientHighWaterBytes() -> Int {
        let everyHighWaterBytes: [Int] = Array(
            self.observedTransientHighWaterBytesByContext.values)
        return everyHighWaterBytes.max() ?? 0
    }

    /**
     * Resolves the transient reserve for one concrete forward context.
     *
     * The ladder widens one dimension class at a time, and each level has a
     * distinct evidence basis:
     *
     * 1. **Exact context** — the same phase, token count, position bucket,
     *    and ownership mode completed a forward before; its high-water is the
     *    tightest known bound for this operation.
     * 2. **Phase-scaled** — no exact observation, but the phase completed
     *    other shapes. Activation workspace scales with the forward's token
     *    count, so each observation is rescaled to this forward's token count
     *    (ceiling division) and the largest scaled value wins. An 8,192-token
     *    chunk's 8 GB window therefore reserves about 2 GB for a 2,048-token
     *    chunk instead of borrowing the absolute 8 GB.
     * 3. **Global maximum** — the phase has no evidence at all. The first
     *    forward of a new phase reserves the largest window ever observed;
     *    its completion records that phase's own evidence and every later
     *    forward in the phase reserves proportionally.
     *
     * Under-projection remains recoverable by design: a typed allocation
     * failure restores the request checkpoint, reclaims the exact deficit,
     * and retries the unchanged forward, which also records the missing
     * shape's evidence. Over-reservation has no such self-correction — it
     * demotes expert ownership outright — so specificity wins over blanket
     * conservatism here.
     *
     * - Parameter adaptiveRamGrowthContext: The forward's exact execution shape.
     * - Returns: The reserve bytes and the evidence level that supplied them.
     */
    public func admissionTransientReserveForContext(
        _ adaptiveRamGrowthContext: AdaptiveRamGrowthContext
    ) -> (reserveBytes: Int, reserveSource: AdaptiveRamGrowthTransientReserveSource) {
        if let exactContextHighWaterBytes: Int =
            self.observedTransientHighWaterBytesByContext[adaptiveRamGrowthContext] {
            return (exactContextHighWaterBytes, .exactContext)
        }
        let scaledPhaseReserveBytes: [Int] = self.observedTransientHighWaterBytesByContext
            .filter({ (observationEntry: (key: AdaptiveRamGrowthContext, value: Int)) in
                observationEntry.key.memoryPhase == adaptiveRamGrowthContext.memoryPhase
            })
            .compactMap({ (observationEntry: (key: AdaptiveRamGrowthContext, value: Int)) -> Int? in
                AdaptiveRamGrowthTransientReserveScaling.scaleToTokenCount(
                    observedHighWaterBytes: observationEntry.value,
                    observedForwardTokenCount: observationEntry.key.forwardTokenCount,
                    targetForwardTokenCount: adaptiveRamGrowthContext.forwardTokenCount)
            })
        if let largestScaledReserveBytes: Int = scaledPhaseReserveBytes.max() {
            return (largestScaledReserveBytes, .phaseScaled)
        }
        return (self.admissionTransientHighWaterBytes(), .globalMaximum)
    }

    /**
     * Returns the retained-payload ceiling that keeps the adaptive growth
     * guard's peak projection inside its limits: the active ceiling plus its
     * transient allowance, minus current active memory, the learned transient
     * reserve, and one routed-page reservation for the next forward, expressed
     * relative to current retained ownership.
     *
     * Decode warming allocates persistent tables that the next forward's
     * admission would otherwise count against the peak limit. Capping warming
     * at this headroom — and reclaiming tables when the headroom is negative —
     * keeps the hot-expert cache from taking memory the adaptive growth guard
     * must hold for transients and KV growth.
     *
     * - Parameters:
     *   - memoryPhase: The phase warming runs inside.
     *   - currentActiveMemoryBytes: MLX active bytes sampled now.
     *   - currentRetainedPayloadBytes: Retained expert payload bytes now.
     *   - routedExpertPageReservationBytes: One routed-page reservation.
     * - Returns: The maximum retained payload warming may hold.
     */
    public func hotExpertRetentionCeilingBytes(
        memoryPhase: MemoryPhase,
        currentActiveMemoryBytes: Int,
        currentRetainedPayloadBytes: Int,
        routedExpertPageReservationBytes: Int
    ) -> Int {
        let transientAllowanceBytes: Int = self.activeMemoryCeilingBytes / 100
        let (allowedSumBytes, allowedOverflowed) = self.activeMemoryCeilingBytes
            .addingReportingOverflow(transientAllowanceBytes)
        let allowedActiveMemoryBytes: Int = allowedOverflowed ? Int.max : allowedSumBytes
        // Warming runs inside one phase, so the reserve it must respect is that
        // phase's own learned workspace. Capping decode warming at the largest
        // workspace ever seen in any phase let one huge prefill suppress decode
        // warming forever (issue #512): the prefill transient is spent by the
        // time decode runs, and prefill admission still reserves against the
        // all-phase maximum through `projectGrowthForContext`, so a later
        // large prefill reclaims warm tables through the existing pressure path
        // instead of needing warming to have predicted it. Before this phase has
        // any observation the all-phase maximum keeps the first warming steps
        // conservative.
        let phaseTransientReserveBytes: Int = self.hasCompletedGrowthObservation(
            memoryPhase: memoryPhase)
            ? self.observedTransientHighWaterBytes(memoryPhase: memoryPhase)
            : self.admissionTransientHighWaterBytes()
        var signedHeadroomBytes: Int = allowedActiveMemoryBytes
        let reservedByteStreams: [Int] = [
            currentActiveMemoryBytes,
            phaseTransientReserveBytes,
            routedExpertPageReservationBytes,
        ]
        for reservedBytes: Int in reservedByteStreams {
            let (remainingHeadroomBytes, headroomUnderflowed) = signedHeadroomBytes
                .subtractingReportingOverflow(reservedBytes)
            signedHeadroomBytes = headroomUnderflowed ? Int.min : remainingHeadroomBytes
        }
        let (signedCeilingSumBytes, ceilingOverflowed) = currentRetainedPayloadBytes
            .addingReportingOverflow(signedHeadroomBytes)
        var signedCeilingBytes: Int = ceilingOverflowed
            ? (signedHeadroomBytes < 0 ? Int.min : Int.max)
            : signedCeilingSumBytes
        if (signedCeilingBytes < 0) {
            signedCeilingBytes = 0
        }
        return signedCeilingBytes
    }

    /**
     * Builds a checked C-stable and P-peak projection from exact-context
     * evidence.
     *
     * The reserve is taken from the most specific evidence available and
     * widened only when it is missing (issue #623): the exact context first,
     * then a token-proportional estimate from the same phase's observations,
     * then the global maximum for a phase with no evidence at all. Charging
     * one shape's absolute transient window to every other shape demoted fully
     * resident expert owners whose own chunk needed a fraction of it.
     *
     * - Parameters:
     *   - adaptiveRamGrowthContext: The forward's exact execution shape.
     *   - currentActiveMemoryBytes: MLX active bytes sampled before the decision.
     *   - exactPersistentGrowthBytes: Exact persistent growth this forward creates.
     *   - routedExpertPageReservationBytes: Maximum bounded expert page that may
     *     coexist with this forward.
     *   - exactTemporaryWorkspaceBytes: Known one-operation workspace.
     * - Returns: The checked projection evidence.
     * - Throws: `AdaptiveRamGrowthGuardError.memoryProjectionOverflow` when any
     *   projection step exceeds the word size.
     */
    public func projectGrowthForContext(
        _ adaptiveRamGrowthContext: AdaptiveRamGrowthContext,
        currentActiveMemoryBytes: Int,
        exactPersistentGrowthBytes: Int,
        routedExpertPageReservationBytes: Int,
        exactTemporaryWorkspaceBytes: Int
    ) throws -> AdaptiveRamGrowthProjection {
        let (observedTransientHighWaterBytes, transientReserveSource) =
            self.admissionTransientReserveForContext(adaptiveRamGrowthContext)
        var stableProjectedBytes: Int = currentActiveMemoryBytes
        let persistentByteStreams: [Int] = [
            exactPersistentGrowthBytes,
            routedExpertPageReservationBytes,
        ]
        for persistentGrowthBytes: Int in persistentByteStreams {
            let (summedStableBytes, stableOverflowed) = stableProjectedBytes
                .addingReportingOverflow(persistentGrowthBytes)
            if (stableOverflowed) {
                throw AdaptiveRamGrowthGuardError.memoryProjectionOverflow
            }
            stableProjectedBytes = summedStableBytes
        }
        // Explicit workspace and learned transient history are additive. Taking
        // only their maximum would under-reserve when a new operation introduces
        // known workspace on top of ordinary activation behavior.
        let (predictedTransientBytes, transientOverflowed) = exactTemporaryWorkspaceBytes
            .addingReportingOverflow(observedTransientHighWaterBytes)
        if (transientOverflowed) {
            throw AdaptiveRamGrowthGuardError.memoryProjectionOverflow
        }
        let (peakProjectedBytes, peakOverflowed) = stableProjectedBytes
            .addingReportingOverflow(predictedTransientBytes)
        if (peakOverflowed) {
            throw AdaptiveRamGrowthGuardError.memoryProjectionOverflow
        }
        let (recoveryProjectedBytes, recoveryOverflowed) = peakProjectedBytes
            .addingReportingOverflow(predictedTransientBytes)
        if (recoveryOverflowed) {
            throw AdaptiveRamGrowthGuardError.memoryProjectionOverflow
        }
        // P is the repository's approved temporary allowance. Stable ownership
        // must fit C; a short-lived peak may use C + 1 percent.
        let transientAllowanceBytes: Int = self.activeMemoryCeilingBytes / 100
        let (allowedSumBytes, allowedOverflowed) = self.activeMemoryCeilingBytes
            .addingReportingOverflow(transientAllowanceBytes)
        let allowedActiveMemoryBytes: Int = allowedOverflowed ? Int.max : allowedSumBytes
        return AdaptiveRamGrowthProjection(
            currentActiveMemoryBytes: currentActiveMemoryBytes,
            exactPersistentGrowthBytes: exactPersistentGrowthBytes,
            routedExpertPageReservationBytes: routedExpertPageReservationBytes,
            exactTemporaryWorkspaceBytes: exactTemporaryWorkspaceBytes,
            observedTransientHighWaterBytes: observedTransientHighWaterBytes,
            transientReserveSource: transientReserveSource,
            stableProjectedBytes: stableProjectedBytes,
            peakProjectedBytes: peakProjectedBytes,
            recoveryProjectedBytes: recoveryProjectedBytes,
            activeMemoryCeilingBytes: self.activeMemoryCeilingBytes,
            allowedActiveMemoryBytes: allowedActiveMemoryBytes)
    }

    /**
     * Retains only recurring-context transient evidence after a successful
     * forward.
     *
     * - Parameters:
     *   - adaptiveRamGrowthContext: The forward's exact execution shape.
     *   - shouldRetainObservation: Whether the forward counts as reusable evidence.
     *   - activeMemoryBytesBeforeGrowth: MLX active bytes sampled before.
     *   - activeMemoryBytesAfterGrowth: MLX active bytes sampled after.
     *   - peakMemoryBytesDuringGrowth: MLX active bytes at the forward's peak.
     *   - exactTemporaryWorkspaceBytes: Known one-operation workspace.
     */
    public func recordCompletedGrowthForContext(
        _ adaptiveRamGrowthContext: AdaptiveRamGrowthContext,
        shouldRetainObservation: Bool,
        activeMemoryBytesBeforeGrowth: Int,
        activeMemoryBytesAfterGrowth: Int,
        peakMemoryBytesDuringGrowth: Int,
        exactTemporaryWorkspaceBytes: Int
    ) {
        self.recordCompletedGrowthExcludingExpertPageStream(
            adaptiveRamGrowthContext,
            shouldRetainObservation: shouldRetainObservation,
            activeMemoryBytesBeforeGrowth: activeMemoryBytesBeforeGrowth,
            activeMemoryBytesAfterGrowth: activeMemoryBytesAfterGrowth,
            peakMemoryBytesDuringGrowth: peakMemoryBytesDuringGrowth,
            exactTemporaryWorkspaceBytes: exactTemporaryWorkspaceBytes,
            retainedExpertPayloadGrowthBytes: 0,
            promotedExpertPageStreamBytes: 0)
    }

    /**
     * Retains recurring-context transient evidence after a forward that
     * promoted mandatory expert pages.
     *
     * `retainedExpertPayloadGrowthBytes` is the resident-payload delta the
     * forward created and `promotedExpertPageStreamBytes` is everything that
     * forward streamed from storage. Pages that stayed resident are already
     * excluded because the post-forward active sample (`max` below) includes
     * the retention delta; subtracting them again would erase real activation
     * headroom. Pages that were streamed and evicted appear only in the peak,
     * so they must leave the residual — otherwise every future activation
     * reserve derived from this evidence inflates by the streamed byte size
     * and shrinks the expert-retention budget (issue #691).
     *
     * - Parameters:
     *   - adaptiveRamGrowthContext: The forward's exact execution shape.
     *   - shouldRetainObservation: Whether the forward counts as reusable evidence.
     *   - activeMemoryBytesBeforeGrowth: MLX active bytes sampled before.
     *   - activeMemoryBytesAfterGrowth: MLX active bytes sampled after.
     *   - peakMemoryBytesDuringGrowth: MLX active bytes at the forward's peak.
     *   - exactTemporaryWorkspaceBytes: Known one-operation workspace.
     *   - retainedExpertPayloadGrowthBytes: Resident-payload delta the forward created.
     *   - promotedExpertPageStreamBytes: Page bytes the forward streamed from storage.
     */
    public func recordCompletedGrowthExcludingExpertPageStream(
        _ adaptiveRamGrowthContext: AdaptiveRamGrowthContext,
        shouldRetainObservation: Bool,
        activeMemoryBytesBeforeGrowth: Int,
        activeMemoryBytesAfterGrowth: Int,
        peakMemoryBytesDuringGrowth: Int,
        exactTemporaryWorkspaceBytes: Int,
        retainedExpertPayloadGrowthBytes: Int,
        promotedExpertPageStreamBytes: Int
    ) {
        if (!shouldRetainObservation) {
            return
        }
        // The post-forward active sample already includes every newly retained
        // expert page. Using it as the stable baseline excludes expert growth
        // from the transient window. Subtracting expert growth again would erase
        // real activation headroom and let the next forward overfill retention.
        let stableActiveMemoryBytes: Int = max(
            activeMemoryBytesBeforeGrowth,
            activeMemoryBytesAfterGrowth)
        // Streamed page bytes inside the peak but outside that stable baseline:
        // promotions beyond the resident delta were read in and evicted, so they
        // are expert-page spike ownership, never activation (issue #691).
        let expertPageBytesOutsideStableBaseline: Int = SaturatingArithmetic.subtractInt(
            promotedExpertPageStreamBytes,
            retainedExpertPayloadGrowthBytes)
        var observedTransientGrowthBytes: Int = SaturatingArithmetic.subtractInt(
            peakMemoryBytesDuringGrowth,
            stableActiveMemoryBytes)
        observedTransientGrowthBytes = SaturatingArithmetic.subtractInt(
            observedTransientGrowthBytes,
            exactTemporaryWorkspaceBytes)
        observedTransientGrowthBytes = SaturatingArithmetic.subtractInt(
            observedTransientGrowthBytes,
            expertPageBytesOutsideStableBaseline)
        // Store the residual only. Explicit workspace is supplied again by the
        // next operation; retaining it in learned history would double-count it.
        let existingHighWaterBytes: Int? =
            self.observedTransientHighWaterBytesByContext[adaptiveRamGrowthContext]
        let retainedHighWaterBytes: Int = max(
            existingHighWaterBytes ?? 0,
            observedTransientGrowthBytes)
        self.observedTransientHighWaterBytesByContext[adaptiveRamGrowthContext] =
            retainedHighWaterBytes
    }
}
