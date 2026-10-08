import Foundation

/**
 * Checked projection evidence for one adaptive growth operation
 * (port of `budget/adaptive_growth_projection.rs`).
 *
 * The immutable outcome of projecting one concrete forward against the stable
 * ceiling, the expected peak, and the diagnostic recovery window. Keeping it
 * beside the transient-reserve vocabulary separates *what admission decided
 * from* (the projection and the source of its reserve) from *how evidence is
 * learned* (the guard). The guard is the single constructor; all external
 * readers go through the exposed fields and derived quantities below.
 */
public struct AdaptiveRamGrowthProjection: Equatable, Hashable, Sendable {

    /// MLX active bytes sampled immediately before this admission decision.
    public let currentActiveMemoryBytes: Int

    /// Exact key/value, recurrent, and caller-declared persistent growth.
    public let exactPersistentGrowthBytes: Int

    /// Maximum bounded expert page that may coexist with this forward.
    public let routedExpertPageReservationBytes: Int

    /// Known one-operation workspace not represented by learned history.
    public let exactTemporaryWorkspaceBytes: Int

    /// Conservative reusable transient evidence resolved for this context.
    public let observedTransientHighWaterBytes: Int

    /// Which evidence level supplied `observedTransientHighWaterBytes`.
    public let transientReserveSource: AdaptiveRamGrowthTransientReserveSource

    /// Stable bytes after persistent state growth and before temporary work.
    public let stableProjectedBytes: Int

    /// Stable bytes plus the exact-context transient high-water window.
    public let peakProjectedBytes: Int

    /// Peak bytes plus one equal diagnostic recovery window.
    public let recoveryProjectedBytes: Int

    /// The stable boundary C this projection was checked against.
    public let activeMemoryCeilingBytes: Int

    /// The configured ceiling plus its approved one-percent transient allowance.
    public let allowedActiveMemoryBytes: Int

    /// Internal because the guard is the single constructor.
    init(
        currentActiveMemoryBytes: Int,
        exactPersistentGrowthBytes: Int,
        routedExpertPageReservationBytes: Int,
        exactTemporaryWorkspaceBytes: Int,
        observedTransientHighWaterBytes: Int,
        transientReserveSource: AdaptiveRamGrowthTransientReserveSource,
        stableProjectedBytes: Int,
        peakProjectedBytes: Int,
        recoveryProjectedBytes: Int,
        activeMemoryCeilingBytes: Int,
        allowedActiveMemoryBytes: Int
    ) {
        self.currentActiveMemoryBytes = currentActiveMemoryBytes
        self.exactPersistentGrowthBytes = exactPersistentGrowthBytes
        self.routedExpertPageReservationBytes = routedExpertPageReservationBytes
        self.exactTemporaryWorkspaceBytes = exactTemporaryWorkspaceBytes
        self.observedTransientHighWaterBytes = observedTransientHighWaterBytes
        self.transientReserveSource = transientReserveSource
        self.stableProjectedBytes = stableProjectedBytes
        self.peakProjectedBytes = peakProjectedBytes
        self.recoveryProjectedBytes = recoveryProjectedBytes
        self.activeMemoryCeilingBytes = activeMemoryCeilingBytes
        self.allowedActiveMemoryBytes = allowedActiveMemoryBytes
    }

    /// The complete non-expert reserve that expert residency must leave
    /// available for this admitted forward through its expected peak.
    ///
    /// Recovery-only shortfall is diagnostic and handled by typed allocation-
    /// failure rollback, exact reclamation, and retry. Passing the expected-peak
    /// difference into the residency planner keeps both policy owners on the
    /// same initial-admission equation.
    public var forwardReserveBytes: Int {
        return SaturatingArithmetic.subtractInt(
            self.peakProjectedBytes,
            self.currentActiveMemoryBytes)
    }

    /// The exact retained-expert reclamation needed by stable and peak work.
    public var operationReclamationRequiredBytes: Int {
        let stableDeficitBytes: Int = SaturatingArithmetic.subtractInt(
            self.stableProjectedBytes,
            self.activeMemoryCeilingBytes)
        let peakDeficitBytes: Int = SaturatingArithmetic.subtractInt(
            self.peakProjectedBytes,
            self.allowedActiveMemoryBytes)
        return max(stableDeficitBytes, peakDeficitBytes)
    }

    /// The diagnostic recovery-reserve shortfall against the transient ceiling.
    public var recoveryReserveShortfallBytes: Int {
        return SaturatingArithmetic.subtractInt(
            self.recoveryProjectedBytes,
            self.allowedActiveMemoryBytes)
    }

    /// Whether stable and expected-peak work fit without any reclamation.
    public var fitsStableAndPeakLimits: Bool {
        return self.operationReclamationRequiredBytes == 0
    }

    /// Whether the diagnostic recovery window fits the transient allowance too.
    public var hasFullRecoveryReserve: Bool {
        return self.recoveryReserveShortfallBytes == 0
    }

    /**
     * Plans preemptive reclamation for stable and expected-peak deficits.
     *
     * Recovery remains diagnostic. A recovery-only shortfall is handled by the
     * typed allocation-failure checkpoint, exact reclamation, and retry path.
     * The peak is passed twice on purpose: `ExpertReclamationPlan` is a pure
     * three-boundary formula also used by stricter callers, and replacing
     * recovery with peak here excludes recovery-only deficits from preemptive
     * eviction while preserving one shared checked-arithmetic implementation.
     *
     * - Parameter retainedExpertPayloadBytes: Resident expert payload bytes
     *   available to reclaim.
     * - Returns: The smallest plan satisfying both admission boundaries.
     */
    public func expertRetentionReclamationPlan(
        retainedExpertPayloadBytes: Int
    ) -> ExpertReclamationPlan {
        return ExpertReclamationPlan.forProjectedMemory(
            stableProjectedBytes: self.stableProjectedBytes,
            peakProjectedBytes: self.peakProjectedBytes,
            recoveryProjectedBytes: self.peakProjectedBytes,
            stableMemoryCeilingBytes: self.activeMemoryCeilingBytes,
            transientMemoryCeilingBytes: self.allowedActiveMemoryBytes,
            retainedExpertPayloadBytes: retainedExpertPayloadBytes)
    }
}
