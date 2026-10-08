import Foundation

/// One live RAM observation that refines context-window reserve and activation
/// headroom (port of `ram_values.rs::MlxRamBudgetMeasurement`).
public struct MlxRamBudgetMeasurement: Equatable, Sendable {

    /// Execution class whose activation high-water this sample may raise.
    public let phase: MemoryPhase

    /// Context size used to choose a monotonic coarse learning bucket.
    public let contextTokenCount: UInt64

    /// Measured request-owned persistent and transient bytes above model core.
    public let measuredContextAndActivationBytes: UInt64

    /// Transient-only high-water independently learned by forward admission.
    public let observedActivationHeadroomBytes: UInt64

    /// Explicit operation workspace already reserved by forward admission.
    public let exactTemporaryWorkspaceBytes: UInt64

    public init(
        phase: MemoryPhase,
        contextTokenCount: UInt64,
        measuredContextAndActivationBytes: UInt64,
        observedActivationHeadroomBytes: UInt64,
        exactTemporaryWorkspaceBytes: UInt64
    ) {
        self.phase = phase
        self.contextTokenCount = contextTokenCount
        self.measuredContextAndActivationBytes = measuredContextAndActivationBytes
        self.observedActivationHeadroomBytes = observedActivationHeadroomBytes
        self.exactTemporaryWorkspaceBytes = exactTemporaryWorkspaceBytes
    }
}
