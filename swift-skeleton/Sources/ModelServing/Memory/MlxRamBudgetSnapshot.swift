import Foundation

/// One composed RAM split for a planned operation
/// (port of `ram_values.rs::MlxRamBudgetSnapshot`).
public struct MlxRamBudgetSnapshot: Equatable, Sendable {

    /// Total MLX active-memory ceiling for this plan.
    public let mlxActiveMemoryCeilingBytes: UInt64

    /// Non-expert model core already charged against the ceiling.
    public let modelCorePayloadBytes: UInt64

    /// Reserved bytes for context-window growth at the planned token count.
    public let contextWindowReserveBytes: UInt64

    /// Reserved bytes for temporary activations / transient workspace.
    public let activationHeadroomBytes: UInt64

    /// Reserved bytes for one complete-layer stream workspace.
    public let completeLayerStreamSlotBytes: UInt64

    /// Any additional fixed non-expert owners (draft model, publication workspace, …).
    public let otherFixedBytes: UInt64

    /// Leftover budget that may pin retained expert layers in MLX.
    public let retainedExpertBudgetBytes: UInt64

    public init(
        mlxActiveMemoryCeilingBytes: UInt64,
        modelCorePayloadBytes: UInt64,
        contextWindowReserveBytes: UInt64,
        activationHeadroomBytes: UInt64,
        completeLayerStreamSlotBytes: UInt64,
        otherFixedBytes: UInt64,
        retainedExpertBudgetBytes: UInt64
    ) {
        self.mlxActiveMemoryCeilingBytes = mlxActiveMemoryCeilingBytes
        self.modelCorePayloadBytes = modelCorePayloadBytes
        self.contextWindowReserveBytes = contextWindowReserveBytes
        self.activationHeadroomBytes = activationHeadroomBytes
        self.completeLayerStreamSlotBytes = completeLayerStreamSlotBytes
        self.otherFixedBytes = otherFixedBytes
        self.retainedExpertBudgetBytes = retainedExpertBudgetBytes
    }
}
