import Foundation

/// Pure measured model geometry feeding every RAM budget composition.
///
/// Family measurement adapters read disk manifests and supply these payload
/// facts; the budget math never re-derives them from artifacts. Byte counts
/// stay in plain bytes internally; user-facing reporting converts to decimal
/// SI (1 GB = 1,000,000,000 bytes).
public struct MlxRamBudgetModelGeometry: Equatable, Sendable {

    /// Non-expert resident model payload (language core, optional vision).
    public let modelCorePayloadBytes: UInt64

    /// Bytes required if every sparse expert is fully resident.
    public let completeExpertPayloadBytes: UInt64

    /// One complete sparse layer; reserved as the streaming workspace.
    public let largestCompleteExpertLayerBytes: UInt64

    /// Largest exact top-K page used by one-token decode.
    public let largestRoutedExpertPageBytes: UInt64

    /// Persistent decoder-state bytes added by one more prompt token.
    public let sequenceStateBytesPerToken: UInt64

    public init(
        modelCorePayloadBytes: UInt64,
        completeExpertPayloadBytes: UInt64,
        largestCompleteExpertLayerBytes: UInt64,
        largestRoutedExpertPageBytes: UInt64,
        sequenceStateBytesPerToken: UInt64
    ) {
        self.modelCorePayloadBytes = modelCorePayloadBytes
        self.completeExpertPayloadBytes = completeExpertPayloadBytes
        self.largestCompleteExpertLayerBytes = largestCompleteExpertLayerBytes
        self.largestRoutedExpertPageBytes = largestRoutedExpertPageBytes
        self.sequenceStateBytesPerToken = sequenceStateBytesPerToken
    }
}
