import Foundation

/// One expert the previous token routed, with the facts admission needs,
/// port of the Rust `PreviousTokenPrefetchCandidate`.
public struct PreviousTokenPrefetchCandidate: Equatable, Sendable {

    public let layerIndex: Int

    public let expertId: Int

    public let payloadBytes: UInt64

    public let isAlreadyResident: Bool

    public init(
        layerIndex: Int,
        expertId: Int,
        payloadBytes: UInt64,
        isAlreadyResident: Bool
    ) {
        self.layerIndex = layerIndex
        self.expertId = expertId
        self.payloadBytes = payloadBytes
        self.isAlreadyResident = isAlreadyResident
    }
}
