import Foundation

/// Exact ownership bytes transferred by one successful atomic commit.
public struct RetainedExpertLayerCommitDelta: Equatable, Sendable {

    /// Bytes released from the page this commit replaced.
    public var releasedPayloadBytes: UInt64

    /// Bytes newly owned by the committed page.
    public var committedPayloadBytes: UInt64

    public init(releasedPayloadBytes: UInt64, committedPayloadBytes: UInt64) {
        self.releasedPayloadBytes = releasedPayloadBytes
        self.committedPayloadBytes = committedPayloadBytes
    }
}
