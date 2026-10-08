import Foundation

/// The per-shard assembly plan of one expert page: which tensors to
/// publish and which bounded intervals to read. Port of the Rust
/// `QuantizedExpertShardManifest` record; a page assembled from several
/// shards carries one manifest per shard.
public struct QuantizedExpertShardManifest: Equatable, Sendable {

    /// Shard file the intervals read from.
    public var sourceFileName: String

    /// Tensor placements the page publishes for this shard.
    public var tensorRanges: [QuantizedExpertTensorRange]

    /// Bounded reads to perform against the shard file.
    public var sourceIntervals: [QuantizedExpertSourceInterval]

    /// Total bytes this shard contributes to the page payload.
    public var payloadByteCount: UInt64

    public init(
        sourceFileName: String,
        tensorRanges: [QuantizedExpertTensorRange],
        sourceIntervals: [QuantizedExpertSourceInterval],
        payloadByteCount: UInt64
    ) {
        self.sourceFileName = sourceFileName
        self.tensorRanges = tensorRanges
        self.sourceIntervals = sourceIntervals
        self.payloadByteCount = payloadByteCount
    }
}
