import Foundation

/// One bounded read interval: where a run of experts inside one tensor
/// lives inside its shard file. Port of the Rust `QuantizedExpertSourceInterval`
/// record consumed by the bounded expert reader.
public struct QuantizedExpertSourceInterval: Equatable, Sendable {

    /// The safetensors tensor the interval reads from.
    public var tensorName: String

    /// First expert the interval covers, inside the tensor's expert axis.
    public var expertStart: Int

    /// Number of experts the interval covers.
    public var expertCount: Int

    /// Absolute offset inside the shard file where the run starts.
    public var sourceFileOffsetBytes: UInt64

    /// Byte length of the run inside the shard file.
    public var sourceByteCount: Int

    /// Where the run lands inside the assembled page's virtual payload.
    public var virtualPayloadOffsetBytes: UInt64

    public init(
        tensorName: String,
        expertStart: Int,
        expertCount: Int,
        sourceFileOffsetBytes: UInt64,
        sourceByteCount: Int,
        virtualPayloadOffsetBytes: UInt64
    ) {
        self.tensorName = tensorName
        self.expertStart = expertStart
        self.expertCount = expertCount
        self.sourceFileOffsetBytes = sourceFileOffsetBytes
        self.sourceByteCount = sourceByteCount
        self.virtualPayloadOffsetBytes = virtualPayloadOffsetBytes
    }
}
