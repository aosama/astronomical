import Foundation

/// Validated source metadata for one full quantized expert tensor within
/// one layer plan. Port of the Rust `QuantizedTensorSource` record: the
/// bounded reader consumes these facts to read expert pages straight from
/// the shard with no per-request geometry work.
public struct QuantizedTensorSource: Equatable, Sendable {

    /// The safetensors tensor name inside the shard.
    public var tensorName: String

    /// The projection the tensor feeds (for example `switch_mlp`).
    public var projectionName: String

    /// The parameter within the projection (`weight`, `scales`, `biases`).
    public var parameterName: String

    /// Storage bit width of the packed payload elements.
    public var quantizationBits: Int32

    /// Affine group size of the packed payload elements.
    public var quantizationGroupSize: Int32

    /// Shard file holding the tensor payload.
    public var sourceFileName: String

    /// Total size of the shard file in bytes.
    public var sourceFileSizeBytes: UInt64

    /// Wire dtype of the stored tensor.
    public var dtype: SafetensorsDtype

    /// Full stored shape of the tensor, experts dimension first.
    public var fullShape: [Int]

    /// Payload offset of this tensor inside its shard file.
    public var tensorPayloadOffsetBytes: UInt64

    /// Packed payload bytes one expert occupies inside this tensor.
    public var bytesPerExpert: Int

    /// Number of experts this tensor stores.
    public var expertCapacity: Int

    public init(
        tensorName: String,
        projectionName: String,
        parameterName: String,
        quantizationBits: Int32,
        quantizationGroupSize: Int32,
        sourceFileName: String,
        sourceFileSizeBytes: UInt64,
        dtype: SafetensorsDtype,
        fullShape: [Int],
        tensorPayloadOffsetBytes: UInt64,
        bytesPerExpert: Int,
        expertCapacity: Int
    ) {
        self.tensorName = tensorName
        self.projectionName = projectionName
        self.parameterName = parameterName
        self.quantizationBits = quantizationBits
        self.quantizationGroupSize = quantizationGroupSize
        self.sourceFileName = sourceFileName
        self.sourceFileSizeBytes = sourceFileSizeBytes
        self.dtype = dtype
        self.fullShape = fullShape
        self.tensorPayloadOffsetBytes = tensorPayloadOffsetBytes
        self.bytesPerExpert = bytesPerExpert
        self.expertCapacity = expertCapacity
    }
}
