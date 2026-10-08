import Foundation

/// One tensor's placement inside an assembled expert page's virtual
/// payload. Port of the Rust `QuantizedExpertTensorRange` record: the
/// rebased safetensors header of a page is generated from these ranges.
public struct QuantizedExpertTensorRange: Equatable, Sendable {

    /// The safetensors tensor name the page publishes.
    public var tensorName: String

    /// The projection the tensor feeds (for example `switch_mlp`).
    public var projectionName: String

    /// The parameter within the projection (`weight`, `scales`, `biases`).
    public var parameterName: String

    /// Wire dtype the page publishes for the tensor.
    public var dtype: SafetensorsDtype

    /// Shape the page publishes for the tensor (the sliced expert run).
    public var shape: [Int]

    /// Where the tensor lands inside the page's virtual payload.
    public var virtualPayloadOffsetBytes: UInt64

    /// Byte length of the tensor inside the page.
    public var byteCount: Int

    public init(
        tensorName: String,
        projectionName: String,
        parameterName: String,
        dtype: SafetensorsDtype,
        shape: [Int],
        virtualPayloadOffsetBytes: UInt64,
        byteCount: Int
    ) {
        self.tensorName = tensorName
        self.projectionName = projectionName
        self.parameterName = parameterName
        self.dtype = dtype
        self.shape = shape
        self.virtualPayloadOffsetBytes = virtualPayloadOffsetBytes
        self.byteCount = byteCount
    }
}
