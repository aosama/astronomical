import Foundation

/// One flattened tensor contract used by a decoder-state persistence
/// file, port of the Rust `DecoderCachePersistedTensorLayout`.
public struct DecoderCachePersistedTensorLayout: Equatable, Hashable, Sendable {

    /// The zero-based decoder-layer position.
    public let decoderLayerIndex: Int

    /// The tensor contract for this persisted decoder-state component.
    public let tensorLayout: DecoderCacheTensorLayout

    init(decoderLayerIndex: Int, tensorLayout: DecoderCacheTensorLayout) {
        self.decoderLayerIndex = decoderLayerIndex
        self.tensorLayout = tensorLayout
    }

    /// The deterministic safetensors name for this component.
    public var persistentTensorName: String {
        "layer_\(decoderLayerIndex)_\(tensorLayout.tensorRoleName)"
    }
}
