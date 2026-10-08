import Foundation

/// Flattened persistence ownership for append-only, rotating, and
/// recurrent state, port of the Rust `persistence_layouts` module.
extension DecoderCacheLayout {

    /// Returns deterministic append-only sequence tensors for one persisted block.
    public func sequenceTensorLayouts() -> [DecoderCachePersistedTensorLayout] {
        return Self.collectLayoutTensors(
            decoderCacheLayout: self,
            includeSequenceTensors: true)
    }

    /// Returns deterministic rotating and recurrent tensors for one boundary.
    public func boundaryTensorLayouts() -> [DecoderCachePersistedTensorLayout] {
        return Self.collectLayoutTensors(
            decoderCacheLayout: self,
            includeSequenceTensors: false)
    }

    private static func collectLayoutTensors(
        decoderCacheLayout: DecoderCacheLayout,
        includeSequenceTensors: Bool
    ) -> [DecoderCachePersistedTensorLayout] {
        var persistedTensorLayouts: [DecoderCachePersistedTensorLayout] = []
        for decoderLayerIndex in 0..<decoderCacheLayout.layerCount {
            if let layerLayout = decoderCacheLayout.layer(decoderLayerIndex) {
                collectLayerTensors(
                    layerLayout,
                    decoderLayerIndex: decoderLayerIndex,
                    includeSequenceTensors: includeSequenceTensors,
                    into: &persistedTensorLayouts)
            }
        }
        return persistedTensorLayouts
    }

    private static func collectLayerTensors(
        _ layerLayout: DecoderCacheLayerLayout,
        decoderLayerIndex: Int,
        includeSequenceTensors: Bool,
        into persistedTensorLayouts: inout [DecoderCachePersistedTensorLayout]
    ) {
        switch layerLayout {
        case .appendOnlyAttention(let keys, let values, _):
            if includeSequenceTensors {
                pushTensor(&persistedTensorLayouts, decoderLayerIndex, keys)
                pushTensor(&persistedTensorLayouts, decoderLayerIndex, values)
            }
        case .rotatingWindowAttention(let keys, let values, _):
            if !includeSequenceTensors {
                pushTensor(&persistedTensorLayouts, decoderLayerIndex, keys)
                pushTensor(&persistedTensorLayouts, decoderLayerIndex, values)
                for counterLayout in DecoderCacheLayerLayout.rotatingWindowCounterLayouts() {
                    pushTensor(&persistedTensorLayouts, decoderLayerIndex, counterLayout)
                }
            }
        case .recurrentTensor(let tensor):
            if !includeSequenceTensors {
                pushTensor(&persistedTensorLayouts, decoderLayerIndex, tensor)
            }
        case .composite(let components):
            for componentLayout in components {
                collectLayerTensors(
                    componentLayout,
                    decoderLayerIndex: decoderLayerIndex,
                    includeSequenceTensors: includeSequenceTensors,
                    into: &persistedTensorLayouts)
            }
        }
    }

    private static func pushTensor(
        _ persistedTensorLayouts: inout [DecoderCachePersistedTensorLayout],
        _ decoderLayerIndex: Int,
        _ tensorLayout: DecoderCacheTensorLayout
    ) {
        persistedTensorLayouts.append(
            DecoderCachePersistedTensorLayout(
                decoderLayerIndex: decoderLayerIndex,
                tensorLayout: tensorLayout))
    }
}
