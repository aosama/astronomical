import Foundation;

import MLX;
import MLXLMCommon;

import RuntimeIntegration;


/// Adapts the engine's live upstream caches and the persistent prompt-cache
/// state owners at request boundaries, port of the Rust engine's
/// state-stack ownership around `RequestDecoderStateStack`. Upstream caches
/// and the state owners reference the same MLX arrays; mirroring copies
/// references, never payloads, so extraction stays allocation-light.
///
/// The mirror is also the dtype authority: the storage contract's scalar
/// types are read from the executed graph's cache tensors, because a
/// configuration activation label does not describe what the live graph
/// actually produces (initializer-weight journeys run float32 under a
/// bfloat16 config label).
public enum Qwen35MoeLiveCacheStateMirror {

    /// Reads the executed per-layer cache dtypes from one probe materialized
    /// cache set, port of the Rust bound-graph dtype replay in a form Swift
    /// can honor without touching weight internals.
    public static func executedLayerCacheDtypes(
        probeCaches: [KVCache]
    ) throws -> [Qwen35DecoderLayerCacheDtypes] {
        var executedLayerCacheDtypes: [Qwen35DecoderLayerCacheDtypes] = [];
        executedLayerCacheDtypes.reserveCapacity(probeCaches.count);
        for (layerIndex, liveCache): (Int, KVCache) in probeCaches.enumerated() {
            if let mambaCache: MambaCache = liveCache as? MambaCache {
                guard let convolutionTensor: MLXArray = mambaCache[0],
                    let recurrentTensor: MLXArray = mambaCache[1]
                else {
                    throw Qwen35MoePromptCacheError.missingLiveCacheTensor(
                        layerIndex: layerIndex, tensorRole: "linear.attention");
                }
                executedLayerCacheDtypes.append(.linearAttention(
                    convolution: try Self.persistedDtype(
                        convolutionTensor.dtype, layerIndex: layerIndex,
                        tensorRole: Qwen35DecoderCacheTensorRole.CONVOLUTION)));
                // The gated-delta recurrent accumulator is float32 in live
                // execution; upstream guarantees it and the layout pins it.
                if recurrentTensor.dtype != .float32 {
                    throw Qwen35MoePromptCacheError.unsupportedLiveCacheDtype(
                        layerIndex: layerIndex,
                        tensorRole: Qwen35DecoderCacheTensorRole.RECURRENCE,
                        dtypeName: String(describing: recurrentTensor.dtype));
                }
                continue;
            }
            guard let attentionCache: KVCacheSimple = liveCache as? KVCacheSimple else {
                throw Qwen35MoePromptCacheError.unsupportedFullAttentionCacheType(
                    layerIndex: layerIndex);
            }
            let attentionState: [MLXArray] = attentionCache.state;
            guard attentionState.count == 2 else {
                throw Qwen35MoePromptCacheError.missingLiveCacheTensor(
                    layerIndex: layerIndex,
                    tensorRole: Qwen35DecoderCacheTensorRole.ATTENTION_KEYS);
            }
            let keysDtype: DecoderCacheTensorDtype = try Self.persistedDtype(
                attentionState[0].dtype, layerIndex: layerIndex,
                tensorRole: Qwen35DecoderCacheTensorRole.ATTENTION_KEYS);
            let valuesDtype: DecoderCacheTensorDtype = try Self.persistedDtype(
                attentionState[1].dtype, layerIndex: layerIndex,
                tensorRole: Qwen35DecoderCacheTensorRole.ATTENTION_VALUES);
            executedLayerCacheDtypes.append(.fullAttention(keys: keysDtype, values: valuesDtype));
        }
        return executedLayerCacheDtypes;
    }

    /// Mirrors one live cache set into state owners the persistent bridge
    /// can extract from. The owners reference the live tensors; the mirror
    /// runs at capture boundaries only.
    public static func mirroredStateStack(
        liveCaches: [KVCache]
    ) throws -> RequestDecoderStateStack {
        var decoderLayerStates: [DecoderCacheState] = [];
        decoderLayerStates.reserveCapacity(liveCaches.count);
        for (layerIndex, liveCache): (Int, KVCache) in liveCaches.enumerated() {
            if let mambaCache: MambaCache = liveCache as? MambaCache {
                guard let convolutionTensor: MLXArray = mambaCache[0],
                    let recurrentTensor: MLXArray = mambaCache[1]
                else {
                    throw Qwen35MoePromptCacheError.missingLiveCacheTensor(
                        layerIndex: layerIndex, tensorRole: "linear.attention");
                }
                let convolutionState: ConvolutionState = ConvolutionState();
                convolutionState.restoreFromSnapshot(convolutionTensor);
                let recurrentState: GatedDeltaRecurrentState = GatedDeltaRecurrentState();
                recurrentState.restoreFromSnapshot(recurrentTensor);
                decoderLayerStates.append(.composite(
                    convolution: convolutionState, recurrent: recurrentState));
                continue;
            }
            guard let attentionCache: KVCacheSimple = liveCache as? KVCacheSimple else {
                throw Qwen35MoePromptCacheError.unsupportedFullAttentionCacheType(
                    layerIndex: layerIndex);
            }
            let attentionState: [MLXArray] = attentionCache.state;
            guard attentionState.count == 2 else {
                throw Qwen35MoePromptCacheError.missingLiveCacheTensor(
                    layerIndex: layerIndex,
                    tensorRole: Qwen35DecoderCacheTensorRole.ATTENTION_KEYS);
            }
            let attentionOwner: FullAttentionKeyValueState = FullAttentionKeyValueState();
            try attentionOwner.restoreFromBlocks(
                restoredKeys: attentionState[0], restoredValues: attentionState[1]);
            decoderLayerStates.append(.appendOnlyAttention(attentionOwner));
        }
        return RequestDecoderStateStack(decoderLayerStates: decoderLayerStates);
    }

    /// Writes one restored state stack back into the live caches: restored
    /// key/value slabs replace the attention state wholesale and the cache
    /// offset advances to the restored token count, so the next forward
    /// appends at the boundary the persistent cache proved.
    ///
    /// The attention slabs are seated as a left-fold of growth-sized
    /// concatenations, the same association order the live prefill produced
    /// them with, rather than one fresh whole-prefix concatenation. At the
    /// artifact's bf16 working precision, a differently-associated but
    /// value-equal slab changes the Metal kernel path enough to flip
    /// near-tie sampled tokens; the Rust engine's over-allocated slabs
    /// carry the same producer-representation guarantee.
    public static func applyRestoredState(
        _ restoredStateStack: RequestDecoderStateStack,
        toLiveCaches liveCaches: [KVCache],
        restoredTokenCount: Int,
        slabGrowthTokens: Int
    ) throws {
        guard restoredStateStack.layerCount == liveCaches.count else {
            throw Qwen35MoePromptCacheError.liveCacheFamilyMismatch(
                layerIndex: restoredStateStack.layerCount);
        }
        for layerIndex: Int in 0..<liveCaches.count {
            switch restoredStateStack.layer(layerIndex: layerIndex) {
            case .appendOnlyAttention(let restoredAttention):
                guard let attentionCache: KVCacheSimple =
                    liveCaches[layerIndex] as? KVCacheSimple
                else {
                    throw Qwen35MoePromptCacheError.unsupportedFullAttentionCacheType(
                        layerIndex: layerIndex);
                }
                guard let restoredKeys: MLXArray = restoredAttention.keysState(),
                    let restoredValues: MLXArray = restoredAttention.valuesState()
                else {
                    throw Qwen35MoePromptCacheError.missingLiveCacheTensor(
                        layerIndex: layerIndex,
                        tensorRole: Qwen35DecoderCacheTensorRole.ATTENTION_KEYS);
                }
                // The upstream state setter installs both slabs and advances
                // the cache offset to the keys' token count in one step.
                attentionCache.state = [
                    Self.producerOrderedSlab(
                        restoredKeys, slabGrowthTokens: slabGrowthTokens),
                    Self.producerOrderedSlab(
                        restoredValues, slabGrowthTokens: slabGrowthTokens),
                ];
            case .composite(let restoredConvolution, let restoredRecurrent):
                guard let mambaCache: MambaCache = liveCaches[layerIndex] as? MambaCache
                else {
                    throw Qwen35MoePromptCacheError.liveCacheFamilyMismatch(
                        layerIndex: layerIndex);
                }
                guard let convolutionTensor: MLXArray = restoredConvolution.state(),
                    let recurrentTensor: MLXArray = restoredRecurrent.state()
                else {
                    throw Qwen35MoePromptCacheError.missingLiveCacheTensor(
                        layerIndex: layerIndex, tensorRole: "linear.attention");
                }
                mambaCache[0] = convolutionTensor;
                mambaCache[1] = recurrentTensor;
                // The upstream arrays-cache state setter does not touch the
                // offset; the restored token count is the logical sequence
                // length every later append continues from.
                mambaCache.offset = restoredTokenCount;
            case nil:
                throw Qwen35MoePromptCacheError.liveCacheFamilyMismatch(
                    layerIndex: layerIndex);
            }
        }
    }

    /// Reassociates one restored slab into the live path's growth-order
    /// concatenation so the restored representation matches the cold run's.
    private static func producerOrderedSlab(
        _ restoredSlab: MLXArray, slabGrowthTokens: Int
    ) -> MLXArray {
        let slabShape: [Int] = restoredSlab.shape;
        if slabShape.count < 3 || slabGrowthTokens <= 0 {
            return restoredSlab;
        }
        let slabTokenCount: Int = slabShape[DecoderCacheStateConstants.TOKEN_AXIS];
        if slabTokenCount <= slabGrowthTokens {
            return restoredSlab;
        }
        var producerOrderedSlab: MLXArray? = nil;
        var sliceStart: Int = 0;
        while sliceStart < slabTokenCount {
            let sliceEnd: Int = min(sliceStart + slabGrowthTokens, slabTokenCount);
            let growthSlice: MLXArray = restoredSlab[0..., 0..., sliceStart..<sliceEnd, 0...];
            if let currentSlab: MLXArray = producerOrderedSlab {
                producerOrderedSlab = concatenated([currentSlab, growthSlice],
                    axis: DecoderCacheStateConstants.TOKEN_AXIS);
            } else {
                producerOrderedSlab = growthSlice;
            }
            sliceStart = sliceEnd;
        }
        return producerOrderedSlab ?? restoredSlab;
    }

    private static func persistedDtype(
        _ liveDtype: DType,
        layerIndex: Int,
        tensorRole: String
    ) throws -> DecoderCacheTensorDtype {
        switch liveDtype {
        case .float16:
            return .float16;
        case .bfloat16:
            return .bfloat16;
        case .float32:
            return .float32;
        default:
            throw Qwen35MoePromptCacheError.unsupportedLiveCacheDtype(
                layerIndex: layerIndex, tensorRole: tensorRole,
                dtypeName: String(describing: liveDtype));
        }
    }
}
