import Foundation;

import MLX;

/// Bridges persistent prompt-cache block tensors and the live in-memory
/// request decoder state, port of the Rust `persistent_state_bridge` and
/// `persistent_state_kv_restore`. The in-memory owners decide how restored
/// tensors become live state; this bridge owns only SSD block extraction,
/// assembly, and the one-concatenation-per-layer restore that replaced
/// per-block slice updates (concatenating every block exactly once is
/// O(restored tokens); per-block recopying was O(tokens × blocks) with one
/// synchronization per block).
extension RequestDecoderStateStack {

    /// Extracts one full-attention key/value block into the persistent
    /// tensor map. The range may cover a complete block boundary or a
    /// partial prefix-cache tail; both produce tensors that participate in
    /// the hash chain and concatenate with other blocks during a later
    /// restore.
    public func extractPersistentPromptCacheKvBlockTensors(
        blockStartTokens: Int,
        blockEndTokens: Int,
        contractBlockTokenCount: Int
    ) throws -> [String: MLXArray] {
        try Self.validatePersistentPromptCacheBlockRange(
            blockStartTokens: blockStartTokens, blockEndTokens: blockEndTokens,
            contractBlockTokenCount: contractBlockTokenCount);
        var kvBlockTensors: [String: MLXArray] = [:];
        kvBlockTensors.reserveCapacity(self.layerCount * 2);
        for layerIndex: Int in 0..<self.layerCount {
            switch self.layer(layerIndex: layerIndex) {
            case .appendOnlyAttention(let attentionState):
                guard let attentionKeys: MLXArray = attentionState.keysState()
                else {
                    throw PersistentPromptCacheStateBridgeError.missingLayerTensor(
                        layerIndex: layerIndex, tensorRole: "keys");
                }
                guard let attentionValues: MLXArray = attentionState.valuesState()
                else {
                    throw PersistentPromptCacheStateBridgeError.missingLayerTensor(
                        layerIndex: layerIndex, tensorRole: "values");
                }
                kvBlockTensors[Self.persistentTensorName(
                    layerIndex: layerIndex, tensorRole: Qwen35DecoderCacheTensorRole
                        .ATTENTION_KEYS)] = try Self.sliceFullAttentionBlock(
                    tensor: attentionKeys, layerIndex: layerIndex, tensorRole: "keys",
                    blockStartTokens: blockStartTokens, blockEndTokens: blockEndTokens);
                kvBlockTensors[Self.persistentTensorName(
                    layerIndex: layerIndex, tensorRole: Qwen35DecoderCacheTensorRole
                        .ATTENTION_VALUES)] = try Self.sliceFullAttentionBlock(
                    tensor: attentionValues, layerIndex: layerIndex, tensorRole: "values",
                    blockStartTokens: blockStartTokens, blockEndTokens: blockEndTokens);
            case .composite:
                break;
            case nil:
                throw PersistentPromptCacheStateBridgeError.missingLayer(layerIndex: layerIndex);
            }
        }
        return kvBlockTensors;
    }

    /// Extracts the current gated-delta recurrent and convolution snapshots.
    public func extractPersistentPromptCacheRecurrentSnapshotTensors() throws
        -> [String: MLXArray] {
        var recurrentSnapshotTensors: [String: MLXArray] = [:];
        recurrentSnapshotTensors.reserveCapacity(self.layerCount * 2);
        for layerIndex: Int in 0..<self.layerCount {
            switch self.layer(layerIndex: layerIndex) {
            case .appendOnlyAttention:
                break;
            case .composite(let convolutionState, let recurrentState):
                guard let convolutionTensor: MLXArray = convolutionState.state()
                else {
                    throw PersistentPromptCacheStateBridgeError.missingLayerTensor(
                        layerIndex: layerIndex, tensorRole: "convolution");
                }
                guard let recurrentTensor: MLXArray = recurrentState.state()
                else {
                    throw PersistentPromptCacheStateBridgeError.missingLayerTensor(
                        layerIndex: layerIndex, tensorRole: "recurrent");
                }
                recurrentSnapshotTensors[Self.persistentTensorName(
                    layerIndex: layerIndex,
                    tensorRole: Qwen35DecoderCacheTensorRole.CONVOLUTION)] = convolutionTensor;
                recurrentSnapshotTensors[Self.persistentTensorName(
                    layerIndex: layerIndex,
                    tensorRole: Qwen35DecoderCacheTensorRole.RECURRENCE)] = recurrentTensor;
            case nil:
                throw PersistentPromptCacheStateBridgeError.missingLayer(layerIndex: layerIndex);
            }
        }
        return recurrentSnapshotTensors;
    }

    /// Restores full-attention KV by concatenating every restored block
    /// slice along the sequence axis in one pass per layer, then
    /// materializing the result so the source blocks release before the
    /// recurrent snapshot loads. Every full-attention layer must concatenate
    /// to exactly `restoredTokenCount`.
    public func restoreFullAttentionKvConcat(
        blockTensorMaps: [[String: MLXArray]],
        restoredTokenCount: Int
    ) throws {
        if blockTensorMaps.isEmpty {
            throw PersistentPromptCacheStateBridgeError.invalidRestoredSequenceTokenCount(
                restoredTokenCount: restoredTokenCount);
        }
        for layerIndex: Int in 0..<self.layerCount {
            switch self.layer(layerIndex: layerIndex) {
            case .appendOnlyAttention(let attentionState):
                let keysTensorName: String = Self.persistentTensorName(
                    layerIndex: layerIndex, tensorRole: Qwen35DecoderCacheTensorRole
                        .ATTENTION_KEYS);
                let valuesTensorName: String = Self.persistentTensorName(
                    layerIndex: layerIndex, tensorRole: Qwen35DecoderCacheTensorRole
                        .ATTENTION_VALUES);
                let keysSlices: [MLXArray] = try Self.blockSliceTensorsByName(
                    blockTensorMaps: blockTensorMaps, layerIndex: layerIndex,
                    tensorName: keysTensorName);
                let valuesSlices: [MLXArray] = try Self.blockSliceTensorsByName(
                    blockTensorMaps: blockTensorMaps, layerIndex: layerIndex,
                    tensorName: valuesTensorName);
                let fullKeys: MLXArray = concatenated(
                    keysSlices, axis: DecoderCacheStateConstants.TOKEN_AXIS);
                let fullValues: MLXArray = concatenated(
                    valuesSlices, axis: DecoderCacheStateConstants.TOKEN_AXIS);
                try Self.validateConcatenatedTokenCount(
                    concatenatedTensor: fullKeys, layerIndex: layerIndex, tensorRole: "keys",
                    restoredTokenCount: restoredTokenCount);
                try Self.validateConcatenatedTokenCount(
                    concatenatedTensor: fullValues, layerIndex: layerIndex, tensorRole: "values",
                    restoredTokenCount: restoredTokenCount);
                try attentionState.restoreFromBlocks(
                    restoredKeys: fullKeys, restoredValues: fullValues);
            case .composite:
                break;
            case nil:
                throw PersistentPromptCacheStateBridgeError.missingLayer(layerIndex: layerIndex);
            }
        }
        self.materializeRestoredFullAttentionTensors();
    }

    /// Installs the newest complete recurrent snapshot after KV blocks are
    /// live. Each absorbed tensor is removed (taken) from the caller's map,
    /// so a partially absorbed snapshot cannot be silently reused.
    public func absorbPersistentPromptCacheRecurrentSnapshot(
        _ recurrentSnapshotTensors: inout [String: MLXArray]
    ) throws {
        for layerIndex: Int in 0..<self.layerCount {
            switch self.layer(layerIndex: layerIndex) {
            case .appendOnlyAttention:
                break;
            case .composite(let convolutionState, let recurrentState):
                let convolutionTensorName: String = Self.persistentTensorName(
                    layerIndex: layerIndex, tensorRole: Qwen35DecoderCacheTensorRole
                        .CONVOLUTION);
                guard let loadedConvolution: MLXArray = recurrentSnapshotTensors
                    .removeValue(forKey: convolutionTensorName)
                else {
                    throw PersistentPromptCacheStateBridgeError.missingBlockTensor(
                        layerIndex: layerIndex, tensorName: convolutionTensorName);
                }
                let recurrentTensorName: String = Self.persistentTensorName(
                    layerIndex: layerIndex, tensorRole: Qwen35DecoderCacheTensorRole.RECURRENCE);
                guard let loadedRecurrent: MLXArray = recurrentSnapshotTensors
                    .removeValue(forKey: recurrentTensorName)
                else {
                    throw PersistentPromptCacheStateBridgeError.missingBlockTensor(
                        layerIndex: layerIndex, tensorName: recurrentTensorName);
                }
                convolutionState.restoreFromSnapshot(loadedConvolution);
                recurrentState.restoreFromSnapshot(loadedRecurrent);
                convolutionState.state()?.eval();
                recurrentState.state()?.eval();
            case nil:
                throw PersistentPromptCacheStateBridgeError.missingLayer(layerIndex: layerIndex);
            }
        }
    }

    /// Materializes all restored state before the first new prefill chunk.
    /// Evaluation places malformed restored state at this single explicit
    /// GPU boundary instead of letting lazy MLX evaluation attribute a later
    /// request failure to unrelated model work.
    public func materializeRestoredPersistentPromptCacheState() {
        for layerIndex: Int in 0..<self.layerCount {
            switch self.layer(layerIndex: layerIndex) {
            case .appendOnlyAttention(let attentionState):
                attentionState.keysState()?.eval();
                attentionState.valuesState()?.eval();
            case .composite(let convolutionState, let recurrentState):
                convolutionState.state()?.eval();
                recurrentState.state()?.eval();
            case nil:
                break;
            }
        }
    }

    private func materializeRestoredFullAttentionTensors() {
        for layerIndex: Int in 0..<self.layerCount {
            switch self.layer(layerIndex: layerIndex) {
            case .appendOnlyAttention(let attentionState):
                attentionState.keysState()?.eval();
                attentionState.valuesState()?.eval();
            case .composite:
                break;
            case nil:
                break;
            }
        }
    }

    private static func persistentTensorName(layerIndex: Int, tensorRole: String) -> String {
        return "layer_\(layerIndex)_\(tensorRole)";
    }

    private static func validatePersistentPromptCacheBlockRange(
        blockStartTokens: Int,
        blockEndTokens: Int,
        contractBlockTokenCount: Int
    ) throws {
        let blockTokenCount: Int = blockEndTokens - blockStartTokens;
        // Partial blocks (prefix-cache tails) hold fewer tokens than a full
        // block while remaining valid restorable state, so only the contract
        // bound and non-emptiness are enforced here.
        if blockTokenCount <= 0 || blockTokenCount > contractBlockTokenCount {
            throw PersistentPromptCacheStateBridgeError.invalidBlockRange(
                blockStartTokens: blockStartTokens, blockEndTokens: blockEndTokens);
        }
    }

    private static func sliceFullAttentionBlock(
        tensor: MLXArray,
        layerIndex: Int,
        tensorRole: String,
        blockStartTokens: Int,
        blockEndTokens: Int
    ) throws -> MLXArray {
        let tensorShape: [Int] = tensor.shape;
        if tensorShape.count != 4 {
            throw PersistentPromptCacheStateBridgeError.invalidLayerTensorShape(
                layerIndex: layerIndex, tensorRole: tensorRole, actualShape: tensorShape);
        }
        if tensorShape[DecoderCacheStateConstants.TOKEN_AXIS] < blockEndTokens {
            throw PersistentPromptCacheStateBridgeError.blockRangeExceedsLayerTensor(
                layerIndex: layerIndex,
                tensorRole: tensorRole,
                requestedEndTokens: blockEndTokens,
                availableTokens: tensorShape[DecoderCacheStateConstants.TOKEN_AXIS]);
        }
        return tensor[0..., 0..., blockStartTokens..<blockEndTokens, 0...];
    }

    private static func blockSliceTensorsByName(
        blockTensorMaps: [[String: MLXArray]],
        layerIndex: Int,
        tensorName: String
    ) throws -> [MLXArray] {
        var blockSlices: [MLXArray] = [];
        blockSlices.reserveCapacity(blockTensorMaps.count);
        for blockTensorMap: [String: MLXArray] in blockTensorMaps {
            guard let blockSlice: MLXArray = blockTensorMap[tensorName]
            else {
                throw PersistentPromptCacheStateBridgeError.missingBlockTensor(
                    layerIndex: layerIndex, tensorName: tensorName);
            }
            blockSlices.append(blockSlice);
        }
        return blockSlices;
    }

    private static func validateConcatenatedTokenCount(
        concatenatedTensor: MLXArray,
        layerIndex: Int,
        tensorRole: String,
        restoredTokenCount: Int
    ) throws {
        let concatenatedShape: [Int] = concatenatedTensor.shape;
        if concatenatedShape.count != 4 {
            throw PersistentPromptCacheStateBridgeError.invalidLayerTensorShape(
                layerIndex: layerIndex, tensorRole: tensorRole,
                actualShape: concatenatedShape);
        }
        let concatenatedTokenCount: Int = concatenatedShape[
            DecoderCacheStateConstants.TOKEN_AXIS];
        if concatenatedTokenCount != restoredTokenCount {
            throw PersistentPromptCacheStateBridgeError.concatenatedTokenCountMismatch(
                layerIndex: layerIndex,
                concatenatedTokenCount: concatenatedTokenCount,
                restoredTokenCount: restoredTokenCount);
        }
    }
}
