import Foundation;

import Testing;

import MLX;

import JourneyCategories;

import ModelServingTestSupport;

@testable import ModelServing;

extension MlxGpuJourneyContainer {

    /// Hermetic journeys for the persistent prompt-cache state bridge: real
    /// MLX tensors flowing between populated in-memory decoder state and the
    /// split block-tensor form the disk store publishes, with restore
    /// assembling blocks back into live state in one concatenation per
    /// layer. Port of the Rust `persistent_prompt_cache_state_bridge`
    /// direct-MLX journeys, hermetic on the frozen Ornith config.
    @Suite(.tags(.hermeticMlxJourney))
    final class PersistentPromptCacheStateBridgeTests {

        init() {
            signal(SIGPIPE, SIG_IGN);
            MLXMetallibLocator.overrideMetallibPathIfNecessary();
        }

        private static func frozenOrnithConfig() throws -> Qwen3_5Config {
            return try Qwen3_5Config.fromJsonBytes(
                configBytes: try Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes());
        }

        private static func fullAttentionLayerIndices(
            qwen35Config: Qwen3_5Config
        ) -> [Int] {
            return (0..<Int(qwen35Config.layerCount())).filter(
                { (decoderLayerIndex: Int) -> Bool in
                    return qwen35Config.decoderLayerIsFullAttention(
                        decoderLayerIndex: decoderLayerIndex);
                });
        }

        private static func linearAttentionLayerIndices(
            qwen35Config: Qwen3_5Config
        ) -> [Int] {
            return (0..<Int(qwen35Config.layerCount())).filter(
                { (decoderLayerIndex: Int) -> Bool in
                    return qwen35Config.decoderLayerIsFullAttention(
                        decoderLayerIndex: decoderLayerIndex) == false;
                });
        }

        /// Builds the standard per-layer state stack for the config: every
        /// full-attention layer gets a KV owner, every hybrid layer gets the
        /// convolution + recurrent pair.
        private static func standardRequestDecoderState(
            qwen35Config: Qwen3_5Config
        ) -> RequestDecoderStateStack {
            let decoderLayerStates: [DecoderCacheState] =
                (0..<Int(qwen35Config.layerCount())).map(
                    { (decoderLayerIndex: Int) -> DecoderCacheState in
                        if qwen35Config.decoderLayerIsFullAttention(
                            decoderLayerIndex: decoderLayerIndex) {
                            return .appendOnlyAttention(FullAttentionKeyValueState());
                        }
                        return .composite(
                            convolution: ConvolutionState(),
                            recurrent: GatedDeltaRecurrentState());
                    });
            return RequestDecoderStateStack(decoderLayerStates: decoderLayerStates);
        }

        /// Populates every layer with config-shaped zero state so extraction
        /// journeys have live tensors to slice.
        private static func populateRequestDecoderState(
            _ requestDecoderState: RequestDecoderStateStack,
            qwen35Config: Qwen3_5Config,
            fullAttentionTokenCount: Int
        ) throws {
            let keyDimensionShape: [Int] = [
                1,
                Int(qwen35Config.keyValueHeadCount()),
                fullAttentionTokenCount,
                Int(qwen35Config.headDimension()),
            ];
            let convolutionShape: [Int] = [
                1,
                max(Int(qwen35Config.linearConvolutionKernelDimension()) - 1, 1),
                Int(qwen35Config.linearConvolutionStateDimension()),
            ];
            let recurrentShape: [Int] = [
                1,
                Int(qwen35Config.linearValueHeadCount()),
                Int(qwen35Config.linearValueHeadDimension()),
                Int(qwen35Config.linearKeyHeadDimension()),
            ];
            for decoderLayerIndex: Int in 0..<Int(qwen35Config.layerCount()) {
                switch requestDecoderState.layer(layerIndex: decoderLayerIndex) {
                case .appendOnlyAttention(let attentionState):
                    try attentionState.restoreFromBlocks(
                        restoredKeys: MLXArray.zeros(keyDimensionShape, type: Float.self)
                            .asType(.bfloat16),
                        restoredValues: MLXArray.zeros(keyDimensionShape, type: Float.self)
                            .asType(.bfloat16));
                case .composite(let convolutionState, let recurrentState):
                    convolutionState.restoreFromSnapshot(
                        MLXArray.zeros(convolutionShape, type: Float.self).asType(.bfloat16));
                    recurrentState.restoreFromSnapshot(
                        MLXArray.zeros(recurrentShape, type: Float.self));
                case nil:
                    break;
                }
            }
        }

        /// Tiny full-attention block tensors for every full-attention layer:
        /// keys hold [base, base+1] and values hold [base+2, base+3].
        private static func tinyKvBlockTensors(
            qwen35Config: Qwen3_5Config,
            tensorValueBase: Float
        ) -> [String: MLXArray] {
            var kvBlockTensors: [String: MLXArray] = [:];
            for decoderLayerIndex: Int in Self.fullAttentionLayerIndices(
                qwen35Config: qwen35Config) {
                kvBlockTensors["layer_\(decoderLayerIndex)_attention.keys"] = MLXArray(
                    [tensorValueBase, tensorValueBase + 1.0], [1, 1, 2, 1]);
                kvBlockTensors["layer_\(decoderLayerIndex)_attention.values"] = MLXArray(
                    [tensorValueBase + 2.0, tensorValueBase + 3.0], [1, 1, 2, 1]);
            }
            return kvBlockTensors;
        }

        /// Tiny recurrent snapshot tensors for every hybrid layer:
        /// convolution holds [base+4] and recurrent holds [base+5].
        private static func tinyRecurrentSnapshotTensors(
            qwen35Config: Qwen3_5Config,
            tensorValueBase: Float
        ) -> [String: MLXArray] {
            var recurrentSnapshotTensors: [String: MLXArray] = [:];
            for decoderLayerIndex: Int in Self.linearAttentionLayerIndices(
                qwen35Config: qwen35Config) {
                recurrentSnapshotTensors["layer_\(decoderLayerIndex)_linear.convolution"] =
                    MLXArray([tensorValueBase + 4.0], [1, 1, 1]);
                recurrentSnapshotTensors["layer_\(decoderLayerIndex)_linear.gated_delta_recurrent"] =
                    MLXArray([tensorValueBase + 5.0], [1, 1, 1]);
            }
            return recurrentSnapshotTensors;
        }

        @Test
        func should_extract_split_persistent_prompt_cache_tensors_from_populated_decoder_state()
            throws {
            let qwen35Config: Qwen3_5Config = try Self.frozenOrnithConfig();
            let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
                .ornithModelContract();
            let requestDecoderState: RequestDecoderStateStack = Self
                .standardRequestDecoderState(qwen35Config: qwen35Config);
            try Self.populateRequestDecoderState(
                requestDecoderState, qwen35Config: qwen35Config,
                fullAttentionTokenCount: modelContract.blockTokenCount);

            let kvBlockTensors: [String: MLXArray] = try requestDecoderState
                .extractPersistentPromptCacheKvBlockTensors(
                    blockStartTokens: 0,
                    blockEndTokens: modelContract.blockTokenCount,
                    contractBlockTokenCount: modelContract.blockTokenCount);
            let recurrentSnapshotTensors: [String: MLXArray] = try requestDecoderState
                .extractPersistentPromptCacheRecurrentSnapshotTensors();

            #expect(kvBlockTensors.count
                == Self.fullAttentionLayerIndices(qwen35Config: qwen35Config).count * 2);
            #expect(recurrentSnapshotTensors.count
                == Self.linearAttentionLayerIndices(qwen35Config: qwen35Config).count * 2);
            for decoderLayerIndex: Int in 0..<Int(qwen35Config.layerCount()) {
                if qwen35Config.decoderLayerIsFullAttention(decoderLayerIndex: decoderLayerIndex) {
                    #expect(kvBlockTensors["layer_\(decoderLayerIndex)_attention.keys"] != nil);
                    #expect(kvBlockTensors["layer_\(decoderLayerIndex)_attention.values"] != nil);
                    #expect(recurrentSnapshotTensors[
                        "layer_\(decoderLayerIndex)_linear.gated_delta_recurrent"] == nil);
                } else {
                    #expect(recurrentSnapshotTensors[
                        "layer_\(decoderLayerIndex)_linear.convolution"] != nil);
                    #expect(kvBlockTensors[
                        "layer_\(decoderLayerIndex)_linear.gated_delta_recurrent"] == nil);
                }
            }
        }

        @Test
        func should_extract_only_the_requested_kv_slice_from_longer_model_state() throws {
            let qwen35Config: Qwen3_5Config = try Self.frozenOrnithConfig();
            let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
                .ornithModelContract();
            let requestDecoderState: RequestDecoderStateStack = Self
                .standardRequestDecoderState(qwen35Config: qwen35Config);
            try Self.populateRequestDecoderState(
                requestDecoderState, qwen35Config: qwen35Config,
                fullAttentionTokenCount: modelContract.blockTokenCount * 2);

            let kvBlockTensors: [String: MLXArray] = try requestDecoderState
                .extractPersistentPromptCacheKvBlockTensors(
                    blockStartTokens: modelContract.blockTokenCount,
                    blockEndTokens: modelContract.blockTokenCount * 2,
                    contractBlockTokenCount: modelContract.blockTokenCount);

            let expectedShape: [Int] = [
                1,
                Int(qwen35Config.keyValueHeadCount()),
                modelContract.blockTokenCount,
                Int(qwen35Config.headDimension()),
            ];
            for decoderLayerIndex: Int in Self.fullAttentionLayerIndices(
                qwen35Config: qwen35Config) {
                let extractedKeys: MLXArray? = kvBlockTensors[
                    "layer_\(decoderLayerIndex)_attention.keys"];
                let extractedValues: MLXArray? = kvBlockTensors[
                    "layer_\(decoderLayerIndex)_attention.values"];
                #expect(extractedKeys?.shape == expectedShape);
                #expect(extractedValues?.shape == expectedShape);
            }
        }

        @Test
        func should_restore_request_decoder_state_from_kv_blocks_and_recurrent_snapshot()
            throws {
            let qwen35Config: Qwen3_5Config = try Self.frozenOrnithConfig();
            let firstKvBlockTensors: [String: MLXArray] = Self.tinyKvBlockTensors(
                qwen35Config: qwen35Config, tensorValueBase: 10.0);
            let secondKvBlockTensors: [String: MLXArray] = Self.tinyKvBlockTensors(
                qwen35Config: qwen35Config, tensorValueBase: 20.0);
            var recurrentSnapshotTensors: [String: MLXArray] = Self
                .tinyRecurrentSnapshotTensors(qwen35Config: qwen35Config, tensorValueBase: 30.0);

            let restoredRequestDecoderState: RequestDecoderStateStack = Self
                .standardRequestDecoderState(qwen35Config: qwen35Config);
            try restoredRequestDecoderState.restoreFullAttentionKvConcat(
                blockTensorMaps: [firstKvBlockTensors, secondKvBlockTensors],
                restoredTokenCount: 4);
            try restoredRequestDecoderState.absorbPersistentPromptCacheRecurrentSnapshot(
                &recurrentSnapshotTensors);

            guard case .appendOnlyAttention(let fullAttentionLayer) =
                restoredRequestDecoderState.layer(layerIndex: 3)
            else {
                Issue.record("layer 3 should be full attention in the frozen Ornith config");
                return;
            }
            #expect(fullAttentionLayer.offsetTokens == 4);
            let restoredKeys: Array<Float> = fullAttentionLayer.keysState()!
                .asArray(Float.self);
            let restoredValues: Array<Float> = fullAttentionLayer.valuesState()!
                .asArray(Float.self);
            #expect(fullAttentionLayer.keysState()?.shape == [1, 1, 4, 1]);
            #expect(restoredKeys == [10.0, 11.0, 20.0, 21.0]);
            #expect(restoredValues == [12.0, 13.0, 22.0, 23.0]);

            guard case .composite(let convolutionState, let recurrentState) =
                restoredRequestDecoderState.layer(layerIndex: 0)
            else {
                Issue.record("layer 0 should be linear attention in the frozen Ornith config");
                return;
            }
            #expect(convolutionState.state()?.asArray(Float.self) == [34.0]);
            #expect(recurrentState.state()?.asArray(Float.self) == [35.0]);
            #expect(recurrentSnapshotTensors.isEmpty,
                "every absorbed snapshot tensor must be taken from the caller's map");
        }

        @Test
        func should_restore_three_kv_blocks_in_sequence_order_at_final_length() throws {
            let qwen35Config: Qwen3_5Config = try Self.frozenOrnithConfig();
            let kvBlockTensors: [[String: MLXArray]] = [
                Self.tinyKvBlockTensors(qwen35Config: qwen35Config, tensorValueBase: 10.0),
                Self.tinyKvBlockTensors(qwen35Config: qwen35Config, tensorValueBase: 20.0),
                Self.tinyKvBlockTensors(qwen35Config: qwen35Config, tensorValueBase: 30.0),
            ];
            var recurrentSnapshotTensors: [String: MLXArray] = Self
                .tinyRecurrentSnapshotTensors(qwen35Config: qwen35Config, tensorValueBase: 40.0);

            let restoredRequestDecoderState: RequestDecoderStateStack = Self
                .standardRequestDecoderState(qwen35Config: qwen35Config);
            try restoredRequestDecoderState.restoreFullAttentionKvConcat(
                blockTensorMaps: kvBlockTensors, restoredTokenCount: 6);
            try restoredRequestDecoderState.absorbPersistentPromptCacheRecurrentSnapshot(
                &recurrentSnapshotTensors);

            guard case .appendOnlyAttention(let fullAttentionLayer) =
                restoredRequestDecoderState.layer(layerIndex: 3)
            else {
                Issue.record("layer 3 should be full attention in the frozen Ornith config");
                return;
            }
            #expect(fullAttentionLayer.offsetTokens == 6);
            #expect(fullAttentionLayer.keysState()?.shape == [1, 1, 6, 1]);
            #expect(fullAttentionLayer.keysState()!.asArray(Float.self)
                == [10.0, 11.0, 20.0, 21.0, 30.0, 31.0]);
        }

        @Test
        func should_materialize_restored_state_before_first_prefill() throws {
            let qwen35Config: Qwen3_5Config = try Self.frozenOrnithConfig();
            let kvBlockTensors: [[String: MLXArray]] = [
                Self.tinyKvBlockTensors(qwen35Config: qwen35Config, tensorValueBase: 10.0)
            ];
            var recurrentSnapshotTensors: [String: MLXArray] = Self
                .tinyRecurrentSnapshotTensors(qwen35Config: qwen35Config, tensorValueBase: 30.0);

            let restoredRequestDecoderState: RequestDecoderStateStack = Self
                .standardRequestDecoderState(qwen35Config: qwen35Config);
            try restoredRequestDecoderState.restoreFullAttentionKvConcat(
                blockTensorMaps: kvBlockTensors, restoredTokenCount: 2);
            try restoredRequestDecoderState.absorbPersistentPromptCacheRecurrentSnapshot(
                &recurrentSnapshotTensors);

            restoredRequestDecoderState.materializeRestoredPersistentPromptCacheState();
            guard case .appendOnlyAttention(let fullAttentionLayer) =
                restoredRequestDecoderState.layer(layerIndex: 3)
            else {
                Issue.record("layer 3 should be full attention in the frozen Ornith config");
                return;
            }
            #expect(fullAttentionLayer.keysState()!.asArray(Float.self) == [10.0, 11.0]);
        }

        @Test
        func should_reject_a_kv_block_tensor_map_missing_a_required_tensor() throws {
            let qwen35Config: Qwen3_5Config = try Self.frozenOrnithConfig();
            var kvBlockTensors: [String: MLXArray] = Self.tinyKvBlockTensors(
                qwen35Config: qwen35Config, tensorValueBase: 10.0);
            kvBlockTensors.removeValue(forKey: "layer_3_attention.keys");

            let restoredRequestDecoderState: RequestDecoderStateStack = Self
                .standardRequestDecoderState(qwen35Config: qwen35Config);

            #expect(throws: PersistentPromptCacheStateBridgeError.self) {
                try restoredRequestDecoderState.restoreFullAttentionKvConcat(
                    blockTensorMaps: [kvBlockTensors], restoredTokenCount: 2);
            }
        }

        @Test
        func should_reject_a_recurrent_snapshot_tensor_map_missing_a_required_tensor() throws {
            let qwen35Config: Qwen3_5Config = try Self.frozenOrnithConfig();
            let kvBlockTensors: [[String: MLXArray]] = [
                Self.tinyKvBlockTensors(qwen35Config: qwen35Config, tensorValueBase: 10.0)
            ];
            var recurrentSnapshotTensors: [String: MLXArray] = Self
                .tinyRecurrentSnapshotTensors(qwen35Config: qwen35Config, tensorValueBase: 30.0);
            recurrentSnapshotTensors.removeValue(
                forKey: "layer_0_linear.gated_delta_recurrent");

            let restoredRequestDecoderState: RequestDecoderStateStack = Self
                .standardRequestDecoderState(qwen35Config: qwen35Config);
            try restoredRequestDecoderState.restoreFullAttentionKvConcat(
                blockTensorMaps: kvBlockTensors, restoredTokenCount: 2);

            #expect(throws: PersistentPromptCacheStateBridgeError.self) {
                try restoredRequestDecoderState.absorbPersistentPromptCacheRecurrentSnapshot(
                    &recurrentSnapshotTensors);
            }
        }
    }
}
