import Foundation;

import Testing;

import MLX;

import JourneyCategories;

import ModelServingTestSupport;

@testable import ModelServing;

extension HermeticMlxJourneyContainer {

    /// Hermetic round-trip journeys: blocks published through the MLX
    /// writer load back by token count into live decoder state through the
    /// state bridge, and a corrupt published file is deleted and untracked
    /// on load instead of ever serving suspect bytes.
    @Suite(.tags(.hermeticMlxJourney))
    final class PersistentPromptCacheLoadBackTests {

        init() {
            signal(SIGPIPE, SIG_IGN);
            MLXMetallibLocator.overrideMetallibPathIfNecessary();
        }

        /// Deterministic rank-four block tensors for one published block:
        /// keys hold (scalar index mod 17) + base and values hold (scalar
        /// index mod 13) + base + 0.25. Every value is exactly representable
        /// in float16 below 2048, so round-trip equality is exact.
        private static func rankFourBlockTensors(
            blockTokenCount: Int,
            valueBase: Float
        ) -> [String: MLXArray] {
            let tensorShape: [Int] = [1, 2, blockTokenCount, 4];
            let scalarCount: Int = tensorShape.reduce(1, *);
            let keysValues: [Float] = (0..<scalarCount).map(
                { (scalarIndex: Int) -> Float in
                    return Float(scalarIndex % 17) + valueBase;
                });
            let valuesValues: [Float] = (0..<scalarCount).map(
                { (scalarIndex: Int) -> Float in
                    return Float(scalarIndex % 13) + valueBase + 0.25;
                });
            return [
                "layer_0_attention.keys": MLXArray(keysValues, tensorShape).asType(.float16),
                "layer_0_attention.values": MLXArray(valuesValues, tensorShape).asType(.float16),
            ];
        }

        /// The expected live-state flat values after two blocks concatenate
        /// along the token axis: axis-2 concatenation interleaves per head,
        /// so each head's slice appends first-block tokens then second-block
        /// tokens.
        private static func expectedConcatenatedValues(
            tensorShape: [Int],
            firstBlockBase: Float,
            secondBlockBase: Float,
            modulus: Int,
            baseOffset: Float
        ) -> [Float] {
            let scalarsPerTensor: Int = tensorShape.reduce(1, *);
            let blockFlatValues: (Float) -> [Float] = { (blockBase: Float) -> [Float] in
                return (0..<scalarsPerTensor).map(
                    { (scalarIndex: Int) -> Float in
                        return Float(scalarIndex % modulus) + blockBase + baseOffset;
                    });
            };
            let firstBlockFlat: [Float] = blockFlatValues(firstBlockBase);
            let secondBlockFlat: [Float] = blockFlatValues(secondBlockBase);
            let headCount: Int = tensorShape[1];
            let scalarsPerHead: Int = scalarsPerTensor / headCount;
            var expectedValues: [Float] = [];
            expectedValues.reserveCapacity(scalarsPerTensor * 2);
            for headIndex: Int in 0..<headCount {
                let headStart: Int = headIndex * scalarsPerHead;
                let headEnd: Int = headStart + scalarsPerHead;
                expectedValues.append(contentsOf: firstBlockFlat[headStart..<headEnd]);
                expectedValues.append(contentsOf: secondBlockFlat[headStart..<headEnd]);
            }
            return expectedValues;
        }

        @Test
        func should_load_published_blocks_back_into_live_decoder_state_by_token_count()
            throws {
            let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
                .syntheticRankFourSequenceContract();
            let globalRoot: URL = FileManager.default.temporaryDirectory
                .appendingPathComponent("prompt-cache-\(UUID().uuidString)", isDirectory: true);
            try FileManager.default.createDirectory(
                at: globalRoot, withIntermediateDirectories: true);
            let diskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
                .openDiskStore(globalRoot: globalRoot, modelContract: modelContract);
            let promptTokens: [UInt32] = PersistentPromptCacheFixture
                .promptTokensWithCompleteBlocksAndTrailingTokens(
                    modelContract: modelContract, completeBlockCount: 2, trailingTokenCount: 0);
            let chainedBlockKeys: [PersistentPromptCacheBlockKey] = try PersistentPromptCacheFixture
                .blockKeysForPrompt(
                    modelContract: modelContract, promptTokens: promptTokens,
                    requestedBlockCount: 2);
            for (blockPosition, chainedBlockKey): (Int, PersistentPromptCacheBlockKey)
            in chainedBlockKeys.enumerated() {
                let mlxStaging: PersistentPromptCacheStateFileMlxStaging =
                    PersistentPromptCacheStateFileMlxStaging(
                        sequenceStateTensors: Self.rankFourBlockTensors(
                            blockTokenCount: modelContract.blockTokenCount,
                            valueBase: blockPosition == 0 ? 0.0 : 40.0),
                        boundaryStateTensors: [:]);
                let parentBlockKey: PersistentPromptCacheBlockKey? = blockPosition == 0
                    ? nil : chainedBlockKeys[blockPosition - 1];
                #expect(try diskStore.publishBlock(
                    staging: mlxStaging, blockKey: chainedBlockKey,
                    parentBlockKey: parentBlockKey) == .published);
            }

            // Restart semantics: reopen the same root so the index rebuilds
            // from disk before any load.
            let reopenedStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
                .openDiskStore(globalRoot: globalRoot, modelContract: modelContract);
            let firstLoadedBlock: [String: MLXArray]? = try reopenedStore.loadKvBlock(
                blockKey: chainedBlockKeys[0]);
            let secondLoadedBlock: [String: MLXArray]? = try reopenedStore.loadKvBlock(
                blockKey: chainedBlockKeys[1]);
            #expect(firstLoadedBlock != nil);
            #expect(secondLoadedBlock != nil);

            let restoredRequestDecoderState: RequestDecoderStateStack =
                RequestDecoderStateStack(decoderLayerStates: [
                    .appendOnlyAttention(FullAttentionKeyValueState()),
                ]);
            try restoredRequestDecoderState.restoreFullAttentionKvConcat(
                blockTensorMaps: [firstLoadedBlock!, secondLoadedBlock!],
                restoredTokenCount: modelContract.blockTokenCount * 2);

            guard case .appendOnlyAttention(let fullAttentionLayer) =
                restoredRequestDecoderState.layer(layerIndex: 0)
            else {
                Issue.record("the single synthetic layer should be full attention");
                return;
            }
            #expect(fullAttentionLayer.offsetTokens == modelContract.blockTokenCount * 2);
            let tensorShape: [Int] = [1, 2, modelContract.blockTokenCount, 4];
            #expect(fullAttentionLayer.keysState()?.shape == [1, 2, modelContract.blockTokenCount * 2, 4]);
            #expect(fullAttentionLayer.keysState()!.asArray(Float.self)
                == Self.expectedConcatenatedValues(
                    tensorShape: tensorShape, firstBlockBase: 0.0,
                    secondBlockBase: 40.0, modulus: 17, baseOffset: 0.0));
            #expect(fullAttentionLayer.valuesState()!.asArray(Float.self)
                == Self.expectedConcatenatedValues(
                    tensorShape: tensorShape, firstBlockBase: 0.0,
                    secondBlockBase: 40.0, modulus: 13, baseOffset: 0.25));
        }

        @Test
        func should_load_a_published_recurrent_snapshot_back_into_live_decoder_state()
            throws {
            let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
                .syntheticCompositeBoundaryContract();
            let globalRoot: URL = FileManager.default.temporaryDirectory
                .appendingPathComponent("prompt-cache-\(UUID().uuidString)", isDirectory: true);
            try FileManager.default.createDirectory(
                at: globalRoot, withIntermediateDirectories: true);
            let diskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
                .openDiskStore(globalRoot: globalRoot, modelContract: modelContract);
            let rootTokens: [UInt32] = PersistentPromptCacheFixture
                .promptTokensWithCompleteBlocksAndTrailingTokens(
                    modelContract: modelContract, completeBlockCount: 1, trailingTokenCount: 0);
            let rootBlockKey: PersistentPromptCacheBlockKey = try PersistentPromptCacheBlockKey
                .forRootBlock(modelContract: modelContract, blockTokens: rootTokens);
            let mlxStaging: PersistentPromptCacheStateFileMlxStaging =
                PersistentPromptCacheStateFileMlxStaging(
                    sequenceStateTensors: [:],
                    boundaryStateTensors: [
                        "layer_0_linear.convolution": MLXArray(
                            [Float]([3.0, 1.0, 4.0, 1.5]), [1, 1, 4]),
                        "layer_0_linear.gated_delta_recurrent": MLXArray(
                            [Float]([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]), [1, 2, 2, 2]),
                    ]);
            #expect(try diskStore.publishBlock(
                staging: mlxStaging, blockKey: rootBlockKey, parentBlockKey: nil)
                == .published);

            let reopenedStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
                .openDiskStore(globalRoot: globalRoot, modelContract: modelContract);
            let loadedSnapshot: [String: MLXArray]? = try reopenedStore.loadRecurrentSnapshot(
                blockKey: rootBlockKey);
            #expect(loadedSnapshot != nil);

            var loadedSnapshotTensors: [String: MLXArray] = loadedSnapshot!;
            let restoredRequestDecoderState: RequestDecoderStateStack =
                RequestDecoderStateStack(decoderLayerStates: [
                    .composite(
                        convolution: ConvolutionState(),
                        recurrent: GatedDeltaRecurrentState()),
                ]);
            try restoredRequestDecoderState.absorbPersistentPromptCacheRecurrentSnapshot(
                &loadedSnapshotTensors);
            guard case .composite(let convolutionState, let recurrentState) =
                restoredRequestDecoderState.layer(layerIndex: 0)
            else {
                Issue.record("the single synthetic layer should be composite");
                return;
            }
            #expect(convolutionState.state()?.asArray(Float.self) == [3.0, 1.0, 4.0, 1.5]);
            #expect(recurrentState.state()?.asArray(Float.self)
                == [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]);
            #expect(loadedSnapshotTensors.isEmpty);
        }

        @Test
        func should_delete_and_untrack_a_corrupt_block_on_load() throws {
            let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
                .syntheticRankFourSequenceContract();
            let globalRoot: URL = FileManager.default.temporaryDirectory
                .appendingPathComponent("prompt-cache-\(UUID().uuidString)", isDirectory: true);
            try FileManager.default.createDirectory(
                at: globalRoot, withIntermediateDirectories: true);
            let diskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
                .openDiskStore(globalRoot: globalRoot, modelContract: modelContract);
            let rootTokens: [UInt32] = PersistentPromptCacheFixture
                .promptTokensWithCompleteBlocksAndTrailingTokens(
                    modelContract: modelContract, completeBlockCount: 1, trailingTokenCount: 0);
            let rootBlockKey: PersistentPromptCacheBlockKey = try PersistentPromptCacheBlockKey
                .forRootBlock(modelContract: modelContract, blockTokens: rootTokens);
            let mlxStaging: PersistentPromptCacheStateFileMlxStaging =
                PersistentPromptCacheStateFileMlxStaging(
                    sequenceStateTensors: Self.rankFourBlockTensors(
                        blockTokenCount: modelContract.blockTokenCount, valueBase: 0.0),
                    boundaryStateTensors: [:]);
            #expect(try diskStore.publishBlock(
                staging: mlxStaging, blockKey: rootBlockKey, parentBlockKey: nil)
                == .published);

            let reopenedStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
                .openDiskStore(globalRoot: globalRoot, modelContract: modelContract);
            #expect(reopenedStore.sequenceStateBlockCount() == 1);
            let stateFileUrl: URL = reopenedStore.blocksDirectory
                .appendingPathComponent(PersistentPromptCacheStoreFile.hexEncode(
                    rootBlockKey.blockHash()))
                .appendingPathComponent(PersistentPromptCacheStoreFile.SEQUENCE_STATE_FILE_NAME);
            try Data([0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3]).write(to: stateFileUrl);

            #expect(throws: PersistentPromptCacheDiskStoreError.self) {
                _ = try reopenedStore.loadKvBlock(blockKey: rootBlockKey);
            }
            #expect(FileManager.default.fileExists(atPath: stateFileUrl.path) == false,
                "the corrupt published file must be deleted before the failure is raised");
            #expect(reopenedStore.sequenceStateBlockCount() == 0,
                "the corrupt block must leave the index after a failed load");
            let reloadedBlock: [String: MLXArray]? = try reopenedStore.loadKvBlock(
                blockKey: rootBlockKey);
            #expect(reloadedBlock == nil,
                "an untracked corrupt block must load as absent");
        }
    }
}
