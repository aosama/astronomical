import Foundation;

import Testing;

import ModelServing;

@testable import ModelServing;

/// Hermetic journeys for the persistent-state storage contract: frozen
/// model geometry derives exact tensor shapes and block sizes, mixed
/// execution dtypes survive the layout, sequence-only and boundary-only
/// models resolve, budget overruns fail closed, and configured block
/// lengths must fit alignment and quota exactly instead of being silently
/// resized. The visual-embedding contract assertions ship with the visual cache slice.
final class PersistentPromptCacheModelContractTests {

    private static let TEST_MLX_MEMORY_CEILING_BYTES: UInt64 = 20_000_000_000;
    private static let TEST_SSD_QUOTA_BYTES: UInt64 = 50_000_000_000;

    @Test
    func should_derive_persistent_prompt_cache_tensor_shapes_from_frozen_model_metadata() throws {
        let ornithConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(
            configBytes: try Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes());
        let decoderLayerCacheDtypes: [Qwen35DecoderLayerCacheDtypes] =
            try PersistentPromptCacheFixture.bfloat16DecoderLayerCacheDtypes(
                qwen35Config: ornithConfig);

        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheModelContract
            .resolve(
                modelId: PersistentPromptCacheFixture.ORNITH_MODEL_ID,
                modelRevision: PersistentPromptCacheFixture.ORNITH_MODEL_REVISION,
                decoderCacheLayout: try Qwen35DecoderCacheLayoutBuilder.buildDecoderCacheLayout(
                    qwen35Config: ornithConfig,
                    fullAttentionKeyValueGrowthTokens: 256,
                    decoderLayerCacheDtypes: decoderLayerCacheDtypes),
                maximumContextTokenCount: Int(ornithConfig.maximumPositionCount()),
                effectiveMlxMemoryCeilingBytes: Self.TEST_MLX_MEMORY_CEILING_BYTES,
                globalSsdQuotaBytes: Self.TEST_SSD_QUOTA_BYTES,
                configuredBlockTokenCount: nil,
                commonPrefixCheckpointStrideBlocks: 4);

        #expect(modelContract.modelId == PersistentPromptCacheFixture.ORNITH_MODEL_ID);
        #expect(modelContract.modelRevision == PersistentPromptCacheFixture.ORNITH_MODEL_REVISION);
        #expect(modelContract.decoderCacheLayout.layerCount == 40);
        #expect(try #require(
            modelContract.decoderCacheLayout.sequenceTensorLayouts().first,
            "the frozen layout should contain sequence state").tensorLayout.dimensions == [1, 2, 0, 256]);
        #expect(try #require(
            modelContract.decoderCacheLayout.boundaryTensorLayouts().first,
            "the frozen layout should contain boundary state").tensorLayout.dimensions == [1, 3, 8_192]);
        #expect(modelContract.decoderCacheLayout.sequenceTensorCount == 20);
        #expect(modelContract.decoderCacheLayout.boundaryTensorCount == 60);
        let sequenceTensorLayouts: [DecoderCachePersistedTensorLayout] = modelContract
            .decoderCacheLayout.sequenceTensorLayouts();
        #expect(sequenceTensorLayouts.allSatisfy({ (tensorLayout) in
            return tensorLayout.tensorLayout.dtype == .bfloat16;
        }));
        let boundaryTensorLayouts: [DecoderCachePersistedTensorLayout] = modelContract
            .decoderCacheLayout.boundaryTensorLayouts();
        #expect(boundaryTensorLayouts.filter({ (tensorLayout) in
            return tensorLayout.tensorLayout.dtype == .bfloat16;
        }).count == 30);
        #expect(boundaryTensorLayouts.filter({ (tensorLayout) in
            return tensorLayout.tensorLayout.dtype == .float32;
        }).count == 30);
        let blockTokenCount: Int = modelContract.blockTokenCount;
        #expect(blockTokenCount % 256 == 0);
        #expect(blockTokenCount <= Int(ornithConfig.maximumPositionCount()));
        #expect(modelContract.sequenceStatePayloadBytesPerBlock
            >= modelContract.boundaryStatePayloadBytes);
        #expect(modelContract.sequenceStatePayloadBytesPerBlock == blockTokenCount * 20_480);
        #expect(modelContract.storageContractFingerprint() != Data(repeating: 0, count: 32));
    }

    @Test
    func should_preserve_mixed_execution_dtypes_in_persistent_prompt_cache_geometry() throws {
        let ornithConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(
            configBytes: try Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes());
        var decoderLayerCacheDtypes: [Qwen35DecoderLayerCacheDtypes] =
            try PersistentPromptCacheFixture.bfloat16DecoderLayerCacheDtypes(
                qwen35Config: ornithConfig);
        decoderLayerCacheDtypes[0] = .linearAttention(convolution: .float32);
        decoderLayerCacheDtypes[3] = .fullAttention(keys: .float32, values: .float32);

        let decoderCacheLayout: DecoderCacheLayout = try Qwen35DecoderCacheLayoutBuilder
            .buildDecoderCacheLayout(
                qwen35Config: ornithConfig,
                fullAttentionKeyValueGrowthTokens: 256,
                decoderLayerCacheDtypes: decoderLayerCacheDtypes);

        let firstBoundaryTensor: DecoderCachePersistedTensorLayout = try #require(
            decoderCacheLayout.boundaryTensorLayouts().first(where: { (tensorLayout) in
                return tensorLayout.decoderLayerIndex == 0;
            }),
            "the first linear-attention layer should have boundary state");
        #expect(firstBoundaryTensor.tensorLayout.dtype == .float32);
        let firstFullAttentionTensors: [DecoderCachePersistedTensorLayout] = decoderCacheLayout
            .sequenceTensorLayouts()
            .filter({ (tensorLayout) in tensorLayout.decoderLayerIndex == 3 });
        #expect(firstFullAttentionTensors.count == 2);
        #expect(firstFullAttentionTensors.allSatisfy({ (tensorLayout) in
            return tensorLayout.tensorLayout.dtype == .float32;
        }));
    }

    @Test
    func should_reject_decoder_cache_execution_dtypes_with_a_missing_layer() throws {
        let ornithConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(
            configBytes: try Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes());
        var decoderLayerCacheDtypes: [Qwen35DecoderLayerCacheDtypes] =
            try PersistentPromptCacheFixture.bfloat16DecoderLayerCacheDtypes(
                qwen35Config: ornithConfig);
        decoderLayerCacheDtypes.removeLast();

        #expect(throws: DecoderCacheLayoutError.executionDtypeLayerCountMismatch(
            expectedLayerCount: 40, actualLayerCount: 39)) {
            _ = try Qwen35DecoderCacheLayoutBuilder.buildDecoderCacheLayout(
                qwen35Config: ornithConfig,
                fullAttentionKeyValueGrowthTokens: 256,
                decoderLayerCacheDtypes: decoderLayerCacheDtypes);
        };
    }

    @Test
    func should_reject_decoder_cache_execution_dtypes_for_the_wrong_attention_family() throws {
        let ornithConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(
            configBytes: try Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes());
        var decoderLayerCacheDtypes: [Qwen35DecoderLayerCacheDtypes] =
            try PersistentPromptCacheFixture.bfloat16DecoderLayerCacheDtypes(
                qwen35Config: ornithConfig);
        decoderLayerCacheDtypes[0] = .fullAttention(keys: .bfloat16, values: .bfloat16);

        #expect(throws: DecoderCacheLayoutError.executionDtypeLayerFamilyMismatch(layerIndex: 0)) {
            _ = try Qwen35DecoderCacheLayoutBuilder.buildDecoderCacheLayout(
                qwen35Config: ornithConfig,
                fullAttentionKeyValueGrowthTokens: 256,
                decoderLayerCacheDtypes: decoderLayerCacheDtypes);
        };
    }

    @Test
    func should_derive_different_block_sizes_from_sequence_and_boundary_geometry() throws {
        let smallBoundaryContract: PersistentPromptCacheModelContract = try Self
            .resolveSyntheticContract(
                modelId: "small-boundary",
                decoderCacheLayout: Self.syntheticHybridLayout(boundaryElementCount: 8),
                maximumContextTokenCount: 128,
                effectiveMlxMemoryCeilingBytes: 1_000_000,
                globalSsdQuotaBytes: 1_000_000);
        let largeBoundaryContract: PersistentPromptCacheModelContract = try Self
            .resolveSyntheticContract(
                modelId: "large-boundary",
                decoderCacheLayout: Self.syntheticHybridLayout(boundaryElementCount: 64),
                maximumContextTokenCount: 128,
                effectiveMlxMemoryCeilingBytes: 1_000_000,
                globalSsdQuotaBytes: 1_000_000);

        #expect(smallBoundaryContract.blockTokenCount == 4);
        #expect(largeBoundaryContract.blockTokenCount == 32);
        #expect(smallBoundaryContract.storageContractFingerprint()
            != largeBoundaryContract.storageContractFingerprint());
    }

    @Test
    func should_resolve_sequence_only_and_boundary_only_storage_contracts() throws {
        let sequenceOnlyLayout: DecoderCacheLayout = try DecoderCacheLayout(layers: [
            .appendOnlyAttention(
                keys: .sequence(
                    tensorRoleName: "attention.keys", dtype: .float16, dimensions: [1, 0, 4],
                    sequenceAxis: 1),
                values: .sequence(
                    tensorRoleName: "attention.values", dtype: .float16, dimensions: [1, 0, 4],
                    sequenceAxis: 1),
                capacityGrowthTokens: 16),
        ]);
        let boundaryOnlyLayout: DecoderCacheLayout = try DecoderCacheLayout(layers: [
            .recurrentTensor(tensor: .fixed(
                tensorRoleName: "recurrent.state", dtype: .float32, dimensions: [25])),
        ]);

        let sequenceOnlyContract: PersistentPromptCacheModelContract = try Self
            .resolveSyntheticContract(
                modelId: "sequence-only", decoderCacheLayout: sequenceOnlyLayout,
                maximumContextTokenCount: 128,
                effectiveMlxMemoryCeilingBytes: 1_000_000, globalSsdQuotaBytes: 1_000_000);
        let boundaryOnlyContract: PersistentPromptCacheModelContract = try Self
            .resolveSyntheticContract(
                modelId: "boundary-only", decoderCacheLayout: boundaryOnlyLayout,
                maximumContextTokenCount: 100,
                effectiveMlxMemoryCeilingBytes: 1_000_000, globalSsdQuotaBytes: 10_000);

        #expect(sequenceOnlyContract.blockTokenCount == 16);
        #expect(sequenceOnlyContract.hasSequenceState);
        #expect(sequenceOnlyContract.hasBoundaryState == false);
        #expect(boundaryOnlyContract.blockTokenCount <= 100);
        #expect(boundaryOnlyContract.hasSequenceState == false);
        #expect(boundaryOnlyContract.hasBoundaryState);
        for storageContract in [sequenceOnlyContract, boundaryOnlyContract] {
            let maximumCommittedBlockCount: Int =
                (storageContract.maximumContextTokenCount
                    + storageContract.blockTokenCount - 1) / storageContract.blockTokenCount;
            #expect(
                storageContract.maximumCommittedBlockBytes
                    * UInt64(maximumCommittedBlockCount)
                    <= (storageContract.hasSequenceState ? 1_000_000 : 10_000),
                "the complete active chain must fit its exact committed-byte quota");
        }
    }

    @Test
    func should_reject_a_contract_when_one_exact_capture_exceeds_a_budget() throws {
        #expect(throws: (any Error).self) {
            _ = try PersistentPromptCacheModelContract.resolve(
                modelId: "fictional-model",
                modelRevision: "revision",
                decoderCacheLayout: Self.syntheticHybridLayout(boundaryElementCount: 64),
                maximumContextTokenCount: 128,
                effectiveMlxMemoryCeilingBytes: 100,
                globalSsdQuotaBytes: 1_000_000,
                configuredBlockTokenCount: nil,
                commonPrefixCheckpointStrideBlocks: 4);
        };
    }

    @Test
    func should_apply_configured_prompt_cache_block_and_common_prefix_boundaries() throws {
        let configuredContract: PersistentPromptCacheModelContract = try PersistentPromptCacheModelContract
            .resolve(
                modelId: "fictional-model",
                modelRevision: "revision",
                decoderCacheLayout: Self.syntheticHybridLayout(boundaryElementCount: 8),
                maximumContextTokenCount: 128,
                effectiveMlxMemoryCeilingBytes: 1_000_000,
                globalSsdQuotaBytes: 1_000_000,
                configuredBlockTokenCount: 16,
                commonPrefixCheckpointStrideBlocks: 6);

        #expect(configuredContract.blockTokenCount == 16);
        #expect(configuredContract.commonPrefixCheckpointStrideBlocks == 6);
    }

    @Test
    func should_reject_a_configured_prompt_cache_block_that_breaks_model_alignment() throws {
        #expect(throws: (any Error).self) {
            _ = try PersistentPromptCacheModelContract.resolve(
                modelId: "fictional-model",
                modelRevision: "revision",
                decoderCacheLayout: Self.syntheticHybridLayout(boundaryElementCount: 8),
                maximumContextTokenCount: 128,
                effectiveMlxMemoryCeilingBytes: 1_000_000,
                globalSsdQuotaBytes: 1_000_000,
                configuredBlockTokenCount: 6,
                commonPrefixCheckpointStrideBlocks: 4);
        };
    }

    @Test
    func should_reject_zero_common_prefix_checkpoint_stride_at_the_storage_boundary() throws {
        #expect(throws: PersistentPromptCacheModelContractError
            .zeroCommonPrefixCheckpointStrideBlocks) {
            _ = try PersistentPromptCacheModelContract.resolve(
                modelId: "fictional-model",
                modelRevision: "revision",
                decoderCacheLayout: Self.syntheticHybridLayout(boundaryElementCount: 8),
                maximumContextTokenCount: 128,
                effectiveMlxMemoryCeilingBytes: 1_000_000,
                globalSsdQuotaBytes: 1_000_000,
                configuredBlockTokenCount: 16,
                commonPrefixCheckpointStrideBlocks: 0);
        };
    }

    @Test
    func should_reject_an_explicit_block_length_instead_of_silently_resizing_it_for_quota() throws {
        do {
            _ = try PersistentPromptCacheModelContract.resolve(
                modelId: "fictional-model",
                modelRevision: "revision",
                decoderCacheLayout: Self.syntheticHybridLayout(boundaryElementCount: 8),
                maximumContextTokenCount: 128,
                effectiveMlxMemoryCeilingBytes: 1_000_000,
                globalSsdQuotaBytes: 1,
                configuredBlockTokenCount: 16,
                commonPrefixCheckpointStrideBlocks: 4);
            Issue.record("an over-quota configured block length must be rejected");
        } catch let rejection as PersistentPromptCacheModelContractError {
            guard case .configuredBlockChainExceedsSsdQuota(let configuredBlockTokens, _, _) =
                rejection
            else {
                Issue.record("expected the quota rejection, got \(rejection)");
                return;
            }
            #expect(configuredBlockTokens == 16);
        }
    }

    private static func resolveSyntheticContract(
        modelId: String,
        decoderCacheLayout: DecoderCacheLayout,
        maximumContextTokenCount: Int,
        effectiveMlxMemoryCeilingBytes: UInt64,
        globalSsdQuotaBytes: UInt64
    ) throws -> PersistentPromptCacheModelContract {
        return try PersistentPromptCacheModelContract.resolve(
            modelId: modelId,
            modelRevision: "revision",
            decoderCacheLayout: decoderCacheLayout,
            maximumContextTokenCount: maximumContextTokenCount,
            effectiveMlxMemoryCeilingBytes: effectiveMlxMemoryCeilingBytes,
            globalSsdQuotaBytes: globalSsdQuotaBytes,
            configuredBlockTokenCount: nil,
            commonPrefixCheckpointStrideBlocks: 4);
    }

    private static func syntheticHybridLayout(
        boundaryElementCount: Int
    ) throws -> DecoderCacheLayout {
        return try DecoderCacheLayout(layers: [
            .composite(components: [
                .appendOnlyAttention(
                    keys: .sequence(
                        tensorRoleName: "attention.keys", dtype: .float16, dimensions: [1, 0, 2],
                        sequenceAxis: 1),
                    values: .sequence(
                        tensorRoleName: "attention.values", dtype: .float16, dimensions: [1, 0, 2],
                        sequenceAxis: 1),
                    capacityGrowthTokens: 4),
                .recurrentTensor(tensor: .fixed(
                    tensorRoleName: "recurrent.state",
                    dtype: .float32,
                    dimensions: [boundaryElementCount])),
            ]),
        ]);
    }
}
