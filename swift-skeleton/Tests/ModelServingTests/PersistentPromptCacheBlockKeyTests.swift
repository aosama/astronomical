import Foundation;

import Testing;

import ModelServing;

@testable import ModelServing;

/// Hermetic journeys for persistent model-state block identity: blocks
/// chain under the resolved storage contract, tokens above the
/// contract-derived block size fail closed, identical tokens stay isolated
/// across model layouts and policies, and model-owned causal input binds at
/// the block where it enters the prompt.
final class PersistentPromptCacheBlockKeyTests {

    @Test
    func should_chain_blocks_under_the_resolved_storage_contract() throws {
        let modelContract: PersistentPromptCacheModelContract =
            PersistentPromptCacheBlockKeyTests.syntheticModelContract(
                modelId: "model", modelRevision: "revision", attentionCapacityGrowthTokens: 8);
        let parentBlockTokens: [UInt32] = [UInt32](repeating: 10, count: 8);
        let parentBlockKey: PersistentPromptCacheBlockKey = try #require(
            try? PersistentPromptCacheBlockKey.forRootBlock(
                modelContract: modelContract, blockTokens: parentBlockTokens),
            "the parent block should hash");
        let childBlockKey: PersistentPromptCacheBlockKey = try #require(
            try? parentBlockKey.forChildBlock(blockTokens: [UInt32](repeating: 20, count: 8)),
            "the child block should hash");

        #expect(parentBlockKey.blockIndex() == 0);
        #expect(childBlockKey.blockIndex() == 1);
        #expect(childBlockKey.blockTokenCount() == 8);
        #expect(parentBlockKey.blockHash() != childBlockKey.blockHash());
    }

    @Test
    func should_reject_tokens_above_the_contract_derived_block_size() throws {
        let modelContract: PersistentPromptCacheModelContract =
            PersistentPromptCacheBlockKeyTests.syntheticModelContract(
                modelId: "model", modelRevision: "revision", attentionCapacityGrowthTokens: 4);

        #expect(throws: (any Error).self) {
            _ = try PersistentPromptCacheBlockKey.forRootBlock(
                modelContract: modelContract, blockTokens: [UInt32](repeating: 1, count: 5));
        };
    }

    @Test
    func should_isolate_identical_tokens_across_model_layouts_and_policies() throws {
        let firstModelContract: PersistentPromptCacheModelContract =
            PersistentPromptCacheBlockKeyTests.syntheticModelContract(
                modelId: "model", modelRevision: "revision", attentionCapacityGrowthTokens: 4);
        let secondModelContract: PersistentPromptCacheModelContract =
            PersistentPromptCacheBlockKeyTests.syntheticModelContract(
                modelId: "model", modelRevision: "revision", attentionCapacityGrowthTokens: 8);
        let blockTokens: [UInt32] = [1, 2, 3, 4];

        let firstBlockKey: PersistentPromptCacheBlockKey = try #require(
            try? PersistentPromptCacheBlockKey.forRootBlock(
                modelContract: firstModelContract, blockTokens: blockTokens),
            "the first block should hash");
        let secondBlockKey: PersistentPromptCacheBlockKey = try #require(
            try? PersistentPromptCacheBlockKey.forRootBlock(
                modelContract: secondModelContract, blockTokens: blockTokens),
            "the second block should hash");

        #expect(firstBlockKey.blockHash() != secondBlockKey.blockHash());
    }

    @Test
    func should_bind_causal_inputs_at_the_block_where_they_enter_the_prompt() throws {
        let modelContract: PersistentPromptCacheModelContract =
            PersistentPromptCacheBlockKeyTests.syntheticModelContract(
                modelId: "model", modelRevision: "revision", attentionCapacityGrowthTokens: 8);
        let rootBlockTokens: [UInt32] = [1, 2, 3, 4];
        let visualBlockTokens: [UInt32] = [5, 6, 7, 8];
        let firstVisualInput: PersistentPromptCacheBlockCausalInput =
            PersistentPromptCacheBlockCausalInput(canonicalBytes: Data([1]));
        let secondVisualInput: PersistentPromptCacheBlockCausalInput =
            PersistentPromptCacheBlockCausalInput(canonicalBytes: Data([2]));

        let rootBlockKey: PersistentPromptCacheBlockKey = try #require(
            try? PersistentPromptCacheBlockKey.forRootBlock(
                modelContract: modelContract, blockTokens: rootBlockTokens),
            "the root block should hash");
        let firstVisualBlockKey: PersistentPromptCacheBlockKey = try #require(
            try? rootBlockKey.forChildBlockWithCausalInput(
                blockTokens: visualBlockTokens, blockCausalInput: firstVisualInput),
            "the first visual child should hash");
        let secondVisualBlockKey: PersistentPromptCacheBlockKey = try #require(
            try? rootBlockKey.forChildBlockWithCausalInput(
                blockTokens: visualBlockTokens, blockCausalInput: secondVisualInput),
            "the second visual child should hash");
        let firstDescendantBlockKey: PersistentPromptCacheBlockKey = try #require(
            try? firstVisualBlockKey.forChildBlock(blockTokens: [9, 10, 11, 12]),
            "the first descendant should hash");
        let secondDescendantBlockKey: PersistentPromptCacheBlockKey = try #require(
            try? secondVisualBlockKey.forChildBlock(blockTokens: [9, 10, 11, 12]),
            "the second descendant should hash");

        #expect(firstVisualBlockKey.blockHash() != secondVisualBlockKey.blockHash());
        #expect(firstDescendantBlockKey.blockHash() != secondDescendantBlockKey.blockHash());
    }

    @Test
    func should_keep_text_only_identity_when_the_causal_input_is_empty() throws {
        let modelContract: PersistentPromptCacheModelContract =
            PersistentPromptCacheBlockKeyTests.syntheticModelContract(
                modelId: "model", modelRevision: "revision", attentionCapacityGrowthTokens: 8);
        let blockTokens: [UInt32] = [1, 2, 3, 4];

        let ordinaryBlockKey: PersistentPromptCacheBlockKey = try #require(
            try? PersistentPromptCacheBlockKey.forRootBlock(
                modelContract: modelContract, blockTokens: blockTokens),
            "the ordinary block should hash");
        let explicitEmptyBlockKey: PersistentPromptCacheBlockKey = try #require(
            try? PersistentPromptCacheBlockKey.forRootBlockWithCausalInput(
                modelContract: modelContract,
                blockTokens: blockTokens,
                blockCausalInput: PersistentPromptCacheBlockCausalInput.empty()),
            "the explicitly empty block should hash");

        #expect(ordinaryBlockKey == explicitEmptyBlockKey);
    }

    /// Builds the synthetic contract the Rust journeys share: one
    /// append-only attention layer whose keys and values grow by the
    /// configured token capacity.
    static func syntheticModelContract(
        modelId: String, modelRevision: String, attentionCapacityGrowthTokens: Int
    ) -> PersistentPromptCacheModelContract {
        let decoderCacheLayout: DecoderCacheLayout = try! DecoderCacheLayout(layers: [
            .appendOnlyAttention(
                keys: .sequence(
                    tensorRoleName: "attention.keys",
                    dtype: .float16,
                    dimensions: [1, 0, 2],
                    sequenceAxis: 1),
                values: .sequence(
                    tensorRoleName: "attention.values",
                    dtype: .float16,
                    dimensions: [1, 0, 2],
                    sequenceAxis: 1),
                capacityGrowthTokens: attentionCapacityGrowthTokens),
        ]);
        return try! PersistentPromptCacheModelContract.resolve(
            modelId: modelId,
            modelRevision: modelRevision,
            decoderCacheLayout: decoderCacheLayout,
            maximumContextTokenCount: 128,
            effectiveMlxMemoryCeilingBytes: 1_000_000,
            globalSsdQuotaBytes: 1_000_000,
            configuredBlockTokenCount: nil,
            commonPrefixCheckpointStrideBlocks: 4);
    }
}
