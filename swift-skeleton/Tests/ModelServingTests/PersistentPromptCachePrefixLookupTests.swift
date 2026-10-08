import Foundation;

import Testing;

import ModelServing;

@testable import ModelServing;

/// Hermetic journeys for the pure prompt-prefix lookup: chain-ordered block
/// hashing, walk-back to the newest usable recurrent snapshot, the final
/// token retained for forward processing, root divergence, causal-input
/// binding, and cold misses with typed diagnostics.
final class PersistentPromptCachePrefixLookupTests {

    @Test
    func should_report_no_cache_hit_when_no_complete_blocks_exist() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let promptTokens: [UInt32] = (0..<1_500).map({ (tokenIndex: Int) -> UInt32 in
            return UInt32(tokenIndex);
        });
        let lookupResult: PersistentPromptCachePrefixLookupResult = PersistentPromptCachePrefixLookup
            .forPrompt(
                modelContract: modelContract,
                promptTokens: promptTokens,
                kvBlockExists: { (_: Data) in false },
                recurrentSnapshotExists: { (_: Data) in false });

        #expect(lookupResult.restoredTokenCount == 0);
        #expect(lookupResult.remainingTokens == promptTokens);
        #expect(lookupResult.lastRestoredBlockKey == nil);
    }

    @Test
    func should_restore_through_the_chain_tip_when_its_recurrent_snapshot_exists() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let promptTokens: [UInt32] = PersistentPromptCacheFixture
            .promptTokensWithCompleteBlocksAndTrailingTokens(
                modelContract: modelContract, completeBlockCount: 2, trailingTokenCount: 100);
        let blockKeys: [PersistentPromptCacheBlockKey] = try PersistentPromptCacheFixture
            .blockKeysForPrompt(
                modelContract: modelContract, promptTokens: promptTokens, requestedBlockCount: 2);
        let chainTipBlockHash: Data = blockKeys[1].blockHash();

        let lookupResult: PersistentPromptCachePrefixLookupResult = PersistentPromptCachePrefixLookup
            .forPrompt(
                modelContract: modelContract,
                promptTokens: promptTokens,
                kvBlockExists: { (blockHash: Data) in
                    return blockKeys.contains(where: { (blockKey: PersistentPromptCacheBlockKey) in
                        return blockKey.blockHash() == blockHash;
                    });
                },
                recurrentSnapshotExists: { (blockHash: Data) in
                    return blockHash == chainTipBlockHash;
                });

        #expect(lookupResult.restoredTokenCount == modelContract.blockTokenCount * 2);
        #expect(lookupResult.remainingTokens.count == 100);
        #expect(lookupResult.lastRestoredBlockKey?.blockHash() == chainTipBlockHash);
    }

    @Test
    func should_walk_back_to_the_latest_available_recurrent_snapshot() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let promptTokens: [UInt32] = PersistentPromptCacheFixture
            .promptTokensWithCompleteBlocksAndTrailingTokens(
                modelContract: modelContract, completeBlockCount: 3, trailingTokenCount: 500);
        let blockKeys: [PersistentPromptCacheBlockKey] = try PersistentPromptCacheFixture
            .blockKeysForPrompt(
                modelContract: modelContract, promptTokens: promptTokens, requestedBlockCount: 3);
        let firstBlockHash: Data = blockKeys[0].blockHash();

        let lookupResult: PersistentPromptCachePrefixLookupResult = PersistentPromptCachePrefixLookup
            .forPrompt(
                modelContract: modelContract,
                promptTokens: promptTokens,
                kvBlockExists: { (blockHash: Data) in
                    return blockKeys.contains(where: { (blockKey: PersistentPromptCacheBlockKey) in
                        return blockKey.blockHash() == blockHash;
                    });
                },
                recurrentSnapshotExists: { (blockHash: Data) in
                    return blockHash == firstBlockHash;
                });

        #expect(lookupResult.restoredTokenCount == modelContract.blockTokenCount);
        #expect(lookupResult.remainingTokens.count == promptTokens.count
            - modelContract.blockTokenCount);
        #expect(lookupResult.lastRestoredBlockKey?.blockHash() == firstBlockHash);
        #expect(lookupResult.diagnostics.matchedSequenceStateBlockCount == 3);
        #expect(lookupResult.diagnostics.newestBoundaryStateSnapshotBlockIndex == 0);
        #expect(lookupResult.diagnostics.missReason == nil);
    }

    @Test
    func should_return_a_cold_miss_when_kv_blocks_exist_without_any_recurrent_snapshot() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let promptTokens: [UInt32] = PersistentPromptCacheFixture
            .promptTokensWithCompleteBlocksAndTrailingTokens(
                modelContract: modelContract, completeBlockCount: 2, trailingTokenCount: 100);
        let blockKeys: [PersistentPromptCacheBlockKey] = try PersistentPromptCacheFixture
            .blockKeysForPrompt(
                modelContract: modelContract, promptTokens: promptTokens, requestedBlockCount: 2);

        let lookupResult: PersistentPromptCachePrefixLookupResult = PersistentPromptCachePrefixLookup
            .forPrompt(
                modelContract: modelContract,
                promptTokens: promptTokens,
                kvBlockExists: { (blockHash: Data) in
                    return blockKeys.contains(where: { (blockKey: PersistentPromptCacheBlockKey) in
                        return blockKey.blockHash() == blockHash;
                    });
                },
                recurrentSnapshotExists: { (_: Data) in false });

        #expect(lookupResult.restoredTokenCount == 0);
        #expect(lookupResult.remainingTokens.count == promptTokens.count);
        #expect(lookupResult.lastRestoredBlockKey == nil);
        #expect(lookupResult.diagnostics.matchedSequenceStateBlockCount == 2);
        #expect(lookupResult.diagnostics.firstMissingSequenceStateBlockIndex == nil);
        #expect(lookupResult.diagnostics.newestBoundaryStateSnapshotBlockIndex == nil);
        #expect(lookupResult.diagnostics.missReason == .boundaryStateSnapshotMissing);
    }

    @Test
    func should_keep_the_final_block_for_forward_processing_when_prompt_ends_on_a_block_boundary()
        throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let promptTokens: [UInt32] = PersistentPromptCacheFixture
            .promptTokensWithCompleteBlocksAndTrailingTokens(
                modelContract: modelContract, completeBlockCount: 2, trailingTokenCount: 0);
        let firstBlockKey: PersistentPromptCacheBlockKey = try #require(
            try PersistentPromptCacheFixture.blockKeysForPrompt(
                modelContract: modelContract, promptTokens: promptTokens, requestedBlockCount: 1)
                .last,
            "the test should produce the first block key");
        let firstBlockHash: Data = firstBlockKey.blockHash();

        let lookupResult: PersistentPromptCachePrefixLookupResult = PersistentPromptCachePrefixLookup
            .forPrompt(
                modelContract: modelContract,
                promptTokens: promptTokens,
                kvBlockExists: { (_: Data) in true },
                recurrentSnapshotExists: { (blockHash: Data) in
                    return blockHash == firstBlockHash;
                });

        #expect(lookupResult.restoredTokenCount == modelContract.blockTokenCount);
        #expect(lookupResult.remainingTokens.count == modelContract.blockTokenCount);
        #expect(lookupResult.lastRestoredBlockKey?.blockHash() == firstBlockHash);
    }

    @Test
    func should_restore_a_complete_exact_boundary_when_the_caller_only_needs_decoder_state()
        throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let promptTokens: [UInt32] = PersistentPromptCacheFixture
            .promptTokensWithCompleteBlocksAndTrailingTokens(
                modelContract: modelContract, completeBlockCount: 2, trailingTokenCount: 0);
        let blockKeys: [PersistentPromptCacheBlockKey] = try PersistentPromptCacheFixture
            .blockKeysForPrompt(
                modelContract: modelContract, promptTokens: promptTokens, requestedBlockCount: 2);
        let finalBlockHash: Data = blockKeys[1].blockHash();

        let lookupResult: PersistentPromptCachePrefixLookupResult = PersistentPromptCachePrefixLookup
            .forCompletePrefix(
                modelContract: modelContract,
                promptTokens: promptTokens,
                kvBlockExists: { (blockHash: Data) in
                    return blockKeys.contains(where: { (blockKey: PersistentPromptCacheBlockKey) in
                        return blockKey.blockHash() == blockHash;
                    });
                },
                recurrentSnapshotExists: { (blockHash: Data) in
                    return blockHash == finalBlockHash;
                });

        #expect(lookupResult.restoredTokenCount == modelContract.blockTokenCount * 2);
        #expect(lookupResult.remainingTokens.isEmpty);
        #expect(lookupResult.lastRestoredBlockKey?.blockHash() == finalBlockHash);
    }

    @Test
    func should_not_match_blocks_when_the_root_block_differs() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let originalPromptTokens: [UInt32] = PersistentPromptCacheFixture
            .promptTokensWithCompleteBlocksAndTrailingTokens(
                modelContract: modelContract, completeBlockCount: 3, trailingTokenCount: 0);
        var modifiedPromptTokens: [UInt32] = originalPromptTokens;
        modifiedPromptTokens[0] = 999;
        let originalRootBlockKey: PersistentPromptCacheBlockKey = try #require(
            try? PersistentPromptCacheBlockKey.forRootBlock(
                modelContract: modelContract,
                blockTokens: Array(originalPromptTokens[..<modelContract.blockTokenCount])),
            "the test should hash the original root block");
        let requestedRootBlockKey: PersistentPromptCacheBlockKey = try #require(
            try? PersistentPromptCacheBlockKey.forRootBlock(
                modelContract: modelContract,
                blockTokens: Array(modifiedPromptTokens[..<modelContract.blockTokenCount])),
            "the test should hash the requested root block");

        let lookupResult: PersistentPromptCachePrefixLookupResult = PersistentPromptCachePrefixLookup
            .forPrompt(
                modelContract: modelContract,
                promptTokens: modifiedPromptTokens,
                kvBlockExists: { (blockHash: Data) in
                    return blockHash == originalRootBlockKey.blockHash();
                },
                recurrentSnapshotExists: { (blockHash: Data) in
                    return blockHash == originalRootBlockKey.blockHash();
                });

        #expect(lookupResult.restoredTokenCount == 0);
        #expect(lookupResult.lastRestoredBlockKey == nil);
        #expect(lookupResult.diagnostics.matchedSequenceStateBlockCount == 0);
        #expect(lookupResult.diagnostics.firstMissingSequenceStateBlockIndex == 0);
        #expect(lookupResult.diagnostics.firstMissingSequenceStateBlockHash
            == requestedRootBlockKey.blockHash());
        #expect(lookupResult.diagnostics.missReason == .rootSequenceStateBlockMissing);
    }

    @Test
    func should_restore_blocks_before_changed_visual_content() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let promptTokens: [UInt32] = PersistentPromptCacheFixture
            .promptTokensWithCompleteBlocksAndTrailingTokens(
                modelContract: modelContract, completeBlockCount: 2, trailingTokenCount: 100);
        let emptyCausalInput: PersistentPromptCacheBlockCausalInput =
            PersistentPromptCacheBlockCausalInput.empty();
        let cachedVisualInput: PersistentPromptCacheBlockCausalInput =
            PersistentPromptCacheBlockCausalInput(canonicalBytes: Data([1]));
        let requestedVisualInput: PersistentPromptCacheBlockCausalInput =
            PersistentPromptCacheBlockCausalInput(canonicalBytes: Data([2]));
        let cachedRootBlockKey: PersistentPromptCacheBlockKey = try #require(
            try? PersistentPromptCacheBlockKey.forRootBlockWithCausalInput(
                modelContract: modelContract,
                blockTokens: Array(promptTokens[..<modelContract.blockTokenCount]),
                blockCausalInput: emptyCausalInput),
            "the root block should hash");
        let cachedChildBlockKey: PersistentPromptCacheBlockKey = try #require(
            try? cachedRootBlockKey.forChildBlockWithCausalInput(
                blockTokens: Array(promptTokens[
                    modelContract.blockTokenCount..<(modelContract.blockTokenCount * 2)]),
                blockCausalInput: cachedVisualInput),
            "the visual child block should hash");

        let lookupResult: PersistentPromptCachePrefixLookupResult = PersistentPromptCachePrefixLookup
            .forPromptWithBlockCausalInputs(
                modelContract: modelContract,
                promptTokens: promptTokens,
                blockCausalInputs: [emptyCausalInput, requestedVisualInput],
                kvBlockExists: { (blockHash: Data) in
                    return blockHash == cachedRootBlockKey.blockHash()
                        || blockHash == cachedChildBlockKey.blockHash();
                },
                recurrentSnapshotExists: { (blockHash: Data) in
                    return blockHash == cachedRootBlockKey.blockHash();
                });

        #expect(lookupResult.restoredTokenCount == modelContract.blockTokenCount);
        #expect(lookupResult.diagnostics.firstMissingSequenceStateBlockIndex == 1);
        #expect(lookupResult.diagnostics.missReason == nil);
    }

    @Test
    func should_report_missing_recurrent_snapshot_before_missing_child_when_matched_prefix_has_no_snapshot()
        throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let promptTokens: [UInt32] = PersistentPromptCacheFixture
            .promptTokensWithCompleteBlocksAndTrailingTokens(
                modelContract: modelContract, completeBlockCount: 3, trailingTokenCount: 100);
        let blockKeys: [PersistentPromptCacheBlockKey] = try PersistentPromptCacheFixture
            .blockKeysForPrompt(
                modelContract: modelContract, promptTokens: promptTokens, requestedBlockCount: 3);
        let firstBlockHash: Data = blockKeys[0].blockHash();

        let lookupResult: PersistentPromptCachePrefixLookupResult = PersistentPromptCachePrefixLookup
            .forPrompt(
                modelContract: modelContract,
                promptTokens: promptTokens,
                kvBlockExists: { (blockHash: Data) in
                    return blockHash == firstBlockHash;
                },
                recurrentSnapshotExists: { (_: Data) in false });

        #expect(lookupResult.restoredTokenCount == 0);
        #expect(lookupResult.diagnostics.completePromptBlockCount == 3);
        #expect(lookupResult.diagnostics.maximumRestorableBlockCount == 3);
        #expect(lookupResult.diagnostics.matchedSequenceStateBlockCount == 1);
        #expect(lookupResult.diagnostics.firstMissingSequenceStateBlockIndex == 1);
        #expect(lookupResult.diagnostics.firstMissingSequenceStateBlockHash
            == blockKeys[1].blockHash());
        #expect(lookupResult.diagnostics.newestBoundaryStateSnapshotBlockIndex == nil);
        #expect(lookupResult.diagnostics.missReason == .boundaryStateSnapshotMissing);
    }
}
