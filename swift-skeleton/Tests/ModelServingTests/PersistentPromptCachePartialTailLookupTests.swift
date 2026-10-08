import Foundation;

import Testing;

import ModelServing;

@testable import ModelServing;

/// Hermetic journeys for partial tail recovery: the longest stored tail
/// under one parent wins, root tails serve prompts shorter than one block,
/// probes stay off when the newest snapshot sits below the chain tip or the
/// uncached suffix holds complete blocks, and the existing lookup
/// constructors stay tail-blind.
final class PersistentPromptCachePartialTailLookupTests {

    @Test
    func should_restore_a_partial_tail_block_on_top_of_a_matched_chain_tip() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let promptTokens: [UInt32] = PersistentPromptCacheFixture
            .promptTokensWithCompleteBlocksAndTrailingTokens(
                modelContract: modelContract, completeBlockCount: 2, trailingTokenCount: 100);
        let blockKeys: [PersistentPromptCacheBlockKey] = try PersistentPromptCacheFixture
            .blockKeysForPrompt(
                modelContract: modelContract, promptTokens: promptTokens, requestedBlockCount: 2);
        let chainTipBlockHash: Data = blockKeys[1].blockHash();
        let tailBlockKey: PersistentPromptCacheBlockKey = try PersistentPromptCachePartialTailLookupTests
            .chainTipBlockKey(chainTip: blockKeys[1], promptTokens: promptTokens, tailTokenCount: 99);

        let lookupResult: PersistentPromptCachePrefixLookupResult = PersistentPromptCachePrefixLookup
            .forPromptWithPartialTailBlockRecovery(
                modelContract: modelContract,
                promptTokens: promptTokens,
                blockCausalInputs: [],
                kvBlockExists: { (blockHash: Data) in
                    return blockKeys.contains(where: { (blockKey: PersistentPromptCacheBlockKey) in
                        return blockKey.blockHash() == blockHash;
                    }) || blockHash == tailBlockKey.blockHash();
                },
                recurrentSnapshotExists: { (blockHash: Data) in
                    return blockHash == chainTipBlockHash || blockHash == tailBlockKey.blockHash();
                });

        #expect(lookupResult.restoredTokenCount == modelContract.blockTokenCount * 2 + 99);
        #expect(lookupResult.remainingTokens.count == 1);
        #expect(lookupResult.lastRestoredBlockKey?.blockHash() == chainTipBlockHash);
        let restoredTailBlockKey: PersistentPromptCacheBlockKey = try #require(
            lookupResult.restoredPartialTailBlockKey, "the stored tail should restore");
        #expect(restoredTailBlockKey.blockHash() == tailBlockKey.blockHash());
        #expect(restoredTailBlockKey.tokenCount() == 99);
        #expect(lookupResult.diagnostics.restoredPartialTailBlockTokenCount == 99);
    }

    @Test
    func should_prefer_the_longest_stored_tail_under_one_parent() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let promptTokens: [UInt32] = PersistentPromptCacheFixture
            .promptTokensWithCompleteBlocksAndTrailingTokens(
                modelContract: modelContract, completeBlockCount: 2, trailingTokenCount: 100);
        let blockKeys: [PersistentPromptCacheBlockKey] = try PersistentPromptCacheFixture
            .blockKeysForPrompt(
                modelContract: modelContract, promptTokens: promptTokens, requestedBlockCount: 2);
        let chainTipBlockHash: Data = blockKeys[1].blockHash();
        let longestTailBlockKey: PersistentPromptCacheBlockKey = try PersistentPromptCachePartialTailLookupTests
            .chainTipBlockKey(chainTip: blockKeys[1], promptTokens: promptTokens, tailTokenCount: 99);
        let shorterTailBlockKey: PersistentPromptCacheBlockKey = try PersistentPromptCachePartialTailLookupTests
            .chainTipBlockKey(chainTip: blockKeys[1], promptTokens: promptTokens, tailTokenCount: 50);

        let lookupResult: PersistentPromptCachePrefixLookupResult = PersistentPromptCachePrefixLookup
            .forPromptWithPartialTailBlockRecovery(
                modelContract: modelContract,
                promptTokens: promptTokens,
                blockCausalInputs: [],
                kvBlockExists: { (blockHash: Data) in
                    return blockKeys.contains(where: { (blockKey: PersistentPromptCacheBlockKey) in
                        return blockKey.blockHash() == blockHash;
                    }) || blockHash == longestTailBlockKey.blockHash()
                        || blockHash == shorterTailBlockKey.blockHash();
                },
                recurrentSnapshotExists: { (blockHash: Data) in
                    return blockHash == chainTipBlockHash
                        || blockHash == longestTailBlockKey.blockHash()
                        || blockHash == shorterTailBlockKey.blockHash();
                });

        #expect(lookupResult.restoredPartialTailBlockKey?.tokenCount() == 99);
    }

    @Test
    func should_skip_the_tail_probe_when_no_tail_files_exist() throws {
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
            .forPromptWithPartialTailBlockRecovery(
                modelContract: modelContract,
                promptTokens: promptTokens,
                blockCausalInputs: [],
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
        #expect(lookupResult.restoredPartialTailBlockKey == nil);
        #expect(lookupResult.diagnostics.restoredPartialTailBlockTokenCount == nil);
    }

    @Test
    func should_restore_a_root_tail_for_a_prompt_shorter_than_one_block() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let promptTokenCount: Int = modelContract.blockTokenCount / 2;
        let promptTokens: [UInt32] = (0..<UInt32(promptTokenCount)).map({ $0 });
        let tailBlockKey: PersistentPromptCacheBlockKey = try #require(
            try? PersistentPromptCacheBlockKey.forRootBlock(
                modelContract: modelContract,
                blockTokens: Array(promptTokens[..<(promptTokenCount - 1)])),
            "the root tail should hash");

        let lookupResult: PersistentPromptCachePrefixLookupResult = PersistentPromptCachePrefixLookup
            .forPromptWithPartialTailBlockRecovery(
                modelContract: modelContract,
                promptTokens: promptTokens,
                blockCausalInputs: [],
                kvBlockExists: { (blockHash: Data) in
                    return blockHash == tailBlockKey.blockHash();
                },
                recurrentSnapshotExists: { (blockHash: Data) in
                    return blockHash == tailBlockKey.blockHash();
                });

        #expect(lookupResult.restoredTokenCount == promptTokenCount - 1);
        #expect(lookupResult.remainingTokens.count == 1);
        #expect(lookupResult.lastRestoredBlockKey == nil);
        #expect(lookupResult.restoredPartialTailBlockKey?.tokenCount() == promptTokenCount - 1);
    }

    @Test
    func should_not_probe_a_tail_when_the_newest_snapshot_sits_below_the_chain_tip() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let promptTokens: [UInt32] = PersistentPromptCacheFixture
            .promptTokensWithCompleteBlocksAndTrailingTokens(
                modelContract: modelContract, completeBlockCount: 2, trailingTokenCount: 100);
        let blockKeys: [PersistentPromptCacheBlockKey] = try PersistentPromptCacheFixture
            .blockKeysForPrompt(
                modelContract: modelContract, promptTokens: promptTokens, requestedBlockCount: 2);
        let firstBlockHash: Data = blockKeys[0].blockHash();

        let lookupResult: PersistentPromptCachePrefixLookupResult = PersistentPromptCachePrefixLookup
            .forPromptWithPartialTailBlockRecovery(
                modelContract: modelContract,
                promptTokens: promptTokens,
                blockCausalInputs: [],
                kvBlockExists: { (blockHash: Data) in
                    return blockKeys.contains(where: { (blockKey: PersistentPromptCacheBlockKey) in
                        return blockKey.blockHash() == blockHash;
                    });
                },
                recurrentSnapshotExists: { (blockHash: Data) in
                    return blockHash == firstBlockHash;
                });

        #expect(lookupResult.restoredTokenCount == modelContract.blockTokenCount);
        #expect(lookupResult.restoredPartialTailBlockKey == nil);
    }

    @Test
    func should_not_probe_a_tail_when_the_uncached_suffix_still_holds_complete_blocks() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let promptTokens: [UInt32] = PersistentPromptCacheFixture
            .promptTokensWithCompleteBlocksAndTrailingTokens(
                modelContract: modelContract, completeBlockCount: 2, trailingTokenCount: 0);
        let blockKeys: [PersistentPromptCacheBlockKey] = try PersistentPromptCacheFixture
            .blockKeysForPrompt(
                modelContract: modelContract, promptTokens: promptTokens, requestedBlockCount: 1);
        let firstBlockHash: Data = blockKeys[0].blockHash();

        let lookupResult: PersistentPromptCachePrefixLookupResult = PersistentPromptCachePrefixLookup
            .forPromptWithPartialTailBlockRecovery(
                modelContract: modelContract,
                promptTokens: promptTokens,
                blockCausalInputs: [],
                kvBlockExists: { (_: Data) in true },
                recurrentSnapshotExists: { (blockHash: Data) in
                    return blockHash == firstBlockHash;
                });

        #expect(lookupResult.restoredTokenCount == modelContract.blockTokenCount);
        #expect(lookupResult.restoredPartialTailBlockKey == nil);
    }

    @Test
    func should_keep_existing_lookup_constructors_tail_blind() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let promptTokens: [UInt32] = PersistentPromptCacheFixture
            .promptTokensWithCompleteBlocksAndTrailingTokens(
                modelContract: modelContract, completeBlockCount: 2, trailingTokenCount: 100);
        let blockKeys: [PersistentPromptCacheBlockKey] = try PersistentPromptCacheFixture
            .blockKeysForPrompt(
                modelContract: modelContract, promptTokens: promptTokens, requestedBlockCount: 2);
        let chainTipBlockHash: Data = blockKeys[1].blockHash();
        let tailBlockKey: PersistentPromptCacheBlockKey = try PersistentPromptCachePartialTailLookupTests
            .chainTipBlockKey(chainTip: blockKeys[1], promptTokens: promptTokens, tailTokenCount: 99);

        let lookupResult: PersistentPromptCachePrefixLookupResult = PersistentPromptCachePrefixLookup
            .forPrompt(
                modelContract: modelContract,
                promptTokens: promptTokens,
                kvBlockExists: { (blockHash: Data) in
                    return blockKeys.contains(where: { (blockKey: PersistentPromptCacheBlockKey) in
                        return blockKey.blockHash() == blockHash;
                    }) || blockHash == tailBlockKey.blockHash();
                },
                recurrentSnapshotExists: { (blockHash: Data) in
                    return blockHash == chainTipBlockHash || blockHash == tailBlockKey.blockHash();
                });

        #expect(lookupResult.restoredTokenCount == modelContract.blockTokenCount * 2);
        #expect(lookupResult.restoredPartialTailBlockKey == nil);
    }

    private static func chainTipBlockKey(
        chainTip: PersistentPromptCacheBlockKey,
        promptTokens: [UInt32],
        tailTokenCount: Int
    ) throws -> PersistentPromptCacheBlockKey {
        let tailStart: Int = promptTokens.count - 100;
        return try chainTip.forChildBlock(
            blockTokens: Array(promptTokens[tailStart..<(tailStart + tailTokenCount)]));
    }
}
