import Foundation;

import Testing;

import ModelServing;

@testable import ModelServing;

/// Hermetic journeys for the publication entry point's partial-tail rules:
/// a partial tail publishes durably and survives a rescan, publishing a
/// longer tail supersedes only strictly shorter siblings at the same chain
/// position, and full-block publication never triggers tail supersede.
final class PersistentPromptCachePartialTailTests {

    @Test
    func should_publish_a_partial_tail_block_and_survive_a_rescan() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
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
            .forRootBlock(
                modelContract: modelContract,
                blockTokens: Array(rootTokens[..<modelContract.blockTokenCount]));
        let tailTokenCount: Int = modelContract.blockTokenCount / 2;
        let tailBlockKey: PersistentPromptCacheBlockKey = try rootBlockKey.forChildBlock(
            blockTokens: PersistentPromptCacheFixture.syntheticTailTokens(
                tokenCount: tailTokenCount, tokenSeed: 90_000));
        let staging: PersistentPromptCacheStateFileStaging = PersistentPromptCacheFixture
            .SyntheticStateFileStaging();

        #expect(try diskStore.publishBlock(
            staging: staging, blockKey: rootBlockKey, parentBlockKey: nil)
            == .published);
        let publicationOutcome: PersistentPromptCachePublicationOutcome = try diskStore
            .publishBlock(staging: staging, blockKey: tailBlockKey, parentBlockKey: rootBlockKey);

        #expect(publicationOutcome == .published);
        #expect(diskStore.sequenceStateBlockCount() == 2);
        let tailBlockDirectory: URL = diskStore.blocksDirectory
            .appendingPathComponent(
                PersistentPromptCacheStoreFile.hexEncode(tailBlockKey.blockHash()),
                isDirectory: true);
        #expect(FileManager.default.fileExists(
            atPath: tailBlockDirectory.appendingPathComponent(
                PersistentPromptCacheStoreFile.SEQUENCE_STATE_FILE_NAME).path));
        #expect(FileManager.default.fileExists(
            atPath: tailBlockDirectory.appendingPathComponent(
                PersistentPromptCacheStoreFile.BOUNDARY_STATE_FILE_NAME).path));

        let rescannedDiskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(globalRoot: globalRoot, modelContract: modelContract);
        #expect(rescannedDiskStore.sequenceStateBlockCount() == 2,
            "the rescan must recover both committed blocks including the tail");
    }

    @Test
    func should_supersede_only_strictly_shorter_tail_siblings() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let diskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(modelContract: modelContract);
        let rootTokens: [UInt32] = PersistentPromptCacheFixture
            .promptTokensWithCompleteBlocksAndTrailingTokens(
                modelContract: modelContract, completeBlockCount: 1, trailingTokenCount: 0);
        let rootBlockKey: PersistentPromptCacheBlockKey = try PersistentPromptCacheBlockKey
            .forRootBlock(
                modelContract: modelContract,
                blockTokens: Array(rootTokens[..<modelContract.blockTokenCount]));
        let fullChildBlockKey: PersistentPromptCacheBlockKey = try rootBlockKey.forChildBlock(
            blockTokens: PersistentPromptCacheFixture.syntheticTailTokens(
                tokenCount: modelContract.blockTokenCount, tokenSeed: 1));
        let staging: PersistentPromptCacheStateFileStaging = PersistentPromptCacheFixture
            .SyntheticStateFileStaging();
        #expect(try diskStore.publishBlock(
            staging: staging, blockKey: rootBlockKey, parentBlockKey: nil) == .published);
        #expect(try diskStore.publishBlock(
            staging: staging, blockKey: fullChildBlockKey, parentBlockKey: rootBlockKey)
            == .published);

        let shorterTailTokenCount: Int = modelContract.blockTokenCount / 4;
        let shorterTailBlockKey: PersistentPromptCacheBlockKey = try rootBlockKey.forChildBlock(
            blockTokens: PersistentPromptCacheFixture.syntheticTailTokens(
                tokenCount: shorterTailTokenCount, tokenSeed: 90_000));
        #expect(try diskStore.publishBlock(
            staging: staging, blockKey: shorterTailBlockKey, parentBlockKey: rootBlockKey)
            == .published);

        // A longer tail at the same chain position supersedes the shorter
        // one but must keep the equal-length sibling and the full block.
        let longerTailTokenCount: Int = modelContract.blockTokenCount / 2;
        let longerTailBlockKey: PersistentPromptCacheBlockKey = try rootBlockKey.forChildBlock(
            blockTokens: PersistentPromptCacheFixture.syntheticTailTokens(
                tokenCount: longerTailTokenCount, tokenSeed: 90_000));
        let equalLengthTailBlockKey: PersistentPromptCacheBlockKey = try rootBlockKey
            .forChildBlock(
                blockTokens: PersistentPromptCacheFixture.syntheticTailTokens(
                    tokenCount: longerTailTokenCount, tokenSeed: 80_000));
        #expect(try diskStore.publishBlock(
            staging: staging, blockKey: equalLengthTailBlockKey, parentBlockKey: rootBlockKey)
            == .published);
        #expect(try diskStore.publishBlock(
            staging: staging, blockKey: longerTailBlockKey, parentBlockKey: rootBlockKey)
            == .published);

        #expect(FileManager.default.fileExists(
            atPath: diskStore.blocksDirectory.appendingPathComponent(
                PersistentPromptCacheStoreFile.hexEncode(shorterTailBlockKey.blockHash()))
                .path) == false,
            "the strictly shorter tail sibling should be removed");
        #expect(FileManager.default.fileExists(
            atPath: diskStore.blocksDirectory.appendingPathComponent(
                PersistentPromptCacheStoreFile.hexEncode(equalLengthTailBlockKey.blockHash()))
                .path),
            "the equal-length tail sibling should survive for divergent conversations");
        #expect(FileManager.default.fileExists(
            atPath: diskStore.blocksDirectory.appendingPathComponent(
                PersistentPromptCacheStoreFile.hexEncode(longerTailBlockKey.blockHash()))
                .path),
            "the longer tail should survive as its conversation's growth path");
        #expect(FileManager.default.fileExists(
            atPath: diskStore.blocksDirectory.appendingPathComponent(
                PersistentPromptCacheStoreFile.hexEncode(fullChildBlockKey.blockHash()))
                .path),
            "full blocks must never be superseded by tails");
        #expect(diskStore.sequenceStateBlockCount() == 4);
    }

    @Test
    func should_keep_full_block_publication_free_of_tail_supersede() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let diskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(modelContract: modelContract);
        let rootTokens: [UInt32] = PersistentPromptCacheFixture
            .promptTokensWithCompleteBlocksAndTrailingTokens(
                modelContract: modelContract, completeBlockCount: 1, trailingTokenCount: 0);
        let rootBlockKey: PersistentPromptCacheBlockKey = try PersistentPromptCacheBlockKey
            .forRootBlock(
                modelContract: modelContract,
                blockTokens: Array(rootTokens[..<modelContract.blockTokenCount]));
        let fullChildBlockKey: PersistentPromptCacheBlockKey = try rootBlockKey.forChildBlock(
            blockTokens: PersistentPromptCacheFixture.syntheticTailTokens(
                tokenCount: modelContract.blockTokenCount, tokenSeed: 1));
        let staging: PersistentPromptCacheStateFileStaging = PersistentPromptCacheFixture
            .SyntheticStateFileStaging();

        #expect(try diskStore.publishBlock(
            staging: staging, blockKey: rootBlockKey, parentBlockKey: nil) == .published);
        #expect(try diskStore.publishBlock(
            staging: staging, blockKey: fullChildBlockKey, parentBlockKey: rootBlockKey)
            == .published);

        #expect(diskStore.sequenceStateBlockCount() == 2);
        #expect(FileManager.default.fileExists(
            atPath: diskStore.blocksDirectory.appendingPathComponent(
                PersistentPromptCacheStoreFile.hexEncode(rootBlockKey.blockHash()))
                .path));
        #expect(FileManager.default.fileExists(
            atPath: diskStore.blocksDirectory.appendingPathComponent(
                PersistentPromptCacheStoreFile.hexEncode(fullChildBlockKey.blockHash()))
                .path));
    }
}
