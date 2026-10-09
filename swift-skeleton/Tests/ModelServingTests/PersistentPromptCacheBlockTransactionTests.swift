import Foundation;

import Testing;

import ModelServing;

@testable import ModelServing;

/// Hermetic journeys for crash-safe block publication: a complete block
/// publishes atomically under its content hash with the staging directory
/// gone and the index tracking real bytes, a duplicate publication fails
/// closed on topology mismatch, and a parent boundary is reclaimed only
/// after the child is durable and only when quota pressure demands it.
final class PersistentPromptCacheBlockTransactionTests {

    @Test
    func should_publish_a_block_atomically_and_reject_a_duplicate() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let diskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(modelContract: modelContract);
        let promptTokens: [UInt32] = PersistentPromptCacheFixture
            .promptTokensWithCompleteBlocksAndTrailingTokens(
                modelContract: modelContract, completeBlockCount: 1, trailingTokenCount: 0);
        let blockKey: PersistentPromptCacheBlockKey = try PersistentPromptCacheBlockKey
            .forRootBlock(
                modelContract: modelContract,
                blockTokens: Array(promptTokens[..<modelContract.blockTokenCount]));
        let staging: PersistentPromptCacheStateFileStaging = PersistentPromptCacheFixture
            .SyntheticStateFileStaging();

        try diskStore.publishNewBlockTransaction(
            staging: staging, blockKey: blockKey, parentBlockKey: nil);

        let blockDirectoryName: String = PersistentPromptCacheStoreFile.hexEncode(
            blockKey.blockHash());
        #expect(FileManager.default.fileExists(
            atPath: diskStore.blocksDirectory
                .appendingPathComponent(blockDirectoryName, isDirectory: true).path),
            "the block must publish under its content hash");
        #expect(diskStore.sequenceStateBlockCount() == 1);
        #expect(diskStore.boundaryStateSnapshotCount() == 1);
        #expect(diskStore.totalSizeBytes() > 0);

        #expect(throws: PersistentPromptCacheDiskStoreError.existingBlockTopologyMismatch(
            blockHash: blockKey.blockHash())) {
            try diskStore.publishNewBlockTransaction(
                staging: staging, blockKey: blockKey, parentBlockKey: nil);
        };
    }

    @Test
    func should_reclaim_a_redundant_parent_boundary_under_quota_pressure() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        // Measure one committed block first, then size the quota so three
        // chained blocks fit only after the non-checkpoint parent boundary
        // is reclaimed at the grandchild's publication.
        let measurementStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(modelContract: modelContract);
        let measurementTokens: [UInt32] = PersistentPromptCacheFixture
            .promptTokensWithCompleteBlocksAndTrailingTokens(
                modelContract: modelContract, completeBlockCount: 1, trailingTokenCount: 0);
        let measurementBlockKey: PersistentPromptCacheBlockKey = try PersistentPromptCacheBlockKey
            .forRootBlock(
                modelContract: modelContract,
                blockTokens: Array(measurementTokens[..<modelContract.blockTokenCount]));
        try measurementStore.publishNewBlockTransaction(
            staging: PersistentPromptCacheFixture.SyntheticStateFileStaging(),
            blockKey: measurementBlockKey, parentBlockKey: nil);
        let singleBlockSizeBytes: UInt64 = measurementStore.totalSizeBytes();
        let boundaryFileSizeBytes: UInt64 = try modelContract
            .boundaryStateFileBytesForBlockTokenCount(
                blockTokenCount: modelContract.blockTokenCount);
        let quotaBytes: UInt64 = singleBlockSizeBytes &* 3 &- boundaryFileSizeBytes &+ 1024;

        let diskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(
                modelContract: modelContract, globalPromptCacheMaximumSizeBytes: quotaBytes);
        let promptTokens: [UInt32] = PersistentPromptCacheFixture
            .promptTokensWithCompleteBlocksAndTrailingTokens(
                modelContract: modelContract, completeBlockCount: 3, trailingTokenCount: 0);
        let blockKeys: [PersistentPromptCacheBlockKey] = try PersistentPromptCacheFixture
            .blockKeysForPrompt(
                modelContract: modelContract, promptTokens: promptTokens, requestedBlockCount: 3);
        let staging: PersistentPromptCacheStateFileStaging = PersistentPromptCacheFixture
            .SyntheticStateFileStaging();
        try diskStore.publishNewBlockTransaction(
            staging: staging, blockKey: blockKeys[0], parentBlockKey: nil);
        try diskStore.publishNewBlockTransaction(
            staging: staging, blockKey: blockKeys[1], parentBlockKey: blockKeys[0]);
        try diskStore.publishNewBlockTransaction(
            staging: staging, blockKey: blockKeys[2], parentBlockKey: blockKeys[1]);

        // The child sits at non-checkpoint index 1, so the grandchild's
        // publication reclaims its boundary under quota pressure while the
        // sequence state stays durable for recompute from the root.
        let childBlockDirectory: URL = diskStore.blocksDirectory
            .appendingPathComponent(
                PersistentPromptCacheStoreFile.hexEncode(blockKeys[1].blockHash()),
                isDirectory: true);
        #expect(FileManager.default.fileExists(
            atPath: childBlockDirectory.appendingPathComponent(
                PersistentPromptCacheStoreFile.SEQUENCE_STATE_FILE_NAME).path),
            "the non-checkpoint parent sequence state must stay durable");
        #expect(FileManager.default.fileExists(
            atPath: childBlockDirectory.appendingPathComponent(
                PersistentPromptCacheStoreFile.BOUNDARY_STATE_FILE_NAME).path) == false,
            "quota pressure must reclaim the redundant parent boundary");
        #expect(diskStore.hasKvBlock(blockHash: blockKeys[0].blockHash()),
            "the root checkpoint boundary and sequence state must stay durable");
        #expect(diskStore.hasRecurrentSnapshot(blockHash: blockKeys[0].blockHash()));
        #expect(diskStore.hasKvBlock(blockHash: blockKeys[2].blockHash()));
        #expect(diskStore.hasRecurrentSnapshot(blockHash: blockKeys[2].blockHash()));
        #expect(diskStore.totalSizeBytes() <= quotaBytes,
            "committed cache bytes must satisfy the configured quota");
    }

}
