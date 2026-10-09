import Foundation;

import Testing;

import ModelServing;

@testable import ModelServing;

/// Hermetic journeys for the store shell's recovery protocol: opening a
/// fresh tree creates the trusted directories, a scan validates committed
/// state before quota runs, clear resets the active index and directories,
/// and reopen re-runs the full recovery protocol.
final class PersistentPromptCacheDiskStoreTests {

    @Test
    func should_open_clear_and_reopen_the_prompt_cache_store() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let globalRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("store-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalRoot); }
        let diskStoreConfig: PersistentPromptCacheDiskStoreConfig = PersistentPromptCacheDiskStoreConfig(
            activeModelPromptCacheDirectory: globalRoot
                .appendingPathComponent("org/model-a/rev-1", isDirectory: true),
            globalPromptCacheRootDirectory: globalRoot,
            globalPromptCacheMaximumSizeBytes: 50_000_000_000);

        let diskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheDiskStore.open(
            diskStoreConfig: diskStoreConfig, modelContract: modelContract);
        #expect(diskStore.sequenceStateBlockCount() == 0);
        #expect(diskStore.boundaryStateSnapshotCount() == 0);
        #expect(FileManager.default.fileExists(atPath: diskStore.blocksDirectory.path));
        #expect(diskStore.totalSizeBytes() == 0);

        let clearOutcome: PersistentPromptCacheClearOutcome = try diskStore.clearPromptCache(
            modelId: "org/model-a");
        #expect(clearOutcome.blocksRemoved == 0);

        let reopenedStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheDiskStore.open(
            diskStoreConfig: diskStoreConfig, modelContract: modelContract);
        #expect(reopenedStore.sequenceStateBlockCount() == 0);
        #expect(reopenedStore.startupCleanupEvidence() == nil,
            "a clean tree must produce no startup cleanup evidence");
    }

    /// The Rust disk_store journey `should_publish_a_block_directly_and_
    /// report_idempotent_republication`: a block publishes durably under
    /// its content hash, and republishing the exact block reports
    /// AlreadyPublished after full revalidation instead of rewriting.
    @Test
    func should_publish_a_block_directly_and_report_idempotent_republication() throws {
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

        let publicationOutcome: PersistentPromptCachePublicationOutcome = try diskStore
            .publishBlock(staging: staging, blockKey: blockKey, parentBlockKey: nil);
        #expect(publicationOutcome == .published);

        let duplicatePublicationOutcome: PersistentPromptCachePublicationOutcome = try diskStore
            .publishBlock(staging: staging, blockKey: blockKey, parentBlockKey: nil);
        #expect(duplicatePublicationOutcome == .alreadyPublished);

        #expect(diskStore.sequenceStateBlockCount() == 1);
        #expect(diskStore.boundaryStateSnapshotCount() == 1);
    }
}
