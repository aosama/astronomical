import Foundation;

import Testing;

import ModelServing;

@testable import ModelServing;

/// Hermetic journey for quota pressure under an active chain: committing a
/// child that would exceed the configured quota evicts the unrelated block
/// instead of the protected parent chain, and the committed bytes stay
/// within the configured quota.
final class PersistentPromptCacheDiskStoreGlobalQuotaTests {

    @Test
    func should_protect_the_active_parent_chain_and_evict_unrelated_blocks_under_quota_pressure()
        throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();

        // Measure one committed block first, mirroring the Rust journey's
        // measurement-then-quota shape.
        let measurementStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(modelContract: modelContract);
        let rootTokens: [UInt32] = PersistentPromptCacheFixture
            .promptTokensWithCompleteBlocksAndTrailingTokens(
                modelContract: modelContract, completeBlockCount: 1, trailingTokenCount: 0);
        let parentBlockKey: PersistentPromptCacheBlockKey = try PersistentPromptCacheBlockKey
            .forRootBlock(
                modelContract: modelContract,
                blockTokens: Array(rootTokens[..<modelContract.blockTokenCount]));
        let childBlockKey: PersistentPromptCacheBlockKey = try parentBlockKey.forChildBlock(
            blockTokens: PersistentPromptCacheFixture.syntheticTailTokens(
                tokenCount: modelContract.blockTokenCount, tokenSeed: 10_000));
        let unrelatedBlockKey: PersistentPromptCacheBlockKey = try PersistentPromptCacheBlockKey
            .forRootBlock(
                modelContract: modelContract,
                blockTokens: PersistentPromptCacheFixture.syntheticTailTokens(
                    tokenCount: modelContract.blockTokenCount, tokenSeed: 99_000));
        let staging: PersistentPromptCacheStateFileStaging = PersistentPromptCacheFixture
            .SyntheticStateFileStaging();
        #expect(try measurementStore.publishBlock(
            staging: staging, blockKey: parentBlockKey, parentBlockKey: nil) == .published);
        let measuredSingleBlockSizeBytes: UInt64 = measurementStore.totalSizeBytes();
        let twoBlockQuotaBytes: UInt64 = measuredSingleBlockSizeBytes &* 2 &+ 1024;

        let globalRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("prompt-cache-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(
            at: globalRoot, withIntermediateDirectories: true);
        let diskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(
                globalRoot: globalRoot, modelContract: modelContract,
                globalPromptCacheMaximumSizeBytes: twoBlockQuotaBytes);

        #expect(try diskStore.publishBlock(
            staging: staging, blockKey: parentBlockKey, parentBlockKey: nil) == .published);
        #expect(try diskStore.publishBlock(
            staging: staging, blockKey: unrelatedBlockKey, parentBlockKey: nil) == .published,
            "the unrelated block should save while the two-block quota has room");
        #expect(try diskStore.publishBlock(
            staging: staging, blockKey: childBlockKey, parentBlockKey: parentBlockKey)
            == .published,
            "the child publication should evict the unrelated block, not the active chain");

        #expect(diskStore.hasKvBlock(blockHash: parentBlockKey.blockHash()),
            "the protected parent sequence state must remain restorable");
        #expect(diskStore.hasKvBlock(blockHash: childBlockKey.blockHash()),
            "the newly published child sequence state must remain restorable");
        #expect(diskStore.hasKvBlock(blockHash: unrelatedBlockKey.blockHash()) == false,
            "quota pressure should evict the unrelated block instead of the active chain");
        #expect(diskStore.totalSizeBytes() <= twoBlockQuotaBytes,
            "committed cache bytes must satisfy the configured two-block quota");
    }

    @Test
    func should_scope_protected_ancestry_to_its_model_namespace_when_hashes_match() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let measurementStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(modelContract: modelContract);
        let rootTokens: [UInt32] = PersistentPromptCacheFixture
            .promptTokensWithCompleteBlocksAndTrailingTokens(
                modelContract: modelContract, completeBlockCount: 1, trailingTokenCount: 0);
        let parentBlockKey: PersistentPromptCacheBlockKey = try PersistentPromptCacheBlockKey
            .forRootBlock(
                modelContract: modelContract,
                blockTokens: Array(rootTokens[..<modelContract.blockTokenCount]));
        let childBlockKey: PersistentPromptCacheBlockKey = try parentBlockKey.forChildBlock(
            blockTokens: PersistentPromptCacheFixture.syntheticTailTokens(
                tokenCount: modelContract.blockTokenCount, tokenSeed: 10_000));
        let staging: PersistentPromptCacheStateFileStaging = PersistentPromptCacheFixture
            .SyntheticStateFileStaging();
        #expect(try measurementStore.publishBlock(
            staging: staging, blockKey: parentBlockKey, parentBlockKey: nil) == .published);
        let measuredBlockSizeBytes: UInt64 = measurementStore.totalSizeBytes();
        let twoBlockQuotaBytes: UInt64 = measuredBlockSizeBytes &* 2 &+ 1_024;

        let globalPromptCacheRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("prompt-cache-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(
            at: globalPromptCacheRoot, withIntermediateDirectories: true);
        let activeCache: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(
                globalRoot: globalPromptCacheRoot,
                activeModelPromptCacheDirectory: globalPromptCacheRoot,
                modelContract: modelContract,
                globalPromptCacheMaximumSizeBytes: twoBlockQuotaBytes);
        #expect(try activeCache.publishBlock(
            staging: staging, blockKey: parentBlockKey, parentBlockKey: nil) == .published);
        let parentBlockDirectory: URL = globalPromptCacheRoot
            .appendingPathComponent("blocks", isDirectory: true)
            .appendingPathComponent(
                PersistentPromptCacheStoreFile.hexEncode(parentBlockKey.blockHash()),
                isDirectory: true);

        // Duplicate the same-hash manifest under a foreign model namespace
        // with enough payload to consume the remaining quota, and backdate
        // it so it becomes the oldest eviction candidate.
        let foreignSameHashDirectory: URL = globalPromptCacheRoot
            .appendingPathComponent("foreign-model/foreign-revision/blocks", isDirectory: true)
            .appendingPathComponent(
                PersistentPromptCacheStoreFile.hexEncode(parentBlockKey.blockHash()),
                isDirectory: true);
        try FileManager.default.createDirectory(
            at: foreignSameHashDirectory, withIntermediateDirectories: true);
        try FileManager.default.copyItem(
            at: parentBlockDirectory.appendingPathComponent("manifest.json"),
            to: foreignSameHashDirectory.appendingPathComponent("manifest.json"));
        try Data(count: 4_096).write(
            to: foreignSameHashDirectory.appendingPathComponent("foreign-payload.bin"));
        try FileManager.default.setAttributes(
            [.modificationDate: Date(timeIntervalSince1970: 0)],
            ofItemAtPath: foreignSameHashDirectory.path);

        #expect(try activeCache.publishBlock(
            staging: staging, blockKey: childBlockKey, parentBlockKey: parentBlockKey)
            == .published,
            Comment("the child should evict only the foreign same-hash namespace"));

        #expect(FileManager.default.fileExists(atPath: parentBlockDirectory.path),
            Comment("the active parent must stay protected even though the foreign namespace carries an identical hash"));
        #expect(FileManager.default.fileExists(
            atPath: globalPromptCacheRoot.appendingPathComponent("blocks", isDirectory: true)
                .appendingPathComponent(
                    PersistentPromptCacheStoreFile.hexEncode(childBlockKey.blockHash()))
                .path));
        #expect(FileManager.default.fileExists(
            atPath: foreignSameHashDirectory.path) == false,
            "hash equality across namespaces must not make foreign bytes unevictable");
    }
}
