import Foundation;

import Testing;

import ModelServing;

@testable import ModelServing;

/// Recovery journeys for ancestry and boundary retention: reopening a chain
/// with one compacted non-checkpoint ancestor keeps every sequence state,
/// recapturing restores only the missing boundary, interrupted compaction
/// completes before startup eviction, unprotected content absorbs pressure
/// first, orphaned descendants are pruned exactly once, and corrupt state
/// never earns idempotent acknowledgement.
final class PersistentPromptCacheBlockTopologyTests {

    @Test
    func should_reopen_a_chain_with_one_compacted_non_checkpoint_ancestor() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let (rootBlockKey, childBlockKey, grandchildBlockKey) = try Self.chainKeys(
            modelContract: modelContract);
        let globalRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("topology-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalRoot); }
        let diskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(
                globalRoot: globalRoot,
                activeModelPromptCacheDirectory: Self.globalPromptCacheActiveDirectory(globalRoot),
                modelContract: modelContract);
        try Self.saveThreeBlockChain(
            diskStore: diskStore, modelContract: modelContract,
            rootBlockKey: rootBlockKey, childBlockKey: childBlockKey,
            grandchildBlockKey: grandchildBlockKey);
        let compactedChildBoundaryFilePath: URL = Self.blockDirectoryPath(
            globalRoot: globalRoot, blockKey: childBlockKey)
            .appendingPathComponent(PersistentPromptCacheStoreFile.BOUNDARY_STATE_FILE_NAME);
        try FileManager.default.removeItem(at: compactedChildBoundaryFilePath);

        let reopenedDiskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(
                globalRoot: globalRoot,
                activeModelPromptCacheDirectory: Self.globalPromptCacheActiveDirectory(globalRoot),
                modelContract: modelContract);

        #expect(reopenedDiskStore.sequenceStateBlockCount() == 3);
        #expect(reopenedDiskStore.boundaryStateSnapshotCount() == 2);
        #expect(reopenedDiskStore.hasKvBlock(blockHash: childBlockKey.blockHash()));
        #expect(reopenedDiskStore.hasRecurrentSnapshot(
            blockHash: childBlockKey.blockHash()) == false);
        #expect(reopenedDiskStore.hasKvBlock(blockHash: grandchildBlockKey.blockHash()));
        #expect(reopenedDiskStore.hasRecurrentSnapshot(blockHash: grandchildBlockKey.blockHash()));
    }

    @Test
    func should_recapture_only_a_compacted_boundary_without_rewriting_sequence_state() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let (rootBlockKey, childBlockKey, grandchildBlockKey) = try Self.chainKeys(
            modelContract: modelContract);
        let globalRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("topology-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalRoot); }
        let diskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(
                globalRoot: globalRoot,
                activeModelPromptCacheDirectory: Self.globalPromptCacheActiveDirectory(globalRoot),
                modelContract: modelContract);
        try Self.saveThreeBlockChain(
            diskStore: diskStore, modelContract: modelContract,
            rootBlockKey: rootBlockKey, childBlockKey: childBlockKey,
            grandchildBlockKey: grandchildBlockKey);
        let childBlockDirectory: URL = Self.blockDirectoryPath(
            globalRoot: globalRoot, blockKey: childBlockKey);
        let childSequenceFilePath: URL = childBlockDirectory.appendingPathComponent(
            PersistentPromptCacheStoreFile.SEQUENCE_STATE_FILE_NAME);
        let childBoundaryFilePath: URL = childBlockDirectory.appendingPathComponent(
            PersistentPromptCacheStoreFile.BOUNDARY_STATE_FILE_NAME);
        try FileManager.default.removeItem(at: childBoundaryFilePath);
        try FileManager.default.setAttributes(
            [.modificationDate: Date(timeIntervalSince1970: 10)],
            ofItemAtPath: childSequenceFilePath.path);

        let reopenedDiskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(
                globalRoot: globalRoot,
                activeModelPromptCacheDirectory: Self.globalPromptCacheActiveDirectory(globalRoot),
                modelContract: modelContract);
        #expect(try reopenedDiskStore.publishBlock(
            staging: PersistentPromptCacheFixture.SyntheticStateFileStaging(),
            blockKey: childBlockKey, parentBlockKey: rootBlockKey) == .published,
            "recapturing the child should restore its missing boundary");

        #expect(FileManager.default.fileExists(atPath: childBoundaryFilePath.path));
        let sequenceAttributes: [FileAttributeKey: Any] = try FileManager.default
            .attributesOfItem(atPath: childSequenceFilePath.path);
        let sequenceModifiedAt: Date = try #require(
            sequenceAttributes[.modificationDate] as? Date);
        #expect(sequenceModifiedAt.timeIntervalSince1970 == 10,
            "recapturing a compacted boundary must not rewrite sequence state");
    }

    @Test
    func should_complete_interrupted_parent_boundary_compaction_before_startup_eviction() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let (rootBlockKey, childBlockKey, grandchildBlockKey) = try Self.chainKeys(
            modelContract: modelContract);
        let globalRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("topology-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalRoot); }
        let diskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(
                globalRoot: globalRoot,
                activeModelPromptCacheDirectory: Self.globalPromptCacheActiveDirectory(globalRoot),
                modelContract: modelContract);
        try Self.saveThreeBlockChain(
            diskStore: diskStore, modelContract: modelContract,
            rootBlockKey: rootBlockKey, childBlockKey: childBlockKey,
            grandchildBlockKey: grandchildBlockKey);
        let interruptedTransactionSizeBytes: UInt64 = diskStore.totalSizeBytes();
        let recoveredBoundaryFileByteCount: UInt64 = try modelContract
            .boundaryStateFileBytesForBlockTokenCount(
                blockTokenCount: modelContract.blockTokenCount);
        let postCompactionQuotaBytes: UInt64 = interruptedTransactionSizeBytes
            &- recoveredBoundaryFileByteCount;

        // The disk image represents a crash after child commit but before
        // redundant parent-boundary deletion; the tightened quota forces the
        // recovery order on reopen.
        let reopenedDiskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(
                globalRoot: globalRoot,
                activeModelPromptCacheDirectory: Self.globalPromptCacheActiveDirectory(globalRoot),
                modelContract: modelContract,
                globalPromptCacheMaximumSizeBytes: postCompactionQuotaBytes);

        #expect(reopenedDiskStore.sequenceStateBlockCount() == 3);
        #expect(reopenedDiskStore.boundaryStateSnapshotCount() == 2);
        #expect(reopenedDiskStore.hasRecurrentSnapshot(
            blockHash: childBlockKey.blockHash()) == false);
        #expect(reopenedDiskStore.hasKvBlock(blockHash: grandchildBlockKey.blockHash()));
        let interruptedTransactionRecovery: PersistentPromptCacheStartupCleanupCategory = try
            #require(reopenedDiskStore.startupCleanupEvidence(),
                "startup compaction should retain cleanup evidence")
            .interruptedTransactionRecovery;
        #expect(interruptedTransactionRecovery.artifactCount == 1);
        #expect(interruptedTransactionRecovery.blockCount == 0);
        #expect(interruptedTransactionRecovery.byteCount == recoveredBoundaryFileByteCount);
    }

    @Test
    func should_evict_unprotected_content_before_compacting_the_active_startup_chain() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let (rootBlockKey, childBlockKey, grandchildBlockKey) = try Self.chainKeys(
            modelContract: modelContract);
        let globalRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("topology-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalRoot); }
        let diskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(
                globalRoot: globalRoot,
                activeModelPromptCacheDirectory: Self.globalPromptCacheActiveDirectory(globalRoot),
                modelContract: modelContract);
        try Self.saveThreeBlockChain(
            diskStore: diskStore, modelContract: modelContract,
            rootBlockKey: rootBlockKey, childBlockKey: childBlockKey,
            grandchildBlockKey: grandchildBlockKey);
        let activeChainSizeBytes: UInt64 = diskStore.totalSizeBytes();
        let unprotectedFilePath: URL = globalRoot.appendingPathComponent("unprotected.bin");
        try Data(count: 4_096).write(to: unprotectedFilePath);

        // Unrelated bytes can satisfy pressure without reducing useful
        // restart points, so they must be selected before any valid
        // active-chain boundary is compacted.
        let reopenedDiskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(
                globalRoot: globalRoot,
                activeModelPromptCacheDirectory: Self.globalPromptCacheActiveDirectory(globalRoot),
                modelContract: modelContract,
                globalPromptCacheMaximumSizeBytes: activeChainSizeBytes);

        #expect(FileManager.default.fileExists(atPath: unprotectedFilePath.path) == false);
        #expect(reopenedDiskStore.sequenceStateBlockCount() == 3);
        #expect(reopenedDiskStore.boundaryStateSnapshotCount() == 3);
    }

    @Test
    func should_remove_descendants_when_their_sequence_ancestor_is_missing_on_reopen() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let (rootBlockKey, childBlockKey, _) = try Self.chainKeys(modelContract: modelContract);
        let globalRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("topology-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalRoot); }
        let diskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(
                globalRoot: globalRoot,
                activeModelPromptCacheDirectory: Self.globalPromptCacheActiveDirectory(globalRoot),
                modelContract: modelContract);
        let staging: PersistentPromptCacheStateFileStaging = PersistentPromptCacheFixture
            .SyntheticStateFileStaging();
        #expect(try diskStore.publishBlock(
            staging: staging, blockKey: rootBlockKey, parentBlockKey: nil) == .published);
        #expect(try diskStore.publishBlock(
            staging: staging, blockKey: childBlockKey, parentBlockKey: rootBlockKey)
            == .published);
        try FileManager.default.removeItem(at: Self.blockDirectoryPath(
            globalRoot: globalRoot, blockKey: rootBlockKey));

        // A child's content hash does not make it independently restorable:
        // removing root must prune the complete orphan chain.
        let reopenedDiskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(
                globalRoot: globalRoot,
                activeModelPromptCacheDirectory: Self.globalPromptCacheActiveDirectory(globalRoot),
                modelContract: modelContract);

        #expect(reopenedDiskStore.sequenceStateBlockCount() == 0);
        #expect(reopenedDiskStore.boundaryStateSnapshotCount() == 0);
        #expect(FileManager.default.fileExists(atPath: Self.blockDirectoryPath(
            globalRoot: globalRoot, blockKey: childBlockKey).path) == false);
        #expect(try #require(reopenedDiskStore.startupCleanupEvidence(),
            "orphan pruning should retain cleanup evidence")
            .corruptCurrentFormat.blockCount == 1,
            "the orphan child should be counted exactly once");
    }

    @Test
    func should_not_acknowledge_an_existing_block_after_its_state_is_corrupted() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let globalRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("topology-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalRoot); }
        let diskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(
                globalRoot: globalRoot,
                activeModelPromptCacheDirectory: Self.globalPromptCacheActiveDirectory(globalRoot),
                modelContract: modelContract);
        let rootTokens: [UInt32] = PersistentPromptCacheFixture
            .promptTokensWithCompleteBlocksAndTrailingTokens(
                modelContract: modelContract, completeBlockCount: 1, trailingTokenCount: 0);
        let blockKey: PersistentPromptCacheBlockKey = try PersistentPromptCacheBlockKey
            .forRootBlock(
                modelContract: modelContract,
                blockTokens: Array(rootTokens[..<modelContract.blockTokenCount]));
        let staging: PersistentPromptCacheStateFileStaging = PersistentPromptCacheFixture
            .SyntheticStateFileStaging();
        #expect(try diskStore.publishBlock(
            staging: staging, blockKey: blockKey, parentBlockKey: nil) == .published);
        try Data("corrupted state".utf8).write(
            to: Self.blockDirectoryPath(globalRoot: globalRoot, blockKey: blockKey)
                .appendingPathComponent(PersistentPromptCacheStoreFile.SEQUENCE_STATE_FILE_NAME));

        // Corrupt existing state must fail full revalidation instead of
        // receiving an idempotent acknowledgement.
        do {
            _ = try diskStore.publishBlock(
                staging: staging, blockKey: blockKey, parentBlockKey: nil);
            Issue.record("corrupt existing state must not receive idempotent acknowledgement");
        } catch let publishError as PersistentPromptCacheDiskStoreError {
            guard case .validateBlock = publishError else {
                Issue.record("expected validateBlock, got \(publishError)");
                return;
            }
        }
    }

    /// The Rust journeys open the store with the temporary root itself as
    /// the active model directory, so block paths are `<root>/blocks/<hash>`.
    private static func globalPromptCacheActiveDirectory(_ globalRoot: URL) -> URL {
        return globalRoot;
    }

    private static func chainKeys(
        modelContract: PersistentPromptCacheModelContract
    ) throws -> (PersistentPromptCacheBlockKey, PersistentPromptCacheBlockKey,
        PersistentPromptCacheBlockKey) {
        let rootBlockKey: PersistentPromptCacheBlockKey = try PersistentPromptCacheBlockKey
            .forRootBlock(
                modelContract: modelContract,
                blockTokens: PersistentPromptCacheFixture.syntheticTailTokens(
                    tokenCount: modelContract.blockTokenCount, tokenSeed: 0));
        let childBlockKey: PersistentPromptCacheBlockKey = try rootBlockKey.forChildBlock(
            blockTokens: PersistentPromptCacheFixture.syntheticTailTokens(
                tokenCount: modelContract.blockTokenCount, tokenSeed: 10_000));
        let grandchildBlockKey: PersistentPromptCacheBlockKey = try childBlockKey.forChildBlock(
            blockTokens: PersistentPromptCacheFixture.syntheticTailTokens(
                tokenCount: modelContract.blockTokenCount, tokenSeed: 20_000));
        return (rootBlockKey, childBlockKey, grandchildBlockKey);
    }

    private static func saveThreeBlockChain(
        diskStore: PersistentPromptCacheDiskStore,
        modelContract: PersistentPromptCacheModelContract,
        rootBlockKey: PersistentPromptCacheBlockKey,
        childBlockKey: PersistentPromptCacheBlockKey,
        grandchildBlockKey: PersistentPromptCacheBlockKey
    ) throws {
        let staging: PersistentPromptCacheStateFileStaging = PersistentPromptCacheFixture
            .SyntheticStateFileStaging();
        #expect(try diskStore.publishBlock(
            staging: staging, blockKey: rootBlockKey, parentBlockKey: nil) == .published);
        #expect(try diskStore.publishBlock(
            staging: staging, blockKey: childBlockKey, parentBlockKey: rootBlockKey)
            == .published);
        #expect(try diskStore.publishBlock(
            staging: staging, blockKey: grandchildBlockKey, parentBlockKey: childBlockKey)
            == .published);
    }

    private static func blockDirectoryPath(
        globalRoot: URL, blockKey: PersistentPromptCacheBlockKey
    ) -> URL {
        return globalRoot.appendingPathComponent("blocks", isDirectory: true)
            .appendingPathComponent(
                PersistentPromptCacheStoreFile.hexEncode(blockKey.blockHash()),
                isDirectory: true);
    }
}
