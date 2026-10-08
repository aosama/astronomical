import Foundation;

import Testing;

import ModelServing;

@testable import ModelServing;

/// Hermetic journeys for publication rollback and retention safety: a
/// fitting child must never trigger eager parent cleanup, even when the
/// parent's boundary file has been replaced by a filesystem-level deletion
/// sentinel.
final class PersistentPromptCacheDiskStoreRollbackTests {

    @Test
    func should_not_touch_a_parent_boundary_when_a_child_fits_without_quota_pressure()
        throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let diskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(modelContract: modelContract);
        let promptTokens: [UInt32] = PersistentPromptCacheFixture
            .promptTokensWithCompleteBlocksAndTrailingTokens(
                modelContract: modelContract, completeBlockCount: 3, trailingTokenCount: 0);
        let blockKeys: [PersistentPromptCacheBlockKey] = try PersistentPromptCacheFixture
            .blockKeysForPrompt(
                modelContract: modelContract, promptTokens: promptTokens, requestedBlockCount: 3);
        let staging: PersistentPromptCacheStateFileStaging = PersistentPromptCacheFixture
            .SyntheticStateFileStaging();
        #expect(try diskStore.publishBlock(
            staging: staging, blockKey: blockKeys[0], parentBlockKey: nil) == .published);
        #expect(try diskStore.publishBlock(
            staging: staging, blockKey: blockKeys[1], parentBlockKey: blockKeys[0])
            == .published);

        // A deletion sentinel directory proves the grandchild publication
        // never writes through the parent's boundary path: an accidental
        // eager parent cleanup would fail here instead of quietly turning a
        // reusable branch point into a missing cache snapshot.
        let childBoundaryFileUrl: URL = diskStore.blocksDirectory
            .appendingPathComponent(
                PersistentPromptCacheStoreFile.hexEncode(blockKeys[1].blockHash()),
                isDirectory: true)
            .appendingPathComponent(PersistentPromptCacheStoreFile.BOUNDARY_STATE_FILE_NAME);
        try FileManager.default.removeItem(at: childBoundaryFileUrl);
        try FileManager.default.createDirectory(
            at: childBoundaryFileUrl, withIntermediateDirectories: false);

        #expect(
            throws: Never.self,
            "a fitting child must retain its parent without quota pressure") {
            _ = try diskStore.publishBlock(
                staging: staging, blockKey: blockKeys[2], parentBlockKey: blockKeys[1]);
        };
        #expect(diskStore.hasKvBlock(blockHash: blockKeys[2].blockHash()));
        #expect(diskStore.hasRecurrentSnapshot(blockHash: blockKeys[2].blockHash()));
        let grandchildBlockDirectory: URL = diskStore.blocksDirectory
            .appendingPathComponent(
                PersistentPromptCacheStoreFile.hexEncode(blockKeys[2].blockHash()),
                isDirectory: true);
        #expect(FileManager.default.fileExists(
            atPath: grandchildBlockDirectory.appendingPathComponent(
                PersistentPromptCacheStoreFile.SEQUENCE_STATE_FILE_NAME).path));
        #expect(FileManager.default.fileExists(
            atPath: grandchildBlockDirectory.appendingPathComponent(
                PersistentPromptCacheStoreFile.BOUNDARY_STATE_FILE_NAME).path));
    }
}
