import Foundation;

import Testing;

import ModelServing;

@testable import ModelServing;

/// Hermetic journeys for storage-layout topology: a sequence-only chain
/// reopens with every block intact, and a boundary-only chain reopens
/// without triggering invalid compaction, proving the store honors
/// contracts that carry only one state kind.
final class PersistentPromptCacheLayoutTopologyTests {

    @Test
    func should_reopen_every_sequence_only_block_in_the_complete_active_chain() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .syntheticSequenceOnlyContract();
        let globalRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("layout-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(
            at: globalRoot, withIntermediateDirectories: true);
        let diskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(
                globalRoot: globalRoot, modelContract: modelContract,
                globalPromptCacheMaximumSizeBytes: 1_000_000);
        let completeBlockCount: Int = modelContract.maximumContextTokenCount
            / modelContract.blockTokenCount;
        let blockKeys: [PersistentPromptCacheBlockKey] = try Self.publishCompleteChain(
            diskStore: diskStore, modelContract: modelContract,
            completeBlockCount: completeBlockCount);
        #expect(blockKeys.count == completeBlockCount);

        let reopenedDiskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(
                globalRoot: globalRoot, modelContract: modelContract,
                globalPromptCacheMaximumSizeBytes: 1_000_000);
        #expect(reopenedDiskStore.sequenceStateBlockCount() == completeBlockCount);
        #expect(reopenedDiskStore.boundaryStateSnapshotCount() == 0);
    }

    @Test
    func should_reopen_every_boundary_only_block_without_invalid_compaction() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .syntheticBoundaryOnlyContract();
        let globalRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("layout-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(
            at: globalRoot, withIntermediateDirectories: true);
        let diskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(
                globalRoot: globalRoot, modelContract: modelContract,
                globalPromptCacheMaximumSizeBytes: 10_000);
        let completeBlockCount: Int = modelContract.maximumContextTokenCount
            / modelContract.blockTokenCount;
        _ = try Self.publishCompleteChain(
            diskStore: diskStore, modelContract: modelContract,
            completeBlockCount: completeBlockCount);

        let reopenedDiskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(
                globalRoot: globalRoot, modelContract: modelContract,
                globalPromptCacheMaximumSizeBytes: 10_000);
        #expect(reopenedDiskStore.sequenceStateBlockCount() == 0);
        #expect(reopenedDiskStore.boundaryStateSnapshotCount() == completeBlockCount,
            "the boundary-only chain must reopen without invalid compaction");
    }

    /// Publishes one complete active chain of `completeBlockCount` blocks,
    /// mirroring the Rust journey chain publisher.
    private static func publishCompleteChain(
        diskStore: PersistentPromptCacheDiskStore,
        modelContract: PersistentPromptCacheModelContract,
        completeBlockCount: Int
    ) throws -> [PersistentPromptCacheBlockKey] {
        let promptTokens: [UInt32] = (0..<(completeBlockCount * modelContract.blockTokenCount))
            .map({ (tokenOffset: Int) -> UInt32 in
                return UInt32(tokenOffset);
            });
        var blockKeys: [PersistentPromptCacheBlockKey] = [];
        var parentBlockKey: PersistentPromptCacheBlockKey? = nil;
        let staging: PersistentPromptCacheStateFileStaging = PersistentPromptCacheFixture
            .SyntheticStateFileStaging();
        for blockIndex: Int in 0..<completeBlockCount {
            let blockStart: Int = blockIndex * modelContract.blockTokenCount;
            let blockKey: PersistentPromptCacheBlockKey;
            if let parentBlockKey: PersistentPromptCacheBlockKey = parentBlockKey {
                blockKey = try parentBlockKey.forChildBlock(
                    blockTokens: Array(promptTokens[blockStart..<(blockStart
                        + modelContract.blockTokenCount)]));
            } else {
                blockKey = try PersistentPromptCacheBlockKey.forRootBlock(
                    modelContract: modelContract,
                    blockTokens: Array(promptTokens[blockStart..<(blockStart
                        + modelContract.blockTokenCount)]));
            }
            #expect(try diskStore.publishBlock(
                staging: staging, blockKey: blockKey, parentBlockKey: parentBlockKey)
                == .published,
                "every complete active-chain block should publish");
            blockKeys.append(blockKey);
            parentBlockKey = blockKey;
        }
        return blockKeys;
    }
}
