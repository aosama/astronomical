import Foundation;

import Testing;

import ModelServing;

@testable import ModelServing;

/// Hermetic journeys for global-quota topology safety: cyclic foreign
/// manifests evict without unbounded walks, eviction subtrees never join
/// across storage-contract fingerprints, and stale transactions evict
/// before older committed content because the transaction class, not the
/// timestamp, controls the first eviction decision.
final class PersistentPromptCacheGlobalQuotaTopologyTests {

    @Test
    func should_bound_global_quota_topology_walk_for_cyclic_foreign_manifests() throws {
        let globalPromptCacheRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("quota-topology-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalPromptCacheRoot); }
        let foreignBlocksDirectory: URL = globalPromptCacheRoot
            .appendingPathComponent("foreign-model/foreign-revision/blocks", isDirectory: true);
        let firstBlockHash: String = String(repeating: "a", count: 64);
        let secondBlockHash: String = String(repeating: "b", count: 64);
        try Self.writeForeignBlock(
            foreignBlocksDirectory: foreignBlocksDirectory, blockHash: firstBlockHash,
            parentBlockHash: secondBlockHash, storageContractFingerprint: "foreign-contract",
            payloadByteCount: 128, modifiedAt: Date(timeIntervalSince1970: 0));
        try Self.writeForeignBlock(
            foreignBlocksDirectory: foreignBlocksDirectory, blockHash: secondBlockHash,
            parentBlockHash: firstBlockHash, storageContractFingerprint: "foreign-contract",
            payloadByteCount: 128, modifiedAt: Date(timeIntervalSince1970: 0));

        // Foreign files bypass active-model startup validation, so the
        // global scanner applies its own cycle guard before quota; opening
        // under a one-byte quota must evict both without recursive overflow.
        _ = try PersistentPromptCacheFixture.openDiskStore(
            globalRoot: globalPromptCacheRoot,
            activeModelPromptCacheDirectory: globalPromptCacheRoot
                .appendingPathComponent("active-model/active-revision", isDirectory: true),
            modelContract: try PersistentPromptCacheFixture.ornithModelContract(),
            globalPromptCacheMaximumSizeBytes: 1);

        #expect(FileManager.default.fileExists(
            atPath: foreignBlocksDirectory.appendingPathComponent(
                firstBlockHash, isDirectory: true).path) == false);
        #expect(FileManager.default.fileExists(
            atPath: foreignBlocksDirectory.appendingPathComponent(
                secondBlockHash, isDirectory: true).path) == false);
    }

    @Test
    func should_not_join_foreign_block_subtrees_across_storage_contract_fingerprints() throws {
        let globalPromptCacheRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("quota-topology-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalPromptCacheRoot); }
        let foreignBlocksDirectory: URL = globalPromptCacheRoot
            .appendingPathComponent("foreign-model/foreign-revision/blocks", isDirectory: true);
        let parentBlockHash: String = String(repeating: "a", count: 64);
        let childBlockHash: String = String(repeating: "b", count: 64);
        try Self.writeForeignBlock(
            foreignBlocksDirectory: foreignBlocksDirectory, blockHash: parentBlockHash,
            parentBlockHash: nil, storageContractFingerprint: String(repeating: "1", count: 64),
            payloadByteCount: 128, modifiedAt: Date(timeIntervalSince1970: 0));
        try Self.writeForeignBlock(
            foreignBlocksDirectory: foreignBlocksDirectory, blockHash: childBlockHash,
            parentBlockHash: parentBlockHash,
            storageContractFingerprint: String(repeating: "2", count: 64),
            payloadByteCount: 128,
            modifiedAt: Date(timeIntervalSince1970: 10));
        let childDirectory: URL = foreignBlocksDirectory.appendingPathComponent(
            childBlockHash, isDirectory: true);
        let childFileSizeBytes: UInt64 = try PersistentPromptCacheFixture
            .directoryFileSizeBytes(directoryPath: childDirectory);

        // The child names the parent's hash but declares incompatible tensor
        // geometry; a one-child quota evicts the oldest candidate, and the
        // parent eviction must not sweep the independently scoped child.
        _ = try PersistentPromptCacheFixture.openDiskStore(
            globalRoot: globalPromptCacheRoot,
            activeModelPromptCacheDirectory: globalPromptCacheRoot
                .appendingPathComponent("active-model/active-revision", isDirectory: true),
            modelContract: try PersistentPromptCacheFixture.ornithModelContract(),
            globalPromptCacheMaximumSizeBytes: childFileSizeBytes);

        #expect(FileManager.default.fileExists(
            atPath: foreignBlocksDirectory.appendingPathComponent(
                parentBlockHash, isDirectory: true).path) == false);
        #expect(FileManager.default.fileExists(atPath: childDirectory.path),
            Comment("evicting the parent must not sweep a foreign child that declares a different storage contract fingerprint"));
    }

    @Test
    func should_remove_stale_transactions_before_evicting_committed_blocks() throws {
        let globalPromptCacheRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("quota-topology-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalPromptCacheRoot); }
        let foreignBlocksDirectory: URL = globalPromptCacheRoot
            .appendingPathComponent("foreign-model/foreign-revision/blocks", isDirectory: true);
        let committedBlockHash: String = String(repeating: "c", count: 64);
        try Self.writeForeignBlock(
            foreignBlocksDirectory: foreignBlocksDirectory, blockHash: committedBlockHash,
            parentBlockHash: nil, storageContractFingerprint: String(repeating: "3", count: 64),
            payloadByteCount: 128, modifiedAt: Date(timeIntervalSince1970: 0));
        let committedBlockDirectory: URL = foreignBlocksDirectory.appendingPathComponent(
            committedBlockHash, isDirectory: true);
        let committedBlockSizeBytes: UInt64 = try PersistentPromptCacheFixture
            .directoryFileSizeBytes(directoryPath: committedBlockDirectory);
        let staleTransactionDirectory: URL = foreignBlocksDirectory
            .appendingPathComponent("\(String(repeating: "d", count: 64)).staging-interrupted",
                isDirectory: true);
        try FileManager.default.createDirectory(
            at: staleTransactionDirectory, withIntermediateDirectories: true);
        try Data(count: 512).write(
            to: staleTransactionDirectory.appendingPathComponent("sequence.safetensors.tmp"));

        // Durable content is older than the staging content: transaction
        // class, not timestamp alone, controls the first eviction decision.
        _ = try PersistentPromptCacheFixture.openDiskStore(
            globalRoot: globalPromptCacheRoot,
            activeModelPromptCacheDirectory: globalPromptCacheRoot
                .appendingPathComponent("active-model/active-revision", isDirectory: true),
            modelContract: try PersistentPromptCacheFixture.ornithModelContract(),
            globalPromptCacheMaximumSizeBytes: committedBlockSizeBytes);

        #expect(FileManager.default.fileExists(atPath: committedBlockDirectory.path),
            Comment("the older committed block must survive while a newer stale transaction exists"));
        #expect(FileManager.default.fileExists(
            atPath: staleTransactionDirectory.path) == false);
    }

    private static func writeForeignBlock(
        foreignBlocksDirectory: URL,
        blockHash: String,
        parentBlockHash: String?,
        storageContractFingerprint: String,
        payloadByteCount: Int,
        modifiedAt: Date
    ) throws {
        let blockDirectory: URL = foreignBlocksDirectory.appendingPathComponent(
            blockHash, isDirectory: true);
        try FileManager.default.createDirectory(
            at: blockDirectory, withIntermediateDirectories: true);
        let manifestObject: [String: Any] = [
            "format_version": "12",
            "block_hash": blockHash,
            "block_index": parentBlockHash == nil ? 0 : 1,
            "parent_block_hash": parentBlockHash ?? NSNull(),
            "storage_contract_fingerprint": storageContractFingerprint,
            "has_sequence_state": true,
            "has_boundary_state": true,
        ];
        try JSONSerialization.data(withJSONObject: manifestObject).write(
            to: blockDirectory.appendingPathComponent("manifest.json"));
        try Data(count: payloadByteCount).write(
            to: blockDirectory.appendingPathComponent("payload.bin"));
        try FileManager.default.setAttributes(
            [.modificationDate: modifiedAt], ofItemAtPath: blockDirectory.path);
    }
}
