import Foundation;

import Testing;

import ModelServing;

@testable import ModelServing;

/// Filesystem journeys for bounded startup-cleanup attribution: interrupted
/// transactions, obsolete retired trees, corrupt blocks, and forced quota
/// evictions each land in their own reason bucket with exact byte totals,
/// directories disappear, and the evidence is consumed exactly once.
final class PersistentPromptCacheStartupCleanupEvidenceTests {

    @Test
    func should_classify_startup_cleanup_and_consume_evidence_once() throws {
        let globalPromptCacheRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("cleanup-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalPromptCacheRoot); }
        let activeModelDirectory: URL = globalPromptCacheRoot
            .appendingPathComponent("fictional-model/fictional-revision", isDirectory: true);
        let blocksDirectory: URL = activeModelDirectory
            .appendingPathComponent("blocks", isDirectory: true);
        try FileManager.default.createDirectory(
            at: blocksDirectory, withIntermediateDirectories: true);

        let abandonedTransactionDirectory: URL = blocksDirectory
            .appendingPathComponent("\(String(repeating: "a", count: 64)).staging-interrupted",
                isDirectory: true);
        try FileManager.default.createDirectory(
            at: abandonedTransactionDirectory, withIntermediateDirectories: true);
        try Data(repeating: 1, count: 31).write(
            to: abandonedTransactionDirectory.appendingPathComponent(
                "sequence.safetensors.tmp"));
        let abandonedTransactionByteCount: UInt64 = try Self.directoryFileSizeBytes(
            directoryPath: abandonedTransactionDirectory);

        let corruptBlockDirectory: URL = blocksDirectory
            .appendingPathComponent(String(repeating: "b", count: 64), isDirectory: true);
        try FileManager.default.createDirectory(
            at: corruptBlockDirectory, withIntermediateDirectories: true);
        try Data("not-json".utf8).write(
            to: corruptBlockDirectory.appendingPathComponent("manifest.json"));
        try Data(repeating: 2, count: 37).write(
            to: corruptBlockDirectory.appendingPathComponent("payload.bin"));
        let corruptBlockByteCount: UInt64 = try Self.directoryFileSizeBytes(
            directoryPath: corruptBlockDirectory);

        let obsoleteSequenceDirectory: URL = activeModelDirectory
            .appendingPathComponent("kv_blocks", isDirectory: true);
        let obsoleteBoundaryDirectory: URL = activeModelDirectory
            .appendingPathComponent("recurrent_snapshots", isDirectory: true);
        try FileManager.default.createDirectory(
            at: obsoleteSequenceDirectory, withIntermediateDirectories: true);
        try FileManager.default.createDirectory(
            at: obsoleteBoundaryDirectory, withIntermediateDirectories: true);
        try Data(repeating: 3, count: 41).write(
            to: obsoleteSequenceDirectory.appendingPathComponent(
                "\(String(repeating: "c", count: 64)).safetensors"));
        try Data(repeating: 4, count: 43).write(
            to: obsoleteBoundaryDirectory.appendingPathComponent(
                "\(String(repeating: "d", count: 64)).safetensors"));

        let promptCache: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(
                globalRoot: globalPromptCacheRoot,
                activeModelPromptCacheDirectory: activeModelDirectory,
                modelContract: try PersistentPromptCacheFixture.ornithModelContract());
        let startupCleanupEvidence: PersistentPromptCacheStartupCleanupEvidence = try #require(
            promptCache.startupCleanupEvidence(),
            "startup cleanup should retain bounded evidence");

        #expect(startupCleanupEvidence.interruptedTransactionRecovery.blockCount == 1);
        #expect(startupCleanupEvidence.interruptedTransactionRecovery.byteCount
            == abandonedTransactionByteCount);
        #expect(startupCleanupEvidence.obsoleteFormat.artifactCount == 2);
        #expect(startupCleanupEvidence.obsoleteFormat.byteCount == 84);
        #expect(startupCleanupEvidence.corruptCurrentFormat.blockCount == 1);
        #expect(startupCleanupEvidence.corruptCurrentFormat.byteCount
            == corruptBlockByteCount);
        #expect(FileManager.default.fileExists(
            atPath: abandonedTransactionDirectory.path) == false);
        #expect(FileManager.default.fileExists(atPath: corruptBlockDirectory.path) == false);
        #expect(FileManager.default.fileExists(
            atPath: obsoleteSequenceDirectory.appendingPathComponent(
                "\(String(repeating: "c", count: 64)).safetensors").path) == false,
            "the obsolete sequence artifact must be removed");
        #expect(FileManager.default.fileExists(
            atPath: obsoleteBoundaryDirectory.appendingPathComponent(
                "\(String(repeating: "d", count: 64)).safetensors").path) == false,
            "the obsolete boundary artifact must be removed");

        #expect(promptCache.takeStartupCleanupEvidence() == startupCleanupEvidence);
        #expect(promptCache.takeStartupCleanupEvidence() == nil);
    }

    @Test
    func should_remove_retired_speculative_prefill_trees_without_touching_other_cache_data()
        throws {
        let globalPromptCacheRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("cleanup-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalPromptCacheRoot); }
        let activeModelDirectory: URL = globalPromptCacheRoot
            .appendingPathComponent("fictional-model/fictional-revision", isDirectory: true);
        let retiredSelectionDirectory: URL = activeModelDirectory
            .appendingPathComponent("speculative_prefill_selections", isDirectory: true);
        let retiredTargetStateDirectory: URL = activeModelDirectory
            .appendingPathComponent("speculative_prefill_target_states", isDirectory: true);
        try FileManager.default.createDirectory(
            at: retiredSelectionDirectory.appendingPathComponent(
                "nested", isDirectory: true), withIntermediateDirectories: true);
        try FileManager.default.createDirectory(
            at: retiredTargetStateDirectory, withIntermediateDirectories: true);
        try Data(repeating: 7, count: 29).write(
            to: retiredSelectionDirectory.appendingPathComponent(
                "nested/selection.safetensors"));
        try Data(repeating: 8, count: 31).write(
            to: retiredTargetStateDirectory.appendingPathComponent("target.safetensors"));
        let retiredByteCount: UInt64 = try Self.directoryFileSizeBytes(
            directoryPath: retiredSelectionDirectory)
            + Self.directoryFileSizeBytes(directoryPath: retiredTargetStateDirectory);
        let unrelatedCacheFile: URL = activeModelDirectory
            .appendingPathComponent("unrelated-cache-metadata");
        try Data("preserve".utf8).write(to: unrelatedCacheFile);

        let promptCache: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(
                globalRoot: globalPromptCacheRoot,
                activeModelPromptCacheDirectory: activeModelDirectory,
                modelContract: try PersistentPromptCacheFixture.ornithModelContract());
        let obsoleteFormatCleanup: PersistentPromptCacheStartupCleanupCategory = try #require(
            promptCache.startupCleanupEvidence(),
            "retired trees should produce startup cleanup evidence").obsoleteFormat;

        #expect(obsoleteFormatCleanup.artifactCount == 2);
        #expect(obsoleteFormatCleanup.byteCount == retiredByteCount);
        #expect(FileManager.default.fileExists(
            atPath: retiredSelectionDirectory.path) == false);
        #expect(FileManager.default.fileExists(
            atPath: retiredTargetStateDirectory.path) == false);
        #expect(FileManager.default.fileExists(atPath: unrelatedCacheFile.path));
    }

    @Test
    func should_count_startup_quota_eviction_by_artifact_and_block() throws {
        let globalPromptCacheRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("cleanup-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalPromptCacheRoot); }
        let foreignRevisionDirectory: URL = globalPromptCacheRoot
            .appendingPathComponent("foreign-model/foreign-revision", isDirectory: true);
        let foreignVisualEmbeddingDirectory: URL = foreignRevisionDirectory
            .appendingPathComponent("visual_embeddings", isDirectory: true);
        try FileManager.default.createDirectory(
            at: foreignVisualEmbeddingDirectory, withIntermediateDirectories: true);
        let foreignStandaloneFile: URL = foreignVisualEmbeddingDirectory
            .appendingPathComponent("\(String(repeating: "e", count: 64)).safetensors");
        try Data(repeating: 5, count: 47).write(to: foreignStandaloneFile);

        let foreignBlockDirectory: URL = foreignRevisionDirectory
            .appendingPathComponent("blocks", isDirectory: true)
            .appendingPathComponent(String(repeating: "f", count: 64), isDirectory: true);
        try FileManager.default.createDirectory(
            at: foreignBlockDirectory, withIntermediateDirectories: true);
        let foreignManifestObject: [String: Any] = [
            "format_version": "12",
            "block_hash": String(repeating: "f", count: 64),
            "block_index": 0,
            "parent_block_hash": NSNull(),
            "storage_contract_fingerprint": "fictional-foreign-contract",
            "has_sequence_state": true,
            "has_boundary_state": true,
        ];
        try JSONSerialization.data(withJSONObject: foreignManifestObject).write(
            to: foreignBlockDirectory.appendingPathComponent("manifest.json"));
        try Data(repeating: 6, count: 53).write(
            to: foreignBlockDirectory.appendingPathComponent("payload.bin"));
        let foreignBlockByteCount: UInt64 = try Self.directoryFileSizeBytes(
            directoryPath: foreignBlockDirectory);

        let promptCache: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(
                globalRoot: globalPromptCacheRoot,
                activeModelPromptCacheDirectory: globalPromptCacheRoot
                    .appendingPathComponent("active-model/active-revision", isDirectory: true),
                modelContract: try PersistentPromptCacheFixture.ornithModelContract(),
                globalPromptCacheMaximumSizeBytes: 0);
        let quotaEviction: PersistentPromptCacheStartupCleanupCategory = try #require(
            promptCache.startupCleanupEvidence(),
            "quota cleanup should retain evidence").quotaEviction;

        #expect(quotaEviction.artifactCount == 1);
        #expect(quotaEviction.blockCount == 1);
        #expect(quotaEviction.byteCount == 47 + foreignBlockByteCount);
        #expect(FileManager.default.fileExists(atPath: foreignStandaloneFile.path) == false);
        #expect(FileManager.default.fileExists(atPath: foreignBlockDirectory.path) == false);
    }

    private static func directoryFileSizeBytes(directoryPath: URL) throws -> UInt64 {
        var pendingDirectories: [URL] = [directoryPath];
        var totalByteCount: UInt64 = 0;
        while let pendingDirectory: URL = pendingDirectories.popLast() {
            for enumeratedEntry: URL in try FileManager.default.contentsOfDirectory(
                at: pendingDirectory, includingPropertiesForKeys: [.fileSizeKey],
                options: []) {
                let entryPath: URL = pendingDirectory.appendingPathComponent(
                    enumeratedEntry.lastPathComponent);
                var isDirectory: ObjCBool = ObjCBool(false);
                FileManager.default.fileExists(atPath: entryPath.path, isDirectory: &isDirectory);
                if isDirectory.boolValue {
                    pendingDirectories.append(entryPath);
                } else {
                    let fileAttributes: [FileAttributeKey: Any] = try FileManager.default
                        .attributesOfItem(atPath: entryPath.path);
                    totalByteCount = totalByteCount &+ ((fileAttributes[.size] as? NSNumber)?
                        .uint64Value ?? 0);
                }
            }
        }
        return totalByteCount;
    }
}
