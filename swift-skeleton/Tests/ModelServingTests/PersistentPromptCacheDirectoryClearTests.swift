import Foundation;

import Testing;

import ModelServing;

@testable import ModelServing;

/// Hermetic journeys for safe cache-tree deletion: a model-scoped clear
/// removes exactly that model's measured blocks and bytes while sibling
/// namespaces survive, a global clear empties the root, an absent root is a
/// measured no-op, and unsafe model identities are rejected before any
/// filesystem access.
final class PersistentPromptCacheDirectoryClearTests {

    @Test
    func should_clear_one_model_namespace_and_preserve_siblings() throws {
        let globalRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("cache-clear-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalRoot); }
        let targetBytes: Data = Data(repeating: 9, count: 512);
        try Self.writeBlockDirectory(
            globalRoot: globalRoot,
            modelPathComponents: ["org", "model-a", "rev-1"],
            blockDirectoryName: String(repeating: "a", count: 64),
            payloadBytes: targetBytes);
        try Self.writeBlockDirectory(
            globalRoot: globalRoot,
            modelPathComponents: ["org", "model-b", "rev-1"],
            blockDirectoryName: String(repeating: "b", count: 64),
            payloadBytes: targetBytes);

        let outcome: PersistentPromptCacheClearOutcome = try PersistentPromptCacheDirectoryClear
            .clearPersistentPromptCacheDirectory(
                globalPromptCacheRootDirectory: globalRoot, modelId: "org/model-a");

        #expect(outcome.modelId == "org/model-a");
        #expect(outcome.blocksRemoved == 1);
        #expect(Int(outcome.bytesFreed) >= 512);
        #expect(FileManager.default.fileExists(
            atPath: globalRoot.appendingPathComponent("org/model-b/rev-1").path),
            "sibling namespaces must survive a model-scoped clear");
    }

    @Test
    func should_clear_the_global_root_and_treat_an_absent_root_as_a_no_op() throws {
        let globalRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("cache-clear-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalRoot); }
        try Self.writeBlockDirectory(
            globalRoot: globalRoot,
            modelPathComponents: ["org", "model-a", "rev-1"],
            blockDirectoryName: String(repeating: "c", count: 64),
            payloadBytes: Data(repeating: 1, count: 256));

        let globalOutcome: PersistentPromptCacheClearOutcome = try PersistentPromptCacheDirectoryClear
            .clearPersistentPromptCacheDirectory(globalPromptCacheRootDirectory: globalRoot,
                modelId: nil);
        #expect(globalOutcome.blocksRemoved == 1);
        let clearedDirectoryEntries: [URL] = try FileManager.default.contentsOfDirectory(
            at: globalRoot, includingPropertiesForKeys: nil, options: []);
        #expect(clearedDirectoryEntries.isEmpty, "a global clear empties the root");

        let absentOutcome: PersistentPromptCacheClearOutcome = try PersistentPromptCacheDirectoryClear
            .clearPersistentPromptCacheDirectory(
                globalPromptCacheRootDirectory: FileManager.default.temporaryDirectory
                    .appendingPathComponent("cache-clear-\(UUID().uuidString)"),
                modelId: "org/model-a");
        #expect(absentOutcome.blocksRemoved == 0);
        #expect(absentOutcome.bytesFreed == 0);
    }

    @Test
    func should_reject_unsafe_model_identities_before_any_filesystem_access() throws {
        let globalRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("cache-clear-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(
            at: globalRoot, withIntermediateDirectories: true);
        defer { try? FileManager.default.removeItem(at: globalRoot); }

        #expect(throws: (any Error).self) {
            _ = try PersistentPromptCacheDirectoryClear.clearPersistentPromptCacheDirectory(
                globalPromptCacheRootDirectory: globalRoot, modelId: "../escape");
        };
        #expect(throws: (any Error).self) {
            _ = try PersistentPromptCacheDirectoryClear.clearPersistentPromptCacheDirectory(
                globalPromptCacheRootDirectory: globalRoot, modelId: "");
        };
    }

    /// Writes one committed block directory under the modeled cache layout:
    /// `<root>/<model>/<revision>/blocks/<hash>/manifest.json` plus payload.
    private static func writeBlockDirectory(
        globalRoot: URL, modelPathComponents: [String], blockDirectoryName: String,
        payloadBytes: Data
    ) throws {
        var blockDirectory: URL = globalRoot;
        for pathComponent: String in modelPathComponents {
            blockDirectory.appendPathComponent(pathComponent);
        }
        blockDirectory.appendPathComponent("blocks", isDirectory: true);
        blockDirectory.appendPathComponent(blockDirectoryName, isDirectory: true);
        try FileManager.default.createDirectory(
            at: blockDirectory, withIntermediateDirectories: true);
        let manifestObject: [String: Any] = [
            "block_hash": blockDirectoryName,
            "block_index": 0,
            "format_version": "12",
            "has_boundary_state": false,
            "has_sequence_state": true,
            "parent_block_hash": NSNull(),
            "storage_contract_fingerprint": String(repeating: "0", count: 64),
        ];
        try JSONSerialization.data(withJSONObject: manifestObject, options: [.sortedKeys])
            .write(to: blockDirectory.appendingPathComponent("manifest.json"));
        try payloadBytes.write(to: blockDirectory.appendingPathComponent("sequence.safetensors"));
    }
}
