import Foundation;

import Testing;

import ModelServing;

@testable import ModelServing;

/// Hermetic journeys for the global quota scan: two models sharing an
/// identical block hash stay separate eviction subtrees (the cross-model
/// ancestry guard), staging transactions sort before durable content, and
/// byte totals count every owned file.
final class PersistentPromptCacheQuotaScanTests {

    @Test
    func should_keep_identical_hashes_in_different_models_separate_and_sort_staging_first()
        throws {
        let globalRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("quota-scan-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalRoot); }
        let blockDirectoryName: String = String(repeating: "e", count: 64);
        let payloadBytes: Data = Data(repeating: 5, count: 256);
        // The same content hash committed under two different models.
        try Self.writeBlockDirectory(
            globalRoot: globalRoot, modelPathComponents: ["org", "model-a"],
            blockDirectoryName: blockDirectoryName, payloadBytes: payloadBytes,
            parentBlockHash: nil);
        try Self.writeBlockDirectory(
            globalRoot: globalRoot, modelPathComponents: ["org", "model-b"],
            blockDirectoryName: blockDirectoryName, payloadBytes: payloadBytes,
            parentBlockHash: nil);
        // One abandoned staging transaction and one committed artifact file.
        let stagingDirectory: URL = globalRoot.appendingPathComponent(
            "org/model-a/blocks/pending.staging-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(
            at: stagingDirectory, withIntermediateDirectories: true);
        try Data(repeating: 1, count: 64).write(
            to: stagingDirectory.appendingPathComponent("sequence.safetensors.tmp"));

        let quotaScan: PersistentPromptCacheQuotaScan = try PersistentPromptCacheQuotaScanEngine
            .scanGlobalPromptCacheQuota(
                globalPromptCacheRootDirectory: globalRoot, excludedDirectory: nil);

        let blockSubtrees: [PersistentPromptCacheEvictionCandidate] = quotaScan
            .evictionCandidatesOldestWrittenFirst
            .filter({ (candidate: PersistentPromptCacheEvictionCandidate) -> Bool in
                if case .blockSubtree = candidate {
                    return true;
                }
                return false;
            });
        // The shared hash must produce two separate single-block subtrees:
        // a parent edge across models would let one eviction orphan the
        // other model's block.
        let singleBlockSubtrees: [PersistentPromptCacheEvictionCandidate] = blockSubtrees
            .filter({ (candidate: PersistentPromptCacheEvictionCandidate) -> Bool in
                return candidate.removedBlockCount == 1;
            });
        #expect(singleBlockSubtrees.count == 2,
            "identical hashes under different models must stay separate subtrees");
        let firstCandidate: PersistentPromptCacheEvictionCandidate = try #require(
            quotaScan.evictionCandidatesOldestWrittenFirst.first);
        #expect(firstCandidate.isUnconditionallyRemovable,
            "the staging transaction must sort before durable content");
        #expect(quotaScan.totalSizeBytes >= UInt64(256 * 2 + 64));
    }

    /// The exact failure mode that let quota pressure delete an in-progress
    /// publication: URL equality breaks across trailing-slash variants and
    /// symlink-resolved prefixes, so the exclusion must hold for the
    /// store-built exclusion URL against enumerated entries.
    @Test
    func should_exclude_the_active_staging_directory_from_eviction_candidates()
        throws {
        let globalRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("quota-scan-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalRoot); }
        let stagingDirectory: URL = globalRoot.appendingPathComponent(
            "org/model-a/blocks/\(String(repeating: "a", count: 64))"
                + ".staging-\(getpid())-123", isDirectory: true);
        try FileManager.default.createDirectory(
            at: stagingDirectory, withIntermediateDirectories: true);
        try Data(repeating: 7, count: 128).write(
            to: stagingDirectory.appendingPathComponent("sequence.safetensors"));

        let quotaScan: PersistentPromptCacheQuotaScan = try PersistentPromptCacheQuotaScanEngine
            .scanGlobalPromptCacheQuota(
                globalPromptCacheRootDirectory: globalRoot,
                excludedDirectory: stagingDirectory);

        #expect(quotaScan.totalSizeBytes == 0,
            "the excluded staging transaction must not count toward the quota");
        #expect(quotaScan.evictionCandidatesOldestWrittenFirst.isEmpty,
            "the excluded staging transaction must never become an eviction candidate");
    }

    /// Store-built paths and scan-emitted paths must share one string form:
    /// enumeration resolves /var to /private/var on macOS, and comparisons
    /// across those forms silently break exclusion, protection, and index
    /// removal.
    @Test
    func should_keep_scanned_candidate_paths_in_the_caller_provided_path_form()
        throws {
        let globalRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("quota-scan-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalRoot); }
        try Self.writeBlockDirectory(
            globalRoot: globalRoot, modelPathComponents: ["org", "model-a"],
            blockDirectoryName: String(repeating: "b", count: 64),
            payloadBytes: Data(repeating: 3, count: 128), parentBlockHash: nil);

        let quotaScan: PersistentPromptCacheQuotaScan = try PersistentPromptCacheQuotaScanEngine
            .scanGlobalPromptCacheQuota(
                globalPromptCacheRootDirectory: globalRoot, excludedDirectory: nil);

        #expect(quotaScan.evictionCandidatesOldestWrittenFirst.isEmpty == false,
            "the committed block must be scanned");
        for candidate: PersistentPromptCacheEvictionCandidate in quotaScan
            .evictionCandidatesOldestWrittenFirst {
            #expect(candidate.tieBreakerPath.hasPrefix(globalRoot.path),
                Comment("scanned candidate paths must stay in the caller-provided form: \(candidate.tieBreakerPath)"));
        }
    }

    private static func writeBlockDirectory(
        globalRoot: URL, modelPathComponents: [String], blockDirectoryName: String,
        payloadBytes: Data, parentBlockHash: Data?
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
            "block_index": parentBlockHash == nil ? 0 : 1,
            "format_version": "12",
            "has_boundary_state": false,
            "has_sequence_state": true,
            "parent_block_hash": parentBlockHash.map(
                { (parentHash: Data) -> String in
                    return PersistentPromptCacheStoreFile.hexEncode(parentHash);
                }) ?? NSNull(),
            "storage_contract_fingerprint": String(repeating: "f", count: 64),
        ];
        try JSONSerialization.data(withJSONObject: manifestObject, options: [.sortedKeys])
            .write(to: blockDirectory.appendingPathComponent("manifest.json"));
        try payloadBytes.write(to: blockDirectory.appendingPathComponent("sequence.safetensors"));
    }
}
