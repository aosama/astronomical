import Foundation;

import Testing;

import ModelServing;

@testable import ModelServing;

/// Hermetic journeys for the startup scan: valid committed blocks survive
/// with their ancestry, stale temporaries and staging transactions are
/// reclaimed as interrupted-transaction evidence, corrupt manifests and
/// orphaned children are removed as corrupt-format evidence, and the
/// compacted-parent boundary exemption honors the retention stride.
final class PersistentPromptCacheDiskStoreScanTests {

    @Test
    func should_keep_a_valid_chain_and_reclaim_corrupt_and_stale_entries() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let blocksDirectory: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("scan-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(
            at: blocksDirectory, withIntermediateDirectories: true);
        defer { try? FileManager.default.removeItem(at: blocksDirectory); }

        let promptTokens: [UInt32] = PersistentPromptCacheFixture
            .promptTokensWithCompleteBlocksAndTrailingTokens(
                modelContract: modelContract, completeBlockCount: 3, trailingTokenCount: 0);
        let blockKeys: [PersistentPromptCacheBlockKey] = try PersistentPromptCacheFixture
            .blockKeysForPrompt(
                modelContract: modelContract, promptTokens: promptTokens, requestedBlockCount: 3);
        for (blockIndex, blockKey): (Int, PersistentPromptCacheBlockKey) in blockKeys.enumerated() {
            let parentBlockKey: PersistentPromptCacheBlockKey? =
                blockIndex == 0 ? nil : blockKeys[blockIndex - 1];
            let blockDirectory: URL = try Self.writeCommittedBlock(
                blocksDirectory: blocksDirectory,
                blockKey: blockKey,
                parentBlockKey: parentBlockKey,
                modelContract: modelContract);
            // Both contract-owned state kinds must be present for a
            // hybrid model: sequence files per block, boundary snapshots
            // per block, each header self-describing its token count.
            _ = try Self.writeStateFile(
                blockDirectory: blockDirectory, blockKey: blockKey,
                modelContract: modelContract, sequenceState: true);
            _ = try Self.writeStateFile(
                blockDirectory: blockDirectory, blockKey: blockKey,
                modelContract: modelContract, sequenceState: false);
        }
        // Stale temporary and an uncommitted staging transaction.
        try Data("stale".utf8).write(
            to: blocksDirectory.appendingPathComponent("blocks.json.tmp"));
        let stagingDirectory: URL = blocksDirectory
            .appendingPathComponent("pending.staging-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(at: stagingDirectory, withIntermediateDirectories: true);
        // A corrupt directory whose manifest does not parse.
        let corruptDirectory: URL = blocksDirectory
            .appendingPathComponent(String(repeating: "a", count: 64), isDirectory: true);
        try FileManager.default.createDirectory(at: corruptDirectory, withIntermediateDirectories: true);
        try Data("not json".utf8).write(to: corruptDirectory.appendingPathComponent("manifest.json"));

        var trackedFiles: PersistentPromptCacheDiskStoreIndex = PersistentPromptCacheDiskStoreIndex();
        var cleanupEvidence: PersistentPromptCacheStartupCleanupEvidence =
            PersistentPromptCacheStartupCleanupEvidence();
        try PersistentPromptCacheDiskStoreScan.scanCurrentFormatBlockDirectories(
            blocksDirectory: blocksDirectory,
            trackedFiles: &trackedFiles,
            modelContract: modelContract,
            startupCleanupEvidence: &cleanupEvidence);

        #expect(trackedFiles.trackedBlocks().count >= 1,
            "the valid committed chain must survive the scan");
        #expect(cleanupEvidence.interruptedTransactionRecovery.blockCount >= 1,
            "the abandoned staging transaction must be reclaimed");
        #expect(cleanupEvidence.corruptCurrentFormat.blockCount >= 1,
            "the unparseable manifest must be reclaimed as corrupt format");
    }

    @Test
    func should_prune_an_orphaned_child_block() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let blocksDirectory: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("scan-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(
            at: blocksDirectory, withIntermediateDirectories: true);
        defer { try? FileManager.default.removeItem(at: blocksDirectory); }

        let promptTokens: [UInt32] = PersistentPromptCacheFixture
            .promptTokensWithCompleteBlocksAndTrailingTokens(
                modelContract: modelContract, completeBlockCount: 2, trailingTokenCount: 0);
        let blockKeys: [PersistentPromptCacheBlockKey] = try PersistentPromptCacheFixture
            .blockKeysForPrompt(
                modelContract: modelContract, promptTokens: promptTokens, requestedBlockCount: 2);
        // Commit only the child; its parent directory is absent, so the
        // child's ancestry is unverifiable and the scan must prune it.
        _ = try Self.writeCommittedBlock(
            blocksDirectory: blocksDirectory,
            blockKey: blockKeys[1],
            parentBlockKey: blockKeys[0],
            modelContract: modelContract);

        var trackedFiles: PersistentPromptCacheDiskStoreIndex = PersistentPromptCacheDiskStoreIndex();
        var cleanupEvidence: PersistentPromptCacheStartupCleanupEvidence =
            PersistentPromptCacheStartupCleanupEvidence();
        try PersistentPromptCacheDiskStoreScan.scanCurrentFormatBlockDirectories(
            blocksDirectory: blocksDirectory,
            trackedFiles: &trackedFiles,
            modelContract: modelContract,
            startupCleanupEvidence: &cleanupEvidence);

        #expect(trackedFiles.trackedBlocks().isEmpty,
            "a child without its committed parent must not survive the scan");
        #expect(cleanupEvidence.corruptCurrentFormat.blockCount == 1);
    }

    /// Writes one committed block directory: hex-hash name, manifest, and
    /// empty state files shaped by the store's own publication names.
    private static func writeCommittedBlock(
        blocksDirectory: URL,
        blockKey: PersistentPromptCacheBlockKey,
        parentBlockKey: PersistentPromptCacheBlockKey?,
        modelContract: PersistentPromptCacheModelContract
    ) throws -> URL {
        let blockDirectory: URL = blocksDirectory.appendingPathComponent(
            PersistentPromptCacheStoreFile.hexEncode(blockKey.blockHash()), isDirectory: true);
        try FileManager.default.createDirectory(
            at: blockDirectory, withIntermediateDirectories: true);
        let manifest: PersistentPromptCacheBlockManifest = try PersistentPromptCacheBlockManifestTests
            .buildManifest(
                blockKey: blockKey, parentBlockKey: parentBlockKey, modelContract: modelContract);
        _ = try manifest.writeToStagingDirectory(stagingBlockDirectory: blockDirectory);
        return blockDirectory;
    }

    /// Writes a minimal state file of either contract kind whose header
    /// self-describes the block's token count and storage contract.
    private static func writeStateFile(
        blockDirectory: URL, blockKey: PersistentPromptCacheBlockKey,
        modelContract: PersistentPromptCacheModelContract, sequenceState: Bool
    ) throws -> URL {
        let stateLayouts: [DecoderCachePersistedTensorLayout] = sequenceState
            ? modelContract.decoderCacheLayout.sequenceTensorLayouts()
            : modelContract.decoderCacheLayout.boundaryTensorLayouts();
        var headerObject: [String: Any] = [:];
        var payloadOffsetBytes: UInt64 = 0;
        for persistedTensorLayout: DecoderCachePersistedTensorLayout in stateLayouts {
            let tensorLayout: DecoderCacheTensorLayout = persistedTensorLayout.tensorLayout;
            let tensorShape: [Int] = tensorLayout.dimensions.enumerated().map(
                { (dimensionEntry: (offset: Int, element: Int)) -> Int in
                    if dimensionEntry.offset == tensorLayout.sequenceAxis {
                        return blockKey.tokenCount();
                    }
                    return dimensionEntry.element;
                });
            var tensorPayloadByteCount: UInt64 = UInt64(tensorLayout.dtype.scalarByteCount);
            for tensorDimension: Int in tensorShape {
                tensorPayloadByteCount = tensorPayloadByteCount &* UInt64(max(tensorDimension, 0));
            }
            headerObject[persistedTensorLayout.persistentTensorName] = [
                "dtype": tensorLayout.dtype.safetensorsDtypeName,
                "shape": tensorShape,
                "data_offsets": [payloadOffsetBytes, payloadOffsetBytes &+ tensorPayloadByteCount],
            ];
            payloadOffsetBytes = payloadOffsetBytes &+ tensorPayloadByteCount;
        }
        headerObject["__metadata__"] = [
            "format_version": "12",
            "block_token_count": String(blockKey.tokenCount()),
            "storage_contract_fingerprint": modelContract.storageContractFingerprintHex(),
        ];
        let sequenceFileUrl: URL = blockDirectory.appendingPathComponent(
            sequenceState
                ? PersistentPromptCacheStoreFile.SEQUENCE_STATE_FILE_NAME
                : PersistentPromptCacheStoreFile.BOUNDARY_STATE_FILE_NAME);
        var fileBytes: Data = Data();
        let headerData: Data = try JSONSerialization.data(
            withJSONObject: headerObject, options: [.sortedKeys]);
        var littleEndianHeaderLength: UInt64 = UInt64(headerData.count);
        withUnsafeBytes(of: &littleEndianHeaderLength) { (valueBuffer: UnsafeRawBufferPointer) in
            fileBytes.append(contentsOf: valueBuffer);
        };
        fileBytes.append(headerData);
        fileBytes.append(Data(count: Int(payloadOffsetBytes)));
        try fileBytes.write(to: sequenceFileUrl);
        return sequenceFileUrl;
    }
    @Test
    func should_enforce_one_global_prompt_cache_quota_across_model_directories() throws {
        let globalPromptCacheRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("scan-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalPromptCacheRoot); }
        _ = try Self.writeCrossModelPromptCacheFile(
            globalPromptCacheRoot: globalPromptCacheRoot, modelDirectoryName: "z-older-model",
            fileNameHexDigit: "a", fileByteCount: Self.FIRST_CROSS_MODEL_FILE_BYTE_COUNT);
        _ = try Self.writeCrossModelPromptCacheFile(
            globalPromptCacheRoot: globalPromptCacheRoot, modelDirectoryName: "a-newer-model",
            fileNameHexDigit: "b", fileByteCount: Self.FIRST_CROSS_MODEL_FILE_BYTE_COUNT);

        let globalPromptCacheMaximumSizeBytes: UInt64 = UInt64(
            Self.FIRST_CROSS_MODEL_FILE_BYTE_COUNT * 2 - 1);
        let persistentPromptCache: PersistentPromptCacheDiskStore = try
            PersistentPromptCacheFixture.openDiskStore(
                globalRoot: globalPromptCacheRoot,
                activeModelPromptCacheDirectory: globalPromptCacheRoot
                    .appendingPathComponent("third-active-model/revision", isDirectory: true),
                modelContract: try PersistentPromptCacheFixture.ornithModelContract(),
                globalPromptCacheMaximumSizeBytes: globalPromptCacheMaximumSizeBytes);

        let actualGlobalSizeBytes: UInt64 = try PersistentPromptCacheFixture
            .directoryFileSizeBytes(directoryPath: globalPromptCacheRoot);
        #expect(actualGlobalSizeBytes <= globalPromptCacheMaximumSizeBytes,
            "global prompt-cache bytes must not exceed one configured maximum");
        #expect(persistentPromptCache.totalSizeBytes() == actualGlobalSizeBytes,
            "reported prompt-cache bytes must describe global root usage");
    }

    @Test
    func should_evict_the_oldest_written_cross_model_file_first() throws {
        let globalPromptCacheRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("scan-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalPromptCacheRoot); }
        let olderFilePath: URL = try Self.writeCrossModelPromptCacheFile(
            globalPromptCacheRoot: globalPromptCacheRoot, modelDirectoryName: "z-older-model",
            fileNameHexDigit: "f", fileByteCount: Self.FIRST_CROSS_MODEL_FILE_BYTE_COUNT);
        let newerFilePath: URL = try Self.writeCrossModelPromptCacheFile(
            globalPromptCacheRoot: globalPromptCacheRoot, modelDirectoryName: "a-newer-model",
            fileNameHexDigit: "a", fileByteCount: Self.FIRST_CROSS_MODEL_FILE_BYTE_COUNT);
        try FileManager.default.setAttributes(
            [.modificationDate: Date(timeIntervalSince1970: 10)],
            ofItemAtPath: olderFilePath.path);
        try FileManager.default.setAttributes(
            [.modificationDate: Date(timeIntervalSince1970: 20)],
            ofItemAtPath: newerFilePath.path);

        _ = try PersistentPromptCacheFixture.openDiskStore(
            globalRoot: globalPromptCacheRoot,
            activeModelPromptCacheDirectory: globalPromptCacheRoot
                .appendingPathComponent("third-active-model/revision", isDirectory: true),
            modelContract: try PersistentPromptCacheFixture.ornithModelContract(),
            globalPromptCacheMaximumSizeBytes: UInt64(Self.FIRST_CROSS_MODEL_FILE_BYTE_COUNT));

        #expect(FileManager.default.fileExists(atPath: olderFilePath.path) == false,
            "the oldest-written cross-model file must evict first");
        #expect(FileManager.default.fileExists(atPath: newerFilePath.path));
    }

    @Test
    func should_evict_a_cross_model_block_parent_with_its_descendants_under_global_quota_pressure()
        throws {
        let globalPromptCacheRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("scan-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalPromptCacheRoot); }
        let parentBlockDirectory: URL = try Self.writeCrossModelBlockDirectory(
            globalPromptCacheRoot: globalPromptCacheRoot, modelDirectoryName: "z-older-model",
            blockHashHex: Self.blockHashHex(forDigit: "1"), parentBlockHashHex: nil,
            stateFileByteCount: 256, modifiedAtSeconds: 10);
        let childBlockDirectory: URL = try Self.writeCrossModelBlockDirectory(
            globalPromptCacheRoot: globalPromptCacheRoot, modelDirectoryName: "z-older-model",
            blockHashHex: Self.blockHashHex(forDigit: "2"),
            parentBlockHashHex: Self.blockHashHex(forDigit: "1"),
            stateFileByteCount: 256, modifiedAtSeconds: 20);
        let unrelatedBlockDirectory: URL = try Self.writeCrossModelBlockDirectory(
            globalPromptCacheRoot: globalPromptCacheRoot, modelDirectoryName: "a-newer-model",
            blockHashHex: Self.blockHashHex(forDigit: "3"), parentBlockHashHex: nil,
            stateFileByteCount: 256, modifiedAtSeconds: 30);

        let unrelatedBlockSizeBytes: UInt64 = try PersistentPromptCacheFixture
            .directoryFileSizeBytes(directoryPath: unrelatedBlockDirectory);
        _ = try PersistentPromptCacheFixture.openDiskStore(
            globalRoot: globalPromptCacheRoot,
            activeModelPromptCacheDirectory: globalPromptCacheRoot
                .appendingPathComponent("active-model/revision", isDirectory: true),
            modelContract: try PersistentPromptCacheFixture.ornithModelContract(),
            globalPromptCacheMaximumSizeBytes: unrelatedBlockSizeBytes);

        #expect(FileManager.default.fileExists(atPath: parentBlockDirectory.path) == false,
            "evicting a parent block must delete the parent directory");
        #expect(FileManager.default.fileExists(atPath: childBlockDirectory.path) == false,
            "evicting a parent block must also delete dependent descendants");
        #expect(FileManager.default.fileExists(atPath: unrelatedBlockDirectory.path),
            "the newer unrelated block should remain after subtree eviction satisfies quota");
    }

    @Test
    func should_delete_stale_cross_model_block_staging_directory_below_global_quota() throws {
        let globalPromptCacheRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("scan-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalPromptCacheRoot); }
        let staleStagingDirectory: URL = globalPromptCacheRoot
            .appendingPathComponent("cross-model/revision/blocks", isDirectory: true)
            .appendingPathComponent("\(Self.blockHashHex(forDigit: "4")).staging-test",
                isDirectory: true);
        try FileManager.default.createDirectory(
            at: staleStagingDirectory, withIntermediateDirectories: true);
        try Data(count: 128).write(
            to: staleStagingDirectory.appendingPathComponent("sequence.safetensors.tmp"));

        _ = try PersistentPromptCacheFixture.openDiskStore(
            globalRoot: globalPromptCacheRoot,
            activeModelPromptCacheDirectory: globalPromptCacheRoot
                .appendingPathComponent("active-model/revision", isDirectory: true),
            modelContract: try PersistentPromptCacheFixture.ornithModelContract(),
            globalPromptCacheMaximumSizeBytes: 10_000);

        #expect(FileManager.default.fileExists(atPath: staleStagingDirectory.path) == false,
            "abandoned block staging directories must not remain globally visible");
    }

    @Test
    func should_return_typed_error_when_cross_model_global_eviction_fails() throws {
        let globalPromptCacheRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("scan-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalPromptCacheRoot); }
        let crossModelFilePath: URL = try Self.writeCrossModelPromptCacheFile(
            globalPromptCacheRoot: globalPromptCacheRoot, modelDirectoryName: "cross-model",
            fileNameHexDigit: "c", fileByteCount: Self.FIRST_CROSS_MODEL_FILE_BYTE_COUNT);
        let crossModelDirectory: URL = crossModelFilePath.deletingLastPathComponent();
        let originalPermissions: [FileAttributeKey: Any] = try FileManager.default
            .attributesOfItem(atPath: crossModelDirectory.path);
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o555], ofItemAtPath: crossModelDirectory.path);
        defer {
            try? FileManager.default.setAttributes(
                originalPermissions, ofItemAtPath: crossModelDirectory.path);
        }

        do {
            _ = try PersistentPromptCacheFixture.openDiskStore(
                globalRoot: globalPromptCacheRoot,
                activeModelPromptCacheDirectory: globalPromptCacheRoot
                    .appendingPathComponent("active-model/revision", isDirectory: true),
                modelContract: try PersistentPromptCacheFixture.ornithModelContract(),
                globalPromptCacheMaximumSizeBytes: 0);
            Issue.record("global quota enforcement must not hide deletion failure");
        } catch let openError as PersistentPromptCacheDiskStoreError {
            guard case let .removeCacheOwnedFile(filePath, _) = openError else {
                Issue.record("expected typed removal error, got \(openError)");
                return;
            }
            #expect(filePath == crossModelFilePath.path);
        }
        #expect(FileManager.default.fileExists(atPath: crossModelFilePath.path),
            "the undeletable cross-model file must remain for permissions to be restored");
    }

    @Test
    func should_delete_cross_model_stale_writer_temp_below_global_quota() throws {
        let globalPromptCacheRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("scan-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalPromptCacheRoot); }
        let crossModelKvBlocksDirectory: URL = globalPromptCacheRoot
            .appendingPathComponent("cross-model/revision/kv_blocks", isDirectory: true);
        try FileManager.default.createDirectory(
            at: crossModelKvBlocksDirectory, withIntermediateDirectories: true);
        let staleWriterTempFilePath: URL = crossModelKvBlocksDirectory
            .appendingPathComponent("\(Self.blockHashHex(forDigit: "d")).safetensors.tmp");
        try Data(count: 128).write(to: staleWriterTempFilePath);

        _ = try PersistentPromptCacheFixture.openDiskStore(
            globalRoot: globalPromptCacheRoot,
            activeModelPromptCacheDirectory: globalPromptCacheRoot
                .appendingPathComponent("active-model/revision", isDirectory: true),
            modelContract: try PersistentPromptCacheFixture.ornithModelContract(),
            globalPromptCacheMaximumSizeBytes: 10_000);

        #expect(FileManager.default.fileExists(atPath: staleWriterTempFilePath.path) == false);
    }

    @Test
    func should_delete_invalid_content_safetensors_file_under_one_byte_quota() throws {
        let persistentPromptCacheRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("scan-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: persistentPromptCacheRoot); }
        let kvBlocksDirectory: URL = persistentPromptCacheRoot
            .appendingPathComponent("kv_blocks", isDirectory: true);
        try FileManager.default.createDirectory(
            at: kvBlocksDirectory, withIntermediateDirectories: true);
        let invalidKvBlockFilePath: URL = kvBlocksDirectory
            .appendingPathComponent("\(Self.blockHashHex(forDigit: "0")).safetensors");
        try Data("not a safetensors file".utf8).write(to: invalidKvBlockFilePath);

        let persistentPromptCache: PersistentPromptCacheDiskStore = try
            PersistentPromptCacheFixture.openDiskStore(
                globalRoot: persistentPromptCacheRoot,
                activeModelPromptCacheDirectory: persistentPromptCacheRoot,
                modelContract: try PersistentPromptCacheFixture.ornithModelContract(),
                globalPromptCacheMaximumSizeBytes: 1);

        #expect(persistentPromptCache.sequenceStateBlockCount() == 0);
        #expect(persistentPromptCache.totalSizeBytes() == 0,
            "cache-owned bytes must be zero after deleting the only invalid file");
        #expect(FileManager.default.fileExists(atPath: invalidKvBlockFilePath.path) == false,
            Comment("invalid cache-owned files must be deleted so they do not consume disk capacity beyond the quota"));
    }

    @Test
    func should_return_remove_prompt_cache_file_error_when_deletion_fails() throws {
        let persistentPromptCacheRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("scan-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: persistentPromptCacheRoot); }
        let kvBlocksDirectory: URL = persistentPromptCacheRoot
            .appendingPathComponent("kv_blocks", isDirectory: true);
        try FileManager.default.createDirectory(
            at: kvBlocksDirectory, withIntermediateDirectories: true);
        let invalidKvBlockFilePath: URL = kvBlocksDirectory
            .appendingPathComponent("\(Self.blockHashHex(forDigit: "0")).safetensors");
        try Data("not a safetensors file".utf8).write(to: invalidKvBlockFilePath);
        let originalPermissions: [FileAttributeKey: Any] = try FileManager.default
            .attributesOfItem(atPath: kvBlocksDirectory.path);
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o555], ofItemAtPath: kvBlocksDirectory.path);
        defer {
            try? FileManager.default.setAttributes(
                originalPermissions, ofItemAtPath: kvBlocksDirectory.path);
        }

        do {
            _ = try PersistentPromptCacheFixture.openDiskStore(
                globalRoot: persistentPromptCacheRoot,
                activeModelPromptCacheDirectory: persistentPromptCacheRoot,
                modelContract: try PersistentPromptCacheFixture.ornithModelContract(),
                globalPromptCacheMaximumSizeBytes: 1);
            Issue.record("a failing cache-owned deletion must fail the open");
        } catch let openError as PersistentPromptCacheDiskStoreError {
            guard case .removeCacheOwnedFile = openError else {
                Issue.record("expected typed removal error, got \(openError)");
                return;
            }
        }
    }

    private static func blockHashHex(forDigit digit: String) -> String {
        return String(repeating: digit, count: 64);
    }

    private static func writeCrossModelPromptCacheFile(
        globalPromptCacheRoot: URL, modelDirectoryName: String, fileNameHexDigit: String,
        fileByteCount: Int
    ) throws -> URL {
        let crossModelVisualEmbeddingsDirectory: URL = globalPromptCacheRoot
            .appendingPathComponent(modelDirectoryName, isDirectory: true)
            .appendingPathComponent("revision", isDirectory: true)
            .appendingPathComponent("visual_embeddings", isDirectory: true);
        try FileManager.default.createDirectory(
            at: crossModelVisualEmbeddingsDirectory, withIntermediateDirectories: true);
        let crossModelFilePath: URL = crossModelVisualEmbeddingsDirectory
            .appendingPathComponent(
                "\(String(repeating: fileNameHexDigit, count: 64)).safetensors");
        try Data(count: fileByteCount).write(to: crossModelFilePath);
        return crossModelFilePath;
    }

    private static func writeCrossModelBlockDirectory(
        globalPromptCacheRoot: URL, modelDirectoryName: String, blockHashHex: String,
        parentBlockHashHex: String?, stateFileByteCount: Int, modifiedAtSeconds: TimeInterval
    ) throws -> URL {
        let crossModelBlockDirectory: URL = globalPromptCacheRoot
            .appendingPathComponent(modelDirectoryName, isDirectory: true)
            .appendingPathComponent("revision", isDirectory: true)
            .appendingPathComponent("blocks", isDirectory: true)
            .appendingPathComponent(blockHashHex, isDirectory: true);
        try FileManager.default.createDirectory(
            at: crossModelBlockDirectory, withIntermediateDirectories: true);
        let manifestObject: [String: Any] = [
            "format_version": "12",
            "block_hash": blockHashHex,
            "block_index": 0,
            "parent_block_hash": parentBlockHashHex ?? NSNull(),
            "storage_contract_fingerprint": "cross-model-contract-fingerprint",
            "has_sequence_state": true,
            "has_boundary_state": true,
        ];
        try JSONSerialization.data(withJSONObject: manifestObject).write(
            to: crossModelBlockDirectory.appendingPathComponent("manifest.json"));
        try Data(repeating: 1, count: stateFileByteCount).write(
            to: crossModelBlockDirectory.appendingPathComponent("sequence.safetensors"));
        try Data(repeating: 2, count: stateFileByteCount).write(
            to: crossModelBlockDirectory.appendingPathComponent("boundary.safetensors"));
        try FileManager.default.setAttributes(
            [.modificationDate: Date(timeIntervalSince1970: modifiedAtSeconds)],
            ofItemAtPath: crossModelBlockDirectory.path);
        return crossModelBlockDirectory;
    }

    private static let FIRST_CROSS_MODEL_FILE_BYTE_COUNT: Int = 1_024;
}
