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
        try manifest.writeToStagingDirectory(stagingBlockDirectory: blockDirectory);
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
}
