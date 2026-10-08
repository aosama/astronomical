import Foundation;

import Testing;

import ModelServing;

@testable import ModelServing;

/// Hermetic journeys for crash-safe block publication: a complete block
/// publishes atomically under its content hash with the staging directory
/// gone and the index tracking real bytes, a duplicate publication fails
/// closed on topology mismatch, and a parent boundary is reclaimed only
/// after the child is durable and only when quota pressure demands it.
final class PersistentPromptCacheBlockTransactionTests {

    /// Writes real header-shaped state files, mirroring the contract's
    /// exact-size geometry so the transaction's size validation holds.
    private final class SyntheticStateFileStaging: PersistentPromptCacheStateFileStaging {

        func stageStateFile(
            stateFileName: String, stagingBlockDirectory: URL, blockTokenCount: Int,
            modelContract: PersistentPromptCacheModelContract
        ) throws -> UInt64 {
            let stateLayouts: [DecoderCachePersistedTensorLayout] = stateFileName
                == PersistentPromptCacheStoreFile.SEQUENCE_STATE_FILE_NAME
                ? modelContract.decoderCacheLayout.sequenceTensorLayouts()
                : modelContract.decoderCacheLayout.boundaryTensorLayouts();
            // The native writer materializes tensors largest-first (name
            // tie-break) so the exact-size geometry stays predictive; the
            // stager must assign payload offsets in that same order, because
            // cumulative offsets pass through different values — and
            // therefore different digit counts — in any other order.
            var tensorEntries: [(
                tensorName: String, dtypeName: String, dimensions: [Int], payloadBytes: UInt64
            )] = [];
            for persistedTensorLayout: DecoderCachePersistedTensorLayout in stateLayouts {
                let tensorLayout: DecoderCacheTensorLayout = persistedTensorLayout.tensorLayout;
                let tensorShape: [Int] = tensorLayout.dimensions.enumerated().map(
                    { (dimensionEntry: (offset: Int, element: Int)) -> Int in
                        if dimensionEntry.offset == tensorLayout.sequenceAxis {
                            return blockTokenCount;
                        }
                        return dimensionEntry.element;
                    });
                var tensorPayloadByteCount: UInt64 = UInt64(tensorLayout.dtype.scalarByteCount);
                for tensorDimension: Int in tensorShape {
                    tensorPayloadByteCount = tensorPayloadByteCount
                        &* UInt64(max(tensorDimension, 0));
                }
                tensorEntries.append((
                    persistedTensorLayout.persistentTensorName,
                    tensorLayout.dtype.safetensorsDtypeName,
                    tensorShape,
                    tensorPayloadByteCount));
            }
            tensorEntries.sort(by: { (leftTensor, rightTensor) -> Bool in
                if leftTensor.payloadBytes != rightTensor.payloadBytes {
                    return leftTensor.payloadBytes > rightTensor.payloadBytes;
                }
                return leftTensor.tensorName < rightTensor.tensorName;
            });
            var headerObject: [String: Any] = [:];
            var payloadOffsetBytes: UInt64 = 0;
            for tensorEntry in tensorEntries {
                headerObject[tensorEntry.tensorName] = [
                    "dtype": tensorEntry.dtypeName,
                    "shape": tensorEntry.dimensions,
                    "data_offsets": [payloadOffsetBytes, payloadOffsetBytes &+ tensorEntry.payloadBytes],
                ];
                payloadOffsetBytes = payloadOffsetBytes &+ tensorEntry.payloadBytes;
            }
            headerObject["__metadata__"] = [
                "format_version": PersistentPromptCacheBlockHeader.FORMAT_VERSION,
                "block_token_count": blockTokenCount,
                "storage_contract_fingerprint": modelContract.storageContractFingerprintHex(),
            ];
            var fileBytes: Data = Data();
            let headerData: Data = try JSONSerialization.data(
                withJSONObject: headerObject, options: [.sortedKeys]);
            var littleEndianHeaderLength: UInt64 = UInt64(headerData.count);
            withUnsafeBytes(of: &littleEndianHeaderLength) { (valueBuffer: UnsafeRawBufferPointer) in
                fileBytes.append(contentsOf: valueBuffer);
            };
            fileBytes.append(headerData);
            fileBytes.append(Data(count: Int(payloadOffsetBytes)));
            try fileBytes.write(
                to: stagingBlockDirectory.appendingPathComponent(stateFileName));
            return UInt64(fileBytes.count);
        }
    }

    @Test
    func should_publish_a_block_atomically_and_reject_a_duplicate() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let diskStore: PersistentPromptCacheDiskStore = try Self.openStore(
            modelContract: modelContract);
        let promptTokens: [UInt32] = PersistentPromptCacheFixture
            .promptTokensWithCompleteBlocksAndTrailingTokens(
                modelContract: modelContract, completeBlockCount: 1, trailingTokenCount: 0);
        let blockKey: PersistentPromptCacheBlockKey = try PersistentPromptCacheBlockKey
            .forRootBlock(
                modelContract: modelContract,
                blockTokens: Array(promptTokens[..<modelContract.blockTokenCount]));
        let staging: PersistentPromptCacheStateFileStaging = SyntheticStateFileStaging();

        try diskStore.publishNewBlockTransaction(
            staging: staging, blockKey: blockKey, parentBlockKey: nil);

        let blockDirectoryName: String = PersistentPromptCacheStoreFile.hexEncode(
            blockKey.blockHash());
        #expect(FileManager.default.fileExists(
            atPath: diskStore.blocksDirectory
                .appendingPathComponent(blockDirectoryName, isDirectory: true).path),
            "the block must publish under its content hash");
        #expect(diskStore.sequenceStateBlockCount() == 1);
        #expect(diskStore.boundaryStateSnapshotCount() == 1);
        #expect(diskStore.totalSizeBytes() > 0);

        #expect(throws: PersistentPromptCacheDiskStoreError.existingBlockTopologyMismatch(
            blockHash: blockKey.blockHash())) {
            try diskStore.publishNewBlockTransaction(
                staging: staging, blockKey: blockKey, parentBlockKey: nil);
        };
    }

    @Test
    func should_reclaim_a_redundant_parent_boundary_under_quota_pressure() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        // Measure one committed block first, then size the quota so three
        // chained blocks fit only after the non-checkpoint parent boundary
        // is reclaimed at the grandchild's publication.
        let measurementStore: PersistentPromptCacheDiskStore = try Self.openStore(
            modelContract: modelContract);
        let measurementTokens: [UInt32] = PersistentPromptCacheFixture
            .promptTokensWithCompleteBlocksAndTrailingTokens(
                modelContract: modelContract, completeBlockCount: 1, trailingTokenCount: 0);
        let measurementBlockKey: PersistentPromptCacheBlockKey = try PersistentPromptCacheBlockKey
            .forRootBlock(
                modelContract: modelContract,
                blockTokens: Array(measurementTokens[..<modelContract.blockTokenCount]));
        try measurementStore.publishNewBlockTransaction(
            staging: SyntheticStateFileStaging(), blockKey: measurementBlockKey,
            parentBlockKey: nil);
        let singleBlockSizeBytes: UInt64 = measurementStore.totalSizeBytes();
        let boundaryFileSizeBytes: UInt64 = try modelContract
            .boundaryStateFileBytesForBlockTokenCount(
                blockTokenCount: modelContract.blockTokenCount);
        let quotaBytes: UInt64 = singleBlockSizeBytes &* 3 &- boundaryFileSizeBytes &+ 1024;

        let diskStore: PersistentPromptCacheDiskStore = try Self.openStore(
            modelContract: modelContract, globalPromptCacheMaximumSizeBytes: quotaBytes);
        let promptTokens: [UInt32] = PersistentPromptCacheFixture
            .promptTokensWithCompleteBlocksAndTrailingTokens(
                modelContract: modelContract, completeBlockCount: 3, trailingTokenCount: 0);
        let blockKeys: [PersistentPromptCacheBlockKey] = try PersistentPromptCacheFixture
            .blockKeysForPrompt(
                modelContract: modelContract, promptTokens: promptTokens, requestedBlockCount: 3);
        let staging: PersistentPromptCacheStateFileStaging = SyntheticStateFileStaging();
        try diskStore.publishNewBlockTransaction(
            staging: staging, blockKey: blockKeys[0], parentBlockKey: nil);
        try diskStore.publishNewBlockTransaction(
            staging: staging, blockKey: blockKeys[1], parentBlockKey: blockKeys[0]);
        try diskStore.publishNewBlockTransaction(
            staging: staging, blockKey: blockKeys[2], parentBlockKey: blockKeys[1]);

        // The child sits at non-checkpoint index 1, so the grandchild's
        // publication reclaims its boundary under quota pressure while the
        // sequence state stays durable for recompute from the root.
        let childBlockDirectory: URL = diskStore.blocksDirectory
            .appendingPathComponent(
                PersistentPromptCacheStoreFile.hexEncode(blockKeys[1].blockHash()),
                isDirectory: true);
        #expect(FileManager.default.fileExists(
            atPath: childBlockDirectory.appendingPathComponent(
                PersistentPromptCacheStoreFile.SEQUENCE_STATE_FILE_NAME).path),
            "the non-checkpoint parent sequence state must stay durable");
        #expect(FileManager.default.fileExists(
            atPath: childBlockDirectory.appendingPathComponent(
                PersistentPromptCacheStoreFile.BOUNDARY_STATE_FILE_NAME).path) == false,
            "quota pressure must reclaim the redundant parent boundary");
        #expect(diskStore.hasKvBlock(blockHash: blockKeys[0].blockHash()),
            "the root checkpoint boundary and sequence state must stay durable");
        #expect(diskStore.hasRecurrentSnapshot(blockHash: blockKeys[0].blockHash()));
        #expect(diskStore.hasKvBlock(blockHash: blockKeys[2].blockHash()));
        #expect(diskStore.hasRecurrentSnapshot(blockHash: blockKeys[2].blockHash()));
        #expect(diskStore.totalSizeBytes() <= quotaBytes,
            "committed cache bytes must satisfy the configured quota");
    }

    private static func openStore(
        modelContract: PersistentPromptCacheModelContract,
        globalPromptCacheMaximumSizeBytes: UInt64 = 50_000_000_000
    ) throws -> PersistentPromptCacheDiskStore {
        let globalRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("txn-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(
            at: globalRoot, withIntermediateDirectories: true);
        return try PersistentPromptCacheDiskStore.open(
            diskStoreConfig: PersistentPromptCacheDiskStoreConfig(
                activeModelPromptCacheDirectory: globalRoot
                    .appendingPathComponent("org/model-a/rev-1", isDirectory: true),
                globalPromptCacheRootDirectory: globalRoot,
                globalPromptCacheMaximumSizeBytes: globalPromptCacheMaximumSizeBytes),
            modelContract: modelContract);
    }
}
