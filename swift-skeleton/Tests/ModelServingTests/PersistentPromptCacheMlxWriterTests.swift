import Foundation;

import Testing;

import MLX;

import JourneyCategories;

import ModelServingTestSupport;

@testable import ModelServing;

extension HermeticMlxJourneyContainer {

    /// Hermetic journeys for the production MLX state-file writer: real
    /// safetensors bytes written from real MLX arrays through the same
    /// transaction the synthetic journeys exercise, with the read-back
    /// validator and rescan recovery proving every file on disk.
    @Suite(.tags(.hermeticMlxJourney))
    final class PersistentPromptCacheMlxWriterTests {

        init() {
            signal(SIGPIPE, SIG_IGN);
            MLXMetallibLocator.overrideMetallibPathIfNecessary();
        }

        /// Fills one contiguous payload with deterministic data: element
        /// value = scalar index modulo 17, so every tensor is distinct yet
        /// reproducible.
        private static func deterministicFloatValues(
            scalarCount: Int
        ) -> [Float] {
            return (0..<scalarCount).map({ (scalarIndex: Int) -> Float in
                return Float(scalarIndex % 17);
            });
        }

        private static func sequenceStateTensors(
            modelContract: PersistentPromptCacheModelContract,
            sequenceAxisTokenCount: Int
        ) -> [String: MLXArray] {
            var sequenceStateTensors: [String: MLXArray] = [:];
            for persistedTensorLayout: DecoderCachePersistedTensorLayout
            in modelContract.decoderCacheLayout.sequenceTensorLayouts() {
                var tensorShape: [Int] = persistedTensorLayout.tensorLayout.dimensions
                    .map({ (dimensionElement: Int) -> Int in
                        return max(dimensionElement, 1);
                    });
                if let sequenceAxis: Int = persistedTensorLayout.tensorLayout.sequenceAxis {
                    tensorShape[sequenceAxis] = sequenceAxisTokenCount;
                }
                let filledArray: MLXArray = MLXArray(
                    Self.deterministicFloatValues(scalarCount: tensorShape.reduce(1, *)),
                    tensorShape);
                sequenceStateTensors[persistedTensorLayout.persistentTensorName] =
                    filledArray.asType(.float16);
            }
            return sequenceStateTensors;
        }

        @Test
        func should_publish_a_required_capture_with_production_mlx_writer_bytes() throws {
            let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
                .syntheticSequenceOnlyContract();
            let globalRoot: URL = FileManager.default.temporaryDirectory
                .appendingPathComponent("prompt-cache-\(UUID().uuidString)", isDirectory: true);
            try FileManager.default.createDirectory(
                at: globalRoot, withIntermediateDirectories: true);
            let diskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
                .openDiskStore(globalRoot: globalRoot, modelContract: modelContract);
            let rootTokens: [UInt32] = PersistentPromptCacheFixture
                .promptTokensWithCompleteBlocksAndTrailingTokens(
                    modelContract: modelContract, completeBlockCount: 1, trailingTokenCount: 0);
            let rootBlockKey: PersistentPromptCacheBlockKey = try PersistentPromptCacheBlockKey
                .forRootBlock(modelContract: modelContract, blockTokens: rootTokens);
            let mlxStaging: PersistentPromptCacheStateFileMlxStaging =
                PersistentPromptCacheStateFileMlxStaging(
                    sequenceStateTensors: Self.sequenceStateTensors(
                        modelContract: modelContract,
                        sequenceAxisTokenCount: modelContract.blockTokenCount),
                    boundaryStateTensors: [:]);

            let publicationOutcome: PersistentPromptCachePublicationOutcome = try diskStore
                .publishBlock(staging: mlxStaging, blockKey: rootBlockKey, parentBlockKey: nil);

            #expect(publicationOutcome == .published);
            #expect(diskStore.sequenceStateBlockCount() == 1);
            let stateFileUrl: URL = diskStore.blocksDirectory
                .appendingPathComponent(PersistentPromptCacheStoreFile.hexEncode(
                    rootBlockKey.blockHash()))
                .appendingPathComponent(PersistentPromptCacheStoreFile.SEQUENCE_STATE_FILE_NAME);
            let projectedFileBytes: UInt64 = try modelContract
                .sequenceStateFileBytesForBlockTokenCount(
                    blockTokenCount: modelContract.blockTokenCount);
            let actualFileAttributes: [FileAttributeKey: Any] = try FileManager.default
                .attributesOfItem(atPath: stateFileUrl.path);
            #expect(UInt64((actualFileAttributes[.size] as? NSNumber)?.intValue ?? -1)
                == projectedFileBytes,
                "the MLX writer must produce the contract-projected byte count");
            let persistedHeader: PersistentPromptCacheBlockHeader = try
                PersistentPromptCacheBlockHeader.readKvBlock(
                    blockFileUrl: stateFileUrl, modelContract: modelContract);
            #expect(persistedHeader.blockTokenCount == modelContract.blockTokenCount);
            #expect(persistedHeader.formatVersion == PersistentPromptCacheBlockHeader
                .FORMAT_VERSION);
            #expect(persistedHeader.tensorCount == 2);

            let republishedOutcome: PersistentPromptCachePublicationOutcome = try diskStore
                .publishBlock(staging: mlxStaging, blockKey: rootBlockKey, parentBlockKey: nil);
            #expect(republishedOutcome == .alreadyPublished,
                "the exact writer's own files must satisfy full idempotency validation");

            let rescannedDiskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
                .openDiskStore(globalRoot: globalRoot, modelContract: modelContract);
            #expect(rescannedDiskStore.sequenceStateBlockCount() == 1,
                "the rescan must recover the MLX-written block from disk");
        }

        @Test
        func should_publish_a_chain_of_four_boundaries_through_the_production_writer() throws {
            let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
                .syntheticSequenceOnlyContract();
            let globalRoot: URL = FileManager.default.temporaryDirectory
                .appendingPathComponent("prompt-cache-\(UUID().uuidString)", isDirectory: true);
            try FileManager.default.createDirectory(
                at: globalRoot, withIntermediateDirectories: true);
            let diskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
                .openDiskStore(globalRoot: globalRoot, modelContract: modelContract);
            let promptTokens: [UInt32] = PersistentPromptCacheFixture
                .promptTokensWithCompleteBlocksAndTrailingTokens(
                    modelContract: modelContract, completeBlockCount: 4, trailingTokenCount: 0);
            let chainedBlockKeys: [PersistentPromptCacheBlockKey] = try PersistentPromptCacheFixture
                .blockKeysForPrompt(
                    modelContract: modelContract, promptTokens: promptTokens,
                    requestedBlockCount: 4);
            let mlxStaging: PersistentPromptCacheStateFileMlxStaging =
                PersistentPromptCacheStateFileMlxStaging(
                    sequenceStateTensors: Self.sequenceStateTensors(
                        modelContract: modelContract,
                        sequenceAxisTokenCount: modelContract.blockTokenCount),
                    boundaryStateTensors: [:]);

            var parentBlockKey: PersistentPromptCacheBlockKey? = nil;
            for chainedBlockKey: PersistentPromptCacheBlockKey in chainedBlockKeys {
                let publicationOutcome: PersistentPromptCachePublicationOutcome = try diskStore
                    .publishBlock(
                        staging: mlxStaging, blockKey: chainedBlockKey,
                        parentBlockKey: parentBlockKey);
                #expect(publicationOutcome == .published);
                parentBlockKey = chainedBlockKey;
            }

            #expect(diskStore.sequenceStateBlockCount() == 4);
            let rescannedDiskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
                .openDiskStore(globalRoot: globalRoot, modelContract: modelContract);
            #expect(rescannedDiskStore.sequenceStateBlockCount() == 4);
        }

        @Test
        func should_write_header_bytes_matching_an_independent_json_construction() throws {
            let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
                .syntheticSequenceOnlyContract();
            let globalRoot: URL = FileManager.default.temporaryDirectory
                .appendingPathComponent("prompt-cache-\(UUID().uuidString)", isDirectory: true);
            try FileManager.default.createDirectory(
                at: globalRoot, withIntermediateDirectories: true);
            let diskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
                .openDiskStore(globalRoot: globalRoot, modelContract: modelContract);
            let rootTokens: [UInt32] = PersistentPromptCacheFixture
                .promptTokensWithCompleteBlocksAndTrailingTokens(
                    modelContract: modelContract, completeBlockCount: 1, trailingTokenCount: 0);
            let rootBlockKey: PersistentPromptCacheBlockKey = try PersistentPromptCacheBlockKey
                .forRootBlock(modelContract: modelContract, blockTokens: rootTokens);
            let mlxStaging: PersistentPromptCacheStateFileMlxStaging =
                PersistentPromptCacheStateFileMlxStaging(
                    sequenceStateTensors: Self.sequenceStateTensors(
                        modelContract: modelContract,
                        sequenceAxisTokenCount: modelContract.blockTokenCount),
                    boundaryStateTensors: [:]);
            #expect(try diskStore.publishBlock(
                staging: mlxStaging, blockKey: rootBlockKey, parentBlockKey: nil) == .published);
            let stateFileUrl: URL = diskStore.blocksDirectory
                .appendingPathComponent(PersistentPromptCacheStoreFile.hexEncode(
                    rootBlockKey.blockHash()))
                .appendingPathComponent(PersistentPromptCacheStoreFile.SEQUENCE_STATE_FILE_NAME);

            let fileBytes: Data = try Data(contentsOf: stateFileUrl);
            #expect(fileBytes.count >= 8);
            var headerLengthBytes: UInt64 = 0;
            withUnsafeMutableBytes(of: &headerLengthBytes) { (valueBuffer: UnsafeMutableRawBufferPointer) in
                for bufferIndex: Int in 0..<8 {
                    valueBuffer.storeBytes(
                        of: UInt8(fileBytes[fileBytes.startIndex + bufferIndex]),
                        toByteOffset: bufferIndex, as: UInt8.self);
                }
            };
            let headerJsonBytes: Data = fileBytes.subdata(
                in: 8..<(8 + Int(headerLengthBytes)));
            // The independent construction serializes the same header through
            // JSONSerialization with sorted keys; byte equality proves the
            // writer's compact, sorted, string-valued header layout without
            // sharing its construction code.
            var expectedHeaderObject: [String: Any] = [:];
            var payloadOffsetBytes: UInt64 = 0;
            for persistedTensorLayout: DecoderCachePersistedTensorLayout
            in modelContract.decoderCacheLayout.sequenceTensorLayouts() {
                let tensorShape: [Int] = persistedTensorLayout.tensorLayout.dimensions
                    .enumerated()
                    .map({ (dimensionEntry: (offset: Int, element: Int)) -> Int in
                        if dimensionEntry.offset == persistedTensorLayout.tensorLayout
                            .sequenceAxis {
                            return modelContract.blockTokenCount;
                        }
                        return dimensionEntry.element;
                    });
                let scalarByteCount: Int = persistedTensorLayout.tensorLayout.dtype
                    .scalarByteCount;
                let payloadByteCount: Int = scalarByteCount
                    * tensorShape.reduce(1, *);
                expectedHeaderObject[persistedTensorLayout.persistentTensorName] = [
                    "dtype": persistedTensorLayout.tensorLayout.dtype.safetensorsDtypeName,
                    "shape": tensorShape,
                    "data_offsets": [payloadOffsetBytes, payloadOffsetBytes + UInt64(payloadByteCount)],
                ];
                payloadOffsetBytes = payloadOffsetBytes + UInt64(payloadByteCount);
            }
            expectedHeaderObject["__metadata__"] = [
                "format_version": PersistentPromptCacheBlockHeader.FORMAT_VERSION,
                "block_token_count": String(modelContract.blockTokenCount),
                "storage_contract_fingerprint": modelContract.storageContractFingerprintHex(),
            ];
            let expectedHeaderJsonBytes: Data = try JSONSerialization.data(
                withJSONObject: expectedHeaderObject, options: [.sortedKeys]);

            #expect(headerJsonBytes == expectedHeaderJsonBytes,
                "the MLX writer's header bytes must match an independent JSON construction");
            #expect(UInt64(fileBytes.count)
                == 8 + UInt64(headerLengthBytes) + payloadOffsetBytes,
                "the file must hold exactly the header section plus raw tensor payloads");
            let parsedMetadata: [String: Any] = try JSONSerialization.jsonObject(
                with: headerJsonBytes) as! [String: Any];
            let metadataObject: [String: Any] = parsedMetadata["__metadata__"] as! [String: Any];
            #expect(metadataObject["block_token_count"] is String,
                "block_token_count must serialize as a JSON string");
            #expect(metadataObject["format_version"] is String);
            #expect(metadataObject["storage_contract_fingerprint"] is String);
        }

        @Test
        func should_publish_a_boundary_snapshot_through_the_production_writer() throws {
            let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
                .syntheticBoundaryOnlyContract();
            let globalRoot: URL = FileManager.default.temporaryDirectory
                .appendingPathComponent("prompt-cache-\(UUID().uuidString)", isDirectory: true);
            try FileManager.default.createDirectory(
                at: globalRoot, withIntermediateDirectories: true);
            let diskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
                .openDiskStore(globalRoot: globalRoot, modelContract: modelContract);
            let rootTokens: [UInt32] = PersistentPromptCacheFixture
                .promptTokensWithCompleteBlocksAndTrailingTokens(
                    modelContract: modelContract, completeBlockCount: 1, trailingTokenCount: 0);
            let rootBlockKey: PersistentPromptCacheBlockKey = try PersistentPromptCacheBlockKey
                .forRootBlock(modelContract: modelContract, blockTokens: rootTokens);
            var boundaryStateTensors: [String: MLXArray] = [:];
            for persistedTensorLayout: DecoderCachePersistedTensorLayout
            in modelContract.decoderCacheLayout.boundaryTensorLayouts() {
                let tensorShape: [Int] = persistedTensorLayout.tensorLayout.dimensions
                    .map({ (dimensionElement: Int) -> Int in
                        return max(dimensionElement, 1);
                    });
                boundaryStateTensors[persistedTensorLayout.persistentTensorName] = MLXArray(
                    Self.deterministicFloatValues(scalarCount: tensorShape.reduce(1, *)),
                    tensorShape);
            }
            let mlxStaging: PersistentPromptCacheStateFileMlxStaging =
                PersistentPromptCacheStateFileMlxStaging(
                    sequenceStateTensors: [:], boundaryStateTensors: boundaryStateTensors);

            let publicationOutcome: PersistentPromptCachePublicationOutcome = try diskStore
                .publishBlock(staging: mlxStaging, blockKey: rootBlockKey, parentBlockKey: nil);

            #expect(publicationOutcome == .published);
            #expect(diskStore.boundaryStateSnapshotCount() == 1);
            let snapshotFileUrl: URL = diskStore.blocksDirectory
                .appendingPathComponent(PersistentPromptCacheStoreFile.hexEncode(
                    rootBlockKey.blockHash()))
                .appendingPathComponent(PersistentPromptCacheStoreFile.BOUNDARY_STATE_FILE_NAME);
            let persistedHeader: PersistentPromptCacheBlockHeader = try
                PersistentPromptCacheBlockHeader.readRecurrentSnapshot(
                    snapshotFileUrl: snapshotFileUrl, modelContract: modelContract);
            #expect(persistedHeader.blockTokenCount == modelContract.blockTokenCount);
            let projectedFileBytes: UInt64 = try modelContract
                .boundaryStateFileBytesForBlockTokenCount(
                    blockTokenCount: modelContract.blockTokenCount);
            let actualFileAttributes: [FileAttributeKey: Any] = try FileManager.default
                .attributesOfItem(atPath: snapshotFileUrl.path);
            #expect(UInt64((actualFileAttributes[.size] as? NSNumber)?.intValue ?? -1)
                == projectedFileBytes);
        }

        @Test
        func should_reject_captured_arrays_that_disagree_with_the_contract() throws {
            let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
                .syntheticSequenceOnlyContract();
            let globalRoot: URL = FileManager.default.temporaryDirectory
                .appendingPathComponent("prompt-cache-\(UUID().uuidString)", isDirectory: true);
            try FileManager.default.createDirectory(
                at: globalRoot, withIntermediateDirectories: true);
            let diskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
                .openDiskStore(globalRoot: globalRoot, modelContract: modelContract);
            let rootTokens: [UInt32] = PersistentPromptCacheFixture
                .promptTokensWithCompleteBlocksAndTrailingTokens(
                    modelContract: modelContract, completeBlockCount: 1, trailingTokenCount: 0);
            let rootBlockKey: PersistentPromptCacheBlockKey = try PersistentPromptCacheBlockKey
                .forRootBlock(modelContract: modelContract, blockTokens: rootTokens);
            let contractShape: [Int] = [1, modelContract.blockTokenCount, 4];
            let mismatchedShape: [Int] = [1, modelContract.blockTokenCount - 1, 4];
            var mismatchedTensors: [String: MLXArray] = [:];
            for persistedTensorLayout: DecoderCachePersistedTensorLayout
            in modelContract.decoderCacheLayout.sequenceTensorLayouts() {
                mismatchedTensors[persistedTensorLayout.persistentTensorName] = MLXArray(
                    Self.deterministicFloatValues(
                        scalarCount: mismatchedShape.reduce(1, *)),
                    mismatchedShape).asType(.float16);
            }
            let mismatchedStaging: PersistentPromptCacheStateFileMlxStaging =
                PersistentPromptCacheStateFileMlxStaging(
                    sequenceStateTensors: mismatchedTensors, boundaryStateTensors: [:]);

            #expect(throws: PersistentPromptCacheDiskStoreError.self) {
                try diskStore.publishBlock(
                    staging: mismatchedStaging, blockKey: rootBlockKey, parentBlockKey: nil);
            }
            #expect(diskStore.sequenceStateBlockCount() == 0,
                "a contract mismatch must fail closed before anything is committed");
            let blocksDirectoryEntries: [URL] = try FileManager.default.contentsOfDirectory(
                at: diskStore.blocksDirectory, includingPropertiesForKeys: nil, options: []);
            #expect(blocksDirectoryEntries.isEmpty,
                "the failed staging transaction must leave no residue in blocks/");

            let conformingStaging: PersistentPromptCacheStateFileMlxStaging =
                PersistentPromptCacheStateFileMlxStaging(
                    sequenceStateTensors: Self.sequenceStateTensors(
                        modelContract: modelContract,
                        sequenceAxisTokenCount: modelContract.blockTokenCount),
                    boundaryStateTensors: [:]);
            let retryOutcome: PersistentPromptCachePublicationOutcome = try diskStore
                .publishBlock(
                    staging: conformingStaging, blockKey: rootBlockKey, parentBlockKey: nil);
            #expect(retryOutcome == .published,
                "conforming arrays must publish cleanly after a failed staging attempt");
            #expect(contractShape[1] == modelContract.blockTokenCount);
        }

        @Test
        func should_reject_a_state_file_name_outside_the_contract() throws {
            let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
                .syntheticSequenceOnlyContract();
            let stagingRoot: URL = FileManager.default.temporaryDirectory
                .appendingPathComponent("prompt-cache-staging-\(UUID().uuidString)",
                    isDirectory: true);
            try FileManager.default.createDirectory(
                at: stagingRoot, withIntermediateDirectories: true);
            let mlxStaging: PersistentPromptCacheStateFileMlxStaging =
                PersistentPromptCacheStateFileMlxStaging(
                    sequenceStateTensors: Self.sequenceStateTensors(
                        modelContract: modelContract,
                        sequenceAxisTokenCount: modelContract.blockTokenCount),
                    boundaryStateTensors: [:]);

            #expect(throws: PersistentPromptCacheDiskStoreError.self) {
                try mlxStaging.stageStateFile(
                    stateFileName: "uncontracted.safetensors",
                    stagingBlockDirectory: stagingRoot,
                    blockTokenCount: modelContract.blockTokenCount,
                    modelContract: modelContract);
            }
        }
    }
}
