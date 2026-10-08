import Foundation;

import MLX;

import RuntimeIntegration;


/// The production state-file stager: writes real safetensors files from the
/// engine's captured MLX arrays, port of the Rust direct writer in
/// `disk_store_file.rs`. One synchronous write per state file streams each
/// tensor's payload straight from the evaluated, contiguous array — there
/// is deliberately no serialized block-sized host buffer. The header bytes
/// come from the same shared layout the quota projection uses, so a
/// published file's size always equals the byte projection that admitted
/// it, and the read-back validator proves the staged file before the
/// transaction commits it.
public final class PersistentPromptCacheStateFileMlxStaging:
    PersistentPromptCacheStateFileStaging {

    private enum StateFileKind {

        case sequenceStateBlock;

        case boundaryStateSnapshot;
    }

    private let sequenceStateTensors: [String: MLXArray];

    private let boundaryStateTensors: [String: MLXArray];

    public init(
        sequenceStateTensors: [String: MLXArray],
        boundaryStateTensors: [String: MLXArray]
    ) {
        self.sequenceStateTensors = sequenceStateTensors;
        self.boundaryStateTensors = boundaryStateTensors;
    }

    public func stageStateFile(
        stateFileName: String,
        stagingBlockDirectory: URL,
        blockTokenCount: Int,
        modelContract: PersistentPromptCacheModelContract
    ) throws -> UInt64 {
        // Accept only the two contract-owned names. Allowing an arbitrary
        // caller name would bypass state-kind validation and broaden cleanup
        // authority.
        let stateFileKind: StateFileKind;
        let capturedTensors: [String: MLXArray];
        let persistedTensorLayouts: [DecoderCachePersistedTensorLayout];
        switch stateFileName {
        case PersistentPromptCacheStoreFile.SEQUENCE_STATE_FILE_NAME:
            stateFileKind = .sequenceStateBlock;
            capturedTensors = self.sequenceStateTensors;
            persistedTensorLayouts = modelContract.decoderCacheLayout.sequenceTensorLayouts();
        case PersistentPromptCacheStoreFile.BOUNDARY_STATE_FILE_NAME:
            stateFileKind = .boundaryStateSnapshot;
            capturedTensors = self.boundaryStateTensors;
            persistedTensorLayouts = modelContract.decoderCacheLayout.boundaryTensorLayouts();
        default:
            throw PersistentPromptCacheDiskStoreError.invalidStateFileName(
                stateFileName: stateFileName);
        }
        try Self.validateCapturedTensorPresence(
            capturedTensors: capturedTensors,
            persistedTensorLayouts: persistedTensorLayouts,
            stateFileName: stateFileName);
        let tensorEntries: [PersistentPromptCacheStateFileHeader.TensorEntry] = try
            PersistentPromptCacheStateFileHeader.tensorEntries(
                persistedTensorLayouts: persistedTensorLayouts,
                blockTokenCount: blockTokenCount);
        try Self.validateCapturedTensorLayouts(
            capturedTensors: capturedTensors, tensorEntries: tensorEntries,
            stateFileName: stateFileName);
        let headerJson: String = try PersistentPromptCacheStateFileHeader.headerJson(
            tensorEntries: tensorEntries,
            blockTokenCount: blockTokenCount,
            storageContractFingerprint: modelContract.storageContractFingerprintHex(),
            formatVersion: PersistentPromptCacheBlockHeader.FORMAT_VERSION);
        let stateFileUrl: URL = stagingBlockDirectory.appendingPathComponent(stateFileName);
        if FileManager.default.createFile(
            atPath: stateFileUrl.path, contents: Data(), attributes: nil) == false {
            throw PersistentPromptCacheDiskStoreError.openTempFile(
                tempFilePath: stateFileUrl.path,
                problem: "the staging state file could not be created");
        }
        let outputFileHandle: FileHandle;
        do {
            outputFileHandle = try FileHandle(forWritingTo: stateFileUrl);
        } catch {
            throw PersistentPromptCacheDiskStoreError.openTempFile(
                tempFilePath: stateFileUrl.path, problem: String(describing: error));
        }
        defer { try? outputFileHandle.close(); }
        try Self.writeHeaderAndTensorPayloads(
            outputFileHandle: outputFileHandle, headerJson: headerJson,
            tensorEntries: tensorEntries, capturedTensors: capturedTensors,
            stateFileUrl: stateFileUrl);
        do {
            try outputFileHandle.synchronize();
        } catch {
            throw PersistentPromptCacheDiskStoreError.synchronizeTempFile(
                tempFilePath: stateFileUrl.path, problem: String(describing: error));
        }
        let actualFileSizeBytes: UInt64 = try Self.actualFileSizeBytes(
            filePath: stateFileUrl.path);
        var expectedFileSizeBytes: UInt64 = 0;
        for tensorEntry: PersistentPromptCacheStateFileHeader.TensorEntry in tensorEntries {
            let (payloadTotalBytes, payloadOverflow) = expectedFileSizeBytes
                .addingReportingOverflow(UInt64(max(tensorEntry.payloadByteCount, 0)));
            if payloadOverflow {
                throw PersistentPromptCacheModelContractError.capturePayloadByteCountOverflow;
            }
            expectedFileSizeBytes = payloadTotalBytes;
        }
        let projectedFileSizeBytes: UInt64 = try PersistentPromptCacheStateFileHeader
            .totalFileBytes(
                headerJsonByteCount: headerJson.utf8.count,
                totalPayloadByteCount: expectedFileSizeBytes);
        if actualFileSizeBytes != projectedFileSizeBytes {
            throw PersistentPromptCacheDiskStoreError.writtenFileSizeMismatch(
                filePath: stateFileUrl.path,
                reportedSizeBytes: projectedFileSizeBytes,
                actualSizeBytes: actualFileSizeBytes);
        }
        // A successful write alone is not sufficient evidence of a complete
        // file: re-read the bounded header and validate it against the same
        // contract every future reader will apply.
        do {
            switch stateFileKind {
            case .sequenceStateBlock:
                _ = try PersistentPromptCacheBlockHeader.readKvBlock(
                    blockFileUrl: stateFileUrl, modelContract: modelContract);
            case .boundaryStateSnapshot:
                _ = try PersistentPromptCacheBlockHeader.readRecurrentSnapshot(
                    snapshotFileUrl: stateFileUrl, modelContract: modelContract);
            }
        } catch {
            throw PersistentPromptCacheDiskStoreError.validateBlock(
                blockFilePath: stateFileUrl.path, problem: String(describing: error));
        }
        return actualFileSizeBytes;
    }

    private static func validateCapturedTensorPresence(
        capturedTensors: [String: MLXArray],
        persistedTensorLayouts: [DecoderCachePersistedTensorLayout],
        stateFileName: String
    ) throws {
        if capturedTensors.count != persistedTensorLayouts.count {
            throw PersistentPromptCacheDiskStoreError.stateKindTensorPresenceMismatch(
                stateFileName: stateFileName,
                expectedTensorCount: persistedTensorLayouts.count,
                actualTensorCount: capturedTensors.count);
        }
    }

    private static func validateCapturedTensorLayouts(
        capturedTensors: [String: MLXArray],
        tensorEntries: [PersistentPromptCacheStateFileHeader.TensorEntry],
        stateFileName: String
    ) throws {
        for tensorEntry: PersistentPromptCacheStateFileHeader.TensorEntry in tensorEntries {
            guard let capturedArray: MLXArray = capturedTensors[tensorEntry.tensorName]
            else {
                throw PersistentPromptCacheDiskStoreError.saveSafetensors(
                    source: .tensorLookupFailed(tensorName: tensorEntry.tensorName));
            }
            let expectedShape: [Int] = tensorEntry.dimensions;
            if capturedArray.shape != expectedShape {
                throw PersistentPromptCacheDiskStoreError.saveSafetensors(
                    source: .runtimeOperation(
                        operation: "stage \(stateFileName)",
                        description: "tensor \(tensorEntry.tensorName) has shape "
                            + "\(capturedArray.shape) but the contract requires \(expectedShape)"));
            }
        }
    }

    private static func writeHeaderAndTensorPayloads(
        outputFileHandle: FileHandle,
        headerJson: String,
        tensorEntries: [PersistentPromptCacheStateFileHeader.TensorEntry],
        capturedTensors: [String: MLXArray],
        stateFileUrl: URL
    ) throws {
        do {
            var headerLengthBytes: UInt64 = UInt64(headerJson.utf8.count);
            // The 8-byte length prefix is little-endian by the safetensors
            // framing contract, written explicitly rather than trusting the
            // host integer layout.
            let headerLengthData: Data = withUnsafeBytes(
                of: &headerLengthBytes) { (valueBuffer: UnsafeRawBufferPointer) -> Data in
                var littleEndianBuffer: [UInt8] = [UInt8](repeating: 0, count: 8);
                for bufferIndex: Int in 0..<8 {
                    littleEndianBuffer[bufferIndex] = valueBuffer[bufferIndex];
                }
                return Data(littleEndianBuffer);
            };
            try outputFileHandle.write(contentsOf: headerLengthData);
            try outputFileHandle.write(contentsOf: Data(headerJson.utf8));
            // Payload order follows the header's offset assignment order, one
            // materialization at a time: the peak is one tensor's contiguous
            // copy rather than the sum of all tensors.
            for tensorEntry: PersistentPromptCacheStateFileHeader.TensorEntry in tensorEntries {
                guard let capturedArray: MLXArray = capturedTensors[tensorEntry.tensorName]
                else {
                    throw PersistentPromptCacheDiskStoreError.saveSafetensors(
                        source: .tensorLookupFailed(tensorName: tensorEntry.tensorName));
                }
                if tensorEntry.payloadByteCount == 0 {
                    continue;
                }
                let materializedArray: MLXArray = capturedArray.contiguous();
                materializedArray.eval();
                let payloadBytes: Data = materializedArray
                    .asData(access: .noCopyIfContiguous).data;
                if payloadBytes.count != tensorEntry.payloadByteCount {
                    throw PersistentPromptCacheDiskStoreError.saveSafetensors(
                        source: .runtimeOperation(
                            operation: "stage \(stateFileUrl.lastPathComponent)",
                            description: "tensor \(tensorEntry.tensorName) materialized "
                                + "\(payloadBytes.count) payload bytes but the contract "
                                + "projects \(tensorEntry.payloadByteCount)"));
                }
                try outputFileHandle.write(contentsOf: payloadBytes);
            }
        } catch let storeError as PersistentPromptCacheDiskStoreError {
            throw storeError;
        } catch {
            throw PersistentPromptCacheDiskStoreError.writeTempFile(
                tempFilePath: stateFileUrl.path, problem: String(describing: error));
        }
    }

    private static func actualFileSizeBytes(filePath: String) throws -> UInt64 {
        let fileAttributes: [FileAttributeKey: Any];
        do {
            fileAttributes = try FileManager.default.attributesOfItem(atPath: filePath);
        } catch {
            throw PersistentPromptCacheDiskStoreError.readBlockMetadata(
                blockFilePath: filePath, problem: String(describing: error));
        }
        guard let fileSizeNumber: NSNumber = fileAttributes[.size] as? NSNumber
        else {
            throw PersistentPromptCacheDiskStoreError.readBlockMetadata(
                blockFilePath: filePath, problem: "the file size attribute is missing");
        }
        return fileSizeNumber.uint64Value;
    }
}
