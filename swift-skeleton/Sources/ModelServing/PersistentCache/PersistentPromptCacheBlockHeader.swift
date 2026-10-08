import Foundation;


/// Bounded validator for persisted model-state files, port of the Rust
/// `PersistentPromptCacheBlockHeader`. This type reads only the
/// length-prefixed JSON header, validates the declared tensor layout against
/// the active architecture-neutral model contract, and returns the metadata
/// the prompt-cache owner needs to decide whether to load the file. It never
/// reads the multi-megabyte tensor payload region.
public struct PersistentPromptCacheBlockHeader: Equatable, Sendable {

    /// Current persistent prompt-cache state version. Bump when the on-disk
    /// layout or execution math changes in a way that invalidates serialized
    /// model state. Version 12 binds model-owned non-token causal input at
    /// the block where it enters the prompt.
    public static let FORMAT_VERSION: String = "12";

    private let formatVersionValue: String;
    private let storageContractFingerprintValue: String;
    private let blockTokenCountValue: Int;
    private let tensorCountValue: Int;

    /// Reads and validates one full-attention key/value block header
    /// without touching the payload.
    public static func readKvBlock(
        blockFileUrl: URL,
        modelContract: PersistentPromptCacheModelContract
    ) throws -> PersistentPromptCacheBlockHeader {
        return try PersistentPromptCacheBlockHeader.readFromKind(
            blockFileUrl: blockFileUrl,
            fileKind: .sequenceStateBlock,
            modelContract: modelContract);
    }

    /// Reads and validates one GatedDeltaNet recurrent snapshot header
    /// without touching the payload.
    public static func readRecurrentSnapshot(
        snapshotFileUrl: URL,
        modelContract: PersistentPromptCacheModelContract
    ) throws -> PersistentPromptCacheBlockHeader {
        return try PersistentPromptCacheBlockHeader.readFromKind(
            blockFileUrl: snapshotFileUrl,
            fileKind: .boundaryStateSnapshot,
            modelContract: modelContract);
    }

    private init(
        formatVersion: String,
        storageContractFingerprint: String,
        blockTokenCount: Int,
        tensorCount: Int
    ) {
        self.formatVersionValue = formatVersion;
        self.storageContractFingerprintValue = storageContractFingerprint;
        self.blockTokenCountValue = blockTokenCount;
        self.tensorCountValue = tensorCount;
    }

    /// The on-disk format version string.
    public var formatVersion: String {
        return self.formatVersionValue;
    }

    /// The number of prompt tokens captured by this block.
    public var blockTokenCount: Int {
        return self.blockTokenCountValue;
    }

    /// The exact model storage-contract fingerprint stamped into the file.
    public var storageContractFingerprint: String {
        return self.storageContractFingerprintValue;
    }

    /// The number of tensors declared in the block header.
    public var tensorCount: Int {
        return self.tensorCountValue;
    }

    private enum FileKind {
        case sequenceStateBlock;
        case boundaryStateSnapshot;
    }

    private static func readFromKind(
        blockFileUrl: URL,
        fileKind: FileKind,
        modelContract: PersistentPromptCacheModelContract
    ) throws -> PersistentPromptCacheBlockHeader {
        // Read and validate only bounded header bytes. Startup scanning must
        // not deserialize every tensor payload just to identify usable SSD
        // entries.
        let parsedHeader: PersistentSafetensorsHeader;
        do {
            parsedHeader = try PersistentSafetensorsHeader.read(fileUrl: blockFileUrl);
        } catch let headerError as PersistentSafetensorsHeaderError {
            throw PersistentPromptCacheBlockHeader.blockError(
                fromHeaderError: headerError);
        }
        let metadata: RequiredMetadata = try PersistentPromptCacheBlockHeader
            .extractRequiredMetadata(
                metadata: parsedHeader.metadata, blockPath: blockFileUrl.path);
        // Model identity and format validation occur before tensor checks so
        // a directory shared with another model revision remains harmless.
        try PersistentPromptCacheBlockHeader.validateMetadata(
            metadata: metadata,
            blockPath: blockFileUrl.path,
            modelContract: modelContract);
        try PersistentPromptCacheBlockHeader.validateTensorLayout(
            tensorViewsByName: parsedHeader.tensorViewsByName,
            blockTokenCount: metadata.blockTokenCount,
            fileKind: fileKind,
            blockPath: blockFileUrl.path,
            modelContract: modelContract);
        try PersistentPromptCacheBlockHeader.validateTensorOffsets(
            tensorViewsByName: parsedHeader.tensorViewsByName,
            dataSectionStartBytes: parsedHeader.dataSectionStartBytes,
            fileSizeBytes: parsedHeader.fileSizeBytes,
            blockPath: blockFileUrl.path);
        return PersistentPromptCacheBlockHeader(
            formatVersion: metadata.formatVersion,
            storageContractFingerprint: metadata.storageContractFingerprint,
            blockTokenCount: metadata.blockTokenCount,
            tensorCount: parsedHeader.tensorViewsByName.count);
    }

    private struct RequiredMetadata {
        var formatVersion: String;
        var storageContractFingerprint: String;
        var blockTokenCount: Int;
    }

    private static func extractRequiredMetadata(
        metadata: [String: String], blockPath: String
    ) throws -> RequiredMetadata {
        guard let formatVersion: String = metadata["format_version"] else {
            throw PersistentPromptCacheBlockError.missingMetadata(
                blockPath: blockPath, fieldName: "format_version");
        }
        guard let storageContractFingerprint: String = metadata["storage_contract_fingerprint"]
        else {
            throw PersistentPromptCacheBlockError.missingMetadata(
                blockPath: blockPath, fieldName: "storage_contract_fingerprint");
        }
        guard let blockTokenCountText: String = metadata["block_token_count"] else {
            throw PersistentPromptCacheBlockError.missingMetadata(
                blockPath: blockPath, fieldName: "block_token_count");
        }
        guard let blockTokenCount: Int = Int(blockTokenCountText) else {
            throw PersistentPromptCacheBlockError.invalidMetadata(
                blockPath: blockPath,
                fieldName: "block_token_count",
                problem: "\(blockTokenCountText) is not a token count");
        }
        return RequiredMetadata(
            formatVersion: formatVersion,
            storageContractFingerprint: storageContractFingerprint,
            blockTokenCount: blockTokenCount);
    }

    private static func validateMetadata(
        metadata: RequiredMetadata,
        blockPath: String,
        modelContract: PersistentPromptCacheModelContract
    ) throws {
        // A content hash proves prompt ancestry, not tensor compatibility.
        // The storage-contract fingerprint binds model identity, revision,
        // tensor geometry, and execution dtype.
        if metadata.formatVersion != PersistentPromptCacheBlockHeader.FORMAT_VERSION {
            throw PersistentPromptCacheBlockError.unsupportedFormatVersion(
                actualFormatVersion: metadata.formatVersion,
                expectedFormatVersion: PersistentPromptCacheBlockHeader.FORMAT_VERSION);
        }
        // Partial blocks (prefix-cache tails) store fewer tokens than a full
        // block while remaining valid restorable state, so the header only
        // has to prove the count fits the contract's block geometry. Tensor
        // layouts are validated against this same header count, keeping each
        // file self-consistent.
        if metadata.blockTokenCount == 0
            || metadata.blockTokenCount > modelContract.blockTokenCount {
            throw PersistentPromptCacheBlockError.blockTokenCountMismatch(
                actualBlockTokenCount: metadata.blockTokenCount,
                expectedBlockTokenCount: modelContract.blockTokenCount);
        }
        if metadata.storageContractFingerprint != modelContract.storageContractFingerprintHex() {
            throw PersistentPromptCacheBlockError.invalidModelSpecificArtifact(
                blockPath: blockPath,
                description: "persistent model-state storage contract fingerprint does not match");
        }
    }

    private static func validateTensorLayout(
        tensorViewsByName: [String: SafetensorsFraming.TensorView],
        blockTokenCount: Int,
        fileKind: FileKind,
        blockPath: String,
        modelContract: PersistentPromptCacheModelContract
    ) throws {
        // Each file kind has a closed layout contract. Be deliberately
        // strict: a plausible-looking subset could produce silent wrong
        // generation rather than a typed request failure.
        let expectedTensorLayouts: [DecoderCachePersistedTensorLayout];
        switch fileKind {
        case .sequenceStateBlock:
            expectedTensorLayouts = modelContract.decoderCacheLayout.sequenceTensorLayouts();
        case .boundaryStateSnapshot:
            expectedTensorLayouts = modelContract.decoderCacheLayout.boundaryTensorLayouts();
        }
        let expectedTensorCount: Int = expectedTensorLayouts.count;
        for expectedTensorLayout: DecoderCachePersistedTensorLayout in expectedTensorLayouts {
            try PersistentPromptCacheBlockHeader.validateExpectedTensorLayout(
                tensorViewsByName: tensorViewsByName,
                blockTokenCount: blockTokenCount,
                expectedTensorLayout: expectedTensorLayout,
                blockPath: blockPath);
        }
        if tensorViewsByName.count != expectedTensorCount {
            throw PersistentPromptCacheBlockError.unexpectedTensorCount(
                blockPath: blockPath,
                expectedTensorCount: expectedTensorCount,
                actualTensorCount: tensorViewsByName.count);
        }
    }

    private static func validateExpectedTensorLayout(
        tensorViewsByName: [String: SafetensorsFraming.TensorView],
        blockTokenCount: Int,
        expectedTensorLayout: DecoderCachePersistedTensorLayout,
        blockPath: String
    ) throws {
        let tensorName: String = expectedTensorLayout.persistentTensorName;
        guard let tensorView: SafetensorsFraming.TensorView = tensorViewsByName[tensorName] else {
            throw PersistentPromptCacheBlockError.missingTensor(
                blockPath: blockPath, tensorName: tensorName);
        }
        let expectedDtype: String = expectedTensorLayout.tensorLayout.dtype.safetensorsDtypeName;
        if tensorView.dtype != expectedDtype {
            throw PersistentPromptCacheBlockError.tensorDtypeMismatch(
                blockPath: blockPath,
                tensorName: tensorName,
                expectedDtype: expectedDtype,
                actualDtype: tensorView.dtype);
        }
        let expectedShape: [Int] = expectedTensorLayout.tensorLayout.dimensions.enumerated()
            .map({ (dimensionEntry: (offset: Int, element: Int)) -> Int in
                if dimensionEntry.offset == expectedTensorLayout.tensorLayout.sequenceAxis {
                    return blockTokenCount;
                }
                return dimensionEntry.element;
            });
        if tensorView.shape != expectedShape {
            throw PersistentPromptCacheBlockError.tensorShapeMismatch(
                blockPath: blockPath,
                tensorName: tensorName,
                expectedShape: expectedShape,
                actualShape: tensorView.shape);
        }
    }

    private static func validateTensorOffsets(
        tensorViewsByName: [String: SafetensorsFraming.TensorView],
        dataSectionStartBytes: UInt64,
        fileSizeBytes: UInt64,
        blockPath: String
    ) throws {
        // Header shape validation alone is insufficient: data offsets are
        // also untrusted and must remain inside the real file before any
        // payload reader opens them.
        for tensorEntry: (key: String, value: SafetensorsFraming.TensorView)
            in tensorViewsByName.sorted(by: { (left, right) -> Bool in
                return left.key < right.key;
            }) {
            let tensorName: String = tensorEntry.key;
            let startOffset: UInt64 = tensorEntry.value.dataStartOffset();
            let endOffset: UInt64 = tensorEntry.value.dataEndOffset();
            if startOffset > endOffset {
                throw PersistentPromptCacheBlockError.invalidDataOffsets(
                    blockPath: blockPath,
                    tensorName: tensorName,
                    startOffset: startOffset,
                    endOffset: endOffset);
            }
            let scalarByteCount: UInt64;
            switch tensorEntry.value.dtype {
            case "F16", "BF16": scalarByteCount = 2;
            case "F32": scalarByteCount = 4;
            default: scalarByteCount = 0;
            }
            var expectedPayloadByteCount: UInt64 = scalarByteCount;
            var payloadOverflowed: Bool = false;
            for tensorDimension: Int in tensorEntry.value.shape {
                let (multipliedBytes, multipliedOverflow) = expectedPayloadByteCount
                    .multipliedReportingOverflow(by: UInt64(max(tensorDimension, 0)));
                if multipliedOverflow {
                    payloadOverflowed = true;
                    break;
                }
                expectedPayloadByteCount = multipliedBytes;
            }
            let actualPayloadByteCount: UInt64 = endOffset >= startOffset
                ? endOffset - startOffset : 0;
            if payloadOverflowed == false
                && actualPayloadByteCount != expectedPayloadByteCount {
                throw PersistentPromptCacheBlockError.tensorPayloadSizeMismatch(
                    blockPath: blockPath,
                    tensorName: tensorName,
                    expectedPayloadByteCount: expectedPayloadByteCount,
                    actualPayloadByteCount: actualPayloadByteCount);
            }
            let (absoluteEndOffset, absoluteOverflow) = dataSectionStartBytes
                .addingReportingOverflow(endOffset);
            if absoluteOverflow || absoluteEndOffset > fileSizeBytes {
                throw PersistentPromptCacheBlockError.offsetBeyondFile(
                    blockPath: blockPath,
                    tensorName: tensorName,
                    endOffset: endOffset,
                    fileSizeBytes: fileSizeBytes);
            }
        }
    }

    private static func blockError(
        fromHeaderError headerError: PersistentSafetensorsHeaderError
    ) -> PersistentPromptCacheBlockError {
        switch headerError {
        case let .readFileMetadata(filePath, problem):
            return .readFileMetadata(blockPath: filePath, problem: problem);
        case let .readHeaderBytes(filePath, problem):
            return .readHeaderBytes(blockPath: filePath, problem: problem);
        case let .headerLengthTooLarge(filePath, headerLengthBytes, maximumHeaderLengthBytes):
            return .headerLengthTooLarge(
                blockPath: filePath,
                headerLengthBytes: headerLengthBytes,
                maximumHeaderLengthBytes: maximumHeaderLengthBytes);
        case let .truncatedFile(filePath, expectedMinimumBytes, actualFileSizeBytes):
            return .truncatedFile(
                blockPath: filePath,
                expectedMinimumBytes: expectedMinimumBytes,
                actualFileSizeBytes: actualFileSizeBytes);
        case let .invalidHeaderJson(filePath, problem):
            return .invalidHeaderJson(blockPath: filePath, problem: problem);
        }
    }
}
