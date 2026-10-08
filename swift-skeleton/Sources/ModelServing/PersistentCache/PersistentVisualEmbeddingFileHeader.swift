import Foundation;

/// Closed, bounded validation for persisted Qwen3.5-MoE projected visual
/// embeddings, port of the Rust `PersistentVisualEmbeddingFileHeader`. The
/// filename digest must re-derive from the recorded image digest, model
/// identity, and format version, so a renamed or cross-model file can never
/// load.
public struct PersistentVisualEmbeddingFileHeader: Equatable, Sendable {

    static let TENSOR_NAME: String = "visual_embeddings";

    static let BF16_SCALAR_BYTE_COUNT: Int = 2;

    public let formatVersion: String;

    public let modelId: String;

    public let modelRevision: String;

    public let encodedImageSha256: Data;

    public let visualTokenCount: Int;

    /// Reads and validates one visual embedding header without loading its
    /// payload.
    public static func readFromFile(
        visualEmbeddingFileUrl: URL,
        modelContract: PersistentVisualEmbeddingModelContract
    ) throws -> PersistentVisualEmbeddingFileHeader {
        let visualEmbeddingFilePath: String = visualEmbeddingFileUrl.path;
        let parsedHeader: PersistentSafetensorsHeader;
        do {
            parsedHeader = try PersistentSafetensorsHeader.read(
                fileUrl: visualEmbeddingFileUrl);
        } catch let headerError as PersistentSafetensorsHeaderError {
            throw PersistentVisualEmbeddingFileError.headerRead(
                visualEmbeddingFilePath: visualEmbeddingFilePath,
                problem: String(describing: headerError));
        }
        let formatVersion: String = try Self.requiredMetadata(
            metadata: parsedHeader.metadata, fieldName: "format_version",
            visualEmbeddingFilePath: visualEmbeddingFilePath);
        let modelId: String = try Self.requiredMetadata(
            metadata: parsedHeader.metadata, fieldName: "model_id",
            visualEmbeddingFilePath: visualEmbeddingFilePath);
        let modelRevision: String = try Self.requiredMetadata(
            metadata: parsedHeader.metadata, fieldName: "model_revision",
            visualEmbeddingFilePath: visualEmbeddingFilePath);
        let encodedImageSha256Text: String = try Self.requiredMetadata(
            metadata: parsedHeader.metadata, fieldName: "encoded_image_sha256",
            visualEmbeddingFilePath: visualEmbeddingFilePath);
        let encodedImageSha256: Data = try Self.decodeLowercaseSha256(
            digestText: encodedImageSha256Text,
            visualEmbeddingFilePath: visualEmbeddingFilePath);
        let visualTokenCountText: String = try Self.requiredMetadata(
            metadata: parsedHeader.metadata, fieldName: "visual_token_count",
            visualEmbeddingFilePath: visualEmbeddingFilePath);
        guard let visualTokenCount: Int = Int(visualTokenCountText)
        else {
            throw PersistentVisualEmbeddingFileError.invalidMetadata(
                visualEmbeddingFilePath: visualEmbeddingFilePath,
                fieldName: "visual_token_count");
        }
        try Self.validateMetadata(
            formatVersion: formatVersion, modelId: modelId, modelRevision: modelRevision,
            visualTokenCount: visualTokenCount,
            visualEmbeddingFilePath: visualEmbeddingFilePath, modelContract: modelContract);
        try Self.validateFilenameDigest(
            visualEmbeddingFilePath: visualEmbeddingFilePath,
            encodedImageSha256: encodedImageSha256, modelContract: modelContract);
        try Self.validateTensorLayout(
            parsedHeader: parsedHeader, visualTokenCount: visualTokenCount,
            visualEmbeddingFilePath: visualEmbeddingFilePath, modelContract: modelContract);
        return PersistentVisualEmbeddingFileHeader(
            formatVersion: formatVersion, modelId: modelId, modelRevision: modelRevision,
            encodedImageSha256: encodedImageSha256, visualTokenCount: visualTokenCount);
    }

    private static func requiredMetadata(
        metadata: [String: String],
        fieldName: String,
        visualEmbeddingFilePath: String
    ) throws -> String {
        guard let metadataValue: String = metadata[fieldName]
        else {
            throw PersistentVisualEmbeddingFileError.missingMetadata(
                visualEmbeddingFilePath: visualEmbeddingFilePath, fieldName: fieldName);
        }
        return metadataValue;
    }

    private static func validateMetadata(
        formatVersion: String,
        modelId: String,
        modelRevision: String,
        visualTokenCount: Int,
        visualEmbeddingFilePath: String,
        modelContract: PersistentVisualEmbeddingModelContract
    ) throws {
        if formatVersion != PersistentVisualEmbeddingKey.FORMAT_VERSION {
            throw PersistentVisualEmbeddingFileError.unsupportedFormatVersion(
                visualEmbeddingFilePath: visualEmbeddingFilePath,
                actualFormatVersion: formatVersion,
                expectedFormatVersion: PersistentVisualEmbeddingKey.FORMAT_VERSION);
        }
        if modelId != modelContract.modelId {
            throw PersistentVisualEmbeddingFileError.foreignModel(
                visualEmbeddingFilePath: visualEmbeddingFilePath, actualModelId: modelId);
        }
        if modelRevision != modelContract.modelRevision {
            throw PersistentVisualEmbeddingFileError.foreignModelRevision(
                visualEmbeddingFilePath: visualEmbeddingFilePath,
                actualModelRevision: modelRevision);
        }
        if visualTokenCount == 0
            || visualTokenCount > modelContract.maximumVisualTokenCount {
            throw PersistentVisualEmbeddingFileError.visualTokenCountOutOfRange(
                visualEmbeddingFilePath: visualEmbeddingFilePath,
                actualVisualTokenCount: visualTokenCount,
                maximumVisualTokenCount: modelContract.maximumVisualTokenCount);
        }
    }

    private static func validateFilenameDigest(
        visualEmbeddingFilePath: String,
        encodedImageSha256: Data,
        modelContract: PersistentVisualEmbeddingModelContract
    ) throws {
        let expectedVisualEmbeddingHash: Data = PersistentVisualEmbeddingKey.forImage(
            encodedImageSha256: encodedImageSha256,
            modelId: modelContract.modelId,
            modelRevision: modelContract.modelRevision).visualEmbeddingHash;
        let fileBaseName: String = URL(fileURLWithPath: visualEmbeddingFilePath)
            .deletingPathExtension().lastPathComponent;
        guard let actualVisualEmbeddingHash: Data =
            PersistentPromptCacheStoreFile.parseBlockHashHex(fileBaseName)
        else {
            throw PersistentVisualEmbeddingFileError.invalidFileName(
                visualEmbeddingFilePath: visualEmbeddingFilePath);
        }
        if actualVisualEmbeddingHash != expectedVisualEmbeddingHash {
            throw PersistentVisualEmbeddingFileError.fileNameDigestMismatch(
                visualEmbeddingFilePath: visualEmbeddingFilePath);
        }
    }

    private static func validateTensorLayout(
        parsedHeader: PersistentSafetensorsHeader,
        visualTokenCount: Int,
        visualEmbeddingFilePath: String,
        modelContract: PersistentVisualEmbeddingModelContract
    ) throws {
        if parsedHeader.tensorViewsByName.count != 1 {
            throw PersistentVisualEmbeddingFileError.unexpectedTensorCount(
                visualEmbeddingFilePath: visualEmbeddingFilePath,
                actualTensorCount: parsedHeader.tensorViewsByName.count);
        }
        guard let visualEmbeddingTensor: SafetensorsFraming.TensorView =
            parsedHeader.tensorViewsByName[PersistentVisualEmbeddingFileHeader.TENSOR_NAME]
        else {
            throw PersistentVisualEmbeddingFileError.missingTensor(
                visualEmbeddingFilePath: visualEmbeddingFilePath);
        }
        if visualEmbeddingTensor.dtype != "BF16" {
            throw PersistentVisualEmbeddingFileError.tensorDtypeMismatch(
                visualEmbeddingFilePath: visualEmbeddingFilePath,
                actualDtype: visualEmbeddingTensor.dtype);
        }
        let expectedTensorShape: [Int] = modelContract.visualEmbeddingShape(
            visualTokenCount: visualTokenCount);
        if visualEmbeddingTensor.shape != expectedTensorShape {
            throw PersistentVisualEmbeddingFileError.tensorShapeMismatch(
                visualEmbeddingFilePath: visualEmbeddingFilePath,
                expectedShape: expectedTensorShape,
                actualShape: visualEmbeddingTensor.shape);
        }
        let expectedPayloadElements: Int = visualTokenCount
            * modelContract.visualEmbeddingHiddenSize;
        let multipliedElements: (partialValue: Int, overflow: Bool) = expectedPayloadElements
            .multipliedReportingOverflow(by: PersistentVisualEmbeddingFileHeader
                .BF16_SCALAR_BYTE_COUNT);
        if multipliedElements.overflow {
            throw PersistentVisualEmbeddingFileError.payloadSizeOverflow(
                visualEmbeddingFilePath: visualEmbeddingFilePath);
        }
        let expectedPayloadByteCount: UInt64 = UInt64(multipliedElements.partialValue);
        let payloadStartBytes: UInt64 = visualEmbeddingTensor.dataStartOffset();
        let payloadEndBytes: UInt64 = visualEmbeddingTensor.dataEndOffset();
        if payloadStartBytes != 0 || payloadEndBytes != expectedPayloadByteCount {
            throw PersistentVisualEmbeddingFileError.invalidTensorDataOffsets(
                visualEmbeddingFilePath: visualEmbeddingFilePath,
                payloadStartBytes: payloadStartBytes,
                payloadEndBytes: payloadEndBytes,
                expectedPayloadByteCount: expectedPayloadByteCount);
        }
        let absolutePayloadEndBytes: (partialValue: UInt64, overflow: Bool) = parsedHeader
            .dataSectionStartBytes.addingReportingOverflow(payloadEndBytes);
        if absolutePayloadEndBytes.overflow
            || absolutePayloadEndBytes.partialValue > parsedHeader.fileSizeBytes {
            throw PersistentVisualEmbeddingFileError.payloadBeyondFile(
                visualEmbeddingFilePath: visualEmbeddingFilePath,
                payloadEndBytes: payloadEndBytes,
                fileSizeBytes: parsedHeader.fileSizeBytes);
        }
    }

    private static func decodeLowercaseSha256(
        digestText: String,
        visualEmbeddingFilePath: String
    ) throws -> Data {
        if digestText.count != 64 || digestText != digestText.lowercased() {
            throw PersistentVisualEmbeddingFileError.invalidImageDigest(
                visualEmbeddingFilePath: visualEmbeddingFilePath);
        }
        let hexScalars: [Character] = Array(digestText);
        var decodedDigest: [UInt8] = [UInt8](repeating: 0, count: 32);
        for digestByteIndex: Int in 0..<32 {
            let hexPair: String = String(
                hexScalars[(digestByteIndex * 2)..<(digestByteIndex * 2 + 2)]);
            guard let decodedByte: UInt8 = UInt8(hexPair, radix: 16)
            else {
                throw PersistentVisualEmbeddingFileError.invalidImageDigest(
                    visualEmbeddingFilePath: visualEmbeddingFilePath);
            }
            decodedDigest[digestByteIndex] = decodedByte;
        }
        return Data(decodedDigest);
    }
}
