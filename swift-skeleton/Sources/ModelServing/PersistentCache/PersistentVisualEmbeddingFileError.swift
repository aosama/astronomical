import Foundation;

/// One bounded failure while validating a persisted visual-embedding file,
/// port of the Rust `PersistentVisualEmbeddingFileError`.
public enum PersistentVisualEmbeddingFileError: Error, Equatable {

    case headerRead(visualEmbeddingFilePath: String, problem: String);

    case missingMetadata(visualEmbeddingFilePath: String, fieldName: String);

    case invalidMetadata(visualEmbeddingFilePath: String, fieldName: String);

    case unsupportedFormatVersion(
        visualEmbeddingFilePath: String,
        actualFormatVersion: String,
        expectedFormatVersion: String);

    case foreignModel(visualEmbeddingFilePath: String, actualModelId: String);

    case foreignModelRevision(visualEmbeddingFilePath: String, actualModelRevision: String);

    case visualTokenCountOutOfRange(
        visualEmbeddingFilePath: String,
        actualVisualTokenCount: Int,
        maximumVisualTokenCount: Int);

    case invalidFileName(visualEmbeddingFilePath: String);

    case fileNameDigestMismatch(visualEmbeddingFilePath: String);

    case unexpectedTensorCount(visualEmbeddingFilePath: String, actualTensorCount: Int);

    case missingTensor(visualEmbeddingFilePath: String);

    case tensorDtypeMismatch(visualEmbeddingFilePath: String, actualDtype: String);

    case tensorShapeMismatch(
        visualEmbeddingFilePath: String,
        expectedShape: [Int],
        actualShape: [Int]);

    case payloadSizeOverflow(visualEmbeddingFilePath: String);

    case invalidTensorDataOffsets(
        visualEmbeddingFilePath: String,
        payloadStartBytes: UInt64,
        payloadEndBytes: UInt64,
        expectedPayloadByteCount: UInt64);

    case payloadBeyondFile(
        visualEmbeddingFilePath: String,
        payloadEndBytes: UInt64,
        fileSizeBytes: UInt64);

    case invalidImageDigest(visualEmbeddingFilePath: String);
}
