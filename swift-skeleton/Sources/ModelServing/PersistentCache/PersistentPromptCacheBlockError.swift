import Foundation;

/// One persistent model-state block could not be read or did not match its
/// storage contract, port of the Rust `PersistentPromptCacheBlockError`.
/// These errors describe an untrusted on-disk artifact rather than a model
/// inference failure: the disk store rejects and removes a bad block, then
/// lets the request use cold prompt processing.
public enum PersistentPromptCacheBlockError: Error, Equatable, Sendable {

    case readFileMetadata(blockPath: String, problem: String);

    case readHeaderBytes(blockPath: String, problem: String);

    case headerLengthTooLarge(
        blockPath: String,
        headerLengthBytes: UInt64,
        maximumHeaderLengthBytes: UInt64);

    case truncatedFile(
        blockPath: String,
        expectedMinimumBytes: UInt64,
        actualFileSizeBytes: UInt64);

    case invalidHeaderJson(blockPath: String, problem: String);

    case missingMetadata(blockPath: String, fieldName: String);

    case invalidMetadata(blockPath: String, fieldName: String, problem: String);

    case unsupportedFormatVersion(
        actualFormatVersion: String,
        expectedFormatVersion: String);

    case blockTokenCountMismatch(
        actualBlockTokenCount: Int,
        expectedBlockTokenCount: Int);

    case missingTensor(blockPath: String, tensorName: String);

    case tensorDtypeMismatch(
        blockPath: String,
        tensorName: String,
        expectedDtype: String,
        actualDtype: String);

    case tensorShapeMismatch(
        blockPath: String,
        tensorName: String,
        expectedShape: [Int],
        actualShape: [Int]);

    case tensorPayloadSizeMismatch(
        blockPath: String,
        tensorName: String,
        expectedPayloadByteCount: UInt64,
        actualPayloadByteCount: UInt64);

    case unexpectedTensorCount(
        blockPath: String,
        expectedTensorCount: Int,
        actualTensorCount: Int);

    case invalidDataOffsets(
        blockPath: String,
        tensorName: String,
        startOffset: UInt64,
        endOffset: UInt64);

    case offsetBeyondFile(
        blockPath: String,
        tensorName: String,
        endOffset: UInt64,
        fileSizeBytes: UInt64);

    case invalidModelSpecificArtifact(blockPath: String, description: String);

    public var errorDescription: String? {
        switch self {
        case let .readFileMetadata(blockPath, problem):
            return "failed to read persistent prompt-cache block metadata at \(blockPath): \(problem)";
        case let .readHeaderBytes(blockPath, problem):
            return "failed to read persistent prompt-cache block header bytes at \(blockPath): \(problem)";
        case let .headerLengthTooLarge(blockPath, headerLengthBytes, maximumHeaderLengthBytes):
            return "persistent prompt-cache block header at \(blockPath) is \(headerLengthBytes) "
                + "bytes, maximum \(maximumHeaderLengthBytes)";
        case let .truncatedFile(blockPath, expectedMinimumBytes, actualFileSizeBytes):
            return "persistent prompt-cache block at \(blockPath) is truncated: expected "
                + "\(expectedMinimumBytes), got \(actualFileSizeBytes)";
        case let .invalidHeaderJson(blockPath, problem):
            return "persistent prompt-cache block header at \(blockPath) is not valid JSON: \(problem)";
        case let .missingMetadata(blockPath, fieldName):
            return "persistent prompt-cache block at \(blockPath) is missing metadata field \(fieldName)";
        case let .invalidMetadata(blockPath, fieldName, problem):
            return "persistent prompt-cache block metadata field \(fieldName) at \(blockPath) is "
                + "invalid: \(problem)";
        case let .unsupportedFormatVersion(actualFormatVersion, expectedFormatVersion):
            return "persistent prompt-cache block format version is \(actualFormatVersion), "
                + "expected \(expectedFormatVersion)";
        case let .blockTokenCountMismatch(actualBlockTokenCount, expectedBlockTokenCount):
            return "persistent prompt-cache block token count is \(actualBlockTokenCount), "
                + "expected \(expectedBlockTokenCount)";
        case let .missingTensor(blockPath, tensorName):
            return "persistent prompt-cache block at \(blockPath) is missing tensor \(tensorName)";
        case let .tensorDtypeMismatch(blockPath, tensorName, expectedDtype, actualDtype):
            return "persistent prompt-cache block tensor \(tensorName) at \(blockPath) has dtype "
                + "\(actualDtype), expected \(expectedDtype)";
        case let .tensorShapeMismatch(blockPath, tensorName, expectedShape, actualShape):
            return "persistent prompt-cache block tensor \(tensorName) at \(blockPath) has shape "
                + "\(actualShape), expected \(expectedShape)";
        case let .tensorPayloadSizeMismatch(blockPath, tensorName, expectedPayloadByteCount, actualPayloadByteCount):
            return "persistent prompt-cache block tensor \(tensorName) at \(blockPath) declares "
                + "\(actualPayloadByteCount) payload bytes, expected \(expectedPayloadByteCount)";
        case let .unexpectedTensorCount(blockPath, expectedTensorCount, actualTensorCount):
            return "persistent prompt-cache block at \(blockPath) has \(actualTensorCount) "
                + "tensors, expected \(expectedTensorCount)";
        case let .invalidDataOffsets(blockPath, tensorName, startOffset, endOffset):
            return "persistent prompt-cache block tensor \(tensorName) at \(blockPath) has "
                + "invalid data offsets [\(startOffset), \(endOffset)]";
        case let .offsetBeyondFile(blockPath, tensorName, endOffset, fileSizeBytes):
            return "persistent prompt-cache block tensor \(tensorName) at \(blockPath) ends at "
                + "\(endOffset) bytes, beyond the \(fileSizeBytes) byte file";
        case let .invalidModelSpecificArtifact(blockPath, description):
            return "persistent prompt-cache model-specific artifact at \(blockPath) is invalid: "
                + "\(description)";
        }
    }
}
