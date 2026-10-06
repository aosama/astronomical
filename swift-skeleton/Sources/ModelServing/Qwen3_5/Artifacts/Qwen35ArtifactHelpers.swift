import Foundation;

/// Small shared helpers for Qwen artifact validation: required-file profile
/// construction, captured byte access, and bounded structural reads. Port of
/// crates/model-serving/src/qwen3_5/artifacts/artifact_helpers.rs.
public enum Qwen35ArtifactHelpers {

    /// A required file whose size is unchecked at the profile level; the
    /// validator enforces actual sizes against manifest expectations.
    public static func requiredFile(fileName: String) -> RequiredFileProfile {
        return RequiredFileProfile(fileName: fileName, sizeBytes: 0);
    }

    /// Captured bytes for a required file retained during validation; runtime
    /// loading never reopens the mutable pathname for captured files.
    public static func capturedRequiredFileBytes(
        requiredFiles: Dictionary<String, ValidatedRequiredFile>,
        fileName: String) throws -> Data {
        guard let capturedBytes: Data = requiredFiles[fileName]?.capturedBytes else {
            throw ArtifactValidationError.profileMissingRequiredFile(fileName: fileName);
        }
        return capturedBytes;
    }

    /// Bounded structural read of a required file through its retained
    /// descriptor, capped at the index byte bound.
    public static func readRequiredFileBytes(
        requiredFile: ValidatedRequiredFile) throws -> Data {
        let maximumIndexBytes: UInt64 = UInt64(Qwen3_5ShardIndex.MAXIMUM_INDEX_BYTES);
        guard let fileSize: Int = Int(exactly: requiredFile.sizeBytes) else {
            throw ArtifactValidationError.capturedRequiredFileTooLarge(
                fileName: requiredFile.fileName,
                actualSizeBytes: requiredFile.sizeBytes,
                maximumSizeBytes: maximumIndexBytes);
        }
        if fileSize > Qwen3_5ShardIndex.MAXIMUM_INDEX_BYTES {
            throw ArtifactValidationError.capturedRequiredFileTooLarge(
                fileName: requiredFile.fileName,
                actualSizeBytes: requiredFile.sizeBytes,
                maximumSizeBytes: maximumIndexBytes);
        }
        guard let fileBytes: Data = try RequiredFiles.positionedReadExactly(
            fileDescriptor: requiredFile.fileHandle.fileDescriptor,
            startOffsetBytes: 0,
            byteCount: fileSize) else {
            throw ArtifactValidationError.readRequiredFileForStructuralValidation(
                fileName: requiredFile.fileName,
                problem: "validated required file ended before its expected size");
        }
        return fileBytes;
    }
}
