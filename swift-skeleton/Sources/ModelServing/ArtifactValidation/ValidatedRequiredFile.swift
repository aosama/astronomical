import Foundation;
import Darwin;

/// POSIX identity that detects a validated file being replaced between
/// validation and later reads. Port of
/// crates/model-serving/src/artifact_validation/validated_artifact.rs.
public struct ValidatedFileIdentity: Equatable, Sendable {
    let deviceId: UInt64;
    let inode: UInt64;
    let sizeBytes: UInt64;
    let modifiedSeconds: Int64;
    let modifiedNanoseconds: Int64;

    /// Reads the identity straight from the retained descriptor via fstat;
    /// nil when the descriptor can no longer be inspected.
    public static func statIdentity(fileDescriptor: Int32) -> ValidatedFileIdentity? {
        guard let statusBuffer: stat = RequiredFiles.fstatStatus(fileDescriptor: fileDescriptor) else {
            return nil;
        }
        return ValidatedFileIdentity.fromStat(statusBuffer);
    }

    static func fromStat(_ statusBuffer: stat) -> ValidatedFileIdentity {
        return ValidatedFileIdentity(
            deviceId: UInt64(statusBuffer.st_dev),
            inode: statusBuffer.st_ino,
            sizeBytes: UInt64(statusBuffer.st_size),
            modifiedSeconds: Int64(statusBuffer.st_mtimespec.tv_sec),
            modifiedNanoseconds: Int64(statusBuffer.st_mtimespec.tv_nsec));
    }
}

/// An open required-file descriptor whose identity was checked during validation.
public final class ValidatedRequiredFile {
    private let retainedFileHandle: FileHandle;
    private let checkedFileIdentity: ValidatedFileIdentity;
    private let validatedFileName: String;
    private let inspectedSizeBytes: UInt64;
    private let retainedCapturedBytes: Data?;

    public init(
        fileHandle: FileHandle, fileIdentity: ValidatedFileIdentity, fileName: String,
        sizeBytes: UInt64, capturedBytes: Data?) {
        self.retainedFileHandle = fileHandle;
        self.checkedFileIdentity = fileIdentity;
        self.validatedFileName = fileName;
        self.inspectedSizeBytes = sizeBytes;
        self.retainedCapturedBytes = capturedBytes;
    }

    public var fileName: String {
        return self.validatedFileName;
    }

    public var sizeBytes: UInt64 {
        return self.inspectedSizeBytes;
    }

    public var capturedBytes: Data? {
        return self.retainedCapturedBytes;
    }

    public var fileHandle: FileHandle {
        return self.retainedFileHandle;
    }

    public var fileIdentity: ValidatedFileIdentity {
        return self.checkedFileIdentity;
    }

    /// Re-checks the descriptor identity before the validated file leaves the
    /// validation layer, mirroring the Rust `into_validated_weights_file`.
    public func intoValidatedWeightsFile() throws -> ValidatedWeightsFile {
        guard let finalIdentity: ValidatedFileIdentity =
            ValidatedFileIdentity.statIdentity(fileDescriptor: self.retainedFileHandle.fileDescriptor) else {
            throw ArtifactValidationError.inspectRequiredFile(
                fileName: self.validatedFileName,
                problem: "the validated file descriptor can no longer be inspected");
        }
        if finalIdentity != self.checkedFileIdentity {
            throw ArtifactValidationError.validatedFileIdentityChanged(
                fileName: self.validatedFileName);
        }
        return ValidatedWeightsFile(validatedRequiredFile: self);
    }
}

/// A validated model-weight descriptor ready for ownership transfer to MLX.
public final class ValidatedWeightsFile {
    private let retainedValidatedRequiredFile: ValidatedRequiredFile;

    public init(validatedRequiredFile: ValidatedRequiredFile) {
        self.retainedValidatedRequiredFile = validatedRequiredFile;
    }

    /// Transfers the validated read-only descriptor to its runtime owner.
    public func intoFile() -> FileHandle {
        return self.retainedValidatedRequiredFile.fileHandle;
    }

    /// Returns the byte length of the validated file.
    public var sizeBytes: UInt64 {
        return self.retainedValidatedRequiredFile.sizeBytes;
    }

    /// Returns the retained validated descriptor for bounded reads.
    public var validatedRequiredFile: ValidatedRequiredFile {
        return self.retainedValidatedRequiredFile;
    }

    /// Returns a strict family-neutral inventory from this retained descriptor.
    public func readRawSafetensorsInventory() throws -> RawSafetensorsInventory {
        return try RawSafetensorsInventoryReader.read(
            validatedRequiredFile: self.retainedValidatedRequiredFile);
    }

    /// Test seam mirroring the Rust raw-inventory projection: the production
    /// reader produces the same data, exposed without crate-private metadata.
    public func readRawSafetensorsInventoryForTests() throws -> RawSafetensorsInventory {
        return try self.readRawSafetensorsInventory();
    }

    /// Test seam for the shared bounded retained-descriptor reader, keeping
    /// tests on the production reader so path-retention and typed-error
    /// behavior are exercised together.
    public func readBoundedBytesForTests(maximumSizeBytes: UInt64) throws -> Data {
        return try RequiredFiles.readBoundedRequiredFileBytes(
            requiredFile: self.retainedValidatedRequiredFile, maximumSizeBytes: maximumSizeBytes);
    }
}
