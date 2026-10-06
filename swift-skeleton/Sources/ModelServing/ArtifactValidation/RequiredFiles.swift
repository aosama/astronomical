import Foundation;
import Darwin;

/// Required-file discovery, validation, and bounded capture. Port of
/// crates/model-serving/src/artifact_validation/required_files.rs.
public enum RequiredFiles {

    private static let CONFIG_FILE_NAME: String = "config.json";
    private static let TOKENIZER_FILE_NAME: String = "tokenizer.json";
    /// Maximum file size for which we capture bytes during validation so runtime
    /// loading never reopens mutable paths. The Qwen3.5-MoE tokenizer is about
    /// 20 MB; this bound covers it and config.json with fixed headroom while
    /// preventing unbounded trust-boundary reads.
    private static let MAXIMUM_CAPTURED_FILE_SIZE_BYTES: UInt64 = 32 * 1024 * 1024;

    /// Validates every profile entry and returns the retained descriptors by
    /// file name. A profile name owns exactly one retained descriptor, so a
    /// repeat is rejected before validating its conflicting metadata.
    public static func validateRequiredFiles(
        modelDirectory: String,
        requiredFileProfiles: Array<RequiredFileProfile>) throws -> Dictionary<String, ValidatedRequiredFile> {
        var requiredFilesByFileName: Dictionary<String, ValidatedRequiredFile> = Dictionary(
            minimumCapacity: requiredFileProfiles.count);
        for requiredFileProfile: RequiredFileProfile in requiredFileProfiles {
            if requiredFilesByFileName[requiredFileProfile.fileName] != nil {
                throw ArtifactValidationError.duplicateProfileFileName(
                    fileName: requiredFileProfile.fileName);
            }
            let validatedFile: ValidatedRequiredFile = try RequiredFiles.validateRequiredFile(
                modelDirectory: modelDirectory, requiredFileProfile: requiredFileProfile);
            requiredFilesByFileName[requiredFileProfile.fileName] = validatedFile;
        }
        return requiredFilesByFileName;
    }

    /// Reads the validated file's full byte content within the caller's limit
    /// through the retained descriptor, never reopening the artifact pathname.
    public static func readBoundedRequiredFileBytes(
        requiredFile: ValidatedRequiredFile, maximumSizeBytes: UInt64) throws -> Data {
        let actualSizeBytes: UInt64 = requiredFile.sizeBytes;
        if actualSizeBytes > maximumSizeBytes {
            throw ArtifactValidationError.boundedRequiredFileTooLarge(
                fileName: requiredFile.fileName,
                actualSizeBytes: actualSizeBytes,
                maximumSizeBytes: maximumSizeBytes);
        }
        guard let requestedByteCount: Int = Int(exactly: actualSizeBytes) else {
            throw ArtifactValidationError.readBoundedRequiredFile(
                fileName: requiredFile.fileName,
                problem: "the validated file size does not fit the platform read length");
        }
        // The bound check precedes the read, so this positioned exact read pulls
        // at most `maximumSizeBytes`; a short read means the validated file ended
        // before its inspected size.
        guard let requiredFileBytes: Data = try RequiredFiles.positionedReadExactly(
            fileDescriptor: requiredFile.fileHandle.fileDescriptor,
            startOffsetBytes: 0,
            byteCount: requestedByteCount) else {
            throw ArtifactValidationError.readBoundedRequiredFile(
                fileName: requiredFile.fileName, problem: "failed to fill whole buffer");
        }
        return requiredFileBytes;
    }

    /// Test seam for required-file path validation without exposing retained
    /// internal metadata.
    public static func validateRequiredFileForTests(
        modelDirectory: String, requiredFileProfile: RequiredFileProfile) throws -> ValidatedWeightsFile {
        let validatedFile: ValidatedRequiredFile = try RequiredFiles.validateRequiredFile(
            modelDirectory: modelDirectory, requiredFileProfile: requiredFileProfile);
        return try validatedFile.intoValidatedWeightsFile();
    }

    public static func validateRequiredFile(
        modelDirectory: String, requiredFileProfile: RequiredFileProfile) throws -> ValidatedRequiredFile {
        try RequiredFiles.validateRequiredFileRelativePath(requiredFileProfile.fileName);
        let requiredFilePath: String = SharedBlobResolution.joinPath(
            modelDirectory, requiredFileProfile.fileName);
        guard let pathMetadata: [FileAttributeKey: Any] =
            SharedBlobResolution.inspectedAttributes(atPath: requiredFilePath) else {
            throw ArtifactValidationError.inspectRequiredFile(
                fileName: requiredFileProfile.fileName,
                problem: "failed to inspect the required file path");
        }

        // Ordinary model directories remain symlink-free. Hugging Face snapshots
        // are the one safe exception because their files intentionally point into
        // the sibling immutable `blobs/` directory.
        let requiredFileOpenPath: String;
        if SharedBlobResolution.attributesDescribeSymlink(pathMetadata) {
            guard let resolvedSnapshotBlobPath: String =
                try RequiredFiles.resolveHuggingFaceSnapshotBlobPath(
                    modelDirectory: modelDirectory,
                    requiredFilePath: requiredFilePath,
                    requiredFileName: requiredFileProfile.fileName) else {
                throw ArtifactValidationError.requiredFileIsSymlink(
                    fileName: requiredFileProfile.fileName);
            }
            requiredFileOpenPath = resolvedSnapshotBlobPath;
        } else if SharedBlobResolution.attributesDescribeRegularFile(pathMetadata) {
            requiredFileOpenPath = requiredFilePath;
        } else {
            throw ArtifactValidationError.requiredFileIsNotRegular(
                fileName: requiredFileProfile.fileName);
        }

        // Open the resolved blob itself with O_NOFOLLOW. If the final path
        // changes after validation, the kernel rejects it instead of following it.
        let fileDescriptor: Int32 = Darwin.open(requiredFileOpenPath, O_RDONLY | O_NOFOLLOW);
        if fileDescriptor < 0 {
            throw ArtifactValidationError.inspectRequiredFile(
                fileName: requiredFileProfile.fileName,
                problem: String(cString: strerror(errno)));
        }
        let requiredFileHandle: FileHandle = FileHandle(
            fileDescriptor: fileDescriptor, closeOnDealloc: true);
        guard let openStatus: OpenedFileStatus =
            RequiredFiles.statOpenedFile(fileDescriptor: fileDescriptor) else {
            throw ArtifactValidationError.inspectRequiredFile(
                fileName: requiredFileProfile.fileName,
                problem: "failed to inspect the opened required file");
        }
        if openStatus.isRegularFile == false {
            throw ArtifactValidationError.requiredFileIsNotRegular(
                fileName: requiredFileProfile.fileName);
        }

        let actualSizeBytes: UInt64 = openStatus.sizeBytes;
        if requiredFileProfile.sizeBytes != 0 && actualSizeBytes != requiredFileProfile.sizeBytes {
            throw ArtifactValidationError.requiredFileSizeMismatch(
                fileName: requiredFileProfile.fileName,
                expectedSizeBytes: requiredFileProfile.sizeBytes,
                actualSizeBytes: actualSizeBytes);
        }

        let shouldCaptureBytes: Bool = requiredFileProfile.fileName == CONFIG_FILE_NAME
            || requiredFileProfile.fileName == TOKENIZER_FILE_NAME;
        if shouldCaptureBytes && actualSizeBytes > MAXIMUM_CAPTURED_FILE_SIZE_BYTES {
            throw ArtifactValidationError.capturedRequiredFileTooLarge(
                fileName: requiredFileProfile.fileName,
                actualSizeBytes: actualSizeBytes,
                maximumSizeBytes: MAXIMUM_CAPTURED_FILE_SIZE_BYTES);
        }
        let capturedBytes: Data?;
        if shouldCaptureBytes {
            capturedBytes = try RequiredFiles.captureFile(
                fileHandle: requiredFileHandle, sizeBytes: actualSizeBytes,
                fileName: requiredFileProfile.fileName);
        } else {
            capturedBytes = nil;
        }
        guard let finalStatus: OpenedFileStatus =
            RequiredFiles.statOpenedFile(fileDescriptor: fileDescriptor) else {
            throw ArtifactValidationError.readRequiredFileForCapture(
                fileName: requiredFileProfile.fileName,
                problem: "failed to re-inspect the captured file");
        }
        if finalStatus.identity != openStatus.identity {
            throw ArtifactValidationError.validatedFileIdentityChanged(
                fileName: requiredFileProfile.fileName);
        }

        return ValidatedRequiredFile(
            fileHandle: requiredFileHandle,
            fileIdentity: openStatus.identity,
            fileName: requiredFileProfile.fileName,
            sizeBytes: actualSizeBytes,
            capturedBytes: capturedBytes);
    }

    /// Model identifier derived from a Hugging Face snapshot layout, used by
    /// artifact discovery when the directory itself carries no manifest.
    static func huggingFaceSnapshotModelId(modelDirectory: String) -> String? {
        guard let modelCacheDirectory: String =
            RequiredFiles.huggingFaceModelCacheDirectory(modelDirectory: modelDirectory) else {
            return nil;
        }
        let cacheDirectoryName: String = (modelCacheDirectory as NSString).lastPathComponent;
        guard let hubRepositoryId: String =
            RequiredFiles.decodeHuggingfaceCacheDirectoryName(cacheDirectoryName) else {
            return nil;
        }
        guard let separatorIndex: String.Index = hubRepositoryId.firstIndex(of: "/") else {
            return hubRepositoryId;
        }
        return String(hubRepositoryId[hubRepositoryId.index(after: separatorIndex)...]);
    }

    // MARK: - Internal helpers

    private struct OpenedFileStatus {
        let identity: ValidatedFileIdentity;
        let isRegularFile: Bool;
        let sizeBytes: UInt64;
    }

    private static func statOpenedFile(fileDescriptor: Int32) -> OpenedFileStatus? {
        guard let statusBuffer: stat = RequiredFiles.fstatStatus(fileDescriptor: fileDescriptor) else {
            return nil;
        }
        return OpenedFileStatus(
            identity: ValidatedFileIdentity.fromStat(statusBuffer),
            isRegularFile: (UInt32(statusBuffer.st_mode) & UInt32(S_IFMT)) == UInt32(S_IFREG),
            sizeBytes: UInt64(statusBuffer.st_size));
    }

    private static func validateRequiredFileRelativePath(_ requiredFileName: String) throws -> Void {
        // Profiles describe one file inside the model directory. Reject absolute,
        // parent, current-directory, and empty components before joining.
        var isPlainRelativeName: Bool =
            requiredFileName.isEmpty == false && requiredFileName.hasPrefix("/") == false;
        if isPlainRelativeName {
            for pathComponent in requiredFileName.split(
                separator: "/", omittingEmptySubsequences: false) {
                let componentName: String = String(pathComponent);
                if componentName.isEmpty || componentName == "." || componentName == ".." {
                    isPlainRelativeName = false;
                    break;
                }
            }
        }
        if isPlainRelativeName == false {
            throw ArtifactValidationError.invalidProfileFileName(fileName: requiredFileName);
        }
    }

    /// Resolves a snapshot symlink into a permitted immutable blob path, or nil
    /// when the symlink target cannot be authenticated for this layout.
    private static func resolveHuggingFaceSnapshotBlobPath(
        modelDirectory: String, requiredFilePath: String,
        requiredFileName: String) throws -> String? {
        guard let modelCacheDirectory: String =
            RequiredFiles.huggingFaceModelCacheDirectory(modelDirectory: modelDirectory) else {
            return nil;
        }

        // Canonicalize both sides before comparison. Path text such as `../` or a
        // nested symlink cannot make an outside target appear to be inside `blobs/`.
        let blobDirectory: String = SharedBlobResolution.joinPath(modelCacheDirectory, "blobs");
        guard let blobDirectoryMetadata: [FileAttributeKey: Any] =
            SharedBlobResolution.inspectedAttributes(atPath: blobDirectory) else {
            throw ArtifactValidationError.inspectRequiredFile(
                fileName: requiredFileName,
                problem: "failed to inspect the snapshot blobs directory");
        }
        if SharedBlobResolution.attributesDescribeSymlink(blobDirectoryMetadata)
            || SharedBlobResolution.attributesDescribeDirectory(blobDirectoryMetadata) == false {
            throw ArtifactValidationError.requiredFileIsSymlink(fileName: requiredFileName);
        }
        let canonicalBlobDirectory: String = SharedBlobResolution.canonicalizedPath(blobDirectory);
        let canonicalBlobPath: String = SharedBlobResolution.canonicalizedPath(requiredFilePath);
        if SharedBlobResolution.isPath(canonicalBlobPath, withinDirectory: canonicalBlobDirectory) {
            return canonicalBlobPath;
        }

        // Newer Hugging Face caches keep large immutable objects once in the hub's
        // shared blob store and link them from each entry's own blob names, so a
        // snapshot file can legitimately resolve past this entry's `blobs/`.
        // Accept that layout only for a blob the snapshot tree record authenticates.
        let hubRootDirectory: String = (modelCacheDirectory as NSString).deletingLastPathComponent;
        if let verifiedSharedBlobPath: String =
            try SharedBlobResolution.resolveVerifiedSharedHubBlobPath(
                hubRootDirectory: hubRootDirectory,
                snapshotDirectory: modelDirectory,
                canonicalResolvedTargetPath: canonicalBlobPath,
                requiredFileName: requiredFileName) {
            return verifiedSharedBlobPath;
        }

        throw ArtifactValidationError.huggingFaceSnapshotSymlinkEscapesBlobDirectory(
            fileName: requiredFileName,
            resolvedTargetPath: canonicalBlobPath,
            expectedBlobDirectory: canonicalBlobDirectory);
    }

    private static func huggingFaceModelCacheDirectory(modelDirectory: String) -> String? {
        let snapshotsDirectory: String = (modelDirectory as NSString).deletingLastPathComponent;
        if (snapshotsDirectory as NSString).lastPathComponent != "snapshots" {
            return nil;
        }
        let modelCacheDirectory: String = (snapshotsDirectory as NSString).deletingLastPathComponent;
        let cacheDirectoryName: String = (modelCacheDirectory as NSString).lastPathComponent;
        if RequiredFiles.decodeHuggingfaceCacheDirectoryName(cacheDirectoryName) == nil {
            return nil;
        }
        return modelCacheDirectory;
    }

    /// Port of astronomical_config's `decode_huggingface_cache_directory_name`
    /// kept module-local so validation does not grow a package dependency.
    private static func decodeHuggingfaceCacheDirectoryName(_ directoryName: String) -> String? {
        guard directoryName.hasPrefix("models--") else {
            return nil;
        }
        let encodedModelId: String = String(directoryName.dropFirst("models--".count));
        if encodedModelId.isEmpty {
            return nil;
        }
        let encodedModelIdParts: Array<Substring> = encodedModelId.split(
            separator: "--", maxSplits: 1, omittingEmptySubsequences: false);
        let organizationName: String = String(encodedModelIdParts[0]);
        if encodedModelIdParts.count > 1 {
            return organizationName + "/" + String(encodedModelIdParts[1]);
        }
        return organizationName;
    }

    private static func captureFile(
        fileHandle: FileHandle, sizeBytes: UInt64, fileName: String) throws -> Data {
        guard let requestedByteCount: Int = Int(exactly: sizeBytes) else {
            throw ArtifactValidationError.readRequiredFileForCapture(
                fileName: fileName,
                problem: "the validated file size does not fit the platform read length");
        }
        guard let capturedBytes: Data = try RequiredFiles.positionedReadExactly(
            fileDescriptor: fileHandle.fileDescriptor,
            startOffsetBytes: 0,
            byteCount: requestedByteCount) else {
            throw ArtifactValidationError.readRequiredFileForCapture(
                fileName: fileName,
                problem: "validated file ended before its inspected size");
        }
        return capturedBytes;
    }

    /// Positioned exact read mirroring the Rust `read_exact_at` loop: reads
    /// straight from the descriptor without touching its cursor state and never
    /// reopens the artifact pathname. nil when the file ends before the
    /// requested bytes are complete.
    static func positionedReadExactly(
        fileDescriptor: Int32, startOffsetBytes: UInt64, byteCount: Int) throws -> Data? {
        var collectedBytes: Array<UInt8> = Array();
        collectedBytes.reserveCapacity(byteCount);
        var readOffsetBytes: UInt64 = startOffsetBytes;
        while collectedBytes.count < byteCount {
            let remainingByteCount: Int = byteCount - collectedBytes.count;
            var chunkBytes: Array<UInt8> = Array(repeating: 0, count: remainingByteCount);
            let bytesReadCount: Int = chunkBytes.withUnsafeMutableBytes({ (chunkBuffer: UnsafeMutableRawBufferPointer) -> Int in
                return pread(fileDescriptor, chunkBuffer.baseAddress, remainingByteCount,
                             off_t(readOffsetBytes));
            });
            if bytesReadCount <= 0 {
                return nil;
            }
            collectedBytes.append(
                contentsOf: chunkBytes[0..<bytesReadCount]);
            readOffsetBytes += UInt64(bytesReadCount);
        }
        return Data(collectedBytes);
    }

    static func fstatStatus(fileDescriptor: Int32) -> stat? {
        var statusBuffer: stat = stat();
        guard fstat(fileDescriptor, &statusBuffer) == 0 else {
            return nil;
        }
        return statusBuffer;
    }
}

extension RequiredFileProfile {
    /// Integration-test seam for validating a complete required-file profile.
    /// Production keeps descriptor ownership module-private; tests need only
    /// the journey outcome and intentionally discard successful descriptors.
    public func validateAllForTests(
        modelDirectory: String, requiredFileProfiles: Array<RequiredFileProfile>) throws -> Void {
        _ = try RequiredFiles.validateRequiredFiles(
            modelDirectory: modelDirectory, requiredFileProfiles: requiredFileProfiles);
    }
}
