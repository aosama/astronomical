import Foundation;

/// Errors produced while failing closed on an invalid model artifact. Port of
/// crates/model-serving/src/artifact_validation/error.rs. Filesystem and JSON
/// sources carry their bounded problem text in place of Rust's `#[source]`.
public enum ArtifactValidationError: Error, Equatable {
    /// The supplied path is not a directory.
    case modelDirectoryNotFound(modelDirectory: String);
    /// A streaming revision manifest is not valid JSON.
    case invalidStreamingManifest(problem: String);
    /// The streaming revision's resident weight bundle is not a valid safetensors file.
    case invalidResidentBundle(problem: String);
    /// A streaming revision manifest declares an unsupported format version.
    case unsupportedStreamingManifestVersion(actualFormatVersion: UInt32);
    /// A streaming revision manifest declares a file that is missing on disk.
    case streamingDeclaredFileMissing(fileName: String, problem: String);
    /// A streaming revision manifest file disagrees with its declared size.
    case streamingDeclaredFileSize(fileName: String, expectedBytes: UInt64, actualBytes: UInt64);
    /// The artifact profile does not include a file needed by validation.
    case profileMissingRequiredFile(fileName: String);
    /// A profile file name is not one plain name relative to the model directory.
    case invalidProfileFileName(fileName: String);
    /// A profile repeats a required file name and would make ownership ambiguous.
    case duplicateProfileFileName(fileName: String);
    /// A required file could not be inspected.
    case inspectRequiredFile(fileName: String, problem: String);
    /// A required file has an unexpected byte length.
    case requiredFileSizeMismatch(fileName: String, expectedSizeBytes: UInt64, actualSizeBytes: UInt64);
    /// A required file is a symlink and would escape byte-identity validation.
    case requiredFileIsSymlink(fileName: String);
    /// A Hugging Face snapshot symlink resolves outside its cache entry's blobs directory.
    case huggingFaceSnapshotSymlinkEscapesBlobDirectory(
        fileName: String, resolvedTargetPath: String, expectedBlobDirectory: String);
    /// A Hugging Face shared cache blob could not be checked against its snapshot tree record.
    case huggingFaceSharedBlobMetadataUnavailable(fileName: String, problem: String);
    /// A resolved Hugging Face shared cache blob is not the content address the snapshot records.
    case huggingFaceSharedBlobIdentityMismatch(fileName: String, recordedDigestText: String);
    /// A resolved Hugging Face shared cache blob disagrees with its recorded size.
    case huggingFaceSharedBlobSizeMismatch(fileName: String, recordedSizeBytes: UInt64, actualSizeBytes: UInt64);
    /// A required path is not a regular file.
    case requiredFileIsNotRegular(fileName: String);
    /// A required file could not be read for bounded capture.
    case readRequiredFileForCapture(fileName: String, problem: String);
    /// A retained required-file descriptor could not supply its validated bytes.
    case readBoundedRequiredFile(fileName: String, problem: String);
    /// A required file could not be read for bounded structural validation.
    case readRequiredFileForStructuralValidation(fileName: String, problem: String);
    /// A validated file changed identity between validation and reopening.
    case validatedFileIdentityChanged(fileName: String);
    /// A config or tokenizer file exceeds the bounded capture limit.
    case capturedRequiredFileTooLarge(fileName: String, actualSizeBytes: UInt64, maximumSizeBytes: UInt64);
    /// A retained required file exceeds its caller's explicit read limit.
    case boundedRequiredFileTooLarge(fileName: String, actualSizeBytes: UInt64, maximumSizeBytes: UInt64);
    /// The safetensors length prefix could not be read from the weights file.
    case readSafetensorsLengthPrefix(fileName: String, problem: String);
    /// The safetensors header length exceeds the maximum accepted bound.
    case safetensorsHeaderLengthTooLarge(
        fileName: String, headerLengthBytes: UInt64, maximumHeaderLengthBytes: UInt64);
    /// The safetensors file is too short for its declared header length.
    case truncatedSafetensorsFile(fileName: String, expectedMinimumBytes: UInt64, actualFileSizeBytes: UInt64);
    /// The safetensors header bytes could not be read from the weights file.
    case readSafetensorsHeader(fileName: String, problem: String);
    /// The safetensors header JSON could not be parsed.
    case invalidSafetensorsHeader(fileName: String, problem: String);
    /// A safetensors tensor name is empty or exceeds the bounded header allowance.
    case invalidSafetensorsTensorName(
        fileName: String, tensorNameLengthBytes: UInt64, maximumTensorNameLengthBytes: UInt64);
    /// A safetensors tensor data offset extends beyond the weights file.
    case safetensorsOffsetBeyondFile(
        fileName: String, tensorName: String, dataEndOffset: UInt64, fileSizeBytes: UInt64);
    /// The safetensors file contains bytes outside all declared tensor payloads.
    case safetensorsPayloadLengthMismatch(
        fileName: String, declaredPayloadBytes: UInt64, actualPayloadBytes: UInt64);
    /// A safetensors tensor has invalid data offsets where the start exceeds the end.
    case safetensorsInvalidDataOffsets(
        fileName: String, tensorName: String, dataStartOffset: UInt64, dataEndOffset: UInt64);
    /// A safetensors tensor declares an unknown dtype string.
    case unknownSafetensorsDtype(fileName: String, tensorName: String, dtypeString: String);
    /// An expected tensor is absent from the safetensors file.
    case tensorMissing(tensorName: String, fileName: String);
    /// The safetensors file contains a tensor absent from the expected profile.
    case unexpectedTensor(tensorName: String);
    /// A tensor has a different dtype than the expected profile allows.
    case tensorDtypeMismatch(tensorName: String, expectedDtype: TensorDtype, actualDtype: String);
    /// A tensor has a different shape than the expected profile allows.
    case tensorShapeMismatch(tensorName: String, expectedShape: Array<Int>, actualShape: Array<Int>);
    /// Tensor payload sizes overflowed the validator's accounting type.
    case tensorPayloadSizeOverflow;
    /// The bounded artifact declared more physical sources than an opaque source ID can represent.
    case tensorSourceCountOverflow;

    public var errorDescription: String? {
        switch self {
        case .modelDirectoryNotFound(let modelDirectory):
            return "model artifact directory does not exist or is not a directory: \"\(modelDirectory)\"";
        case .invalidStreamingManifest(let problem):
            return "streaming revision manifest is not valid JSON: \(problem)";
        case .invalidResidentBundle(let problem):
            return "streaming revision resident weight bundle is not a valid safetensors file: \(problem)";
        case .unsupportedStreamingManifestVersion(let actualFormatVersion):
            return "streaming revision manifest declares unsupported format version \(actualFormatVersion) (expected 3)";
        case .streamingDeclaredFileMissing(let fileName, let problem):
            return "streaming revision manifest declares missing file \(fileName): \(problem)";
        case .streamingDeclaredFileSize(let fileName, let expectedBytes, let actualBytes):
            return "streaming revision file \(fileName) has \(actualBytes) bytes, manifest declares \(expectedBytes)";
        case .profileMissingRequiredFile(let fileName):
            return "artifact profile is missing required file \(fileName)";
        case .invalidProfileFileName(let fileName):
            return "artifact profile entry \"\(fileName)\" must be one plain file name";
        case .duplicateProfileFileName(let fileName):
            return "artifact profile contains duplicate required file name \(fileName)";
        case .inspectRequiredFile(let fileName, let problem):
            return "failed to inspect required model file \(fileName): \(problem)";
        case .requiredFileSizeMismatch(let fileName, let expectedSizeBytes, let actualSizeBytes):
            return "required model file \(fileName) has \(actualSizeBytes) bytes, expected \(expectedSizeBytes) bytes";
        case .requiredFileIsSymlink(let fileName):
            return "required model file \(fileName) is a symlink";
        case .huggingFaceSnapshotSymlinkEscapesBlobDirectory(let fileName, let resolvedTargetPath, let expectedBlobDirectory):
            return "Hugging Face snapshot file \(fileName) resolves to \"\(resolvedTargetPath)\", outside expected blob directory \"\(expectedBlobDirectory)\"";
        case .huggingFaceSharedBlobMetadataUnavailable(let fileName, let problem):
            return "Hugging Face shared cache file \(fileName) could not be verified against its snapshot tree record: \(problem)";
        case .huggingFaceSharedBlobIdentityMismatch(let fileName, let recordedDigestText):
            return "Hugging Face snapshot file \(fileName) resolves to a shared cache blob that is not the content-addressed object for digest \(recordedDigestText)";
        case .huggingFaceSharedBlobSizeMismatch(let fileName, let recordedSizeBytes, let actualSizeBytes):
            return "Hugging Face shared cache file \(fileName) has \(actualSizeBytes) bytes, but its snapshot tree record declares \(recordedSizeBytes) bytes";
        case .requiredFileIsNotRegular(let fileName):
            return "required model file \(fileName) is not a regular file";
        case .readRequiredFileForCapture(let fileName, let problem):
            return "failed to read required model file \(fileName) for bounded capture: \(problem)";
        case .readBoundedRequiredFile(let fileName, let problem):
            return "failed to read validated required model file \(fileName) within its byte limit: \(problem)";
        case .readRequiredFileForStructuralValidation(let fileName, let problem):
            return "failed to read required model file \(fileName) for bounded structural validation: \(problem)";
        case .validatedFileIdentityChanged(let fileName):
            return "validated required file \(fileName) changed identity after validation";
        case .capturedRequiredFileTooLarge(let fileName, let actualSizeBytes, let maximumSizeBytes):
            return "required model file \(fileName) is \(actualSizeBytes) bytes, exceeding captured-file limit \(maximumSizeBytes)";
        case .boundedRequiredFileTooLarge(let fileName, let actualSizeBytes, let maximumSizeBytes):
            return "required model file \(fileName) is \(actualSizeBytes) bytes, exceeding bounded-read limit \(maximumSizeBytes)";
        case .readSafetensorsLengthPrefix(let fileName, let problem):
            return "failed to read safetensors length prefix from weight file \(fileName): \(problem)";
        case .safetensorsHeaderLengthTooLarge(let fileName, let headerLengthBytes, let maximumHeaderLengthBytes):
            return "safetensors header length \(headerLengthBytes) exceeds maximum \(maximumHeaderLengthBytes) in weight file \(fileName)";
        case .truncatedSafetensorsFile(let fileName, let expectedMinimumBytes, let actualFileSizeBytes):
            return "safetensors file \(fileName) is \(actualFileSizeBytes) bytes, shorter than the \(expectedMinimumBytes) bytes required by its header length";
        case .readSafetensorsHeader(let fileName, let problem):
            return "failed to read safetensors header from weight file \(fileName): \(problem)";
        case .invalidSafetensorsHeader(let fileName, let problem):
            return "invalid safetensors header JSON in weight file \(fileName): \(problem)";
        case .invalidSafetensorsTensorName(let fileName, let tensorNameLengthBytes, let maximumTensorNameLengthBytes):
            return "safetensors tensor name has \(tensorNameLengthBytes) bytes, expected 1..=\(maximumTensorNameLengthBytes) bytes in weight file \(fileName)";
        case .safetensorsOffsetBeyondFile(let fileName, let tensorName, let dataEndOffset, let fileSizeBytes):
            return "safetensors tensor \(tensorName) data end offset \(dataEndOffset) exceeds file size \(fileSizeBytes) in weight file \(fileName)";
        case .safetensorsPayloadLengthMismatch(let fileName, let declaredPayloadBytes, let actualPayloadBytes):
            return "safetensors payload length mismatch in \(fileName): declared \(declaredPayloadBytes) bytes, file contains \(actualPayloadBytes) bytes";
        case .safetensorsInvalidDataOffsets(let fileName, let tensorName, let dataStartOffset, let dataEndOffset):
            return "safetensors tensor \(tensorName) has invalid data offsets: start \(dataStartOffset) exceeds end \(dataEndOffset) in weight file \(fileName)";
        case .unknownSafetensorsDtype(let fileName, let tensorName, let dtypeString):
            return "safetensors tensor \(tensorName) has unknown dtype \(dtypeString) in weight file \(fileName)";
        case .tensorMissing(let tensorName, let fileName):
            return "tensor \(tensorName) is missing from the safetensors weight file \(fileName)";
        case .unexpectedTensor(let tensorName):
            return "extra tensor \(tensorName) in the safetensors weight file";
        case .tensorDtypeMismatch(let tensorName, let expectedDtype, let actualDtype):
            return "tensor \(tensorName) dtype mismatch: expected \(rustDebugText(expectedDtype)), got \(actualDtype)";
        case .tensorShapeMismatch(let tensorName, let expectedShape, let actualShape):
            return "tensor \(tensorName) shape mismatch: expected \(rustDebugArrayText(expectedShape)), got \(rustDebugArrayText(actualShape))";
        case .tensorPayloadSizeOverflow:
            return "safetensors payload byte count overflowed u64";
        case .tensorSourceCountOverflow:
            return "safetensors source count overflowed the validated source identity range";
        }
    }

    /// Returns a bounded, path-free explanation suitable for a public load error.
    public func publicFailureReason() -> String {
        let unboundedReason: String;
        switch self {
        case .modelDirectoryNotFound:
            unboundedReason = "model artifact directory does not exist or is not a directory";
        case .huggingFaceSnapshotSymlinkEscapesBlobDirectory:
            unboundedReason = "a model file resolves outside its permitted Hugging Face snapshot";
        case .huggingFaceSharedBlobMetadataUnavailable:
            unboundedReason = "a Hugging Face shared cache file could not be verified against its snapshot tree record";
        case .huggingFaceSharedBlobIdentityMismatch:
            unboundedReason = "a Hugging Face shared cache file is not the object its snapshot record declares";
        case .huggingFaceSharedBlobSizeMismatch:
            unboundedReason = "a Hugging Face shared cache file disagrees with the size its snapshot record declares";
        case .truncatedSafetensorsFile:
            unboundedReason = errorDescription?
                .replacingOccurrences(of: "safetensors file", with: "truncated safetensors file", options: .anchored)
                ?? errorDescription ?? "";
        default:
            unboundedReason = errorDescription ?? String(describing: self);
        }
        return ArtifactValidationError.boundPublicArtifactFailureReason(unboundedReason);
    }

    private static func boundPublicArtifactFailureReason(_ unboundedReason: String) -> String {
        let MAX_PUBLIC_REASON_CHARACTERS: Int = 512;
        let pathFreeReason: String = unboundedReason
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "\\", with: "_");
        var boundedReason: String = String(pathFreeReason.prefix(MAX_PUBLIC_REASON_CHARACTERS));
        if pathFreeReason.count > MAX_PUBLIC_REASON_CHARACTERS {
            boundedReason.removeLast();
            boundedReason.append("…");
        }
        return boundedReason;
    }

    /// Reproduces Rust's `{:?}` rendering of the TensorDtype enum, e.g. `BF16`.
    private func rustDebugText(_ dtype: TensorDtype) -> String {
        switch dtype {
        case .affineQuantizationFloat: return "AffineQuantizationFloat";
        case .modelFloat: return "ModelFloat";
        case .bfloat16: return "BFloat16";
        case .float32: return "Float32";
        case .uint32: return "UInt32";
        }
    }

    /// Reproduces Rust's `{:?}` slice rendering, e.g. `[2, 3]`.
    private func rustDebugArrayText(_ values: Array<Int>) -> String {
        return "[\(values.map({ (value: Int) -> String in String(value) }).joined(separator: ", "))]";
    }
}
