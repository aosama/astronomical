import Foundation;
import IpcProtocol;

/// Bounded safetensors validator that never allocates the entire weights
/// file. Port of crates/model-serving/src/artifact_validation/bounded_safetensors.rs:
/// only the length prefix and the bounded header are read, every tensor offset
/// is validated against the open file size, and the multi-hundred-megabyte
/// payload region is never touched.
public enum BoundedSafetensors {

    public static let MAXIMUM_ARTIFACT_SAFETENSORS_HEADER_LENGTH_BYTES: UInt64 = 16 * 1024 * 1024;

    /// Parsed safetensors metadata extracted from a bounded header read.
    private struct BoundedSafetensorsMetadata {
        /// Total tensor payload bytes across all validated tensors.
        let totalPayloadBytes: UInt64;
    }

    /// Parsed metadata returned by partial-profile safetensors validation.
    public struct PartialProfileMetadata {
        /// Total tensor payload bytes across all tensors in the shard.
        public let totalPayloadBytes: UInt64;
    }

    /// One parsed artifact header: per-tensor views plus where payloads begin.
    private struct ArtifactSafetensorsHeader {
        let tensorsByName: Dictionary<String, SafetensorsFraming.TensorView>;
        let dataSectionStartBytes: UInt64;
    }

    /// Validates a safetensors shard where some tensors have strict dtype/shape
    /// profiles and the remaining tensors are accepted by name only.
    ///
    /// Qwen3.5-MoE uses exact dtype and shape profiles generated from its
    /// validated config.
    public static func validateBoundedSafetensorsWithPartialProfiles(
        fileHandle: FileHandle, fileSizeBytes: UInt64, weightsFileName: String,
        profiledTensorProfiles: Array<TensorProfile>,
        acceptedExtraTensorNames: Set<String>) throws -> PartialProfileMetadata {
        var requiredTensorNames: Set<String> = Set();
        for tensorProfile: TensorProfile in profiledTensorProfiles {
            requiredTensorNames.insert(tensorProfile.name);
        }
        let boundedMetadata: BoundedSafetensorsMetadata = try BoundedSafetensors
            .validateBoundedSafetensorsInternal(
                fileHandle: fileHandle, fileSizeBytes: fileSizeBytes,
                weightsFileName: weightsFileName,
                requiredTensorNames: requiredTensorNames,
                acceptedExtraTensorNames: acceptedExtraTensorNames,
                requiredTensorProfiles: profiledTensorProfiles,
                recognizedTensorProfiles: Array());

        return PartialProfileMetadata(totalPayloadBytes: boundedMetadata.totalPayloadBytes);
    }

    /// Maps a bounded framing failure onto the shared artifact error contract.
    static func artifactSafetensorsHeaderError(
        _ boundedHeaderError: SafetensorsFraming.BoundedHeaderError,
        weightsFileName: String) -> ArtifactValidationError {
        switch boundedHeaderError {
        case .readLengthPrefix(let problem):
            return .readSafetensorsLengthPrefix(fileName: weightsFileName, problem: problem);
        case .headerLengthTooLarge(let headerLengthBytes, let maximumHeaderLengthBytes):
            return .safetensorsHeaderLengthTooLarge(
                fileName: weightsFileName, headerLengthBytes: headerLengthBytes,
                maximumHeaderLengthBytes: maximumHeaderLengthBytes);
        case .headerBeyondFile(let headerEndOffsetBytes, let fileSizeBytes):
            return .truncatedSafetensorsFile(
                fileName: weightsFileName, expectedMinimumBytes: headerEndOffsetBytes,
                actualFileSizeBytes: fileSizeBytes);
        case .readHeader(let problem):
            return .readSafetensorsHeader(fileName: weightsFileName, problem: problem);
        case .invalidHeaderJson(let problem):
            return .invalidSafetensorsHeader(fileName: weightsFileName, problem: problem);
        }
    }

    private static func validateBoundedSafetensorsInternal(
        fileHandle: FileHandle, fileSizeBytes: UInt64, weightsFileName: String,
        requiredTensorNames: Set<String>, acceptedExtraTensorNames: Set<String>,
        requiredTensorProfiles: Array<TensorProfile>,
        recognizedTensorProfiles: Array<TensorProfile>) throws -> BoundedSafetensorsMetadata {
        let artifactHeader: ArtifactSafetensorsHeader = try BoundedSafetensors
            .readArtifactSafetensorsHeader(
                fileHandle: fileHandle, fileSizeBytes: fileSizeBytes,
                weightsFileName: weightsFileName);
        let boundedMetadata: BoundedSafetensorsMetadata = try BoundedSafetensors.validateAllTensors(
            tensorsByName: artifactHeader.tensorsByName,
            requiredTensorNames: requiredTensorNames,
            acceptedExtraTensorNames: acceptedExtraTensorNames,
            requiredTensorProfiles: requiredTensorProfiles,
            recognizedTensorProfiles: recognizedTensorProfiles,
            dataSectionStartBytes: artifactHeader.dataSectionStartBytes,
            fileSizeBytes: fileSizeBytes,
            weightsFileName: weightsFileName);
        let (actualPayloadBytes, payloadUnderflowed): (UInt64, Bool) =
            fileSizeBytes.subtractingReportingOverflow(artifactHeader.dataSectionStartBytes);
        if payloadUnderflowed {
            throw ArtifactValidationError.truncatedSafetensorsFile(
                fileName: weightsFileName,
                expectedMinimumBytes: artifactHeader.dataSectionStartBytes,
                actualFileSizeBytes: fileSizeBytes);
        }
        if boundedMetadata.totalPayloadBytes != actualPayloadBytes {
            throw ArtifactValidationError.safetensorsPayloadLengthMismatch(
                fileName: weightsFileName,
                declaredPayloadBytes: boundedMetadata.totalPayloadBytes,
                actualPayloadBytes: actualPayloadBytes);
        }
        return boundedMetadata;
    }

    /// Validates that tensor data offsets are contiguous starting from zero and
    /// that each tensor's data range matches its declared shape and dtype size.
    private static func validateTensorDataConsistency(
        tensorsByName: Dictionary<String, SafetensorsFraming.TensorView>,
        weightsFileName: String) throws -> Void {
        let orderedTensors: Array<(tensorName: String, tensorView: SafetensorsFraming.TensorView)> =
            tensorsByName
                .map({ (tensorName: String, tensorView: SafetensorsFraming.TensorView) -> (tensorName: String, tensorView: SafetensorsFraming.TensorView) in
                    return (tensorName, tensorView);
                })
                .sorted(by: { (leftEntry: (tensorName: String, tensorView: SafetensorsFraming.TensorView), rightEntry: (tensorName: String, tensorView: SafetensorsFraming.TensorView)) -> Bool in
                    if leftEntry.tensorView.dataStartOffset() != rightEntry.tensorView.dataStartOffset() {
                        return leftEntry.tensorView.dataStartOffset() < rightEntry.tensorView.dataStartOffset();
                    }
                    return leftEntry.tensorName < rightEntry.tensorName;
                });

        var expectedStartOffset: UInt64 = 0;
        for orderedTensor: (tensorName: String, tensorView: SafetensorsFraming.TensorView) in orderedTensors {
            if orderedTensor.tensorView.dataStartOffset() != expectedStartOffset {
                throw ArtifactValidationError.safetensorsInvalidDataOffsets(
                    fileName: weightsFileName,
                    tensorName: orderedTensor.tensorName,
                    dataStartOffset: orderedTensor.tensorView.dataStartOffset(),
                    dataEndOffset: orderedTensor.tensorView.dataEndOffset());
            }
            if orderedTensor.tensorView.dataEndOffset() < orderedTensor.tensorView.dataStartOffset() {
                throw ArtifactValidationError.safetensorsInvalidDataOffsets(
                    fileName: weightsFileName,
                    tensorName: orderedTensor.tensorName,
                    dataStartOffset: orderedTensor.tensorView.dataStartOffset(),
                    dataEndOffset: orderedTensor.tensorView.dataEndOffset());
            }

            let elementCount: UInt64 = try SafetensorsDtypeHelpers.elementCount(
                fromShape: orderedTensor.tensorView.shape);
            let bitsPerElement: UInt64 = try SafetensorsDtypeHelpers.dtypeBitsPerElement(
                dtypeString: orderedTensor.tensorView.dtype,
                fileName: weightsFileName,
                tensorName: orderedTensor.tensorName);
            guard let expectedDataBytes: UInt64 = try SafetensorsDtypeHelpers
                .checkedSafetensorsPayloadBytes(
                    elementCount: elementCount, bitsPerElement: bitsPerElement) else {
                throw ArtifactValidationError.safetensorsInvalidDataOffsets(
                    fileName: weightsFileName,
                    tensorName: orderedTensor.tensorName,
                    dataStartOffset: orderedTensor.tensorView.dataStartOffset(),
                    dataEndOffset: orderedTensor.tensorView.dataEndOffset());
            }
            let (actualDataBytes, dataBytesUnderflowed): (UInt64, Bool) = orderedTensor.tensorView
                .dataEndOffset()
                .subtractingReportingOverflow(orderedTensor.tensorView.dataStartOffset());
            if dataBytesUnderflowed {
                throw ArtifactValidationError.safetensorsInvalidDataOffsets(
                    fileName: weightsFileName,
                    tensorName: orderedTensor.tensorName,
                    dataStartOffset: orderedTensor.tensorView.dataStartOffset(),
                    dataEndOffset: orderedTensor.tensorView.dataEndOffset());
            }
            if actualDataBytes != expectedDataBytes {
                throw ArtifactValidationError.safetensorsInvalidDataOffsets(
                    fileName: weightsFileName,
                    tensorName: orderedTensor.tensorName,
                    dataStartOffset: orderedTensor.tensorView.dataStartOffset(),
                    dataEndOffset: orderedTensor.tensorView.dataEndOffset());
            }

            expectedStartOffset = orderedTensor.tensorView.dataEndOffset();
        }
    }

    private static func readArtifactSafetensorsHeader(
        fileHandle: FileHandle, fileSizeBytes: UInt64,
        weightsFileName: String) throws -> ArtifactSafetensorsHeader {
        let boundedJsonHeader: SafetensorsFraming.BoundedJsonHeader;
        do {
            boundedJsonHeader = try SafetensorsFraming.readBoundedJsonHeader(
                fileHandle: fileHandle, fileSizeBytes: fileSizeBytes,
                maximumHeaderLengthBytes: MAXIMUM_ARTIFACT_SAFETENSORS_HEADER_LENGTH_BYTES);
        } catch let boundedHeaderError as SafetensorsFraming.BoundedHeaderError {
            throw BoundedSafetensors.artifactSafetensorsHeaderError(
                boundedHeaderError, weightsFileName: weightsFileName);
        }
        if let metadataJsonValue: JsonWireValue = boundedJsonHeader.metadataJsonValue {
            try BoundedSafetensors.requireStringValuedMetadata(
                metadataJsonValue: metadataJsonValue, weightsFileName: weightsFileName);
        }
        var tensorsByName: Dictionary<String, SafetensorsFraming.TensorView> = Dictionary(
            minimumCapacity: boundedJsonHeader.tensorJsonValues.count);
        for headerEntry: (tensorName: String, headerValue: JsonWireValue) in boundedJsonHeader.tensorJsonValues {
            do {
                tensorsByName[headerEntry.tensorName] = try SafetensorsFraming.TensorView.decoded(
                    wireValue: headerEntry.headerValue);
            } catch let jsonWireProblem as JsonWireProblem {
                throw ArtifactValidationError.invalidSafetensorsHeader(
                    fileName: weightsFileName, problem: jsonWireProblem.description);
            }
        }
        return ArtifactSafetensorsHeader(
            tensorsByName: tensorsByName,
            dataSectionStartBytes: boundedJsonHeader.dataSectionStartBytes);
    }

    /// Rust parses `__metadata__` into a `HashMap<String, String>`; every entry
    /// must therefore be a string.
    private static func requireStringValuedMetadata(
        metadataJsonValue: JsonWireValue, weightsFileName: String) throws -> Void {
        let metadataObject: JsonWireObject;
        do {
            metadataObject = try JsonWireValue.extractObject(metadataJsonValue);
        } catch let jsonWireProblem as JsonWireProblem {
            throw ArtifactValidationError.invalidSafetensorsHeader(
                fileName: weightsFileName, problem: jsonWireProblem.description);
        }
        for metadataEntry: (key: String, value: JsonWireValue) in metadataObject.entries {
            if case .string(_) = metadataEntry.value {
                continue;
            }
            throw ArtifactValidationError.invalidSafetensorsHeader(
                fileName: weightsFileName,
                problem: "invalid type: metadata value for \(metadataEntry.key) is not a string");
        }
    }

    private static func validateAllTensors(
        tensorsByName: Dictionary<String, SafetensorsFraming.TensorView>,
        requiredTensorNames: Set<String>, acceptedExtraTensorNames: Set<String>,
        requiredTensorProfiles: Array<TensorProfile>,
        recognizedTensorProfiles: Array<TensorProfile>,
        dataSectionStartBytes: UInt64, fileSizeBytes: UInt64,
        weightsFileName: String) throws -> BoundedSafetensorsMetadata {
        try BoundedSafetensors.validateTensorDataConsistency(
            tensorsByName: tensorsByName, weightsFileName: weightsFileName);

        let orderedTensorNames: Array<String> = tensorsByName.keys.sorted();
        for tensorName: String in orderedTensorNames {
            if requiredTensorNames.contains(tensorName) || acceptedExtraTensorNames.contains(tensorName) {
                continue;
            }
            throw ArtifactValidationError.unexpectedTensor(tensorName: tensorName);
        }
        for tensorName: String in orderedTensorNames {
            try BoundedSafetensors.validateTensorOffsets(
                tensorName: tensorName,
                tensorView: tensorsByName[tensorName]!,
                dataSectionStartBytes: dataSectionStartBytes,
                fileSizeBytes: fileSizeBytes,
                fileName: weightsFileName);
        }

        for requiredTensorName: String in requiredTensorNames.sorted() {
            if tensorsByName[requiredTensorName] == nil {
                throw ArtifactValidationError.tensorMissing(
                    tensorName: requiredTensorName, fileName: weightsFileName);
            }
        }

        for tensorProfile: TensorProfile in requiredTensorProfiles {
            guard let tensorView: SafetensorsFraming.TensorView = tensorsByName[tensorProfile.name] else {
                throw ArtifactValidationError.tensorMissing(
                    tensorName: tensorProfile.name, fileName: weightsFileName);
            }
            try BoundedSafetensors.validateTensorDtype(
                tensorProfile: tensorProfile, tensorView: tensorView, fileName: weightsFileName);
            try BoundedSafetensors.validateTensorShape(tensorProfile: tensorProfile, tensorView: tensorView);
        }
        for tensorProfile: TensorProfile in recognizedTensorProfiles {
            guard let tensorView: SafetensorsFraming.TensorView = tensorsByName[tensorProfile.name] else {
                continue;
            }
            try BoundedSafetensors.validateTensorDtype(
                tensorProfile: tensorProfile, tensorView: tensorView, fileName: weightsFileName);
            try BoundedSafetensors.validateTensorShape(tensorProfile: tensorProfile, tensorView: tensorView);
        }

        var totalPayloadBytes: UInt64 = 0;
        for tensorName: String in orderedTensorNames {
            let tensorView: SafetensorsFraming.TensorView = tensorsByName[tensorName]!;
            let (tensorDataBytes, dataBytesUnderflowed): (UInt64, Bool) = tensorView
                .dataEndOffset()
                .subtractingReportingOverflow(tensorView.dataStartOffset());
            if dataBytesUnderflowed {
                throw ArtifactValidationError.tensorPayloadSizeOverflow;
            }
            let (summedPayloadBytes, payloadOverflowed): (UInt64, Bool) =
                totalPayloadBytes.addingReportingOverflow(tensorDataBytes);
            if payloadOverflowed {
                throw ArtifactValidationError.tensorPayloadSizeOverflow;
            }
            totalPayloadBytes = summedPayloadBytes;
        }

        return BoundedSafetensorsMetadata(totalPayloadBytes: totalPayloadBytes);
    }

    private static func validateTensorOffsets(
        tensorName: String, tensorView: SafetensorsFraming.TensorView,
        dataSectionStartBytes: UInt64, fileSizeBytes: UInt64, fileName: String) throws -> Void {
        if tensorView.dataStartOffset() > tensorView.dataEndOffset() {
            throw ArtifactValidationError.safetensorsInvalidDataOffsets(
                fileName: fileName, tensorName: tensorName,
                dataStartOffset: tensorView.dataStartOffset(),
                dataEndOffset: tensorView.dataEndOffset());
        }
        let (absoluteEndOffset, endOffsetOverflowed): (UInt64, Bool) = dataSectionStartBytes
            .addingReportingOverflow(tensorView.dataEndOffset());
        if endOffsetOverflowed {
            throw ArtifactValidationError.safetensorsOffsetBeyondFile(
                fileName: fileName, tensorName: tensorName,
                dataEndOffset: tensorView.dataEndOffset(), fileSizeBytes: fileSizeBytes);
        }
        if absoluteEndOffset > fileSizeBytes {
            throw ArtifactValidationError.safetensorsOffsetBeyondFile(
                fileName: fileName, tensorName: tensorName,
                dataEndOffset: tensorView.dataEndOffset(), fileSizeBytes: fileSizeBytes);
        }
    }

    private static func validateTensorDtype(
        tensorProfile: TensorProfile, tensorView: SafetensorsFraming.TensorView,
        fileName: String) throws -> Void {
        let acceptedDtypeNames: Set<String>;
        switch tensorProfile.dtype {
        case .affineQuantizationFloat, .modelFloat:
            acceptedDtypeNames = ["F16", "BF16", "F32"];
        case .bfloat16:
            acceptedDtypeNames = ["BF16"];
        case .float32:
            acceptedDtypeNames = ["F32"];
        case .uint32:
            acceptedDtypeNames = ["U32"];
        }
        if acceptedDtypeNames.contains(tensorView.dtype) {
            return;
        }
        let actualDtype: SafetensorsDtype = try SafetensorsDtypeHelpers.parseSafetensorsDtype(
            dtypeString: tensorView.dtype, fileName: fileName, tensorName: tensorProfile.name);
        throw ArtifactValidationError.tensorDtypeMismatch(
            tensorName: tensorProfile.name, expectedDtype: tensorProfile.dtype,
            actualDtype: actualDtype.canonicalName);
    }

    private static func validateTensorShape(
        tensorProfile: TensorProfile,
        tensorView: SafetensorsFraming.TensorView) throws -> Void {
        let shapeMatches: Bool = tensorView.shape == tensorProfile.shape
            || tensorProfile.equivalentPublishedShapes.contains(tensorView.shape);
        if shapeMatches == false {
            throw ArtifactValidationError.tensorShapeMismatch(
                tensorName: tensorProfile.name, expectedShape: tensorProfile.shape,
                actualShape: tensorView.shape);
        }
    }
}
