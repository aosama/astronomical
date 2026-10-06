import Foundation;
import IpcProtocol;

/// A deterministic raw inventory produced before family normalization. Port
/// of crates/model-serving/src/artifact_validation/raw_safetensors_inventory.rs.
public struct RawSafetensorsInventory {
    /// Tensor descriptors sorted lexically by their unmodified raw names.
    public let tensorDescriptors: Array<RawSafetensorsTensorDescriptor>;
    /// Exact bytes covered by all contiguous tensor payload intervals.
    public let shardPayloadBytes: UInt64;
}

/// One format-valid tensor declaration with absolute source-file intervals.
public struct RawSafetensorsTensorDescriptor {
    /// Unmodified tensor key from the safetensors header.
    public let tensorName: String;
    /// Actual storage dtype declared by the shard.
    public let dtype: SafetensorsDtype;
    /// Actual row-major tensor dimensions declared by the shard.
    public let shape: Array<Int>;
    /// Inclusive byte offset from the beginning of the retained file.
    public let dataStartOffsetBytes: UInt64;
    /// Exclusive byte offset from the beginning of the retained file.
    public let dataEndOffsetBytes: UInt64;
    /// Exact checked bytes in this tensor's source interval.
    public let tensorPayloadBytes: UInt64;
}

/// Reads and validates raw tensor declarations from one retained artifact
/// descriptor, before any family-owned normalization runs.
public enum RawSafetensorsInventoryReader {

    /// Parsed declaration retained until aggregate accounting is complete.
    private struct RawSafetensorsTensorDeclaration {
        let tensorName: String;
        let dtype: SafetensorsDtype;
        let shape: Array<Int>;
        let dataStartOffset: UInt64;
        let dataEndOffset: UInt64;
        let tensorPayloadBytes: UInt64;
    }

    public static func read(
        validatedRequiredFile: ValidatedRequiredFile) throws -> RawSafetensorsInventory {
        let weightsFileName: String = validatedRequiredFile.fileName;
        let fileSizeBytes: UInt64 = validatedRequiredFile.sizeBytes;
        let boundedJsonHeader: SafetensorsFraming.BoundedJsonHeader;
        do {
            boundedJsonHeader = try SafetensorsFraming.readBoundedJsonHeader(
                fileHandle: validatedRequiredFile.fileHandle, fileSizeBytes: fileSizeBytes,
                maximumHeaderLengthBytes:
                    BoundedSafetensors.MAXIMUM_ARTIFACT_SAFETENSORS_HEADER_LENGTH_BYTES);
        } catch let headerError as SafetensorsFraming.BoundedHeaderError {
            throw BoundedSafetensors.artifactSafetensorsHeaderError(
                headerError, weightsFileName: weightsFileName);
        }

        // Metadata remains validated as a string map but never enters the
        // tensor inventory consumed by family-owned normalization.
        if let metadataJsonValue: JsonWireValue = boundedJsonHeader.metadataJsonValue {
            try RawSafetensorsInventoryReader.requireStringValuedMetadata(
                metadataJsonValue: metadataJsonValue, weightsFileName: weightsFileName);
        }

        let dataSectionStartBytes: UInt64 = boundedJsonHeader.dataSectionStartBytes;
        var tensorDeclarations: Array<RawSafetensorsTensorDeclaration> = Array();
        tensorDeclarations.reserveCapacity(boundedJsonHeader.tensorJsonValues.count);
        var shardPayloadBytes: UInt64 = 0;
        for headerEntry: (tensorName: String, headerValue: JsonWireValue) in boundedJsonHeader.tensorJsonValues {
            try RawSafetensorsInventoryReader.validateTensorName(
                tensorName: headerEntry.tensorName, weightsFileName: weightsFileName);
            let tensorView: SafetensorsFraming.TensorView;
            do {
                tensorView = try SafetensorsFraming.TensorView.decoded(
                    wireValue: headerEntry.headerValue);
            } catch let jsonWireProblem as JsonWireProblem {
                throw ArtifactValidationError.invalidSafetensorsHeader(
                    fileName: weightsFileName, problem: jsonWireProblem.description);
            }
            let dtype: SafetensorsDtype = try SafetensorsDtypeHelpers.parseRawSafetensorsDtype(
                dtypeString: tensorView.dtype, fileName: weightsFileName,
                tensorName: headerEntry.tensorName);
            let tensorPayloadBytes: UInt64 = try RawSafetensorsInventoryReader
                .validateTensorPayloadBytes(
                    tensorName: headerEntry.tensorName, tensorView: tensorView,
                    dtype: dtype, weightsFileName: weightsFileName);
            let (summedPayloadBytes, payloadOverflowed): (UInt64, Bool) =
                shardPayloadBytes.addingReportingOverflow(tensorPayloadBytes);
            if payloadOverflowed {
                throw ArtifactValidationError.tensorPayloadSizeOverflow;
            }
            shardPayloadBytes = summedPayloadBytes;
            tensorDeclarations.append(RawSafetensorsTensorDeclaration(
                tensorName: headerEntry.tensorName,
                dtype: dtype,
                shape: tensorView.shape,
                dataStartOffset: tensorView.dataStartOffset(),
                dataEndOffset: tensorView.dataEndOffset(),
                tensorPayloadBytes: tensorPayloadBytes));
        }

        var tensorDescriptors: Array<RawSafetensorsTensorDescriptor> = Array();
        tensorDescriptors.reserveCapacity(tensorDeclarations.count);
        for tensorDeclaration: RawSafetensorsTensorDeclaration in tensorDeclarations {
            let dataStartOffsetBytes: UInt64 = try RawSafetensorsInventoryReader
                .absoluteTensorOffset(
                    dataSectionStartBytes: dataSectionStartBytes,
                    relativeOffsetBytes: tensorDeclaration.dataStartOffset,
                    dataEndOffset: tensorDeclaration.dataEndOffset,
                    fileSizeBytes: fileSizeBytes, weightsFileName: weightsFileName,
                    tensorName: tensorDeclaration.tensorName);
            let dataEndOffsetBytes: UInt64 = try RawSafetensorsInventoryReader
                .absoluteTensorOffset(
                    dataSectionStartBytes: dataSectionStartBytes,
                    relativeOffsetBytes: tensorDeclaration.dataEndOffset,
                    dataEndOffset: tensorDeclaration.dataEndOffset,
                    fileSizeBytes: fileSizeBytes, weightsFileName: weightsFileName,
                    tensorName: tensorDeclaration.tensorName);
            tensorDescriptors.append(RawSafetensorsTensorDescriptor(
                tensorName: tensorDeclaration.tensorName,
                dtype: tensorDeclaration.dtype,
                shape: tensorDeclaration.shape,
                dataStartOffsetBytes: dataStartOffsetBytes,
                dataEndOffsetBytes: dataEndOffsetBytes,
                tensorPayloadBytes: tensorDeclaration.tensorPayloadBytes));
        }

        try RawSafetensorsInventoryReader.validateContiguousIntervals(
            tensorDescriptors: tensorDescriptors,
            dataSectionStartBytes: dataSectionStartBytes,
            weightsFileName: weightsFileName);
        let (actualPayloadBytes, payloadUnderflowed): (UInt64, Bool) =
            fileSizeBytes.subtractingReportingOverflow(dataSectionStartBytes);
        if payloadUnderflowed {
            throw ArtifactValidationError.truncatedSafetensorsFile(
                fileName: weightsFileName, expectedMinimumBytes: dataSectionStartBytes,
                actualFileSizeBytes: fileSizeBytes);
        }
        if shardPayloadBytes != actualPayloadBytes {
            throw ArtifactValidationError.safetensorsPayloadLengthMismatch(
                fileName: weightsFileName, declaredPayloadBytes: shardPayloadBytes,
                actualPayloadBytes: actualPayloadBytes);
        }

        tensorDescriptors.sort(by: { (leftDescriptor: RawSafetensorsTensorDescriptor, rightDescriptor: RawSafetensorsTensorDescriptor) -> Bool in
            return leftDescriptor.tensorName < rightDescriptor.tensorName;
        });
        return RawSafetensorsInventory(
            tensorDescriptors: tensorDescriptors, shardPayloadBytes: shardPayloadBytes);
    }

    private static func validateTensorName(
        tensorName: String, weightsFileName: String) throws -> Void {
        let tensorNameLengthBytes: UInt64 = UInt64(tensorName.utf8.count);
        if tensorNameLengthBytes == 0
            || tensorNameLengthBytes > BoundedSafetensors.MAXIMUM_ARTIFACT_SAFETENSORS_HEADER_LENGTH_BYTES {
            throw ArtifactValidationError.invalidSafetensorsTensorName(
                fileName: weightsFileName, tensorNameLengthBytes: tensorNameLengthBytes,
                maximumTensorNameLengthBytes:
                    BoundedSafetensors.MAXIMUM_ARTIFACT_SAFETENSORS_HEADER_LENGTH_BYTES);
        }
    }

    private static func validateTensorPayloadBytes(
        tensorName: String, tensorView: SafetensorsFraming.TensorView,
        dtype: SafetensorsDtype, weightsFileName: String) throws -> UInt64 {
        let elementCount: UInt64 = try SafetensorsDtypeHelpers.elementCount(
            fromShape: tensorView.shape);
        let expectedPayloadBytes: UInt64? = try SafetensorsDtypeHelpers
            .checkedSafetensorsPayloadBytes(elementCount: elementCount, bitsPerElement: dtype.bitsize);
        let (actualPayloadBytes, intervalUnderflowed): (UInt64, Bool) = tensorView
            .dataEndOffset()
            .subtractingReportingOverflow(tensorView.dataStartOffset());
        if intervalUnderflowed {
            throw RawSafetensorsInventoryReader.invalidDataOffsets(
                weightsFileName: weightsFileName, tensorName: tensorName, tensorView: tensorView);
        }
        if expectedPayloadBytes != actualPayloadBytes {
            throw RawSafetensorsInventoryReader.invalidDataOffsets(
                weightsFileName: weightsFileName, tensorName: tensorName, tensorView: tensorView);
        }
        return actualPayloadBytes;
    }

    private static func absoluteTensorOffset(
        dataSectionStartBytes: UInt64, relativeOffsetBytes: UInt64, dataEndOffset: UInt64,
        fileSizeBytes: UInt64, weightsFileName: String, tensorName: String) throws -> UInt64 {
        let (absoluteOffsetBytes, offsetOverflowed): (UInt64, Bool) =
            dataSectionStartBytes.addingReportingOverflow(relativeOffsetBytes);
        if offsetOverflowed {
            throw RawSafetensorsInventoryReader.offsetBeyondFile(
                weightsFileName: weightsFileName, tensorName: tensorName,
                dataEndOffset: dataEndOffset, fileSizeBytes: fileSizeBytes);
        }
        if absoluteOffsetBytes > fileSizeBytes {
            throw RawSafetensorsInventoryReader.offsetBeyondFile(
                weightsFileName: weightsFileName, tensorName: tensorName,
                dataEndOffset: dataEndOffset, fileSizeBytes: fileSizeBytes);
        }
        return absoluteOffsetBytes;
    }

    private static func validateContiguousIntervals(
        tensorDescriptors: Array<RawSafetensorsTensorDescriptor>,
        dataSectionStartBytes: UInt64, weightsFileName: String) throws -> Void {
        let intervalOrder: Array<RawSafetensorsTensorDescriptor> = tensorDescriptors
            .sorted(by: { (leftDescriptor: RawSafetensorsTensorDescriptor, rightDescriptor: RawSafetensorsTensorDescriptor) -> Bool in
                if leftDescriptor.dataStartOffsetBytes != rightDescriptor.dataStartOffsetBytes {
                    return leftDescriptor.dataStartOffsetBytes < rightDescriptor.dataStartOffsetBytes;
                }
                return leftDescriptor.tensorName < rightDescriptor.tensorName;
            });
        var expectedStartOffsetBytes: UInt64 = dataSectionStartBytes;
        for tensorDescriptor: RawSafetensorsTensorDescriptor in intervalOrder {
            if tensorDescriptor.dataStartOffsetBytes != expectedStartOffsetBytes {
                throw ArtifactValidationError.safetensorsInvalidDataOffsets(
                    fileName: weightsFileName, tensorName: tensorDescriptor.tensorName,
                    dataStartOffset: RawSafetensorsInventoryReader.saturatingSubtract(
                        tensorDescriptor.dataStartOffsetBytes, dataSectionStartBytes),
                    dataEndOffset: RawSafetensorsInventoryReader.saturatingSubtract(
                        tensorDescriptor.dataEndOffsetBytes, dataSectionStartBytes));
            }
            expectedStartOffsetBytes = tensorDescriptor.dataEndOffsetBytes;
        }
    }

    private static func saturatingSubtract(_ minuend: UInt64, _ subtrahend: UInt64) -> UInt64 {
        let (difference, underflowed): (UInt64, Bool) = minuend.subtractingReportingOverflow(subtrahend);
        return underflowed ? 0 : difference;
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

    private static func invalidDataOffsets(
        weightsFileName: String, tensorName: String,
        tensorView: SafetensorsFraming.TensorView) -> ArtifactValidationError {
        return .safetensorsInvalidDataOffsets(
            fileName: weightsFileName, tensorName: tensorName,
            dataStartOffset: tensorView.dataStartOffset(),
            dataEndOffset: tensorView.dataEndOffset());
    }

    private static func offsetBeyondFile(
        weightsFileName: String, tensorName: String, dataEndOffset: UInt64,
        fileSizeBytes: UInt64) -> ArtifactValidationError {
        return .safetensorsOffsetBeyondFile(
            fileName: weightsFileName, tensorName: tensorName, dataEndOffset: dataEndOffset,
            fileSizeBytes: fileSizeBytes);
    }
}
