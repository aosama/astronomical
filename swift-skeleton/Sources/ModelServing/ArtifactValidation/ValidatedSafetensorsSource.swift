import Foundation;
import IpcProtocol;

/// Already-open validated safetensors descriptor with one retained bounded
/// header. Port of
/// crates/model-serving/src/artifact_validation/validated_safetensors_source.rs.
public final class ValidatedSafetensorsSource {
    /// Same 16 MiB header bound the Rust source applies independently of the
    /// bounded-artifact reader.
    private static let MAXIMUM_SAFETENSORS_HEADER_BYTES: UInt64 = 16 * 1024 * 1024;

    private let sourceIdValue: TensorSourceId;
    private let requiredFile: ValidatedRequiredFile;
    private let tensorMetadataByStoredName: Dictionary<String, SafetensorsFraming.TensorView>;
    private let payloadBytesValue: UInt64;

    private init(
        sourceId: TensorSourceId, requiredFile: ValidatedRequiredFile,
        tensorMetadataByStoredName: Dictionary<String, SafetensorsFraming.TensorView>,
        payloadBytes: UInt64) {
        self.sourceIdValue = sourceId;
        self.requiredFile = requiredFile;
        self.tensorMetadataByStoredName = tensorMetadataByStoredName;
        self.payloadBytesValue = payloadBytes;
    }

    /// Exercises the required/optional profile partition through the real
    /// descriptor and header path. The returned boolean is false only when the
    /// requested optional feature has a profile defect and therefore must be
    /// disabled atomically; a thrown error means required source validity failed.
    public static func validateSafetensorsProfilePartitionsForTests(
        modelDirectory: String, relativeFileName: String, inventory: TensorInventory,
        canonicalProfiles: Array<TensorProfile>,
        optionalFeature: TensorFeature) throws -> Bool {
        let requiredFile: ValidatedRequiredFile = try RequiredFiles.validateRequiredFile(
            modelDirectory: modelDirectory,
            requiredFileProfile: RequiredFileProfile(
                fileName: relativeFileName, sizeBytes: 0));
        let source: ValidatedSafetensorsSource = try ValidatedSafetensorsSource.parse(
            sourceId: TensorSourceId(sourceNumber: 1), requiredFile: requiredFile);
        try source.validateRequiredInventoryProfiles(
            inventory: inventory, canonicalProfiles: canonicalProfiles);
        do {
            try source.validateFeatureInventoryProfiles(
                inventory: inventory, canonicalProfiles: canonicalProfiles,
                feature: optionalFeature);
        } catch {
            return false;
        }
        return true;
    }

    public static func parse(
        sourceId: TensorSourceId, requiredFile: ValidatedRequiredFile) throws -> ValidatedSafetensorsSource {
        let boundedJsonHeader: SafetensorsFraming.BoundedJsonHeader;
        do {
            boundedJsonHeader = try SafetensorsFraming.readBoundedJsonHeader(
                fileHandle: requiredFile.fileHandle, fileSizeBytes: requiredFile.sizeBytes,
                maximumHeaderLengthBytes: ValidatedSafetensorsSource.MAXIMUM_SAFETENSORS_HEADER_BYTES);
        } catch let boundedHeaderError as SafetensorsFraming.BoundedHeaderError {
            throw BoundedSafetensors.artifactSafetensorsHeaderError(
                boundedHeaderError, weightsFileName: requiredFile.fileName);
        }
        // The safetensors spec allows metadata to be a JSON object of strings at
        // the top level. OptiQ sidecars embed richer metadata (quantization
        // bits, group sizes, policies) that do not conform to the flat-string
        // schema, so a non-conforming metadata block is non-fatal here — the
        // sidecar is still valid as long as its tensor entries parse
        // successfully. Metadata never enters the tensor inventory.
        var tensorMetadataByStoredName: Dictionary<String, SafetensorsFraming.TensorView> =
            Dictionary();
        for headerEntry: (tensorName: String, headerValue: JsonWireValue) in boundedJsonHeader.tensorJsonValues {
            let tensorMetadata: SafetensorsFraming.TensorView;
            do {
                tensorMetadata = try SafetensorsFraming.TensorView.decoded(
                    wireValue: headerEntry.headerValue);
            } catch let jsonWireProblem as JsonWireProblem {
                throw ArtifactValidationError.invalidSafetensorsHeader(
                    fileName: requiredFile.fileName, problem: jsonWireProblem.description);
            }
            tensorMetadataByStoredName[headerEntry.tensorName] = tensorMetadata;
        }
        let payloadBytes: UInt64 = try ValidatedSafetensorsSource.validatePhysicalMetadata(
            tensorsByName: tensorMetadataByStoredName,
            dataSectionStartBytes: boundedJsonHeader.dataSectionStartBytes,
            fileSizeBytes: boundedJsonHeader.fileSizeBytes,
            fileName: requiredFile.fileName);
        return ValidatedSafetensorsSource(
            sourceId: sourceId, requiredFile: requiredFile,
            tensorMetadataByStoredName: tensorMetadataByStoredName, payloadBytes: payloadBytes);
    }

    public var sourceId: TensorSourceId {
        return self.sourceIdValue;
    }

    public var fileName: String {
        return self.requiredFile.fileName;
    }

    public var payloadBytes: UInt64 {
        return self.payloadBytesValue;
    }

    /// Stored tensor names in canonical-name order, mirroring the Rust
    /// B-tree key iteration.
    public func storedTensorNames() -> Array<String> {
        return self.tensorMetadataByStoredName.keys.sorted();
    }

    /// Retained header view for one stored tensor, used for structured
    /// validation diagnostics.
    public func storedTensorView(storedName: String) -> SafetensorsFraming.TensorView? {
        return self.tensorMetadataByStoredName[storedName];
    }

    /// Validates canonical profiles against physical names without reparsing
    /// the header.
    public func validateInventoryProfiles(
        inventory: TensorInventory, canonicalProfiles: Array<TensorProfile>) throws -> Void {
        let sourceLocations: Array<TensorLocation> = self.sourceLocations(inventory: inventory);
        try self.validateExactPhysicalInventory(sourceLocations: sourceLocations);
        try self.validateLocations(locations: sourceLocations, canonicalProfiles: canonicalProfiles);
    }

    /// Validates required target and vision profiles while leaving optional
    /// features atomic.
    ///
    /// A target shard may physically contain an optional MTP head. A wrong
    /// optional dtype or shape must disable that complete feature, not reject
    /// otherwise valid target weights. Physical-name and offset validation
    /// still covers the entire source before this split, so ignoring optional
    /// profile semantics cannot hide an undeclared or structurally unsafe tensor.
    public func validateRequiredInventoryProfiles(
        inventory: TensorInventory, canonicalProfiles: Array<TensorProfile>) throws -> Void {
        let sourceLocations: Array<TensorLocation> = self.sourceLocations(inventory: inventory);
        try self.validateExactPhysicalInventory(sourceLocations: sourceLocations);
        let requiredLocations: Array<TensorLocation> = sourceLocations
            .filter({ (tensorLocation: TensorLocation) -> Bool in
                return tensorLocation.feature == nil;
            });
        try self.validateLocations(locations: requiredLocations, canonicalProfiles: canonicalProfiles);
    }

    /// Validates one optional feature independently after required profiles
    /// are known safe.
    public func validateFeatureInventoryProfiles(
        inventory: TensorInventory, canonicalProfiles: Array<TensorProfile>,
        feature: TensorFeature) throws -> Void {
        let featureLocations: Array<TensorLocation> = self.sourceLocations(inventory: inventory)
            .filter({ (tensorLocation: TensorLocation) -> Bool in
                return tensorLocation.feature == feature;
            });
        try self.validateLocations(locations: featureLocations, canonicalProfiles: canonicalProfiles);
    }

    public func intoValidatedWeightsFile() throws -> ValidatedWeightsFile {
        return try self.requiredFile.intoValidatedWeightsFile();
    }

    private func sourceLocations(inventory: TensorInventory) -> Array<TensorLocation> {
        return inventory.locations().filter({ (tensorLocation: TensorLocation) -> Bool in
            return tensorLocation.sourceId == self.sourceIdValue;
        });
    }

    private func validateExactPhysicalInventory(
        sourceLocations: Array<TensorLocation>) throws -> Void {
        var declaredStoredNames: Set<String> = Set();
        for tensorLocation: TensorLocation in sourceLocations {
            declaredStoredNames.insert(tensorLocation.storedName);
        }
        let physicalStoredNames: Set<String> = Set(self.tensorMetadataByStoredName.keys);
        if declaredStoredNames == physicalStoredNames {
            return;
        }
        let unresolvedStoredName: String =
            physicalStoredNames.subtracting(declaredStoredNames).sorted().first
                ?? declaredStoredNames.subtracting(physicalStoredNames).sorted().first
                ?? "unresolved tensor inventory";
        throw ArtifactValidationError.unexpectedTensor(tensorName: unresolvedStoredName);
    }

    private func validateLocations(
        locations: Array<TensorLocation>, canonicalProfiles: Array<TensorProfile>) throws -> Void {
        var profileByCanonicalName: Dictionary<String, TensorProfile> = Dictionary();
        for tensorProfile: TensorProfile in canonicalProfiles {
            profileByCanonicalName[tensorProfile.name] = tensorProfile;
        }
        for tensorLocation: TensorLocation in locations {
            guard let tensorProfile: TensorProfile =
                profileByCanonicalName[tensorLocation.canonicalName] else {
                throw ArtifactValidationError.tensorMissing(
                    tensorName: tensorLocation.canonicalName, fileName: self.fileName);
            }
            guard let tensorMetadata: SafetensorsFraming.TensorView =
                self.tensorMetadataByStoredName[tensorLocation.storedName] else {
                throw ArtifactValidationError.tensorMissing(
                    tensorName: tensorLocation.canonicalName, fileName: self.fileName);
            }
            try ValidatedSafetensorsSource.validateProfile(
                profile: tensorProfile, metadata: tensorMetadata);
        }
    }

    /// Contiguity, per-tensor byte accounting, and whole-shard payload checks
    /// over the physical header alone.
    private static func validatePhysicalMetadata(
        tensorsByName: Dictionary<String, SafetensorsFraming.TensorView>,
        dataSectionStartBytes: UInt64, fileSizeBytes: UInt64,
        fileName: String) throws -> UInt64 {
        let orderedTensors: Array<(storedName: String, metadata: SafetensorsFraming.TensorView)> =
            tensorsByName
                .map({ (storedName: String, metadata: SafetensorsFraming.TensorView) -> (storedName: String, metadata: SafetensorsFraming.TensorView) in
                    return (storedName, metadata);
                })
                .sorted(by: { (leftEntry: (storedName: String, metadata: SafetensorsFraming.TensorView), rightEntry: (storedName: String, metadata: SafetensorsFraming.TensorView)) -> Bool in
                    if leftEntry.metadata.dataStartOffset() != rightEntry.metadata.dataStartOffset() {
                        return leftEntry.metadata.dataStartOffset() < rightEntry.metadata.dataStartOffset();
                    }
                    return leftEntry.storedName < rightEntry.storedName;
                });
        var expectedPayloadOffset: UInt64 = 0;
        for orderedTensor: (storedName: String, metadata: SafetensorsFraming.TensorView) in orderedTensors {
            let elementCount: UInt64 = try SafetensorsDtypeHelpers.elementCount(
                fromShape: orderedTensor.metadata.shape);
            let expectedBitsPerElement: UInt64 = try SafetensorsDtypeHelpers.dtypeBitsPerElement(
                dtypeString: orderedTensor.metadata.dtype, fileName: fileName,
                tensorName: orderedTensor.storedName);
            let (expectedBits, bitsOverflowed): (UInt64, Bool) = elementCount
                .multipliedReportingOverflow(by: expectedBitsPerElement);
            if bitsOverflowed {
                throw ArtifactValidationError.tensorPayloadSizeOverflow;
            }
            // SafeTensors requires every tensor payload to end on a complete byte.
            if expectedBits % 8 != 0 {
                throw ArtifactValidationError.safetensorsInvalidDataOffsets(
                    fileName: fileName, tensorName: orderedTensor.storedName,
                    dataStartOffset: orderedTensor.metadata.dataStartOffset(),
                    dataEndOffset: orderedTensor.metadata.dataEndOffset());
            }
            let expectedBytes: UInt64 = expectedBits / 8;
            let (actualIntervalBytes, intervalUnderflowed): (UInt64, Bool) = orderedTensor.metadata
                .dataEndOffset()
                .subtractingReportingOverflow(orderedTensor.metadata.dataStartOffset());
            if orderedTensor.metadata.dataStartOffset() != expectedPayloadOffset
                || intervalUnderflowed
                || actualIntervalBytes != expectedBytes {
                throw ArtifactValidationError.safetensorsInvalidDataOffsets(
                    fileName: fileName, tensorName: orderedTensor.storedName,
                    dataStartOffset: orderedTensor.metadata.dataStartOffset(),
                    dataEndOffset: orderedTensor.metadata.dataEndOffset());
            }
            expectedPayloadOffset = orderedTensor.metadata.dataEndOffset();
        }
        let (actualPayloadBytes, payloadUnderflowed): (UInt64, Bool) =
            fileSizeBytes.subtractingReportingOverflow(dataSectionStartBytes);
        if payloadUnderflowed {
            throw ArtifactValidationError.truncatedSafetensorsFile(
                fileName: fileName, expectedMinimumBytes: dataSectionStartBytes,
                actualFileSizeBytes: fileSizeBytes);
        }
        if expectedPayloadOffset != actualPayloadBytes {
            throw ArtifactValidationError.safetensorsPayloadLengthMismatch(
                fileName: fileName, declaredPayloadBytes: expectedPayloadOffset,
                actualPayloadBytes: actualPayloadBytes);
        }
        return actualPayloadBytes;
    }

    /// Profile checks after physical validation. A dtype mismatch inside an
    /// inventory location surfaces as an unexpected tensor because the
    /// canonical identity itself cannot be served from this source.
    private static func validateProfile(
        profile: TensorProfile, metadata: SafetensorsFraming.TensorView) throws -> Void {
        let acceptedDtypeNames: Set<String>;
        switch profile.dtype {
        case .affineQuantizationFloat, .modelFloat:
            acceptedDtypeNames = ["F16", "BF16", "F32"];
        case .bfloat16:
            acceptedDtypeNames = ["BF16"];
        case .float32:
            acceptedDtypeNames = ["F32"];
        case .uint32:
            acceptedDtypeNames = ["U32"];
        }
        if acceptedDtypeNames.contains(metadata.dtype) == false {
            throw ArtifactValidationError.unexpectedTensor(tensorName: profile.name);
        }
        let shapeMatches: Bool = metadata.shape == profile.shape
            || profile.equivalentPublishedShapes.contains(metadata.shape);
        if shapeMatches == false {
            throw ArtifactValidationError.tensorShapeMismatch(
                tensorName: profile.name, expectedShape: profile.shape,
                actualShape: metadata.shape);
        }
    }
}
