import Foundation;
import IpcProtocol;

/// Strict metadata for the measured portion of the pinned OptiQ bit map, port
/// of crates/model-serving/src/qwen3_5/quantizations/optiq/metadata.rs.

private let MAXIMUM_OPTIQ_METADATA_BYTES: Int = 128 * 1024;

public struct OptiQMetadata {
    private let measuredModuleProfileEntries: SortedModuleProfiles;

    fileprivate init(measuredModuleProfiles: SortedModuleProfiles) {
        self.measuredModuleProfileEntries = measuredModuleProfiles;
    }

    /// Parses and validates the complete bounded OptiQ metadata document.
    ///
    /// Two document shapes are accepted. A document carrying `per_layer` is a
    /// measured bit-map and keeps the exact strict contract: unknown fields are
    /// rejected and every measured profile must match the config. A document
    /// without `per_layer` is provenance-only (expert-compressed variants such
    /// as REAP-pruned or merged artifacts publish compression-run evidence
    /// here); it makes no per-module quantization claims, so it contributes
    /// zero measured modules and does not constrain execution. The byte bound
    /// applies to both shapes.
    public static func fromJsonBytes(metadataBytes: Array<UInt8>) throws -> OptiQMetadata {
        if metadataBytes.count > MAXIMUM_OPTIQ_METADATA_BYTES {
            throw OptiQMetadataError.metadataTooLarge(
                actualSizeBytes: metadataBytes.count, maximumSizeBytes: MAXIMUM_OPTIQ_METADATA_BYTES);
        }
        do {
            let metadataDocument: OptiQMetadataDocument = try OptiQMetadataDocument.decoded(
                wireValue: try JsonWireParser.parseDocument(documentBytes: Data(metadataBytes)));
            var measuredModuleProfiles: SortedModuleProfiles = SortedModuleProfiles(entries: Array());
            for moduleEntry: (moduleName: String, overrideValue: OptiQQuantizationOverride) in metadataDocument.perLayer {
                if isMlxAffineQuantizationGroupSizeSupported(groupSize: moduleEntry.overrideValue.groupSize) == false {
                    throw OptiQMetadataError.unsupportedGroupSize(
                        moduleName: moduleEntry.moduleName,
                        actualGroupSize: moduleEntry.overrideValue.groupSize);
                }
                if isMlxAffineQuantizationBitWidthSupported(bitWidth: moduleEntry.overrideValue.bits) == false {
                    throw OptiQMetadataError.unsupportedBits(
                        moduleName: moduleEntry.moduleName, actualBits: moduleEntry.overrideValue.bits);
                }
                measuredModuleProfiles.insert(
                    profile: OptiQQuantizationProfile(
                        bits: moduleEntry.overrideValue.bits,
                        groupSize: moduleEntry.overrideValue.groupSize),
                    forKey: moduleEntry.moduleName);
            }
            return OptiQMetadata(measuredModuleProfiles: measuredModuleProfiles);
        } catch let metadataProblem as OptiQMetadataError {
            throw metadataProblem;
        } catch {
            return try Self.provenanceOnlyFromJsonBytes(
                metadataBytes: metadataBytes,
                strictDeserializationProblem: error);
        }
    }

    /// Accepts a bounded provenance-only document that declares no measured
    /// bit-map. Strict parsing failed, so the document must be a JSON object
    /// without any `per_layer` key: such a document makes no per-module
    /// quantization claims, so it cannot constrain execution and contributes
    /// zero measured modules. A malformed document that does claim measurements
    /// still fails with the strict error unchanged.
    private static func provenanceOnlyFromJsonBytes(
        metadataBytes: Array<UInt8>,
        strictDeserializationProblem: Error) throws -> OptiQMetadata {
        guard let jsonValue: JsonWireValue = try? JsonWireParser.parseDocument(
            documentBytes: Data(metadataBytes)) else {
            throw OptiQMetadataError.deserializeMetadata;
        }
        var isProvenanceOnly: Bool = false;
        if case let .object(documentObject) = jsonValue {
            isProvenanceOnly = documentObject.value(forKey: "per_layer") == nil;
        }
        if isProvenanceOnly {
            return OptiQMetadata(measuredModuleProfiles: SortedModuleProfiles(entries: Array()));
        }
        // A malformed document claiming measurements fails with the strict
        // error; mirror the Rust shape by rethrowing the original problem.
        _ = strictDeserializationProblem;
        throw OptiQMetadataError.deserializeMetadata;
    }

    /// Returns the number of sensitivity-measured quantized modules.
    public func measuredModuleCount() -> Int {
        return self.measuredModuleProfileEntries.count;
    }

    /// Requires every declared measured module profile to equal a config override.
    public func validateAgainstConfig(qwen3_5Config: Qwen3_5Config) throws -> Void {
        var expectedMeasuredModuleProfiles: Array<(moduleName: String, profile: OptiQQuantizationProfile)> = Array();
        for moduleEntry: (moduleName: String, profile: OptiQQuantizationProfile) in qwen3_5Config.quantizedModuleProfiles().entries {
            if moduleEntry.profile.isUnquantized() == false {
                expectedMeasuredModuleProfiles.append(moduleEntry);
            }
        }
        for measuredEntry: (moduleName: String, profile: OptiQQuantizationProfile) in self.measuredModuleProfileEntries.entries {
            guard let expectedProfile: OptiQQuantizationProfile = expectedMeasuredModuleProfiles
                .first(where: { (entry: (moduleName: String, profile: OptiQQuantizationProfile)) -> Bool in entry.moduleName == measuredEntry.moduleName })?
                .profile else {
                throw OptiQMetadataError.unexpectedMeasuredModule(moduleName: measuredEntry.moduleName);
            }
            if measuredEntry.profile.bits != expectedProfile.bits {
                throw OptiQMetadataError.configBitMismatch(
                    moduleName: measuredEntry.moduleName,
                    configBits: expectedProfile.bits,
                    metadataBits: measuredEntry.profile.bits);
            }
            if measuredEntry.profile.groupSize != expectedProfile.groupSize {
                throw OptiQMetadataError.configGroupSizeMismatch(
                    moduleName: measuredEntry.moduleName,
                    configGroupSize: expectedProfile.groupSize,
                    metadataGroupSize: measuredEntry.profile.groupSize);
            }
        }
    }
}

/// Strict wire shape of the OptiQ metadata document.
private struct OptiQMetadataDocument {
    var perLayer: Array<(moduleName: String, overrideValue: OptiQQuantizationOverride)>;

    static func decoded(wireValue: JsonWireValue) throws -> OptiQMetadataDocument {
        let documentObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        try documentObject.rejectUnknownFields(allowedFieldNames: [
            "method", "base_model", "reference", "sensitivity_measured_on",
            "target_bpw", "achieved_bpw", "n_high_bits", "n_low_bits", "threshold", "per_layer"]);
        _ = try documentObject.decodeString(fieldName: "method");
        _ = try documentObject.decodeString(fieldName: "base_model");
        _ = try documentObject.decodeString(fieldName: "reference");
        _ = try documentObject.decodeOptionalStringAllowingAbsent(fieldName: "sensitivity_measured_on");
        _ = try Self.decodedFloat64Field(documentObject, fieldName: "target_bpw");
        _ = try Self.decodedFloat64Field(documentObject, fieldName: "achieved_bpw");
        _ = try Self.decodedUsizeField(documentObject, fieldName: "n_high_bits");
        _ = try Self.decodedUsizeField(documentObject, fieldName: "n_low_bits");
        _ = try Self.decodedFloat64Field(documentObject, fieldName: "threshold");
        var perLayerEntries: Array<(moduleName: String, overrideValue: OptiQQuantizationOverride)> = Array();
        let perLayerObject: JsonWireObject = try documentObject.decodeObject(fieldName: "per_layer");
        for propertyName: String in perLayerObject.keyNames {
            let overrideObject: JsonWireObject = try JsonWireValue.extractObject(
                try perLayerObject.requireObjectValue(fieldName: propertyName));
            try overrideObject.rejectUnknownFields(allowedFieldNames: ["bits", "group_size"]);
            perLayerEntries.append((
                propertyName,
                OptiQQuantizationOverride(
                    groupSize: try overrideObject.decodeUInt32(fieldName: "group_size"),
                    bits: try overrideObject.decodeUInt32(fieldName: "bits"))));
        }
        perLayerEntries.sort { (leftEntry: (moduleName: String, overrideValue: OptiQQuantizationOverride), rightEntry: (moduleName: String, overrideValue: OptiQQuantizationOverride)) -> Bool in
            return Array(leftEntry.moduleName.utf8).lexicographicallyPrecedes(Array(rightEntry.moduleName.utf8));
        };
        return OptiQMetadataDocument(perLayer: perLayerEntries);
    }

    private static func decodedFloat64Field(_ documentObject: JsonWireObject, fieldName propertyName: String) throws -> Double {
        return try JsonWireValue.extractFloat64(
            try documentObject.requireObjectValue(fieldName: propertyName));
    }

    private static func decodedUsizeField(_ documentObject: JsonWireObject, fieldName propertyName: String) throws -> Int {
        let rawValue: UInt64 = try documentObject.decodeUInt64(fieldName: propertyName);
        guard rawValue <= UInt64(Int.max) else {
            throw JsonWireProblem.invalidType(
                expectedTypeName: "usize", found: "integer `\(rawValue)`");
        }
        return Int(rawValue);
    }
}

/// A strict mismatch in the pinned OptiQ sensitivity metadata.
public enum OptiQMetadataError: Error, Equatable {
    case metadataTooLarge(actualSizeBytes: Int, maximumSizeBytes: Int);
    case deserializeMetadata;
    case unsupportedGroupSize(moduleName: String, actualGroupSize: UInt32);
    case unsupportedBits(moduleName: String, actualBits: UInt32);
    case configBitMismatch(moduleName: String, configBits: UInt32, metadataBits: UInt32);
    case configGroupSizeMismatch(moduleName: String, configGroupSize: UInt32, metadataGroupSize: UInt32);
    case unexpectedMeasuredModule(moduleName: String);

    public var errorDescription: String? {
        switch self {
        case .metadataTooLarge(let actualSizeBytes, let maximumSizeBytes):
            return "OptiQ metadata is \(actualSizeBytes) bytes, exceeding \(maximumSizeBytes)";
        case .deserializeMetadata:
            return "failed to decode OptiQ metadata JSON";
        case .unsupportedGroupSize(let moduleName, let actualGroupSize):
            return "OptiQ metadata module '\(moduleName)' uses unsupported group size \(actualGroupSize)";
        case .unsupportedBits(let moduleName, let actualBits):
            return "OptiQ metadata module '\(moduleName)' uses unsupported \(actualBits)-bit quantization";
        case .configBitMismatch(let moduleName, let configBits, let metadataBits):
            return "OptiQ module '\(moduleName)' is \(configBits)-bit in config and \(metadataBits)-bit in metadata";
        case .configGroupSizeMismatch(let moduleName, let configGroupSize, let metadataGroupSize):
            return "OptiQ module '\(moduleName)' has group size \(configGroupSize) in config and \(metadataGroupSize) in metadata";
        case .unexpectedMeasuredModule(let moduleName):
            return "OptiQ metadata contains unexpected measured module '\(moduleName)'";
        }
    }
}
