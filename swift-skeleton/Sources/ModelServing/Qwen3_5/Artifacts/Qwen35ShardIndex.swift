import Foundation;
import IpcProtocol;

/// The bounded executable tensor-to-shard inventory for a Qwen3.5 artifact,
/// port of crates/model-serving/src/qwen3_5/artifacts/shard_index.rs.
///
/// Tracks both language model tensors (`language_model.*`) and vision tower
/// tensors (`vision_tower.*`) declared only by the main shard index. The
/// strict-json duplicate rejection the Rust side gains from
/// DuplicateAwareJsonValue is inherent in JsonWireParser, which rejects a
/// repeated object key before replacement.
public struct Qwen3_5ShardIndex: Equatable {
    private let MAXIMUM_TENSOR_NAME_BYTES: Int = 512;
    public static let MAXIMUM_INDEX_BYTES: Int = 1024 * 1024;

    /// Name-keyed maps kept in Rust BTreeMap (UTF-8 byte) order.
    private var languageTensorShardEntries: Array<(tensorName: String, shardFileName: String)>;
    private var visionTensorShardEntries: Array<(tensorName: String, shardFileName: String)>;
    private let totalPayloadBytesValue: UInt64;
    private let modelShardFileNamesValue: Array<String>;
    private let visionSidecarFileNamesValue: Array<String>;

    /// Parses and independently validates the executable language inventory.
    /// Also collects vision tower tensor mappings for embedded and sidecar storage.
    public static func fromJsonBytes(
        indexBytes: Array<UInt8>,
        languageTensorProfiles: Array<TensorProfile>) throws -> Qwen3_5ShardIndex {
        if indexBytes.count > Qwen3_5ShardIndex.MAXIMUM_INDEX_BYTES {
            throw Qwen3_5ArtifactError.indexTooLarge(
                actualIndexBytes: indexBytes.count,
                maximumIndexBytes: Qwen3_5ShardIndex.MAXIMUM_INDEX_BYTES);
        }
        let indexDocument: ShardIndexDocument;
        do {
            indexDocument = try ShardIndexDocument.decoded(
                wireValue: try JsonWireParser.parseDocument(documentBytes: Data(indexBytes)));
        } catch let jsonWireProblem as JsonWireProblem {
            throw Qwen3_5ArtifactError.deserializeIndex(problem: jsonWireProblem.description);
        } catch {
            throw Qwen3_5ArtifactError.deserializeIndex(problem: "unexpected index document shape");
        }
        let totalPayloadBytes: UInt64 = indexDocument.metadata.totalSize;
        // Collect actual model shard file names from the index rather than
        // hardcoding names. The index remains authoritative for logical tensor
        // ownership; physical duplicate tensors are handled during validation.
        var languageTensorNames: Set<String> = Set();
        var languageTensorShardEntries: Array<(tensorName: String, shardFileName: String)> = Array();
        var visionTensorShardEntries: Array<(tensorName: String, shardFileName: String)> = Array();
        var languageShardFileNames: Set<String> = Set();
        var visionShardFileNames: Set<String> = Set();
        // Iterate the weight map in BTreeMap (UTF-8 byte) order.
        for weightEntry: (tensorName: String, shardFileName: String) in indexDocument.weightMap {
            try Qwen3_5ShardIndex.validateTensorName(tensorName: weightEntry.tensorName);
            if weightEntry.tensorName.hasPrefix("language_model.") {
                languageShardFileNames.insert(weightEntry.shardFileName);
                languageTensorNames.insert(weightEntry.tensorName);
                languageTensorShardEntries.append(weightEntry);
            } else if weightEntry.tensorName.hasPrefix("vision_tower.") {
                // Embedded vision tensors may share language shards or occupy
                // dedicated vision files. A file containing only vision tensors
                // is loaded through the separate vision-tower path, regardless
                // of its filename.
                visionShardFileNames.insert(weightEntry.shardFileName);
                visionTensorShardEntries.append(weightEntry);
            }
            // Other tensor prefixes are silently skipped; the language
            // tensor-name validation below fails closed on unknown names.
        }
        try Qwen3_5TensorSpec.validateLanguageTensorNames(
            actualLanguageTensorNames: languageTensorNames,
            languageTensorProfiles: languageTensorProfiles);
        let byteOrderedSort: ((String, String) -> Bool) = { (leftName: String, rightName: String) -> Bool in
            return Array(leftName.utf8).lexicographicallyPrecedes(Array(rightName.utf8));
        };
        let sortedByteOrder: (String, String) -> Bool = byteOrderedSort;
        let visionOnlyShardFileNames: Array<String> = visionShardFileNames
            .subtracting(languageShardFileNames)
            .sorted(by: sortedByteOrder);
        let modelShardFileNames: Array<String> = languageShardFileNames
            .sorted(by: sortedByteOrder);
        func sortedByName(_ entries: Array<(tensorName: String, shardFileName: String)>) -> Array<(tensorName: String, shardFileName: String)> {
            return entries.sorted(by: { (leftEntry: (tensorName: String, shardFileName: String), rightEntry: (tensorName: String, shardFileName: String)) -> Bool in
                return Array(leftEntry.tensorName.utf8).lexicographicallyPrecedes(Array(rightEntry.tensorName.utf8));
            });
        }
        return Qwen3_5ShardIndex(
            languageTensorShardEntries: sortedByName(languageTensorShardEntries),
            visionTensorShardEntries: sortedByName(visionTensorShardEntries),
            totalPayloadBytesValue: totalPayloadBytes,
            modelShardFileNamesValue: modelShardFileNames,
            visionSidecarFileNamesValue: visionOnlyShardFileNames);
    }

    private init(
        languageTensorShardEntries: Array<(tensorName: String, shardFileName: String)>,
        visionTensorShardEntries: Array<(tensorName: String, shardFileName: String)>,
        totalPayloadBytesValue: UInt64, modelShardFileNamesValue: Array<String>,
        visionSidecarFileNamesValue: Array<String>) {
        self.languageTensorShardEntries = languageTensorShardEntries;
        self.visionTensorShardEntries = visionTensorShardEntries;
        self.totalPayloadBytesValue = totalPayloadBytesValue;
        self.modelShardFileNamesValue = modelShardFileNamesValue;
        self.visionSidecarFileNamesValue = visionSidecarFileNamesValue;
    }

    /// Returns the total payload bytes declared by the index.
    public func totalPayloadBytes() -> UInt64 {
        return self.totalPayloadBytesValue;
    }

    /// Returns the executable text-model tensor count.
    public func tensorCount() -> Int {
        return self.languageTensorShardEntries.count;
    }

    /// Returns the count of executable text-model tensors.
    public func languageTensorCount() -> Int {
        return self.languageTensorShardEntries.count;
    }

    /// Returns the vision tower tensor name to shard file name mapping in byte order.
    public func visionTensorNameToShardFileName() -> Array<(tensorName: String, shardFileName: String)> {
        return self.visionTensorShardEntries;
    }

    /// Returns vision tensor names that belong to one indexed file.
    public func visionTensorNamesForShard(shardFileName: String) -> Array<String> {
        return self.visionTensorShardEntries
            .filter({ (entry: (tensorName: String, shardFileName: String)) -> Bool in entry.shardFileName == shardFileName })
            .map({ (entry: (tensorName: String, shardFileName: String)) -> String in entry.tensorName });
    }

    /// Returns executable model shard file names in sorted order.
    ///
    /// Includes target-language files and language files with embedded vision.
    public func modelShardFileNames() -> Array<String> {
        return self.modelShardFileNamesValue;
    }

    /// Returns vision-only files that are loaded separately from language shards.
    public func visionSidecarFileNames() -> Array<String> {
        return self.visionSidecarFileNamesValue;
    }

    /// Returns whether the indexed file is a separately loaded vision file.
    public func isVisionSidecarFile(shardFileName: String) -> Bool {
        return self.visionSidecarFileNamesValue.contains(shardFileName);
    }

    /// Returns the mapping from language tensor names to their containing shard
    /// file names in byte order. Used by expert paging to locate weight tensors
    /// in shard files.
    public func languageTensorNameToShardFileName() -> Array<(tensorName: String, shardFileName: String)> {
        return self.languageTensorShardEntries;
    }

    /// Returns the number of executable model shards.
    public func shardCount() -> Int {
        return self.modelShardFileNamesValue.count;
    }

    /// Resolves one exact tensor name to its shard file.
    public func shardFileNameForTensor(tensorName: String) -> String? {
        return self.languageTensorShardEntries
            .first(where: { (entry: (tensorName: String, shardFileName: String)) -> Bool in entry.tensorName == tensorName })?
            .shardFileName;
    }

    /// Returns language tensor names that belong to one shard.
    public func languageTensorNamesForShard(shardFileName: String) -> Array<String> {
        return self.languageTensorShardEntries
            .filter({ (entry: (tensorName: String, shardFileName: String)) -> Bool in
                return entry.shardFileName == shardFileName && entry.tensorName.hasPrefix("language_model.");
            })
            .map({ (entry: (tensorName: String, shardFileName: String)) -> String in entry.tensorName });
    }

    /// Extracts the set of language tensor names from the safetensors index JSON
    /// without performing any validation against tensor profiles.
    ///
    /// This is used to determine which target modules are quantized vs.
    /// unquantized by checking for affine companion tensors before the full
    /// validation pass that requires complete tensor profiles.
    public static func extractLanguageTensorNamesFromJson(
        indexBytes: Array<UInt8>) throws -> Set<String> {
        if indexBytes.count > Qwen3_5ShardIndex.MAXIMUM_INDEX_BYTES {
            throw Qwen3_5ArtifactError.indexTooLarge(
                actualIndexBytes: indexBytes.count,
                maximumIndexBytes: Qwen3_5ShardIndex.MAXIMUM_INDEX_BYTES);
        }
        let indexDocument: ShardIndexDocument;
        do {
            indexDocument = try ShardIndexDocument.decoded(
                wireValue: try JsonWireParser.parseDocument(documentBytes: Data(indexBytes)));
        } catch let jsonWireProblem as JsonWireProblem {
            throw Qwen3_5ArtifactError.deserializeIndex(problem: jsonWireProblem.description);
        } catch {
            throw Qwen3_5ArtifactError.deserializeIndex(problem: "unexpected index document shape");
        }
        var languageTensorNames: Set<String> = Set();
        for weightEntry: (tensorName: String, shardFileName: String) in indexDocument.weightMap {
            if weightEntry.tensorName.hasPrefix("language_model.") {
                languageTensorNames.insert(weightEntry.tensorName);
            }
        }
        return languageTensorNames;
    }

    private static func validateTensorName(tensorName: String) throws -> Void {
        let maximumTensorNameBytes: Int = 512;
        if tensorName.isEmpty || tensorName.utf8.count > maximumTensorNameBytes {
            throw Qwen3_5ArtifactError.invalidTensorNameLength(
                tensorName: tensorName, maximumTensorNameBytes: maximumTensorNameBytes);
        }
    }

    public static func == (lhsValue: Qwen3_5ShardIndex, rhsValue: Qwen3_5ShardIndex) -> Bool {
        func entriesEqual(_ leftEntries: Array<(tensorName: String, shardFileName: String)>, _ rightEntries: Array<(tensorName: String, shardFileName: String)>) -> Bool {
            return leftEntries.elementsEqual(rightEntries, by: { (leftEntry: (tensorName: String, shardFileName: String), rightEntry: (tensorName: String, shardFileName: String)) -> Bool in
                return leftEntry.tensorName == rightEntry.tensorName
                    && leftEntry.shardFileName == rightEntry.shardFileName;
            });
        }
        return lhsValue.totalPayloadBytesValue == rhsValue.totalPayloadBytesValue
            && entriesEqual(lhsValue.languageTensorShardEntries, rhsValue.languageTensorShardEntries)
            && entriesEqual(lhsValue.visionTensorShardEntries, rhsValue.visionTensorShardEntries)
            && lhsValue.modelShardFileNamesValue == rhsValue.modelShardFileNamesValue
            && lhsValue.visionSidecarFileNamesValue == rhsValue.visionSidecarFileNamesValue;
    }
}

/// The strict shard-index wire document.
private struct ShardIndexDocument {
    var metadata: ShardIndexMetadata;
    var weightMap: Array<(tensorName: String, shardFileName: String)>;

    static func decoded(wireValue: JsonWireValue) throws -> ShardIndexDocument {
        let documentObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        try documentObject.rejectUnknownFields(allowedFieldNames: ["metadata", "weight_map"]);
        let metadataObject: JsonWireObject = try documentObject.decodeObject(fieldName: "metadata");
        try metadataObject.rejectUnknownFields(allowedFieldNames: ["total_size", "total_parameters"]);
        let weightMapObject: JsonWireObject = try documentObject.decodeObject(fieldName: "weight_map");
        var weightMapEntries: Array<(tensorName: String, shardFileName: String)> = Array();
        for propertyName: String in weightMapObject.keyNames {
            weightMapEntries.append((
                propertyName,
                try weightMapObject.decodeString(fieldName: propertyName)));
        }
        weightMapEntries.sort { (leftEntry: (tensorName: String, shardFileName: String), rightEntry: (tensorName: String, shardFileName: String)) -> Bool in
            return Array(leftEntry.tensorName.utf8).lexicographicallyPrecedes(Array(rightEntry.tensorName.utf8));
        };
        return ShardIndexDocument(
            metadata: ShardIndexMetadata(
                totalSize: try metadataObject.decodeUInt64(fieldName: "total_size"),
                totalParameters: try metadataObject.decodeOptionalUInt64AllowingAbsent(fieldName: "total_parameters")),
            weightMap: weightMapEntries);
    }
}

private struct ShardIndexMetadata {
    var totalSize: UInt64;
    var totalParameters: UInt64?;
}
