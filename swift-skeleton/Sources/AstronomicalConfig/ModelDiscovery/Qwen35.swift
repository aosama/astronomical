import Foundation;
import Darwin;

/**
 * Family-owned shallow discovery rules for executable Qwen3.5 artifacts,
 * porting crates/config/src/model_discovery/qwen3_5.rs. The caseless enum is
 * the Swift equivalent of the Rust module of free functions and constants.
 */
internal enum Qwen35 {

    /**
     * Family-derived metadata returned to neutral discovery orchestration,
     * projected into the chat capabilities of the neutral discovered model.
     */
    internal struct DiscoveredModelMetadata: Equatable, Sendable {
        internal let contextWindowTokens: UInt32;
        internal let maximumInputTokens: UInt32;
        internal let maximumOutputTokens: UInt32;
        internal let hasVision: Bool;
        internal let supportsReasoning: Bool;
        internal let supportsToolCalls: Bool;
        internal let modelSizeBytes: UInt64;

        internal init(
            contextWindowTokens: UInt32,
            maximumInputTokens: UInt32,
            maximumOutputTokens: UInt32,
            hasVision: Bool,
            supportsReasoning: Bool,
            supportsToolCalls: Bool,
            modelSizeBytes: UInt64
        ) {
            self.contextWindowTokens = contextWindowTokens;
            self.maximumInputTokens = maximumInputTokens;
            self.maximumOutputTokens = maximumOutputTokens;
            self.hasVision = hasVision;
            self.supportsReasoning = supportsReasoning;
            self.supportsToolCalls = supportsToolCalls;
            self.modelSizeBytes = modelSizeBytes;
        }
    }

    /**
     * Lowest context window disk discovery and download preflight accept: one
     * token for the prompt and one for the completion. Anything smaller
     * cannot serve a single user turn.
     */
    internal static let MINIMUM_SERVABLE_CONTEXT_WINDOW_TOKENS: UInt64 = 2;

    /** Recognizes every Qwen model type served by the existing runtime. */
    internal static func recognizesModelType(_ modelType: String?) -> Bool {
        guard let presentModelType: String = modelType else {
            return false;
        }
        return presentModelType == "qwen3_5"
            || presentModelType == "qwen3_5_moe"
            || presentModelType == "qwen3_5_moe_vision";
    }

    /**
     * Validates shallow Qwen completeness and derives public discovery
     * metadata. Nil means the directory is not a complete executable Qwen
     * artifact.
     */
    internal static func discoverModelMetadata(
        modelDirectory: FilePath,
        configObject: Dictionary<String, Any>
    ) -> Qwen35.DiscoveredModelMetadata? {
        // A converted per-expert streaming revision declares itself with
        // manifest.json (format version 3) and carries no shard index; it
        // discovers through its manifest and resident weight bundle.
        if DiscoveryPathNavigation.isExistingRegularFile(path: modelDirectory.appending(component: "manifest.json")) {
            return Qwen35.discoverStreamingModelMetadata(modelDirectory: modelDirectory, configObject: configObject);
        }
        if !DiscoveryPathNavigation.isExistingRegularFile(path: modelDirectory.appending(component: "model.safetensors.index.json"))
            || !DiscoveryPathNavigation.isExistingRegularFile(path: modelDirectory.appending(component: "tokenizer.json"))
        {
            return nil;
        }
        let indexPath: FilePath = modelDirectory.appending(component: "model.safetensors.index.json");
        var hasVision: Bool = false;
        if let weightMap: Dictionary<String, Any> = Qwen35.shardIndexWeightMap(indexPath: indexPath) {
            let shardFileNames: Set<String> = Qwen35.allShardFileNames(weightMap: weightMap);
            let requiredShardFileNames: Set<String> = Qwen35.requiredShardFileNames(weightMap: weightMap);
            for shardFileName: String in shardFileNames {
                let shardPath: FilePath = modelDirectory.appending(component: shardFileName);
                if !DiscoveryPathNavigation.isExistingRegularFile(path: shardPath) && requiredShardFileNames.contains(shardFileName) {
                    return nil;
                }
            }
            for tensorName: String in weightMap.keys {
                if tensorName.hasPrefix("vision_tower.") {
                    hasVision = true;
                    break;
                }
            }
        }
        return Qwen35.assembledMetadata(hasVision: hasVision, configObject: configObject, modelDirectory: modelDirectory);
    }

    /**
     * Reads the effective context window exactly like disk discovery: the
     * text config carries it first, and a plain text-only config declares it
     * at the root. Zero means the document declares no usable context window.
     */
    internal static func contextWindowTokens(configObject: Dictionary<String, Any>) -> UInt64 {
        var declaredValue: Any? = nil;
        if let textConfigObject: Dictionary<String, Any> = configObject["text_config"] as? Dictionary<String, Any> {
            declaredValue = textConfigObject["max_position_embeddings"];
        }
        if declaredValue == nil {
            declaredValue = configObject["max_position_embeddings"];
        }
        guard let presentValue: Any = declaredValue else {
            return 0;
        }
        return Qwen35.unsignedInteger64Value(of: presentValue) ?? 0;
    }

    /**
     * Extracts the shard file names disk discovery and download preflight
     * both treat as mandatory: every shard that carries at least one tensor.
     * A missing shard never leaves discovery valid. Entries that do not map a
     * tensor to a string shard are skipped, mirroring disk discovery's
     * tolerance.
     */
    internal static func requiredShardFileNames(weightMap: Dictionary<String, Any>) -> Set<String> {
        var requiredShardFileNames: Set<String> = Set<String>();
        for (key: _, value: tensorShardValue) in weightMap {
            guard let shardFileName: String = tensorShardValue as? String else {
                continue;
            }
            requiredShardFileNames.insert(shardFileName);
        }
        return requiredShardFileNames;
    }

    private static func allShardFileNames(weightMap: Dictionary<String, Any>) -> Set<String> {
        var shardFileNames: Set<String> = Set<String>();
        for tensorShardValue: Any in weightMap.values {
            guard let shardFileName: String = tensorShardValue as? String else {
                continue;
            }
            shardFileNames.insert(shardFileName);
        }
        return shardFileNames;
    }

    /**
     * Discovers a converted per-expert streaming revision from its manifest
     * and resident weight bundle. The manifest replaces the shard index; the
     * vision bundle travels beside the resident weights unchanged.
     */
    private static func discoverStreamingModelMetadata(
        modelDirectory: FilePath,
        configObject: Dictionary<String, Any>
    ) -> Qwen35.DiscoveredModelMetadata? {
        let manifestPath: FilePath = modelDirectory.appending(component: "manifest.json");
        guard let manifestBytes: Data = FileManager.default.contents(atPath: manifestPath.string) else {
            return nil;
        }
        let manifestFormatVersion: UInt32;
        do {
            let parsedManifestDocument: Any = try DiscoveryStrictJsonDocument.parseDocument(bytes: manifestBytes);
            guard let manifestObject: Dictionary<String, Any> = parsedManifestDocument as? Dictionary<String, Any> else {
                return nil;
            }
            // serde's derive rejects a missing or mistyped format_version, so
            // the probe fails the same way for both.
            manifestFormatVersion = try StrictJson.requiredUnsignedInteger(object: manifestObject, fieldName: "format_version");
        } catch {
            return nil;
        }
        if manifestFormatVersion != 3 || !DiscoveryPathNavigation.isExistingRegularFile(path: modelDirectory.appending(component: "tokenizer.json")) {
            return nil;
        }
        if !DiscoveryPathNavigation.isExistingRegularFile(path: modelDirectory.appending(component: "resident.safetensors")) {
            return nil;
        }
        let hasVision: Bool = DiscoveryPathNavigation.isExistingRegularFile(
            path: modelDirectory.appending(component: "optiq").appending(component: "optiq_vision.safetensors")
        );
        return Qwen35.assembledMetadata(hasVision: hasVision, configObject: configObject, modelDirectory: modelDirectory);
    }

    /** Shared tail of both discovery paths: context-window checks and size measurement. */
    private static func assembledMetadata(
        hasVision: Bool,
        configObject: Dictionary<String, Any>,
        modelDirectory: FilePath
    ) -> Qwen35.DiscoveredModelMetadata? {
        let contextWindowTokens: UInt64 = Qwen35.contextWindowTokens(configObject: configObject);
        guard let contextWindow: UInt32 = UInt32(exactly: contextWindowTokens) else {
            return nil;
        }
        if contextWindowTokens < Qwen35.MINIMUM_SERVABLE_CONTEXT_WINDOW_TOKENS {
            return nil;
        }
        let protocolMaximumOutputTokens: UInt32 = min(UInt32(UInt16.max), contextWindow - 1);
        guard let modelSizeBytes: UInt64 = Qwen35.measureModelSafetensorsBytes(modelDirectory: modelDirectory) else {
            return nil;
        }
        return Qwen35.DiscoveredModelMetadata(
            contextWindowTokens: contextWindow,
            maximumInputTokens: contextWindow - 1,
            maximumOutputTokens: protocolMaximumOutputTokens,
            hasVision: hasVision,
            // The Qwen text processor owns both structured output contracts.
            supportsReasoning: true,
            supportsToolCalls: true,
            modelSizeBytes: modelSizeBytes
        );
    }

    /**
     * Reads and parses the shard index weight map. Nil mirrors the Rust
     * `else { false }` arm: a missing, unreadable, malformed, or shapeless
     * index only suppresses vision detection and shard validation.
     */
    private static func shardIndexWeightMap(indexPath: FilePath) -> Dictionary<String, Any>? {
        guard let indexBytes: Data = FileManager.default.contents(atPath: indexPath.string) else {
            return nil;
        }
        let parsedIndexDocument: Any;
        do {
            parsedIndexDocument = try DiscoveryStrictJsonDocument.parseDocument(bytes: indexBytes);
        } catch {
            return nil;
        }
        guard let indexDocumentObject: Dictionary<String, Any> = parsedIndexDocument as? Dictionary<String, Any> else {
            return nil;
        }
        return indexDocumentObject["weight_map"] as? Dictionary<String, Any>;
    }

    /**
     * Measures the total bytes of every weight payload under the model
     * directory. Nil means the measurement itself failed (an unreadable
     * directory or entry, an unresolvable path, or a size overflow), which
     * fails the whole discovery rather than reporting a partial size.
     */
    private static func measureModelSafetensorsBytes(modelDirectory: FilePath) -> UInt64? {
        var pendingDirectories: Array<FilePath> = Array<FilePath>();
        pendingDirectories.append(modelDirectory);
        var measuredEntryPaths: Set<String> = Set<String>();
        var modelSizeBytes: UInt64 = 0;
        while (pendingDirectories.isEmpty == false) {
            let pendingDirectory: FilePath = pendingDirectories.removeLast();
            let entryNames: Array<String>;
            do {
                entryNames = try FileManager.default.contentsOfDirectory(atPath: pendingDirectory.string);
            } catch {
                return nil;
            }
            for entryName: String in entryNames {
                let entryPath: FilePath = pendingDirectory.appending(component: entryName);
                // Directory entries report their own type without following
                // symlinks, so a symlinked subdirectory is not descended.
                if DiscoveryPathNavigation.isSymlinkTargetedDirectory(path: entryPath) {
                    pendingDirectories.append(entryPath);
                    continue;
                }
                let entryNameExtension: String = (entryName as NSString).pathExtension;
                let isWeightPayloadFile: Bool = entryNameExtension == "safetensors";
                if !isWeightPayloadFile || !DiscoveryPathNavigation.isExistingRegularFile(path: entryPath) {
                    continue;
                }
                guard let canonicalEntryPath: String = Qwen35.canonicalizedPath(path: entryPath) else {
                    return nil;
                }
                let insertionOutcome: (inserted: Bool, memberAfterInsert: String) = measuredEntryPaths.insert(canonicalEntryPath);
                if (insertionOutcome.inserted) {
                    guard let entryFileBytes: UInt64 = Qwen35.regularFileSizeBytes(path: entryPath) else {
                        return nil;
                    }
                    let additionOutcome: (partialValue: UInt64, overflow: Bool) = modelSizeBytes.addingReportingOverflow(entryFileBytes);
                    if (additionOutcome.overflow) {
                        return nil;
                    }
                    modelSizeBytes = additionOutcome.partialValue;
                }
            }
        }
        return modelSizeBytes;
    }

    /**
     * serde_json's `Value::as_u64` over the JSONSerialization-shaped tree:
     * only numbers stored as unsigned integers qualify, so a whole-valued
     * float, a negative integer, a boolean, and a non-number all miss.
     */
    private static func unsignedInteger64Value(of jsonValue: Any) -> UInt64? {
        if (jsonValue is NSNull) {
            return nil;
        }
        guard let numberValue: NSNumber = jsonValue as? NSNumber else {
            return nil;
        }
        if CFGetTypeID(numberValue as CFTypeRef) == CFBooleanGetTypeID() {
            return nil;
        }
        let storageTypeDescription: String = String(cString: numberValue.objCType);
        if storageTypeDescription == "d" || storageTypeDescription == "f" {
            return nil;
        }
        if storageTypeDescription == "Q" {
            return numberValue.uint64Value;
        }
        let storedSignedValue: Int64 = numberValue.int64Value;
        guard storedSignedValue >= 0 else {
            return nil;
        }
        return UInt64(storedSignedValue);
    }

    /** `fs::canonicalize`: nil when the path cannot be resolved. */
    private static func canonicalizedPath(path: FilePath) -> String? {
        guard let resolvedPathBuffer: UnsafeMutablePointer<CChar> = realpath(path.string, nil) else {
            return nil;
        }
        let resolvedPath: String = String(cString: resolvedPathBuffer);
        free(resolvedPathBuffer);
        return resolvedPath;
    }

    /** `fs::metadata(path).len()`: symlink-following file size in bytes. */
    private static func regularFileSizeBytes(path: FilePath) -> UInt64? {
        var pathStatus: stat = stat();
        guard Darwin.fstatat(Darwin.AT_FDCWD, path.string, &pathStatus, 0) == 0 else {
            return nil;
        }
        return UInt64(pathStatus.st_size);
    }
}
