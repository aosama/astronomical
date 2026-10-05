import Foundation;
import Darwin;

/**
 * Family-owned shallow classification and executable discovery for the K2
 * Horizon MoVA family, porting crates/config/src/model_discovery/
 * k2_horizon_mova.rs. The caseless enum is the Swift equivalent of the Rust
 * module of free functions. Recognition is `model_type` only; completeness
 * requires a stacked affine index, shards, tokenizer, and standalone chat
 * template, and unstacked per-expert tensors stay unpublished.
 */
internal enum K2HorizonMova {

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

    /** Recognizes the K2 Horizon MoVA family marker without claiming execution support. */
    internal static func recognizesModelType(_ modelType: String?) -> Bool {
        guard let presentModelType: String = modelType else {
            return false;
        }
        return presentModelType == "k2_horizon_mova";
    }

    /**
     * Predicts whether startup can execute one stacked affine family member.
     * The shard index must carry the stacked tensor markers (switch MLP or
     * value experts); an index holding only unstacked per-expert tensors
     * stays unpublished. Nil means the directory is not a complete
     * executable family member.
     */
    internal static func discoverModelMetadata(
        modelDirectory: FilePath,
        configBytes: Data
    ) -> K2HorizonMova.DiscoveredModelMetadata? {
        let modelType: String;
        let maximumPositionEmbeddings: UInt32?;
        do {
            let configRootValue: Any = try DiscoveryStrictJsonDocument.parseDocument(bytes: configBytes);
            guard let configObject: Dictionary<String, Any> = configRootValue as? Dictionary<String, Any> else {
                return nil;
            }
            modelType = try StrictJson.requiredString(object: configObject, fieldName: "model_type");
            maximumPositionEmbeddings = try StrictJson.optionalUnsignedInteger(
                object: configObject,
                fieldName: "max_position_embeddings"
            );
        } catch {
            return nil;
        }
        if (modelType != "k2_horizon_mova") {
            return nil;
        }
        guard let contextWindowTokens: UInt32 = maximumPositionEmbeddings, contextWindowTokens >= 2 else {
            return nil;
        }
        if !DiscoveryPathNavigation.isExistingRegularFile(path: modelDirectory.appending(component: "tokenizer.json"))
            || !DiscoveryPathNavigation.isExistingRegularFile(path: modelDirectory.appending(component: "chat_template.jinja"))
            || !DiscoveryPathNavigation.isExistingRegularFile(path: modelDirectory.appending(component: "model.safetensors.index.json"))
        {
            return nil;
        }
        guard let indexWeightMap: Dictionary<String, Any> = K2HorizonMova.shardIndexWeightMap(modelDirectory: modelDirectory) else {
            return nil;
        }
        var sawStacked: Bool = false;
        var sawUnstackedOnly: Bool = false;
        var shardFileNames: Set<String> = Set<String>();
        for tensorName: String in indexWeightMap.keys {
            guard let tensorShardValue: Any = indexWeightMap[tensorName] else {
                continue;
            }
            if let shardFileName: String = tensorShardValue as? String {
                shardFileNames.insert(shardFileName);
            }
            if (tensorName.contains(".mlp.switch_mlp.") || tensorName.contains(".self_attn.v_experts.weight")) {
                sawStacked = true;
            }
            if (tensorName.contains(".mlp.experts.0.") || tensorName.contains(".self_attn.v_experts.0.")) {
                sawUnstackedOnly = true;
            }
        }
        if (sawUnstackedOnly && !sawStacked) {
            return nil;
        }
        if (!sawStacked) {
            return nil;
        }
        for shardFileName: String in shardFileNames {
            if !DiscoveryPathNavigation.isExistingRegularFile(path: modelDirectory.appending(component: shardFileName)) {
                return nil;
            }
        }
        var modelSizeBytes: UInt64 = 0;
        for shardFileName: String in shardFileNames {
            let shardPath: FilePath = modelDirectory.appending(component: shardFileName);
            guard let shardFileBytes: UInt64 = K2HorizonMova.regularFileSizeBytes(path: shardPath) else {
                return nil;
            }
            let additionOutcome: (partialValue: UInt64, overflow: Bool) = modelSizeBytes.addingReportingOverflow(shardFileBytes);
            if (additionOutcome.overflow) {
                return nil;
            }
            modelSizeBytes = additionOutcome.partialValue;
        }
        return K2HorizonMova.DiscoveredModelMetadata(
            contextWindowTokens: contextWindowTokens,
            maximumInputTokens: contextWindowTokens - 1,
            maximumOutputTokens: Swift.min(UInt32(UInt16.max), contextWindowTokens - 1),
            hasVision: false,
            supportsReasoning: true,
            supportsToolCalls: true,
            modelSizeBytes: modelSizeBytes
        );
    }

    /**
     * Reads and parses the shard index weight map. Nil mirrors the serde
     * derivation: a missing, unreadable, malformed, or shapeless index fails
     * the whole discovery.
     */
    private static func shardIndexWeightMap(modelDirectory: FilePath) -> Dictionary<String, Any>? {
        let indexPath: FilePath = modelDirectory.appending(component: "model.safetensors.index.json");
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
        do {
            return try StrictJson.objectValue(object: indexDocumentObject, fieldName: "weight_map");
        } catch {
            return nil;
        }
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
