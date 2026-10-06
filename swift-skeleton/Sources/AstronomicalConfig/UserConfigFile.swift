import Foundation;

/**
 * Root of the strict schema-version-1 user configuration document, the Swift
 * port of the Rust `UserConfigFile`. Field names on the wire follow the
 * serde snake_case names of the v1 schema exactly.
 */
internal struct UserConfigFile: Equatable {
    internal let schemaReference: String;
    internal let schemaVersion: UInt32;
    internal let runtime: RuntimeConfigFile;
    internal let promptCache: PromptCacheConfigFile?;
    internal let chunking: ChunkingConfigFile?;
    internal let models: Dictionary<String, ModelConfigFile>;
    internal let diagnostics: DiagnosticsConfigFile?;

    internal init(
        schemaReference: String,
        schemaVersion: UInt32,
        runtime: RuntimeConfigFile,
        promptCache: PromptCacheConfigFile?,
        chunking: ChunkingConfigFile?,
        models: Dictionary<String, ModelConfigFile>,
        diagnostics: DiagnosticsConfigFile?
    ) {
        self.schemaReference = schemaReference;
        self.schemaVersion = schemaVersion;
        self.runtime = runtime;
        self.promptCache = promptCache;
        self.chunking = chunking;
        self.models = models;
        self.diagnostics = diagnostics;
    }

    /**
     * Document written on first run. The chunking defaults live here until
     * the chunking-config slice lands its own resolution unit.
     */
    internal static func minimal() -> UserConfigFile {
        return UserConfigFile(
            schemaReference: "./astronomical-config.schema.json",
            schemaVersion: 1,
            runtime: RuntimeConfigFile(
                modelDirectories: Array<String>(),
                maximumMlxMemoryGb: nil,
                defaultModel: nil,
                experimentalQwenThinkingChannelSeedEnabled: nil
            ),
            promptCache: nil,
            chunking: ChunkingConfigFile(
                fixedPromptProcessingChunkSizeTokens: UserConfigFile.DEFAULT_FIXED_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS,
                fixedSsdStreamingPromptProcessingChunkSizeTokens: UserConfigFile.DEFAULT_FIXED_SSD_STREAMING_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS,
                fullAttentionKeyValueGrowthTokens: nil,
                prefillGraphSubmissionLayerInterval: UserConfigFile.DEFAULT_PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL,
                experimentalSsdPagingPrefillGraphSubmissionLayerInterval: UserConfigFile.DEFAULT_EXPERIMENTAL_SSD_PAGING_PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL,
                experimentalSsdPagingGenerationGraphSubmissionLayerInterval: nil,
                promptCacheBlockTokens: nil,
                promptCacheCommonPrefixStrideBlocks: nil,
                experimentalDecodeStageAttributionEnabled: nil,
                experimentalQuantizedKvCacheEnabled: nil,
                experimentalFusedMoeDecodeEnabled: nil
            ),
            models: Dictionary<String, ModelConfigFile>(),
            diagnostics: nil
        );
    }

    private static let DEFAULT_FIXED_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS: UInt32 = 2048;
    private static let DEFAULT_FIXED_SSD_STREAMING_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS: UInt32 = 2048;
    private static let DEFAULT_PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL: UInt32 = 0;
    private static let DEFAULT_EXPERIMENTAL_SSD_PAGING_PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL: UInt32 = 1;

    internal static func fromJsonObject(_ jsonObject: Any) throws -> UserConfigFile {
        guard let rootObject: Dictionary<String, Any> = jsonObject as? Dictionary<String, Any> else {
            throw StrictJsonError(fieldName: "config", problem: "top-level value must be a JSON object");
        }
        try StrictJson.requireKnownKeys(
            object: rootObject,
            knownKeys: ["$schema", "schema_version", "runtime", "prompt_cache", "chunking", "models", "diagnostics"],
            fieldName: ""
        );
        let runtimeObject: Dictionary<String, Any> = try StrictJson.objectValue(object: rootObject, fieldName: "runtime");
        let promptCacheObject: Dictionary<String, Any>? = try StrictJson.optionalObject(
            object: rootObject,
            fieldName: "prompt_cache"
        );
        let chunkingObject: Dictionary<String, Any>? = try StrictJson.optionalObject(
            object: rootObject,
            fieldName: "chunking"
        );
        let modelsObject: Dictionary<String, Any>? = try StrictJson.optionalObject(
            object: rootObject,
            fieldName: "models"
        );
        let diagnosticsObject: Dictionary<String, Any>? = try StrictJson.optionalObject(
            object: rootObject,
            fieldName: "diagnostics"
        );
        var loadedModels: Dictionary<String, ModelConfigFile> = Dictionary<String, ModelConfigFile>();
        if let presentModelsObject: Dictionary<String, Any> = modelsObject {
            for modelId: String in presentModelsObject.keys {
                guard let modelObjectValue: Any = presentModelsObject[modelId] else {
                    throw StrictJsonError(fieldName: "models." + modelId, problem: "is required");
                }
                guard let modelObject: Dictionary<String, Any> = modelObjectValue as? Dictionary<String, Any> else {
                    throw StrictJsonError(fieldName: "models." + modelId, problem: "must be an object");
                }
                loadedModels[modelId] = try ModelConfigFile.fromJsonObject(modelObject);
            }
        }
        return UserConfigFile(
            schemaReference: try StrictJson.requiredString(object: rootObject, fieldName: "$schema"),
            schemaVersion: try StrictJson.requiredUnsignedInteger(object: rootObject, fieldName: "schema_version"),
            runtime: try RuntimeConfigFile.fromJsonObject(runtimeObject),
            promptCache: try StrictJson.decodeOptional(
                promptCacheObject,
                decode: PromptCacheConfigFile.fromJsonObject
            ),
            chunking: try StrictJson.decodeOptional(
                chunkingObject,
                decode: ChunkingConfigFile.fromJsonObject
            ),
            models: loadedModels,
            diagnostics: try StrictJson.decodeOptional(
                diagnosticsObject,
                decode: DiagnosticsConfigFile.fromJsonObject
            )
        );
    }

    internal func toJsonObject() -> Any {
        var jsonObject: Dictionary<String, Any> = Dictionary<String, Any>();
        jsonObject["$schema"] = self.schemaReference;
        jsonObject["schema_version"] = self.schemaVersion;
        jsonObject["runtime"] = self.runtime.toJsonObject();
        if let promptCache: PromptCacheConfigFile = self.promptCache {
            jsonObject["prompt_cache"] = promptCache.toJsonObject();
        }
        if let chunking: ChunkingConfigFile = self.chunking {
            jsonObject["chunking"] = chunking.toJsonObject();
        }
        if (self.models.isEmpty == false) {
            var modelsObject: Dictionary<String, Any> = Dictionary<String, Any>();
            for (key: modelId, value: modelConfigFile) in self.models {
                modelsObject[modelId] = modelConfigFile.toJsonObject();
            }
            jsonObject["models"] = modelsObject;
        }
        if let diagnostics: DiagnosticsConfigFile = self.diagnostics {
            jsonObject["diagnostics"] = diagnostics.toJsonObject();
        }
        return jsonObject;
    }
}
