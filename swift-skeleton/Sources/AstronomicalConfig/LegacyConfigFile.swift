import Foundation;

/// Typed view of the unversioned (legacy) configuration document, porting
/// LegacyConfigFile from crates/config/src/legacy_config_migration.rs. Unknown
/// keys are rejected like serde's deny_unknown_fields. The three legacy
/// boolean switches use present-or-absent semantics: an explicit JSON null on
/// them is a parse failure, matching the Rust deserialize_present_boolean
/// helper that only accepts real booleans.
internal struct LegacyConfigFile {

    internal var modelDirectories: Array<String>;
    internal var maximumOutputTokens: UInt32?;
    internal var chunking: ChunkingConfigFile;
    internal var performanceAttributionEnabled: Bool?;
    internal var persistentPromptCacheEnabled: Bool?;
    internal var maximumMlxMemoryGb: UInt64?;
    internal var mtpEnabled: Bool?;
    internal var mtpDraftDepth: UInt8?;
    internal var supervisor: LegacySupervisorConfigFile?;
    internal var promptCacheMaxSizeGb: UInt64?;
    internal var logging: LegacyLoggingConfigFile?;

    internal init() {
        self.modelDirectories = Array<String>();
        self.maximumOutputTokens = nil;
        self.chunking = LegacyConfigFile.defaultLegacyChunkingConfigFile();
        self.performanceAttributionEnabled = nil;
        self.persistentPromptCacheEnabled = nil;
        self.maximumMlxMemoryGb = nil;
        self.mtpEnabled = nil;
        self.mtpDraftDepth = nil;
        self.supervisor = nil;
        self.promptCacheMaxSizeGb = nil;
        self.logging = nil;
    }

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> LegacyConfigFile {
        try StrictJson.requireKnownKeys(
            object: jsonObject,
            knownKeys: [
                "model_directories",
                "max_output_tokens",
                "chunking",
                "performance_attribution_enabled",
                "persistent_prompt_cache_enabled",
                "maximum_mlx_memory_gb",
                "mtp_enabled",
                "mtp_draft_depth",
                "supervisor",
                "prompt_cache_max_size_gb",
                "logging"
            ],
            fieldName: "legacy configuration"
        );
        var legacyConfigFile = LegacyConfigFile();
        legacyConfigFile.modelDirectories = try parseLegacyModelDirectories(jsonObject: jsonObject);
        legacyConfigFile.maximumOutputTokens = try StrictJson.optionalUnsignedInteger(object: jsonObject, fieldName: "max_output_tokens");
        legacyConfigFile.chunking = try LegacyConfigFile.parseLegacyChunking(jsonObject: jsonObject);
        legacyConfigFile.performanceAttributionEnabled = try LegacyConfigFile.parsePresentBoolean(object: jsonObject, fieldName: "performance_attribution_enabled");
        legacyConfigFile.persistentPromptCacheEnabled = try LegacyConfigFile.parsePresentBoolean(object: jsonObject, fieldName: "persistent_prompt_cache_enabled");
        legacyConfigFile.maximumMlxMemoryGb = try StrictJson.optionalUnsignedInteger(object: jsonObject, fieldName: "maximum_mlx_memory_gb");
        legacyConfigFile.mtpEnabled = try LegacyConfigFile.parsePresentBoolean(object: jsonObject, fieldName: "mtp_enabled");
        legacyConfigFile.mtpDraftDepth = try StrictJson.optionalUnsignedInteger(object: jsonObject, fieldName: "mtp_draft_depth");
        legacyConfigFile.supervisor = try StrictJson.decodeOptional(
            StrictJson.optionalObject(object: jsonObject, fieldName: "supervisor"),
            decode: LegacySupervisorConfigFile.fromJsonObject
        );
        legacyConfigFile.promptCacheMaxSizeGb = try StrictJson.optionalUnsignedInteger(object: jsonObject, fieldName: "prompt_cache_max_size_gb");
        legacyConfigFile.logging = try StrictJson.decodeOptional(
            StrictJson.optionalObject(object: jsonObject, fieldName: "logging"),
            decode: LegacyLoggingConfigFile.fromJsonObject
        );
        return legacyConfigFile;
    }

    private static func defaultLegacyChunkingConfigFile() -> ChunkingConfigFile {
        return ChunkingConfigFile(
            fixedPromptProcessingChunkSizeTokens: nil,
            fixedSsdStreamingPromptProcessingChunkSizeTokens: nil,
            fullAttentionKeyValueGrowthTokens: nil,
            prefillGraphSubmissionLayerInterval: nil,
            experimentalSsdPagingPrefillGraphSubmissionLayerInterval: nil,
            experimentalSsdPagingGenerationGraphSubmissionLayerInterval: nil,
            promptCacheBlockTokens: nil,
            promptCacheCommonPrefixStrideBlocks: nil,
            experimentalDecodeStageAttributionEnabled: nil,
            experimentalQuantizedKvCacheEnabled: nil,
            experimentalFusedMoeDecodeEnabled: nil
        );
    }

    private static func parseLegacyModelDirectories(jsonObject: Dictionary<String, Any>) throws -> Array<String> {
        if jsonObject["model_directories"] == nil {
            return Array<String>();
        }
        return try StrictJson.requiredStringArray(object: jsonObject, fieldName: "model_directories");
    }

    private static func parseLegacyChunking(jsonObject: Dictionary<String, Any>) throws -> ChunkingConfigFile {
        let presentChunkingObject: Dictionary<String, Any>? = try StrictJson.optionalObject(object: jsonObject, fieldName: "chunking");
        guard let unwrappedChunkingObject = presentChunkingObject else {
            return LegacyConfigFile.defaultLegacyChunkingConfigFile();
        }
        return try ChunkingConfigFile.fromJsonObject(unwrappedChunkingObject);
    }
}

/// Legacy supervisor section; only bind_address exists and v1 cannot
/// represent it, so the migration validator rejects any value.
internal struct LegacySupervisorConfigFile {

    internal let bindAddress: String?;

    internal init(bindAddress: String?) {
        self.bindAddress = bindAddress;
    }

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> LegacySupervisorConfigFile {
        try StrictJson.requireKnownKeys(object: jsonObject, knownKeys: ["bind_address"], fieldName: "supervisor");
        return LegacySupervisorConfigFile(bindAddress: try StrictJson.optionalString(object: jsonObject, fieldName: "bind_address"));
    }
}

/// Legacy logging section; the level falls back to warn when absent, matching
/// LogLevel's serde default.
internal struct LegacyLoggingConfigFile {

    internal let level: LogLevel;
    internal let retainedFiles: Int32?;

    internal init(level: LogLevel, retainedFiles: Int32?) {
        self.level = level;
        self.retainedFiles = retainedFiles;
    }

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> LegacyLoggingConfigFile {
        try StrictJson.requireKnownKeys(object: jsonObject, knownKeys: ["level", "retained_files"], fieldName: "logging");
        let parsedLevel: LogLevel;
        if let presentLevelName: String = try StrictJson.optionalString(object: jsonObject, fieldName: "level") {
            guard let wireNameLevel: LogLevel = LogLevel.fromWireName(presentLevelName) else {
                throw StrictJsonError(fieldName: "level", problem: "unknown log level \(presentLevelName)");
            }
            parsedLevel = wireNameLevel;
        } else {
            parsedLevel = LogLevel.defaultValue;
        }
        return LegacyLoggingConfigFile(level: parsedLevel, retainedFiles: try StrictJson.optionalSignedInteger(object: jsonObject, fieldName: "retained_files"));
    }
}

/// A legacy boolean switch is accepted only when it holds a real boolean;
/// an explicit null is a parse failure, mirroring deserialize_present_boolean.
internal extension LegacyConfigFile {

    static func parsePresentBoolean(object: Dictionary<String, Any>, fieldName: String) throws -> Bool? {
        guard object[fieldName] != nil else {
            return nil;
        }
        if object[fieldName] is NSNull {
            throw StrictJsonError(fieldName: fieldName, problem: "must be a boolean");
        }
        return try StrictJson.optionalBoolean(object: object, fieldName: fieldName);
    }
}
