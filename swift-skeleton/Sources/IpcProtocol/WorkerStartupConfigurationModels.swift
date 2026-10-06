import Foundation;

/// Logging verbosity supplied by the supervisor when starting a worker.
public enum WorkerLogLevel: Equatable {
    case error;
    case warn;
    case info;
    case debug;
    case trace;

    private static let expectedVariantNames: Array<String> = ["error", "warn", "info", "debug", "trace"];

    internal func wireValue() -> JsonWireValue {
        switch (self) {
        case .error: return .string("error");
        case .warn: return .string("warn");
        case .info: return .string("info");
        case .debug: return .string("debug");
        case .trace: return .string("trace");
        }
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerLogLevel {
        let wireName = try JsonWireValue.extractString(wireValue);
        switch wireName {
        case "error": return .error;
        case "warn": return .warn;
        case "info": return .info;
        case "debug": return .debug;
        case "trace": return .trace;
        default:
            throw JsonWireProblem.malformedDocument(problem: "unknown variant `\(wireName)`, expected one of \(JsonWireProblem.formattedFieldList(WorkerLogLevel.expectedVariantNames))");
        }
    }
}

/// Worker-acknowledged feature settings safe to expose through local status.
///
/// This intentionally excludes startup paths and model locations. It proves the
/// effective policy of the worker process that will serve requests.
public struct WorkerRuntimeFeatureConfiguration: Equatable {
    /// Semantic generation applied by this exact worker process.
    public let configurationGeneration: String;
    /// Whether the worker will persist and restore ordinary prompt state.
    public let persistentPromptCacheEnabled: Bool;
    /// Effective global cache capacity, without disclosing its local directory.
    public let promptCacheMaximumSizeBytes: UInt64;
    /// Present only after a swap binds one concrete model policy.
    public let loadedModel: WorkerLoadedModelRuntimeConfiguration?;

    public init(
        configurationGeneration: String,
        persistentPromptCacheEnabled: Bool,
        promptCacheMaximumSizeBytes: UInt64,
        loadedModel: WorkerLoadedModelRuntimeConfiguration?
    ) {
        self.configurationGeneration = configurationGeneration;
        self.persistentPromptCacheEnabled = persistentPromptCacheEnabled;
        self.promptCacheMaximumSizeBytes = promptCacheMaximumSizeBytes;
        self.loadedModel = loadedModel;
    }

    public func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "configuration_generation", value: .string(self.configurationGeneration));
        wireObject.appendEntry(key: "persistent_prompt_cache_enabled", value: .boolean(self.persistentPromptCacheEnabled));
        wireObject.appendEntry(key: "prompt_cache_maximum_size_bytes", value: .unsignedInteger(self.promptCacheMaximumSizeBytes));
        wireObject.appendEntry(key: "loaded_model", value: WorkerRuntimeFeatureConfiguration.optionalLoadedModelWireValue(self.loadedModel));
        return .object(wireObject);
    }

    // This struct carries no deny_unknown_fields in Rust, so serde silently
    // ignores unrecognized keys; decode mirrors that leniency on purpose.
    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerRuntimeFeatureConfiguration {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedConfiguration = WorkerRuntimeFeatureConfiguration(
            configurationGeneration: try wireObject.decodeString(fieldName: "configuration_generation"),
            persistentPromptCacheEnabled: try wireObject.decodeBool(fieldName: "persistent_prompt_cache_enabled"),
            promptCacheMaximumSizeBytes: try wireObject.decodeUInt64(fieldName: "prompt_cache_maximum_size_bytes"),
            loadedModel: try WorkerRuntimeFeatureConfiguration.decodeOptionalLoadedModel(wireObject: wireObject));
        return parsedConfiguration;
    }

    private static func decodeOptionalLoadedModel(wireObject: JsonWireObject) throws -> WorkerLoadedModelRuntimeConfiguration? {
        guard let loadedModelObject = try wireObject.decodeOptionalObject(fieldName: "loaded_model") else {
            return nil;
        }
        return try WorkerLoadedModelRuntimeConfiguration.fromWireValue(.object(loadedModelObject));
    }

    private static func optionalLoadedModelWireValue(_ optionalValue: WorkerLoadedModelRuntimeConfiguration?) -> JsonWireValue {
        guard let unwrappedValue = optionalValue else {
            return .null;
        }
        return unwrappedValue.wireValue();
    }
}

/// Fully resolved worker-owned startup settings.
public struct WorkerStartupConfiguration: Equatable {
    public let configurationGeneration: String;
    public let globalPromptCacheRootDirectory: String;
    public let globalPromptCacheMaximumSizeBytes: UInt64;
    public let persistentPromptCacheEnabled: Bool;
    public let configuredMaximumMlxMemoryBytes: UInt64?;
    public let performanceAttributionEnabled: Bool;
    public let loggingDirectory: String;
    public let loggingLevel: WorkerLogLevel;
    public let retainedLogFileCount: Int;

    public init(
        configurationGeneration: String,
        globalPromptCacheRootDirectory: String,
        globalPromptCacheMaximumSizeBytes: UInt64,
        persistentPromptCacheEnabled: Bool,
        configuredMaximumMlxMemoryBytes: UInt64?,
        performanceAttributionEnabled: Bool,
        loggingDirectory: String,
        loggingLevel: WorkerLogLevel,
        retainedLogFileCount: Int
    ) {
        self.configurationGeneration = configurationGeneration;
        self.globalPromptCacheRootDirectory = globalPromptCacheRootDirectory;
        self.globalPromptCacheMaximumSizeBytes = globalPromptCacheMaximumSizeBytes;
        self.persistentPromptCacheEnabled = persistentPromptCacheEnabled;
        self.configuredMaximumMlxMemoryBytes = configuredMaximumMlxMemoryBytes;
        self.performanceAttributionEnabled = performanceAttributionEnabled;
        self.loggingDirectory = loggingDirectory;
        self.loggingLevel = loggingLevel;
        self.retainedLogFileCount = retainedLogFileCount;
    }

    internal static let wireFieldNames: Array<String> = [
        "configuration_generation", "global_prompt_cache_root_directory",
        "global_prompt_cache_maximum_size_bytes", "persistent_prompt_cache_enabled",
        "configured_maximum_mlx_memory_bytes", "performance_attribution_enabled",
        "logging_directory", "logging_level", "retained_log_file_count",
    ];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "configuration_generation", value: .string(self.configurationGeneration));
        wireObject.appendEntry(key: "global_prompt_cache_root_directory", value: .string(self.globalPromptCacheRootDirectory));
        wireObject.appendEntry(key: "global_prompt_cache_maximum_size_bytes", value: .unsignedInteger(self.globalPromptCacheMaximumSizeBytes));
        wireObject.appendEntry(key: "persistent_prompt_cache_enabled", value: .boolean(self.persistentPromptCacheEnabled));
        wireObject.appendEntry(key: "configured_maximum_mlx_memory_bytes", value: WorkerStartupConfiguration.optionalUInt64WireValue(self.configuredMaximumMlxMemoryBytes));
        wireObject.appendEntry(key: "performance_attribution_enabled", value: .boolean(self.performanceAttributionEnabled));
        wireObject.appendEntry(key: "logging_directory", value: .string(self.loggingDirectory));
        wireObject.appendEntry(key: "logging_level", value: self.loggingLevel.wireValue());
        wireObject.appendEntry(key: "retained_log_file_count", value: .unsignedInteger(UInt64(self.retainedLogFileCount)));
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerStartupConfiguration {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedConfiguration = WorkerStartupConfiguration(
            configurationGeneration: try wireObject.decodeString(fieldName: "configuration_generation"),
            globalPromptCacheRootDirectory: try wireObject.decodeString(fieldName: "global_prompt_cache_root_directory"),
            globalPromptCacheMaximumSizeBytes: try wireObject.decodeUInt64(fieldName: "global_prompt_cache_maximum_size_bytes"),
            persistentPromptCacheEnabled: try wireObject.decodeBool(fieldName: "persistent_prompt_cache_enabled"),
            configuredMaximumMlxMemoryBytes: try wireObject.decodeOptionalUInt64(fieldName: "configured_maximum_mlx_memory_bytes"),
            performanceAttributionEnabled: try wireObject.decodeBool(fieldName: "performance_attribution_enabled"),
            loggingDirectory: try wireObject.decodeString(fieldName: "logging_directory"),
            loggingLevel: try WorkerLogLevel.fromWireValue(try wireObject.requireObjectValue(fieldName: "logging_level")),
            retainedLogFileCount: Int(try wireObject.decodeUInt64(fieldName: "retained_log_file_count")));
        try wireObject.rejectUnknownFields(allowedFieldNames: WorkerStartupConfiguration.wireFieldNames);
        return parsedConfiguration;
    }

    private static func optionalUInt64WireValue(_ optionalValue: UInt64?) -> JsonWireValue {
        guard let unwrappedValue = optionalValue else {
            return .null;
        }
        return .unsignedInteger(unwrappedValue);
    }
}
