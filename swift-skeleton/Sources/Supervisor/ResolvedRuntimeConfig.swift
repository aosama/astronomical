import Foundation;

import AstronomicalConfig;
import IpcProtocol;

/// Immutable snapshot of every resolved runtime value the supervisor needs
/// to decide whether a config reload requires a worker restart, a full REST
/// API restart, or only an in-place update.
///
/// Migrates the struct half of apps/supervisor/src/config_reload.rs.
public struct ResolvedRuntimeConfig: Equatable, Sendable {

    /// Privacy-safe identity of the accepted semantic configuration document.
    public var configurationGeneration: String;
    /// Resolved worker executable path used to spawn or replace the worker.
    public var workerExecutablePath: FilePath;
    /// All models discovered from the resolved directories.
    public var discoveredModels: Array<DiscoveryDiscoveredModel>;
    /// Path-safe authored-root conflicts excluded from executable discovery.
    public var modelDiscoveryDiagnostics: Array<DiscoveryModelDiscoveryDiagnostic>;
    /// Configured discovery roots, including roots that contain no model.
    public var configuredModelDirectories: Array<FilePath>;
    /// Canonical requestable identity to resolved execution policy.
    public var modelPolicyCatalog: Dictionary<String, RuntimeModelPolicy>;
    /// Configured preferences retained while their canonical target is absent.
    public var unmatchedModelConfigIds: Array<String>;
    /// Optional user-configured MLX memory ceiling in exact decimal SI bytes.
    public var maximumMlxMemoryBytes: UInt64?;
    /// Performance attribution preference captured by the worker at startup.
    public var performanceAttributionEnabled: Bool;
    /// Whether completion attribution captures emitted tool calls.
    public var completionAttributionEnabled: Bool;
    /// Whether REST requests may read the optional Qwen thinking-channel seed.
    public var experimentalQwenThinkingChannelSeedEnabled: Bool;
    /// Whether the worker may read and write the persistent prompt cache.
    public var persistentPromptCacheEnabled: Bool;
    /// Authored cache toggle before the enabled-by-default policy is applied.
    public var configuredPersistentPromptCacheEnabled: Bool?;
    /// Authored cache capacity before the 50 GB default is applied.
    public var configuredPromptCacheMaximumSizeBytes: UInt64?;
    /// Resolved SSD-backed prompt-cache policy (worker-startup field).
    public var promptCacheConfig: PromptCacheConfig;
    /// Resolved supervisor bind address (REST API restart required to change).
    public var bindAddress: String;
    /// The typed endpoint behind `bindAddress`; the REST listener binds this.
    public var bindEndpoint: SocketEndpoint;
    /// Resolved logging policy (REST API restart required to change).
    public var loggingConfig: LoggingConfig;

    public init(
        configurationGeneration: String,
        workerExecutablePath: FilePath,
        discoveredModels: Array<DiscoveryDiscoveredModel>,
        modelDiscoveryDiagnostics: Array<DiscoveryModelDiscoveryDiagnostic>,
        configuredModelDirectories: Array<FilePath>,
        modelPolicyCatalog: Dictionary<String, RuntimeModelPolicy>,
        unmatchedModelConfigIds: Array<String>,
        maximumMlxMemoryBytes: UInt64?,
        performanceAttributionEnabled: Bool,
        completionAttributionEnabled: Bool,
        experimentalQwenThinkingChannelSeedEnabled: Bool,
        persistentPromptCacheEnabled: Bool,
        configuredPersistentPromptCacheEnabled: Bool?,
        configuredPromptCacheMaximumSizeBytes: UInt64?,
        promptCacheConfig: PromptCacheConfig,
        bindAddress: String,
        bindEndpoint: SocketEndpoint,
        loggingConfig: LoggingConfig
    ) {
        self.configurationGeneration = configurationGeneration;
        self.workerExecutablePath = workerExecutablePath;
        self.discoveredModels = discoveredModels;
        self.modelDiscoveryDiagnostics = modelDiscoveryDiagnostics;
        self.configuredModelDirectories = configuredModelDirectories;
        self.modelPolicyCatalog = modelPolicyCatalog;
        self.unmatchedModelConfigIds = unmatchedModelConfigIds;
        self.maximumMlxMemoryBytes = maximumMlxMemoryBytes;
        self.performanceAttributionEnabled = performanceAttributionEnabled;
        self.completionAttributionEnabled = completionAttributionEnabled;
        self.experimentalQwenThinkingChannelSeedEnabled = experimentalQwenThinkingChannelSeedEnabled;
        self.persistentPromptCacheEnabled = persistentPromptCacheEnabled;
        self.configuredPersistentPromptCacheEnabled = configuredPersistentPromptCacheEnabled;
        self.configuredPromptCacheMaximumSizeBytes = configuredPromptCacheMaximumSizeBytes;
        self.promptCacheConfig = promptCacheConfig;
        self.bindAddress = bindAddress;
        self.bindEndpoint = bindEndpoint;
        self.loggingConfig = loggingConfig;
    }

    /// Converts supervisor-resolved worker settings into the IPC bootstrap DTO.
    public func workerStartupConfiguration() -> WorkerStartupConfiguration {
        let loggingLevel: WorkerLogLevel;
        switch (self.loggingConfig.level) {
        case .error: loggingLevel = .error;
        case .warn: loggingLevel = .warn;
        case .info: loggingLevel = .info;
        case .debug: loggingLevel = .debug;
        case .trace: loggingLevel = .trace;
        }
        return WorkerStartupConfiguration(
            configurationGeneration: self.configurationGeneration,
            globalPromptCacheRootDirectory: self.promptCacheConfig.globalPromptCacheRootDirectory.string,
            globalPromptCacheMaximumSizeBytes: self.promptCacheConfig.globalPromptCacheMaximumSizeBytes,
            persistentPromptCacheEnabled: self.persistentPromptCacheEnabled,
            configuredMaximumMlxMemoryBytes: self.maximumMlxMemoryBytes,
            performanceAttributionEnabled: self.performanceAttributionEnabled,
            loggingDirectory: self.loggingConfig.directory.string,
            loggingLevel: loggingLevel,
            retainedLogFileCount: self.loggingConfig.retainedFiles);
    }
}
