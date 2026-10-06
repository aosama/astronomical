import Foundation;

/**
 * Loaded configuration boundary for one Astronomical instance, the Swift
 * port of the Rust `AstronomicalConfig::load_from_instance_paths` journey.
 * The struct name intentionally repeats the module name the way swift-log's
 * `Logger` ecosystem types do, so client code reads
 * `AstronomicalConfig.loadFromInstancePaths(...)` after one import.
 *
 * MIGRATION MARKER — the worker-startup resolution fields are ported
 * (prompt cache, logging, MLX ceiling, attribution, generation digest);
 * still deferred to their own slices: chunking-config resolution and
 * persist-back, resolved-model-config inheritance, model-policy catalog
 * resolution, and the reload diff. The configuration-generation digest
 * feeds the resolved generation later, which also mixes in the discovered
 * models and policy catalog.
 */
public struct AstronomicalConfig {
    private let loadedInstancePaths: AstronomicalInstancePaths;
    private let loadedUserConfigFile: UserConfigFile;

    internal init(instancePaths: AstronomicalInstancePaths, userConfigFile: UserConfigFile) {
        self.loadedInstancePaths = instancePaths;
        self.loadedUserConfigFile = userConfigFile;
    }

    /** Loads the instance config, creating the first-run document when absent. */
    public static func loadFromInstancePaths(_ instancePaths: AstronomicalInstancePaths) throws -> AstronomicalConfig {
        let userConfigFile: UserConfigFile = try ConfigFileStore.readUserConfigFile(
            configFilePath: instancePaths.configFilePath
        );
        return AstronomicalConfig(instancePaths: instancePaths, userConfigFile: userConfigFile);
    }

    /**
     * Loads the Development channel for `homeDirectory`. Only the
     * `.astronomical-dev` state beneath it is ever read or written.
     */
    public static func loadFromDevelopmentHomeDirectory(_ homeDirectory: FilePath) throws -> AstronomicalConfig {
        let developmentInstancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            homeDirectory,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );
        return try loadFromInstancePaths(developmentInstancePaths);
    }

    public var instancePaths: AstronomicalInstancePaths {
        return self.loadedInstancePaths;
    }

    public var modelDirectories: Array<FilePath> {
        var directoryPaths: Array<FilePath> = Array<FilePath>();
        for modelDirectoryString: String in self.loadedUserConfigFile.runtime.modelDirectories {
            directoryPaths.append(FilePath(string: modelDirectoryString));
        }
        return directoryPaths;
    }

    public func supervisorBindAddress() throws -> SocketEndpoint {
        let supervisorBindAddress: SocketEndpoint = self.loadedInstancePaths.defaultBindAddress;
        guard (supervisorBindAddress.host.hasPrefix("127.") || supervisorBindAddress.host == "::1") else {
            throw AstronomicalConfigError.nonLoopbackBindAddress(supervisorBindAddress: supervisorBindAddress);
        }
        return supervisorBindAddress;
    }

    /// Canonical digest of the accepted configuration document. The resolved
    /// generation later mixes this with the discovered models and policy
    /// catalog, so the raw digest stays the config-only fingerprint.
    public func generation() throws -> String {
        return try ConfigurationGeneration.configurationGeneration(
            configFilePath: self.loadedInstancePaths.configFilePath,
            userConfigFile: self.loadedUserConfigFile);
    }

    /// Resolved prompt-cache policy for worker startup: the instance cache
    /// root plus the configured decimal-SI capacity, defaulting to 50 GB.
    public func promptCache() throws -> PromptCacheConfig {
        let globalPromptCacheMaximumSizeBytes: UInt64 = try PromptCacheResolution.promptCacheMaximumSizeGbToBytes(
            self.loadedUserConfigFile.promptCache?.maximumSizeGb
                ?? PromptCacheResolution.defaultPromptCacheMaximumSizeGb);
        return PromptCacheConfig(
            rootDirectory: self.loadedInstancePaths.promptCacheDirectory,
            maximumSizeBytes: globalPromptCacheMaximumSizeBytes);
    }

    /// Whether the SSD-backed prompt cache is enabled; absent means enabled.
    public func persistentPromptCacheEnabled() -> Bool {
        return self.loadedUserConfigFile.promptCache?.enabled ?? true;
    }

    /// The authored cache toggle without replacing absence with policy.
    public func configuredPersistentPromptCacheEnabled() -> Bool? {
        return self.loadedUserConfigFile.promptCache?.enabled;
    }

    /// The authored decimal-SI cache capacity without applying its default.
    public func configuredPromptCacheMaximumSizeBytes() throws -> UInt64? {
        guard let configuredMaximumSizeGb: UInt64 = self.loadedUserConfigFile.promptCache?.maximumSizeGb else {
            return nil;
        }
        return try PromptCacheResolution.promptCacheMaximumSizeGbToBytes(configuredMaximumSizeGb);
    }

    /// Resolved bounded hourly file logging for the supervisor or worker.
    public func logging() -> LoggingConfig {
        let configuredDiagnostics: DiagnosticsConfigFile? = self.loadedUserConfigFile.diagnostics;
        let configuredLogLevel: LogLevel? = configuredDiagnostics?.logLevel.flatMap { (logLevelWireName: String) -> LogLevel? in
            return LogLevel.fromWireName(logLevelWireName);
        };
        let retainedLogFiles: Int = configuredDiagnostics?.retainedLogFiles.map { (retainedLogFiles: Int32) -> Int in
            return Int(retainedLogFiles)
        } ?? LoggingConfig.defaultRetainedLogFiles;
        return LoggingConfig(
            directory: self.loadedInstancePaths.loggingDirectory,
            level: configuredLogLevel ?? LogLevel.defaultValue,
            retainedFiles: retainedLogFiles);
    }

    /// Optional user-configured MLX memory ceiling in exact decimal SI bytes;
    /// `nil` means startup or live control should use the machine maximum.
    public func maximumMlxMemoryBytes() throws -> UInt64? {
        guard let configuredMaximumMlxMemoryGb: UInt64 = self.loadedUserConfigFile.runtime.maximumMlxMemoryGb else {
            return nil;
        }
        return try MaximumMlxMemory.maximumMlxMemoryGbToBytes(configuredMaximumMlxMemoryGb);
    }

    /// Whether detailed critical-path performance attribution is enabled;
    /// attribution defaults to disabled so normal inference avoids timing
    /// and serialization work unless explicitly requested.
    public func performanceAttributionEnabled() -> Bool {
        return self.loadedUserConfigFile.diagnostics?.performanceAttributionEnabled ?? false;
    }

    /// Whether completion attribution is enabled; defaults to disabled so
    /// normal inference never pays the attribution cost.
    public func completionAttributionEnabled() -> Bool {
        return self.loadedUserConfigFile.diagnostics?.completionAttributionEnabled ?? false;
    }

    /// Whether REST requests may read the optional Qwen thinking-channel seed.
    public func experimentalQwenThinkingChannelSeedEnabled() -> Bool {
        return self.loadedUserConfigFile.runtime.experimentalQwenThinkingChannelSeedEnabled ?? false;
    }

    /// Canonical model identities the config carries preferences for.
    public var configuredModelIds: Array<String> {
        return Array<String>(self.loadedUserConfigFile.models.keys).sorted();
    }
}
