import Foundation

import AstronomicalConfig;

/// Arguments for `astronomical validate config`, porting validate_config_arguments.rs.
public struct ValidateConfigArguments: Equatable {

    public let runtimeInstance: AstronomicalRuntimeInstance;
    public let renderJson: Bool;

    public init(runtimeInstance: AstronomicalRuntimeInstance, renderJson: Bool) {
        self.runtimeInstance = runtimeInstance;
        self.renderJson = renderJson;
    }
}

enum ValidateConfigArgumentParser {

    static func parseValidateConfigArguments(
        _ remainingArguments: Array<String>
    ) -> Result<ValidateConfigArguments, UsageError> {
        var runtimeInstance: AstronomicalRuntimeInstance = .development;
        var instanceFlagSeen: Bool = false;
        var renderJson: Bool = false;

        var argumentIndex: Int = 0;
        while (argumentIndex < remainingArguments.count) {
            let argumentText: String = remainingArguments[argumentIndex];
            switch (argumentText) {
            case "--instance":
                if (instanceFlagSeen) {
                    return .failure(.repeatedArgument("--instance"));
                }
                guard argumentIndex + 1 < remainingArguments.count else {
                    return .failure(.missingValue("--instance"));
                }
                let rawInstanceName: String = remainingArguments[argumentIndex + 1];
                switch (rawInstanceName) {
                case "stable":
                    runtimeInstance = .stable;
                case "development":
                    runtimeInstance = .development;
                default:
                    return .failure(.unknownInstance(rawInstanceName));
                }
                instanceFlagSeen = true;
                argumentIndex += 2;
            case "--json":
                renderJson = true;
                argumentIndex += 1;
            default:
                return .failure(.unknownArgument(argumentText));
            }
        }

        return .success(ValidateConfigArguments(runtimeInstance: runtimeInstance, renderJson: renderJson));
    }
}

/// Failures of `astronomical validate config`, porting the error half of validate_config.rs.
public enum ValidateConfigError: Error, CustomStringConvertible {

    case configFileMissing(configFilePath: String)
    case configFileUnreadable(configFilePath: String, cause: String)
    case invalidConfiguration(configFilePath: String, cause: String)
    case inconsistentConfiguration(cause: String)
    case reportWriteFailed

    public var description: String {
        switch (self) {
        case let .configFileMissing(configFilePath):
            return "No configuration file at \(configFilePath). Start Astronomical once to create one."
        case let .configFileUnreadable(configFilePath, cause):
            return "could not read \(configFilePath): \(cause)"
        case let .invalidConfiguration(configFilePath, cause):
            return "invalid configuration at \(configFilePath): \(cause)"
        case let .inconsistentConfiguration(cause):
            return "configuration is internally inconsistent: \(cause)"
        case .reportWriteFailed:
            return "could not write report"
        }
    }
}

/**
 * `astronomical validate config`, porting validate_config.rs: loads one
 * instance's configuration file in-process and reports the effective values
 * Astronomical will run with.
 */
public enum ValidateConfigCommand {

    /// Runs the validate-config journey and writes the report.
    public static func run(
        validateArguments: ValidateConfigArguments,
        renderedOutput: TextOutputWriter
    ) throws -> Void {
        let instancePaths: AstronomicalInstancePaths;
        do {
            instancePaths = try AstronomicalInstancePaths.forCurrentUser(
                runtimeInstance: validateArguments.runtimeInstance
            );
        } catch let configError {
            throw ValidateConfigError.inconsistentConfiguration(
                cause: String(describing: configError));
        }
        return try ValidateConfigCommand.run(
            validateArguments: validateArguments,
            instancePaths: instancePaths,
            renderedOutput: renderedOutput
        );
    }

    /// Runs the validate-config journey against one instance's state
    /// directory, the seam hermetic journeys use to redirect the file reads.
    public static func run(
        validateArguments: ValidateConfigArguments,
        instancePaths: AstronomicalInstancePaths,
        renderedOutput: TextOutputWriter
    ) throws -> Void {
        let configFilePath: String = instancePaths.configFilePath.string;
        let configDocumentText: String = try ValidateConfigCommand.readConfigDocumentText(
            configFilePath: configFilePath
        );
        let loadedConfig: AstronomicalConfig;
        do {
            loadedConfig = try AstronomicalConfig.loadFromInstancePaths(instancePaths);
        } catch let configError {
            throw ValidateConfigError.invalidConfiguration(
                configFilePath: configFilePath,
                cause: String(describing: configError)
            );
        }
        _ = configDocumentText;
        let effectiveValues: EffectiveConfigurationValues = try ValidateConfigCommand.effectiveValues(
            loadedConfig: loadedConfig,
            configFilePath: configFilePath
        );
        let reportText: String = validateArguments.renderJson
            ? ValidateConfigCommand.renderJsonReport(effectiveValues)
            : ValidateConfigCommand.renderTextReport(effectiveValues);
        if !renderedOutput.write(reportText) {
            throw ValidateConfigError.reportWriteFailed;
        }
    }

    private static func readConfigDocumentText(configFilePath: String) throws -> String {
        let configDocumentData: Data;
        do {
            configDocumentData = try Data(contentsOf: URL(fileURLWithPath: configFilePath));
        } catch let readError where (readError as NSError).code == NSFileReadNoSuchFileError {
            throw ValidateConfigError.configFileMissing(configFilePath: configFilePath);
        } catch let readError {
            throw ValidateConfigError.configFileUnreadable(
                configFilePath: configFilePath,
                cause: readError.localizedDescription
            );
        }
        return String(decoding: configDocumentData, as: UTF8.self);
    }

    private static func effectiveValues(
        loadedConfig: AstronomicalConfig,
        configFilePath: String
    ) throws -> EffectiveConfigurationValues {
        let supervisorBindAddress: SocketEndpoint;
        let maximumMlxMemoryBytes: UInt64?;
        do {
            supervisorBindAddress = try loadedConfig.supervisorBindAddress();
            maximumMlxMemoryBytes = try loadedConfig.maximumMlxMemoryBytes();
        } catch let configError {
            throw ValidateConfigError.inconsistentConfiguration(
                cause: String(describing: configError));
        }
        let loggingValues: LoggingConfig = loadedConfig.logging();
        return EffectiveConfigurationValues(
            configurationFilePath: configFilePath,
            generation: (try? loadedConfig.generation()) ?? "",
            runtimeInstance: loadedConfig.instancePaths.runtimeInstance ?? .development,
            supervisorBindAddress: supervisorBindAddress,
            modelDirectories: loadedConfig.modelDirectories,
            configuredModelIds: loadedConfig.configuredModelIds,
            persistentPromptCacheEnabled: loadedConfig.persistentPromptCacheEnabled(),
            maximumMlxMemoryBytes: maximumMlxMemoryBytes,
            loggingLevel: loggingValues.level.asStr(),
            loggingRetainedFiles: loggingValues.retainedFiles,
            loggingDirectory: loggingValues.directory.string,
            performanceAttributionEnabled: loadedConfig.performanceAttributionEnabled()
        );
    }

    private static func renderTextReport(
        _ effectiveValues: EffectiveConfigurationValues
    ) -> String {
        var reportLines: Array<String> = [];
        reportLines.append("Configuration file: \(effectiveValues.configurationFilePath)");
        reportLines.append("Generation: \(effectiveValues.generation)");
        reportLines.append("Runtime instance: \(ValidateConfigCommand.runtimeInstanceSlug(effectiveValues.runtimeInstance))");
        reportLines.append("Supervisor bind address: \(effectiveValues.supervisorBindAddress)");
        reportLines.append("Model directories: \(effectiveValues.modelDirectories.count)");
        for modelDirectory: FilePath in effectiveValues.modelDirectories {
            reportLines.append("  \(modelDirectory.string)");
        }
        reportLines.append("Configured model ids: \(effectiveValues.configuredModelIds.isEmpty ? "none" : effectiveValues.configuredModelIds.joined(separator: ", "))");
        reportLines.append("Persistent prompt cache: \(ValidateConfigCommand.enabledWord(effectiveValues.persistentPromptCacheEnabled))");
        reportLines.append("Maximum MLX memory: \(ValidateConfigCommand.renderedMaximumMlxMemory(effectiveValues.maximumMlxMemoryBytes))");
        reportLines.append("Logging level: \(effectiveValues.loggingLevel)");
        reportLines.append("Logging retained files: \(effectiveValues.loggingRetainedFiles)");
        reportLines.append("Logging directory: \(effectiveValues.loggingDirectory)");
        reportLines.append("Performance attribution: \(ValidateConfigCommand.enabledWord(effectiveValues.performanceAttributionEnabled))");
        return reportLines.joined(separator: "\n") + "\n";
    }

    private static func renderJsonReport(
        _ effectiveValues: EffectiveConfigurationValues
    ) -> String {
        var loggingDocument: Dictionary<String, Any> = [:];
        loggingDocument["level"] = effectiveValues.loggingLevel;
        loggingDocument["retained_files"] = effectiveValues.loggingRetainedFiles;
        loggingDocument["directory"] = effectiveValues.loggingDirectory;
        var reportDocument: Dictionary<String, Any> = [:];
        reportDocument["configuration_file_path"] = effectiveValues.configurationFilePath;
        reportDocument["generation"] = effectiveValues.generation;
        reportDocument["runtime_instance"] = ValidateConfigCommand.runtimeInstanceSlug(effectiveValues.runtimeInstance);
        reportDocument["supervisor_bind_address"] = effectiveValues.supervisorBindAddress.description;
        reportDocument["model_directories"] = effectiveValues.modelDirectories.map { (modelDirectory: FilePath) -> String in
            return modelDirectory.string;
        };
        reportDocument["configured_model_ids"] = effectiveValues.configuredModelIds;
        reportDocument["persistent_prompt_cache_enabled"] = effectiveValues.persistentPromptCacheEnabled;
        reportDocument["maximum_mlx_memory_bytes"] = effectiveValues.maximumMlxMemoryBytes.map { (maximumBytes: UInt64) -> Any in
            return Int(maximumBytes);
        } ?? NSNull();
        reportDocument["logging"] = loggingDocument;
        reportDocument["performance_attribution_enabled"] = effectiveValues.performanceAttributionEnabled;
        guard let renderedData: Data = try? JSONSerialization.data(
            withJSONObject: reportDocument,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        ) else {
            return "{}\n";
        }
        return String(decoding: renderedData, as: UTF8.self) + "\n";
    }

    private static func runtimeInstanceSlug(
        _ runtimeInstance: AstronomicalRuntimeInstance
    ) -> String {
        switch (runtimeInstance) {
        case .stable:
            return "stable";
        case .development:
            return "development";
        }
    }

    private static func enabledWord(_ isEnabled: Bool) -> String {
        return isEnabled ? "enabled" : "disabled";
    }

    /// Renders a byte count as decimal SI gigabytes, truncated to one tenth
    /// so the value never overstates available memory.
    private static func renderedMaximumMlxMemory(_ maximumMlxMemoryBytes: UInt64?) -> String {
        let bytesPerGigabyte: UInt64 = 1_000_000_000;
        let tenthsPerGigabyte: UInt64 = 100_000_000;
        guard let totalBytes: UInt64 = maximumMlxMemoryBytes else {
            return "not set";
        }
        let wholeGigabytes: UInt64 = totalBytes / bytesPerGigabyte;
        let gigabyteTenths: UInt64 = (totalBytes % bytesPerGigabyte) / tenthsPerGigabyte;
        if (gigabyteTenths == 0) {
            return "\(wholeGigabytes) GB";
        }
        return "\(wholeGigabytes).\(gigabyteTenths) GB";
    }
}

/// The effective values one validate report carries.
struct EffectiveConfigurationValues {

    let configurationFilePath: String;
    let generation: String;
    let runtimeInstance: AstronomicalRuntimeInstance;
    let supervisorBindAddress: SocketEndpoint;
    let modelDirectories: Array<FilePath>;
    let configuredModelIds: Array<String>;
    let persistentPromptCacheEnabled: Bool;
    let maximumMlxMemoryBytes: UInt64?;
    let loggingLevel: String;
    let loggingRetainedFiles: Int;
    let loggingDirectory: String;
    let performanceAttributionEnabled: Bool;
}
