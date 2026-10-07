import Foundation;

/**
 * Failure while loading or resolving Astronomical runtime configuration.
 */
public enum AstronomicalConfigError: Error, CustomStringConvertible {
    case homeDirectoryRequired
    case resolveHomeDirectory(homeDirectory: FilePath, underlyingError: any Error)
    case homeDirectoryMustNotBeRoot
    case invalidRuntimeInstance(rawInstance: String)
    case pathMustBeAbsolute(fieldName: String, configuredPath: FilePath)
    case standardInstanceBindAddressMismatch(
        configuredBindAddress: SocketEndpoint,
        expectedBindAddress: SocketEndpoint
    )
    case readConfigFile(configFilePath: FilePath, underlyingError: any Error)
    case parseConfigFile(configFilePath: FilePath, underlyingDescription: String)
    case serializeConfigFile(configFilePath: FilePath)
    case writeConfigFile(configFilePath: FilePath, underlyingError: any Error)
    case configFileTooLarge(configFilePath: FilePath, maximumBytes: Int)
    case unsupportedSchemaVersion(schemaVersion: UInt32)
    case invalidSchemaReference
    case nonLoopbackBindAddress(supervisorBindAddress: SocketEndpoint)
    case invalidMaximumMlxMemoryGb(description: String)
    case invalidPromptCacheMaxSizeGb(description: String)
    case invalidDefaultModel(description: String)
    case configuredContextExceedsArtifact(
        modelId: String,
        configuredMaximumContextTokens: UInt32,
        artifactMaximumContextTokens: UInt32)
    case configuredOutputNotSmallerThanContext(
        modelId: String,
        configuredMaximumOutputTokens: UInt32,
        effectiveMaximumContextTokens: UInt32)
    case configChangedDuringUpdate

    public var description: String {
        switch (self) {
        case .homeDirectoryRequired:
            return "HOME is required to derive the Astronomical instance directory";
        case .resolveHomeDirectory(let homeDirectory, let underlyingError):
            return "failed to resolve HOME at \"\(homeDirectory.string)\" (\(underlyingError.localizedDescription))";
        case .homeDirectoryMustNotBeRoot:
            return "HOME must not resolve to the filesystem root";
        case .invalidRuntimeInstance(let rawInstance):
            return "runtime instance must be 'stable' or 'development', got '\(rawInstance)'";
        case .pathMustBeAbsolute(let fieldName, let configuredPath):
            return "\(fieldName) must be an absolute path, got \"\(configuredPath.string)\"";
        case .standardInstanceBindAddressMismatch(let configuredBindAddress, let expectedBindAddress):
            return "standard runtime instance must bind to \(expectedBindAddress), not \(configuredBindAddress)";
        case .readConfigFile(let configFilePath, let underlyingError):
            return "failed to read config file at \"\(configFilePath.string)\" (\(underlyingError.localizedDescription))";
        case .parseConfigFile(let configFilePath, let underlyingDescription):
            return "failed to parse config file at \"\(configFilePath.string)\" (\(underlyingDescription))";
        case .serializeConfigFile(let configFilePath):
            return "failed to serialize the config file for \"\(configFilePath.string)\"";
        case .writeConfigFile(let configFilePath, let underlyingError):
            return "failed to write config file at \"\(configFilePath.string)\" (\(underlyingError.localizedDescription))";
        case .configFileTooLarge(let configFilePath, let maximumBytes):
            return "config file at \"\(configFilePath.string)\" exceeds the \(maximumBytes) byte cap";
        case .unsupportedSchemaVersion(let schemaVersion):
            return "config schema version \(schemaVersion) is not supported, only version 1 is";
        case .invalidSchemaReference:
            return "config file must reference \"./astronomical-config.schema.json\" through its $schema field";
        case .nonLoopbackBindAddress(let supervisorBindAddress):
            return "supervisor bind address \(supervisorBindAddress) must be a loopback address";
        case .invalidPromptCacheMaxSizeGb(let problemDescription):
            return problemDescription;
        case .invalidDefaultModel(let problemDescription):
            return problemDescription;
        case let .configuredContextExceedsArtifact(modelId, configuredMaximumContextTokens, artifactMaximumContextTokens):
            return "models[\(modelId)].limits.maximum_context_tokens (\(configuredMaximumContextTokens)) exceeds the discovered artifact maximum (\(artifactMaximumContextTokens))";
        case let .configuredOutputNotSmallerThanContext(modelId, configuredMaximumOutputTokens, effectiveMaximumContextTokens):
            return "models[\(modelId)].generation_defaults.maximum_output_tokens (\(configuredMaximumOutputTokens)) must be smaller than the effective context (\(effectiveMaximumContextTokens))";
        case .invalidMaximumMlxMemoryGb(let problemDescription):
            return "invalid maximum_mlx_memory_gb: \(problemDescription)";
        case .configChangedDuringUpdate:
            return "the config file changed while the update was being prepared; retry the update";
        }
    }
}
