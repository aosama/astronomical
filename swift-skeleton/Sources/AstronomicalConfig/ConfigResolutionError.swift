import Foundation;

/// Error variants that the configuration resolution modules raise but that do
/// not yet exist on the central AstronomicalConfigError enum. The variants
/// mirror the Rust error variants of crates/config one for one; merging them
/// into AstronomicalConfigError is a central integration step.
internal enum ConfigResolutionError: Error, CustomStringConvertible {

    case invalidDefaultModel(description: String);
    case invalidMaximumMlxMemoryGb(description: String);
    case invalidPromptCacheMaxSizeGb(description: String);
    case invalidChunkingValue(fieldName: String, description: String);
    case invalidRetainedLogFileCount;
    case duplicateConfigKey(configFilePath: FilePath, duplicateKey: String);
    case configChangedDuringUpdate;
    case legacyMigration(description: String);

    internal var description: String {
        switch self {
        case let .invalidDefaultModel(descriptionValue): return "invalid default model: \(descriptionValue)";
        case let .invalidMaximumMlxMemoryGb(descriptionValue): return "invalid maximum MLX memory: \(descriptionValue)";
        case let .invalidPromptCacheMaxSizeGb(descriptionValue): return "invalid prompt-cache maximum size: \(descriptionValue)";
        case let .invalidChunkingValue(fieldName, descriptionValue): return "invalid chunking value for \(fieldName): \(descriptionValue)";
        case .invalidRetainedLogFileCount: return "diagnostics.retained_log_files must be a positive count";
        case let .duplicateConfigKey(configFilePath, duplicateKey): return "duplicate config key \(duplicateKey) in \(configFilePath.string)";
        case .configChangedDuringUpdate: return "the config file changed during the update and the update was abandoned";
        case let .legacyMigration(descriptionValue): return "legacy migration failed: \(descriptionValue)";
        }
    }
}
