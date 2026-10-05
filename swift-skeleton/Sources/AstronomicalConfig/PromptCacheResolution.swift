import Foundation;

/// Prompt-cache directory layout and size conversion, porting
/// crates/config/src/prompt_cache_config.rs plus the
/// prompt_cache_size_gb_to_bytes conversion from crates/config/src/lib.rs.
internal struct PromptCacheConfig {

    internal let globalPromptCacheRootDirectory: FilePath;
    internal let activeModelPromptCacheDirectory: FilePath;
    internal let globalPromptCacheMaximumSizeBytes: UInt64;

    internal init(rootDirectory: FilePath, maximumSizeBytes: UInt64) {
        self.globalPromptCacheRootDirectory = rootDirectory;
        self.activeModelPromptCacheDirectory = rootDirectory;
        self.globalPromptCacheMaximumSizeBytes = maximumSizeBytes;
    }

    private init(rootDirectory: FilePath, activeModelDirectory: FilePath, maximumSizeBytes: UInt64) {
        self.globalPromptCacheRootDirectory = rootDirectory;
        self.activeModelPromptCacheDirectory = activeModelDirectory;
        self.globalPromptCacheMaximumSizeBytes = maximumSizeBytes;
    }

    internal func forModel(modelId: String, revision: String) -> PromptCacheConfig {
        return PromptCacheConfig(
            rootDirectory: self.globalPromptCacheRootDirectory,
            activeModelDirectory: self.globalPromptCacheRootDirectory
                .appending(component: modelId)
                .appending(component: revision),
            maximumSizeBytes: self.globalPromptCacheMaximumSizeBytes
        );
    }
}

internal enum PromptCacheResolution {

    internal static let bytesPerConfiguredGigabyte: UInt64 = 1_000_000_000;

    /// The facade default before the user configures a prompt-cache budget.
    internal static let defaultPromptCacheMaximumSizeGb: UInt64 = 50;

    /// Converts the configured gigabyte budget to bytes. The configured unit is
    /// decimal (1 GB = 1,000,000,000 bytes), matching BYTES_PER_CONFIGURED_GIGABYTE.
    internal static func promptCacheMaximumSizeGbToBytes(_ maximumSizeGb: UInt64) throws -> UInt64 {
        if maximumSizeGb == 0 {
            throw ConfigResolutionError.invalidPromptCacheMaxSizeGb(description: "prompt-cache max size must be positive");
        }
        let (convertedBytes, conversionOverflowed) = maximumSizeGb.multipliedReportingOverflow(by: PromptCacheResolution.bytesPerConfiguredGigabyte);
        if conversionOverflowed {
            throw ConfigResolutionError.invalidPromptCacheMaxSizeGb(description: "prompt-cache max size exceeds the byte range");
        }
        return convertedBytes;
    }
}
