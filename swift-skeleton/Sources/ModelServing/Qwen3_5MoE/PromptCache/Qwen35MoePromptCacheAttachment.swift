import Foundation;


/// The caller-supplied persistent prompt-cache locations and budgets for one
/// resident MoE engine, port of the store-opening inputs the Rust worker
/// passes to the engine state. The engine resolves the storage contract from
/// its bound execution dtypes and opens the disk store at attach time; the
/// directories and quotas belong to the deployment, never to this file.
public struct Qwen35MoePromptCacheAttachment: Sendable {

    /// The active model's isolated prompt-cache namespace directory.
    public let activeModelPromptCacheDirectory: URL;

    /// The shared global prompt-cache root whose quota spans every model.
    public let globalPromptCacheRootDirectory: URL;

    /// The global SSD quota in bytes shared by every model's cache.
    public let globalPromptCacheMaximumSizeBytes: UInt64;

    /// The live MLX active-memory ceiling the contract's memory projections
    /// assume; mirrors the value recorded through `applyMlxMemoryLimit`.
    public let effectiveMlxMemoryCeilingBytes: UInt64;

    /// The model identity the cache namespace is derived from.
    public let modelId: String;

    /// The model revision the cache namespace is derived from.
    public let modelRevision: String;

    /// The user-configured block length in tokens, or nil for automatic
    /// sizing from the contract's quota-derived policy.
    public let configuredBlockTokenCount: Int?;

    /// The checkpoint stride that retention keeps between reusable
    /// recurrent snapshots.
    public let commonPrefixCheckpointStrideBlocks: UInt32;

    public init(
        activeModelPromptCacheDirectory: URL,
        globalPromptCacheRootDirectory: URL,
        globalPromptCacheMaximumSizeBytes: UInt64,
        effectiveMlxMemoryCeilingBytes: UInt64,
        modelId: String,
        modelRevision: String,
        configuredBlockTokenCount: Int? = nil,
        commonPrefixCheckpointStrideBlocks: UInt32 = 4
    ) {
        self.activeModelPromptCacheDirectory = activeModelPromptCacheDirectory;
        self.globalPromptCacheRootDirectory = globalPromptCacheRootDirectory;
        self.globalPromptCacheMaximumSizeBytes = globalPromptCacheMaximumSizeBytes;
        self.effectiveMlxMemoryCeilingBytes = effectiveMlxMemoryCeilingBytes;
        self.modelId = modelId;
        self.modelRevision = modelRevision;
        self.configuredBlockTokenCount = configuredBlockTokenCount;
        self.commonPrefixCheckpointStrideBlocks = commonPrefixCheckpointStrideBlocks;
    }
}
