import Foundation;


/// The worker-spawn prompt-cache policy derived from the startup
/// configuration, port of the Rust worker startup's prompt-cache intake:
/// the daemon sends the global cache root, the quota, and the enable flag;
/// the worker resolves the effective memory ceiling and derives every
/// attachment from this policy when a MoE runtime is built.
public struct Qwen35MoePromptCacheSpawnPolicy: Sendable {

    /// The shared global prompt-cache root whose quota spans every model.
    public let globalPromptCacheRootDirectory: URL;

    /// The global SSD quota in bytes shared by every model's cache.
    public let globalPromptCacheMaximumSizeBytes: UInt64;

    /// The resolved live MLX active-memory ceiling for this worker.
    public let effectiveMlxMemoryCeilingBytes: UInt64;

    public init(
        globalPromptCacheRootDirectory: URL,
        globalPromptCacheMaximumSizeBytes: UInt64,
        effectiveMlxMemoryCeilingBytes: UInt64
    ) {
        self.globalPromptCacheRootDirectory = globalPromptCacheRootDirectory;
        self.globalPromptCacheMaximumSizeBytes = globalPromptCacheMaximumSizeBytes;
        self.effectiveMlxMemoryCeilingBytes = effectiveMlxMemoryCeilingBytes;
    }

    /// Derives one engine attachment from this policy: the model namespace
    /// hangs off the global root, and the caller supplies the model
    /// identity plus the chunking configuration's block length.
    public func makeAttachment(
        modelId: String,
        modelRevision: String,
        configuredBlockTokenCount: Int?
    ) -> Qwen35MoePromptCacheAttachment {
        return Qwen35MoePromptCacheAttachment(
            activeModelPromptCacheDirectory: self.globalPromptCacheRootDirectory
                .appendingPathComponent(modelId, isDirectory: true)
                .appendingPathComponent(modelRevision, isDirectory: true),
            globalPromptCacheRootDirectory: self.globalPromptCacheRootDirectory,
            globalPromptCacheMaximumSizeBytes: self.globalPromptCacheMaximumSizeBytes,
            effectiveMlxMemoryCeilingBytes: self.effectiveMlxMemoryCeilingBytes,
            modelId: modelId,
            modelRevision: modelRevision,
            configuredBlockTokenCount: configuredBlockTokenCount);
    }
}
