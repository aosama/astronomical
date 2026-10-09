import Foundation;


/// Typed failures of the MoE engine's persistent prompt-cache wiring, port
/// of the engine-side error translations in the Rust capture and restore
/// owners. Required persistence failures stop the request; best-effort tail
/// failures are logged and skipped by the caller.
public enum Qwen35MoePromptCacheError: Error, Equatable {

    /// The live full-attention cache is not a plain key/value cache; the
    /// persistent cache supports no other execution representation.
    case unsupportedFullAttentionCacheType(layerIndex: Int);

    /// A live cache tensor that the storage contract requires was absent.
    case missingLiveCacheTensor(layerIndex: Int, tensorRole: String);

    /// The live state dtype cannot be persisted.
    case unsupportedLiveCacheDtype(layerIndex: Int, tensorRole: String, dtypeName: String);

    /// The live cache families disagree with the validated configuration's
    /// per-layer layout.
    case liveCacheFamilyMismatch(layerIndex: Int);

    /// The store was consulted before the engine attached it.
    case promptCacheNotAttached;

    /// Restore loaded a block that lookup had proven present, but the load
    /// found nothing; the store index and the filesystem disagree.
    case restoredBlockVanished(blockIndex: UInt32);

    /// A required capture step failed validation and stopped the request.
    case requiredCaptureFailure(stage: String, description: String);
}
