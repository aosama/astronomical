import Foundation;

import IpcProtocol;

/// The synchronous inference-engine contract every model family plugs in.
///
/// Mirrors the MLX-owner half of crates/model-serving/src/inference_engine/
/// contract.rs (`MlxInferenceExecution`). The async Tokio wrapper of the Rust
/// worker evaporates here: the Swift worker loop is the single MLX owner and
/// advances one engine step per loop turn, interleaving supervisor commands
/// between steps through `ProtocolReader.pollNextCommand`.
///
/// Perf-attribution contract: implementations wrap load, each prefill chunk,
/// and each decode step in the switchable performance attribution spans so
/// prompt processing and token generation stay attributable end to end.
public protocol InferenceEngine: AnyObject {

    /// Loads engine resources before the worker reports readiness.
    func load() throws -> EngineLoadResult;

    /// Creates one engine-side request after prompt preparation succeeds.
    func startGeneration(
        _ inferenceRequest: any PreparedInferenceRequest
    ) throws -> EngineGenerationStart;

    /// Advances one bounded prefill or generated-token boundary.
    func decodeNextToken(requestId: RequestId) throws -> GeneratedToken;

    /// Adds tokenized model-visible feedback before decoding continues.
    func injectInputTokens(requestId: RequestId, inputTokenIds: Array<UInt32>) throws;

    /// Cancels and releases engine-side state for one active request.
    func cancelGeneration(requestId: RequestId) throws -> GenerationFinalization;

    /// Collects the current idle MLX memory observation when supported;
    /// `nil` means the engine has no observation for this poll.
    func collectMlxMemorySnapshot() -> WorkerMlxMemorySnapshot?;

    /// Records a validated live MLX memory ceiling on the engine's context.
    func applyMlxMemoryLimit(_ requestedMlxMemoryCeilingBytes: UInt64) throws;

    /// Collects the persistent prompt-cache stats snapshot when the engine
    /// has a cache attached; `nil` by default. Mirrors the Rust engine's
    /// `collect_persistent_prompt_cache_stats`.
    func collectPersistentPromptCacheStats() -> WorkerPersistentPromptCacheStats?;

    /// Clears the persisted prompt-cache state for one model identifier, or
    /// for the active model when the identifier is `nil`. The default
    /// engine carries no persistent cache and reports the empty deletion.
    func clearPersistentPromptCache(modelId: String?) throws
        -> PersistentPromptCacheClearOutcome;

    /// The persisted sequence-state bytes one context token occupies for
    /// this engine, or `nil` when the engine has no context-workspace
    /// charge. The memory governor multiplies this by the request's
    /// context token need for its admission verdict.
    func contextWorkspaceBytesPerToken() -> Int?;
}

/// Default persistent prompt-cache surface for engines without a cache.
extension InferenceEngine {

    public func collectPersistentPromptCacheStats() -> WorkerPersistentPromptCacheStats? {
        return nil;
    }

    public func clearPersistentPromptCache(
        modelId: String?
    ) throws -> PersistentPromptCacheClearOutcome {
        return PersistentPromptCacheClearOutcome(
            modelId: modelId, blocksRemoved: 0, bytesFreed: 0);
    }

    public func contextWorkspaceBytesPerToken() -> Int? {
        return nil;
    }
}
