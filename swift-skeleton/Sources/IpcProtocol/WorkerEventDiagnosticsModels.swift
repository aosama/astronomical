import Foundation;

extension WorkerEvent {
    /// Returns a bounded diagnostic summary without exposing model-generated
    /// payloads, mirroring the Rust `WorkerEvent::diagnostic_summary`.
    public func diagnosticSummary() -> String {
        switch (self) {
        case .runtimeFeatureConfigurationApplied: return "runtime_feature_configuration_applied";
        case .idle: return "idle";
        case .mlxMemorySample: return "mlx_memory_sample";
        case .mlxMemoryLimitChanged: return "mlx_memory_limit_changed";
        case .mlxMemoryLimitRejected: return "mlx_memory_limit_rejected";
        case .expertMemoryModeChanged: return "expert_memory_mode_changed";
        case let .generationFinalized(requestId, _, _, _): return WorkerEvent.summarizedVariant("generation_finalized", requestId: requestId);
        case let .imageGenerationProgress(requestId, _, _, _, _, _): return WorkerEvent.summarizedVariant("image_generation_progress", requestId: requestId);
        case let .imageGenerationCompleted(requestId, _, _): return WorkerEvent.summarizedVariant("image_generation_completed", requestId: requestId);
        case let .imageGenerationFailed(requestId, _): return WorkerEvent.summarizedVariant("image_generation_failed", requestId: requestId);
        case let .imageGenerationFinalized(requestId, _, _): return WorkerEvent.summarizedVariant("image_generation_finalized", requestId: requestId);
        case let .embeddingsCompleted(requestId, _, _, _): return WorkerEvent.summarizedVariant("embeddings_completed", requestId: requestId);
        case let .embeddingsFailed(requestId, _): return WorkerEvent.summarizedVariant("embeddings_failed", requestId: requestId);
        case let .embeddingsFinalized(requestId, _, _): return WorkerEvent.summarizedVariant("embeddings_finalized", requestId: requestId);
        case .ready: return "ready";
        case let .output(requestId, _, _, _, _, _): return WorkerEvent.summarizedVariant("output", requestId: requestId);
        case let .prefillProgress(requestId, _, _, _, _, _, _, _, _): return WorkerEvent.summarizedVariant("prefill_progress", requestId: requestId);
        case let .generationPreparationStarted(requestId, _, _, _, _): return WorkerEvent.summarizedVariant("generation_preparation_started", requestId: requestId);
        case let .generationProgress(requestId, _, _, _, _, _): return WorkerEvent.summarizedVariant("generation_progress", requestId: requestId);
        case let .firstDecodeCompleted(requestId, _): return WorkerEvent.summarizedVariant("first_decode_completed", requestId: requestId);
        case let .promptWorkReuse(requestId, _): return WorkerEvent.summarizedVariant("prompt_work_reuse", requestId: requestId);
        case let .completed(requestId, _, _, _, _, _, _): return WorkerEvent.summarizedVariant("completed", requestId: requestId);
        case let .failed(requestId, _): return WorkerEvent.summarizedVariant("failed", requestId: requestId);
        case .modelSwapped: return "model_swapped";
        case .modelSwapFailed: return "model_swap_failed";
        case .persistentPromptCacheStats: return "persistent_prompt_cache_stats";
        case let .promptCacheCleared(modelId, _, _): return "prompt_cache_cleared model_id=\(WorkerEvent.rustDebugDescription(modelId))";
        }
    }

    private static func summarizedVariant(_ variantName: String, requestId: RequestId) -> String {
        return "\(variantName) request_id=\(requestId.value())";
    }

    /// Renders an optional model identifier the way Rust's `{:?}` formatting
    /// prints `Option<String>`, so log lines match the reference daemon.
    private static func rustDebugDescription(_ optionalModelId: String?) -> String {
        guard let unwrappedModelId = optionalModelId else {
            return "None";
        }
        return "Some(\"\(unwrappedModelId)\")";
    }
}
