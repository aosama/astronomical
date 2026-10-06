import Foundation;

extension WorkerEvent {
    internal func wireValue() -> JsonWireValue {
        switch (self) {
        case let .runtimeFeatureConfigurationApplied(workerRuntimeFeatureConfiguration):
            return WorkerEvent.taggedWireObject(variantName: "runtime_feature_configuration_applied", fieldAppender: { (wireObject: inout JsonWireObject) in
                wireObject.appendEntry(key: "worker_runtime_feature_configuration", value: workerRuntimeFeatureConfiguration.wireValue());
            });
        case let .idle(machineMlxMemoryCeilingBytes, effectiveMlxMemoryCeilingBytes, minimumMlxMemoryCeilingBytes):
            return WorkerEvent.taggedWireObject(variantName: "idle", fieldAppender: { (wireObject: inout JsonWireObject) in
                wireObject.appendEntry(key: "machine_mlx_memory_ceiling_bytes", value: .unsignedInteger(machineMlxMemoryCeilingBytes));
                wireObject.appendEntry(key: "effective_mlx_memory_ceiling_bytes", value: .unsignedInteger(effectiveMlxMemoryCeilingBytes));
                wireObject.appendEntry(key: "minimum_mlx_memory_ceiling_bytes", value: .unsignedInteger(minimumMlxMemoryCeilingBytes));
            });
        case let .mlxMemorySample(mlxMemorySnapshot, expertResidency):
            return WorkerEvent.taggedWireObject(variantName: "mlx_memory_sample", fieldAppender: { (wireObject: inout JsonWireObject) in
                wireObject.appendEntry(key: "mlx_memory_snapshot", value: WorkerEventWireValues.optionalMlxMemorySnapshotWireValue(mlxMemorySnapshot));
                wireObject.appendEntry(key: "expert_residency", value: WorkerEventWireValues.optionalExpertResidencyWireValue(expertResidency));
            });
        case let .mlxMemoryLimitChanged(effectiveMlxMemoryCeilingBytes, minimumMlxMemoryCeilingBytes, expertMemoryMode, mlxMemorySnapshot, expertResidency):
            return WorkerEvent.taggedWireObject(variantName: "mlx_memory_limit_changed", fieldAppender: { (wireObject: inout JsonWireObject) in
                wireObject.appendEntry(key: "effective_mlx_memory_ceiling_bytes", value: .unsignedInteger(effectiveMlxMemoryCeilingBytes));
                wireObject.appendEntry(key: "minimum_mlx_memory_ceiling_bytes", value: .unsignedInteger(minimumMlxMemoryCeilingBytes));
                wireObject.appendEntry(key: "expert_memory_mode", value: expertMemoryMode.wireValue());
                wireObject.appendEntry(key: "mlx_memory_snapshot", value: WorkerEventWireValues.optionalMlxMemorySnapshotWireValue(mlxMemorySnapshot));
                wireObject.appendEntry(key: "expert_residency", value: WorkerEventWireValues.optionalExpertResidencyWireValue(expertResidency));
            });
        case let .mlxMemoryLimitRejected(requestedMlxMemoryCeilingBytes, minimumMlxMemoryCeilingBytes, machineMlxMemoryCeilingBytes, reason):
            return WorkerEvent.taggedWireObject(variantName: "mlx_memory_limit_rejected", fieldAppender: { (wireObject: inout JsonWireObject) in
                wireObject.appendEntry(key: "requested_mlx_memory_ceiling_bytes", value: .unsignedInteger(requestedMlxMemoryCeilingBytes));
                wireObject.appendEntry(key: "minimum_mlx_memory_ceiling_bytes", value: .unsignedInteger(minimumMlxMemoryCeilingBytes));
                wireObject.appendEntry(key: "machine_mlx_memory_ceiling_bytes", value: .unsignedInteger(machineMlxMemoryCeilingBytes));
                wireObject.appendEntry(key: "reason", value: .string(reason));
            });
        case let .expertMemoryModeChanged(expertMemoryMode):
            return WorkerEvent.taggedWireObject(variantName: "expert_memory_mode_changed", fieldAppender: { (wireObject: inout JsonWireObject) in
                wireObject.appendEntry(key: "expert_memory_mode", value: expertMemoryMode.wireValue());
            });
        case let .generationFinalized(requestId, expertMemoryMode, mlxMemorySnapshot, expertResidency):
            return WorkerEvent.taggedWireObject(variantName: "generation_finalized", fieldAppender: { (wireObject: inout JsonWireObject) in
                wireObject.appendEntry(key: "request_id", value: requestId.wireValue());
                wireObject.appendEntry(key: "expert_memory_mode", value: WorkerEventWireValues.optionalExpertMemoryModeWireValue(expertMemoryMode));
                wireObject.appendEntry(key: "mlx_memory_snapshot", value: WorkerEventWireValues.optionalMlxMemorySnapshotWireValue(mlxMemorySnapshot));
                wireObject.appendEntry(key: "expert_residency", value: WorkerEventWireValues.optionalExpertResidencyWireValue(expertResidency));
            });
        case let .imageGenerationProgress(requestId, phase, completedSteps, totalSteps, elapsedMillis, mlxMemorySnapshot):
            return WorkerEvent.taggedWireObject(variantName: "image_generation_progress", fieldAppender: { (wireObject: inout JsonWireObject) in
                wireObject.appendEntry(key: "request_id", value: requestId.wireValue());
                wireObject.appendEntry(key: "phase", value: phase.wireValue());
                wireObject.appendEntry(key: "completed_steps", value: .unsignedInteger(UInt64(completedSteps)));
                wireObject.appendEntry(key: "total_steps", value: .unsignedInteger(UInt64(totalSteps)));
                wireObject.appendEntry(key: "elapsed_millis", value: .unsignedInteger(elapsedMillis));
                wireObject.appendEntry(key: "mlx_memory_snapshot", value: WorkerEventWireValues.optionalMlxMemorySnapshotWireValue(mlxMemorySnapshot));
            });
        case let .imageGenerationCompleted(requestId, generatedImage, resultMetadata):
            return WorkerEvent.taggedWireObject(variantName: "image_generation_completed", fieldAppender: { (wireObject: inout JsonWireObject) in
                wireObject.appendEntry(key: "request_id", value: requestId.wireValue());
                wireObject.appendEntry(key: "generated_image", value: generatedImage.wireValue());
                wireObject.appendEntry(key: "result_metadata", value: resultMetadata.wireValue());
            });
        case let .imageGenerationFailed(requestId, reason):
            return WorkerEvent.taggedWireObject(variantName: "image_generation_failed", fieldAppender: { (wireObject: inout JsonWireObject) in
                wireObject.appendEntry(key: "request_id", value: requestId.wireValue());
                wireObject.appendEntry(key: "reason", value: reason.wireValue());
            });
        case let .imageGenerationFinalized(requestId, elapsedMillis, mlxMemorySnapshot):
            return WorkerEvent.taggedWireObject(variantName: "image_generation_finalized", fieldAppender: { (wireObject: inout JsonWireObject) in
                wireObject.appendEntry(key: "request_id", value: requestId.wireValue());
                wireObject.appendEntry(key: "elapsed_millis", value: .unsignedInteger(elapsedMillis));
                wireObject.appendEntry(key: "mlx_memory_snapshot", value: WorkerEventWireValues.optionalMlxMemorySnapshotWireValue(mlxMemorySnapshot));
            });
        case let .embeddingsCompleted(requestId, embeddings, inputTokenCounts, elapsedMillis):
            return WorkerEvent.taggedWireObject(variantName: "embeddings_completed", fieldAppender: { (wireObject: inout JsonWireObject) in
                wireObject.appendEntry(key: "request_id", value: requestId.wireValue());
                wireObject.appendEntry(key: "embeddings", value: WorkerEventWireValues.nestedFloat32ArrayWireValue(embeddings));
                wireObject.appendEntry(key: "input_token_counts", value: WorkerEventWireValues.uint32ArrayWireValue(inputTokenCounts));
                wireObject.appendEntry(key: "elapsed_millis", value: .unsignedInteger(elapsedMillis));
            });
        case let .embeddingsFailed(requestId, reason):
            return WorkerEvent.taggedWireObject(variantName: "embeddings_failed", fieldAppender: { (wireObject: inout JsonWireObject) in
                wireObject.appendEntry(key: "request_id", value: requestId.wireValue());
                wireObject.appendEntry(key: "reason", value: reason.wireValue());
            });
        case let .embeddingsFinalized(requestId, elapsedMillis, mlxMemorySnapshot):
            return WorkerEvent.taggedWireObject(variantName: "embeddings_finalized", fieldAppender: { (wireObject: inout JsonWireObject) in
                wireObject.appendEntry(key: "request_id", value: requestId.wireValue());
                wireObject.appendEntry(key: "elapsed_millis", value: .unsignedInteger(elapsedMillis));
                wireObject.appendEntry(key: "mlx_memory_snapshot", value: WorkerEventWireValues.optionalMlxMemorySnapshotWireValue(mlxMemorySnapshot));
            });
        case let .ready(modelId, capabilities):
            return WorkerEvent.taggedWireObject(variantName: "ready", fieldAppender: { (wireObject: inout JsonWireObject) in
                wireObject.appendEntry(key: "model_id", value: .string(modelId));
                wireObject.appendEntry(key: "capabilities", value: capabilities.wireValue());
            });
        case let .output(requestId, sequenceNumber, generatedTokenCount, outputs, mlxMemorySnapshot, expertResidency):
            return WorkerEvent.taggedWireObject(variantName: "output", fieldAppender: { (wireObject: inout JsonWireObject) in
                wireObject.appendEntry(key: "request_id", value: requestId.wireValue());
                wireObject.appendEntry(key: "sequence_number", value: .unsignedInteger(UInt64(sequenceNumber)));
                wireObject.appendEntry(key: "generated_token_count", value: .unsignedInteger(UInt64(generatedTokenCount)));
                wireObject.appendEntry(key: "outputs", value: WorkerEventWireValues.chatGenerationOutputArrayWireValue(outputs));
                wireObject.appendEntry(key: "mlx_memory_snapshot", value: WorkerEventWireValues.optionalMlxMemorySnapshotWireValue(mlxMemorySnapshot));
                wireObject.appendEntry(key: "expert_residency", value: WorkerEventWireValues.optionalExpertResidencyWireValue(expertResidency));
            });
        case let .prefillProgress(requestId, promptProcessingPhase, processedTokens, totalTokens, elapsedMillis, forwardPrefillChunkElapsedMillis, completedPrefillChunkTokens, mlxMemorySnapshot, expertResidency):
            return WorkerEvent.taggedWireObject(variantName: "prefill_progress", fieldAppender: { (wireObject: inout JsonWireObject) in
                wireObject.appendEntry(key: "request_id", value: requestId.wireValue());
                wireObject.appendEntry(key: "prompt_processing_phase", value: promptProcessingPhase.wireValue());
                wireObject.appendEntry(key: "processed_tokens", value: .unsignedInteger(UInt64(processedTokens)));
                wireObject.appendEntry(key: "total_tokens", value: .unsignedInteger(UInt64(totalTokens)));
                wireObject.appendEntry(key: "elapsed_millis", value: .unsignedInteger(elapsedMillis));
                wireObject.appendEntry(key: "forward_prefill_chunk_elapsed_millis", value: WorkerEventWireValues.optionalUInt64WireValue(forwardPrefillChunkElapsedMillis));
                wireObject.appendEntry(key: "completed_prefill_chunk_tokens", value: WorkerEventWireValues.optionalUInt32WireValue(completedPrefillChunkTokens));
                wireObject.appendEntry(key: "mlx_memory_snapshot", value: WorkerEventWireValues.optionalMlxMemorySnapshotWireValue(mlxMemorySnapshot));
                wireObject.appendEntry(key: "expert_residency", value: WorkerEventWireValues.optionalExpertResidencyWireValue(expertResidency));
            });
        case let .generationPreparationStarted(requestId, totalLayerCount, residentExpertCount, residentExpertPayloadBytes, mlxMemorySnapshot):
            return WorkerEvent.taggedWireObject(variantName: "generation_preparation_started", fieldAppender: { (wireObject: inout JsonWireObject) in
                wireObject.appendEntry(key: "request_id", value: requestId.wireValue());
                wireObject.appendEntry(key: "total_layer_count", value: .unsignedInteger(UInt64(totalLayerCount)));
                wireObject.appendEntry(key: "resident_expert_count", value: .unsignedInteger(UInt64(residentExpertCount)));
                wireObject.appendEntry(key: "resident_expert_payload_bytes", value: .unsignedInteger(residentExpertPayloadBytes));
                wireObject.appendEntry(key: "mlx_memory_snapshot", value: WorkerEventWireValues.optionalMlxMemorySnapshotWireValue(mlxMemorySnapshot));
            });
        case let .generationProgress(requestId, generatedTokenCount, maximumOutputTokens, elapsedMillis, mlxMemorySnapshot, expertResidency):
            return WorkerEvent.taggedWireObject(variantName: "generation_progress", fieldAppender: { (wireObject: inout JsonWireObject) in
                wireObject.appendEntry(key: "request_id", value: requestId.wireValue());
                wireObject.appendEntry(key: "generated_token_count", value: .unsignedInteger(UInt64(generatedTokenCount)));
                wireObject.appendEntry(key: "maximum_output_tokens", value: .unsignedInteger(UInt64(maximumOutputTokens)));
                wireObject.appendEntry(key: "elapsed_millis", value: .unsignedInteger(elapsedMillis));
                wireObject.appendEntry(key: "mlx_memory_snapshot", value: WorkerEventWireValues.optionalMlxMemorySnapshotWireValue(mlxMemorySnapshot));
                wireObject.appendEntry(key: "expert_residency", value: WorkerEventWireValues.optionalExpertResidencyWireValue(expertResidency));
            });
        case let .firstDecodeCompleted(requestId, elapsedMillis):
            return WorkerEvent.taggedWireObject(variantName: "first_decode_completed", fieldAppender: { (wireObject: inout JsonWireObject) in
                wireObject.appendEntry(key: "request_id", value: requestId.wireValue());
                wireObject.appendEntry(key: "elapsed_millis", value: .unsignedInteger(elapsedMillis));
            });
        case let .promptWorkReuse(requestId, promptWorkReuse):
            return WorkerEvent.taggedWireObject(variantName: "prompt_work_reuse", fieldAppender: { (wireObject: inout JsonWireObject) in
                wireObject.appendEntry(key: "request_id", value: requestId.wireValue());
                wireObject.appendEntry(key: "prompt_work_reuse", value: promptWorkReuse.wireValue());
            });
        case let .completed(requestId, promptTokenCount, generatedTokenCount, reasoningTokenCount, cachedTokenCount, persistentPromptCacheDiagnostics, reason):
            return WorkerEvent.taggedWireObject(variantName: "completed", fieldAppender: { (wireObject: inout JsonWireObject) in
                wireObject.appendEntry(key: "request_id", value: requestId.wireValue());
                wireObject.appendEntry(key: "prompt_token_count", value: .unsignedInteger(UInt64(promptTokenCount)));
                wireObject.appendEntry(key: "generated_token_count", value: .unsignedInteger(UInt64(generatedTokenCount)));
                wireObject.appendEntry(key: "reasoning_token_count", value: .unsignedInteger(UInt64(reasoningTokenCount)));
                wireObject.appendEntry(key: "cached_token_count", value: .unsignedInteger(UInt64(cachedTokenCount)));
                wireObject.appendEntry(key: "persistent_prompt_cache_diagnostics", value: WorkerEventWireValues.optionalPromptCacheDiagnosticsWireValue(persistentPromptCacheDiagnostics));
                wireObject.appendEntry(key: "reason", value: reason.wireValue());
            });
        case let .failed(requestId, reason):
            return WorkerEvent.taggedWireObject(variantName: "failed", fieldAppender: { (wireObject: inout JsonWireObject) in
                wireObject.appendEntry(key: "request_id", value: requestId.wireValue());
                wireObject.appendEntry(key: "reason", value: reason.wireValue());
            });
        case let .modelSwapped(modelId, capabilities, expertMemoryMode, minimumMlxMemoryCeilingBytes):
            return WorkerEvent.taggedWireObject(variantName: "model_swapped", fieldAppender: { (wireObject: inout JsonWireObject) in
                wireObject.appendEntry(key: "model_id", value: .string(modelId));
                wireObject.appendEntry(key: "capabilities", value: capabilities.wireValue());
                wireObject.appendEntry(key: "expert_memory_mode", value: WorkerEventWireValues.optionalExpertMemoryModeWireValue(expertMemoryMode));
                wireObject.appendEntry(key: "minimum_mlx_memory_ceiling_bytes", value: .unsignedInteger(minimumMlxMemoryCeilingBytes));
            });
        case let .modelSwapFailed(loadedModelRemainsReady, modelLoadFailureReason):
            return WorkerEvent.taggedWireObject(variantName: "model_swap_failed", fieldAppender: { (wireObject: inout JsonWireObject) in
                wireObject.appendEntry(key: "loaded_model_remains_ready", value: .boolean(loadedModelRemainsReady));
                wireObject.appendEntry(key: "model_load_failure_reason", value: .string(modelLoadFailureReason));
            });
        case let .persistentPromptCacheStats(persistentPromptCacheStats):
            return WorkerEvent.taggedWireObject(variantName: "persistent_prompt_cache_stats", fieldAppender: { (wireObject: inout JsonWireObject) in
                guard case let .object(statsObject) = persistentPromptCacheStats.wireValue() else {
                    return;
                }
                for statsEntry in statsObject.entries {
                    wireObject.appendEntry(key: statsEntry.key, value: statsEntry.value);
                }
            });
        case let .promptCacheCleared(modelId, blocksRemoved, bytesFreed):
            return WorkerEvent.taggedWireObject(variantName: "prompt_cache_cleared", fieldAppender: { (wireObject: inout JsonWireObject) in
                wireObject.appendEntry(key: "model_id", value: WorkerEventWireValues.optionalStringWireValue(modelId));
                wireObject.appendEntry(key: "blocks_removed", value: .unsignedInteger(blocksRemoved));
                wireObject.appendEntry(key: "bytes_freed", value: .unsignedInteger(bytesFreed));
            });
        }
    }

    /// Builds the internally tagged wire object: the `kind` tag first, then
    /// each variant field in Rust declaration order, matching serde's
    /// internally-tagged struct-variant serialization.
    private static func taggedWireObject(variantName: String, fieldAppender: (inout JsonWireObject) -> Void) -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "kind", value: .string(variantName));
        fieldAppender(&wireObject);
        return .object(wireObject);
    }
}
