import Foundation;

extension WorkerEvent {
    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerEvent {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        switch try wireObject.decodeTaggedVariantName(tagFieldName: "kind", expectedVariantNames: WorkerEvent.expectedVariantNames) {
        case "runtime_feature_configuration_applied":
            let parsedEvent = WorkerEvent.runtimeFeatureConfigurationApplied(
                try WorkerRuntimeFeatureConfiguration.fromWireValue(try wireObject.requireObjectValue(fieldName: "worker_runtime_feature_configuration")));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["worker_runtime_feature_configuration"]);
            return parsedEvent;
        case "idle":
            let parsedEvent = WorkerEvent.idle(
                machineMlxMemoryCeilingBytes: try wireObject.decodeUInt64(fieldName: "machine_mlx_memory_ceiling_bytes"),
                effectiveMlxMemoryCeilingBytes: try wireObject.decodeUInt64(fieldName: "effective_mlx_memory_ceiling_bytes"),
                minimumMlxMemoryCeilingBytes: try wireObject.decodeUInt64(fieldName: "minimum_mlx_memory_ceiling_bytes"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: [
                "machine_mlx_memory_ceiling_bytes", "effective_mlx_memory_ceiling_bytes", "minimum_mlx_memory_ceiling_bytes",
            ]);
            return parsedEvent;
        case "mlx_memory_sample":
            let parsedEvent = WorkerEvent.mlxMemorySample(
                mlxMemorySnapshot: try WorkerEventWireValues.decodeOptionalMlxMemorySnapshot(wireObject: wireObject, fieldName: "mlx_memory_snapshot"),
                expertResidency: try WorkerEventWireValues.decodeOptionalExpertResidency(wireObject: wireObject, fieldName: "expert_residency"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["mlx_memory_snapshot", "expert_residency"]);
            return parsedEvent;
        case "mlx_memory_limit_changed":
            let parsedEvent = WorkerEvent.mlxMemoryLimitChanged(
                effectiveMlxMemoryCeilingBytes: try wireObject.decodeUInt64(fieldName: "effective_mlx_memory_ceiling_bytes"),
                minimumMlxMemoryCeilingBytes: try wireObject.decodeUInt64(fieldName: "minimum_mlx_memory_ceiling_bytes"),
                expertMemoryMode: try ExpertMemoryMode.fromWireValue(try wireObject.requireObjectValue(fieldName: "expert_memory_mode")),
                mlxMemorySnapshot: try WorkerEventWireValues.decodeOptionalMlxMemorySnapshot(wireObject: wireObject, fieldName: "mlx_memory_snapshot"),
                expertResidency: try WorkerEventWireValues.decodeOptionalExpertResidency(wireObject: wireObject, fieldName: "expert_residency"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: [
                "effective_mlx_memory_ceiling_bytes", "minimum_mlx_memory_ceiling_bytes", "expert_memory_mode",
                "mlx_memory_snapshot", "expert_residency",
            ]);
            return parsedEvent;
        case "mlx_memory_limit_rejected":
            let parsedEvent = WorkerEvent.mlxMemoryLimitRejected(
                requestedMlxMemoryCeilingBytes: try wireObject.decodeUInt64(fieldName: "requested_mlx_memory_ceiling_bytes"),
                minimumMlxMemoryCeilingBytes: try wireObject.decodeUInt64(fieldName: "minimum_mlx_memory_ceiling_bytes"),
                machineMlxMemoryCeilingBytes: try wireObject.decodeUInt64(fieldName: "machine_mlx_memory_ceiling_bytes"),
                reason: try wireObject.decodeString(fieldName: "reason"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: [
                "requested_mlx_memory_ceiling_bytes", "minimum_mlx_memory_ceiling_bytes", "machine_mlx_memory_ceiling_bytes", "reason",
            ]);
            return parsedEvent;
        case "expert_memory_mode_changed":
            let parsedEvent = WorkerEvent.expertMemoryModeChanged(
                expertMemoryMode: try ExpertMemoryMode.fromWireValue(try wireObject.requireObjectValue(fieldName: "expert_memory_mode")));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["expert_memory_mode"]);
            return parsedEvent;
        case "generation_finalized":
            let parsedEvent = WorkerEvent.generationFinalized(
                requestId: try RequestId.fromWireValue(try wireObject.requireObjectValue(fieldName: "request_id")),
                expertMemoryMode: try WorkerEventWireValues.decodeOptionalExpertMemoryMode(wireObject: wireObject, fieldName: "expert_memory_mode"),
                mlxMemorySnapshot: try WorkerEventWireValues.decodeOptionalMlxMemorySnapshot(wireObject: wireObject, fieldName: "mlx_memory_snapshot"),
                expertResidency: try WorkerEventWireValues.decodeOptionalExpertResidency(wireObject: wireObject, fieldName: "expert_residency"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: [
                "request_id", "expert_memory_mode", "mlx_memory_snapshot", "expert_residency",
            ]);
            return parsedEvent;
        case "image_generation_progress":
            let parsedEvent = WorkerEvent.imageGenerationProgress(
                requestId: try RequestId.fromWireValue(try wireObject.requireObjectValue(fieldName: "request_id")),
                phase: try ImageGenerationPhase.fromWireValue(try wireObject.requireObjectValue(fieldName: "phase")),
                completedSteps: try wireObject.decodeUInt16(fieldName: "completed_steps"),
                totalSteps: try wireObject.decodeUInt16(fieldName: "total_steps"),
                elapsedMillis: try wireObject.decodeUInt64(fieldName: "elapsed_millis"),
                mlxMemorySnapshot: try WorkerEventWireValues.decodeOptionalMlxMemorySnapshot(wireObject: wireObject, fieldName: "mlx_memory_snapshot"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: [
                "request_id", "phase", "completed_steps", "total_steps", "elapsed_millis", "mlx_memory_snapshot",
            ]);
            return parsedEvent;
        case "image_generation_completed":
            let parsedEvent = WorkerEvent.imageGenerationCompleted(
                requestId: try RequestId.fromWireValue(try wireObject.requireObjectValue(fieldName: "request_id")),
                generatedImage: try GeneratedImage.fromWireValue(try wireObject.requireObjectValue(fieldName: "generated_image")),
                resultMetadata: try ImageGenerationResultMetadata.fromWireValue(try wireObject.requireObjectValue(fieldName: "result_metadata")));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: [
                "request_id", "generated_image", "result_metadata",
            ]);
            return parsedEvent;
        case "image_generation_failed":
            let parsedEvent = WorkerEvent.imageGenerationFailed(
                requestId: try RequestId.fromWireValue(try wireObject.requireObjectValue(fieldName: "request_id")),
                reason: try ImageGenerationFailureReason.fromWireValue(try wireObject.requireObjectValue(fieldName: "reason")));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["request_id", "reason"]);
            return parsedEvent;
        case "image_generation_finalized":
            let parsedEvent = WorkerEvent.imageGenerationFinalized(
                requestId: try RequestId.fromWireValue(try wireObject.requireObjectValue(fieldName: "request_id")),
                elapsedMillis: try wireObject.decodeUInt64(fieldName: "elapsed_millis"),
                mlxMemorySnapshot: try WorkerEventWireValues.decodeOptionalMlxMemorySnapshot(wireObject: wireObject, fieldName: "mlx_memory_snapshot"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: [
                "request_id", "elapsed_millis", "mlx_memory_snapshot",
            ]);
            return parsedEvent;
        case "embeddings_completed":
            let parsedEvent = WorkerEvent.embeddingsCompleted(
                requestId: try RequestId.fromWireValue(try wireObject.requireObjectValue(fieldName: "request_id")),
                embeddings: try WorkerEventWireValues.nestedFloat32ArrayFromWireValue(try wireObject.requireObjectValue(fieldName: "embeddings")),
                inputTokenCounts: try WorkerEventWireValues.uint32ArrayFromWireValue(try wireObject.requireObjectValue(fieldName: "input_token_counts")),
                elapsedMillis: try wireObject.decodeUInt64(fieldName: "elapsed_millis"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: [
                "request_id", "embeddings", "input_token_counts", "elapsed_millis",
            ]);
            return parsedEvent;
        case "embeddings_failed":
            let parsedEvent = WorkerEvent.embeddingsFailed(
                requestId: try RequestId.fromWireValue(try wireObject.requireObjectValue(fieldName: "request_id")),
                reason: try EmbeddingsFailureReason.fromWireValue(try wireObject.requireObjectValue(fieldName: "reason")));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["request_id", "reason"]);
            return parsedEvent;
        case "embeddings_finalized":
            let parsedEvent = WorkerEvent.embeddingsFinalized(
                requestId: try RequestId.fromWireValue(try wireObject.requireObjectValue(fieldName: "request_id")),
                elapsedMillis: try wireObject.decodeUInt64(fieldName: "elapsed_millis"),
                mlxMemorySnapshot: try WorkerEventWireValues.decodeOptionalMlxMemorySnapshot(wireObject: wireObject, fieldName: "mlx_memory_snapshot"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: [
                "request_id", "elapsed_millis", "mlx_memory_snapshot",
            ]);
            return parsedEvent;
        case "ready":
            let parsedEvent = WorkerEvent.ready(
                modelId: try wireObject.decodeString(fieldName: "model_id"),
                capabilities: try WorkerModelCapabilities.fromWireValue(try wireObject.requireObjectValue(fieldName: "capabilities")));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: [
                "model_id", "capabilities",
            ]);
            return parsedEvent;
        case "output":
            let parsedEvent = WorkerEvent.output(
                requestId: try RequestId.fromWireValue(try wireObject.requireObjectValue(fieldName: "request_id")),
                sequenceNumber: try wireObject.decodeUInt16(fieldName: "sequence_number"),
                generatedTokenCount: try wireObject.decodeUInt16(fieldName: "generated_token_count"),
                outputs: try WorkerEventWireValues.chatGenerationOutputArrayFromWireValue(try wireObject.requireObjectValue(fieldName: "outputs")),
                mlxMemorySnapshot: try WorkerEventWireValues.decodeOptionalMlxMemorySnapshot(wireObject: wireObject, fieldName: "mlx_memory_snapshot"),
                expertResidency: try WorkerEventWireValues.decodeOptionalExpertResidency(wireObject: wireObject, fieldName: "expert_residency"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: [
                "request_id", "sequence_number", "generated_token_count", "outputs", "mlx_memory_snapshot", "expert_residency",
            ]);
            return parsedEvent;
        case "prefill_progress":
            let parsedEvent = WorkerEvent.prefillProgress(
                requestId: try RequestId.fromWireValue(try wireObject.requireObjectValue(fieldName: "request_id")),
                promptProcessingPhase: try WorkerPromptProcessingPhase.fromWireValue(try wireObject.requireObjectValue(fieldName: "prompt_processing_phase")),
                processedTokens: try wireObject.decodeUInt32(fieldName: "processed_tokens"),
                totalTokens: try wireObject.decodeUInt32(fieldName: "total_tokens"),
                elapsedMillis: try wireObject.decodeUInt64(fieldName: "elapsed_millis"),
                forwardPrefillChunkElapsedMillis: try wireObject.decodeOptionalUInt64(fieldName: "forward_prefill_chunk_elapsed_millis"),
                completedPrefillChunkTokens: try wireObject.decodeOptionalUInt32(fieldName: "completed_prefill_chunk_tokens"),
                mlxMemorySnapshot: try WorkerEventWireValues.decodeOptionalMlxMemorySnapshot(wireObject: wireObject, fieldName: "mlx_memory_snapshot"),
                expertResidency: try WorkerEventWireValues.decodeOptionalExpertResidency(wireObject: wireObject, fieldName: "expert_residency"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: [
                "request_id", "prompt_processing_phase", "processed_tokens", "total_tokens", "elapsed_millis",
                "forward_prefill_chunk_elapsed_millis", "completed_prefill_chunk_tokens", "mlx_memory_snapshot", "expert_residency",
            ]);
            return parsedEvent;
        case "generation_preparation_started":
            let parsedEvent = WorkerEvent.generationPreparationStarted(
                requestId: try RequestId.fromWireValue(try wireObject.requireObjectValue(fieldName: "request_id")),
                totalLayerCount: try wireObject.decodeUInt32(fieldName: "total_layer_count"),
                residentExpertCount: try wireObject.decodeUInt32(fieldName: "resident_expert_count"),
                residentExpertPayloadBytes: try wireObject.decodeUInt64(fieldName: "resident_expert_payload_bytes"),
                mlxMemorySnapshot: try WorkerEventWireValues.decodeOptionalMlxMemorySnapshot(wireObject: wireObject, fieldName: "mlx_memory_snapshot"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: [
                "request_id", "total_layer_count", "resident_expert_count", "resident_expert_payload_bytes", "mlx_memory_snapshot",
            ]);
            return parsedEvent;
        case "generation_progress":
            let parsedEvent = WorkerEvent.generationProgress(
                requestId: try RequestId.fromWireValue(try wireObject.requireObjectValue(fieldName: "request_id")),
                generatedTokenCount: try wireObject.decodeUInt16(fieldName: "generated_token_count"),
                maximumOutputTokens: try wireObject.decodeUInt16(fieldName: "maximum_output_tokens"),
                elapsedMillis: try wireObject.decodeUInt64(fieldName: "elapsed_millis"),
                mlxMemorySnapshot: try WorkerEventWireValues.decodeOptionalMlxMemorySnapshot(wireObject: wireObject, fieldName: "mlx_memory_snapshot"),
                expertResidency: try WorkerEventWireValues.decodeOptionalExpertResidency(wireObject: wireObject, fieldName: "expert_residency"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: [
                "request_id", "generated_token_count", "maximum_output_tokens", "elapsed_millis", "mlx_memory_snapshot", "expert_residency",
            ]);
            return parsedEvent;
        case "first_decode_completed":
            let parsedEvent = WorkerEvent.firstDecodeCompleted(
                requestId: try RequestId.fromWireValue(try wireObject.requireObjectValue(fieldName: "request_id")),
                elapsedMillis: try wireObject.decodeUInt64(fieldName: "elapsed_millis"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["request_id", "elapsed_millis"]);
            return parsedEvent;
        case "prompt_work_reuse":
            let parsedEvent = WorkerEvent.promptWorkReuse(
                requestId: try RequestId.fromWireValue(try wireObject.requireObjectValue(fieldName: "request_id")),
                promptWorkReuse: try WorkerPromptWorkReuse.fromWireValue(try wireObject.requireObjectValue(fieldName: "prompt_work_reuse")));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["request_id", "prompt_work_reuse"]);
            return parsedEvent;
        case "completed":
            let parsedEvent = WorkerEvent.completed(
                requestId: try RequestId.fromWireValue(try wireObject.requireObjectValue(fieldName: "request_id")),
                promptTokenCount: try wireObject.decodeUInt32(fieldName: "prompt_token_count"),
                generatedTokenCount: try wireObject.decodeUInt16(fieldName: "generated_token_count"),
                reasoningTokenCount: try wireObject.decodeUInt16(fieldName: "reasoning_token_count"),
                cachedTokenCount: try wireObject.decodeUInt32(fieldName: "cached_token_count"),
                persistentPromptCacheDiagnostics: try WorkerEventWireValues.decodeOptionalPromptCacheDiagnostics(wireObject: wireObject, fieldName: "persistent_prompt_cache_diagnostics"),
                reason: try ChatGenerationCompletionReason.fromWireValue(try wireObject.requireObjectValue(fieldName: "reason")));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: [
                "request_id", "prompt_token_count", "generated_token_count", "reasoning_token_count",
                "cached_token_count", "persistent_prompt_cache_diagnostics", "reason",
            ]);
            return parsedEvent;
        case "failed":
            let parsedEvent = WorkerEvent.failed(
                requestId: try RequestId.fromWireValue(try wireObject.requireObjectValue(fieldName: "request_id")),
                reason: try ChatGenerationFailureReason.fromWireValue(try wireObject.requireObjectValue(fieldName: "reason")));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["request_id", "reason"]);
            return parsedEvent;
        case "model_swapped":
            let parsedEvent = WorkerEvent.modelSwapped(
                modelId: try wireObject.decodeString(fieldName: "model_id"),
                capabilities: try WorkerModelCapabilities.fromWireValue(try wireObject.requireObjectValue(fieldName: "capabilities")),
                expertMemoryMode: try WorkerEventWireValues.decodeOptionalExpertMemoryMode(wireObject: wireObject, fieldName: "expert_memory_mode"),
                minimumMlxMemoryCeilingBytes: try wireObject.decodeUInt64(fieldName: "minimum_mlx_memory_ceiling_bytes"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: [
                "model_id", "capabilities", "expert_memory_mode", "minimum_mlx_memory_ceiling_bytes",
            ]);
            return parsedEvent;
        case "model_swap_failed":
            let parsedEvent = WorkerEvent.modelSwapFailed(
                loadedModelRemainsReady: try wireObject.decodeBool(fieldName: "loaded_model_remains_ready"),
                modelLoadFailureReason: try wireObject.decodeString(fieldName: "model_load_failure_reason"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: [
                "loaded_model_remains_ready", "model_load_failure_reason",
            ]);
            return parsedEvent;
        case "persistent_prompt_cache_stats":
            // The Rust wire form flattens the stats fields into the tagged
            // envelope; the decoder strips the tag and reuses the object
            // decoder so field validation stays in one place.
            var statsWireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
            for wireEntry in wireObject.entries {
                if wireEntry.key != "kind" {
                    statsWireObject.appendEntry(key: wireEntry.key, value: wireEntry.value);
                }
            }
            let parsedEvent = WorkerEvent.persistentPromptCacheStats(
                try WorkerPersistentPromptCacheStats.fromWireValue(.object(statsWireObject)));
            try wireObject.rejectUnknownFieldsBesidesTag(
                tagFieldName: "kind",
                allowedFieldNames: WorkerPersistentPromptCacheStats.wireFieldNames);
            return parsedEvent;
        default:
            let parsedEvent = WorkerEvent.promptCacheCleared(
                modelId: try wireObject.decodeOptionalString(fieldName: "model_id"),
                blocksRemoved: try wireObject.decodeUInt64(fieldName: "blocks_removed"),
                bytesFreed: try wireObject.decodeUInt64(fieldName: "bytes_freed"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: [
                "model_id", "blocks_removed", "bytes_freed",
            ]);
            return parsedEvent;
        }
    }
}
