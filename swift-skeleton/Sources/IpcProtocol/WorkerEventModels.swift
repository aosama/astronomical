import Foundation;

/// One cumulative persistent prompt-cache observability snapshot reported by
/// the worker.
///
/// Mirrors the inline `PersistentPromptCacheStats` struct variant of the Rust
/// `WorkerEvent` enum; all counters are cumulative since worker startup.
public struct WorkerPersistentPromptCacheStats: Equatable {
    public let persistentPromptCacheHits: UInt64;
    public let persistentPromptCacheMisses: UInt64;
    public let persistentPromptCacheTokensSaved: UInt64;
    public let persistentPromptCachePartialTailHits: UInt64;
    public let persistentPromptCacheBlockTokenCount: UInt64;
    public let persistentPromptCacheSequenceStateBlockCount: UInt64;
    public let persistentPromptCacheBoundaryStateSnapshotCount: UInt64;
    public let persistentPromptCacheVisualEmbeddingCount: UInt64;
    public let persistentPromptCacheTotalSizeBytes: UInt64;
    public let persistentPromptCacheVisualEmbeddingTotalSizeBytes: UInt64;
    public let persistentPromptCacheMaximumSizeBytes: UInt64;
    public let persistentPromptCacheVisualEmbeddingHits: UInt64;
    public let persistentPromptCacheVisualEmbeddingMisses: UInt64;
    public let persistentPromptCacheVisualEmbeddingRowsLoaded: UInt64;

    public init(
        persistentPromptCacheHits: UInt64,
        persistentPromptCacheMisses: UInt64,
        persistentPromptCacheTokensSaved: UInt64,
        persistentPromptCachePartialTailHits: UInt64,
        persistentPromptCacheBlockTokenCount: UInt64,
        persistentPromptCacheSequenceStateBlockCount: UInt64,
        persistentPromptCacheBoundaryStateSnapshotCount: UInt64,
        persistentPromptCacheVisualEmbeddingCount: UInt64,
        persistentPromptCacheTotalSizeBytes: UInt64,
        persistentPromptCacheVisualEmbeddingTotalSizeBytes: UInt64,
        persistentPromptCacheMaximumSizeBytes: UInt64,
        persistentPromptCacheVisualEmbeddingHits: UInt64,
        persistentPromptCacheVisualEmbeddingMisses: UInt64,
        persistentPromptCacheVisualEmbeddingRowsLoaded: UInt64
    ) {
        self.persistentPromptCacheHits = persistentPromptCacheHits;
        self.persistentPromptCacheMisses = persistentPromptCacheMisses;
        self.persistentPromptCacheTokensSaved = persistentPromptCacheTokensSaved;
        self.persistentPromptCachePartialTailHits = persistentPromptCachePartialTailHits;
        self.persistentPromptCacheBlockTokenCount = persistentPromptCacheBlockTokenCount;
        self.persistentPromptCacheSequenceStateBlockCount = persistentPromptCacheSequenceStateBlockCount;
        self.persistentPromptCacheBoundaryStateSnapshotCount = persistentPromptCacheBoundaryStateSnapshotCount;
        self.persistentPromptCacheVisualEmbeddingCount = persistentPromptCacheVisualEmbeddingCount;
        self.persistentPromptCacheTotalSizeBytes = persistentPromptCacheTotalSizeBytes;
        self.persistentPromptCacheVisualEmbeddingTotalSizeBytes = persistentPromptCacheVisualEmbeddingTotalSizeBytes;
        self.persistentPromptCacheMaximumSizeBytes = persistentPromptCacheMaximumSizeBytes;
        self.persistentPromptCacheVisualEmbeddingHits = persistentPromptCacheVisualEmbeddingHits;
        self.persistentPromptCacheVisualEmbeddingMisses = persistentPromptCacheVisualEmbeddingMisses;
        self.persistentPromptCacheVisualEmbeddingRowsLoaded = persistentPromptCacheVisualEmbeddingRowsLoaded;
    }

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "persistent_prompt_cache_hits", value: .unsignedInteger(self.persistentPromptCacheHits));
        wireObject.appendEntry(key: "persistent_prompt_cache_misses", value: .unsignedInteger(self.persistentPromptCacheMisses));
        wireObject.appendEntry(key: "persistent_prompt_cache_tokens_saved", value: .unsignedInteger(self.persistentPromptCacheTokensSaved));
        wireObject.appendEntry(key: "persistent_prompt_cache_partial_tail_hits", value: .unsignedInteger(self.persistentPromptCachePartialTailHits));
        wireObject.appendEntry(key: "persistent_prompt_cache_block_token_count", value: .unsignedInteger(self.persistentPromptCacheBlockTokenCount));
        wireObject.appendEntry(key: "persistent_prompt_cache_sequence_state_block_count", value: .unsignedInteger(self.persistentPromptCacheSequenceStateBlockCount));
        wireObject.appendEntry(key: "persistent_prompt_cache_boundary_state_snapshot_count", value: .unsignedInteger(self.persistentPromptCacheBoundaryStateSnapshotCount));
        wireObject.appendEntry(key: "persistent_prompt_cache_visual_embedding_count", value: .unsignedInteger(self.persistentPromptCacheVisualEmbeddingCount));
        wireObject.appendEntry(key: "persistent_prompt_cache_total_size_bytes", value: .unsignedInteger(self.persistentPromptCacheTotalSizeBytes));
        wireObject.appendEntry(key: "persistent_prompt_cache_visual_embedding_total_size_bytes", value: .unsignedInteger(self.persistentPromptCacheVisualEmbeddingTotalSizeBytes));
        wireObject.appendEntry(key: "persistent_prompt_cache_maximum_size_bytes", value: .unsignedInteger(self.persistentPromptCacheMaximumSizeBytes));
        wireObject.appendEntry(key: "persistent_prompt_cache_visual_embedding_hits", value: .unsignedInteger(self.persistentPromptCacheVisualEmbeddingHits));
        wireObject.appendEntry(key: "persistent_prompt_cache_visual_embedding_misses", value: .unsignedInteger(self.persistentPromptCacheVisualEmbeddingMisses));
        wireObject.appendEntry(key: "persistent_prompt_cache_visual_embedding_rows_loaded", value: .unsignedInteger(self.persistentPromptCacheVisualEmbeddingRowsLoaded));
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerPersistentPromptCacheStats {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedStats = WorkerPersistentPromptCacheStats(
            persistentPromptCacheHits: try wireObject.decodeUInt64(fieldName: "persistent_prompt_cache_hits"),
            persistentPromptCacheMisses: try wireObject.decodeUInt64(fieldName: "persistent_prompt_cache_misses"),
            persistentPromptCacheTokensSaved: try wireObject.decodeUInt64(fieldName: "persistent_prompt_cache_tokens_saved"),
            persistentPromptCachePartialTailHits: try wireObject.decodeUInt64(fieldName: "persistent_prompt_cache_partial_tail_hits"),
            persistentPromptCacheBlockTokenCount: try wireObject.decodeUInt64(fieldName: "persistent_prompt_cache_block_token_count"),
            persistentPromptCacheSequenceStateBlockCount: try wireObject.decodeUInt64(fieldName: "persistent_prompt_cache_sequence_state_block_count"),
            persistentPromptCacheBoundaryStateSnapshotCount: try wireObject.decodeUInt64(fieldName: "persistent_prompt_cache_boundary_state_snapshot_count"),
            persistentPromptCacheVisualEmbeddingCount: try wireObject.decodeUInt64(fieldName: "persistent_prompt_cache_visual_embedding_count"),
            persistentPromptCacheTotalSizeBytes: try wireObject.decodeUInt64(fieldName: "persistent_prompt_cache_total_size_bytes"),
            persistentPromptCacheVisualEmbeddingTotalSizeBytes: try wireObject.decodeUInt64(fieldName: "persistent_prompt_cache_visual_embedding_total_size_bytes"),
            persistentPromptCacheMaximumSizeBytes: try wireObject.decodeUInt64(fieldName: "persistent_prompt_cache_maximum_size_bytes"),
            persistentPromptCacheVisualEmbeddingHits: try wireObject.decodeUInt64(fieldName: "persistent_prompt_cache_visual_embedding_hits"),
            persistentPromptCacheVisualEmbeddingMisses: try wireObject.decodeUInt64(fieldName: "persistent_prompt_cache_visual_embedding_misses"),
            persistentPromptCacheVisualEmbeddingRowsLoaded: try wireObject.decodeUInt64(fieldName: "persistent_prompt_cache_visual_embedding_rows_loaded"));
        try wireObject.rejectUnknownFields(allowedFieldNames: WorkerPersistentPromptCacheStats.wireFieldNames);
        return parsedStats;
    }

    static let wireFieldNames: Array<String> = [
        "persistent_prompt_cache_hits", "persistent_prompt_cache_misses", "persistent_prompt_cache_tokens_saved",
        "persistent_prompt_cache_partial_tail_hits", "persistent_prompt_cache_block_token_count",
        "persistent_prompt_cache_sequence_state_block_count", "persistent_prompt_cache_boundary_state_snapshot_count",
        "persistent_prompt_cache_visual_embedding_count", "persistent_prompt_cache_total_size_bytes",
        "persistent_prompt_cache_visual_embedding_total_size_bytes", "persistent_prompt_cache_maximum_size_bytes",
        "persistent_prompt_cache_visual_embedding_hits", "persistent_prompt_cache_visual_embedding_misses",
        "persistent_prompt_cache_visual_embedding_rows_loaded",
    ];
}

/// An event emitted by the inference worker.
///
/// Wire shape is internally tagged with `kind` in snake_case and rejects
/// unknown fields, mirroring the Rust enum's
/// `#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]`.
public enum WorkerEvent: Equatable {
    /// Confirms the feature settings applied by the currently running worker.
    case runtimeFeatureConfigurationApplied(WorkerRuntimeFeatureConfiguration);
    /// Reports that the worker process is running without a model loaded.
    case idle(machineMlxMemoryCeilingBytes: UInt64, effectiveMlxMemoryCeilingBytes: UInt64, minimumMlxMemoryCeilingBytes: UInt64);
    /// Delivers an explicitly requested idle or model-load MLX observation.
    case mlxMemorySample(mlxMemorySnapshot: WorkerMlxMemorySnapshot?, expertResidency: WorkerExpertResidencySnapshot?);
    /// Reports one accepted live MLX memory-ceiling adjustment.
    case mlxMemoryLimitChanged(
        effectiveMlxMemoryCeilingBytes: UInt64,
        minimumMlxMemoryCeilingBytes: UInt64,
        expertMemoryMode: ExpertMemoryMode,
        mlxMemorySnapshot: WorkerMlxMemorySnapshot?,
        expertResidency: WorkerExpertResidencySnapshot?);
    /// Reports that an MLX memory-ceiling adjustment was rejected without mutation.
    case mlxMemoryLimitRejected(
        requestedMlxMemoryCeilingBytes: UInt64,
        minimumMlxMemoryCeilingBytes: UInt64,
        machineMlxMemoryCeilingBytes: UInt64,
        reason: String);
    /// Reports a change in sparse-expert residency without affecting model output.
    case expertMemoryModeChanged(expertMemoryMode: ExpertMemoryMode);
    /// Reports final engine residency and MLX memory after request cleanup.
    case generationFinalized(
        requestId: RequestId,
        expertMemoryMode: ExpertMemoryMode?,
        mlxMemorySnapshot: WorkerMlxMemorySnapshot?,
        expertResidency: WorkerExpertResidencySnapshot?);
    /// Reports image-generation phase and denoising-step progress.
    case imageGenerationProgress(
        requestId: RequestId,
        phase: ImageGenerationPhase,
        completedSteps: UInt16,
        totalSteps: UInt16,
        elapsedMillis: UInt64,
        mlxMemorySnapshot: WorkerMlxMemorySnapshot?);
    /// Delivers one completed encoded image and its reproducibility metadata.
    case imageGenerationCompleted(requestId: RequestId, generatedImage: GeneratedImage, resultMetadata: ImageGenerationResultMetadata);
    /// Reports a request-scoped image failure that leaves the worker responsive.
    case imageGenerationFailed(requestId: RequestId, reason: ImageGenerationFailureReason);
    /// Confirms that image request state has been released after any outcome.
    case imageGenerationFinalized(requestId: RequestId, elapsedMillis: UInt64, mlxMemorySnapshot: WorkerMlxMemorySnapshot?);
    /// Delivers one completed embedding vector per input, in request order.
    case embeddingsCompleted(
        requestId: RequestId,
        embeddings: Array<Array<Float>>,
        inputTokenCounts: Array<UInt32>,
        elapsedMillis: UInt64);
    /// Reports a request-scoped embeddings failure that leaves the worker responsive.
    case embeddingsFailed(requestId: RequestId, reason: EmbeddingsFailureReason);
    /// Confirms that embeddings request state has been released after any outcome.
    case embeddingsFinalized(requestId: RequestId, elapsedMillis: UInt64, mlxMemorySnapshot: WorkerMlxMemorySnapshot?);
    /// Reports that the configured model finished loading.
    case ready(
        modelId: String,
        capabilities: WorkerModelCapabilities);
    /// Delivers one or more ordered model outputs in a single frame.
    case output(
        requestId: RequestId,
        sequenceNumber: UInt16,
        generatedTokenCount: UInt16,
        outputs: Array<ChatGenerationOutput>,
        mlxMemorySnapshot: WorkerMlxMemorySnapshot?,
        expertResidency: WorkerExpertResidencySnapshot?);
    /// Reports initial prompt-processing status or one completed prompt-processing chunk.
    case prefillProgress(
        requestId: RequestId,
        promptProcessingPhase: WorkerPromptProcessingPhase,
        processedTokens: UInt32,
        totalTokens: UInt32,
        elapsedMillis: UInt64,
        forwardPrefillChunkElapsedMillis: UInt64?,
        completedPrefillChunkTokens: UInt32?,
        mlxMemorySnapshot: WorkerMlxMemorySnapshot?,
        expertResidency: WorkerExpertResidencySnapshot?);
    /// Reports the explicit barrier between final prompt processing and first decode.
    case generationPreparationStarted(
        requestId: RequestId,
        totalLayerCount: UInt32,
        residentExpertCount: UInt32,
        residentExpertPayloadBytes: UInt64,
        mlxMemorySnapshot: WorkerMlxMemorySnapshot?);
    /// Reports generated-token progress that has not necessarily produced public output yet.
    case generationProgress(
        requestId: RequestId,
        generatedTokenCount: UInt16,
        maximumOutputTokens: UInt16,
        elapsedMillis: UInt64,
        mlxMemorySnapshot: WorkerMlxMemorySnapshot?,
        expertResidency: WorkerExpertResidencySnapshot?);
    /// Reports the measured first decode forward independently from preparation and output.
    case firstDecodeCompleted(requestId: RequestId, elapsedMillis: UInt64);
    /// Reports model-row work avoided through exact reusable prompt state.
    case promptWorkReuse(requestId: RequestId, promptWorkReuse: WorkerPromptWorkReuse);
    /// Reports normal completion, including cancellation.
    case completed(
        requestId: RequestId,
        promptTokenCount: UInt32,
        generatedTokenCount: UInt16,
        reasoningTokenCount: UInt16,
        cachedTokenCount: UInt32,
        persistentPromptCacheDiagnostics: WorkerPersistentPromptCacheRequestDiagnostics?,
        reason: ChatGenerationCompletionReason);
    /// Reports a request-scoped failure that leaves the process responsive.
    case failed(requestId: RequestId, reason: ChatGenerationFailureReason);
    /// Reports that a model swap completed successfully and the new model is loaded.
    case modelSwapped(
        modelId: String,
        capabilities: WorkerModelCapabilities,
        expertMemoryMode: ExpertMemoryMode?,
        minimumMlxMemoryCeilingBytes: UInt64);
    /// Reports that a model swap failed while the worker process remained responsive.
    case modelSwapFailed(loadedModelRemainsReady: Bool, modelLoadFailureReason: String);
    /// Reports cumulative persistent prompt-cache observability counters and disk footprint.
    case persistentPromptCacheStats(WorkerPersistentPromptCacheStats);
    /// Confirms a persistent prompt-cache deletion completed.
    case promptCacheCleared(modelId: String?, blocksRemoved: UInt64, bytesFreed: UInt64);

    internal static let expectedVariantNames: Array<String> = [
        "runtime_feature_configuration_applied", "idle", "mlx_memory_sample", "mlx_memory_limit_changed",
        "mlx_memory_limit_rejected", "expert_memory_mode_changed", "generation_finalized",
        "image_generation_progress", "image_generation_completed", "image_generation_failed",
        "image_generation_finalized", "embeddings_completed", "embeddings_failed", "embeddings_finalized",
        "ready", "output", "prefill_progress", "generation_preparation_started", "generation_progress",
        "first_decode_completed", "prompt_work_reuse", "completed", "failed", "model_swapped",
        "model_swap_failed", "persistent_prompt_cache_stats", "prompt_cache_cleared",
    ];
}

/// Shared wire encoders and decoders for the optional and collection fields of
/// `WorkerEvent` variants; every variant's Option serializes as an explicit
/// `null` when absent, matching serde's default `Option` behavior.
internal enum WorkerEventWireValues {
    internal static func optionalMlxMemorySnapshotWireValue(_ optionalSnapshot: WorkerMlxMemorySnapshot?) -> JsonWireValue {
        guard let unwrappedSnapshot = optionalSnapshot else {
            return .null;
        }
        return unwrappedSnapshot.wireValue();
    }

    internal static func optionalExpertResidencyWireValue(_ optionalResidency: WorkerExpertResidencySnapshot?) -> JsonWireValue {
        guard let unwrappedResidency = optionalResidency else {
            return .null;
        }
        return unwrappedResidency.wireValue();
    }

    internal static func optionalExpertMemoryModeWireValue(_ optionalExpertMemoryMode: ExpertMemoryMode?) -> JsonWireValue {
        guard let unwrappedExpertMemoryMode = optionalExpertMemoryMode else {
            return .null;
        }
        return unwrappedExpertMemoryMode.wireValue();
    }

    internal static func optionalPromptCacheDiagnosticsWireValue(_ optionalDiagnostics: WorkerPersistentPromptCacheRequestDiagnostics?) -> JsonWireValue {
        guard let unwrappedDiagnostics = optionalDiagnostics else {
            return .null;
        }
        return unwrappedDiagnostics.wireValue();
    }

    internal static func optionalStringWireValue(_ optionalValue: String?) -> JsonWireValue {
        guard let unwrappedValue = optionalValue else {
            return .null;
        }
        return .string(unwrappedValue);
    }

    internal static func optionalUInt64WireValue(_ optionalValue: UInt64?) -> JsonWireValue {
        guard let unwrappedValue = optionalValue else {
            return .null;
        }
        return .unsignedInteger(unwrappedValue);
    }

    internal static func optionalUInt32WireValue(_ optionalValue: UInt32?) -> JsonWireValue {
        guard let unwrappedValue = optionalValue else {
            return .null;
        }
        return .unsignedInteger(UInt64(unwrappedValue));
    }

    internal static func nestedFloat32ArrayWireValue(_ vectors: Array<Array<Float>>) -> JsonWireValue {
        return JsonWireValue.mappedArray(vectors, mappedWireValue: { (vector: Array<Float>) -> JsonWireValue in
            JsonWireValue.mappedArray(vector, mappedWireValue: { (vectorElement: Float) -> JsonWireValue in .float32(vectorElement) })
        });
    }

    internal static func nestedFloat32ArrayFromWireValue(_ wireValue: JsonWireValue) throws -> Array<Array<Float>> {
        return try JsonWireValue.extractArray(wireValue, mappedElement: { (vectorWireValue: JsonWireValue) throws -> Array<Float> in
            try JsonWireValue.extractArray(vectorWireValue, mappedElement: WorkerEventWireValues.float32FromWireValue)
        });
    }

    internal static func uint32ArrayWireValue(_ numericValues: Array<UInt32>) -> JsonWireValue {
        return JsonWireValue.mappedArray(numericValues, mappedWireValue: { (numericValue: UInt32) -> JsonWireValue in .unsignedInteger(UInt64(numericValue)) });
    }

    internal static func uint32ArrayFromWireValue(_ wireValue: JsonWireValue) throws -> Array<UInt32> {
        return try JsonWireValue.extractArray(wireValue, mappedElement: { (elementWireValue: JsonWireValue) throws -> UInt32 in
            try JsonWireValue.clampToUInt32(try JsonWireValue.extractUInt64(elementWireValue))
        });
    }

    internal static func chatGenerationOutputArrayWireValue(_ outputs: Array<ChatGenerationOutput>) -> JsonWireValue {
        return JsonWireValue.mappedArray(outputs, mappedWireValue: { (output: ChatGenerationOutput) -> JsonWireValue in output.wireValue() });
    }

    internal static func chatGenerationOutputArrayFromWireValue(_ wireValue: JsonWireValue) throws -> Array<ChatGenerationOutput> {
        return try JsonWireValue.extractArray(wireValue, mappedElement: { (elementWireValue: JsonWireValue) throws -> ChatGenerationOutput in
            try ChatGenerationOutput.fromWireValue(elementWireValue)
        });
    }

    internal static func decodeOptionalMlxMemorySnapshot(wireObject: JsonWireObject, fieldName: String) throws -> WorkerMlxMemorySnapshot? {
        guard let snapshotObject = try wireObject.decodeOptionalObject(fieldName: fieldName) else {
            return nil;
        }
        return try WorkerMlxMemorySnapshot.fromWireValue(.object(snapshotObject));
    }

    internal static func decodeOptionalExpertResidency(wireObject: JsonWireObject, fieldName: String) throws -> WorkerExpertResidencySnapshot? {
        guard let residencyObject = try wireObject.decodeOptionalObject(fieldName: fieldName) else {
            return nil;
        }
        return try WorkerExpertResidencySnapshot.fromWireValue(.object(residencyObject));
    }

    internal static func decodeOptionalExpertMemoryMode(wireObject: JsonWireObject, fieldName: String) throws -> ExpertMemoryMode? {
        guard let expertMemoryModeName = try wireObject.decodeOptionalString(fieldName: fieldName) else {
            return nil;
        }
        return try ExpertMemoryMode.fromWireValue(.string(expertMemoryModeName));
    }

    internal static func decodeOptionalPromptCacheDiagnostics(wireObject: JsonWireObject, fieldName: String) throws -> WorkerPersistentPromptCacheRequestDiagnostics? {
        guard let diagnosticsObject = try wireObject.decodeOptionalObject(fieldName: fieldName) else {
            return nil;
        }
        return try WorkerPersistentPromptCacheRequestDiagnostics.fromWireValue(.object(diagnosticsObject));
    }

    private static func float32FromWireValue(_ elementWireValue: JsonWireValue) throws -> Float {
        switch elementWireValue {
        case let .float32(numericValue): return numericValue;
        case let .double(numericValue): return Float(numericValue);
        case let .unsignedInteger(numericValue): return Float(numericValue);
        case let .signedInteger(numericValue): return Float(numericValue);
        default: throw JsonWireProblem.invalidType(expectedTypeName: "f32", found: elementWireValue.foundDescription);
        }
    }
}
