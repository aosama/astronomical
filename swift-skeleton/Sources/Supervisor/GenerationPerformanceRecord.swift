import Foundation

import IpcProtocol

/**
 * One row in the generation performance log, mirroring the Rust record from
 * apps/supervisor/src/generation_performance_log.rs.
 *
 * Each completed generation request appends one JSON line to the selected
 * instance's `logs/performance.jsonl`. Fields are chosen to answer "how well
 * is the model performing?" at a glance:
 *
 * - Throughput: `prefill_tok_per_second` and `generation_tok_per_second`
 * - Latency: `total_elapsed_millis`, `prefill_elapsed_millis`, `generation_elapsed_millis`
 * - Scale: `prompt_token_count`, `cached_token_count`, `generated_token_count`
 * - Resources: `mlx_peak_memory_bytes`, `mlx_active_memory_bytes`
 * - Identity: `request_id`, `model_id`, `completion_reason`
 */
public struct GenerationPerformanceRecord: Equatable {

    /// Unix epoch milliseconds when the record was written (request completion time).
    public let timestampMillis: UInt64

    /// Supervisor-local monotonic request identifier.
    public let requestId: UInt64

    /// The model that produced this generation.
    public let modelId: String

    /// Total prompt tokens (including cached tokens).
    public let promptTokenCount: UInt32

    /// Prompt tokens restored from the persistent SSD cache.
    public let cachedTokenCount: UInt32

    /// Output tokens the model produced.
    public let generatedTokenCount: UInt16

    /// Why the generation stopped: `end_of_sequence`, `tool_calls`, or `maximum_output_tokens`.
    public let completionReason: String

    /// Accumulated prompt-processing time across all prefill chunks, in milliseconds.
    public let prefillElapsedMillis: UInt64

    /// Wall-clock decode time from the first output token to completion, in milliseconds.
    public let generationElapsedMillis: UInt64

    /// Wall-clock time from request arrival to completion, in milliseconds.
    public let totalElapsedMillis: UInt64

    /// User-visible latency from accepted request to first public output.
    public let timeToFirstOutputMillis: UInt64?

    /// Time between explicit preparation start and generation progress/output.
    public let generationPreparationElapsedMillis: UInt64?

    public let firstDecodeForwardElapsedMillis: UInt64?

    /// Source bytes read solely during preparation; zero proves no eager warm scan.
    public let generationPreparationExpertSourceReadByteCount: UInt64

    public let finalResidentExpertCount: UInt32?

    public let finalResidentExpertPayloadBytes: UInt64?

    /// Prefill throughput; nil when the entire prompt was cached (0 ms prefill).
    public let prefillTokPerSecond: Double?

    /// Decode throughput; nil when generation took 0 ms.
    public let generationTokPerSecond: Double?

    /// Peak MLX GPU memory observed during prefill, in bytes.
    public let mlxPeakMemoryBytes: UInt64?

    /// Active MLX GPU memory at last prefill progress event, in bytes.
    public let mlxActiveMemoryBytes: UInt64?

    /// Bounded cache lookup, publication, and reclamation attribution for this request.
    public let persistentPromptCacheDiagnostics: WorkerPersistentPromptCacheRequestDiagnostics?

    public init(
        timestampMillis: UInt64,
        requestId: UInt64,
        modelId: String,
        promptTokenCount: UInt32,
        cachedTokenCount: UInt32,
        generatedTokenCount: UInt16,
        completionReason: String,
        prefillElapsedMillis: UInt64,
        generationElapsedMillis: UInt64,
        totalElapsedMillis: UInt64,
        timeToFirstOutputMillis: UInt64?,
        generationPreparationElapsedMillis: UInt64?,
        firstDecodeForwardElapsedMillis: UInt64?,
        generationPreparationExpertSourceReadByteCount: UInt64,
        finalResidentExpertCount: UInt32?,
        finalResidentExpertPayloadBytes: UInt64?,
        prefillTokPerSecond: Double?,
        generationTokPerSecond: Double?,
        mlxPeakMemoryBytes: UInt64?,
        mlxActiveMemoryBytes: UInt64?,
        persistentPromptCacheDiagnostics: WorkerPersistentPromptCacheRequestDiagnostics?
    ) {
        self.timestampMillis = timestampMillis
        self.requestId = requestId
        self.modelId = modelId
        self.promptTokenCount = promptTokenCount
        self.cachedTokenCount = cachedTokenCount
        self.generatedTokenCount = generatedTokenCount
        self.completionReason = completionReason
        self.prefillElapsedMillis = prefillElapsedMillis
        self.generationElapsedMillis = generationElapsedMillis
        self.totalElapsedMillis = totalElapsedMillis
        self.timeToFirstOutputMillis = timeToFirstOutputMillis
        self.generationPreparationElapsedMillis = generationPreparationElapsedMillis
        self.firstDecodeForwardElapsedMillis = firstDecodeForwardElapsedMillis
        self.generationPreparationExpertSourceReadByteCount = generationPreparationExpertSourceReadByteCount
        self.finalResidentExpertCount = finalResidentExpertCount
        self.finalResidentExpertPayloadBytes = finalResidentExpertPayloadBytes
        self.prefillTokPerSecond = prefillTokPerSecond
        self.generationTokPerSecond = generationTokPerSecond
        self.mlxPeakMemoryBytes = mlxPeakMemoryBytes
        self.mlxActiveMemoryBytes = mlxActiveMemoryBytes
        self.persistentPromptCacheDiagnostics = persistentPromptCacheDiagnostics
    }

    /**
     * Computes the throughput fields from raw counters and elapsed times.
     *
     * - `prefillTokPerSecond` is nil when `prefillElapsedMillis == 0` (fully cached)
     *   or when no uncached tokens remain.
     * - `generationTokPerSecond` is nil when `generationElapsedMillis == 0`
     *   or when no tokens were generated.
     *
     * - Returns: the prefill and generation throughput pair, each optional.
     */
    public static func computeThroughput(
        promptTokenCount: UInt32,
        cachedTokenCount: UInt32,
        generatedTokenCount: UInt16,
        prefillElapsedMillis: UInt64,
        generationElapsedMillis: UInt64
    ) -> (prefillTokPerSecond: Double?, generationTokPerSecond: Double?) {
        let uncachedPromptTokens: UInt32 = promptTokenCount >= cachedTokenCount
            ? promptTokenCount - cachedTokenCount
            : 0
        let prefillTokPerSecond: Double? = (prefillElapsedMillis > 0 && uncachedPromptTokens > 0)
            ? Double(uncachedPromptTokens) / (Double(prefillElapsedMillis) / 1000.0)
            : nil
        let generationTokPerSecond: Double? = (generationElapsedMillis > 0 && generatedTokenCount > 0)
            ? Double(generatedTokenCount) / (Double(generationElapsedMillis) / 1000.0)
            : nil
        return (prefillTokPerSecond, generationTokPerSecond)
    }

    /// The serde-shaped JSON object written to `performance.jsonl`; key order
    /// mirrors the Rust struct declaration so rows diff cleanly across stacks.
    public func jsonlWireValue() -> JsonWireValue {
        let wireObject: JsonWireObject = JsonWireObject(entries: [])
        return .object(GenerationPerformanceRecord.wireObject(from: self, into: wireObject))
    }

    private static func wireObject(
        from record: GenerationPerformanceRecord,
        into wireObject: JsonWireObject
    ) -> JsonWireObject {
        var builtObject: JsonWireObject = wireObject
        builtObject.appendEntry(key: "timestamp_millis", value: .unsignedInteger(record.timestampMillis))
        builtObject.appendEntry(key: "request_id", value: .unsignedInteger(record.requestId))
        builtObject.appendEntry(key: "model_id", value: .string(record.modelId))
        builtObject.appendEntry(key: "prompt_token_count", value: .unsignedInteger(UInt64(record.promptTokenCount)))
        builtObject.appendEntry(key: "cached_token_count", value: .unsignedInteger(UInt64(record.cachedTokenCount)))
        builtObject.appendEntry(key: "generated_token_count", value: .unsignedInteger(UInt64(record.generatedTokenCount)))
        builtObject.appendEntry(key: "completion_reason", value: .string(record.completionReason))
        builtObject.appendEntry(key: "prefill_elapsed_millis", value: .unsignedInteger(record.prefillElapsedMillis))
        builtObject.appendEntry(key: "generation_elapsed_millis", value: .unsignedInteger(record.generationElapsedMillis))
        builtObject.appendEntry(key: "total_elapsed_millis", value: .unsignedInteger(record.totalElapsedMillis))
        builtObject.appendEntry(key: "time_to_first_output_millis", value: GenerationPerformanceRecord.optionalUInteger(record.timeToFirstOutputMillis))
        builtObject.appendEntry(key: "generation_preparation_elapsed_millis", value: GenerationPerformanceRecord.optionalUInteger(record.generationPreparationElapsedMillis))
        builtObject.appendEntry(key: "first_decode_forward_elapsed_millis", value: GenerationPerformanceRecord.optionalUInteger(record.firstDecodeForwardElapsedMillis))
        builtObject.appendEntry(key: "generation_preparation_expert_source_read_byte_count", value: .unsignedInteger(record.generationPreparationExpertSourceReadByteCount))
        builtObject.appendEntry(key: "final_resident_expert_count", value: GenerationPerformanceRecord.optionalUInteger(record.finalResidentExpertCount.map({ (expertCount: UInt32) -> UInt64 in return UInt64(expertCount) })))
        builtObject.appendEntry(key: "final_resident_expert_payload_bytes", value: GenerationPerformanceRecord.optionalUInteger(record.finalResidentExpertPayloadBytes))
        builtObject.appendEntry(key: "prefill_tok_per_second", value: GenerationPerformanceRecord.optionalDouble(record.prefillTokPerSecond))
        builtObject.appendEntry(key: "generation_tok_per_second", value: GenerationPerformanceRecord.optionalDouble(record.generationTokPerSecond))
        builtObject.appendEntry(key: "mlx_peak_memory_bytes", value: GenerationPerformanceRecord.optionalUInteger(record.mlxPeakMemoryBytes))
        builtObject.appendEntry(key: "mlx_active_memory_bytes", value: GenerationPerformanceRecord.optionalUInteger(record.mlxActiveMemoryBytes))
        builtObject.appendEntry(key: "persistent_prompt_cache_diagnostics", value: record.persistentPromptCacheDiagnostics?.performanceLogWireValue() ?? .null)
        return builtObject
    }

    private static func optionalUInteger(_ optionalValue: UInt64?) -> JsonWireValue {
        guard let unwrappedValue: UInt64 = optionalValue else {
            return .null
        }
        return .unsignedInteger(unwrappedValue)
    }

    private static func optionalDouble(_ optionalValue: Double?) -> JsonWireValue {
        guard let unwrappedValue: Double = optionalValue else {
            return .null
        }
        return .double(unwrappedValue)
    }
}
