import Foundation;

/// Model-loading payload of one serialized attribution report.
public struct ModelLoadingPerformanceAttributionReport: Sendable {

    let common: CommonPerformanceAttributionReport;
    let modelId: String?;
    let modelRevision: String?;
    let prefillTransientObservationCompleted: Bool;
    let prefillObservedTransientHighWaterBytes: UInt64;
    let totalArtifactPayloadBytes: UInt64?;
    let residentModelPayloadBytes: UInt64?;
    let modelShardCount: Int?;
    let mlxActiveMemoryBytes: UInt64?;
    let mlxAllocatorCacheMemoryBytes: UInt64?;
    let mlxPeakMemoryBytes: UInt64?;
    let failureDescription: String?;

    func encodePayloadFields(
        into container: inout KeyedEncodingContainer<AnyCodingKey>
    ) throws -> Void {
        try common.encodeFields(into: &container);
        encodeOptionalString(modelId, into: &container, key: "model_id");
        encodeOptionalString(modelRevision, into: &container, key: "model_revision");
        try container.encode(
            prefillTransientObservationCompleted,
            forKey: AnyCodingKey("prefill_transient_observation_completed"));
        try container.encode(
            prefillObservedTransientHighWaterBytes,
            forKey: AnyCodingKey("prefill_observed_transient_high_water_bytes"));
        encodeOptionalUInt64(totalArtifactPayloadBytes, into: &container, key: "total_artifact_payload_bytes");
        encodeOptionalUInt64(residentModelPayloadBytes, into: &container, key: "resident_model_payload_bytes");
        encodeOptionalInt(modelShardCount, into: &container, key: "model_shard_count");
        encodeOptionalUInt64(mlxActiveMemoryBytes, into: &container, key: "mlx_active_memory_bytes");
        encodeOptionalUInt64(mlxAllocatorCacheMemoryBytes, into: &container, key: "mlx_allocator_cache_memory_bytes");
        encodeOptionalUInt64(mlxPeakMemoryBytes, into: &container, key: "mlx_peak_memory_bytes");
        encodeOptionalString(failureDescription, into: &container, key: "failure_description");
    }
}

/// Generation payload of one serialized attribution report.
public struct GenerationPerformanceAttributionReport: Sendable {

    let common: CommonPerformanceAttributionReport;
    let modelId: String;
    let modelRevision: String;
    let prefillTransientObservationCompleted: Bool;
    let prefillObservedTransientHighWaterBytes: UInt64;
    let requestId: UInt64;
    let configuredMaximumOutputTokens: UInt16;
    let mlxActiveMemoryBytes: UInt64?;
    let mlxAllocatorCacheMemoryBytes: UInt64?;
    let mlxPeakMemoryBytes: UInt64?;
    let failureDescription: String?;
    let previousTokenExpertRouteReuseByLayer: [PreviousTokenExpertRouteReuseByLayerReport];
    let expertStreamingSourceSummaries: [ExpertStreamingSourceSummary];

    func encodePayloadFields(
        into container: inout KeyedEncodingContainer<AnyCodingKey>
    ) throws -> Void {
        try common.encodeFields(into: &container);
        try container.encode(modelId, forKey: AnyCodingKey("model_id"));
        try container.encode(modelRevision, forKey: AnyCodingKey("model_revision"));
        try container.encode(
            prefillTransientObservationCompleted,
            forKey: AnyCodingKey("prefill_transient_observation_completed"));
        try container.encode(
            prefillObservedTransientHighWaterBytes,
            forKey: AnyCodingKey("prefill_observed_transient_high_water_bytes"));
        try container.encode(requestId, forKey: AnyCodingKey("request_id"));
        try container.encode(
            configuredMaximumOutputTokens,
            forKey: AnyCodingKey("configured_maximum_output_tokens"));
        encodeOptionalUInt64(mlxActiveMemoryBytes, into: &container, key: "mlx_active_memory_bytes");
        encodeOptionalUInt64(mlxAllocatorCacheMemoryBytes, into: &container, key: "mlx_allocator_cache_memory_bytes");
        encodeOptionalUInt64(mlxPeakMemoryBytes, into: &container, key: "mlx_peak_memory_bytes");
        encodeOptionalString(failureDescription, into: &container, key: "failure_description");
        var routeReuseContainer = container.nestedUnkeyedContainer(
            forKey: AnyCodingKey("previous_token_expert_route_reuse_by_layer"));
        for routeReuseReport in previousTokenExpertRouteReuseByLayer {
            var routeReuseRow = routeReuseContainer.nestedContainer(keyedBy: AnyCodingKey.self);
            try routeReuseReport.encodeFields(into: &routeReuseRow);
        }
        var sourceSummariesContainer = container.nestedUnkeyedContainer(
            forKey: AnyCodingKey("expert_streaming_source_summaries"));
        for sourceSummary in expertStreamingSourceSummaries {
            var sourceSummaryRow = sourceSummariesContainer.nestedContainer(keyedBy: AnyCodingKey.self);
            try sourceSummary.encodeFields(into: &sourceSummaryRow);
        }
    }
}

/// One per-layer previous-token route reuse row, serialized without expert
/// identifiers so diagnostics never carry route contents.
struct PreviousTokenExpertRouteReuseByLayerReport: Sendable {

    let layerIndex: Int;
    let predictedExpertCount: UInt64;
    let matchedExpertCount: UInt64;
    let completelyMatchedLayerCount: UInt64;
    let examinedLayerCount: UInt64;

    func encodeFields(
        into container: inout KeyedEncodingContainer<AnyCodingKey>
    ) throws -> Void {
        try container.encode(layerIndex, forKey: AnyCodingKey("layer_index"));
        try container.encode(predictedExpertCount, forKey: AnyCodingKey("predicted_expert_count"));
        try container.encode(matchedExpertCount, forKey: AnyCodingKey("matched_expert_count"));
        try container.encode(
            completelyMatchedLayerCount,
            forKey: AnyCodingKey("completely_matched_layer_count"));
        try container.encode(examinedLayerCount, forKey: AnyCodingKey("examined_layer_count"));
    }
}

extension ExpertStreamingSourceSummary {

    func encodeFields(
        into container: inout KeyedEncodingContainer<AnyCodingKey>
    ) throws -> Void {
        try container.encode(phase.rawValue, forKey: AnyCodingKey("phase"));
        try container.encode(layerIndex, forKey: AnyCodingKey("layer_index"));
        try container.encode(sourcePlanCount, forKey: AnyCodingKey("source_plan_count"));
        try container.encode(totalRouteTokenCount, forKey: AnyCodingKey("total_route_token_count"));
        try container.encode(totalRoutedExpertCount, forKey: AnyCodingKey("total_routed_expert_count"));
        try container.encode(totalStreamedExpertCount, forKey: AnyCodingKey("total_streamed_expert_count"));
        try container.encode(totalSourceShardCount, forKey: AnyCodingKey("total_source_shard_count"));
        try container.encode(payloadByteCount, forKey: AnyCodingKey("payload_byte_count"));
    }
}

/// One memory snapshot boundary inside an image-generation report.
struct ImageGenerationMemorySnapshot: Sendable {

    let phase: String;
    let mlxActiveMemoryBytes: UInt64?;
    let mlxAllocatorCacheMemoryBytes: UInt64?;
    let mlxPeakMemoryBytes: UInt64?;

    func encodeFields(
        into container: inout KeyedEncodingContainer<AnyCodingKey>
    ) throws -> Void {
        try container.encode(phase, forKey: AnyCodingKey("phase"));
        encodeOptionalUInt64(mlxActiveMemoryBytes, into: &container, key: "mlx_active_memory_bytes");
        encodeOptionalUInt64(mlxAllocatorCacheMemoryBytes, into: &container, key: "mlx_allocator_cache_memory_bytes");
        encodeOptionalUInt64(mlxPeakMemoryBytes, into: &container, key: "mlx_peak_memory_bytes");
    }
}

/// Image-generation payload of one serialized attribution report.
public struct ImageGenerationPerformanceAttributionReport: Sendable {

    let common: CommonPerformanceAttributionReport;
    let requestId: UInt64;
    let modelId: String;
    let modelRevision: String;
    let widthPixels: UInt32;
    let heightPixels: UInt32;
    let steps: UInt16;
    let guidanceThousandths: UInt32;
    let seed: UInt64;
    let encodedBytes: UInt64?;
    let memorySnapshots: [ImageGenerationMemorySnapshot];
    let failureDescription: String?;

    func encodePayloadFields(
        into container: inout KeyedEncodingContainer<AnyCodingKey>
    ) throws -> Void {
        try common.encodeFields(into: &container);
        try container.encode(requestId, forKey: AnyCodingKey("request_id"));
        try container.encode(modelId, forKey: AnyCodingKey("model_id"));
        try container.encode(modelRevision, forKey: AnyCodingKey("model_revision"));
        try container.encode(widthPixels, forKey: AnyCodingKey("width_pixels"));
        try container.encode(heightPixels, forKey: AnyCodingKey("height_pixels"));
        try container.encode(steps, forKey: AnyCodingKey("steps"));
        try container.encode(guidanceThousandths, forKey: AnyCodingKey("guidance_thousandths"));
        try container.encode(seed, forKey: AnyCodingKey("seed"));
        encodeOptionalUInt64(encodedBytes, into: &container, key: "encoded_bytes");
        var memorySnapshotsContainer = container.nestedUnkeyedContainer(
            forKey: AnyCodingKey("memory_snapshots"));
        for memorySnapshot in memorySnapshots {
            var memorySnapshotRow = memorySnapshotsContainer.nestedContainer(keyedBy: AnyCodingKey.self);
            try memorySnapshot.encodeFields(into: &memorySnapshotRow);
        }
        encodeOptionalString(failureDescription, into: &container, key: "failure_description");
    }
}

/// Embeddings payload of one serialized attribution report.
public struct EmbeddingsPerformanceAttributionReport: Sendable {

    let common: CommonPerformanceAttributionReport;
    let requestId: UInt64;
    let modelId: String;
    let inputCount: Int;
    let totalInputTokens: UInt32;
    let vectorWidth: UInt32;
    let failureDescription: String?;

    func encodePayloadFields(
        into container: inout KeyedEncodingContainer<AnyCodingKey>
    ) throws -> Void {
        try common.encodeFields(into: &container);
        try container.encode(requestId, forKey: AnyCodingKey("request_id"));
        try container.encode(modelId, forKey: AnyCodingKey("model_id"));
        try container.encode(inputCount, forKey: AnyCodingKey("input_count"));
        try container.encode(totalInputTokens, forKey: AnyCodingKey("total_input_tokens"));
        try container.encode(vectorWidth, forKey: AnyCodingKey("vector_width"));
        encodeOptionalString(failureDescription, into: &container, key: "failure_description");
    }
}
