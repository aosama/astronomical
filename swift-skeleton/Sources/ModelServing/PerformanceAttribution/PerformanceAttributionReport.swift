import Foundation;

/// Coding key that carries its serialized snake_case name explicitly, so the
/// attribution report's JSON contract stays independent of Swift property
/// naming and of any encoder key-mapping strategy.
struct AnyCodingKey: CodingKey {

    let stringValue: String;
    let intValue: Int?;

    init(_ stringValue: String) {
        self.stringValue = stringValue;
        self.intValue = nil;
    }

    init?(stringValue: String) {
        self.init(stringValue);
    }

    init?(intValue: Int) {
        self.stringValue = String(intValue);
        self.intValue = intValue;
    }
}

/// One serialized row of the fixed operation catalog.
struct PerformanceOperationReport: Sendable {

    let operation: String;
    let occurrenceCount: UInt64;
    let totalElapsedNanoseconds: UInt64;
    let minimumElapsedNanoseconds: UInt64;
    let maximumElapsedNanoseconds: UInt64;
    let firstStartedOffsetNanoseconds: UInt64;
    let lastEndedOffsetNanoseconds: UInt64;

    func encodeFields(
        into container: inout KeyedEncodingContainer<AnyCodingKey>
    ) throws -> Void {
        try container.encode(operation, forKey: AnyCodingKey("operation"));
        try container.encode(occurrenceCount, forKey: AnyCodingKey("occurrence_count"));
        try container.encode(totalElapsedNanoseconds, forKey: AnyCodingKey("total_elapsed_nanoseconds"));
        try container.encode(minimumElapsedNanoseconds, forKey: AnyCodingKey("minimum_elapsed_nanoseconds"));
        try container.encode(maximumElapsedNanoseconds, forKey: AnyCodingKey("maximum_elapsed_nanoseconds"));
        try container.encode(firstStartedOffsetNanoseconds, forKey: AnyCodingKey("first_started_offset_nanoseconds"));
        try container.encode(lastEndedOffsetNanoseconds, forKey: AnyCodingKey("last_ended_offset_nanoseconds"));
    }
}

/// One serialized row of the fixed counter catalog.
struct PerformanceCounterReport: Sendable {

    let counter: String;
    let amount: UInt64;
}

/// Fields shared by every attribution report kind, serialized first in each
/// payload so the JSON row stays layout-stable across report kinds.
struct CommonPerformanceAttributionReport: Sendable {

    let startedAtUnixMillis: UInt64;
    let endedAtUnixMillis: UInt64;
    let reportElapsedNanoseconds: UInt64;
    let attributedElapsedNanoseconds: UInt64;
    let unattributedElapsedNanoseconds: UInt64;
    let attributedPercent: Double;
    let outcome: PerformanceAttributionOutcome;
    let operations: [PerformanceOperationReport];
    let counters: [PerformanceCounterReport];
    let processPhysicalDiskReadBytes: UInt64?;
    let processPhysicalDiskWrittenBytes: UInt64?;
    let processIoUnavailabilityReason: String?;

    func encodeFields(
        into container: inout KeyedEncodingContainer<AnyCodingKey>
    ) throws -> Void {
        try container.encode(startedAtUnixMillis, forKey: AnyCodingKey("started_at_unix_millis"));
        try container.encode(endedAtUnixMillis, forKey: AnyCodingKey("ended_at_unix_millis"));
        try container.encode(reportElapsedNanoseconds, forKey: AnyCodingKey("report_elapsed_nanoseconds"));
        try container.encode(attributedElapsedNanoseconds, forKey: AnyCodingKey("attributed_elapsed_nanoseconds"));
        try container.encode(unattributedElapsedNanoseconds, forKey: AnyCodingKey("unattributed_elapsed_nanoseconds"));
        try container.encode(attributedPercent, forKey: AnyCodingKey("attributed_percent"));
        try container.encode(outcome.rawValue, forKey: AnyCodingKey("outcome"));
        var operationsContainer = container.nestedUnkeyedContainer(forKey: AnyCodingKey("operations"));
        for operationReport in operations {
            var operationContainer = operationsContainer.nestedContainer(keyedBy: AnyCodingKey.self);
            try operationReport.encodeFields(into: &operationContainer);
        }
        var countersContainer = container.nestedUnkeyedContainer(forKey: AnyCodingKey("counters"));
        for counterReport in counters {
            var counterContainer = countersContainer.nestedContainer(keyedBy: AnyCodingKey.self);
            try counterContainer.encode(counterReport.counter, forKey: AnyCodingKey("counter"));
            try counterContainer.encode(counterReport.amount, forKey: AnyCodingKey("amount"));
        }
        encodeOptionalUInt64(
            processPhysicalDiskReadBytes,
            into: &container,
            key: "process_physical_disk_read_bytes");
        encodeOptionalUInt64(
            processPhysicalDiskWrittenBytes,
            into: &container,
            key: "process_physical_disk_written_bytes");
        if let processIoUnavailabilityReason = processIoUnavailabilityReason {
            try container.encode(
                processIoUnavailabilityReason,
                forKey: AnyCodingKey("process_io_unavailability_reason"));
        } else {
            try container.encodeNil(forKey: AnyCodingKey("process_io_unavailability_reason"));
        }
    }
}

/// Encodes an optional byte count with serde-compatible `null` semantics, so
/// consumers can distinguish an unavailable sample from measured zero traffic.
func encodeOptionalUInt64(
    _ optionalAmount: UInt64?,
    into container: inout KeyedEncodingContainer<AnyCodingKey>,
    key: String
) -> Void {
    if let optionalAmount = optionalAmount {
        try? container.encode(optionalAmount, forKey: AnyCodingKey(key));
    } else {
        try? container.encodeNil(forKey: AnyCodingKey(key));
    }
}

/// Encodes an optional text field with serde-compatible `null` semantics.
func encodeOptionalString(
    _ optionalText: String?,
    into container: inout KeyedEncodingContainer<AnyCodingKey>,
    key: String
) -> Void {
    if let optionalText = optionalText {
        try? container.encode(optionalText, forKey: AnyCodingKey(key));
    } else {
        try? container.encodeNil(forKey: AnyCodingKey(key));
    }
}

/// Encodes an optional counted quantity with serde-compatible `null` semantics.
func encodeOptionalInt(
    _ optionalAmount: Int?,
    into container: inout KeyedEncodingContainer<AnyCodingKey>,
    key: String
) -> Void {
    if let optionalAmount = optionalAmount {
        try? container.encode(optionalAmount, forKey: AnyCodingKey(key));
    } else {
        try? container.encodeNil(forKey: AnyCodingKey(key));
    }
}

/// One serialized attribution report, tagged by `report_kind`.
public enum PerformanceAttributionReport: Encodable, Sendable {

    case modelLoading(ModelLoadingPerformanceAttributionReport);
    case generation(GenerationPerformanceAttributionReport);
    case imageGeneration(ImageGenerationPerformanceAttributionReport);
    case embeddings(EmbeddingsPerformanceAttributionReport);

    public func encode(to encoder: Encoder) throws -> Void {
        var container = encoder.container(keyedBy: AnyCodingKey.self);
        switch self {
        case .modelLoading(let modelLoadingReport):
            try container.encode("model_loading", forKey: AnyCodingKey("report_kind"));
            try modelLoadingReport.encodePayloadFields(into: &container);
        case .generation(let generationReport):
            try container.encode("generation", forKey: AnyCodingKey("report_kind"));
            try generationReport.encodePayloadFields(into: &container);
        case .imageGeneration(let imageGenerationReport):
            try container.encode("image_generation", forKey: AnyCodingKey("report_kind"));
            try imageGenerationReport.encodePayloadFields(into: &container);
        case .embeddings(let embeddingsReport):
            try container.encode("embeddings", forKey: AnyCodingKey("report_kind"));
            try embeddingsReport.encodePayloadFields(into: &container);
        }
    }
}
