import Foundation;

/// Lifecycle point at which the worker observed MLX allocator memory.
public enum MlxMemorySnapshotSource: Equatable {
    /// The model became resident in the worker.
    case modelLoaded;
    /// A prompt-processing chunk completed.
    case prefill;
    /// One-token-ahead decode work was submitted to MLX.
    case decodeSubmitted;
    /// Request state and reclaimable allocator memory were released.
    case finalized;
    /// The ready worker was idle when the supervisor requested a refresh.
    case idlePoll;
    /// A live MLX memory-ceiling control operation completed.
    case memoryLimitAdjusted;
    /// One image render step completed while generation was still active.
    case imageGenerationStep;

    internal var wireName: String {
        switch self {
        case .modelLoaded: return "model_loaded";
        case .prefill: return "prefill";
        case .decodeSubmitted: return "decode_submitted";
        case .finalized: return "finalized";
        case .idlePoll: return "idle_poll";
        case .memoryLimitAdjusted: return "memory_limit_adjusted";
        case .imageGenerationStep: return "image_generation_step";
        }
    }

    internal func wireValue() -> JsonWireValue {
        return .string(self.wireName);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> MlxMemorySnapshotSource {
        let wireName = try JsonWireValue.extractString(wireValue);
        switch wireName {
        case "model_loaded": return .modelLoaded;
        case "prefill": return .prefill;
        case "decode_submitted": return .decodeSubmitted;
        case "finalized": return .finalized;
        case "idle_poll": return .idlePoll;
        case "memory_limit_adjusted": return .memoryLimitAdjusted;
        case "image_generation_step": return .imageGenerationStep;
        default: throw JsonWireProblem.malformedDocument(problem: "unknown variant `\(wireName)`, expected one of `model_loaded`, `prefill`, `decode_submitted`, `finalized`, `idle_poll`, `memory_limit_adjusted`, `image_generation_step`");
        }
    }
}

/// One utilization decomposition published beside its memory snapshot.
public struct WorkerMemoryCeilingUtilizationSnapshot: Equatable {
    public let unusedHeadroomBytes: UInt64;
    public let reservedModelCoreSlackBytes: UInt64;
    public let reservedContextGrowthBytes: UInt64;
    public let reservedActivationAndWorkspaceBytes: UInt64;
    public let unseatedExpertEntitlementBytes: UInt64;
    public let unexplainedHeadroomBytes: UInt64;
    public let ownerOverrunBytes: UInt64;

    public init(
        unusedHeadroomBytes: UInt64,
        reservedModelCoreSlackBytes: UInt64,
        reservedContextGrowthBytes: UInt64,
        reservedActivationAndWorkspaceBytes: UInt64,
        unseatedExpertEntitlementBytes: UInt64,
        unexplainedHeadroomBytes: UInt64,
        ownerOverrunBytes: UInt64
    ) {
        self.unusedHeadroomBytes = unusedHeadroomBytes;
        self.reservedModelCoreSlackBytes = reservedModelCoreSlackBytes;
        self.reservedContextGrowthBytes = reservedContextGrowthBytes;
        self.reservedActivationAndWorkspaceBytes = reservedActivationAndWorkspaceBytes;
        self.unseatedExpertEntitlementBytes = unseatedExpertEntitlementBytes;
        self.unexplainedHeadroomBytes = unexplainedHeadroomBytes;
        self.ownerOverrunBytes = ownerOverrunBytes;
    }

    internal static let wireFieldNames: Array<String> = [
        "unused_headroom_bytes", "reserved_model_core_slack_bytes", "reserved_context_growth_bytes",
        "reserved_activation_and_workspace_bytes", "unseated_expert_entitlement_bytes",
        "unexplained_headroom_bytes", "owner_overrun_bytes",
    ];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "unused_headroom_bytes", value: .unsignedInteger(self.unusedHeadroomBytes));
        wireObject.appendEntry(key: "reserved_model_core_slack_bytes", value: .unsignedInteger(self.reservedModelCoreSlackBytes));
        wireObject.appendEntry(key: "reserved_context_growth_bytes", value: .unsignedInteger(self.reservedContextGrowthBytes));
        wireObject.appendEntry(key: "reserved_activation_and_workspace_bytes", value: .unsignedInteger(self.reservedActivationAndWorkspaceBytes));
        wireObject.appendEntry(key: "unseated_expert_entitlement_bytes", value: .unsignedInteger(self.unseatedExpertEntitlementBytes));
        wireObject.appendEntry(key: "unexplained_headroom_bytes", value: .unsignedInteger(self.unexplainedHeadroomBytes));
        wireObject.appendEntry(key: "owner_overrun_bytes", value: .unsignedInteger(self.ownerOverrunBytes));
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerMemoryCeilingUtilizationSnapshot {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedSnapshot = WorkerMemoryCeilingUtilizationSnapshot(
            unusedHeadroomBytes: try wireObject.decodeUInt64(fieldName: "unused_headroom_bytes"),
            reservedModelCoreSlackBytes: try wireObject.decodeUInt64(fieldName: "reserved_model_core_slack_bytes"),
            reservedContextGrowthBytes: try wireObject.decodeUInt64(fieldName: "reserved_context_growth_bytes"),
            reservedActivationAndWorkspaceBytes: try wireObject.decodeUInt64(fieldName: "reserved_activation_and_workspace_bytes"),
            unseatedExpertEntitlementBytes: try wireObject.decodeUInt64(fieldName: "unseated_expert_entitlement_bytes"),
            unexplainedHeadroomBytes: try wireObject.decodeUInt64(fieldName: "unexplained_headroom_bytes"),
            ownerOverrunBytes: try wireObject.decodeUInt64(fieldName: "owner_overrun_bytes"));
        try wireObject.rejectUnknownFields(allowedFieldNames: WorkerMemoryCeilingUtilizationSnapshot.wireFieldNames);
        return parsedSnapshot;
    }
}

/// One worker-owned MLX allocator observation reconciled into user-visible owners.
public struct WorkerMlxMemorySnapshot: Equatable {
    public let source: MlxMemorySnapshotSource;
    public let activeMemoryBytes: UInt64;
    public let allocatorCacheMemoryBytes: UInt64;
    public let peakMemoryBytes: UInt64;
    public let expertPayloadBytes: UInt64;
    public let modelCorePayloadBytes: UInt64;
    public let contextStatePayloadBytes: UInt64;
    /// Reason-tagged split of the ceiling's unused headroom at this instant
    /// (issue #510). `nil` from engines without a composed RAM budget. The
    /// wire field is `#[serde(default)]`: absent decodes as nil while still
    /// serializing as an explicit null.
    public let memoryCeilingUtilization: WorkerMemoryCeilingUtilizationSnapshot?;

    public init(
        source: MlxMemorySnapshotSource,
        activeMemoryBytes: UInt64,
        allocatorCacheMemoryBytes: UInt64,
        peakMemoryBytes: UInt64,
        expertPayloadBytes: UInt64,
        modelCorePayloadBytes: UInt64,
        contextStatePayloadBytes: UInt64,
        memoryCeilingUtilization: WorkerMemoryCeilingUtilizationSnapshot?
    ) {
        self.source = source;
        self.activeMemoryBytes = activeMemoryBytes;
        self.allocatorCacheMemoryBytes = allocatorCacheMemoryBytes;
        self.peakMemoryBytes = peakMemoryBytes;
        self.expertPayloadBytes = expertPayloadBytes;
        self.modelCorePayloadBytes = modelCorePayloadBytes;
        self.contextStatePayloadBytes = contextStatePayloadBytes;
        self.memoryCeilingUtilization = memoryCeilingUtilization;
    }

    internal static let wireFieldNames: Array<String> = [
        "source", "active_memory_bytes", "allocator_cache_memory_bytes", "peak_memory_bytes",
        "expert_payload_bytes", "model_core_payload_bytes", "context_state_payload_bytes",
        "memory_ceiling_utilization",
    ];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "source", value: self.source.wireValue());
        wireObject.appendEntry(key: "active_memory_bytes", value: .unsignedInteger(self.activeMemoryBytes));
        wireObject.appendEntry(key: "allocator_cache_memory_bytes", value: .unsignedInteger(self.allocatorCacheMemoryBytes));
        wireObject.appendEntry(key: "peak_memory_bytes", value: .unsignedInteger(self.peakMemoryBytes));
        wireObject.appendEntry(key: "expert_payload_bytes", value: .unsignedInteger(self.expertPayloadBytes));
        wireObject.appendEntry(key: "model_core_payload_bytes", value: .unsignedInteger(self.modelCorePayloadBytes));
        wireObject.appendEntry(key: "context_state_payload_bytes", value: .unsignedInteger(self.contextStatePayloadBytes));
        if let ceilingUtilization = self.memoryCeilingUtilization {
            wireObject.appendEntry(key: "memory_ceiling_utilization", value: ceilingUtilization.wireValue());
        } else {
            wireObject.appendEntry(key: "memory_ceiling_utilization", value: .null);
        }
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerMlxMemorySnapshot {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        var parsedCeilingUtilization: WorkerMemoryCeilingUtilizationSnapshot? = nil;
        if let ceilingWireValue = wireObject.value(forKey: "memory_ceiling_utilization"), ceilingWireValue.isNull == false {
            parsedCeilingUtilization = try WorkerMemoryCeilingUtilizationSnapshot.fromWireValue(ceilingWireValue);
        }
        let parsedSnapshot = WorkerMlxMemorySnapshot(
            source: try MlxMemorySnapshotSource.fromWireValue(try wireObject.requireObjectValue(fieldName: "source")),
            activeMemoryBytes: try wireObject.decodeUInt64(fieldName: "active_memory_bytes"),
            allocatorCacheMemoryBytes: try wireObject.decodeUInt64(fieldName: "allocator_cache_memory_bytes"),
            peakMemoryBytes: try wireObject.decodeUInt64(fieldName: "peak_memory_bytes"),
            expertPayloadBytes: try wireObject.decodeUInt64(fieldName: "expert_payload_bytes"),
            modelCorePayloadBytes: try wireObject.decodeUInt64(fieldName: "model_core_payload_bytes"),
            contextStatePayloadBytes: try wireObject.decodeUInt64(fieldName: "context_state_payload_bytes"),
            memoryCeilingUtilization: parsedCeilingUtilization);
        try wireObject.rejectUnknownFields(allowedFieldNames: WorkerMlxMemorySnapshot.wireFieldNames);
        return parsedSnapshot;
    }
}
