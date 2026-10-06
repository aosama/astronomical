import Foundation;

/// Current sparse-expert weight residency exposed by the local worker.
public enum ExpertMemoryMode: Equatable {
    /// Every decoder layer has complete sparse experts resident.
    case resident;
    /// Some routed experts are retained while misses still page.
    case hybrid;
    /// No sparse expert payload is retained; every miss pages from storage.
    case paged;

    public var wireName: String {
        switch self {
        case .resident: return "resident";
        case .hybrid: return "hybrid";
        case .paged: return "paged";
        }
    }

    internal func wireValue() -> JsonWireValue {
        return .string(self.wireName);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> ExpertMemoryMode {
        let wireName = try JsonWireValue.extractString(wireValue);
        switch wireName {
        case "resident": return .resident;
        case "hybrid": return .hybrid;
        case "paged": return .paged;
        default: throw JsonWireProblem.malformedDocument(problem: "unknown variant `\(wireName)`, expected one of `resident`, `hybrid`, `paged`");
        }
    }
}

/// Final concrete sparse-expert topology copied from the model owner.
public struct WorkerExpertResidencySnapshot: Equatable {
    public let totalLayerCount: UInt32;
    public let residentExpertCount: UInt32;
    public let residentExpertPayloadBytes: UInt64;

    public init(totalLayerCount: UInt32, residentExpertCount: UInt32, residentExpertPayloadBytes: UInt64) {
        self.totalLayerCount = totalLayerCount;
        self.residentExpertCount = residentExpertCount;
        self.residentExpertPayloadBytes = residentExpertPayloadBytes;
    }

    internal static let wireFieldNames: Array<String> = ["total_layer_count", "resident_expert_count", "resident_expert_payload_bytes"];

    public func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "total_layer_count", value: .unsignedInteger(UInt64(self.totalLayerCount)));
        wireObject.appendEntry(key: "resident_expert_count", value: .unsignedInteger(UInt64(self.residentExpertCount)));
        wireObject.appendEntry(key: "resident_expert_payload_bytes", value: .unsignedInteger(self.residentExpertPayloadBytes));
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerExpertResidencySnapshot {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedSnapshot = WorkerExpertResidencySnapshot(
            totalLayerCount: try wireObject.decodeUInt32(fieldName: "total_layer_count"),
            residentExpertCount: try wireObject.decodeUInt32(fieldName: "resident_expert_count"),
            residentExpertPayloadBytes: try wireObject.decodeUInt64(fieldName: "resident_expert_payload_bytes"));
        try wireObject.rejectUnknownFields(allowedFieldNames: WorkerExpertResidencySnapshot.wireFieldNames);
        return parsedSnapshot;
    }
}
