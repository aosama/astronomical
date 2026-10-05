import Foundation;

/// Per-request target-model work that was eligible for and restored from reusable prompt state.
public struct WorkerPromptWorkReuse: Equatable {
    public let targetEligibleTokenCount: UInt64;
    public let targetRestoredTokenCount: UInt64;

    public init(targetEligibleTokenCount: UInt64, targetRestoredTokenCount: UInt64) {
        self.targetEligibleTokenCount = targetEligibleTokenCount;
        self.targetRestoredTokenCount = targetRestoredTokenCount;
    }

    internal static let wireFieldNames: Array<String> = ["target_eligible_token_count", "target_restored_token_count"];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "target_eligible_token_count", value: .unsignedInteger(self.targetEligibleTokenCount));
        wireObject.appendEntry(key: "target_restored_token_count", value: .unsignedInteger(self.targetRestoredTokenCount));
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerPromptWorkReuse {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedReuse = WorkerPromptWorkReuse(
            targetEligibleTokenCount: try wireObject.decodeUInt64(fieldName: "target_eligible_token_count"),
            targetRestoredTokenCount: try wireObject.decodeUInt64(fieldName: "target_restored_token_count"));
        try wireObject.rejectUnknownFields(allowedFieldNames: WorkerPromptWorkReuse.wireFieldNames);
        return parsedReuse;
    }
}

/// Model currently processing the active prompt phase.
public enum WorkerPromptProcessingPhase: Equatable {
    /// The target model is processing protected or selected prompt work.
    case target;

    internal var wireName: String {
        switch self {
        case .target: return "target";
        }
    }

    internal func wireValue() -> JsonWireValue {
        return .string(self.wireName);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerPromptProcessingPhase {
        let wireName = try JsonWireValue.extractString(wireValue);
        switch wireName {
        case "target": return .target;
        default: throw JsonWireProblem.malformedDocument(problem: "unknown variant `\(wireName)`, expected `target`");
        }
    }
}
