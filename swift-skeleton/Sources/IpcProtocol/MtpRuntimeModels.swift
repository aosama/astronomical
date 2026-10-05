import Foundation;

/// Runtime execution state of native multi-token prediction (MTP).
public enum MtpRuntimeState: Equatable {
    /// The user preference is false.
    case disabled;
    /// Preference is true and the selected model has no compatible MTP inventory.
    case targetOnly;
    /// Preference is true, the head is compatible, and native MTP decode is available.
    case active;
    /// Preference is true but MTP inventory or initialization failed.
    case unavailable;

    internal var wireName: String {
        switch self {
        case .disabled: return "disabled";
        case .targetOnly: return "target_only";
        case .active: return "active";
        case .unavailable: return "unavailable";
        }
    }

    internal func wireValue() -> JsonWireValue {
        return .string(self.wireName);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> MtpRuntimeState {
        let wireName = try JsonWireValue.extractString(wireValue);
        switch wireName {
        case "disabled": return .disabled;
        case "target_only": return .targetOnly;
        case "active": return .active;
        case "unavailable": return .unavailable;
        default: throw JsonWireProblem.malformedDocument(problem: "unknown variant `\(wireName)`, expected one of `disabled`, `target_only`, `active`, `unavailable`");
        }
    }
}

/// Bounded explanation when MTP depth resolution changes or cautions user intent.
public enum MtpDepthResolutionReason: Equatable, CustomStringConvertible, Sendable {
    case configuredDepthClampedToArtifactMaximum;
    case configuredDepthExceedsAutomaticGuidance;

    internal var wireName: String {
        switch self {
        case .configuredDepthClampedToArtifactMaximum: return "configured_depth_clamped_to_artifact_maximum";
        case .configuredDepthExceedsAutomaticGuidance: return "configured_depth_exceeds_automatic_guidance";
        }
    }

    public var description: String {
        switch self {
        case .configuredDepthClampedToArtifactMaximum: return "configured MTP draft depth was clamped to the declared artifact maximum";
        case .configuredDepthExceedsAutomaticGuidance: return "configured MTP draft depth exceeds the automatic depth-one guidance";
        }
    }

    internal func wireValue() -> JsonWireValue {
        return .string(self.wireName);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> MtpDepthResolutionReason {
        let wireName = try JsonWireValue.extractString(wireValue);
        switch wireName {
        case "configured_depth_clamped_to_artifact_maximum": return .configuredDepthClampedToArtifactMaximum;
        case "configured_depth_exceeds_automatic_guidance": return .configuredDepthExceedsAutomaticGuidance;
        default: throw JsonWireProblem.malformedDocument(problem: "unknown variant `\(wireName)`, expected one of `configured_depth_clamped_to_artifact_maximum`, `configured_depth_exceeds_automatic_guidance`");
        }
    }
}

/// Fixed MTP depth metadata resolved by the loaded model and active executor.
public struct MtpDepthStatus: Equatable, Sendable {
    public let configuredDraftDepth: UInt8?;
    public let artifactMaximumDraftDepth: UInt8?;
    public let artifactDefaultDraftDepth: UInt8?;
    public let resolvedRequestedDraftDepth: UInt8?;
    public let cappedDraftDepth: UInt8?;
    public let effectiveExecutionDraftDepth: UInt8?;
    public let resolutionReason: MtpDepthResolutionReason?;

    public init(
        configuredDraftDepth: UInt8?,
        artifactMaximumDraftDepth: UInt8?,
        artifactDefaultDraftDepth: UInt8?,
        resolvedRequestedDraftDepth: UInt8?,
        cappedDraftDepth: UInt8?,
        effectiveExecutionDraftDepth: UInt8?,
        resolutionReason: MtpDepthResolutionReason?
    ) {
        self.configuredDraftDepth = configuredDraftDepth;
        self.artifactMaximumDraftDepth = artifactMaximumDraftDepth;
        self.artifactDefaultDraftDepth = artifactDefaultDraftDepth;
        self.resolvedRequestedDraftDepth = resolvedRequestedDraftDepth;
        self.cappedDraftDepth = cappedDraftDepth;
        self.effectiveExecutionDraftDepth = effectiveExecutionDraftDepth;
        self.resolutionReason = resolutionReason;
    }

    public static let empty: MtpDepthStatus = MtpDepthStatus(
        configuredDraftDepth: nil,
        artifactMaximumDraftDepth: nil,
        artifactDefaultDraftDepth: nil,
        resolvedRequestedDraftDepth: nil,
        cappedDraftDepth: nil,
        effectiveExecutionDraftDepth: nil,
        resolutionReason: nil);

    internal static let wireFieldNames: Array<String> = [
        "configured_draft_depth", "artifact_maximum_draft_depth", "artifact_default_draft_depth",
        "resolved_requested_draft_depth", "capped_draft_depth", "effective_execution_draft_depth",
        "resolution_reason",
    ];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "configured_draft_depth", value: MtpDepthStatus.optionalUInt8WireValue(self.configuredDraftDepth));
        wireObject.appendEntry(key: "artifact_maximum_draft_depth", value: MtpDepthStatus.optionalUInt8WireValue(self.artifactMaximumDraftDepth));
        wireObject.appendEntry(key: "artifact_default_draft_depth", value: MtpDepthStatus.optionalUInt8WireValue(self.artifactDefaultDraftDepth));
        wireObject.appendEntry(key: "resolved_requested_draft_depth", value: MtpDepthStatus.optionalUInt8WireValue(self.resolvedRequestedDraftDepth));
        wireObject.appendEntry(key: "capped_draft_depth", value: MtpDepthStatus.optionalUInt8WireValue(self.cappedDraftDepth));
        wireObject.appendEntry(key: "effective_execution_draft_depth", value: MtpDepthStatus.optionalUInt8WireValue(self.effectiveExecutionDraftDepth));
        wireObject.appendEntry(key: "resolution_reason", value: MtpDepthStatus.optionalWireValue(self.resolutionReason, mappedWireValue: { (reason: MtpDepthResolutionReason) -> JsonWireValue in reason.wireValue() }));
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> MtpDepthStatus {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedStatus = MtpDepthStatus(
            configuredDraftDepth: try wireObject.decodeOptionalUInt8(fieldName: "configured_draft_depth"),
            artifactMaximumDraftDepth: try wireObject.decodeOptionalUInt8(fieldName: "artifact_maximum_draft_depth"),
            artifactDefaultDraftDepth: try wireObject.decodeOptionalUInt8(fieldName: "artifact_default_draft_depth"),
            resolvedRequestedDraftDepth: try wireObject.decodeOptionalUInt8(fieldName: "resolved_requested_draft_depth"),
            cappedDraftDepth: try wireObject.decodeOptionalUInt8(fieldName: "capped_draft_depth"),
            effectiveExecutionDraftDepth: try wireObject.decodeOptionalUInt8(fieldName: "effective_execution_draft_depth"),
            resolutionReason: try MtpDepthStatus.decodeOptionalMapped(wireObject, fieldName: "resolution_reason", { (reasonWireValue: JsonWireValue) throws -> MtpDepthResolutionReason in try MtpDepthResolutionReason.fromWireValue(reasonWireValue) }));
        try wireObject.rejectUnknownFields(allowedFieldNames: MtpDepthStatus.wireFieldNames);
        return parsedStatus;
    }

    private static func optionalUInt8WireValue(_ rawValue: UInt8?) -> JsonWireValue {
        guard let unwrappedValue = rawValue else {
            return .null;
        }
        return .unsignedInteger(UInt64(unwrappedValue));
    }

    private static func optionalWireValue<T>(_ optionalValue: T?, mappedWireValue: (T) -> JsonWireValue) -> JsonWireValue {
        guard let unwrappedValue = optionalValue else {
            return .null;
        }
        return mappedWireValue(unwrappedValue);
    }

    private static func decodeOptionalMapped<T>(_ wireObject: JsonWireObject, fieldName propertyName: String, _ mappedValue: (JsonWireValue) throws -> T) throws -> T? {
        let fieldValue = try wireObject.requireObjectValue(fieldName: propertyName);
        if fieldValue.isNull {
            return nil;
        }
        return try mappedValue(fieldValue);
    }
}
