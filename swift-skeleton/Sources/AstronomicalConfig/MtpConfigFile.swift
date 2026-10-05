import Foundation;

/**
 * `models.<id>.acceleration.mtp` multi-token-prediction stanza of the v1
 * user configuration document.
 */
internal struct MtpConfigFile: Equatable {
    internal let enabled: Bool?;
    internal let draftDepth: UInt8?;

    internal init(enabled: Bool?, draftDepth: UInt8?) {
        self.enabled = enabled;
        self.draftDepth = draftDepth;
    }

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> MtpConfigFile {
        try StrictJson.requireKnownKeys(
            object: jsonObject,
            knownKeys: ["enabled", "draft_depth"],
            fieldName: "acceleration.mtp"
        );
        return MtpConfigFile(
            enabled: try StrictJson.optionalBoolean(object: jsonObject, fieldName: "enabled"),
            draftDepth: try StrictJson.optionalUnsignedInteger(object: jsonObject, fieldName: "draft_depth")
        );
    }

    internal func toJsonObject() -> Any {
        var jsonObject: Dictionary<String, Any> = Dictionary<String, Any>();
        if let enabled: Bool = self.enabled {
            jsonObject["enabled"] = enabled;
        }
        if let draftDepth: UInt8 = self.draftDepth {
            jsonObject["draft_depth"] = draftDepth;
        }
        return jsonObject;
    }
}
