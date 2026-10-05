import Foundation;

/** `prompt_cache` section of the v1 user configuration document. */
internal struct PromptCacheConfigFile: Equatable {
    internal let enabled: Bool?;
    internal let maximumSizeGb: UInt64?;

    internal init(enabled: Bool?, maximumSizeGb: UInt64?) {
        self.enabled = enabled;
        self.maximumSizeGb = maximumSizeGb;
    }

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> PromptCacheConfigFile {
        try StrictJson.requireKnownKeys(
            object: jsonObject,
            knownKeys: ["enabled", "maximum_size_gb"],
            fieldName: "prompt_cache"
        );
        return PromptCacheConfigFile(
            enabled: try StrictJson.optionalBoolean(object: jsonObject, fieldName: "enabled"),
            maximumSizeGb: try StrictJson.optionalUnsignedInteger(
                object: jsonObject,
                fieldName: "maximum_size_gb"
            )
        );
    }

    internal func toJsonObject() -> Any {
        var jsonObject: Dictionary<String, Any> = Dictionary<String, Any>();
        if let enabled: Bool = self.enabled {
            jsonObject["enabled"] = enabled;
        }
        if let maximumSizeGb: UInt64 = self.maximumSizeGb {
            jsonObject["maximum_size_gb"] = maximumSizeGb;
        }
        return jsonObject;
    }
}
