import Foundation;

/** `models.<id>.limits` stanza of the v1 user configuration document. */
internal struct ModelLimitsConfigFile: Equatable {
    internal let maximumContextTokens: UInt32?;

    internal init(maximumContextTokens: UInt32?) {
        self.maximumContextTokens = maximumContextTokens;
    }

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> ModelLimitsConfigFile {
        try StrictJson.requireKnownKeys(object: jsonObject, knownKeys: ["maximum_context_tokens"], fieldName: "limits");
        return ModelLimitsConfigFile(
            maximumContextTokens: try StrictJson.optionalUnsignedInteger(
                object: jsonObject,
                fieldName: "maximum_context_tokens"
            )
        );
    }

    internal func toJsonObject() -> Any {
        var jsonObject: Dictionary<String, Any> = Dictionary<String, Any>();
        if let maximumContextTokens: UInt32 = self.maximumContextTokens {
            jsonObject["maximum_context_tokens"] = maximumContextTokens;
        }
        return jsonObject;
    }
}
