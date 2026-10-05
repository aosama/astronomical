import Foundation;

/**
 * `models.<id>.generation_defaults` stanza of the v1 user configuration
 * document.
 *
 * MIGRATION MARKER — deferred from this slice: the Rust unit also validates
 * temperature (0.0...2.0, representable in thousandths), top_p, and the
 * maximum-output-token range; those checks land with resolved-model-config.
 */
internal struct GenerationDefaultsConfigFile: Equatable {
    internal let temperature: Float?;
    internal let topP: Float?;
    internal let maximumOutputTokens: UInt32?;

    internal init(temperature: Float?, topP: Float?, maximumOutputTokens: UInt32?) {
        self.temperature = temperature;
        self.topP = topP;
        self.maximumOutputTokens = maximumOutputTokens;
    }

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> GenerationDefaultsConfigFile {
        try StrictJson.requireKnownKeys(
            object: jsonObject,
            knownKeys: ["temperature", "top_p", "maximum_output_tokens"],
            fieldName: "generation_defaults"
        );
        return GenerationDefaultsConfigFile(
            temperature: try StrictJson.optionalFloat(object: jsonObject, fieldName: "temperature"),
            topP: try StrictJson.optionalFloat(object: jsonObject, fieldName: "top_p"),
            maximumOutputTokens: try StrictJson.optionalUnsignedInteger(
                object: jsonObject,
                fieldName: "maximum_output_tokens"
            )
        );
    }

    internal func toJsonObject() -> Any {
        var jsonObject: Dictionary<String, Any> = Dictionary<String, Any>();
        if let temperature: Float = self.temperature {
            jsonObject["temperature"] = temperature;
        }
        if let topP: Float = self.topP {
            jsonObject["top_p"] = topP;
        }
        if let maximumOutputTokens: UInt32 = self.maximumOutputTokens {
            jsonObject["maximum_output_tokens"] = maximumOutputTokens;
        }
        return jsonObject;
    }
}
