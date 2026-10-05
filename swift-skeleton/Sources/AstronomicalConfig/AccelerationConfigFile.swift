import Foundation;

/**
 * `models.<id>.acceleration` stanza of the v1 user configuration document.
 *
 * MIGRATION MARKER — deferred from this slice: the draft-depth 1...3 range
 * check lands with resolved-model-config.
 */
internal struct AccelerationConfigFile: Equatable {
    internal let mtp: MtpConfigFile?;

    internal init(mtp: MtpConfigFile?) {
        self.mtp = mtp;
    }

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> AccelerationConfigFile {
        try StrictJson.requireKnownKeys(object: jsonObject, knownKeys: ["mtp"], fieldName: "acceleration");
        return AccelerationConfigFile(
            mtp: try StrictJson.decodeOptional(
                try StrictJson.optionalObject(object: jsonObject, fieldName: "mtp"),
                decode: MtpConfigFile.fromJsonObject
            )
        );
    }

    internal func toJsonObject() -> Any {
        var jsonObject: Dictionary<String, Any> = Dictionary<String, Any>();
        if let mtp: MtpConfigFile = self.mtp {
            jsonObject["mtp"] = mtp.toJsonObject();
        }
        return jsonObject;
    }
}
