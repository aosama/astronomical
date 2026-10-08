import Foundation;

/** `runtime` section of the v1 user configuration document. */
internal struct RuntimeConfigFile: Equatable {
    internal let modelDirectories: Array<String>;
    internal let maximumMlxMemoryGb: UInt64?;
    internal let defaultModel: String?;

    internal init(
        modelDirectories: Array<String>,
        maximumMlxMemoryGb: UInt64?,
        defaultModel: String?
    ) {
        self.modelDirectories = modelDirectories;
        self.maximumMlxMemoryGb = maximumMlxMemoryGb;
        self.defaultModel = defaultModel;
    }

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> RuntimeConfigFile {
        try StrictJson.requireKnownKeys(
            object: jsonObject,
            knownKeys: [
                "model_directories", "maximum_mlx_memory_gb", "default_model"
            ],
            fieldName: "runtime"
        );
        return RuntimeConfigFile(
            modelDirectories: try StrictJson.requiredStringArray(
                object: jsonObject,
                fieldName: "model_directories"
            ),
            maximumMlxMemoryGb: try StrictJson.optionalUnsignedInteger(
                object: jsonObject,
                fieldName: "maximum_mlx_memory_gb"
            ),
            defaultModel: try StrictJson.optionalString(object: jsonObject, fieldName: "default_model")
        );
    }

    internal func toJsonObject() -> Any {
        var jsonObject: Dictionary<String, Any> = Dictionary<String, Any>();
        jsonObject["model_directories"] = self.modelDirectories;
        if let maximumMlxMemoryGb: UInt64 = self.maximumMlxMemoryGb {
            jsonObject["maximum_mlx_memory_gb"] = maximumMlxMemoryGb;
        }
        if let defaultModel: String = self.defaultModel {
            jsonObject["default_model"] = defaultModel;
        }
        return jsonObject;
    }
}
