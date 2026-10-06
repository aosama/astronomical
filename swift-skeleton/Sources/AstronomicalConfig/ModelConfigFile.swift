import Foundation;

/**
 * Per-model `models.<id>` entry of the v1 user configuration document.
 *
 * MIGRATION MARKER — deferred from this slice: model-ID hygiene (non-empty,
 * no surrounding whitespace or control characters) and the per-model chunking
 * merge against the global section land with resolved-model-config.
 */
internal struct ModelConfigFile: Equatable {
    internal let limits: ModelLimitsConfigFile?;
    internal let generationDefaults: GenerationDefaultsConfigFile?;
    internal let chunking: ChunkingConfigFile?;

    internal init(
        limits: ModelLimitsConfigFile?,
        generationDefaults: GenerationDefaultsConfigFile?,
        chunking: ChunkingConfigFile?
    ) {
        self.limits = limits;
        self.generationDefaults = generationDefaults;
        self.chunking = chunking;
    }

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> ModelConfigFile {
        try StrictJson.requireKnownKeys(
            object: jsonObject,
            knownKeys: ["limits", "generation_defaults", "chunking"],
            fieldName: "model"
        );
        let limitsObject: Dictionary<String, Any>? = try StrictJson.optionalObject(
            object: jsonObject,
            fieldName: "limits"
        );
        let generationDefaultsObject: Dictionary<String, Any>? = try StrictJson.optionalObject(
            object: jsonObject,
            fieldName: "generation_defaults"
        );
        let chunkingObject: Dictionary<String, Any>? = try StrictJson.optionalObject(
            object: jsonObject,
            fieldName: "chunking"
        );
        return ModelConfigFile(
            limits: try StrictJson.decodeOptional(limitsObject, decode: ModelLimitsConfigFile.fromJsonObject),
            generationDefaults: try StrictJson.decodeOptional(
                generationDefaultsObject,
                decode: GenerationDefaultsConfigFile.fromJsonObject
            ),
            chunking: try StrictJson.decodeOptional(chunkingObject, decode: ChunkingConfigFile.fromJsonObject)
        );
    }

    internal func toJsonObject() -> Any {
        var jsonObject: Dictionary<String, Any> = Dictionary<String, Any>();
        if let limits: ModelLimitsConfigFile = self.limits {
            jsonObject["limits"] = limits.toJsonObject();
        }
        if let generationDefaults: GenerationDefaultsConfigFile = self.generationDefaults {
            jsonObject["generation_defaults"] = generationDefaults.toJsonObject();
        }
        if let chunking: ChunkingConfigFile = self.chunking {
            jsonObject["chunking"] = chunking.toJsonObject();
        }
        return jsonObject;
    }
}
