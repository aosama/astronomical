import Foundation;

/// V1 document validation and retired-field stripping, porting the
/// parse_and_validate_v1 / validate_user_config_file /
/// strip_retired_speculative_prefill_config trio from
/// crates/config/src/config_file.rs plus UserConfigFile::validate from
/// crates/config/src/config_document.rs.
internal enum ConfigDocumentValidation {

    internal static let configSchemaReference: String = "./astronomical-config.schema.json";

    internal static func parseAndValidateV1Document(configFilePath: FilePath, configJson: Any, performanceAttributionEnabled: Bool = false) throws -> UserConfigFile {
        let passStart = ConfigPerformanceAttribution.startedPass(operationName: "parse_and_validate_v1", performanceAttributionEnabled: performanceAttributionEnabled);
        let parsedUserConfigFile: UserConfigFile;
        do {
            parsedUserConfigFile = try UserConfigFile.fromJsonObject(configJson);
        } catch let strictJsonError as StrictJsonError {
            ConfigPerformanceAttribution.finishedPass(operationName: "parse_and_validate_v1", passStart: passStart, passOutcome: "failure", performanceAttributionEnabled: performanceAttributionEnabled);
            throw AstronomicalConfigError.parseConfigFile(
                configFilePath: configFilePath,
                underlyingDescription: "field \(strictJsonError.fieldName): \(strictJsonError.problem)"
            );
        }
        try validateV1UserConfigFile(parsedUserConfigFile);
        ConfigPerformanceAttribution.finishedPass(operationName: "parse_and_validate_v1", passStart: passStart, passOutcome: "success", performanceAttributionEnabled: performanceAttributionEnabled);
        return parsedUserConfigFile;
    }

    internal static func validateV1UserConfigFile(_ userConfigFile: UserConfigFile) throws {
        if userConfigFile.schemaVersion != 1 {
            throw AstronomicalConfigError.unsupportedSchemaVersion(schemaVersion: userConfigFile.schemaVersion);
        }
        if userConfigFile.schemaReference != ConfigDocumentValidation.configSchemaReference {
            throw AstronomicalConfigError.invalidSchemaReference;
        }
        for modelDirectory in userConfigFile.runtime.modelDirectories {
            let modelDirectoryPath = FilePath(string: modelDirectory);
            if modelDirectoryPath.isAbsolute == false {
                throw AstronomicalConfigError.pathMustBeAbsolute(fieldName: "runtime.model_directories", configuredPath: modelDirectoryPath);
            }
        }
        if let maximumMlxMemoryGb: UInt64 = userConfigFile.runtime.maximumMlxMemoryGb {
            let _ = try MaximumMlxMemory.maximumMlxMemoryGbToBytes(maximumMlxMemoryGb);
        }
        if let promptCacheConfigFile: PromptCacheConfigFile = userConfigFile.promptCache {
            if let promptCacheMaximumSizeGb: UInt64 = promptCacheConfigFile.maximumSizeGb {
                let _ = try PromptCacheResolution.promptCacheMaximumSizeGbToBytes(promptCacheMaximumSizeGb);
            }
        }
        if let diagnosticsConfigFile: DiagnosticsConfigFile = userConfigFile.diagnostics {
            if let retainedLogFiles: Int32 = diagnosticsConfigFile.retainedLogFiles {
                if retainedLogFiles == 0 {
                    throw ConfigResolutionError.invalidRetainedLogFileCount;
                }
            }
        }
        // Per-model chunking validation lands with the resolved-model-config
        // slice; only the global resolution is validated here.
        let globalChunkingFile = userConfigFile.chunking ?? ChunkingConfigFile(
            fixedPromptProcessingChunkSizeTokens: nil,
            fixedSsdStreamingPromptProcessingChunkSizeTokens: nil,
            fullAttentionKeyValueGrowthTokens: nil,
            prefillGraphSubmissionLayerInterval: nil,
            experimentalSsdPagingPrefillGraphSubmissionLayerInterval: nil,
            experimentalSsdPagingGenerationGraphSubmissionLayerInterval: nil,
            promptCacheBlockTokens: nil,
            promptCacheCommonPrefixStrideBlocks: nil,
            experimentalDecodeStageAttributionEnabled: nil,
            experimentalQuantizedKvCacheEnabled: nil,
            experimentalFusedMoeDecodeEnabled: nil
        );
        let _ = try ChunkingConfig.resolve(configuredChunkingFile: globalChunkingFile);
    }

    /// Removes retired speculative-prefill settings from an untyped v1
    /// document and reports whether anything was removed. Every candidate
    /// location is examined even after an earlier removal succeeds, mirroring
    /// the non-short-circuit `|=` accumulation in the Rust implementation.
    internal static func stripRetiredSpeculativePrefillConfig(_ configJson: inout Dictionary<String, Any>) -> Bool {
        var removedAnyRetiredField: Bool = false;
        if removeObjectField(configJson: &configJson, fieldName: "speculative_prefill") {
            removedAnyRetiredField = true;
        }
        if removeNestedObjectField(configJson: &configJson, containerFieldName: "chunking", fieldName: "speculative_prefill_draft_forward_tokens") {
            removedAnyRetiredField = true;
        }
        if let presentModelsValue: Any = configJson["models"], var modelsObject = presentModelsValue as? Dictionary<String, Any> {
            for (key: modelId, value: modelValue) in modelsObject {
                guard var modelObject = modelValue as? Dictionary<String, Any> else {
                    continue;
                }
                var removedForModel: Bool = false;
                if removeNestedObjectField(configJson: &modelObject, containerFieldName: "chunking", fieldName: "speculative_prefill_draft_forward_tokens") {
                    removedForModel = true;
                }
                if removeNestedObjectField(configJson: &modelObject, containerFieldName: "acceleration", fieldName: "speculative_prefill") {
                    removedForModel = true;
                }
                // A retired member leaves an empty retired container behind;
                // strict parsing rejects unknown fields, so drop it too.
                if (modelObject["acceleration"] as? Dictionary<String, Any>)?.isEmpty == true {
                    modelObject.removeValue(forKey: "acceleration");
                    removedForModel = true;
                }
                if removedForModel {
                    modelsObject[modelId] = modelObject;
                    removedAnyRetiredField = true;
                }
            }
            configJson["models"] = modelsObject;
        }
        return removedAnyRetiredField;
    }

    private static func removeObjectField(configJson: inout Dictionary<String, Any>, fieldName: String) -> Bool {
        if configJson[fieldName] == nil {
            return false;
        }
        configJson.removeValue(forKey: fieldName);
        return true;
    }

    private static func removeNestedObjectField(configJson: inout Dictionary<String, Any>, containerFieldName: String, fieldName: String) -> Bool {
        guard let presentContainerValue: Any = configJson[containerFieldName], var containerObject = presentContainerValue as? Dictionary<String, Any> else {
            return false;
        }
        if containerObject[fieldName] == nil {
            return false;
        }
        containerObject.removeValue(forKey: fieldName);
        configJson[containerFieldName] = containerObject;
        return true;
    }
}
