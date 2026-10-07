import Foundation

/**
 * Persists the user's default model id for one-shot CLI verbs, porting
 * crates/config/src/default_model.rs. The built-in fallback is shared by the
 * daemon's effective-default resolution and the CLI's request resolution so
 * the two can never drift.
 */
public enum DefaultModel {

    /** Built-in chat model used when the user has configured no default. */
    public static let builtinDefaultModelId: String = "Qwen3.5-2B-4bit";

    private static let CONFIG_FILE_NAME: String = "config.json";

    /**
     * Builds a validated byte transaction without mutating the source of
     * truth. `nil` clears the persisted default.
     */
    public static func prepareDefaultModelUpdate(
        stateDirectory: FilePath,
        defaultModel: String?
    ) throws -> DefaultModelConfigUpdate {
        if let requestedDefaultModel: String = defaultModel {
            try DefaultModel.validateDefaultModelId(requestedDefaultModel);
        }
        let configFilePath: FilePath = stateDirectory.appending(
            component: DefaultModel.CONFIG_FILE_NAME);
        let priorConfigBytes: Data? = try ConfigFileStore.readExistingConfigFileBytes(
            configFilePath: configFilePath);
        let candidateUserConfigFile: UserConfigFile;
        if let existingConfigBytes: Data = priorConfigBytes {
            let parsedConfigJsonValue: Any = try DuplicateKeyJson.parseJsonRejectingDuplicates(
                configFilePath: configFilePath,
                configBytes: existingConfigBytes
            );
            guard let configJson: Dictionary<String, Any> = parsedConfigJsonValue as? Dictionary<String, Any> else {
                throw AstronomicalConfigError.parseConfigFile(
                    configFilePath: configFilePath,
                    underlyingDescription: "top-level value must be a JSON object"
                );
            }
            if (configJson["schema_version"] == nil) {
                candidateUserConfigFile = try LegacyConfigMigration.prepareLegacyConfigMigration(
                    configFilePath: configFilePath,
                    legacyJson: configJson
                );
            } else {
                candidateUserConfigFile = try ConfigDocumentValidation.parseAndValidateV1Document(
                    configFilePath: configFilePath,
                    configJson: configJson
                );
            }
        } else {
            candidateUserConfigFile = UserConfigFile.minimal();
        }
        let updatedUserConfigFile: UserConfigFile = DefaultModel.rebuildUserConfigFile(
            sourceUserConfigFile: candidateUserConfigFile,
            defaultModel: defaultModel
        );
        try ConfigDocumentValidation.validateV1UserConfigFile(updatedUserConfigFile);
        let candidateConfigBytes: Data = try ConfigDocumentSerialization.serializeConfigFileBytes(
            configFilePath: configFilePath,
            userConfigFile: updatedUserConfigFile
        );
        return DefaultModelConfigUpdate(
            priorConfigBytes: priorConfigBytes,
            candidateConfigBytes: candidateConfigBytes
        );
    }

    /**
     * Commits a prepared update only while the document it was based on still
     * owns the file.
     */
    public static func commitDefaultModelUpdate(
        stateDirectory: FilePath,
        configUpdate: DefaultModelConfigUpdate
    ) throws -> Void {
        let configFilePath: FilePath = stateDirectory.appending(
            component: DefaultModel.CONFIG_FILE_NAME);
        let currentConfigBytes: Data? = try ConfigFileStore.readExistingConfigFileBytes(
            configFilePath: configFilePath);
        if (currentConfigBytes != configUpdate.priorConfigBytes) {
            throw AstronomicalConfigError.configChangedDuringUpdate;
        }
        if let priorConfigBytes: Data = configUpdate.priorConfigBytes {
            let parsedPriorJsonValue: Any = try DuplicateKeyJson.parseJsonRejectingDuplicates(
                configFilePath: configFilePath,
                configBytes: priorConfigBytes
            );
            guard let priorConfigJson: Dictionary<String, Any> = parsedPriorJsonValue as? Dictionary<String, Any> else {
                throw AstronomicalConfigError.parseConfigFile(
                    configFilePath: configFilePath,
                    underlyingDescription: "top-level value must be a JSON object"
                );
            }
            if (priorConfigJson["schema_version"] == nil) {
                try LegacyConfigMigration.preserveLegacyConfigBackup(
                    configFilePath: configFilePath,
                    legacyConfigBytes: priorConfigBytes
                );
            }
        }
        // The schema precedes the document so every committed config remains locally inspectable.
        try ConfigFileStore.writeAdjacentSchema(configFilePath: configFilePath);
        try ConfigFileStore.writeConfigFileBytesAtomically(configFilePath, bytes: configUpdate.candidateConfigBytes);
    }

    /**
     * Persists the default model id (or clears it) and returns its exact byte
     * transaction.
     */
    public static func writeDefaultModel(
        stateDirectory: FilePath,
        defaultModel: String?
    ) throws -> DefaultModelConfigUpdate {
        let configUpdate: DefaultModelConfigUpdate = try DefaultModel.prepareDefaultModelUpdate(
            stateDirectory: stateDirectory,
            defaultModel: defaultModel
        );
        try DefaultModel.commitDefaultModelUpdate(
            stateDirectory: stateDirectory,
            configUpdate: configUpdate
        );
        return configUpdate;
    }

    private static func validateDefaultModelId(_ modelId: String) throws -> Void {
        let trimmedModelId: String = modelId.trimmingCharacters(
            in: CharacterSet.whitespacesAndNewlines
        );
        if (modelId.isEmpty || trimmedModelId != modelId) {
            throw AstronomicalConfigError.invalidDefaultModel(
                description: "default model id must be a non-empty model id without surrounding whitespace"
            );
        }
    }

    /**
     * Runtime and document sections are immutable value types, so applying the
     * override rebuilds both instead of mutating in place.
     */
    private static func rebuildUserConfigFile(
        sourceUserConfigFile: UserConfigFile,
        defaultModel: String?
    ) -> UserConfigFile {
        let updatedRuntimeConfigFile: RuntimeConfigFile = RuntimeConfigFile(
            modelDirectories: sourceUserConfigFile.runtime.modelDirectories,
            maximumMlxMemoryGb: sourceUserConfigFile.runtime.maximumMlxMemoryGb,
            defaultModel: defaultModel,
            experimentalQwenThinkingChannelSeedEnabled: sourceUserConfigFile.runtime.experimentalQwenThinkingChannelSeedEnabled
        );
        return UserConfigFile(
            schemaReference: sourceUserConfigFile.schemaReference,
            schemaVersion: sourceUserConfigFile.schemaVersion,
            runtime: updatedRuntimeConfigFile,
            promptCache: sourceUserConfigFile.promptCache,
            chunking: sourceUserConfigFile.chunking,
            models: sourceUserConfigFile.models,
            diagnostics: sourceUserConfigFile.diagnostics
        );
    }
}

/** Exact before/after bytes for one atomic default-model configuration mutation. */
public struct DefaultModelConfigUpdate: Equatable {

    public let priorConfigBytes: Data?;
    public let candidateConfigBytes: Data;

    public init(priorConfigBytes: Data?, candidateConfigBytes: Data) {
        self.priorConfigBytes = priorConfigBytes;
        self.candidateConfigBytes = candidateConfigBytes;
    }
}
