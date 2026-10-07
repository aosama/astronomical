import Foundation;

/**
 * Validates and persists the optional `maximum_mlx_memory_gb` runtime setting
 * as an all-or-nothing byte transaction, porting
 * crates/config/src/maximum_mlx_memory.rs. Prepare builds and serializes the
 * candidate document without touching the source of truth; commit swaps it in
 * only while the file still holds the bytes the candidate was derived from.
 */
public enum MaximumMlxMemory {

    private static let BYTES_PER_DECIMAL_GIGABYTE: UInt64 = 1_000_000_000;
    private static let CONFIG_FILE_NAME: String = "config.json";

    /**
     * Converts a positive decimal SI gigabyte setting to exact bytes, rejecting
     * zero and values whose byte product does not fit in 64 bits.
     */
    public static func maximumMlxMemoryGbToBytes(_ maximumMlxMemoryGb: UInt64) throws -> UInt64 {
        if (maximumMlxMemoryGb == 0) {
            throw AstronomicalConfigError.invalidMaximumMlxMemoryGb(
                description: "maximum MLX memory must be positive"
            );
        }
        let productBytes: (partialValue: UInt64, overflow: Bool) = maximumMlxMemoryGb.multipliedReportingOverflow(
            by: MaximumMlxMemory.BYTES_PER_DECIMAL_GIGABYTE
        );
        if (productBytes.overflow) {
            throw AstronomicalConfigError.invalidMaximumMlxMemoryGb(
                description: "maximum MLX memory exceeds the byte range"
            );
        }
        return productBytes.partialValue;
    }

    /**
     * Builds a validated byte transaction without mutating the source of truth.
     */
    public static func prepareMaximumMlxMemoryGbUpdate(
        stateDirectory: FilePath,
        maximumMlxMemoryGb: UInt64?
    ) throws -> MaximumMlxMemoryConfigUpdate {
        if let requestedMaximumMlxMemoryGb: UInt64 = maximumMlxMemoryGb {
            let _ = try MaximumMlxMemory.maximumMlxMemoryGbToBytes(requestedMaximumMlxMemoryGb);
        }
        let configFilePath: FilePath = stateDirectory.appending(component: MaximumMlxMemory.CONFIG_FILE_NAME);
        let priorConfigBytes: Data? = try ConfigFileStore.readExistingConfigFileBytes(configFilePath: configFilePath);
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
            };
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
        let updatedUserConfigFile: UserConfigFile = MaximumMlxMemory.rebuildUserConfigFile(
            sourceUserConfigFile: candidateUserConfigFile,
            maximumMlxMemoryGb: maximumMlxMemoryGb
        );
        try ConfigDocumentValidation.validateV1UserConfigFile(updatedUserConfigFile);
        let candidateConfigBytes: Data = try ConfigDocumentSerialization.serializeConfigFileBytes(
            configFilePath: configFilePath,
            userConfigFile: updatedUserConfigFile
        );
        return MaximumMlxMemoryConfigUpdate(
            priorConfigBytes: priorConfigBytes,
            candidateConfigBytes: candidateConfigBytes
        );
    }

    /**
     * Commits a prepared update only while the document it was based on still
     * owns the file.
     */
    public static func commitMaximumMlxMemoryGbUpdate(
        stateDirectory: FilePath,
        configUpdate: MaximumMlxMemoryConfigUpdate
    ) throws -> Void {
        let configFilePath: FilePath = stateDirectory.appending(component: MaximumMlxMemory.CONFIG_FILE_NAME);
        let currentConfigBytes: Data? = try ConfigFileStore.readExistingConfigFileBytes(configFilePath: configFilePath);
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
            };
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
     * Persists the optional MLX memory override and returns its exact byte
     * transaction.
     */
    internal static func writeMaximumMlxMemoryGb(
        stateDirectory: FilePath,
        maximumMlxMemoryGb: UInt64?
    ) throws -> MaximumMlxMemoryConfigUpdate {
        let configUpdate: MaximumMlxMemoryConfigUpdate = try MaximumMlxMemory.prepareMaximumMlxMemoryGbUpdate(
            stateDirectory: stateDirectory,
            maximumMlxMemoryGb: maximumMlxMemoryGb
        );
        try MaximumMlxMemory.commitMaximumMlxMemoryGbUpdate(
            stateDirectory: stateDirectory,
            configUpdate: configUpdate
        );
        return configUpdate;
    }

    /**
     * Runtime and document sections are immutable value types, so applying the
     * override rebuilds both instead of mutating in place.
     */
    private static func rebuildUserConfigFile(
        sourceUserConfigFile: UserConfigFile,
        maximumMlxMemoryGb: UInt64?
    ) -> UserConfigFile {
        let updatedRuntimeConfigFile: RuntimeConfigFile = RuntimeConfigFile(
            modelDirectories: sourceUserConfigFile.runtime.modelDirectories,
            maximumMlxMemoryGb: maximumMlxMemoryGb,
            defaultModel: sourceUserConfigFile.runtime.defaultModel,
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
