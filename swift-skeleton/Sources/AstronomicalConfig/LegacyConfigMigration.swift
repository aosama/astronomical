import Foundation;
import Darwin;

/// Closure seam for model discovery during legacy migration. The Rust code
/// calls crate::discover_models directly; the Swift ModelDiscovery slice is
/// still landing, so the coordinator wires this seam to the discovery scans
/// and maps each scan to its discovered model ids.
internal typealias LegacyModelIdDiscovery = (Array<String>) throws -> Array<String>;

/// One-time migration of an unversioned configuration document into v1,
/// porting crates/config/src/legacy_config_migration.rs. The migration is a
/// byte-transaction: backup first, schema second, atomic commit last.
internal enum LegacyConfigMigration {

    internal static let legacyConfigBackupFileName: String = "config.legacy-v0.json";

    internal static func migrateLegacyConfig(configFilePath: FilePath, legacyConfigBytes: Data, legacyJson: Dictionary<String, Any>, discoverModelIds: LegacyModelIdDiscovery? = nil, performanceAttributionEnabled: Bool = false) throws -> UserConfigFile {
        let migrationStart = ConfigPerformanceAttribution.startedPass(operationName: "legacy-config-migration", performanceAttributionEnabled: performanceAttributionEnabled);
        do {
            let migratedUserConfigFile = try executeLegacyConfigMigration(configFilePath: configFilePath, legacyConfigBytes: legacyConfigBytes, legacyJson: legacyJson, discoverModelIds: discoverModelIds, performanceAttributionEnabled: performanceAttributionEnabled);
            ConfigPerformanceAttribution.finishedPass(operationName: "legacy-config-migration", passStart: migrationStart, passOutcome: "success", performanceAttributionEnabled: performanceAttributionEnabled);
            return migratedUserConfigFile;
        } catch let migrationError {
            ConfigPerformanceAttribution.finishedPass(operationName: "legacy-config-migration", passStart: migrationStart, passOutcome: "failed", performanceAttributionEnabled: performanceAttributionEnabled);
            throw migrationError;
        }
    }

    private static func executeLegacyConfigMigration(configFilePath: FilePath, legacyConfigBytes: Data, legacyJson: Dictionary<String, Any>, discoverModelIds: LegacyModelIdDiscovery?, performanceAttributionEnabled: Bool) throws -> UserConfigFile {
        let validatedConfig = try prepareLegacyConfigMigration(configFilePath: configFilePath, legacyJson: legacyJson, discoverModelIds: discoverModelIds, performanceAttributionEnabled: performanceAttributionEnabled);
        let migratedBytes = try ConfigDocumentSerialization.serializeConfigFileBytes(configFilePath: configFilePath, userConfigFile: validatedConfig);
        // A successful one-way migration must retain recovery material before its commit point.
        try preserveLegacyConfigBackup(configFilePath: configFilePath, legacyConfigBytes: legacyConfigBytes);
        try ConfigFileStore.writeAdjacentSchema(configFilePath: configFilePath);
        try ConfigFileStore.writeConfigFileBytesAtomically(configFilePath, bytes: migratedBytes);
        return validatedConfig;
    }

    /// Resolves legacy intent without writing so compare-and-commit callers retain ownership.
    internal static func prepareLegacyConfigMigration(configFilePath: FilePath, legacyJson: Dictionary<String, Any>, discoverModelIds: LegacyModelIdDiscovery? = nil, performanceAttributionEnabled: Bool = false) throws -> UserConfigFile {
        let passStart = ConfigPerformanceAttribution.startedPass(operationName: "prepare-legacy-config-migration", performanceAttributionEnabled: performanceAttributionEnabled);
        var workingLegacyJson = legacyJson;
        let _ = ConfigDocumentValidation.stripRetiredSpeculativePrefillConfig(&workingLegacyJson);
        let legacyConfig: LegacyConfigFile;
        do {
            legacyConfig = try LegacyConfigFile.fromJsonObject(workingLegacyJson);
        } catch let strictJsonError as StrictJsonError {
            ConfigPerformanceAttribution.finishedPass(operationName: "prepare-legacy-config-migration", passStart: passStart, passOutcome: "failure", performanceAttributionEnabled: performanceAttributionEnabled);
            throw AstronomicalConfigError.parseConfigFile(
                configFilePath: configFilePath,
                underlyingDescription: "field \(strictJsonError.fieldName): \(strictJsonError.problem)"
            );
        }
        try validateLegacyConfig(legacyConfig: legacyConfig);
        let discoveredModelIds = try discoverModelIdsRequiredForMigration(legacyConfig: legacyConfig, discoverModelIds: discoverModelIds);
        let migratedConfig = buildMigratedConfig(legacyConfig: legacyConfig, discoveredModelIds: discoveredModelIds);
        let migratedJson: Any = migratedConfig.toJsonObject();
        let validatedV1Config = try ConfigDocumentValidation.parseAndValidateV1Document(configFilePath: configFilePath, configJson: migratedJson, performanceAttributionEnabled: performanceAttributionEnabled);
        ConfigPerformanceAttribution.finishedPass(operationName: "prepare-legacy-config-migration", passStart: passStart, passOutcome: "success", performanceAttributionEnabled: performanceAttributionEnabled);
        return validatedV1Config;
    }

    /// Writes the one-time legacy backup with owner-only permissions and
    /// refuses to follow symlinks at the backup path, so an attacker cannot
    /// redirect the recovery copy; an already-present backup is accepted only
    /// when it byte-matches the document being migrated.
    internal static func preserveLegacyConfigBackup(configFilePath: FilePath, legacyConfigBytes: Data) throws {
        guard let parentDirectoryPath = configFilePath.parentDirectory() else {
            throw AstronomicalConfigError.writeConfigFile(
                configFilePath: configFilePath,
                underlyingError: NSError(
                    domain: NSPOSIXErrorDomain,
                    code: Int(EINVAL),
                    userInfo: [NSLocalizedDescriptionKey: "config file has no parent directory"]
                )
            );
        }
        let legacyBackupPath = parentDirectoryPath.appending(component: LegacyConfigMigration.legacyConfigBackupFileName);
        let legacyBackupDescriptor: Int32 = legacyBackupPath.string.withCString { (backupPathCString: UnsafePointer<CChar>) -> Int32 in
            return open(backupPathCString, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, mode_t(0o600));
        };
        if legacyBackupDescriptor < 0 {
            let openError = posixSystemError();
            if openError.code == Int32(EEXIST) {
                return try acceptMatchingExistingBackup(legacyBackupPath: legacyBackupPath, legacyConfigBytes: legacyConfigBytes);
            }
            throw AstronomicalConfigError.writeConfigFile(configFilePath: legacyBackupPath, underlyingError: openError);
        }
        let backupWriteError = writeBytesToFileDescriptor(fileDescriptor: legacyBackupDescriptor, outputBytes: legacyConfigBytes);
        if backupWriteError == nil {
            if fsync(legacyBackupDescriptor) != 0 {
                // The failed backup is incomplete and must not linger: the
                // next migration attempt then recreates it from scratch.
                let _ = unlink(legacyBackupPath.string);
            }
        }
        close(legacyBackupDescriptor);
        if let unwrappedBackupWriteError = backupWriteError {
            throw AstronomicalConfigError.writeConfigFile(configFilePath: legacyBackupPath, underlyingError: unwrappedBackupWriteError);
        }
        try synchronizeDirectoryContents(directoryPath: parentDirectoryPath, errorReportingPath: legacyBackupPath);
    }

    private static func acceptMatchingExistingBackup(legacyBackupPath: FilePath, legacyConfigBytes: Data) throws {
        let existingBackupBytes: Data? = try ConfigFileStore.readExistingConfigFileBytes(configFilePath: legacyBackupPath);
        if existingBackupBytes == legacyConfigBytes {
            return;
        }
        throw ConfigResolutionError.legacyMigration(
            description: "the one-time backup at \(legacyBackupPath.string) already exists with different content; preserve both files and resolve the conflict before retrying"
        );
    }

    private static func discoverModelIdsRequiredForMigration(legacyConfig: LegacyConfigFile, discoverModelIds: LegacyModelIdDiscovery?) throws -> Array<String> {
        if legacyConfig.maximumOutputTokens == nil {
            return Array<String>();
        }
        guard let unwrappedDiscoverModelIds = discoverModelIds else {
            throw ConfigResolutionError.legacyMigration(
                description: "could not discover models needed to preserve global policy: model discovery is not wired into this slice"
            );
        }
        let discoveredModelIds: Array<String>;
        do {
            discoveredModelIds = try unwrappedDiscoverModelIds(legacyConfig.modelDirectories);
        } catch let discoveryError {
            throw ConfigResolutionError.legacyMigration(
                description: "could not discover models needed to preserve global policy: \(discoveryError)"
            );
        }
        if discoveredModelIds.isEmpty {
            throw ConfigResolutionError.legacyMigration(
                description: "global model policy requires at least one currently discovered model; repair model_directories and retry"
            );
        }
        return discoveredModelIds;
    }

    private static func buildMigratedConfig(legacyConfig: LegacyConfigFile, discoveredModelIds: Array<String>) -> UserConfigFile {
        var migratedModels: Dictionary<String, ModelConfigFile> = Dictionary<String, ModelConfigFile>();
        for discoveredModelId in discoveredModelIds {
            var migratedModelConfig = ModelConfigFile(limits: nil, generationDefaults: nil, chunking: nil);
            if let maximumOutputTokens: UInt32 = legacyConfig.maximumOutputTokens {
                migratedModelConfig = ModelConfigFile(
                    limits: nil,
                    generationDefaults: GenerationDefaultsConfigFile(temperature: nil, topP: nil, maximumOutputTokens: maximumOutputTokens),
                    chunking: nil
                );
            }
            migratedModels[discoveredModelId] = migratedModelConfig;
        }
        return UserConfigFile(
            schemaReference: ConfigDocumentValidation.configSchemaReference,
            schemaVersion: 1,
            runtime: RuntimeConfigFile(
                modelDirectories: legacyConfig.modelDirectories,
                maximumMlxMemoryGb: legacyConfig.maximumMlxMemoryGb,
                defaultModel: nil
            ),
            promptCache: PromptCacheConfigFile(enabled: legacyConfig.persistentPromptCacheEnabled, maximumSizeGb: legacyConfig.promptCacheMaxSizeGb),
            chunking: legacyConfig.chunking,
            models: migratedModels,
            diagnostics: DiagnosticsConfigFile(
                performanceAttributionEnabled: legacyConfig.performanceAttributionEnabled,
                completionAttributionEnabled: nil,
                logLevel: legacyConfig.logging?.level.asStr(),
                retainedLogFiles: legacyConfig.logging?.retainedFiles
            )
        );
    }

    private static func validateLegacyConfig(legacyConfig: LegacyConfigFile) throws {
        for modelDirectory: String in legacyConfig.modelDirectories {
            let modelDirectoryPath: FilePath = FilePath(string: modelDirectory);
            if modelDirectoryPath.isAbsolute == false {
                throw AstronomicalConfigError.pathMustBeAbsolute(fieldName: "model_directories", configuredPath: modelDirectoryPath);
            }
        }
        let _ = try ChunkingConfig.resolve(configuredChunkingFile: legacyConfig.chunking);
        if let legacySupervisor = legacyConfig.supervisor {
            if legacySupervisor.bindAddress != nil {
                throw ConfigResolutionError.legacyMigration(
                    description: "legacy supervisor.bind_address cannot be represented because v1 derives the endpoint from the runtime channel; remove the setting to migrate"
                );
            }
        }
        if let maximumOutputTokens: UInt32 = legacyConfig.maximumOutputTokens {
            if maximumOutputTokens == 0 {
                throw ConfigResolutionError.legacyMigration(description: "legacy max_output_tokens must be positive");
            }
        }
    }

    private static func writeBytesToFileDescriptor(fileDescriptor: Int32, outputBytes: Data) -> NSError? {
        var writtenByteCount: Int = 0;
        var writeFailure: NSError? = nil;
        outputBytes.withUnsafeBytes { (outputRawBytes: UnsafeRawBufferPointer) -> Void in
            while writtenByteCount < outputBytes.count {
                let remainingByteCount = outputBytes.count - writtenByteCount;
                let writeResult: Int;
                if let outputBaseAddress = outputRawBytes.baseAddress {
                    writeResult = write(fileDescriptor, outputBaseAddress + writtenByteCount, remainingByteCount);
                } else {
                    writeResult = 0;
                }
                if writeResult < 0 {
                    if errno == EINTR {
                        continue;
                    }
                    writeFailure = posixSystemError();
                    return;
                }
                writtenByteCount += writeResult;
            }
        };
        return writeFailure;
    }

    private static func synchronizeDirectoryContents(directoryPath: FilePath, errorReportingPath: FilePath) throws {
        let directoryDescriptor: Int32 = directoryPath.string.withCString { (directoryPathCString: UnsafePointer<CChar>) -> Int32 in
            return open(directoryPathCString, O_RDONLY);
        };
        if directoryDescriptor < 0 {
            throw AstronomicalConfigError.writeConfigFile(configFilePath: errorReportingPath, underlyingError: posixSystemError());
        }
        let directorySyncSucceeded = fsync(directoryDescriptor) == 0;
        let directorySyncErrno = errno;
        close(directoryDescriptor);
        if directorySyncSucceeded == false {
            let directorySyncError = NSError(
                domain: NSPOSIXErrorDomain,
                code: Int(directorySyncErrno),
                userInfo: [NSLocalizedDescriptionKey: String(cString: strerror(directorySyncErrno))]
            );
            throw AstronomicalConfigError.writeConfigFile(configFilePath: errorReportingPath, underlyingError: directorySyncError);
        }
    }

    private static func posixSystemError() -> NSError {
        let capturedErrno = errno;
        return NSError(
            domain: NSPOSIXErrorDomain,
            code: Int(capturedErrno),
            userInfo: [NSLocalizedDescriptionKey: String(cString: strerror(capturedErrno))]
        );
    }
}
