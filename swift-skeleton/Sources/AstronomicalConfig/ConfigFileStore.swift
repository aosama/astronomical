import Foundation;
import Darwin;
import os;

/**
 * Read-or-create journey for the strict v1 configuration file, the Swift
 * port of the Rust `config_file.rs` unit.
 */
internal enum ConfigFileStore {
    internal static let MAXIMUM_CONFIG_FILE_BYTES: Int = 1_048_576;
    internal static let CONFIG_SCHEMA_FILE_NAME: String = "astronomical-config.schema.json";

    private static let EXPECTED_SCHEMA_REFERENCE: String = "./astronomical-config.schema.json";
    private static let TEMPORARY_CONFIG_FILE_ATTEMPT_LIMIT: Int = 100;
    /** Monotonic suffix so concurrent first-run writers never share a temp name. */
    private static let temporaryFileSequenceState: OSAllocatedUnfairLock<UInt64> = OSAllocatedUnfairLock<UInt64>(
        initialState: 0
    );

    internal static func readUserConfigFile(configFilePath: FilePath) throws -> UserConfigFile {
        guard let existingConfigBytes: Data = try ConfigFileStore.readExistingConfigFileBytes(configFilePath: configFilePath) else {
            return try ConfigFileStore.createFirstRunConfig(configFilePath: configFilePath);
        }
        return try ConfigFileStore.parseAndValidateV1(configFilePath: configFilePath, configBytes: existingConfigBytes);
    }

    private static func parseAndValidateV1(configFilePath: FilePath, configBytes: Data) throws -> UserConfigFile {
        // Duplicate object keys must fail the strict read exactly as the Rust
        // unit does; a silently last-wins dictionary would mask operator typos.
        let jsonRootValue: Any = try DuplicateKeyJson.parseJsonRejectingDuplicates(
            configFilePath: configFilePath,
            configBytes: configBytes
        );
        let userConfigFile: UserConfigFile;
        do {
            userConfigFile = try UserConfigFile.fromJsonObject(jsonRootValue);
        } catch let strictJsonError as StrictJsonError {
            throw AstronomicalConfigError.parseConfigFile(
                configFilePath: configFilePath,
                underlyingDescription: "field " + strictJsonError.fieldName + ": " + strictJsonError.problem
            );
        }
        try ConfigFileStore.validateUserConfigFile(userConfigFile);
        return userConfigFile;
    }

    private static func validateUserConfigFile(_ userConfigFile: UserConfigFile) throws -> Void {
        if (userConfigFile.schemaVersion != 1) {
            throw AstronomicalConfigError.unsupportedSchemaVersion(schemaVersion: userConfigFile.schemaVersion);
        }
        if (userConfigFile.schemaReference != ConfigFileStore.EXPECTED_SCHEMA_REFERENCE) {
            throw AstronomicalConfigError.invalidSchemaReference;
        }
        for modelDirectoryString: String in userConfigFile.runtime.modelDirectories {
            let modelDirectoryPath: FilePath = FilePath(string: modelDirectoryString);
            if (!modelDirectoryPath.isAbsolute) {
                throw AstronomicalConfigError.pathMustBeAbsolute(
                    fieldName: "runtime.model_directories",
                    configuredPath: modelDirectoryPath
                );
            }
        }
    }

    private static func createFirstRunConfig(configFilePath: FilePath) throws -> UserConfigFile {
        let firstRunConfigFile: UserConfigFile = UserConfigFile.minimal();
        let firstRunConfigBytes: Data = try ConfigFileStore.serializeConfigFile(
            configFilePath: configFilePath,
            userConfigFile: firstRunConfigFile
        );
        try ConfigFileStore.writeAdjacentSchema(configFilePath: configFilePath);
        try ConfigFileStore.writeConfigFileBytesAtomically(configFilePath, bytes: firstRunConfigBytes);
        return firstRunConfigFile;
    }

    private static func serializeConfigFile(configFilePath: FilePath, userConfigFile: UserConfigFile) throws -> Data {
        do {
            return try JSONSerialization.data(
                withJSONObject: userConfigFile.toJsonObject(),
                options: [.prettyPrinted, .sortedKeys]
            );
        } catch {
            throw AstronomicalConfigError.serializeConfigFile(configFilePath: configFilePath);
        }
    }

    /**
     * The validation schema rides beside config.json so editors can offer
     * completions against the exact version the daemon enforces.
     */
    internal static func writeAdjacentSchema(configFilePath: FilePath) throws -> Void {
        guard let stateDirectoryPath: FilePath = configFilePath.parentDirectory() else {
            throw AstronomicalConfigError.writeConfigFile(
                configFilePath: configFilePath,
                underlyingError: CocoaError(.fileWriteUnknown)
            );
        }
        let schemaDestinationPath: FilePath = stateDirectoryPath.appending(
            component: ConfigFileStore.CONFIG_SCHEMA_FILE_NAME
        );
        let schemaBytes: Data = try ConfigFileStore.loadBundledSchemaBytes(schemaDestinationPath: schemaDestinationPath);
        try ConfigFileStore.writeConfigFileBytesAtomically(schemaDestinationPath, bytes: schemaBytes);
    }

    private static func loadBundledSchemaBytes(schemaDestinationPath: FilePath) throws -> Data {
        // The file name carries a double extension, so the resource name is
        // "astronomical-config.schema" with extension "json"; SwiftPM may or
        // may not preserve the Resources/ directory inside the bundle.
        let schemaResourceUrl: URL? = Bundle.module.url(
            forResource: "astronomical-config.schema",
            withExtension: "json",
            subdirectory: "Resources"
        ) ?? Bundle.module.url(forResource: "astronomical-config.schema", withExtension: "json");
        guard let presentSchemaResourceUrl: URL = schemaResourceUrl else {
            throw AstronomicalConfigError.writeConfigFile(
                configFilePath: schemaDestinationPath,
                underlyingError: CocoaError(.fileReadNoSuchFile)
            );
        }
        do {
            return try Data(contentsOf: presentSchemaResourceUrl);
        } catch let schemaReadError as NSError {
            throw AstronomicalConfigError.writeConfigFile(
                configFilePath: schemaDestinationPath,
                underlyingError: schemaReadError
            );
        }
    }

    /**
     * Missing file means first run, not an error. The size cap is checked
     * after the read do/catch so the structured too-large error cannot be
     * re-mapped into a read failure.
     */
    internal static func readExistingConfigFileBytes(configFilePath: FilePath) throws -> Data? {
        let configBytes: Data;
        do {
            let readFileHandle: FileHandle = try FileHandle(
                forReadingFrom: URL(fileURLWithPath: configFilePath.string)
            );
            configBytes = try readFileHandle.read(upToCount: ConfigFileStore.MAXIMUM_CONFIG_FILE_BYTES + 1) ?? Data();
            try readFileHandle.close();
        } catch let underlyingError as CocoaError where (underlyingError.code == .fileNoSuchFile) {
            return nil;
        } catch let posixOpenError as NSError
        where (posixOpenError.domain == NSPOSIXErrorDomain && posixOpenError.code == ENOENT) {
            return nil;
        } catch let readFailureError as NSError {
            throw AstronomicalConfigError.readConfigFile(
                configFilePath: configFilePath,
                underlyingError: readFailureError
            );
        }
        if (configBytes.count > ConfigFileStore.MAXIMUM_CONFIG_FILE_BYTES) {
            throw AstronomicalConfigError.configFileTooLarge(
                configFilePath: configFilePath,
                maximumBytes: ConfigFileStore.MAXIMUM_CONFIG_FILE_BYTES
            );
        }
        return configBytes;
    }

    /**
     * Byte-exact atomic transaction: create the parent directory, write a
     * private temporary file, fsync it, then rename over the destination so
     * a reader never observes a torn configuration.
     */
    internal static func writeConfigFileBytesAtomically(_ destinationPath: FilePath, bytes: Data) throws -> Void {
        guard let parentDirectoryPath: FilePath = destinationPath.parentDirectory() else {
            throw AstronomicalConfigError.writeConfigFile(
                configFilePath: destinationPath,
                underlyingError: CocoaError(.fileWriteUnknown)
            );
        }
        do {
            try FileManager.default.createDirectory(
                atPath: parentDirectoryPath.string,
                withIntermediateDirectories: true
            );
        } catch let directoryCreationError as NSError {
            throw AstronomicalConfigError.writeConfigFile(
                configFilePath: destinationPath,
                underlyingError: directoryCreationError
            );
        }
        let temporaryFilePath: FilePath = try ConfigFileStore.createTemporaryConfigFile(
            destinationPath: destinationPath,
            bytes: bytes
        );
        let renameOutcome: Int32 = destinationPath.string.withCString { (destinationCString: UnsafePointer<CChar>) in
            return temporaryFilePath.string.withCString { (temporaryCString: UnsafePointer<CChar>) in
                return rename(temporaryCString, destinationCString);
            };
        };
        if (renameOutcome != 0) {
            let renameErrnoValue: Int32 = errno;
            do {
                try FileManager.default.removeItem(atPath: temporaryFilePath.string);
            } catch {
                // Best-effort cleanup: the rename failure below is the
                // failure the caller must see, not this secondary one.
            }
            throw AstronomicalConfigError.writeConfigFile(
                configFilePath: destinationPath,
                underlyingError: NSError(domain: NSPOSIXErrorDomain, code: Int(renameErrnoValue))
            );
        }
        ConfigFileStore.synchronizeDirectoryBestEffort(parentDirectoryPath);
    }

    /**
     * Directory fsync so the rename itself survives a crash. The Rust unit
     * only warns when this fails, so neither outcome propagates here.
     */
    private static func synchronizeDirectoryBestEffort(_ directoryPath: FilePath) -> Void {
        let directoryFileDescriptor: Int32 = directoryPath.string.withCString { (directoryCString: UnsafePointer<CChar>) in
            return open(directoryCString, O_RDONLY);
        };
        if (directoryFileDescriptor == -1) {
            return;
        }
        _ = fsync(directoryFileDescriptor);
        close(directoryFileDescriptor);
    }

    private static func createTemporaryConfigFile(destinationPath: FilePath, bytes: Data) throws -> FilePath {
        guard let parentDirectoryPath: FilePath = destinationPath.parentDirectory() else {
            throw AstronomicalConfigError.writeConfigFile(
                configFilePath: destinationPath,
                underlyingError: CocoaError(.fileWriteUnknown)
            );
        }
        let processIdentifier: pid_t = getpid();
        let sequenceNumber: UInt64 = ConfigFileStore.temporaryFileSequenceState.withLock({ (counterState: inout UInt64) in
            counterState += 1;
            return counterState;
        });
        for attemptIndex: Int in 0..<ConfigFileStore.TEMPORARY_CONFIG_FILE_ATTEMPT_LIMIT {
            let temporaryFilePath: FilePath = parentDirectoryPath.appending(
                component: ".config.json.tmp.\(processIdentifier).\(sequenceNumber + UInt64(attemptIndex))"
            );
            let creationFileDescriptor: Int32 = temporaryFilePath.string.withCString { (temporaryCString: UnsafePointer<CChar>) in
                return open(temporaryCString, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, mode_t(0o600));
            };
            if (creationFileDescriptor == -1) {
                let creationErrnoValue: Int32 = errno;
                if (creationErrnoValue == EEXIST) {
                    continue;
                }
                throw AstronomicalConfigError.writeConfigFile(
                    configFilePath: destinationPath,
                    underlyingError: NSError(domain: NSPOSIXErrorDomain, code: Int(creationErrnoValue))
                );
            }
            let temporaryFileHandle: FileHandle = FileHandle(
                fileDescriptor: creationFileDescriptor,
                closeOnDealloc: true
            );
            do {
                try temporaryFileHandle.write(contentsOf: bytes);
                try temporaryFileHandle.synchronize();
            } catch let temporaryWriteError as NSError {
                do {
                    try FileManager.default.removeItem(atPath: temporaryFilePath.string);
                } catch {
                    // Best-effort cleanup: the write failure below is the
                    // failure the caller must see, not this secondary one.
                }
                throw AstronomicalConfigError.writeConfigFile(
                    configFilePath: destinationPath,
                    underlyingError: temporaryWriteError
                );
            }
            return temporaryFilePath;
        }
        throw AstronomicalConfigError.writeConfigFile(
            configFilePath: destinationPath,
            underlyingError: NSError(
                domain: NSPOSIXErrorDomain,
                code: Int(EEXIST),
                userInfo: [
                    NSLocalizedDescriptionKey: "exhausted all temporary config file name attempts"
                ]
            )
        );
    }
}
