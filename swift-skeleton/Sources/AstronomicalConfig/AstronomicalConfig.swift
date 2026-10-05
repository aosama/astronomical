import Foundation;

/**
 * Loaded configuration boundary for one Astronomical instance, the Swift
 * port of the Rust `AstronomicalConfig::load_from_instance_paths` journey.
 * The struct name intentionally repeats the module name the way swift-log's
 * `Logger` ecosystem types do, so client code reads
 * `AstronomicalConfig.loadFromInstancePaths(...)` after one import.
 *
 * MIGRATION MARKER — deferred from this slice: chunking-config resolution
 * and persist-back, logging config, maximum-MLX-memory resolution,
 * resolved-model-config, legacy migration, duplicate-key rejection, and the
 * configuration-generation digest.
 */
public struct AstronomicalConfig {
    private let loadedInstancePaths: AstronomicalInstancePaths;
    private let loadedUserConfigFile: UserConfigFile;

    internal init(instancePaths: AstronomicalInstancePaths, userConfigFile: UserConfigFile) {
        self.loadedInstancePaths = instancePaths;
        self.loadedUserConfigFile = userConfigFile;
    }

    /** Loads the instance config, creating the first-run document when absent. */
    public static func loadFromInstancePaths(_ instancePaths: AstronomicalInstancePaths) throws -> AstronomicalConfig {
        let userConfigFile: UserConfigFile = try ConfigFileStore.readUserConfigFile(
            configFilePath: instancePaths.configFilePath
        );
        return AstronomicalConfig(instancePaths: instancePaths, userConfigFile: userConfigFile);
    }

    /**
     * Loads the Development channel for `homeDirectory`. Only the
     * `.astronomical-dev` state beneath it is ever read or written.
     */
    public static func loadFromDevelopmentHomeDirectory(_ homeDirectory: FilePath) throws -> AstronomicalConfig {
        let developmentInstancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            homeDirectory,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );
        return try loadFromInstancePaths(developmentInstancePaths);
    }

    public var instancePaths: AstronomicalInstancePaths {
        return self.loadedInstancePaths;
    }

    public var modelDirectories: Array<FilePath> {
        var directoryPaths: Array<FilePath> = Array<FilePath>();
        for modelDirectoryString: String in self.loadedUserConfigFile.runtime.modelDirectories {
            directoryPaths.append(FilePath(string: modelDirectoryString));
        }
        return directoryPaths;
    }

    public func supervisorBindAddress() throws -> SocketEndpoint {
        let supervisorBindAddress: SocketEndpoint = self.loadedInstancePaths.defaultBindAddress;
        guard (supervisorBindAddress.host.hasPrefix("127.") || supervisorBindAddress.host == "::1") else {
            throw AstronomicalConfigError.nonLoopbackBindAddress(supervisorBindAddress: supervisorBindAddress);
        }
        return supervisorBindAddress;
    }
}
