import Foundation;

/** Complete writable path and endpoint boundary for one Astronomical instance. */
public struct AstronomicalInstancePaths: Equatable, Sendable {
    private static let STABLE_STATE_DIRECTORY_NAME: String = ".astronomical";
    private static let DEVELOPMENT_STATE_DIRECTORY_NAME: String = ".astronomical-dev";
    // App Store channel state roots. Sandboxed apps may write only inside
    // their container, and the platform-standard Application Support
    // directory is mapped into that container automatically, so the store
    // build derives all state from it instead of a home-directory
    // dot-folder (App Review guideline 2.4.5(ii)).
    private static let APPLICATION_SUPPORT_STABLE_DIRECTORY_NAME: String = "Astronomical";
    private static let APPLICATION_SUPPORT_DEVELOPMENT_DIRECTORY_NAME: String = "Astronomical Development";

    private let resolvedRuntimeInstance: AstronomicalRuntimeInstance?;
    private let rootStateDirectory: FilePath;
    private let resolvedDefaultBindAddress: SocketEndpoint;
    private let resolvedIsStandardStateDirectory: Bool;

    private init(
        resolvedRuntimeInstance: AstronomicalRuntimeInstance?,
        rootStateDirectory: FilePath,
        resolvedDefaultBindAddress: SocketEndpoint,
        resolvedIsStandardStateDirectory: Bool
    ) {
        self.resolvedRuntimeInstance = resolvedRuntimeInstance;
        self.rootStateDirectory = rootStateDirectory;
        self.resolvedDefaultBindAddress = resolvedDefaultBindAddress;
        self.resolvedIsStandardStateDirectory = resolvedIsStandardStateDirectory;
    }

    public static func forCurrentUser(runtimeInstance: AstronomicalRuntimeInstance) throws -> AstronomicalInstancePaths {
        let homeEnvironmentValue: String? = ProcessInfo.processInfo.environment["HOME"];
        guard let rawHomeDirectory: String = homeEnvironmentValue, !rawHomeDirectory.isEmpty else {
            throw AstronomicalConfigError.homeDirectoryRequired;
        }
        let homeDirectory: FilePath = FilePath(string: rawHomeDirectory);
        return try AstronomicalInstancePaths.forUserHomeDirectory(homeDirectory, runtimeInstance: runtimeInstance);
    }

    /**
     * Resolves an active user's existing home before deriving standard state.
     * Acceptance fixtures use `forHomeDirectory` when no real home exists.
     */
    public static func forUserHomeDirectory(
        _ homeDirectory: FilePath,
        runtimeInstance: AstronomicalRuntimeInstance
    ) throws -> AstronomicalInstancePaths {
        if (!homeDirectory.isAbsolute) {
            throw AstronomicalConfigError.pathMustBeAbsolute(fieldName: "HOME", configuredPath: homeDirectory);
        }
        // URL.resolvingSymlinksInPath() succeeds for missing paths, so
        // existence is checked explicitly to keep the canonicalize failure
        // semantics the Rust channel relies on.
        if (!FileManager.default.fileExists(atPath: homeDirectory.string)) {
            throw AstronomicalConfigError.resolveHomeDirectory(
                homeDirectory: homeDirectory,
                underlyingError: CocoaError(.fileReadNoSuchFile)
            );
        }
        let canonicalHomeUrl: URL = URL(fileURLWithPath: homeDirectory.string).resolvingSymlinksInPath();
        guard let canonicalHomeDirectory: FilePath = FilePath(url: canonicalHomeUrl) else {
            throw AstronomicalConfigError.resolveHomeDirectory(
                homeDirectory: homeDirectory,
                underlyingError: CocoaError(.fileReadInvalidFileName)
            );
        }
        if (canonicalHomeDirectory.isRoot) {
            throw AstronomicalConfigError.homeDirectoryMustNotBeRoot;
        }
        return AstronomicalInstancePaths.forHomeDirectory(canonicalHomeDirectory, runtimeInstance: runtimeInstance);
    }

    /** Instance paths for the default user location of `runtimeInstance`. */
    public static func forHomeDirectory(
        _ homeDirectory: FilePath,
        runtimeInstance: AstronomicalRuntimeInstance
    ) -> AstronomicalInstancePaths {
        let stateDirectoryName: String;
        switch (runtimeInstance) {
        case .stable:
            stateDirectoryName = AstronomicalInstancePaths.STABLE_STATE_DIRECTORY_NAME;
        case .development:
            stateDirectoryName = AstronomicalInstancePaths.DEVELOPMENT_STATE_DIRECTORY_NAME;
        }
        return AstronomicalInstancePaths.forStateDirectoryWithStandardEndpoint(
            homeDirectory.appending(component: stateDirectoryName),
            runtimeInstance: runtimeInstance
        );
    }

    /**
     * Resolves instance state beneath the platform-standard macOS Application
     * Support directory. The App Store channel uses this instead of the
     * home-directory dot-folder because sandboxed store builds may write only
     * inside their container, and Application Support is the container-mapped
     * location Apple's file-system requirements name for persistent state.
     * Standard-instance semantics (loopback endpoint guards) carry over
     * unchanged so the store build keeps the same endpoint discipline as the
     * direct channel.
     */
    public static func forApplicationSupportDirectory(
        _ applicationSupportDirectory: FilePath,
        runtimeInstance: AstronomicalRuntimeInstance
    ) -> AstronomicalInstancePaths {
        let stateDirectoryName: String;
        switch (runtimeInstance) {
        case .stable:
            stateDirectoryName = AstronomicalInstancePaths.APPLICATION_SUPPORT_STABLE_DIRECTORY_NAME;
        case .development:
            stateDirectoryName = AstronomicalInstancePaths.APPLICATION_SUPPORT_DEVELOPMENT_DIRECTORY_NAME;
        }
        return AstronomicalInstancePaths.forStateDirectoryWithStandardEndpoint(
            applicationSupportDirectory.appending(component: stateDirectoryName),
            runtimeInstance: runtimeInstance
        );
    }

    private static func forStateDirectoryWithStandardEndpoint(
        _ stateDirectory: FilePath,
        runtimeInstance: AstronomicalRuntimeInstance
    ) -> AstronomicalInstancePaths {
        return AstronomicalInstancePaths(
            resolvedRuntimeInstance: runtimeInstance,
            rootStateDirectory: stateDirectory,
            resolvedDefaultBindAddress: runtimeInstance.loopbackEndpoint,
            resolvedIsStandardStateDirectory: true
        );
    }

    public static func forStateDirectory(
        _ stateDirectory: FilePath,
        runtimeInstance: AstronomicalRuntimeInstance
    ) -> AstronomicalInstancePaths {
        // Custom state must coexist with installed channels and parallel test
        // instances without restoring a user-editable endpoint to the strict
        // public configuration document.
        return AstronomicalInstancePaths(
            resolvedRuntimeInstance: runtimeInstance,
            rootStateDirectory: stateDirectory,
            resolvedDefaultBindAddress: SocketEndpoint.loopback(port: 0),
            resolvedIsStandardStateDirectory: false
        );
    }

    public static func forExplicitStateDirectory(
        _ stateDirectory: FilePath,
        defaultBindAddress: SocketEndpoint
    ) -> AstronomicalInstancePaths {
        return AstronomicalInstancePaths(
            resolvedRuntimeInstance: nil,
            rootStateDirectory: stateDirectory,
            resolvedDefaultBindAddress: defaultBindAddress,
            resolvedIsStandardStateDirectory: false
        );
    }

    public var runtimeInstance: AstronomicalRuntimeInstance? {
        return self.resolvedRuntimeInstance;
    }

    public var stateDirectory: FilePath {
        return self.rootStateDirectory;
    }

    public var defaultBindAddress: SocketEndpoint {
        return self.resolvedDefaultBindAddress;
    }

    public var isStandardStateDirectory: Bool {
        return self.resolvedIsStandardStateDirectory;
    }

    /**
     * Prevents a standard Stable or Development instance from adopting the
     * other channel's endpoint while leaving explicit test state configurable.
     */
    public func validateConfiguredBindAddress(_ configuredBindAddress: SocketEndpoint) throws -> SocketEndpoint {
        if (self.resolvedIsStandardStateDirectory && configuredBindAddress != self.resolvedDefaultBindAddress) {
            throw AstronomicalConfigError.standardInstanceBindAddressMismatch(
                configuredBindAddress: configuredBindAddress,
                expectedBindAddress: self.resolvedDefaultBindAddress
            );
        }
        return configuredBindAddress;
    }

    public var configFilePath: FilePath {
        return self.rootStateDirectory.appending(component: "config.json");
    }

    public var promptCacheDirectory: FilePath {
        return self.rootStateDirectory.appending(component: "cache");
    }

    public var modelsDirectory: FilePath {
        return self.rootStateDirectory.appending(component: "models");
    }

    public var loggingDirectory: FilePath {
        return self.rootStateDirectory.appending(component: "logs");
    }

    public var daemonOwnershipFilePath: FilePath {
        return self.rootStateDirectory.appending(component: "menu-owned-daemon.json");
    }

    public var instanceLockFilePath: FilePath {
        return self.rootStateDirectory.appending(component: "instance.lock");
    }

    /** Private local IPC socket the daemon serves ephemeral CLI verbs on. */
    public var ipcSocketFilePath: FilePath {
        return self.rootStateDirectory.appending(component: "ipc.sock");
    }
}
