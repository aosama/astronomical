import Foundation;

import AstronomicalConfig;

/// Where the daemon finds its worker executable when the configuration
/// names no explicit override.
///
/// Migrates the fallback_worker_executable_path derivation from
/// apps/supervisor/src/main.rs: the worker binary lives beside the daemon
/// binary, named for the current platform.
public enum FallbackWorkerExecutablePath {

    public static let workerExecutableName: String = "astronomical-inference-worker";

    /// The worker executable path derived from the running daemon's own
    /// directory; errors when the daemon cannot locate itself.
    public static func derive() throws -> FilePath {
        guard let daemonExecutableUrl: URL = Bundle.main.executableURL else {
            throw FallbackWorkerExecutablePathError.missingCurrentExecutable;
        }
        let daemonExecutablePath: String = daemonExecutableUrl.path;
        let daemonExecutableDirectory: String = (daemonExecutablePath as NSString).deletingLastPathComponent;
        guard !daemonExecutableDirectory.isEmpty else {
            throw FallbackWorkerExecutablePathError.missingExecutableDirectory;
        }
        return FilePath(string: daemonExecutableDirectory)
            .appending(component: FallbackWorkerExecutablePath.workerExecutableName);
    }
}

/// Failures while locating the bundled worker executable.
public enum FallbackWorkerExecutablePathError: Error, CustomStringConvertible {

    case missingCurrentExecutable;
    case missingExecutableDirectory;

    public var description: String {
        switch (self) {
        case .missingCurrentExecutable:
            return "the daemon could not resolve its own executable path";
        case .missingExecutableDirectory:
            return "the daemon executable has no containing directory";
        }
    }
}
