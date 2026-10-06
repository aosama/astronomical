import Foundation;

import AstronomicalConfig;

/// Parses the astronomicald process arguments into one daemon command.
///
/// Mirrors apps/supervisor/src/daemon_arguments.rs: only --instance and
/// --state-directory carry values, each may appear at most once, and the
/// state-directory override must be an absolute non-root path.
public struct DaemonArguments: Equatable {

    private static let HELP_TEXT: String = "Astronomical local model runner\n\nUsage: astronomicald [--instance stable|development] [--state-directory PATH]\n       astronomicald --help\n       astronomicald --version\n\nOptions:\n  --instance INSTANCE      Runtime instance (default: development)\n  --state-directory PATH   Absolute writable state root for this invocation\n  -h, --help               Show this help\n  --version                Show exact build identity\n";

    public static func parse(processArguments: Array<String>) throws -> DaemonCommand {
        let suppliedArguments: Array<String> = Array(processArguments.dropFirst());
        if suppliedArguments.contains("--help") || suppliedArguments.contains("-h") {
            return .help;
        }
        if suppliedArguments.contains("--version") {
            return .version;
        }

        var runtimeInstance: AstronomicalRuntimeInstance = AstronomicalRuntimeInstance.development;
        var stateDirectoryOverride: String? = nil;
        var hasRuntimeInstanceArgument: Bool = false;
        var hasStateDirectoryArgument: Bool = false;
        var argumentIndex: Int = 0;
        while argumentIndex < suppliedArguments.count {
            let argument: String = suppliedArguments[argumentIndex];
            if argument == "--instance" {
                if hasRuntimeInstanceArgument {
                    throw DaemonArgumentError.repeatedArgument(argumentName: "--instance");
                }
                guard argumentIndex + 1 < suppliedArguments.count else {
                    throw DaemonArgumentError.missingValue(argumentName: "--instance");
                }
                let rawRuntimeInstance: String = suppliedArguments[argumentIndex + 1];
                guard let parsedInstance: AstronomicalRuntimeInstance = AstronomicalRuntimeInstance(rawValue: rawRuntimeInstance) else {
                    throw DaemonArgumentError.invalidInstance(rawValue: rawRuntimeInstance);
                }
                runtimeInstance = parsedInstance;
                hasRuntimeInstanceArgument = true;
                argumentIndex += 2;
                continue;
            }
            if argument == "--state-directory" {
                if hasStateDirectoryArgument {
                    throw DaemonArgumentError.repeatedArgument(argumentName: "--state-directory");
                }
                guard argumentIndex + 1 < suppliedArguments.count else {
                    throw DaemonArgumentError.missingValue(argumentName: "--state-directory");
                }
                let stateDirectory: String = suppliedArguments[argumentIndex + 1];
                guard stateDirectory.hasPrefix("/"), stateDirectory != "/" else {
                    throw DaemonArgumentError.invalidStateDirectory(path: stateDirectory);
                }
                stateDirectoryOverride = stateDirectory;
                hasStateDirectoryArgument = true;
                argumentIndex += 2;
                continue;
            }
            throw DaemonArgumentError.unknownArgument(argument: argument);
        }
        return .run(DaemonArguments(
            runtimeInstance: runtimeInstance,
            stateDirectoryOverride: stateDirectoryOverride));
    }

    public static func helpText() -> String {
        return HELP_TEXT;
    }

    let runtimeInstance: AstronomicalRuntimeInstance;
    let stateDirectoryOverride: String?;

    private init(runtimeInstance: AstronomicalRuntimeInstance, stateDirectoryOverride: String?) {
        self.runtimeInstance = runtimeInstance;
        self.stateDirectoryOverride = stateDirectoryOverride;
    }

    /// Resolves the instance paths from the override or the standard per-user
    /// home location, exactly like the Rust daemon startup does.
    public func resolveInstancePaths() throws -> AstronomicalInstancePaths {
        if let stateDirectoryOverride: String = self.stateDirectoryOverride {
            return AstronomicalInstancePaths.forStateDirectory(
                FilePath(string: stateDirectoryOverride),
                runtimeInstance: self.runtimeInstance);
        }
        return try AstronomicalInstancePaths.forCurrentUser(
            runtimeInstance: self.runtimeInstance);
    }
}

/// One action the daemon entry point takes for its process arguments.
public enum DaemonCommand: Equatable {
    case run(DaemonArguments)
    case help
    case version
}

public enum DaemonArgumentError: Error, Equatable {
    case missingValue(argumentName: String)
    case invalidInstance(rawValue: String)
    case repeatedArgument(argumentName: String)
    case invalidStateDirectory(path: String)
    case unknownArgument(argument: String)
}
