import Foundation

import AstronomicalConfig;
import IpcProtocol;

/**
 * The astronomical CLI entry point, porting main.rs: dispatches every verb,
 * maps failures to exit codes (usage and ModelUnavailable exit 2, transient
 * failures exit 1), and execs the launch harness in place.
 */
@main
final class AstronomicalCliMain {

    static func main() {
        let processArguments: Array<String> = Array(CommandLine.arguments.dropFirst());
        switch (CliArgumentParser.parseCommand(processArguments)) {
        case .success(.help):
            print(CliArgumentParser.helpText(), terminator: "");
            exit(0);
        case .success(.version):
            print(AstronomicalCliVersion.version);
            exit(0);
        case let .success(.launch(launchArguments)):
            exit(AstronomicalCliMain.runLaunchVerb(launchArguments: launchArguments));
        case let .success(.schema(schemaArguments)):
            let standardOutput: StandardStreamTextOutputWriter = StandardStreamTextOutputWriter(
                fileHandle: FileHandle.standardOutput);
            if (SchemaCommand.run(schemaArguments, renderedOutput: standardOutput)) {
                exit(0);
            }
            FileHandle.standardError.write(Data("astronomical: could not write the schema document\n".utf8));
            exit(1);
        case let .success(.validateConfig(validateArguments)):
            exit(AstronomicalCliMain.runValidateConfigVerb(validateArguments: validateArguments));
        case let .success(.respond(respondArguments)):
            exit(AstronomicalCliMain.runRespondVerb(respondArguments: respondArguments));
        case let .success(.embed(embedArguments)):
            exit(AstronomicalCliMain.runEmbedVerb(embedArguments: embedArguments));
        case let .success(.models(modelsCommand)):
            exit(AstronomicalCliMain.runModelsVerb(modelsCommand: modelsCommand));
        case .success(.status):
            exit(AstronomicalCliMain.runStatusVerb());
        case let .failure(usageError):
            FileHandle.standardError.write(Data("astronomical: \(usageError)\n\n".utf8));
            FileHandle.standardError.write(Data(CliArgumentParser.helpText().utf8));
            exit(2);
        }
    }

    private static func runLaunchVerb(launchArguments: LaunchArguments) -> Int32 {
        let executablePath: String = CommandLine.arguments[0];
        let preferredInstance: AstronomicalRuntimeInstance = CliInstanceResolution
            .runtimeInstanceFromExecutablePath(executablePath);
        let candidateBindEndpoints: Array<(host: String, port: UInt16)> = CliInstanceResolution
            .candidateInstances(preferredInstance)
            .map { (runtimeInstance: AstronomicalRuntimeInstance) -> (host: String, port: UInt16) in
                let loopbackEndpoint: SocketEndpoint = runtimeInstance.loopbackEndpoint;
                return ("127.0.0.1", loopbackEndpoint.port);
            };
        let isInteractive: Bool = isatty(STDIN_FILENO) == 1;
        let launchDependencies: LaunchCommand.LaunchDependencies = LaunchCommand.LaunchDependencies(
            candidateBindEndpoints: candidateBindEndpoints,
            pathValue: ProcessInfo.processInfo.environment["PATH"] ?? "",
            isInteractive: isInteractive,
            selectionInput: nil,
            stderr: StandardStreamTextOutputWriter(fileHandle: FileHandle.standardError)
        );
        switch (LaunchCommand.prepareLaunch(
            launchArguments: launchArguments,
            launchDependencies: launchDependencies
        )) {
        case let .success(preparedLaunch):
            return AstronomicalCliMain.execPreparedLaunch(preparedLaunch);
        case let .failure(launchError):
            FileHandle.standardError.write(Data("\(launchError)\n".utf8));
            return 1;
        }
    }

    /// Replaces this process with the harness, mirroring Rust's `Command::exec`.
    /// The return value only runs when the exec fails.
    private static func execPreparedLaunch(_ preparedLaunch: LaunchCommand.PreparedLaunch) -> Int32 {
        var environmentVariables: [String: String] = ProcessInfo.processInfo.environment;
        for (environmentName, environmentValue) in preparedLaunch.extraEnvironment {
            environmentVariables[environmentName] = environmentValue;
        }
        let environmentBlock: [String] = environmentVariables.map { (environmentEntry: (key: String, value: String)) -> String in
            return "\(environmentEntry.key)=\(environmentEntry.value)";
        };
        let programPathBytes: [Int8] = preparedLaunch.programPath.utf8CString.map { (codeUnit: CChar) -> Int8 in
            return Int8(codeUnit);
        };
        let execOutcome: Int32 = programPathBytes.withUnsafeBufferPointer { (programBuffer: UnsafeBufferPointer<Int8>) -> Int32 in
            guard let programPointer: UnsafePointer<Int8> = programBuffer.baseAddress else {
                return -1;
            }
            return environmentBlock.withUnsafeBufferPointer { (environmentBuffer: UnsafeBufferPointer<String>) -> Int32 in
                var environmentPointers: [UnsafeMutablePointer<CChar>?] = environmentBuffer.map { (environmentLine: String) -> UnsafeMutablePointer<CChar>? in
                    return strdup(environmentLine);
                }
                environmentPointers.append(nil);
                defer {
                    for environmentPointer: UnsafeMutablePointer<CChar>? in environmentPointers {
                        if let environmentPointer: UnsafeMutablePointer<CChar> = environmentPointer {
                            free(environmentPointer);
                        }
                    }
                }
                return environmentPointers.withUnsafeBufferPointer { (pointerBuffer: UnsafeBufferPointer<UnsafeMutablePointer<CChar>?>) -> Int32 in
                    return execve(programPointer, UnsafeMutablePointer(mutating: pointerBuffer.baseAddress), environ);
                };
            };
        };
        let launchError: LaunchError = .toolStartFailed(
            program: preparedLaunch.programPath,
            cause: String(cString: strerror(execOutcome == -1 ? errno : 0))
        );
        FileHandle.standardError.write(Data("\(launchError)\n".utf8));
        return 1;
    }

    private static func runValidateConfigVerb(
        validateArguments: ValidateConfigArguments
    ) -> Int32 {
        let standardOutput: StandardStreamTextOutputWriter = StandardStreamTextOutputWriter(
            fileHandle: FileHandle.standardOutput);
        do {
            try ValidateConfigCommand.run(
                validateArguments: validateArguments,
                renderedOutput: standardOutput
            );
            return 0;
        } catch let validateError {
            FileHandle.standardError.write(Data("astronomical: \(validateError)\n".utf8));
            return 1;
        }
    }

    private static func runRespondVerb(respondArguments: RespondArguments) -> Int32 {
        let respondDependencies: RespondDependencies = RespondDependencies(
            candidateSocketPaths: AstronomicalCliMain.candidateIpcSocketPaths(),
            stdout: StandardStreamTextOutputWriter(fileHandle: FileHandle.standardOutput),
            stderr: StandardStreamTextOutputWriter(fileHandle: FileHandle.standardError)
        );
        do {
            try RespondCommand.run(
                respondArguments: respondArguments,
                respondDependencies: respondDependencies
            );
            return 0;
        } catch let respondError as RespondError {
            FileHandle.standardError.write(Data("astronomical: \(respondError)\n".utf8));
            // A missing or capability-mismatched model is a usage error, so
            // it exits 2 like a bad flag.
            if case .modelUnavailable = respondError {
                return 2;
            }
            return 1;
        } catch let otherError {
            FileHandle.standardError.write(Data("astronomical: \(otherError)\n".utf8));
            return 1;
        }
    }

    private static func runEmbedVerb(embedArguments: EmbedArguments) -> Int32 {
        let standardInputText: String = String(
            decoding: FileHandle.standardInput.readDataToEndOfFile(),
            as: UTF8.self
        );
        let embedDependencies: EmbedDependencies = EmbedDependencies(
            candidateSocketPaths: AstronomicalCliMain.candidateIpcSocketPaths(),
            standardInputText: standardInputText,
            stdout: StandardStreamTextOutputWriter(fileHandle: FileHandle.standardOutput),
            stderr: StandardStreamTextOutputWriter(fileHandle: FileHandle.standardError)
        );
        do {
            try EmbedCommand.run(
                embedArguments: embedArguments,
                embedDependencies: embedDependencies
            );
            return 0;
        } catch let embedError as EmbedError {
            FileHandle.standardError.write(Data("astronomical: \(embedError)\n".utf8));
            // A missing or capability-mismatched model is a usage error, so
            // it exits 2 like a bad flag.
            if case .modelUnavailable = embedError {
                return 2;
            }
            return 1;
        } catch let otherError {
            FileHandle.standardError.write(Data("astronomical: \(otherError)\n".utf8));
            return 1;
        }
    }

    private static func runModelsVerb(modelsCommand: ModelsSubcommand) -> Int32 {
        let modelsDependencies: ModelsDependencies = ModelsDependencies(
            candidateSocketPaths: AstronomicalCliMain.candidateIpcSocketPaths(),
            stdout: StandardStreamTextOutputWriter(fileHandle: FileHandle.standardOutput),
            stderr: StandardStreamTextOutputWriter(fileHandle: FileHandle.standardError)
        );
        do {
            try ModelsVerb.run(
                modelsCommand: modelsCommand,
                modelsDependencies: modelsDependencies
            );
            return 0;
        } catch let modelsError as ModelsVerbError {
            FileHandle.standardError.write(Data("astronomical: \(modelsError)\n".utf8));
            // A missing model on `models download` is a usage error, so it
            // exits 2 like a bad flag.
            if case .modelUnavailable = modelsError {
                return 2;
            }
            return 1;
        } catch let otherError {
            FileHandle.standardError.write(Data("astronomical: \(otherError)\n".utf8));
            return 1;
        }
    }

    private static func runStatusVerb() -> Int32 {
        do {
            let instancePaths: AstronomicalInstancePaths = try AstronomicalInstancePaths.forCurrentUser(
                runtimeInstance: .development
            );
            let report: String = try StatusCommand.run(instancePaths: instancePaths);
            FileHandle.standardOutput.write(Data(report.utf8));
            return 0;
        } catch let statusError as StatusError {
            FileHandle.standardError.write(Data("astronomical: \(statusError)\n".utf8));
            return 1;
        } catch let otherError {
            FileHandle.standardError.write(Data("astronomical: \(otherError)\n".utf8));
            return 1;
        }
    }

    /// IPC sockets of every instance the running executable could serve,
    /// most preferred first: its own instance, then the others.
    private static func candidateIpcSocketPaths() -> Array<String> {
        let executablePath: String = CommandLine.arguments[0];
        let preferredInstance: AstronomicalRuntimeInstance = CliInstanceResolution
            .runtimeInstanceFromExecutablePath(executablePath);
        var socketPaths: Array<String> = [];
        for runtimeInstance: AstronomicalRuntimeInstance in CliInstanceResolution.candidateInstances(preferredInstance) {
            if let instancePaths: AstronomicalInstancePaths = try? AstronomicalInstancePaths.forCurrentUser(
                runtimeInstance: runtimeInstance
            ) {
                socketPaths.append(instancePaths.ipcSocketFilePath.string);
            }
        }
        return socketPaths;
    }
}
