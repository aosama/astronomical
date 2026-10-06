import Foundation;

#if canImport(Glibc)
import Glibc;
#else
import Darwin;
#endif

import IpcProtocol;

/// The creation half of one worker process: pipes, spawn attributes, and
/// `posix_spawn` itself. Kept apart from the lifecycle half so process
/// creation and process ownership stay separately readable concerns.
extension WorkerProcess {

    /// Creates the three pipe pairs, spawns the child over them, and closes
    /// the child-side ends in the parent immediately. Every descriptor
    /// carries `FD_CLOEXEC`, so an exec'd descendant can never inherit a
    /// parent-side end even if a descriptor momentarily outlives its owner.
    static func spawnWorker(
        workerExecutablePath: String,
        workerArguments: Array<String>,
        workerStartupConfiguration: WorkerStartupConfiguration?
    ) throws -> WorkerProcess {
        var commandPipeFileDescriptors: [Int32] = [-1, -1];
        var eventPipeFileDescriptors: [Int32] = [-1, -1];
        var stderrPipeFileDescriptors: [Int32] = [-1, -1];
        guard pipe(&commandPipeFileDescriptors) == 0,
            pipe(&eventPipeFileDescriptors) == 0,
            pipe(&stderrPipeFileDescriptors) == 0 else {
            WorkerProcess.closeSpawnPipes(
                commandPipeFileDescriptors, eventPipeFileDescriptors, stderrPipeFileDescriptors);
            throw WorkerControlError.startWorker(
                underlyingDescription: "worker stdio pipe creation failed: \(String(cString: strerror(errno)))");
        }
        for fileDescriptor: Int32 in commandPipeFileDescriptors + eventPipeFileDescriptors + stderrPipeFileDescriptors {
            let descriptorFlags: Int32 = fcntl(fileDescriptor, F_GETFD);
            if descriptorFlags >= 0 {
                _ = fcntl(fileDescriptor, F_SETFD, descriptorFlags | FD_CLOEXEC);
            }
        }

        var spawnFileActions: posix_spawn_file_actions_t? = nil;
        posix_spawn_file_actions_init(&spawnFileActions);
        posix_spawn_file_actions_adddup2(&spawnFileActions, commandPipeFileDescriptors[0], STDIN_FILENO);
        posix_spawn_file_actions_adddup2(&spawnFileActions, eventPipeFileDescriptors[1], STDOUT_FILENO);
        posix_spawn_file_actions_adddup2(&spawnFileActions, stderrPipeFileDescriptors[1], STDERR_FILENO);

        // An ignored disposition is the one signal state that survives exec:
        // without resetting it, a supervisor that ignores SIGTERM (or SIGPIPE)
        // would spawn workers that no longer answer graceful shutdown, and the
        // escalation ladder would collapse onto SIGKILL. Foundation's Process
        // and Rust's std::process::Command apply the same reset.
        var spawnAttributes: posix_spawnattr_t? = nil;
        posix_spawnattr_init(&spawnAttributes);
        var defaultDispositionSignals: sigset_t = sigset_t();
        sigemptyset(&defaultDispositionSignals);
        for controlSignalNumber: Int32 in [SIGHUP, SIGINT, SIGQUIT, SIGTERM, SIGPIPE] {
            sigaddset(&defaultDispositionSignals, controlSignalNumber);
        }
        posix_spawnattr_setsigdefault(&spawnAttributes, &defaultDispositionSignals);
        var emptySignalMask: sigset_t = sigset_t();
        sigemptyset(&emptySignalMask);
        posix_spawnattr_setsigmask(&spawnAttributes, &emptySignalMask);
        posix_spawnattr_setflags(&spawnAttributes, Int16(POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK));

        var childProcessIdentifier: pid_t = 0;
        var argumentPointers: Array<UnsafeMutablePointer<CChar>?> =
            WorkerProcess.nullTerminatedArgumentPointers(
                workerExecutablePath: workerExecutablePath,
                workerArguments: workerArguments);
        defer {
            for case let allocatedArgumentPointer in argumentPointers {
                if let allocatedArgumentPointer {
                    free(allocatedArgumentPointer);
                }
            }
        }
        let spawnOutcome: Int32 = posix_spawn(
            &childProcessIdentifier,
            workerExecutablePath,
            &spawnFileActions,
            &spawnAttributes,
            &argumentPointers,
            environ);
        posix_spawn_file_actions_destroy(&spawnFileActions);
        posix_spawnattr_destroy(&spawnAttributes);
        if spawnOutcome != 0 {
            WorkerProcess.closeSpawnPipes(
                commandPipeFileDescriptors, eventPipeFileDescriptors, stderrPipeFileDescriptors);
            throw WorkerControlError.startWorker(
                underlyingDescription: "worker spawn failed: \(String(cString: strerror(spawnOutcome)))");
        }

        // The child owns dup'ed copies as its stdio; the parent drops the
        // child-side ends right here so EOF and EPIPE semantics stay exact.
        WorkerProcess.closeDescriptor(commandPipeFileDescriptors[0]);
        WorkerProcess.closeDescriptor(eventPipeFileDescriptors[1]);
        WorkerProcess.closeDescriptor(stderrPipeFileDescriptors[1]);

        let commandWriter: ProtocolWriter = ProtocolWriter(transport: PipeFrameTransport(
            fileDescriptor: commandPipeFileDescriptors[1],
            isWriteEnd: true));
        let eventReader: ProtocolReader = ProtocolReader(transport: PipeFrameTransport(
            fileDescriptor: eventPipeFileDescriptors[0],
            isWriteEnd: false));
        return WorkerProcess(
            childProcessIdentifier: childProcessIdentifier,
            commandWriteFileDescriptor: commandPipeFileDescriptors[1],
            eventReadFileDescriptor: eventPipeFileDescriptors[0],
            stderrReadFileDescriptor: stderrPipeFileDescriptors[0],
            commandWriter: commandWriter,
            eventReader: eventReader,
            workerExecutablePath: workerExecutablePath,
            workerArguments: workerArguments,
            startupConfiguration: workerStartupConfiguration);
    }

    private static func nullTerminatedArgumentPointers(
        workerExecutablePath: String,
        workerArguments: Array<String>
    ) -> Array<UnsafeMutablePointer<CChar>?> {
        var pointerList: Array<UnsafeMutablePointer<CChar>?> = Array<UnsafeMutablePointer<CChar>?>();
        for argumentText: String in [workerExecutablePath] + workerArguments {
            pointerList.append(strdup(argumentText));
        }
        pointerList.append(nil);
        return pointerList;
    }

    /// Closes whichever spawn pipes exist after a partial setup failure;
    /// `-1` marks a pair that was never created.
    private static func closeSpawnPipes(
        _ commandPipeFileDescriptors: [Int32],
        _ eventPipeFileDescriptors: [Int32],
        _ stderrPipeFileDescriptors: [Int32]
    ) -> Void {
        for fileDescriptor: Int32 in commandPipeFileDescriptors + eventPipeFileDescriptors + stderrPipeFileDescriptors where fileDescriptor >= 0 {
            WorkerProcess.closeDescriptor(fileDescriptor);
        }
    }

    /// The instance method `close()` shadows C `close` inside this type, so
    /// every raw descriptor close goes through this module-qualified helper.
    static func closeDescriptor(_ fileDescriptor: Int32) -> Void {
        if fileDescriptor >= 0 {
            #if canImport(Glibc)
            _ = Glibc.close(fileDescriptor);
            #else
            _ = Darwin.close(fileDescriptor);
            #endif
        }
    }

    static func signalProcess(_ processIdentifier: pid_t, _ signalNumber: Int32) -> Void {
        if processIdentifier > 0 {
            kill(processIdentifier, signalNumber);
        }
    }
}
