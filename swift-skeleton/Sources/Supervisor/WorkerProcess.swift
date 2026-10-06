import Foundation;

#if canImport(Glibc)
import Glibc;
#else
import Darwin;
#endif

import IpcProtocol;

/// One spawned inference-worker child process and its framed command/event
/// pipes.
///
/// Mirrors the lifecycle spine of apps/supervisor/src/worker_process.rs:
/// launch with piped stdio, send commands and read events over the same
/// length-delimited JSON framing the sockets use, keep a bounded stderr tail
/// for diagnostics, and terminate by half-closing the command side first,
/// escalating to SIGTERM and then SIGKILL when the worker ignores EOF.
/// The process handle is confined to the supervisor's control thread; the
/// only cross-thread state is the lock-guarded stderr tail.
extension WorkerProcess: @unchecked Sendable {
}

public final class WorkerProcess {

    private static let COMMAND_WRITE_TIMEOUT_SECONDS: TimeInterval = 5;
    private static let SHUTDOWN_TIMEOUT_SECONDS: TimeInterval = 5;
    private static let STDERR_TAIL_MAXIMUM_BYTE_COUNT: Int = 8_192;

    private var process: Process;
    private var commandWriter: ProtocolWriter;
    private var eventReader: ProtocolReader;
    private let stderrTailLock: NSLock;
    private var stderrTailBytes: Array<UInt8>;
    private let workerStartedAt: Date;
    private let workerExecutablePath: String;
    private let workerArguments: Array<String>;
    private let startupConfiguration: WorkerStartupConfiguration?;
    private let lifecycleLock: NSLock;
    private var startupRuntimeConfigurationAppliedFlag: Bool;

    private init(
        process: Process,
        commandWriter: ProtocolWriter,
        eventReader: ProtocolReader,
        workerExecutablePath: String,
        workerArguments: Array<String>,
        startupConfiguration: WorkerStartupConfiguration?
    ) {
        self.process = process;
        self.commandWriter = commandWriter;
        self.eventReader = eventReader;
        self.stderrTailLock = NSLock();
        self.stderrTailBytes = Array<UInt8>();
        self.workerStartedAt = Date();
        self.workerExecutablePath = workerExecutablePath;
        self.workerArguments = workerArguments;
        self.startupConfiguration = startupConfiguration;
        self.lifecycleLock = NSLock();
        self.startupRuntimeConfigurationAppliedFlag = false;
    }

    /// Starts the worker executable with piped stdio and a draining stderr tail.
    public static func launch(workerExecutablePath: String) throws -> WorkerProcess {
        return try WorkerProcess.launch(
            workerExecutablePath: workerExecutablePath,
            arguments: Array<String>());
    }

    /// Starts the worker executable with launch arguments, piped stdio, and a
    /// draining stderr tail.
    public static func launch(
        workerExecutablePath: String,
        arguments: Array<String>
    ) throws -> WorkerProcess {
        return try WorkerProcess.launch(
            workerExecutablePath: workerExecutablePath,
            arguments: arguments,
            workerStartupConfiguration: nil);
    }

    /// Starts the worker executable and writes the InitializeWorker startup
    /// policy over the command pipe, mirroring the launch path of
    /// apps/supervisor/src/worker_process.rs. A failed policy write is
    /// followed by a best-effort graceful termination of the just-started
    /// child before the failure surfaces.
    public static func launch(
        workerExecutablePath: String,
        arguments: Array<String>,
        workerStartupConfiguration: WorkerStartupConfiguration?
    ) throws -> WorkerProcess {
        let workerProcess: Process = Process();
        workerProcess.executableURL = URL(fileURLWithPath: workerExecutablePath);
        workerProcess.arguments = arguments;
        let standardInputPipe: Pipe = Pipe();
        let standardOutputPipe: Pipe = Pipe();
        return try WorkerProcess.launchInternal(
            workerProcess: workerProcess,
            workerExecutablePath: workerExecutablePath,
            workerArguments: arguments,
            standardInputPipe: standardInputPipe,
            standardOutputPipe: standardOutputPipe,
            workerStartupConfiguration: workerStartupConfiguration);
    }

    private static func launchInternal(
        workerProcess: Process,
        workerExecutablePath: String,
        workerArguments: Array<String>,
        standardInputPipe: Pipe,
        standardOutputPipe: Pipe,
        workerStartupConfiguration: WorkerStartupConfiguration?
    ) throws -> WorkerProcess {
        workerProcess.standardInput = standardInputPipe;
        workerProcess.standardOutput = standardOutputPipe;
        let standardErrorPipe: Pipe = Pipe();
        workerProcess.standardError = standardErrorPipe;
        do {
            try workerProcess.run();
        } catch let launchError as NSError {
            throw WorkerControlError.startWorker(underlyingDescription: launchError.localizedDescription);
        }
        let commandWriter: ProtocolWriter = ProtocolWriter(transport: PipeFrameTransport(
            fileDescriptor: standardInputPipe.fileHandleForWriting.fileDescriptor,
            isWriteEnd: true));
        let eventReader: ProtocolReader = ProtocolReader(transport: PipeFrameTransport(
            fileDescriptor: standardOutputPipe.fileHandleForReading.fileDescriptor,
            isWriteEnd: false));
        let launchedWorker: WorkerProcess = WorkerProcess(
            process: workerProcess,
            commandWriter: commandWriter,
            eventReader: eventReader,
            workerExecutablePath: workerExecutablePath,
            workerArguments: workerArguments,
            startupConfiguration: workerStartupConfiguration);
        launchedWorker.drainStandardError(standardErrorPipe: standardErrorPipe);
        if let startupConfiguration: WorkerStartupConfiguration = workerStartupConfiguration {
            do {
                try launchedWorker.sendCommand(.initializeWorker(startupConfiguration));
            } catch let initializationFailure {
                throw WorkerProcess.cleanUpFailedLaunch(
                    launchedWorker,
                    operationFailure: initializationFailure);
            }
        }
        return launchedWorker;
    }

    /// Terminates a worker whose launch handshake failed, combining both
    /// failures when the cleanup cannot run, as the Rust startup cleanup does.
    private static func cleanUpFailedLaunch(
        _ failedWorker: WorkerProcess,
        operationFailure: Error
    ) -> Error {
        do {
            _ = try failedWorker.close();
        } catch {
            return WorkerControlError.operationAndCleanupFailed(
                operationDescription: String(describing: operationFailure),
                cleanupDescription: String(describing: error));
        }
        return operationFailure;
    }

    /// The configuration generation this worker was launched to acknowledge.
    public func expectedConfigurationGeneration() -> String? {
        return self.startupConfiguration?.configurationGeneration;
    }

    /// Whether the startup runtime-policy wait already completed for this
    /// process; live memory updates must not wait again.
    public func isStartupRuntimeConfigurationApplied() -> Bool {
        self.lifecycleLock.lock();
        defer { self.lifecycleLock.unlock(); }
        return self.startupRuntimeConfigurationAppliedFlag;
    }

    public func markStartupRuntimeConfigurationApplied() {
        self.lifecycleLock.lock();
        defer { self.lifecycleLock.unlock(); }
        self.startupRuntimeConfigurationAppliedFlag = true;
    }

    /// The child process identifier while the worker is alive.
    public var processId: Int32? {
        if self.process.isRunning {
            return Int32(self.process.processIdentifier);
        }
        return nil;
    }

    /// The most recent stderr bytes, bounded to the tail maximum.
    public var stderrTail: String {
        self.stderrTailLock.lock();
        defer { self.stderrTailLock.unlock(); }
        return String(decoding: self.stderrTailBytes, as: UTF8.self);
    }

    /// Sends one framed command to the worker.
    public func sendCommand(_ workerCommand: WorkerCommand) throws {
        try self.commandWriter.sendCommand(workerCommand);
    }

    /// Reads the next framed event, or `nil` when the worker closed its output.
    public func nextEvent() throws -> WorkerEvent? {
        return try self.eventReader.nextEvent();
    }

    /// Terminates the worker: half-close the command side and wait a bounded
    /// time for EOF-driven exit, escalate to SIGTERM, then SIGKILL, then reap.
    /// The outcome reports whether EOF alone sufficed and whether the child
    /// reported a successful exit, mirroring the Rust close path.
    public func close() throws -> WorkerTerminationOutcome {
        try self.commandWriter.close();
        let shutdownDeadline: Date = Date().addingTimeInterval(
            WorkerProcess.SHUTDOWN_TIMEOUT_SECONDS);
        while self.process.isRunning && Date() < shutdownDeadline {
            Thread.sleep(forTimeInterval: 0.02);
        }
        if !self.process.isRunning {
            self.process.waitUntilExit();
            return .graceful(
                processExitSuccessful: self.process.terminationStatus == 0);
        }
        if let processId: Int32 = self.processId {
            kill(processId, SIGTERM);
        }
        let terminateDeadline: Date = Date().addingTimeInterval(
            WorkerProcess.SHUTDOWN_TIMEOUT_SECONDS);
        while self.process.isRunning && Date() < terminateDeadline {
            Thread.sleep(forTimeInterval: 0.02);
        }
        if self.process.isRunning {
            if let processId: Int32 = self.processId {
                kill(processId, SIGKILL);
            }
        }
        self.process.waitUntilExit();
        return .forced(
            processExitSuccessful: self.process.terminationStatus == 0);
    }

    /// Kills the worker outright and reaps it; the containment path for a
    /// worker that can no longer be trusted to answer a graceful close.
    public func forceTerminate() throws -> WorkerTerminationOutcome {
        if let processId: Int32 = self.processId {
            kill(processId, SIGKILL);
        }
        self.process.waitUntilExit();
        return .forced(
            processExitSuccessful: self.process.terminationStatus == 0);
    }

    /// Whether the worker process is still alive.
    public func hasLivingProcess() -> Bool {
        return self.process.isRunning;
    }

    /// Starts a clean process from this worker's exact portable launch inputs
    /// after the previous process ended, resetting the once-per-process
    /// startup flag; mirrors the Rust relaunch_after_termination.
    public func relaunchAfterTermination() throws -> Void {
        self.eventReader.closeTransportFileDescriptor();
        // Best-effort close of the replaced child's command pipe; the child
        // already ended, so a failure here leaves nothing to recover.
        do {
            try self.commandWriter.close();
        } catch {
            self.commandWriter.closeTransportFileDescriptor();
        }
        let replacementWorker: WorkerProcess = try WorkerProcess.launch(
            workerExecutablePath: self.workerExecutablePath,
            arguments: self.workerArguments,
            workerStartupConfiguration: self.startupConfiguration);
        self.process = replacementWorker.process;
        self.commandWriter = replacementWorker.commandWriter;
        self.eventReader = replacementWorker.eventReader;
        self.lifecycleLock.lock();
        self.startupRuntimeConfigurationAppliedFlag = false;
        self.lifecycleLock.unlock();
    }

    private func drainStandardError(standardErrorPipe: Pipe) {
        standardErrorPipe.fileHandleForReading.readabilityHandler = { [weak self] (fileHandle: FileHandle) -> Void in
            let receivedData: Data = fileHandle.availableData;
            if receivedData.isEmpty {
                fileHandle.readabilityHandler = nil;
                return;
            }
            guard let strongSelf: WorkerProcess = self else {
                return;
            }
            strongSelf.stderrTailLock.lock();
            strongSelf.stderrTailBytes.append(contentsOf: Array(receivedData));
            if strongSelf.stderrTailBytes.count > WorkerProcess.STDERR_TAIL_MAXIMUM_BYTE_COUNT {
                strongSelf.stderrTailBytes.removeFirst(
                    strongSelf.stderrTailBytes.count - WorkerProcess.STDERR_TAIL_MAXIMUM_BYTE_COUNT);
            }
            strongSelf.stderrTailLock.unlock();
        };
    }
}
