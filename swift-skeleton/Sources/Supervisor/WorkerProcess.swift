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
///
/// The child is spawned through `posix_spawn` over pipes this type creates
/// itself (see `WorkerProcessSpawning.swift`). Foundation's `Process`/`Pipe`
/// close their file descriptors at unpredictable deallocation times, while
/// descriptor numbers freed earlier get handed to new pipes — a stale
/// `FileHandle` then closes a descriptor another worker already owns
/// (observed as `EBADF` reads and spurious EOF). Owning every descriptor
/// end to end makes each close happen exactly once, at a known point, so
/// concurrent worker lifecycles cannot corrupt each other's transports.
/// The process handle is confined to the supervisor's control thread;
/// cross-thread state is the lock-guarded stderr tail, the drain
/// generation, and the reap state polled by `hasLivingProcess`.
extension WorkerProcess: @unchecked Sendable {
}

public final class WorkerProcess {

    private static let SHUTDOWN_TIMEOUT_SECONDS: TimeInterval = 5;
    private static let STDERR_TAIL_MAXIMUM_BYTE_COUNT: Int = 8_192;
    private static let EXIT_POLL_INTERVAL_SECONDS: TimeInterval = 0.02;
    private static let STDERR_DRAIN_CHUNK_BYTE_COUNT: Int = 4_096;

    private let workerExecutablePath: String;
    private let workerArguments: Array<String>;
    private let startupConfiguration: WorkerStartupConfiguration?;
    private let workerStartedAt: Date;

    private let lifecycleLock: NSLock;
    private var childProcessIdentifier: pid_t;
    private var commandWriteFileDescriptor: Int32;
    private var eventReadFileDescriptor: Int32;
    private var stderrReadFileDescriptor: Int32;
    private var hasReapedExit: Bool;
    private var exitWasSuccessful: Bool;
    private var escalatedToSignal: Bool;
    private var adoptedByAnotherLifecycle: Bool;
    private var startupRuntimeConfigurationAppliedFlag: Bool;

    private var commandWriter: ProtocolWriter;
    private var eventReader: ProtocolReader;

    private let stderrTailLock: NSLock;
    private var stderrTailBytes: Array<UInt8>;
    private var stderrDrainGeneration: Int;

    init(
        childProcessIdentifier: pid_t,
        commandWriteFileDescriptor: Int32,
        eventReadFileDescriptor: Int32,
        stderrReadFileDescriptor: Int32,
        commandWriter: ProtocolWriter,
        eventReader: ProtocolReader,
        workerExecutablePath: String,
        workerArguments: Array<String>,
        startupConfiguration: WorkerStartupConfiguration?
    ) {
        self.workerExecutablePath = workerExecutablePath;
        self.workerArguments = workerArguments;
        self.startupConfiguration = startupConfiguration;
        self.workerStartedAt = Date();
        self.lifecycleLock = NSLock();
        self.childProcessIdentifier = childProcessIdentifier;
        self.commandWriteFileDescriptor = commandWriteFileDescriptor;
        self.eventReadFileDescriptor = eventReadFileDescriptor;
        self.stderrReadFileDescriptor = stderrReadFileDescriptor;
        self.hasReapedExit = false;
        self.exitWasSuccessful = false;
        self.escalatedToSignal = false;
        self.adoptedByAnotherLifecycle = false;
        self.startupRuntimeConfigurationAppliedFlag = false;
        self.commandWriter = commandWriter;
        self.eventReader = eventReader;
        self.stderrTailLock = NSLock();
        self.stderrTailBytes = Array<UInt8>();
        self.stderrDrainGeneration = 1;
    }

    deinit {
        // Safety net for an owner that never closed: reap the child if it is
        // still alive, then close whichever transport descriptors remain. A
        // lifecycle adopted by a relaunching owner is skipped entirely.
        if !self.adoptedByAnotherLifecycle {
            if !self.hasReapedExit {
                WorkerProcess.signalProcess(self.childProcessIdentifier, SIGKILL);
                var waitStatus: Int32 = 0;
                _ = waitpid(self.childProcessIdentifier, &waitStatus, 0);
            }
            WorkerProcess.closeDescriptor(self.commandWriteFileDescriptor);
            WorkerProcess.closeDescriptor(self.eventReadFileDescriptor);
            WorkerProcess.closeDescriptor(self.stderrReadFileDescriptor);
        }
    }

    /// Starts the worker executable with piped stdio and a draining stderr tail.
    public static func launch(workerExecutablePath: String) throws -> WorkerProcess {
        return try WorkerProcess.launch(
            workerExecutablePath: workerExecutablePath,
            arguments: Array<String>());
    }

    /// Starts the worker executable with launch arguments, piped stdio, and
    /// a draining stderr tail.
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
    /// followed by a best-effort termination of the just-started child
    /// before the failure surfaces.
    public static func launch(
        workerExecutablePath: String,
        arguments: Array<String>,
        workerStartupConfiguration: WorkerStartupConfiguration?
    ) throws -> WorkerProcess {
        let spawnedWorker: WorkerProcess = try WorkerProcess.launchConfiguredWorker(
            workerExecutablePath: workerExecutablePath,
            workerArguments: arguments,
            workerStartupConfiguration: workerStartupConfiguration);
        spawnedWorker.startStandardErrorDrain();
        return spawnedWorker;
    }

    /// Spawns the worker and delivers the startup policy without starting a
    /// stderr drain: the public launch drains its own child, while a
    /// relaunching owner adopts the descriptors and drains them itself — a
    /// second drain here would race the owner's for the same bytes.
    private static func launchConfiguredWorker(
        workerExecutablePath: String,
        workerArguments: Array<String>,
        workerStartupConfiguration: WorkerStartupConfiguration?
    ) throws -> WorkerProcess {
        let spawnedWorker: WorkerProcess = try WorkerProcess.spawnWorker(
            workerExecutablePath: workerExecutablePath,
            workerArguments: workerArguments,
            workerStartupConfiguration: workerStartupConfiguration);
        if let startupConfiguration: WorkerStartupConfiguration = workerStartupConfiguration {
            do {
                try spawnedWorker.sendCommand(.initializeWorker(startupConfiguration));
            } catch let initializationFailure {
                _ = try? spawnedWorker.close();
                throw initializationFailure;
            }
        }
        return spawnedWorker;
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
        if self.hasLivingProcess() {
            return Int32(self.childProcessIdentifier);
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
        self.halfCloseCommandSide();
        if !self.waitForExit(within: WorkerProcess.SHUTDOWN_TIMEOUT_SECONDS) {
            self.escalateToSignal(SIGTERM);
            if !self.waitForExit(within: WorkerProcess.SHUTDOWN_TIMEOUT_SECONDS) {
                self.escalateToSignal(SIGKILL);
            }
        }
        let exitWasSuccessful: Bool = self.reapExitIfPending();
        self.closeTransportDescriptors();
        let escalatedToSignal: Bool = {
            self.lifecycleLock.lock();
            defer { self.lifecycleLock.unlock(); }
            return self.escalatedToSignal;
        }();
        if escalatedToSignal {
            return .forced(processExitSuccessful: exitWasSuccessful);
        }
        return .graceful(processExitSuccessful: exitWasSuccessful);
    }

    /// Kills the worker outright and reaps it; the containment path for a
    /// worker that can no longer be trusted to answer a graceful close.
    public func forceTerminate() throws -> WorkerTerminationOutcome {
        if let processId: Int32 = self.processId {
            kill(processId, SIGKILL);
        }
        let exitWasSuccessful: Bool = self.reapExitIfPending();
        self.closeTransportDescriptors();
        return .forced(processExitSuccessful: exitWasSuccessful);
    }

    /// Whether the worker process is still alive; reaps a pending exit so
    /// the answer and the recorded outcome stay in sync.
    public func hasLivingProcess() -> Bool {
        self.lifecycleLock.lock();
        defer { self.lifecycleLock.unlock(); }
        if self.hasReapedExit {
            return false;
        }
        var waitStatus: Int32 = 0;
        let waitOutcome: pid_t = waitpid(self.childProcessIdentifier, &waitStatus, WNOHANG);
        if waitOutcome == 0 {
            return true;
        }
        self.recordReapedExit(waitStatus: waitStatus);
        return false;
    }

    /// Starts a clean process from this worker's exact portable launch inputs
    /// after the previous process ended, resetting the once-per-process
    /// startup flag; mirrors the Rust relaunch_after_termination. All
    /// previous descriptors close here, exactly once, before the
    /// replacement allocates its own.
    public func relaunchAfterTermination() throws -> Void {
        self.closeTransportDescriptors();
        let replacementWorker: WorkerProcess = try WorkerProcess.launchConfiguredWorker(
            workerExecutablePath: self.workerExecutablePath,
            workerArguments: self.workerArguments,
            workerStartupConfiguration: self.startupConfiguration);
        self.lifecycleLock.lock();
        self.childProcessIdentifier = replacementWorker.currentChildProcessIdentifier();
        self.commandWriteFileDescriptor = replacementWorker.currentCommandWriteFileDescriptor();
        self.eventReadFileDescriptor = replacementWorker.currentEventReadFileDescriptor();
        self.stderrReadFileDescriptor = replacementWorker.currentStderrReadFileDescriptor();
        self.hasReapedExit = false;
        self.exitWasSuccessful = false;
        self.escalatedToSignal = false;
        self.startupRuntimeConfigurationAppliedFlag = false;
        self.commandWriter = replacementWorker.currentCommandWriter();
        self.eventReader = replacementWorker.currentEventReader();
        self.lifecycleLock.unlock();
        replacementWorker.markAdoptedByAnotherLifecycle();
        self.stderrTailLock.lock();
        self.stderrTailBytes.removeAll();
        self.stderrTailLock.unlock();
        self.startStandardErrorDrain();
    }

    /// Closes every transport descriptor this worker still owns. Each close
    /// is recorded so no descriptor is ever closed twice — the property that
    /// keeps concurrent worker lifecycles from corrupting each other.
    private func closeTransportDescriptors() {
        self.lifecycleLock.lock();
        let commandWriteFileDescriptor: Int32 = self.commandWriteFileDescriptor;
        let eventReadFileDescriptor: Int32 = self.eventReadFileDescriptor;
        let stderrReadFileDescriptor: Int32 = self.stderrReadFileDescriptor;
        self.commandWriteFileDescriptor = -1;
        self.eventReadFileDescriptor = -1;
        self.stderrReadFileDescriptor = -1;
        self.lifecycleLock.unlock();
        WorkerProcess.closeDescriptor(commandWriteFileDescriptor);
        WorkerProcess.closeDescriptor(eventReadFileDescriptor);
        WorkerProcess.closeDescriptor(stderrReadFileDescriptor);
        self.stderrTailLock.lock();
        self.stderrDrainGeneration += 1;
        self.stderrTailLock.unlock();
    }

    /// Half-closes the supervisor's command side: the worker reads EOF on its
    /// stdin, which is the graceful-shutdown signal in the Rust supervisor.
    private func halfCloseCommandSide() {
        self.lifecycleLock.lock();
        let commandWriteFileDescriptor: Int32 = self.commandWriteFileDescriptor;
        self.commandWriteFileDescriptor = -1;
        self.lifecycleLock.unlock();
        WorkerProcess.closeDescriptor(commandWriteFileDescriptor);
    }

    /// Sends one escalation signal to a still-living child.
    private func escalateToSignal(_ signalNumber: Int32) {
        if let processId: Int32 = self.processId {
            kill(processId, signalNumber);
            self.lifecycleLock.lock();
            self.escalatedToSignal = true;
            self.lifecycleLock.unlock();
        }
    }

    /// Waits a bounded time for the child to exit on its own, reaping a
    /// pending exit as soon as it lands.
    private func waitForExit(within timeoutSeconds: TimeInterval) -> Bool {
        let exitDeadline: Date = Date().addingTimeInterval(timeoutSeconds);
        while Date() < exitDeadline {
            if !self.hasLivingProcess() {
                return true;
            }
            Thread.sleep(forTimeInterval: WorkerProcess.EXIT_POLL_INTERVAL_SECONDS);
        }
        return !self.hasLivingProcess();
    }

    /// Reaps the exit if it has not been reaped yet, returning whether the
    /// child reported a successful exit.
    private func reapExitIfPending() -> Bool {
        self.lifecycleLock.lock();
        defer { self.lifecycleLock.unlock(); }
        if self.hasReapedExit {
            return self.exitWasSuccessful;
        }
        var waitStatus: Int32 = 0;
        _ = waitpid(self.childProcessIdentifier, &waitStatus, 0);
        self.recordReapedExit(waitStatus: waitStatus);
        return self.exitWasSuccessful;
    }

    /// Caller holds `lifecycleLock`. The raw wait-status decode mirrors the
    /// C macros: `(status & 0x7F) == 0` is a normal exit and bits 8..15 the
    /// exit code; Swift's Darwin overlay does not export `WIFEXITED`.
    private func recordReapedExit(waitStatus: Int32) {
        self.hasReapedExit = true;
        self.exitWasSuccessful = (waitStatus & 0x7F) == 0 && ((waitStatus >> 8) & 0xFF) == 0;
    }

    private func currentChildProcessIdentifier() -> pid_t {
        return self.childProcessIdentifier;
    }

    private func currentCommandWriteFileDescriptor() -> Int32 {
        return self.commandWriteFileDescriptor;
    }

    private func currentEventReadFileDescriptor() -> Int32 {
        return self.eventReadFileDescriptor;
    }

    private func currentStderrReadFileDescriptor() -> Int32 {
        return self.stderrReadFileDescriptor;
    }

    private func currentCommandWriter() -> ProtocolWriter {
        return self.commandWriter;
    }

    private func currentEventReader() -> ProtocolReader {
        return self.eventReader;
    }

    /// The relaunching owner adopted this child's pid, descriptors, and
    /// transports; the deinit safety net must never touch them.
    private func markAdoptedByAnotherLifecycle() {
        self.lifecycleLock.lock();
        self.adoptedByAnotherLifecycle = true;
        self.childProcessIdentifier = -1;
        self.commandWriteFileDescriptor = -1;
        self.eventReadFileDescriptor = -1;
        self.stderrReadFileDescriptor = -1;
        self.hasReapedExit = true;
        self.lifecycleLock.unlock();
    }

    private func startStandardErrorDrain() {
        self.lifecycleLock.lock();
        let stderrReadFileDescriptor: Int32 = self.stderrReadFileDescriptor;
        self.lifecycleLock.unlock();
        guard stderrReadFileDescriptor >= 0 else {
            return;
        }
        self.stderrTailLock.lock();
        self.stderrDrainGeneration += 1;
        let drainGeneration: Int = self.stderrDrainGeneration;
        self.stderrTailLock.unlock();
        let drainThread: Thread = Thread(block: { [weak self] in
            WorkerProcess.drainStandardError(
                self,
                fileDescriptor: stderrReadFileDescriptor,
                drainGeneration: drainGeneration);
        });
        drainThread.name = "astronomicald-worker-stderr-drain";
        drainThread.start();
    }

    /// The adopted drain generation gate keeps a thread parked on a replaced
    /// child's stderr from appending into the relaunched tail.
    private static func drainStandardError(
        _ owner: WorkerProcess?,
        fileDescriptor: Int32,
        drainGeneration: Int
    ) -> Void {
        var readBuffer: Array<UInt8> = Array(repeating: 0, count: WorkerProcess.STDERR_DRAIN_CHUNK_BYTE_COUNT);
        while true {
            let receivedByteCount: Int = read(fileDescriptor, &readBuffer, readBuffer.count);
            if receivedByteCount <= 0 {
                return;
            }
            guard let strongOwner: WorkerProcess = owner else {
                return;
            }
            strongOwner.stderrTailLock.lock();
            if strongOwner.stderrDrainGeneration == drainGeneration {
                strongOwner.stderrTailBytes.append(contentsOf: readBuffer[0..<receivedByteCount]);
                if strongOwner.stderrTailBytes.count > WorkerProcess.STDERR_TAIL_MAXIMUM_BYTE_COUNT {
                    strongOwner.stderrTailBytes.removeFirst(
                        strongOwner.stderrTailBytes.count - WorkerProcess.STDERR_TAIL_MAXIMUM_BYTE_COUNT);
                }
            }
            strongOwner.stderrTailLock.unlock();
        }
    }
}
