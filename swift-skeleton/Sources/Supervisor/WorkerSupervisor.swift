import Foundation;

import IpcProtocol;

/// Supervisor-side owner of the one local inference-worker process.
///
/// Migrates the synchronous core of apps/supervisor/src/worker_handle.rs +
/// worker.rs + worker_generate.rs: launch with the resolved startup policy,
/// serialize chat execution (one generation at a time, swapped onto the
/// requested model on demand), route worker events to health state, contain
/// failed workers, and shut down cleanly. Rust spreads these across an async
/// command loop; the synchronous Swift serving model runs each operation on
/// the caller's thread and guards the small mutable state with one lock.
///
/// The Rust FIFO queue depth (eight waiters behind one active request) is a
/// REST-surface contract and lands with the REST chat endpoint slice; this
/// boundary rejects a second concurrent request with capacityUnavailable.
public final class WorkerSupervisor: @unchecked Sendable, ChatGenerationExecuting {

    static let shutdownDrainWaitSeconds: TimeInterval = 10;
    static let promptCacheClearTimeoutSeconds: TimeInterval = 60;
    static let generationPollSeconds: TimeInterval = 0.25;

    let stateLock: NSCondition;
    var workerProcess: WorkerProcess?;
    var eventPump: WorkerEventPump?;
    var issuedAdmissionTicketCount: Int;
    var servedAdmissionTicket: Int;
    var abandonedAdmissionTickets: Set<Int>;
    var pendingMlxMemoryLimitUpdate: PendingMlxMemoryLimitUpdate?;
    var pendingPromptCacheClear: PendingPromptCacheClear?;
    var isShutdownRequested: Bool;
    let healthState: WorkerHealthState;
    let modelPolicyCatalog: Dictionary<String, RuntimeModelPolicy>;
    let workerExecutablePath: String;
    let workerArguments: Array<String>;
    let workerStartupConfiguration: WorkerStartupConfiguration?;
    let modelLoadTimeout: TimeInterval;

    private init(
        workerExecutablePath: String,
        workerArguments: Array<String>,
        workerStartupConfiguration: WorkerStartupConfiguration?,
        modelPolicyCatalog: Dictionary<String, RuntimeModelPolicy>,
        modelLoadTimeout: TimeInterval
    ) {
        self.stateLock = NSCondition();
        self.workerProcess = nil;
        self.eventPump = nil;
        self.issuedAdmissionTicketCount = 0;
        self.servedAdmissionTicket = 0;
        self.abandonedAdmissionTickets = Set<Int>();
        self.pendingMlxMemoryLimitUpdate = nil;
        self.pendingPromptCacheClear = nil;
        self.isShutdownRequested = false;
        self.healthState = WorkerHealthState();
        self.modelPolicyCatalog = modelPolicyCatalog;
        self.workerExecutablePath = workerExecutablePath;
        self.workerArguments = workerArguments;
        self.workerStartupConfiguration = workerStartupConfiguration;
        self.modelLoadTimeout = modelLoadTimeout;
    }

    /// Creates a handle that reports an unavailable worker without starting a
    /// process, mirroring WorkerHandle::unavailable; the daemon degrades to
    /// this when the spawn fails instead of refusing to start.
    public static func unavailable(
        modelPolicyCatalog: Dictionary<String, RuntimeModelPolicy>
    ) -> WorkerSupervisor {
        return WorkerSupervisor(
            workerExecutablePath: "",
            workerArguments: [],
            workerStartupConfiguration: nil,
            modelPolicyCatalog: modelPolicyCatalog,
            modelLoadTimeout: 0);
    }

    /// Spawns the worker with the supervisor-resolved bootstrap settings and
    /// waits a bounded time for its readiness and runtime-policy
    /// acknowledgement. A failure contains the just-started child before the
    /// error surfaces.
    public static func launch(
        workerExecutablePath: String,
        workerArguments: Array<String>,
        workerStartupConfiguration: WorkerStartupConfiguration?,
        modelPolicyCatalog: Dictionary<String, RuntimeModelPolicy>,
        modelLoadTimeout: TimeInterval
    ) throws -> WorkerSupervisor {
        let supervisor: WorkerSupervisor = WorkerSupervisor(
            workerExecutablePath: workerExecutablePath,
            workerArguments: workerArguments,
            workerStartupConfiguration: workerStartupConfiguration,
            modelPolicyCatalog: modelPolicyCatalog,
            modelLoadTimeout: modelLoadTimeout);
        let launchedWorker: WorkerProcess;
        do {
            launchedWorker = try WorkerProcess.launch(
                workerExecutablePath: workerExecutablePath,
                arguments: workerArguments,
                workerStartupConfiguration: workerStartupConfiguration);
        } catch {
            supervisor.healthState.publish(.unavailable(.unavailable));
            throw error;
        }
        supervisor.stateLock.lock();
        supervisor.workerProcess = launchedWorker;
        supervisor.stateLock.unlock();
        let eventPump: WorkerEventPump = WorkerEventPump(workerProcess: launchedWorker);
        supervisor.stateLock.lock();
        supervisor.eventPump = eventPump;
        supervisor.stateLock.unlock();
        do {
            try WorkerStartupRuntime.waitForStartupRuntimeConfiguration(
                workerProcess: launchedWorker,
                eventPump: eventPump,
                healthState: supervisor.healthState,
                modelLoadTimeout: modelLoadTimeout);
        } catch {
            supervisor.containAndAttemptRelaunch(controlError: error);
            throw error;
        }
        return supervisor;
    }

    /// The current worker health snapshot.
    public func workerHealthSnapshot() -> WorkerHealthSnapshot {
        return self.healthState.currentSnapshot();
    }

    /// The health state this supervisor owns and publishes into. The daemon
    /// hands it to the REST routes and the IPC status verb so every surface
    /// reads the same worker facts.
    public func ownedHealthState() -> WorkerHealthState {
        return self.healthState;
    }

    /// Whether a living worker owns this handle.
    public func isAvailable() -> Bool {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        return self.workerProcess != nil && !self.isShutdownRequested;
    }

    /// Runs one bounded chat generation to its terminal event.
    ///
    /// Requests are serialized through the bounded FIFO admission queue: the
    /// first request owns the worker until its terminal event, model swap
    /// included; further requests wait in the queue while it has room and
    /// are rejected with capacityUnavailable once it is full.
    public func startChatGeneration(
        _ generationCommand: ChatGenerationCommand
    ) throws -> Array<ChatGenerationStreamEvent> {
        try self.admitGenerationSlot();
        defer { self.finishAdmissionSlot(); }
        return try self.runGeneration(generationCommand);
    }

    /// Runs one bounded embeddings request to its completed output, swapping
    /// the resident model on demand exactly as chat does.
    public func startEmbeddingsGeneration(
        _ embeddingsCommand: EmbeddingsCommand
    ) throws -> EmbeddingsOutput {
        try self.admitGenerationSlot();
        defer { self.finishAdmissionSlot(); }
        return try self.runEmbeddingsGeneration(embeddingsCommand);
    }

    /// Runs one bounded image request to its completed output under the
    /// execution and progress-stall bounds the Rust executor enforces.
    public func startImageGeneration(
        _ imageGenerationCommand: ImageGenerationCommand
    ) throws -> ImageGenerationOutput {
        try self.admitGenerationSlot();
        defer { self.finishAdmissionSlot(); }
        return try self.runImageGeneration(
            imageGenerationCommand,
            timeouts: ImageGenerationTimeouts.default);
    }

    /// Shuts down and reaps the owned inference worker process. Queued
    /// waiters wake and abandon their tickets; a generation still holding
    /// the worker is interrupted through its bounded drain and waited out
    /// before the close, so shutdown never races a live request.
    public func shutdown() throws -> WorkerTerminationOutcome {
        self.stateLock.lock();
        self.isShutdownRequested = true;
        self.stateLock.broadcast();
        self.stateLock.unlock();
        let drainDeadline: Date = Date().addingTimeInterval(WorkerSupervisor.shutdownDrainWaitSeconds);
        while true {
            self.stateLock.lock();
            let hasOutstandingAdmissionTicket: Bool =
                self.issuedAdmissionTicketCount > self.servedAdmissionTicket;
            self.stateLock.unlock();
            if !hasOutstandingAdmissionTicket || Date() >= drainDeadline {
                break;
            }
            Thread.sleep(forTimeInterval: 0.05);
        }
        self.stateLock.lock();
        let workerProcess: WorkerProcess? = self.workerProcess;
        self.workerProcess = nil;
        self.eventPump = nil;
        self.stateLock.unlock();
        guard let workerProcess = workerProcess else {
            return .graceful(processExitSuccessful: true);
        }
        return try WorkerLifecycle.shutdown(workerProcess: workerProcess, healthState: self.healthState);
    }

    // MARK: - Containment

    /// Terminates an untrusted worker, then brings a replacement up so the
    /// daemon keeps serving, unless shutdown already claimed the handle.
    func containAndAttemptRelaunch(controlError: Error) -> Void {
        self.stateLock.lock();
        let workerProcess: WorkerProcess? = self.workerProcess;
        let eventPump: WorkerEventPump? = self.eventPump;
        self.stateLock.unlock();
        guard let workerProcess = workerProcess, let eventPump = eventPump else {
            return;
        }
        WorkerLifecycle.containFailure(
            workerProcess: workerProcess,
            healthState: self.healthState,
            operationFailure: controlError);
        self.stateLock.lock();
        let isShutdownRequested: Bool = self.isShutdownRequested;
        self.stateLock.unlock();
        if isShutdownRequested {
            return;
        }
        do {
            try WorkerLifecycle.relaunchAfterFailure(
                workerProcess: workerProcess,
                eventPump: eventPump,
                healthState: self.healthState,
                recoveryAcknowledgementTimeout: self.modelLoadTimeout);
        } catch let relaunchError {
            WorkerLifecycle.containFailure(
                workerProcess: workerProcess,
                healthState: self.healthState,
                operationFailure: relaunchError);
        }
    }
}
