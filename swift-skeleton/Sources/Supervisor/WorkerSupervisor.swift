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

    private static let shutdownDrainWaitSeconds: TimeInterval = 10;
    private static let generationPollSeconds: TimeInterval = 0.25;

    private let stateLock: NSLock;
    private var workerProcess: WorkerProcess?;
    private var eventPump: WorkerEventPump?;
    private var isGenerationAdmitted: Bool;
    private var isShutdownRequested: Bool;
    private let healthState: WorkerHealthState;
    private let modelPolicyCatalog: Dictionary<String, RuntimeModelPolicy>;
    private let workerExecutablePath: String;
    private let workerArguments: Array<String>;
    private let workerStartupConfiguration: WorkerStartupConfiguration?;
    private let modelLoadTimeout: TimeInterval;

    private init(
        workerExecutablePath: String,
        workerArguments: Array<String>,
        workerStartupConfiguration: WorkerStartupConfiguration?,
        modelPolicyCatalog: Dictionary<String, RuntimeModelPolicy>,
        modelLoadTimeout: TimeInterval
    ) {
        self.stateLock = NSLock();
        self.workerProcess = nil;
        self.eventPump = nil;
        self.isGenerationAdmitted = false;
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
    /// Requests are serialized: the first admitted request owns the worker
    /// until its terminal event, model swap included; a concurrent request is
    /// rejected with capacityUnavailable instead of queueing.
    public func startChatGeneration(
        _ generationCommand: ChatGenerationCommand
    ) throws -> Array<ChatGenerationStreamEvent> {
        try self.admitGeneration();
        defer { self.releaseAdmission(); }
        return try self.runGeneration(generationCommand);
    }

    /// Runs one bounded embeddings request to its completed output, swapping
    /// the resident model on demand exactly as chat does.
    public func startEmbeddingsGeneration(
        _ embeddingsCommand: EmbeddingsCommand
    ) throws -> EmbeddingsOutput {
        try self.admitGeneration();
        defer { self.releaseAdmission(); }
        return try self.runEmbeddingsGeneration(embeddingsCommand);
    }

    private func admitGeneration() throws -> Void {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        if self.isShutdownRequested || self.workerProcess == nil {
            throw GenerationStartError.workerUnavailable;
        }
        if self.isGenerationAdmitted {
            throw GenerationStartError.capacityUnavailable;
        }
        self.isGenerationAdmitted = true;
    }

    private func releaseAdmission() -> Void {
        self.stateLock.lock();
        self.isGenerationAdmitted = false;
        self.stateLock.unlock();
    }

    /// Shuts down and reaps the owned inference worker process. A generation
    /// still holding the worker is interrupted through its bounded drain and
    /// waited out before the close, so shutdown never races a live request.
    public func shutdown() throws -> WorkerTerminationOutcome {
        self.stateLock.lock();
        self.isShutdownRequested = true;
        self.stateLock.unlock();
        let drainDeadline: Date = Date().addingTimeInterval(WorkerSupervisor.shutdownDrainWaitSeconds);
        while true {
            self.stateLock.lock();
            let isGenerationAdmitted: Bool = self.isGenerationAdmitted;
            self.stateLock.unlock();
            if !isGenerationAdmitted || Date() >= drainDeadline {
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

    // MARK: - Generation execution

    private func runGeneration(
        _ generationCommand: ChatGenerationCommand
    ) throws -> Array<ChatGenerationStreamEvent> {
        self.stateLock.lock();
        let workerProcess: WorkerProcess? = self.workerProcess;
        let eventPump: WorkerEventPump? = self.eventPump;
        self.stateLock.unlock();
        guard let workerProcess = workerProcess, let eventPump = eventPump else {
            throw GenerationStartError.workerUnavailable;
        }
        do {
            try WorkerStartupRuntime.waitForStartupRuntimeConfiguration(
                workerProcess: workerProcess,
                eventPump: eventPump,
                healthState: self.healthState,
                modelLoadTimeout: self.modelLoadTimeout);
        } catch {
            // The startup acknowledgement never arrived for this process; a
            // fresh attempt is the only recovery.
            self.containAndAttemptRelaunch(controlError: error);
            throw GenerationStartError.workerUnavailable;
        }
        try WorkerGenerate.prepareResidentModel(
            targetModelId: generationCommand.model,
            workerProcess: workerProcess,
            eventPump: eventPump,
            healthState: self.healthState,
            modelPolicyCatalog: self.modelPolicyCatalog,
            modelLoadTimeout: self.modelLoadTimeout,
            containment: { (controlError: Error) -> Void in
                self.containAndAttemptRelaunch(controlError: controlError);
            });
        do {
            try workerProcess.sendCommand(.generate(generationCommand));
        } catch let protocolError as ProtocolError {
            if case let .outgoingMessageTooLarge(actualMessageBytes, maximumMessageBytes) = protocolError {
                throw GenerationStartError.requestTooLarge(
                    actualIpcMessageBytes: actualMessageBytes,
                    maximumIpcMessageBytes: maximumMessageBytes);
            }
            self.containAndAttemptRelaunch(controlError: protocolError);
            throw GenerationStartError.workerUnavailable;
        } catch let sendError {
            self.containAndAttemptRelaunch(controlError: sendError);
            throw GenerationStartError.workerUnavailable;
        }
        do {
            return try self.collectGenerationEvents(
                generationCommand.requestId,
                eventPump: eventPump);
        } catch let controlError as WorkerControlError {
            self.containAndAttemptRelaunch(controlError: controlError);
            throw GenerationStartError.workerUnavailable;
        }
    }

    /// Drains worker events until this request's terminal event, routing
    /// process-scoped events to health state and generation events for other
    /// requests to protocol violations, mirroring the Rust loop's
    /// active-request matching.
    private func collectGenerationEvents(        _ requestId: RequestId,
        eventPump: WorkerEventPump
    ) throws -> Array<ChatGenerationStreamEvent> {
        var streamEvents: Array<ChatGenerationStreamEvent> = Array<ChatGenerationStreamEvent>();
        while true {
            self.stateLock.lock();
            let isShutdownRequested: Bool = self.isShutdownRequested;
            self.stateLock.unlock();
            if isShutdownRequested {
                throw WorkerControlError.workerEventStreamClosed;
            }
            guard let workerEvent: WorkerEvent = try eventPump.nextEvent(
                within: WorkerSupervisor.generationPollSeconds) else {
                continue;
            }
            switch (workerEvent) {
            case let .output(eventRequestId, _, _, outputs, _, _):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                for workerOutput: ChatGenerationOutput in outputs {
                    streamEvents.append(ChatGenerationStreamEvent.fromWorkerOutput(workerOutput));
                }
            case let .prefillProgress(
                eventRequestId,
                _,
                processedTokens,
                totalTokens,
                elapsedMillis,
                forwardPrefillChunkElapsedMillis,
                completedPrefillChunkTokens,
                mlxMemorySnapshot,
                _):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                streamEvents.append(.prefillProgress(
                    processedTokens: processedTokens,
                    totalTokens: totalTokens,
                    elapsedMillis: elapsedMillis,
                    forwardPrefillChunkElapsedMillis: forwardPrefillChunkElapsedMillis,
                    completedPrefillChunkTokens: completedPrefillChunkTokens,
                    mlxActiveMemoryBytes: mlxMemorySnapshot?.activeMemoryBytes,
                    mlxAllocatorCacheMemoryBytes: mlxMemorySnapshot?.allocatorCacheMemoryBytes,
                    mlxPeakMemoryBytes: mlxMemorySnapshot?.peakMemoryBytes));
            case let .completed(
                eventRequestId,
                promptTokenCount,
                generatedTokenCount,
                reasoningTokenCount,
                cachedTokenCount,
                _,
                reason):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                streamEvents.append(.completed(
                    promptTokenCount: promptTokenCount,
                    generatedTokenCount: generatedTokenCount,
                    reasoningTokenCount: reasoningTokenCount,
                    cachedTokenCount: cachedTokenCount,
                    reason: reason));
                return streamEvents;
            case let .failed(eventRequestId, reason):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                streamEvents.append(.failed(reason: reason));
                return streamEvents;
            case .generationPreparationStarted, .generationProgress, .firstDecodeCompleted,
                 .promptWorkReuse, .generationFinalized:
                // Request-scoped telemetry without CLI presentation; the
                // attribution slice consumes it from health state later.
                continue;
            default:
                try WorkerEventHandler.handle(workerEvent, healthState: self.healthState);
            }
        }
    }

    /// Runs one embeddings command through the worker: startup wait, model
    /// swap, command send, and the completed output collection, mirroring the
    /// chat generation path's containment.
    private func runEmbeddingsGeneration(
        _ embeddingsCommand: EmbeddingsCommand
    ) throws -> EmbeddingsOutput {
        self.stateLock.lock();
        let workerProcess: WorkerProcess? = self.workerProcess;
        let eventPump: WorkerEventPump? = self.eventPump;
        self.stateLock.unlock();
        guard let workerProcess = workerProcess, let eventPump = eventPump else {
            throw GenerationStartError.workerUnavailable;
        }
        do {
            try WorkerStartupRuntime.waitForStartupRuntimeConfiguration(
                workerProcess: workerProcess,
                eventPump: eventPump,
                healthState: self.healthState,
                modelLoadTimeout: self.modelLoadTimeout);
        } catch {
            // The startup acknowledgement never arrived for this process; a
            // fresh attempt is the only recovery.
            self.containAndAttemptRelaunch(controlError: error);
            throw GenerationStartError.workerUnavailable;
        }
        try WorkerGenerate.prepareResidentModel(
            targetModelId: embeddingsCommand.model,
            workerProcess: workerProcess,
            eventPump: eventPump,
            healthState: self.healthState,
            modelPolicyCatalog: self.modelPolicyCatalog,
            modelLoadTimeout: self.modelLoadTimeout,
            containment: { (controlError: Error) -> Void in
                self.containAndAttemptRelaunch(controlError: controlError);
            });
        do {
            try workerProcess.sendCommand(.generateEmbeddings(embeddingsCommand));
        } catch let protocolError as ProtocolError {
            if case let .outgoingMessageTooLarge(actualMessageBytes, maximumMessageBytes) = protocolError {
                throw GenerationStartError.requestTooLarge(
                    actualIpcMessageBytes: actualMessageBytes,
                    maximumIpcMessageBytes: maximumMessageBytes);
            }
            self.containAndAttemptRelaunch(controlError: protocolError);
            throw GenerationStartError.workerUnavailable;
        } catch let sendError {
            self.containAndAttemptRelaunch(controlError: sendError);
            throw GenerationStartError.workerUnavailable;
        }
        do {
            return try self.collectEmbeddingsEvents(
                embeddingsCommand.requestId,
                eventPump: eventPump);
        } catch let controlError as WorkerControlError {
            self.containAndAttemptRelaunch(controlError: controlError);
            throw EmbeddingsExecutionError.workerUnavailable;
        }
    }

    /// Drains worker events until this embeddings request completes or fails,
    /// applying the same active-request matching as the chat collection.
    private func collectEmbeddingsEvents(
        _ requestId: RequestId,
        eventPump: WorkerEventPump
    ) throws -> EmbeddingsOutput {
        while true {
            self.stateLock.lock();
            let isShutdownRequested: Bool = self.isShutdownRequested;
            self.stateLock.unlock();
            if isShutdownRequested {
                throw WorkerControlError.workerEventStreamClosed;
            }
            guard let workerEvent: WorkerEvent = try eventPump.nextEvent(
                within: WorkerSupervisor.generationPollSeconds) else {
                continue;
            }
            switch (workerEvent) {
            case let .embeddingsCompleted(
                eventRequestId, embeddings, inputTokenCounts, _):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                return EmbeddingsOutput(
                    embeddings: embeddings,
                    inputTokenCounts: inputTokenCounts);
            case let .embeddingsFailed(eventRequestId, reason):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                throw EmbeddingsExecutionError.workerFailure(reason);
            case .embeddingsFinalized:
                // The request-scoped release acknowledgement; the outcome
                // event always arrives first.
                continue;
            case let .output(eventRequestId, _, _, _, _, _):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                throw WorkerControlError.workerProtocolViolation(
                    description: "the embeddings request received a chat output frame");
            case let .completed(eventRequestId, _, _, _, _, _, _):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                throw WorkerControlError.workerProtocolViolation(
                    description: "the embeddings request received a chat terminal frame");
            case let .failed(eventRequestId, _):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                throw WorkerControlError.workerProtocolViolation(
                    description: "the embeddings request received a chat terminal frame");
            case .generationPreparationStarted, .generationProgress, .firstDecodeCompleted,
                 .promptWorkReuse, .generationFinalized, .prefillProgress:
                // Request-scoped telemetry without endpoint presentation.
                continue;
            default:
                try WorkerEventHandler.handle(workerEvent, healthState: self.healthState);
            }
        }
    }

    private static func requireActiveRequest(
        _ eventRequestId: RequestId,
        requestId: RequestId
    ) throws -> Void {
        if eventRequestId != requestId {
            throw WorkerControlError.workerProtocolViolation(
                description: "generation event \(eventRequestId.value()) does not belong to the active request");
        }
    }

    // MARK: - Containment

    /// Terminates an untrusted worker, then brings a replacement up so the
    /// daemon keeps serving, unless shutdown already claimed the handle.
    private func containAndAttemptRelaunch(controlError: Error) -> Void {
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
