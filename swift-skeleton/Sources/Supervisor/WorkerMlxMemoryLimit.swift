import Foundation;

import IpcProtocol;

/// Completion state of one supervisor-side MLX memory-ceiling request,
/// mirroring apps/supervisor/src/worker_memory_limit.rs's
/// MlxMemoryLimitUpdateOutcome.
public enum MlxMemoryLimitUpdateOutcome: Equatable {

    /// The worker applied the new ceiling immediately.
    case applied;
    /// A generation is active; the ceiling raise waits behind it.
    case queued;
    /// The worker refused the ceiling without mutating its accounting.
    case rejected;
}

/// A live ceiling raise that waits because a generation is already running.
struct PendingMlxMemoryLimitUpdate {

    let effectiveMlxMemoryCeilingBytes: UInt64;
    let configurationGeneration: String;
}

/// The supervisor's side of the live MLX memory-ceiling control: immediate
/// application on an idle worker, queueing behind an active generation with
/// the pending ceiling published to health state, and application of the
/// queued raise during the admission-slot release so the next request starts
/// against the new ceiling (issue #515's guarantee).
extension WorkerSupervisor {

    /// Whether the worker accepts a control action right now: no admission
    /// ticket is outstanding, so no generation is active or queued,
    /// mirroring WorkerHandle::is_generation_idle_for_control_action.
    public func isGenerationIdleForControlAction() -> Bool {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        return self.issuedAdmissionTicketCount == self.servedAdmissionTicket;
    }

    /// Stages generation attribution before a memory command can race to
    /// acknowledgement, mirroring
    /// WorkerHandle::stage_memory_configuration_generation.
    public func stageMemoryConfigurationGeneration(_ configurationGeneration: String) -> Void {
        try? self.healthState.apply({ (snapshot: inout WorkerHealthSnapshot) in
            snapshot.pendingConfigurationGeneration = configurationGeneration;
        });
    }

    /// Finalizes generation attribution after the memory control outcome is
    /// known, mirroring
    /// WorkerHandle::record_memory_configuration_generation. A queued raise
    /// keeps its staged generation until the deferred application lands.
    public func recordMemoryConfigurationGeneration(
        _ configurationGeneration: String,
        _ updateOutcome: MlxMemoryLimitUpdateOutcome
    ) -> Void {
        try? self.healthState.apply({ (snapshot: inout WorkerHealthSnapshot) in
            if updateOutcome == .applied {
                if let acknowledgedConfiguration: WorkerRuntimeFeatureConfiguration = snapshot.workerRuntimeFeatureConfiguration {
                    snapshot.workerRuntimeFeatureConfiguration = WorkerRuntimeFeatureConfiguration(
                        configurationGeneration: configurationGeneration,
                        persistentPromptCacheEnabled: acknowledgedConfiguration.persistentPromptCacheEnabled,
                        promptCacheMaximumSizeBytes: acknowledgedConfiguration.promptCacheMaximumSizeBytes,
                        loadedModel: acknowledgedConfiguration.loadedModel);
                }
                snapshot.pendingConfigurationGeneration = nil;
            } else if updateOutcome == .rejected {
                snapshot.pendingConfigurationGeneration = nil;
            }
        });
    }

    /// Applies an idle MLX ceiling immediately or queues it behind one
    /// active generation, mirroring WorkerHandle::update_mlx_memory_limit.
    ///
    /// The idle-application window races a concurrent admission only in the
    /// favorable direction: the worker itself refuses ceiling changes during
    /// a generation, so the worst case is a `rejected` outcome, never a
    /// mutated ceiling under a running request.
    public func updateMlxMemoryLimit(
        _ effectiveMlxMemoryCeilingBytes: UInt64,
        configurationGeneration: String
    ) throws -> MlxMemoryLimitUpdateOutcome {
        self.stateLock.lock();
        guard let workerProcess: WorkerProcess = self.workerProcess,
              let eventPump: WorkerEventPump = self.eventPump else {
            self.stateLock.unlock();
            throw WorkerControlError.missingActiveWorker;
        }
        let hasActiveGenerationSlot: Bool = self.issuedAdmissionTicketCount > self.servedAdmissionTicket;
        if hasActiveGenerationSlot {
            self.pendingMlxMemoryLimitUpdate = PendingMlxMemoryLimitUpdate(
                effectiveMlxMemoryCeilingBytes: effectiveMlxMemoryCeilingBytes,
                configurationGeneration: configurationGeneration);
            self.stateLock.unlock();
            try self.healthState.apply({ (snapshot: inout WorkerHealthSnapshot) in
                snapshot.pendingMlxMemoryCeilingBytes = effectiveMlxMemoryCeilingBytes;
            });
            return .queued;
        }
        self.stateLock.unlock();
        do {
            return try self.applyMlxMemoryLimitUpdate(
                PendingMlxMemoryLimitUpdate(
                    effectiveMlxMemoryCeilingBytes: effectiveMlxMemoryCeilingBytes,
                    configurationGeneration: configurationGeneration),
                workerProcess: workerProcess,
                eventPump: eventPump)
        } catch let controlError {
            // An unacknowledged or failed ceiling change cannot trust the
            // worker anymore: contain it exactly as the Rust handle does
            // before the typed error surfaces.
            self.containWorkerFailure(controlError: controlError)
            throw controlError
        }
    }

    /// Applies the raise a finished generation left queued, so the release of
    /// the admission slot never hands the worker to the next request against
    /// a ceiling the user already replaced.
    func applyPendingMlxMemoryLimitUpdateAfterFinalization() -> Void {
        self.stateLock.lock();
        let pendingUpdate: PendingMlxMemoryLimitUpdate? = self.pendingMlxMemoryLimitUpdate;
        self.pendingMlxMemoryLimitUpdate = nil;
        let isShutdownRequested: Bool = self.isShutdownRequested;
        let workerProcess: WorkerProcess? = self.workerProcess;
        let eventPump: WorkerEventPump? = self.eventPump;
        self.stateLock.unlock();
        guard let pendingUpdate = pendingUpdate, !isShutdownRequested,
              let workerProcess = workerProcess, let eventPump = eventPump else {
            return;
        }
        do {
            _ = try self.applyMlxMemoryLimitUpdate(
                pendingUpdate,
                workerProcess: workerProcess,
                eventPump: eventPump);
        } catch let controlError {
            self.containWorkerFailure(controlError: controlError);
        }
    }

    /// Sends one ceiling command and pumps events until the worker
    /// acknowledges it, publishing the outcome to health state and bounding
    /// the wait exactly as worker_memory_limit.rs's apply_mlx_memory_limit.
    func applyMlxMemoryLimitUpdate(
        _ pendingUpdate: PendingMlxMemoryLimitUpdate,
        workerProcess: WorkerProcess,
        eventPump: WorkerEventPump
    ) throws -> MlxMemoryLimitUpdateOutcome {
        try workerProcess.sendCommand(.updateMlxMemoryLimit(
            effectiveMlxMemoryCeilingBytes: pendingUpdate.effectiveMlxMemoryCeilingBytes,
            configurationGeneration: pendingUpdate.configurationGeneration));
        let updateDeadline: Date = Date().addingTimeInterval(self.modelLoadTimeout);
        while true {
            self.stateLock.lock();
            let isShutdownRequested: Bool = self.isShutdownRequested;
            self.stateLock.unlock();
            if isShutdownRequested {
                throw WorkerControlError.workerEventStreamClosed;
            }
            guard let workerEvent: WorkerEvent = try eventPump.nextEvent(
                within: WorkerSupervisor.generationPollSeconds) else {
                if Date() >= updateDeadline {
                    throw WorkerControlError.mlxMemoryLimitUpdateTimeout(
                        memoryLimitUpdateTimeoutMillis: UInt64(self.modelLoadTimeout * 1000));
                }
                continue;
            }
            switch (workerEvent) {
            case .mlxMemoryLimitChanged, .mlxMemoryLimitRejected:
                try WorkerEventHandler.handle(workerEvent, healthState: self.healthState);
                if case .mlxMemoryLimitChanged = workerEvent {
                    return .applied;
                }
                return .rejected;
            default:
                try WorkerEventHandler.handle(workerEvent, healthState: self.healthState);
            }
        }
    }
}
