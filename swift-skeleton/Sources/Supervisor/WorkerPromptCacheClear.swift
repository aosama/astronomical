import Foundation;

import IpcProtocol;

/// Result returned to the HTTP request that submitted one cache clear,
/// mirroring worker_cache_clear.rs's PromptCacheClearOutcome.
public enum PromptCacheClearOutcome: Equatable, Sendable {

    /// The worker deleted the cache content and reported the deletion.
    case applied(modelId: String?, blocksRemoved: UInt64, bytesFreed: UInt64);
    /// A generation holds or waits for the worker; the clear waits behind it.
    case queued;
}

/// A cache clear that waits because generations hold or wait for the worker.
public struct PendingPromptCacheClear: Equatable, Sendable {

    public let modelId: String?;
}

/// The cache-clear control surface the REST endpoint drives, so journeys and
/// the daemon can pass the real supervisor or a scripted double alike.
public protocol PromptCacheClearControlling: Sendable {

    func clearPromptCache(modelId: String?) throws -> PromptCacheClearOutcome;
}

/// The supervisor's side of the SSD prompt-cache clear, migrating
/// apps/supervisor/src/worker_cache_clear.rs: an idle worker clears
/// synchronously under a bounded acknowledgement wait, a busy worker takes
/// only the newest queued clear, and the queued clear applies once the
/// active generation finished and no queued waiters remain.
extension WorkerSupervisor {

    /// Queues a busy clear or applies it synchronously while the worker is
    /// idle, mirroring WorkerHandle::clear_prompt_cache.
    public func clearPromptCache(modelId: String?) throws -> PromptCacheClearOutcome {
        self.stateLock.lock();
        guard let workerProcess: WorkerProcess = self.workerProcess,
              let eventPump: WorkerEventPump = self.eventPump else {
            self.stateLock.unlock();
            throw WorkerControlError.missingActiveWorker;
        }
        let isGenerationControlIdle: Bool = self.issuedAdmissionTicketCount == self.servedAdmissionTicket;
        if !isGenerationControlIdle {
            self.pendingPromptCacheClear = PendingPromptCacheClear(modelId: modelId);
            self.stateLock.unlock();
            try self.healthState.apply({ (snapshot: inout WorkerHealthSnapshot) in
                snapshot.pendingPromptCacheClear = PendingPromptCacheClear(modelId: modelId);
            });
            return .queued;
        }
        self.stateLock.unlock();
        return try self.applyPromptCacheClear(
            modelId: modelId,
            workerProcess: workerProcess,
            eventPump: eventPump);
    }

    /// Applies the newest queued clear once the active generation finished
    /// and no queued waiters remain, so the deletion never lands under a
    /// resident prompt-cache consumer.
    func applyPendingPromptCacheClearIfIdle() -> Void {
        self.stateLock.lock();
        let isGenerationControlIdle: Bool = self.issuedAdmissionTicketCount == self.servedAdmissionTicket;
        let pendingClear: PendingPromptCacheClear? = self.pendingPromptCacheClear;
        let workerProcess: WorkerProcess? = self.workerProcess;
        let eventPump: WorkerEventPump? = self.eventPump;
        self.stateLock.unlock();
        guard isGenerationControlIdle, let pendingClear = pendingClear,
              let workerProcess = workerProcess, let eventPump = eventPump else {
            return;
        }
        try? self.healthState.apply({ (snapshot: inout WorkerHealthSnapshot) in
            snapshot.pendingPromptCacheClear = nil;
        });
        _ = try? self.applyPromptCacheClear(
            modelId: pendingClear.modelId,
            workerProcess: workerProcess,
            eventPump: eventPump);
    }

    /// Sends one clear command and pumps events until this scope's
    /// acknowledgement, containing the worker on timeout exactly as
    /// apply_prompt_cache_clear does.
    func applyPromptCacheClear(
        modelId: String?,
        workerProcess: WorkerProcess,
        eventPump: WorkerEventPump
    ) throws -> PromptCacheClearOutcome {
        try workerProcess.sendCommand(.clearPromptCache(modelId: modelId));
        let clearDeadline: Date = Date().addingTimeInterval(WorkerSupervisor.promptCacheClearTimeoutSeconds);
        while true {
            self.stateLock.lock();
            let isShutdownRequested: Bool = self.isShutdownRequested;
            self.stateLock.unlock();
            if isShutdownRequested {
                throw WorkerControlError.workerEventStreamClosed;
            }
            guard let workerEvent: WorkerEvent = try eventPump.nextEvent(
                within: WorkerSupervisor.generationPollSeconds) else {
                if Date() >= clearDeadline {
                    let timeoutError: WorkerControlError = WorkerControlError.promptCacheClearTimeout(
                        cacheClearTimeoutMillis: UInt64(
                            WorkerSupervisor.promptCacheClearTimeoutSeconds * 1000));
                    self.containAndAttemptRelaunch(controlError: timeoutError);
                    throw WorkerControlError.missingActiveWorker;
                }
                continue;
            }
            switch (workerEvent) {
            case let .promptCacheCleared(clearedModelId, blocksRemoved, bytesFreed):
                if clearedModelId != modelId {
                    let scopeError: WorkerControlError = WorkerControlError.workerProtocolViolation(
                        description: "prompt-cache clear acknowledgement scope mismatch");
                    self.containAndAttemptRelaunch(controlError: scopeError);
                    throw WorkerControlError.missingActiveWorker;
                }
                return .applied(
                    modelId: clearedModelId,
                    blocksRemoved: blocksRemoved,
                    bytesFreed: bytesFreed);
            default:
                try WorkerEventHandler.handle(workerEvent, healthState: self.healthState);
            }
        }
    }
}

extension WorkerSupervisor: PromptCacheClearControlling {}
