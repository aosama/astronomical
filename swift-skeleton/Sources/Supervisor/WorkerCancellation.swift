import Foundation;

import IpcProtocol;

/// The cancellation path of an abandoned generation, migrating
/// apps/supervisor/src/worker_containment.rs's cancel_active_generation and
/// cancel_worker_request: the client stopped consuming, so the supervisor
/// cancels the in-flight request, publishes whatever process telemetry the
/// worker still offers, and — only when the worker cannot acknowledge the
/// cancellation — terminates it and relaunches a replacement from the same
/// launch inputs so serving continues.
extension WorkerSupervisor {

    /// The number of cancellation-driven containments this supervisor has
    /// driven; journeys assert this monotonic counter because the
    /// loading-window between containment and relaunch can complete between
    /// two health polls.
    var observedCancellationContainmentCount: Int {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        return self.cancellationContainmentCount;
    }

    /// Cancels one abandoned request and restores a trusted worker: a
    /// responsive worker stays ready, an unacknowledging or misbehaving one
    /// is terminated and replaced. Health ends Ready or Unavailable, never
    /// silently Loading.
    func cancelActiveGeneration(
        requestId: RequestId,
        workerProcess: WorkerProcess,
        eventPump: WorkerEventPump,
        expectsImageFinalization: Bool
    ) -> Void {
        self.publishServingActivity(.idle, progress: nil);
        do {
            try WorkerCancellation.drainCancellationAcknowledgement(
                requestId: requestId,
                workerProcess: workerProcess,
                eventPump: eventPump,
                healthState: self.healthState,
                cancellationAcknowledgementTimeout: self.cancellationAcknowledgementTimeout,
                expectsImageFinalization: expectsImageFinalization);
            return;
        } catch let cancellationError {
            self.stateLock.lock();
            self.cancellationContainmentCount += 1;
            self.stateLock.unlock();
            WorkerLifecycle.writeContainmentLine(
                "worker cancellation failed; replacing worker: \(WorkerControlError.describe(cancellationError))");
            self.healthState.publish(.unavailable(.loading));
            do {
                _ = try workerProcess.forceTerminate();
            } catch let terminationError {
                WorkerLifecycle.writeContainmentLine(
                    "failed to terminate worker after cancellation failure: \(terminationError)");
                self.healthState.publish(.unavailable(.unavailable));
                return;
            }
            do {
                try WorkerLifecycle.relaunchAfterFailure(
                    workerProcess: workerProcess,
                    eventPump: eventPump,
                    healthState: self.healthState,
                    recoveryAcknowledgementTimeout: self.modelLoadTimeout);
            } catch let recoveryError {
                WorkerLifecycle.writeContainmentLine(
                    "failed to replace worker after cancellation containment: \(recoveryError)");
                _ = try? workerProcess.forceTerminate();
                self.healthState.publish(.unavailable(.unavailable));
            }
        }
    }
}

/// The bounded drain that watches one cancellation through to its
/// acknowledgement, publishing process-scoped telemetry observed on the way.
enum WorkerCancellation {

    /// Sends the cancel command and drains worker events until this
    /// request's terminal acknowledgement, mirroring cancel_worker_request:
    /// same-request progress is ignored, process telemetry is published, any
    /// other event is a protocol breach that fails the cancellation.
    static func drainCancellationAcknowledgement(
        requestId: RequestId,
        workerProcess: WorkerProcess,
        eventPump: WorkerEventPump,
        healthState: WorkerHealthState,
        cancellationAcknowledgementTimeout: TimeInterval,
        expectsImageFinalization: Bool
    ) throws -> Void {
        try workerProcess.sendCommand(.cancel(requestId: requestId));
        let acknowledgementDeadline: Date = Date().addingTimeInterval(cancellationAcknowledgementTimeout);
        while true {
            let remainingWait: TimeInterval = acknowledgementDeadline.timeIntervalSinceNow;
            if remainingWait <= 0 {
                throw WorkerControlError.cancellationAckTimeout(
                    cancellationTimeoutMillis: UInt64(
                        (cancellationAcknowledgementTimeout * 1000).rounded()));
            }
            guard let workerEvent: WorkerEvent = try eventPump.nextEvent(within: remainingWait) else {
                throw WorkerControlError.cancellationAckTimeout(
                    cancellationTimeoutMillis: UInt64(
                        (cancellationAcknowledgementTimeout * 1000).rounded()));
            }
            switch (workerEvent) {
            case let .output(eventRequestId, _, _, _, _, _)
                where eventRequestId == requestId:
                continue;
            case let .prefillProgress(eventRequestId, _, _, _, _, _, _, _, _)
                where eventRequestId == requestId:
                continue;
            case let .generationPreparationStarted(eventRequestId, _, _, _, _)
                where eventRequestId == requestId:
                continue;
            case let .generationProgress(eventRequestId, _, _, _, _, _)
                where eventRequestId == requestId:
                continue;
            case let .firstDecodeCompleted(eventRequestId, _)
                where eventRequestId == requestId:
                continue;
            case let .promptWorkReuse(eventRequestId, _)
                where eventRequestId == requestId:
                continue;
            case let .imageGenerationProgress(eventRequestId, _, _, _, _, _)
                where expectsImageFinalization && eventRequestId == requestId:
                continue;
            case let .imageGenerationCompleted(eventRequestId, _, _)
                where expectsImageFinalization && eventRequestId == requestId:
                continue;
            case let .imageGenerationFailed(eventRequestId, _)
                where expectsImageFinalization && eventRequestId == requestId:
                continue;
            case let .imageGenerationFinalized(eventRequestId, _, mlxMemorySnapshot):
                guard expectsImageFinalization && eventRequestId == requestId else {
                    throw WorkerControlError.unexpectedCancellationEvent(
                        requestId: requestId.value(),
                        unexpectedWorkerEventSummary: workerEvent.diagnosticSummary());
                }
                if let mlxMemorySnapshot = mlxMemorySnapshot {
                    try WorkerEventHandler.handle(
                        .mlxMemorySample(
                            mlxMemorySnapshot: mlxMemorySnapshot,
                            expertResidency: nil),
                        healthState: healthState);
                }
                return;
            case .expertMemoryModeChanged:
                try WorkerEventHandler.handle(workerEvent, healthState: healthState);
            case let .generationFinalized(
                _,
                expertMemoryMode,
                mlxMemorySnapshot,
                expertResidency):
                if let expertMemoryMode = expertMemoryMode {
                    try WorkerEventHandler.handle(
                        .expertMemoryModeChanged(expertMemoryMode: expertMemoryMode),
                        healthState: healthState);
                }
                try WorkerEventHandler.handle(
                    .mlxMemorySample(
                        mlxMemorySnapshot: mlxMemorySnapshot,
                        expertResidency: expertResidency),
                    healthState: healthState);
            case .persistentPromptCacheStats:
                // Cache publication can finish after the client drops the
                // stream. Keep the worker reusable and record the telemetry.
                try WorkerEventHandler.handle(workerEvent, healthState: healthState);
            case .mlxMemorySample:
                try WorkerEventHandler.handle(workerEvent, healthState: healthState);
            case let .completed(eventRequestId, _, _, _, _, _, _)
                where eventRequestId == requestId:
                return;
            case let .failed(eventRequestId, _)
                where eventRequestId == requestId:
                return;
            default:
                throw WorkerControlError.unexpectedCancellationEvent(
                    requestId: requestId.value(),
                    unexpectedWorkerEventSummary: workerEvent.diagnosticSummary());
            }
        }
    }
}
