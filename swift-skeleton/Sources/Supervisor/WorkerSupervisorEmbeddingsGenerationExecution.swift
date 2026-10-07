import Foundation;

import IpcProtocol;

/// The per-request execution pipeline of the owning supervisor for
/// embeddings requests, mirroring the chat and image execution files:
/// startup wait, on-demand model swap, command send, and the bounded
/// event collection with the same containment and active-request
/// matching on every path.
extension WorkerSupervisor {
    /// swap, command send, and the completed output collection, mirroring the
    /// chat generation path's containment.
    func runEmbeddingsGeneration(
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
            self.containWorkerFailure(controlError: error);
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
                self.containWorkerFailure(controlError: controlError);
            });
        do {
            try workerProcess.sendCommand(.generateEmbeddings(embeddingsCommand));
        } catch let protocolError as ProtocolError {
            if case let .outgoingMessageTooLarge(actualMessageBytes, maximumMessageBytes) = protocolError {
                throw GenerationStartError.requestTooLarge(
                    actualIpcMessageBytes: actualMessageBytes,
                    maximumIpcMessageBytes: maximumMessageBytes);
            }
            self.containWorkerFailure(controlError: protocolError);
            throw GenerationStartError.workerUnavailable;
        } catch let sendError {
            self.containWorkerFailure(controlError: sendError);
            throw GenerationStartError.workerUnavailable;
        }
        do {
            return try self.collectEmbeddingsEvents(
                embeddingsCommand.requestId,
                eventPump: eventPump);
        } catch let controlError as WorkerControlError {
            self.containWorkerFailure(controlError: controlError);
            throw EmbeddingsExecutionError.workerUnavailable;
        }
    }

    /// Drains worker events until this embeddings request completes or fails,
    /// applying the same active-request matching as the chat collection.
    func collectEmbeddingsEvents(
        _ requestId: RequestId,
        eventPump: WorkerEventPump
    ) throws -> EmbeddingsOutput {
        var stagedEmbeddingsOutcome: Result<EmbeddingsOutput, EmbeddingsExecutionError>? = nil;
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
                if (stagedEmbeddingsOutcome != nil) {
                    throw WorkerControlError.workerProtocolViolation(
                        description: "duplicate embeddings terminal outcome");
                }
                stagedEmbeddingsOutcome = .success(EmbeddingsOutput(
                    embeddings: embeddings,
                    inputTokenCounts: inputTokenCounts));
                continue
            case let .embeddingsFailed(eventRequestId, reason):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                if (stagedEmbeddingsOutcome != nil) {
                    throw WorkerControlError.workerProtocolViolation(
                        description: "duplicate embeddings terminal outcome");
                }
                stagedEmbeddingsOutcome = .failure(EmbeddingsExecutionError.workerFailure(reason));
                continue
            case let .embeddingsFinalized(eventRequestId, _, mlxMemorySnapshot):
                // The request resolves at its release acknowledgement exactly
                // as the Rust executor pairs the terminal frame with the
                // finalized frame, so no trailing release frame is ever left
                // for the next waiter to trip over.
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                if let mlxMemorySnapshot = mlxMemorySnapshot {
                    try WorkerEventHandler.handle(
                        .mlxMemorySample(
                            mlxMemorySnapshot: mlxMemorySnapshot,
                            expertResidency: nil),
                        healthState: self.healthState);
                }
                self.publishServingActivity(.idle, progress: nil);
                guard let resolvedEmbeddingsOutcome: Result<EmbeddingsOutput, EmbeddingsExecutionError> =
                    stagedEmbeddingsOutcome else {
                    throw WorkerControlError.workerProtocolViolation(
                        description: "embeddings finalized before a terminal outcome");
                }
                switch (resolvedEmbeddingsOutcome) {
                case let .success(embeddingsOutput):
                    return embeddingsOutput;
                case let .failure(embeddingsError):
                    throw embeddingsError;
                }
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

}
