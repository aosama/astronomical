import Foundation;

import IpcProtocol;

/// The per-request execution pipeline of the owning supervisor: startup
/// wait, on-demand model swap, command send, and the bounded event
/// collection for chat, embeddings, and image requests, with the same
/// containment and active-request matching on every path.
extension WorkerSupervisor {

    // MARK: - Generation execution

    func runGeneration(
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
    func collectGenerationEvents(        _ requestId: RequestId,
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
    func collectEmbeddingsEvents(
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

    // MARK: - Image execution

    /// Runs one image command through the worker: startup wait, model swap,
    /// command send, and bounded output collection.
    func runImageGeneration(
        _ imageGenerationCommand: ImageGenerationCommand,
        timeouts: ImageGenerationTimeouts
    ) throws -> ImageGenerationOutput {
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
            targetModelId: imageGenerationCommand.model,
            workerProcess: workerProcess,
            eventPump: eventPump,
            healthState: self.healthState,
            modelPolicyCatalog: self.modelPolicyCatalog,
            modelLoadTimeout: self.modelLoadTimeout,
            containment: { (controlError: Error) -> Void in
                self.containAndAttemptRelaunch(controlError: controlError);
            });
        do {
            try workerProcess.sendCommand(.generateImage(imageGenerationCommand));
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
            return try self.collectImageGenerationEvents(
                imageGenerationCommand.requestId,
                eventPump: eventPump,
                timeouts: timeouts);
        } catch let controlError as WorkerControlError {
            self.containAndAttemptRelaunch(controlError: controlError);
            throw ImageGenerationExecutionError.workerUnavailable;
        }
    }

    /// Drains worker events until this image request completes or fails,
    /// bounding both the total execution and the gap without any forward
    /// event, exactly as the Rust executor's two timeouts do.
    func collectImageGenerationEvents(
        _ requestId: RequestId,
        eventPump: WorkerEventPump,
        timeouts: ImageGenerationTimeouts
    ) throws -> ImageGenerationOutput {
        let executionDeadline: Date = Date().addingTimeInterval(timeouts.executionTimeoutSeconds);
        var lastProgressDate: Date = Date();
        while true {
            self.stateLock.lock();
            let isShutdownRequested: Bool = self.isShutdownRequested;
            self.stateLock.unlock();
            if isShutdownRequested {
                throw WorkerControlError.workerEventStreamClosed;
            }
            guard let workerEvent: WorkerEvent = try eventPump.nextEvent(
                within: WorkerSupervisor.generationPollSeconds) else {
                if Date() >= executionDeadline || Date().addingTimeInterval(
                    -timeouts.progressStallTimeoutSeconds) > lastProgressDate {
                    throw ImageGenerationExecutionError.deadlineExceeded;
                }
                continue;
            }
            lastProgressDate = Date();
            switch (workerEvent) {
            case let .imageGenerationCompleted(eventRequestId, generatedImage, resultMetadata):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                return ImageGenerationOutput(
                    generatedImage: generatedImage,
                    resultMetadata: resultMetadata);
            case let .imageGenerationFailed(eventRequestId, reason):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                throw ImageGenerationExecutionError.workerFailure(reason);
            case .imageGenerationFinalized:
                // The request-scoped release acknowledgement; the outcome
                // event always arrives first.
                continue;
            case .generationPreparationStarted, .generationProgress, .firstDecodeCompleted,
                 .promptWorkReuse, .generationFinalized, .prefillProgress:
                // Request-scoped telemetry resets the forward-progress bound.
                continue;
            default:
                try WorkerEventHandler.handle(workerEvent, healthState: self.healthState);
            }
        }
    }

}
