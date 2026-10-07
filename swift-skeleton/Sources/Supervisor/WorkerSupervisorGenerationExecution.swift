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
        self.publishServingActivity(.promptProcessing, progress: nil);
        do {
            return try self.collectGenerationEvents(
                generationCommand.requestId,
                requestStartedAt: Date(),
                maximumOutputTokens: generationCommand.settings.maxOutputTokens,
                eventPump: eventPump);
        } catch let controlError as WorkerControlError {
            self.containAndAttemptRelaunch(controlError: controlError);
            throw GenerationStartError.workerUnavailable;
        }
    }

    /// Publishes the request-phase activity and its latest progress
    /// observation to health state, mirroring the Rust publish_activity +
    /// publish_active_request_progress pair.
    func publishServingActivity(
        _ activity: WorkerActivity,
        progress: ActiveRequestProgress?
    ) -> Void {
        try? self.healthState.apply({ (snapshot: inout WorkerHealthSnapshot) -> Void in
            snapshot.activity = activity;
            snapshot.activeRequestProgress = progress;
        });
    }

    /// Drains worker events until this request's terminal event, routing
    /// process-scoped events to health state and generation events for other
    /// requests to protocol violations, mirroring the Rust loop's
    /// active-request matching.
    func collectGenerationEvents(        _ requestId: RequestId,
        requestStartedAt: Date,
        maximumOutputTokens: UInt16,
        eventPump: WorkerEventPump
    ) throws -> Array<ChatGenerationStreamEvent> {
        var streamEvents: Array<ChatGenerationStreamEvent> = Array<ChatGenerationStreamEvent>();
        var generationStartedAt: Date?;
        var prefillElapsedMillis: UInt64 = 0;
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
            case let .output(eventRequestId, _, generatedTokenCount, outputs, _, _):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                if generationStartedAt == nil {
                    generationStartedAt = Date();
                }
                self.publishGenerationProgress(
                    generatedTokenCount: UInt32(generatedTokenCount),
                    maximumOutputTokens: UInt32(maximumOutputTokens),
                    generationStartedAt: generationStartedAt!);
                for workerOutput: ChatGenerationOutput in outputs {
                    streamEvents.append(ChatGenerationStreamEvent.fromWorkerOutput(workerOutput));
                }
            case let .prefillProgress(
                eventRequestId,
                promptProcessingPhase,
                processedTokens,
                totalTokens,
                elapsedMillis,
                forwardPrefillChunkElapsedMillis,
                completedPrefillChunkTokens,
                mlxMemorySnapshot,
                _):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                prefillElapsedMillis = max(prefillElapsedMillis, elapsedMillis);
                if let mlxMemorySnapshot = mlxMemorySnapshot {
                    try WorkerEventHandler.handle(
                        .mlxMemorySample(
                            mlxMemorySnapshot: mlxMemorySnapshot,
                            expertResidency: nil),
                        healthState: self.healthState);
                }
                self.publishServingActivity(.promptProcessing, progress: .prefill(
                    promptProcessingPhase: promptProcessingPhase,
                    processedTokens: processedTokens,
                    totalTokens: totalTokens,
                    requestStartedAt: requestStartedAt,
                    elapsedMillis: elapsedMillis,
                    completedPrefillChunkTokens: completedPrefillChunkTokens));
                streamEvents.append(.prefillProgress(
                    processedTokens: processedTokens,
                    totalTokens: totalTokens,
                    elapsedMillis: elapsedMillis,
                    forwardPrefillChunkElapsedMillis: forwardPrefillChunkElapsedMillis,
                    completedPrefillChunkTokens: completedPrefillChunkTokens,
                    mlxActiveMemoryBytes: mlxMemorySnapshot?.activeMemoryBytes,
                    mlxAllocatorCacheMemoryBytes: mlxMemorySnapshot?.allocatorCacheMemoryBytes,
                    mlxPeakMemoryBytes: mlxMemorySnapshot?.peakMemoryBytes));
            case let .generationPreparationStarted(
                eventRequestId,
                totalLayerCount,
                residentExpertCount,
                residentExpertPayloadBytes,
                _):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                let preparationStartedAt: Date = Date();
                self.publishServingActivity(.generationPreparation, progress: .generationPreparation(
                    requestStartedAt: requestStartedAt,
                    preparationStartedAt: preparationStartedAt,
                    totalLayerCount: totalLayerCount,
                    residentExpertCount: residentExpertCount,
                    residentExpertPayloadBytes: residentExpertPayloadBytes));
                continue;
            case let .generationProgress(
                eventRequestId,
                generatedTokenCount,
                eventMaximumOutputTokens,
                _,
                _,
                _):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                if generationStartedAt == nil {
                    generationStartedAt = Date();
                }
                self.publishGenerationProgress(
                    generatedTokenCount: UInt32(generatedTokenCount),
                    maximumOutputTokens: UInt32(eventMaximumOutputTokens),
                    generationStartedAt: generationStartedAt!);
                continue;
            case let .firstDecodeCompleted(eventRequestId, _):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                if generationStartedAt == nil {
                    generationStartedAt = Date();
                }
                self.publishGenerationProgress(
                    generatedTokenCount: 1,
                    maximumOutputTokens: UInt32(maximumOutputTokens),
                    generationStartedAt: generationStartedAt!);
                continue;
            case let .promptWorkReuse(eventRequestId, promptWorkReuse):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                try? self.healthState.apply({ (snapshot: inout WorkerHealthSnapshot) -> Void in
                    snapshot.servingSession.recordPromptWorkReuse(promptWorkReuse);
                });
                continue;
            case let .completed(
                eventRequestId,
                promptTokenCount,
                generatedTokenCount,
                reasoningTokenCount,
                cachedTokenCount,
                _,
                reason):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                self.recordServingSessionTotals(
                    promptTokenCount: promptTokenCount,
                    cachedTokenCount: cachedTokenCount,
                    generatedTokenCount: generatedTokenCount,
                    prefillElapsedMillis: prefillElapsedMillis,
                    generationStartedAt: generationStartedAt);
                self.publishServingActivity(.idle, progress: nil);
                streamEvents.append(.completed(
                    promptTokenCount: promptTokenCount,
                    generatedTokenCount: generatedTokenCount,
                    reasoningTokenCount: reasoningTokenCount,
                    cachedTokenCount: cachedTokenCount,
                    reason: reason));
                return streamEvents;
            case let .failed(eventRequestId, reason):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                self.publishServingActivity(.idle, progress: nil);
                streamEvents.append(.failed(reason: reason));
                return streamEvents;
            case .generationFinalized:
                // The request-scoped release acknowledgement; the outcome
                // event always arrives first and already restored idle.
                continue;
            default:
                try WorkerEventHandler.handle(workerEvent, healthState: self.healthState);
            }
        }
    }

    /// Records one completed request into the lifetime serving totals with
    /// its measured throughput, mirroring the Rust completion-event path.
    private func recordServingSessionTotals(
        promptTokenCount: UInt32,
        cachedTokenCount: UInt32,
        generatedTokenCount: UInt16,
        prefillElapsedMillis: UInt64,
        generationStartedAt: Date?
    ) -> Void {
        let generationElapsedMillis: UInt64 = generationStartedAt.map({ (startedAt: Date) -> UInt64 in
            return UInt64((Date().timeIntervalSince(startedAt) * 1000).rounded());
        }) ?? 0;
        let throughput: (prefillTokPerSecond: Double?, generationTokPerSecond: Double?) =
            ServingSessionSnapshot.computeThroughput(
                promptTokenCount: promptTokenCount,
                cachedTokenCount: cachedTokenCount,
                generatedTokenCount: generatedTokenCount,
                prefillElapsedMillis: prefillElapsedMillis,
                generationElapsedMillis: generationElapsedMillis);
        try? self.healthState.apply({ (snapshot: inout WorkerHealthSnapshot) -> Void in
            snapshot.servingSession.recordCompletedRequest(
                promptTokenCount: promptTokenCount,
                cachedTokenCount: cachedTokenCount,
                prefillTokPerSecond: throughput.prefillTokPerSecond,
                generationTokPerSecond: throughput.generationTokPerSecond);
        });
    }

    /// Publishes the generating activity with elapsed time measured from the
    /// first decode observation, mirroring the Rust generation-output path.
    private func publishGenerationProgress(
        generatedTokenCount: UInt32,
        maximumOutputTokens: UInt32,
        generationStartedAt: Date
    ) -> Void {
        let elapsedMillis: UInt64 = UInt64(
            (Date().timeIntervalSince(generationStartedAt) * 1000).rounded());
        self.publishServingActivity(.generating, progress: .generation(
            generatedTokenCount: generatedTokenCount,
            maximumOutputTokens: maximumOutputTokens,
            elapsedMillis: elapsedMillis));
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
                self.publishServingActivity(.idle, progress: nil);
                return EmbeddingsOutput(
                    embeddings: embeddings,
                    inputTokenCounts: inputTokenCounts);
            case let .embeddingsFailed(eventRequestId, reason):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                self.publishServingActivity(.idle, progress: nil);
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
        self.publishServingActivity(.imageGeneration, progress: nil);
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
            case let .imageGenerationProgress(
                eventRequestId,
                phase,
                completedSteps,
                totalSteps,
                elapsedMillis,
                mlxMemorySnapshot):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                if let mlxMemorySnapshot = mlxMemorySnapshot {
                    try WorkerEventHandler.handle(
                        .mlxMemorySample(
                            mlxMemorySnapshot: mlxMemorySnapshot,
                            expertResidency: nil),
                        healthState: self.healthState);
                }
                self.publishServingActivity(.imageGeneration, progress: .imageGeneration(
                    phase: phase,
                    completedSteps: completedSteps,
                    totalSteps: totalSteps,
                    elapsedMillis: elapsedMillis));
                continue;
            case let .imageGenerationCompleted(eventRequestId, generatedImage, resultMetadata):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                self.publishServingActivity(.idle, progress: nil);
                return ImageGenerationOutput(
                    generatedImage: generatedImage,
                    resultMetadata: resultMetadata);
            case let .imageGenerationFailed(eventRequestId, reason):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                self.publishServingActivity(.idle, progress: nil);
                throw ImageGenerationExecutionError.workerFailure(reason);
            case .imageGenerationFinalized:
                // The request-scoped release acknowledgement; the outcome
                // event always arrives first and already restored idle.
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
