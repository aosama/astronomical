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
            self.containWorkerFailure(controlError: error);
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
                self.containWorkerFailure(controlError: controlError);
            });
        do {
            try workerProcess.sendCommand(.generate(generationCommand));
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
        self.publishServingActivity(.promptProcessing, progress: nil);
        do {
            return try self.collectGenerationEvents(
                generationCommand.requestId,
                requestStartedAt: Date(),
                maximumOutputTokens: generationCommand.settings.maxOutputTokens,
                eventPump: eventPump);
        } catch let controlError as WorkerControlError {
            self.containWorkerFailure(controlError: controlError);
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
    /// active-request matching. The optional sink receives each public
    /// stream event as it is produced (the streaming serving surface), and
    /// the optional abandonment probe diverts into cancellation the moment
    /// the client stops consuming.
    func collectGenerationEvents(        _ requestId: RequestId,
        requestStartedAt: Date,
        maximumOutputTokens: UInt16,
        eventPump: WorkerEventPump,
        onStreamEvent: ((ChatGenerationStreamEvent) -> Void)? = nil,
        isClientAbandoned: (() -> Bool)? = nil
    ) throws -> Array<ChatGenerationStreamEvent> {
        var streamEvents: Array<ChatGenerationStreamEvent> = Array<ChatGenerationStreamEvent>();
        func emitStreamEvent(_ streamEvent: ChatGenerationStreamEvent) -> Void {
            streamEvents.append(streamEvent);
            onStreamEvent?(streamEvent);
        }
        var validationState: ChatGenerationRequestValidationState = ChatGenerationRequestValidationState();
        var generationStartedAt: Date?;
        var firstOutputAt: Date?;
        var prefillElapsedMillis: UInt64 = 0;
        var maximumMlxPeakMemoryBytes: UInt64?;
        var lastMlxActiveMemoryBytes: UInt64?;
        while true {
            self.stateLock.lock();
            let isShutdownRequested: Bool = self.isShutdownRequested;
            self.stateLock.unlock();
            if isShutdownRequested {
                throw WorkerControlError.workerEventStreamClosed;
            }
            if let isClientAbandoned = isClientAbandoned, isClientAbandoned() {
                throw ChatGenerationClientAbandonment.abandonedByClient;
            }
            guard let workerEvent: WorkerEvent = try eventPump.nextEvent(
                within: WorkerSupervisor.generationPollSeconds) else {
                continue;
            }
            switch (workerEvent) {
            case let .output(eventRequestId, sequenceNumber, generatedTokenCount, outputs, mlxMemorySnapshot, expertResidency):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                try ChatGenerationEventValidation.acceptOutputBatch(
                    sequenceNumber: sequenceNumber,
                    generatedTokenCount: generatedTokenCount,
                    outputs: outputs,
                    maximumOutputTokens: maximumOutputTokens,
                    validationState: &validationState);
                if generationStartedAt == nil {
                    generationStartedAt = Date();
                }
                if firstOutputAt == nil {
                    firstOutputAt = Date();
                }
                self.publishGenerationProgress(
                    generatedTokenCount: UInt32(generatedTokenCount),
                    maximumOutputTokens: UInt32(maximumOutputTokens),
                    generationStartedAt: generationStartedAt!);
                try self.publishGenerationTelemetry(
                    mlxMemorySnapshot: mlxMemorySnapshot,
                    expertResidency: expertResidency);
                for workerOutput: ChatGenerationOutput in outputs {
                    emitStreamEvent(ChatGenerationStreamEvent.fromWorkerOutput(workerOutput));
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
                    maximumMlxPeakMemoryBytes = max(
                        maximumMlxPeakMemoryBytes ?? 0,
                        mlxMemorySnapshot.peakMemoryBytes);
                    lastMlxActiveMemoryBytes = mlxMemorySnapshot.activeMemoryBytes;
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
                emitStreamEvent(.prefillProgress(
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
                preparationMlxMemorySnapshot):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                try ChatGenerationEventValidation.acceptGenerationPreparation(
                    residentExpertCount: residentExpertCount,
                    residentExpertPayloadBytes: residentExpertPayloadBytes,
                    validationState: &validationState);
                if let preparationMlxMemorySnapshot = preparationMlxMemorySnapshot {
                    try WorkerEventHandler.handle(
                        .mlxMemorySample(
                            mlxMemorySnapshot: preparationMlxMemorySnapshot,
                            expertResidency: nil),
                        healthState: self.healthState);
                }
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
                progressMlxMemorySnapshot,
                progressExpertResidency):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                try ChatGenerationEventValidation.acceptGenerationProgress(
                    generatedTokenCount: generatedTokenCount,
                    eventMaximumOutputTokens: eventMaximumOutputTokens,
                    maximumOutputTokens: maximumOutputTokens,
                    validationState: &validationState);
                if generationStartedAt == nil {
                    generationStartedAt = Date();
                }
                self.publishGenerationProgress(
                    generatedTokenCount: UInt32(generatedTokenCount),
                    maximumOutputTokens: UInt32(eventMaximumOutputTokens),
                    generationStartedAt: generationStartedAt!);
                try self.publishGenerationTelemetry(
                    mlxMemorySnapshot: progressMlxMemorySnapshot,
                    expertResidency: progressExpertResidency);
                continue;
            case let .firstDecodeCompleted(eventRequestId, _):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                try ChatGenerationEventValidation.acceptFirstDecodeCompleted(
                    validationState: &validationState);
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
                persistentPromptCacheDiagnostics,
                reason):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                try ChatGenerationEventValidation.acceptCompletion(
                    generatedTokenCount: generatedTokenCount,
                    reason: reason,
                    maximumOutputTokens: maximumOutputTokens,
                    validationState: &validationState);
                self.recordServingSessionTotals(
                    promptTokenCount: promptTokenCount,
                    cachedTokenCount: cachedTokenCount,
                    generatedTokenCount: generatedTokenCount,
                    prefillElapsedMillis: prefillElapsedMillis,
                    generationStartedAt: generationStartedAt);
                self.recordCompletionAttribution(
                    requestId: requestId,
                    requestStartedAt: requestStartedAt,
                    generationStartedAt: generationStartedAt,
                    firstOutputAt: firstOutputAt,
                    promptTokenCount: promptTokenCount,
                    cachedTokenCount: cachedTokenCount,
                    generatedTokenCount: generatedTokenCount,
                    prefillElapsedMillis: prefillElapsedMillis,
                    maximumMlxPeakMemoryBytes: maximumMlxPeakMemoryBytes,
                    lastMlxActiveMemoryBytes: lastMlxActiveMemoryBytes,
                    persistentPromptCacheDiagnostics: persistentPromptCacheDiagnostics,
                    completionReason: reason,
                    streamEvents: streamEvents);
                self.publishServingActivity(.idle, progress: nil);
                emitStreamEvent(.completed(
                    promptTokenCount: promptTokenCount,
                    generatedTokenCount: generatedTokenCount,
                    reasoningTokenCount: reasoningTokenCount,
                    cachedTokenCount: cachedTokenCount,
                    reason: reason));
                return streamEvents;
            case let .failed(eventRequestId, reason):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                self.publishServingActivity(.idle, progress: nil);
                emitStreamEvent(.failed(reason: reason));
                return streamEvents;
            case let .generationFinalized(
                eventRequestId,
                finalizedExpertMemoryMode,
                finalizedMlxMemorySnapshot,
                finalizedExpertResidency):
                // The request-scoped release acknowledgement: it carries the
                // final residency memory that replaces any prefill-era
                // telemetry, exactly as the Rust finalized arm publishes it.
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                try WorkerEventHandler.handle(
                    .mlxMemorySample(
                        mlxMemorySnapshot: finalizedMlxMemorySnapshot,
                        expertResidency: finalizedExpertResidency),
                    healthState: self.healthState);
                if let finalizedExpertMemoryMode = finalizedExpertMemoryMode {
                    try self.healthState.apply({ (snapshot: inout WorkerHealthSnapshot) -> Void in
                        snapshot.expertMemoryMode = snapshot.readyModelId.map({ (_: String) -> ExpertMemoryMode in
                            return finalizedExpertMemoryMode;
                        });
                    });
                }
                continue;
            default:
                try WorkerEventHandler.handle(workerEvent, healthState: self.healthState);
            }
        }
    }

    /// Publishes one generation frame's live memory observation and expert
    /// residency into health state, mirroring the Rust output and progress
    /// arms: an absent snapshot leaves the last observation untouched, and
    /// residency only ever replaces itself when the frame carries one.
    private func publishGenerationTelemetry(
        mlxMemorySnapshot: WorkerMlxMemorySnapshot?,
        expertResidency: WorkerExpertResidencySnapshot?
    ) throws -> Void {
        if let mlxMemorySnapshot = mlxMemorySnapshot {
            try WorkerEventHandler.handle(
                .mlxMemorySample(
                    mlxMemorySnapshot: mlxMemorySnapshot,
                    expertResidency: expertResidency),
                healthState: self.healthState);
            return;
        }
        if let expertResidency = expertResidency {
            try self.healthState.apply({ (snapshot: inout WorkerHealthSnapshot) -> Void in
                snapshot.expertResidency = expertResidency;
            });
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

    static func requireActiveRequest(
        _ eventRequestId: RequestId,
        requestId: RequestId
    ) throws -> Void {
        if eventRequestId != requestId {
            throw WorkerControlError.workerProtocolViolation(
                description: "generation event \(eventRequestId.value()) does not belong to the active request");
        }
    }
}
