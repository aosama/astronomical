import Foundation;

import IpcProtocol;

/// Image-request execution over the worker process: startup wait, model
/// swap, dispatch guarded against an abandoned requester, and the bounded
/// collection that resolves the request at its release frame.
extension WorkerSupervisor {

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
            self.containWorkerFailure(controlError: error);
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
                self.containWorkerFailure(controlError: controlError);
            });
        // A requester that went away while the model swap was still in
        // flight must not have its image dispatched into a worker the
        // supervisor is already shutting down — the synchronous analog of
        // the Rust executor dropping the request future at its await point.
        self.stateLock.lock();
        let isShutdownRequested: Bool = self.isShutdownRequested;
        self.stateLock.unlock();
        if (isShutdownRequested) {
            throw GenerationStartError.workerUnavailable;
        }
        do {
            try workerProcess.sendCommand(.generateImage(imageGenerationCommand));
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
        self.publishServingActivity(.imageGeneration, progress: nil);
        do {
            return try self.collectImageGenerationEvents(
                imageGenerationCommand.requestId,
                eventPump: eventPump,
                timeouts: timeouts);
        } catch let controlError as WorkerControlError {
            self.containWorkerFailure(controlError: controlError);
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
        var stagedImageOutcome: Result<ImageGenerationOutput, ImageGenerationExecutionError>? = nil;
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
                if (stagedImageOutcome != nil) {
                    throw WorkerControlError.workerProtocolViolation(
                        description: "duplicate image terminal outcome");
                }
                stagedImageOutcome = .success(ImageGenerationOutput(
                    generatedImage: generatedImage,
                    resultMetadata: resultMetadata));
                continue;
            case let .imageGenerationFailed(eventRequestId, reason):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                if (stagedImageOutcome != nil) {
                    throw WorkerControlError.workerProtocolViolation(
                        description: "duplicate image terminal outcome");
                }
                stagedImageOutcome = .failure(ImageGenerationExecutionError.workerFailure(reason));
                continue;
            case let .imageGenerationFinalized(eventRequestId, _, mlxMemorySnapshot):
                // The request resolves at its release acknowledgement, the
                // exact terminal-plus-finalized pairing the Rust executor
                // enforces, so no trailing release frame is ever left in the
                // stream for the next waiter to trip over.
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                if let mlxMemorySnapshot = mlxMemorySnapshot {
                    try WorkerEventHandler.handle(
                        .mlxMemorySample(
                            mlxMemorySnapshot: mlxMemorySnapshot,
                            expertResidency: nil),
                        healthState: self.healthState);
                }
                self.publishServingActivity(.idle, progress: nil);
                guard let resolvedImageOutcome: Result<ImageGenerationOutput, ImageGenerationExecutionError> =
                    stagedImageOutcome else {
                    throw WorkerControlError.workerProtocolViolation(
                        description: "image finalized before a terminal outcome");
                }
                switch (resolvedImageOutcome) {
                case let .success(imageGenerationOutput):
                    return imageGenerationOutput;
                case let .failure(imageGenerationError):
                    throw imageGenerationError;
                }
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
