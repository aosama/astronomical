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
        timeouts: ImageGenerationTimeouts,
        isClientAbandoned: (@Sendable () -> Bool)?,
        journeyTiming: ImageGenerationExecutionTiming
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
        let swapLoadStartedAt: Date = Date();
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
        journeyTiming.swapLoadElapsedMillis = ImageGenerationExecutionTiming.elapsedMillis(
            since: swapLoadStartedAt);
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
        journeyTiming.executionStartedAt = Date();
        self.publishServingActivity(.imageGeneration, progress: nil);
        do {
            return try self.collectImageGenerationEvents(
                imageGenerationCommand,
                eventPump: eventPump,
                timeouts: timeouts,
                isClientAbandoned: isClientAbandoned,
                journeyTiming: journeyTiming);
        } catch ImageGenerationClientAbandonment.abandonedByClient {
            // The requester stopped waiting: free the shared worker through
            // the bounded cancellation path before this thread unwinds.
            self.cancelActiveGeneration(
                requestId: imageGenerationCommand.requestId,
                workerProcess: workerProcess,
                eventPump: eventPump,
                expectsImageFinalization: true);
            throw ImageGenerationExecutionError.workerFailure(.cancelled);
        } catch ImageGenerationExecutionError.deadlineExceeded {
            // A bound ran out: cancel the in-flight request so the worker's
            // terminal pair cannot leak into the next waiter.
            self.cancelActiveGeneration(
                requestId: imageGenerationCommand.requestId,
                workerProcess: workerProcess,
                eventPump: eventPump,
                expectsImageFinalization: true);
            throw ImageGenerationExecutionError.deadlineExceeded;
        } catch let imageExecutionError as ImageGenerationExecutionError {
            throw imageExecutionError;
        } catch let controlError {
            self.containWorkerFailure(controlError: controlError);
            throw ImageGenerationExecutionError.workerUnavailable;
        }
    }

    /// Drains worker events until this image request completes or fails,
    /// bounding both the total execution and the gap without forward
    /// progress, exactly as the Rust executor's two timeouts do. The
    /// forward-progress bound refreshes only on genuinely advanced image
    /// progress for this request, and every event must satisfy the
    /// monotonicity and settings-match contract of
    /// apps/supervisor/src/worker_image_event.rs.
    func collectImageGenerationEvents(
        _ imageGenerationCommand: ImageGenerationCommand,
        eventPump: WorkerEventPump,
        timeouts: ImageGenerationTimeouts,
        isClientAbandoned: (@Sendable () -> Bool)?,
        journeyTiming: ImageGenerationExecutionTiming
    ) throws -> ImageGenerationOutput {
        let requestId: RequestId = imageGenerationCommand.requestId;
        let settings: ImageGenerationSettings = imageGenerationCommand.settings;
        let executionDeadline: Date = Date().addingTimeInterval(timeouts.executionTimeoutSeconds);
        var lastAdvancedProgressDate: Date = Date();
        var stagedImageOutcome: Result<ImageGenerationOutput, ImageGenerationExecutionError>? = nil;
        var latestPhase: ImageGenerationPhase? = nil;
        var latestCompletedSteps: UInt16 = 0;
        var latestElapsedMillis: UInt64 = 0;
        while true {
            self.stateLock.lock();
            let isShutdownRequested: Bool = self.isShutdownRequested;
            self.stateLock.unlock();
            if isShutdownRequested {
                throw WorkerControlError.workerEventStreamClosed;
            }
            if let isClientAbandoned = isClientAbandoned, isClientAbandoned() {
                throw ImageGenerationClientAbandonment.abandonedByClient;
            }
            guard let workerEvent: WorkerEvent = try eventPump.nextEvent(
                within: WorkerSupervisor.generationPollSeconds) else {
                if Date() >= executionDeadline || Date().addingTimeInterval(
                    -timeouts.progressStallTimeoutSeconds) > lastAdvancedProgressDate {
                    throw ImageGenerationExecutionError.deadlineExceeded;
                }
                continue;
            }
            switch (workerEvent) {
            case let .imageGenerationProgress(
                eventRequestId,
                phase,
                completedSteps,
                totalSteps,
                elapsedMillis,
                mlxMemorySnapshot):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                if (stagedImageOutcome != nil
                    || totalSteps != settings.steps
                    || completedSteps < latestCompletedSteps
                    || completedSteps > totalSteps
                    || elapsedMillis < latestElapsedMillis
                    || (latestPhase.map { (latestPhase: ImageGenerationPhase) -> Bool in
                        return WorkerSupervisor.imagePhaseRank(phase)
                            < WorkerSupervisor.imagePhaseRank(latestPhase);
                    } ?? false)) {
                    throw WorkerControlError.workerProtocolViolation(
                        description: "invalid image generation progress");
                }
                let progressAdvanced: Bool = latestPhase != phase
                    || completedSteps > latestCompletedSteps
                    || elapsedMillis > latestElapsedMillis;
                latestPhase = phase;
                latestCompletedSteps = completedSteps;
                latestElapsedMillis = elapsedMillis;
                if (progressAdvanced) {
                    lastAdvancedProgressDate = Date();
                }
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
                if (stagedImageOutcome != nil
                    || resultMetadata.widthPixels != settings.widthPixels
                    || resultMetadata.heightPixels != settings.heightPixels
                    || resultMetadata.steps != settings.steps
                    || resultMetadata.guidanceThousandths != settings.guidanceThousandths
                    || resultMetadata.seed != settings.seed
                    || resultMetadata.elapsedMillis < latestElapsedMillis
                    || generatedImage.mimeType != "image/png"
                    || generatedImage.encodedBytes.isEmpty) {
                    throw WorkerControlError.workerProtocolViolation(
                        description: "invalid completed image result");
                }
                stagedImageOutcome = .success(ImageGenerationOutput(
                    generatedImage: generatedImage,
                    resultMetadata: resultMetadata));
                journeyTiming.terminalReceivedAt = Date();
                continue;
            case let .imageGenerationFailed(eventRequestId, reason):
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                if (stagedImageOutcome != nil) {
                    throw WorkerControlError.workerProtocolViolation(
                        description: "duplicate image terminal outcome");
                }
                stagedImageOutcome = .failure(ImageGenerationExecutionError.workerFailure(reason));
                journeyTiming.terminalReceivedAt = Date();
                continue;
            case let .imageGenerationFinalized(eventRequestId, workerReportedElapsedMillis, mlxMemorySnapshot):
                // The request resolves at its release acknowledgement, the
                // exact terminal-plus-finalized pairing the Rust executor
                // enforces, so no trailing release frame is ever left in the
                // stream for the next waiter to trip over.
                try WorkerSupervisor.requireActiveRequest(eventRequestId, requestId: requestId);
                if (workerReportedElapsedMillis < latestElapsedMillis) {
                    throw WorkerControlError.workerProtocolViolation(
                        description: "image finalization elapsed time regressed");
                }
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
                self.generationPerformanceLog.recordImage(
                    journeyTiming.performanceRecord(
                        imageGenerationCommand: imageGenerationCommand,
                        completionOutcome: resolvedImageOutcome.isCompleted
                            ? "completed"
                            : "failed",
                        workerReportedElapsedMillis: workerReportedElapsedMillis,
                        encodedImageBytes: resolvedImageOutcome.encodedImageByteCount,
                        mlxPeakMemoryBytes: mlxMemorySnapshot?.peakMemoryBytes,
                        mlxActiveMemoryBytes: mlxMemorySnapshot?.activeMemoryBytes));
                switch (resolvedImageOutcome) {
                case let .success(imageGenerationOutput):
                    return imageGenerationOutput;
                case let .failure(imageGenerationError):
                    throw imageGenerationError;
                }
            case .generationPreparationStarted, .generationProgress, .firstDecodeCompleted,
                 .promptWorkReuse, .generationFinalized, .prefillProgress:
                // Foreign-request telemetry: never advances this request's
                // forward-progress bound, matching the Rust active-request
                // matching that ignores other requests' events.
                continue;
            default:
                try WorkerEventHandler.handle(workerEvent, healthState: self.healthState);
            }
        }
    }

    /// The serving order of the render phases; progress may not move to a
    /// lower-ranked phase, mirroring the Rust image_phase_rank.
    private static func imagePhaseRank(_ phase: ImageGenerationPhase) -> Int {
        switch (phase) {
        case .preparing: return 0;
        case .encodingPrompt: return 1;
        case .denoising: return 2;
        case .decoding: return 3;
        case .encodingImage: return 4;
        }
    }
}

/// The phase stamps one image request collects between arrival and cleanup
/// finalization; they attribute the performance row the way the Rust
/// executor's ActiveImageGeneration does. A class so the executor's deeper
/// frames stamp phases without threading it back out by hand.
final class ImageGenerationExecutionTiming {

    var requestArrivedAt: Date;
    var admissionServedAt: Date;
    var swapLoadElapsedMillis: UInt64;
    var executionStartedAt: Date;
    var terminalReceivedAt: Date?;

    init(requestArrivedAt: Date, admissionServedAt: Date) {
        self.requestArrivedAt = requestArrivedAt;
        self.admissionServedAt = admissionServedAt;
        self.swapLoadElapsedMillis = 0;
        self.executionStartedAt = admissionServedAt;
        self.terminalReceivedAt = nil;
    }

    static func elapsedMillis(since earlierDate: Date) -> UInt64 {
        return UInt64(max(0, Date().timeIntervalSince(earlierDate) * 1_000));
    }

    static func millisBetween(_ earlierDate: Date, _ laterDate: Date) -> UInt64 {
        return UInt64(max(0, laterDate.timeIntervalSince(earlierDate) * 1_000));
    }

    func performanceRecord(
        imageGenerationCommand: ImageGenerationCommand,
        completionOutcome: String,
        workerReportedElapsedMillis: UInt64,
        encodedImageBytes: UInt64?,
        mlxPeakMemoryBytes: UInt64?,
        mlxActiveMemoryBytes: UInt64?
    ) -> ImageGenerationPerformanceRecord {
        let finalizedAt: Date = Date();
        let terminalReceivedAt: Date = self.terminalReceivedAt ?? finalizedAt;
        return ImageGenerationPerformanceRecord(
            operation: "image_generation",
            timestampMillis: UInt64(finalizedAt.timeIntervalSince1970 * 1_000),
            requestId: imageGenerationCommand.requestId.value(),
            modelId: imageGenerationCommand.model,
            widthPixels: imageGenerationCommand.settings.widthPixels,
            heightPixels: imageGenerationCommand.settings.heightPixels,
            steps: imageGenerationCommand.settings.steps,
            completionOutcome: completionOutcome,
            totalElapsedMillis: ImageGenerationExecutionTiming.millisBetween(
                self.admissionServedAt, finalizedAt),
            queueWaitElapsedMillis: ImageGenerationExecutionTiming.millisBetween(
                self.requestArrivedAt, self.admissionServedAt),
            swapLoadElapsedMillis: self.swapLoadElapsedMillis,
            executionElapsedMillis: ImageGenerationExecutionTiming.millisBetween(
                self.executionStartedAt, terminalReceivedAt),
            finalizationElapsedMillis: ImageGenerationExecutionTiming.millisBetween(
                terminalReceivedAt, finalizedAt),
            workerReportedElapsedMillis: workerReportedElapsedMillis,
            encodedImageBytes: encodedImageBytes,
            mlxPeakMemoryBytes: mlxPeakMemoryBytes,
            mlxActiveMemoryBytes: mlxActiveMemoryBytes);
    }
}

/// The abandonment signal one image requester raises when its client stops
/// waiting; the executor diverts into the bounded cancellation path.
enum ImageGenerationClientAbandonment: Error, Equatable {

    case abandonedByClient;
}

extension Result where Success == ImageGenerationOutput, Failure == ImageGenerationExecutionError {

    var isCompleted: Bool {
        switch (self) {
        case .success: return true;
        case .failure: return false;
        }
    }

    var encodedImageByteCount: UInt64? {
        switch (self) {
        case let .success(imageGenerationOutput):
            return UInt64(imageGenerationOutput.generatedImage.encodedBytes.count);
        case .failure:
            return nil;
        }
    }
}
