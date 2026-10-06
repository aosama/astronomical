import Foundation;

import IpcProtocol;

/// Result of waiting for a worker model-swap acknowledgement.
public enum ModelSwapWaitOutcome: Equatable {

    /// The worker acknowledged the swap and its policy is recorded.
    case loaded;

    /// The worker rejected the requested model while remaining responsive.
    case rejected(modelLoadFailureReason: String);
}

/// Drains worker events until a model swap succeeds or is rejected.
///
/// Migrates apps/supervisor/src/worker_model_swap.rs. Process-scoped
/// telemetry and configuration events may have been queued before SwapModel;
/// they remain valid while the swap acknowledgement is pending and must
/// update supervisor health rather than terminate the worker. The swap
/// acknowledgement and its runtime-policy acknowledgement are staged and
/// published together so health never shows a model without its policy.
public enum WorkerModelSwap {

    /// Blocks until the swap completes, is rejected, or the bounded wait
    /// expires. `nil` from the pump is exclusively a deadline expiry; stream
    /// closure and read failures surface as thrown errors.
    public static func waitForModelSwap(
        eventPump: WorkerEventPump,
        healthState: WorkerHealthState,
        expectedConfigurationGeneration: String?,
        expectedModelRuntimeConfiguration: WorkerLoadedModelRuntimeConfiguration,
        modelLoadTimeout: TimeInterval
    ) throws -> ModelSwapWaitOutcome {
        var stagedModelSwapEvent: WorkerEvent?;
        var stagedRuntimeConfiguration: WorkerRuntimeFeatureConfiguration?;
        let waitDeadline: Date = Date().addingTimeInterval(modelLoadTimeout);
        while true {
            let remainingWait: TimeInterval = waitDeadline.timeIntervalSinceNow;
            if remainingWait <= 0 {
                throw WorkerControlError.modelLoadTimeout(
                    modelLoadTimeoutMillis: UInt64((modelLoadTimeout * 1000).rounded()));
            }
            guard let workerEvent: WorkerEvent = try eventPump.nextEvent(within: remainingWait) else {
                throw WorkerControlError.modelLoadTimeout(
                    modelLoadTimeoutMillis: UInt64((modelLoadTimeout * 1000).rounded()));
            }
            switch (workerEvent) {
            case .modelSwapped:
                if stagedModelSwapEvent != nil {
                    throw WorkerControlError.workerProtocolViolation(
                        description: "duplicate model swap acknowledgement");
                }
                try WorkerModelSwap.validateModelSwapEvent(
                    workerEvent,
                    expectedRuntimeConfiguration: expectedModelRuntimeConfiguration);
                stagedModelSwapEvent = workerEvent;
            case let .modelSwapFailed(loadedModelRemainsReady, modelLoadFailureReason):
                // A failed first load leaves a healthy model-less worker. A
                // failed replacement may leave the prior model ready; its
                // existing health snapshot must remain intact in that case.
                if !loadedModelRemainsReady {
                    try healthState.apply({ (snapshot: inout WorkerHealthSnapshot) throws -> Void in
                        snapshot = WorkerHealthSnapshot.readyWithoutModel(
                            machineMlxMemoryCeilingBytes: snapshot.machineMlxMemoryCeilingBytes,
                            effectiveMlxMemoryCeilingBytes: snapshot.effectiveMlxMemoryCeilingBytes,
                            minimumMlxMemoryCeilingBytes: snapshot.minimumMlxMemoryCeilingBytes);
                    });
                }
                return .rejected(modelLoadFailureReason: modelLoadFailureReason);
            case let .runtimeFeatureConfigurationApplied(configuration):
                var isStagedRuntimeAcknowledgement: Bool = false;
                if let expectedGeneration: String = expectedConfigurationGeneration {
                    if configuration.configurationGeneration != expectedGeneration
                        || configuration.loadedModel != expectedModelRuntimeConfiguration {
                        throw WorkerControlError.workerProtocolViolation(
                            description: "model swap runtime policy acknowledgement mismatch");
                    }
                    if stagedRuntimeConfiguration != nil {
                        throw WorkerControlError.workerProtocolViolation(
                            description: "duplicate model swap runtime policy acknowledgement");
                    }
                    stagedRuntimeConfiguration = configuration;
                    isStagedRuntimeAcknowledgement = true;
                }
                if !isStagedRuntimeAcknowledgement {
                    try WorkerEventHandler.handle(workerEvent, healthState: healthState);
                }
            default:
                // The central handler accepts process-scoped updates and
                // rejects generation events whose active-request contract is
                // absent, so both wait loops enforce the same protocol rules.
                try WorkerEventHandler.handle(workerEvent, healthState: healthState);
            }
            let policyIsReady: Bool = expectedConfigurationGeneration == nil
                || stagedRuntimeConfiguration != nil;
            if stagedModelSwapEvent != nil && policyIsReady {
                guard let completedSwapEvent: WorkerEvent = stagedModelSwapEvent else {
                    throw WorkerControlError.workerProtocolViolation(
                        description: "model swap acknowledgement staging failed");
                }
                return try WorkerModelSwap.publishStagedModelSwap(
                    healthState: healthState,
                    modelSwapEvent: completedSwapEvent,
                    runtimeConfiguration: stagedRuntimeConfiguration);
            }
        }
    }

    /// Rejects swap acknowledgements whose identity or capability geometry
    /// does not match the requested model policy.
    private static func validateModelSwapEvent(
        _ modelSwapEvent: WorkerEvent,
        expectedRuntimeConfiguration: WorkerLoadedModelRuntimeConfiguration
    ) throws -> Void {
        guard case let .modelSwapped(modelId, capabilities, _, _) = modelSwapEvent else {
            throw WorkerControlError.workerProtocolViolation(
                description: "non-model event staged as model swap acknowledgement");
        }
        if modelId != expectedRuntimeConfiguration.modelId() {
            throw WorkerControlError.workerProtocolViolation(
                description: "model swap identity acknowledgement mismatch");
        }
        var capabilitiesMatchPolicy: Bool = false;
        switch (expectedRuntimeConfiguration) {
        case let .autoregressive(configuration):
            if let chatCapabilities: ChatModelCapabilities = capabilities.chat {
                let maximumContextTokens: UInt32 = configuration.maximumContextTokens;
                let saturatingContextMinusOne: UInt32 = maximumContextTokens > 0
                    ? maximumContextTokens - 1
                    : 0;
                capabilitiesMatchPolicy =
                    chatCapabilities.contextWindow == maximumContextTokens
                    && chatCapabilities.maxOutputTokens == configuration.maximumOutputTokens
                    && chatCapabilities.maxInputTokens == saturatingContextMinusOne;
            }
            capabilitiesMatchPolicy = capabilitiesMatchPolicy && capabilities.imageGeneration == nil;
        case .flux2Klein, .qwenImage21:
            capabilitiesMatchPolicy = capabilities.chat == nil && capabilities.imageGeneration != nil;
        case .embeddings:
            capabilitiesMatchPolicy = capabilities.chat == nil
                && capabilities.imageGeneration == nil
                && capabilities.embeddings != nil;
        }
        if !capabilitiesMatchPolicy {
            throw WorkerControlError.workerProtocolViolation(
                description: "model swap capabilities acknowledgement mismatch");
        }
    }

    /// Publishes the staged swap and its policy acknowledgement together.
    private static func publishStagedModelSwap(
        healthState: WorkerHealthState,
        modelSwapEvent: WorkerEvent,
        runtimeConfiguration: WorkerRuntimeFeatureConfiguration?
    ) throws -> ModelSwapWaitOutcome {
        guard case let .modelSwapped(
            modelId,
            capabilities,
            expertMemoryMode,
            minimumMlxMemoryCeilingBytes) = modelSwapEvent else {
            throw WorkerControlError.workerProtocolViolation(
                description: "non-model event staged as model swap acknowledgement");
        }
        try healthState.apply({ (snapshot: inout WorkerHealthSnapshot) throws -> Void in
            var replacementHealthSnapshot: WorkerHealthSnapshot = WorkerHealthSnapshot.readyWithReplacementModel(
                modelId: modelId,
                capabilities: capabilities,
                minimumMlxMemoryCeilingBytes: minimumMlxMemoryCeilingBytes,
                previousHealthSnapshot: snapshot);
            replacementHealthSnapshot.expertMemoryMode = expertMemoryMode;
            if let stagedConfiguration: WorkerRuntimeFeatureConfiguration = runtimeConfiguration {
                replacementHealthSnapshot.workerRuntimeFeatureConfiguration = stagedConfiguration;
            }
            snapshot = replacementHealthSnapshot;
        });
        return .loaded;
    }
}
