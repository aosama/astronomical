import Foundation;

import IpcProtocol;

/// Applies one typed worker event to supervisor-owned health state.
///
/// Migrates the process-scoped arms of
/// apps/supervisor/src/worker_event_handler.rs. Events stay valid by their
/// own payload in this shell: readiness, idle, memory samples, and the
/// startup runtime acknowledgement arrive before any request exists. Events
/// that require an active request or a dedicated wait loop are protocol
/// violations here, exactly as the Rust handler rejects them outside their
/// wait.
public enum WorkerEventHandler {

    public static func handle(
        _ workerEvent: WorkerEvent,
        healthState: WorkerHealthState
    ) throws -> Void {
        switch (workerEvent) {
        case let .runtimeFeatureConfigurationApplied(workerRuntimeFeatureConfiguration):
            return try WorkerEventHandler.applyRuntimeFeatureConfiguration(
                workerRuntimeFeatureConfiguration,
                healthState: healthState);
        case let .ready(modelId, capabilities):
            return try WorkerEventHandler.applyReady(
                modelId: modelId,
                capabilities: capabilities,
                healthState: healthState);
        case let .idle(
            machineMlxMemoryCeilingBytes,
            effectiveMlxMemoryCeilingBytes,
            minimumMlxMemoryCeilingBytes):
            return try WorkerEventHandler.applyIdle(
                machineMlxMemoryCeilingBytes: machineMlxMemoryCeilingBytes,
                effectiveMlxMemoryCeilingBytes: effectiveMlxMemoryCeilingBytes,
                minimumMlxMemoryCeilingBytes: minimumMlxMemoryCeilingBytes,
                healthState: healthState);
        case let .mlxMemorySample(mlxMemorySnapshot, expertResidency):
            return try WorkerEventHandler.applyMlxMemorySample(
                mlxMemorySnapshot: mlxMemorySnapshot,
                expertResidency: expertResidency,
                healthState: healthState);
        case .modelSwapped:
            throw WorkerControlError.workerProtocolViolation(
                description: "model swap acknowledgement outside model swap wait");
        case .modelSwapFailed:
            throw WorkerControlError.workerProtocolViolation(
                description: "model swap failure outside model swap wait");
        case .promptCacheCleared:
            throw WorkerControlError.workerProtocolViolation(
                description: "prompt-cache clear acknowledgement outside cache-clear wait");
        case .failed:
            throw WorkerControlError.workerProtocolViolation(
                description: "failure without an active request");
        case .output, .prefillProgress, .generationPreparationStarted, .generationProgress,
             .firstDecodeCompleted, .promptWorkReuse, .completed, .generationFinalized:
            throw WorkerControlError.workerProtocolViolation(
                description: "chat event without an active chat request");
        case .imageGenerationProgress, .imageGenerationCompleted, .imageGenerationFailed,
             .imageGenerationFinalized, .embeddingsCompleted, .embeddingsFailed,
             .embeddingsFinalized:
            throw WorkerControlError.workerProtocolViolation(
                description: "image or embeddings event without an active request");
        case let .mlxMemoryLimitChanged(
            effectiveMlxMemoryCeilingBytes,
            minimumMlxMemoryCeilingBytes,
            expertMemoryMode,
            mlxMemorySnapshot,
            expertResidency):
            return try healthState.apply({ (snapshot: inout WorkerHealthSnapshot) in
                snapshot.effectiveMlxMemoryCeilingBytes = effectiveMlxMemoryCeilingBytes;
                snapshot.minimumMlxMemoryCeilingBytes = minimumMlxMemoryCeilingBytes;
                snapshot.pendingMlxMemoryCeilingBytes = nil;
                snapshot.mlxMemoryLimitError = nil;
                snapshot.latestMlxMemorySnapshot = mlxMemorySnapshot;
                snapshot.expertMemoryMode = snapshot.readyModelId.map({ _ in expertMemoryMode });
                if let expertResidency = expertResidency {
                    snapshot.expertResidency = expertResidency;
                }
            });
        case let .mlxMemoryLimitRejected(_, minimumMlxMemoryCeilingBytes, _, reason):
            return try healthState.apply({ (snapshot: inout WorkerHealthSnapshot) in
                snapshot.minimumMlxMemoryCeilingBytes = minimumMlxMemoryCeilingBytes;
                snapshot.pendingMlxMemoryCeilingBytes = nil;
                snapshot.mlxMemoryLimitError = reason;
            });
        case .expertMemoryModeChanged, .persistentPromptCacheStats:
            throw WorkerControlError.workerProtocolViolation(
                description: "live memory or cache event before its supervisor surface is wired");
        }
    }

    /// Records the startup feature policy acknowledged by this exact worker,
    /// rejecting acknowledgements that contradict the published model.
    private static func applyRuntimeFeatureConfiguration(
        _ workerRuntimeFeatureConfiguration: WorkerRuntimeFeatureConfiguration,
        healthState: WorkerHealthState
    ) throws -> Void {
        return try healthState.apply({ (snapshot: inout WorkerHealthSnapshot) throws -> Void in
            let acknowledgedModelId: String? = workerRuntimeFeatureConfiguration.loadedModel.map({
                (loadedModel: WorkerLoadedModelRuntimeConfiguration) -> String in
                return loadedModel.modelId();
            });
            if acknowledgedModelId != snapshot.readyModelId {
                throw WorkerControlError.workerProtocolViolation(
                    description: "runtime policy acknowledgement does not match the published model");
            }
            if let previousConfiguration: WorkerRuntimeFeatureConfiguration = snapshot.workerRuntimeFeatureConfiguration,
               previousConfiguration.loadedModel != workerRuntimeFeatureConfiguration.loadedModel {
                throw WorkerControlError.workerProtocolViolation(
                    description: "runtime policy changed without an atomic model transition");
            }
            snapshot.workerRuntimeFeatureConfiguration = workerRuntimeFeatureConfiguration;
        });
    }

    private static func applyReady(
        modelId: String,
        capabilities: WorkerModelCapabilities,
        healthState: WorkerHealthState
    ) throws -> Void {
        if healthState.hasAcknowledgedLifecycle() {
            throw WorkerControlError.workerProtocolViolation(description: "duplicate worker readiness");
        }
        healthState.markLifecycleAcknowledged();
        healthState.publish(WorkerHealthSnapshot.readyWithModel(
            modelId: modelId,
            capabilities: capabilities));
    }

    private static func applyIdle(
        machineMlxMemoryCeilingBytes: UInt64,
        effectiveMlxMemoryCeilingBytes: UInt64,
        minimumMlxMemoryCeilingBytes: UInt64,
        healthState: WorkerHealthState
    ) throws -> Void {
        if healthState.hasAcknowledgedLifecycle() {
            throw WorkerControlError.workerProtocolViolation(description: "duplicate worker idle event");
        }
        healthState.markLifecycleAcknowledged();
        healthState.publish(WorkerHealthSnapshot.readyWithoutModel(
            machineMlxMemoryCeilingBytes: machineMlxMemoryCeilingBytes,
            effectiveMlxMemoryCeilingBytes: effectiveMlxMemoryCeilingBytes,
            minimumMlxMemoryCeilingBytes: minimumMlxMemoryCeilingBytes));
    }

    private static func applyMlxMemorySample(
        mlxMemorySnapshot: WorkerMlxMemorySnapshot?,
        expertResidency: WorkerExpertResidencySnapshot?,
        healthState: WorkerHealthState
    ) throws -> Void {
        // An absent snapshot clears the stale observation; residency only
        // ever accumulates, matching the Rust publisher helpers.
        return try healthState.apply({ (snapshot: inout WorkerHealthSnapshot) throws -> Void in
            snapshot.latestMlxMemorySnapshot = mlxMemorySnapshot;
            if let workerExpertResidency: WorkerExpertResidencySnapshot = expertResidency {
                snapshot.expertResidency = workerExpertResidency;
            }
        });
    }
}
