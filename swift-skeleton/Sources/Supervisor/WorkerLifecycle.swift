import Foundation;

import IpcProtocol;

/// Supervisor-side worker lifecycle ownership: the shutdown path, the
/// containment path for a worker that failed, and the relaunch recovery
/// acknowledgement.
///
/// Migrates the generation-free core of
/// apps/supervisor/src/worker_containment.rs. There is no active request in
/// this shell, so the fail-active-generation arms join with the generation
/// slices.
public enum WorkerLifecycle {

    /// Publishes the shutdown health transition and closes the worker
    /// process, returning the termination outcome for the shutdown reply.
    public static func shutdown(
        workerProcess: WorkerProcess,
        healthState: WorkerHealthState
    ) throws -> WorkerTerminationOutcome {
        healthState.publish(.unavailable(.unavailable));
        if workerProcess.hasLivingProcess() {
            return try workerProcess.close();
        }
        return .graceful(processExitSuccessful: true);
    }

    /// Terminates an untrusted worker and marks health unavailable. Cleanup
    /// failures are reported but never masked: the worker is gone from
    /// health either way.
    public static func containFailure(
        workerProcess: WorkerProcess,
        healthState: WorkerHealthState,
        operationFailure: Error
    ) -> Void {
        WorkerLifecycle.writeContainmentLine(
            "worker failed (\(operationFailure)); terminating local worker process");
        do {
            let terminationOutcome: WorkerTerminationOutcome = try workerProcess.forceTerminate();
            WorkerLifecycle.writeContainmentLine(
                "local worker process terminated after failure: \(terminationOutcome)");
        } catch {
            WorkerLifecycle.writeContainmentLine(
                "failed to terminate local worker process: \(error)");
        }
        healthState.publish(.unavailable(.unavailable));
    }

    /// Restarts the worker from its exact launch inputs and waits a bounded
    /// time for the replacement's readiness and runtime-policy
    /// acknowledgement, then publishes them. The event pump keeps working
    /// across the relaunch because it holds the process reference and re-reads
    /// from it every iteration.
    public static func relaunchAfterFailure(
        workerProcess: WorkerProcess,
        eventPump: WorkerEventPump,
        healthState: WorkerHealthState,
        recoveryAcknowledgementTimeout: TimeInterval
    ) throws -> Void {
        try workerProcess.relaunchAfterTermination();
        eventPump.resumeAfterRelaunch();
        let expectedConfigurationGeneration: String? =
            workerProcess.expectedConfigurationGeneration();
        let recoveryOutcome: RecoveryAcknowledgement = try WorkerLifecycle.readRecoveryAcknowledgement(
            eventPump: eventPump,
            healthState: healthState,
            expectedConfigurationGeneration: expectedConfigurationGeneration,
            recoveryAcknowledgementTimeout: recoveryAcknowledgementTimeout);
        try WorkerLifecycle.publishRecoveryAcknowledgement(
            healthState: healthState,
            recoveryOutcome: recoveryOutcome);
    }

    private struct RecoveryAcknowledgement {
        var readinessEvent: WorkerEvent;
        var runtimeConfiguration: WorkerRuntimeFeatureConfiguration?;
    }

    /// Drains replacement-worker events until readiness plus the expected
    /// policy acknowledgement arrive, rejecting duplicates and mismatches.
    private static func readRecoveryAcknowledgement(
        eventPump: WorkerEventPump,
        healthState: WorkerHealthState,
        expectedConfigurationGeneration: String?,
        recoveryAcknowledgementTimeout: TimeInterval
    ) throws -> RecoveryAcknowledgement {
        var readinessEvent: WorkerEvent?;
        var runtimeConfiguration: WorkerRuntimeFeatureConfiguration?;
        let waitDeadline: Date = Date().addingTimeInterval(recoveryAcknowledgementTimeout);
        while true {
            if let acknowledgedReadiness: WorkerEvent = readinessEvent,
               expectedConfigurationGeneration == nil || runtimeConfiguration != nil {
                return RecoveryAcknowledgement(
                    readinessEvent: acknowledgedReadiness,
                    runtimeConfiguration: runtimeConfiguration);
            }
            let remainingWait: TimeInterval = waitDeadline.timeIntervalSinceNow;
            if remainingWait <= 0 {
                throw WorkerControlError.candidateAcknowledgementTimeout(
                    acknowledgementTimeoutMillis: UInt64(
                        (recoveryAcknowledgementTimeout * 1000).rounded()));
            }
            // `nil` is exclusively a bounded-wait expiry; closure and read
            // failures surface as thrown errors from the pump.
            guard let workerEvent: WorkerEvent = try eventPump.nextEvent(within: remainingWait) else {
                throw WorkerControlError.candidateAcknowledgementTimeout(
                    acknowledgementTimeoutMillis: UInt64(
                        (recoveryAcknowledgementTimeout * 1000).rounded()));
            }
            switch (workerEvent) {
            case .idle, .ready:
                if readinessEvent != nil {
                    throw WorkerControlError.workerProtocolViolation(
                        description: "replacement worker emitted duplicate readiness");
                }
                readinessEvent = workerEvent;
            case let .runtimeFeatureConfigurationApplied(configuration):
                if expectedConfigurationGeneration != configuration.configurationGeneration {
                    throw WorkerControlError.workerProtocolViolation(
                        description: "replacement worker runtime configuration mismatch");
                }
                if runtimeConfiguration != nil {
                    throw WorkerControlError.workerProtocolViolation(
                        description: "replacement worker emitted duplicate runtime configuration");
                }
                runtimeConfiguration = configuration;
            case .mlxMemorySample, .expertMemoryModeChanged, .persistentPromptCacheStats:
                continue;
            default:
                throw WorkerControlError.workerProtocolViolation(
                    description: "replacement worker emitted an unexpected startup event");
            }
        }
    }

    /// Cross-checks the paired acknowledgements and publishes the recovered
    /// snapshot; the replacement is ready and its policy recorded.
    private static func publishRecoveryAcknowledgement(
        healthState: WorkerHealthState,
        recoveryOutcome: RecoveryAcknowledgement
    ) throws -> Void {
        let runtimeConfiguration: WorkerRuntimeFeatureConfiguration? =
            recoveryOutcome.runtimeConfiguration;
        switch (recoveryOutcome.readinessEvent, runtimeConfiguration) {
        case (.idle, .some(let configuration)) where configuration.loadedModel != nil:
            throw WorkerControlError.workerProtocolViolation(
                description: "idle replacement worker acknowledged a loaded model");
        case let (.ready(modelId, _), .some(configuration))
            where configuration.loadedModel?.modelId() != modelId:
            throw WorkerControlError.workerProtocolViolation(
                description: "ready replacement worker policy did not match its model");
        default:
            break;
        }
        var recoveredSnapshot: WorkerHealthSnapshot;
        switch (recoveryOutcome.readinessEvent) {
        case let .idle(
            machineMlxMemoryCeilingBytes,
            effectiveMlxMemoryCeilingBytes,
            minimumMlxMemoryCeilingBytes):
            recoveredSnapshot = WorkerHealthSnapshot.readyWithoutModel(
                machineMlxMemoryCeilingBytes: machineMlxMemoryCeilingBytes,
                effectiveMlxMemoryCeilingBytes: effectiveMlxMemoryCeilingBytes,
                minimumMlxMemoryCeilingBytes: minimumMlxMemoryCeilingBytes);
        case let .ready(modelId, capabilities):
            recoveredSnapshot = WorkerHealthSnapshot.readyWithModel(
                modelId: modelId,
                capabilities: capabilities);
        default:
            throw WorkerControlError.workerProtocolViolation(
                description: "replacement worker acknowledgement lost readiness");
        }
        recoveredSnapshot.workerRuntimeFeatureConfiguration = runtimeConfiguration;
        healthState.publish(recoveredSnapshot);
        healthState.markLifecycleAcknowledged();
    }

    /// Containment events go to stderr so an attached operator sees them live.
    static func writeContainmentLine(_ line: String) -> Void {
        FileHandle.standardError.write(Data("astronomicald: \(line)\n".utf8));
    }
}
