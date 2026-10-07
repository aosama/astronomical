import Foundation;

import IpcProtocol;

/// The transactional replacement of one trusted worker process with a
/// candidate launched from supervisor-resolved bootstrap settings.
///
/// Migrates apps/supervisor/src/worker_replacement.rs: the candidate is
/// launched and fully acknowledged before the trusted worker is touched, so
/// a rejected candidate never exposes partial startup state and the serving
/// worker keeps its exact previous health. The acknowledged runtime
/// configuration is the commit receipt the reload endpoint records as the
/// new effective generation.
extension WorkerSupervisor {

    /// Replaces the owned worker with a candidate that must acknowledge the
    /// candidate startup configuration's exact generation, mirroring
    /// WorkerHandle::restart_worker_with_startup_configuration. The REST
    /// application stays alive across the swap.
    public func restartWorkerWithStartupConfiguration(
        candidateWorkerExecutablePath: String,
        candidateWorkerArguments: Array<String>,
        candidateModelPolicyCatalog: Dictionary<String, RuntimeModelPolicy>,
        candidateStartupConfiguration: WorkerStartupConfiguration
    ) throws -> WorkerRuntimeFeatureConfiguration {
        if !self.isGenerationIdleForControlAction() {
            throw WorkerControlError.generationBusy;
        }
        self.stateLock.lock();
        let trustedWorkerProcess: WorkerProcess? = self.workerProcess;
        self.stateLock.unlock();
        guard let trustedWorkerProcess: WorkerProcess = trustedWorkerProcess else {
            throw WorkerControlError.missingActiveWorker;
        }
        let candidateWorkerProcess: WorkerProcess;
        do {
            candidateWorkerProcess = try WorkerProcess.launch(
                workerExecutablePath: candidateWorkerExecutablePath,
                arguments: candidateWorkerArguments,
                workerStartupConfiguration: candidateStartupConfiguration);
        } catch let candidateLaunchError {
            throw WorkerControlError.startWorker(
                underlyingDescription: String(describing: candidateLaunchError));
        }
        let candidateEventPump: WorkerEventPump = WorkerEventPump(workerProcess: candidateWorkerProcess);
        let candidateAcknowledgement: CandidateWorkerAcknowledgement;
        do {
            candidateAcknowledgement = try WorkerTransactionalReplacement.readCandidateAcknowledgement(
                candidateEventPump: candidateEventPump,
                expectedConfigurationGeneration: candidateStartupConfiguration.configurationGeneration,
                candidateModelPolicyCatalog: candidateModelPolicyCatalog,
                acknowledgementTimeout: self.modelLoadTimeout);
        } catch let candidateError {
            throw WorkerTransactionalReplacement.rejectCandidate(
                candidateWorkerProcess: candidateWorkerProcess,
                candidateError: candidateError);
        }
        return try WorkerTransactionalReplacement.commitCandidate(
            trustedWorkerProcess: trustedWorkerProcess,
            candidateWorkerProcess: candidateWorkerProcess,
            candidateEventPump: candidateEventPump,
            candidateModelPolicyCatalog: candidateModelPolicyCatalog,
            candidateAcknowledgement: candidateAcknowledgement,
            supervisor: self);
    }
}

/// One candidate's paired startup acknowledgements plus the process events
/// that arrived before the supervisor was ready to publish them.
struct CandidateWorkerAcknowledgement {

    var readinessEvent: WorkerEvent;
    var runtimeFeatureConfiguration: WorkerRuntimeFeatureConfiguration;
}

/// The launch-acknowledge-commit transaction over one candidate worker.
enum WorkerTransactionalReplacement {

    /// The number of deferred candidate process events tolerated before the
    /// candidate is judged chatty beyond protocol and rejected.
    private static let maximumDeferredCandidateProcessEventCount: Int = 64;

    /// Drains candidate events until readiness plus the expected generation's
    /// runtime-policy acknowledgement arrive, mirroring
    /// read_candidate_acknowledgement. Duplicate acknowledgements, a foreign
    /// generation, or any non-startup event reject the candidate.
    fileprivate static func readCandidateAcknowledgement(
        candidateEventPump: WorkerEventPump,
        expectedConfigurationGeneration: String,
        candidateModelPolicyCatalog: Dictionary<String, RuntimeModelPolicy>,
        acknowledgementTimeout: TimeInterval
    ) throws -> CandidateWorkerAcknowledgement {
        var readinessEvent: WorkerEvent?;
        var runtimeFeatureConfiguration: WorkerRuntimeFeatureConfiguration?;
        var deferredProcessEventCount: Int = 0;
        let waitDeadline: Date = Date().addingTimeInterval(acknowledgementTimeout);
        while true {
            let remainingWait: TimeInterval = waitDeadline.timeIntervalSinceNow;
            if remainingWait <= 0 {
                throw WorkerControlError.candidateAcknowledgementTimeout(
                    acknowledgementTimeoutMillis: UInt64(
                        (acknowledgementTimeout * 1000).rounded()));
            }
            // `nil` is exclusively a bounded-wait expiry; stream closure and
            // read failures surface as thrown errors from the pump.
            guard let workerEvent: WorkerEvent = try candidateEventPump.nextEvent(within: remainingWait) else {
                throw WorkerControlError.candidateAcknowledgementTimeout(
                    acknowledgementTimeoutMillis: UInt64(
                        (acknowledgementTimeout * 1000).rounded()));
            }
            switch (workerEvent) {
            case .idle, .ready:
                if readinessEvent != nil {
                    throw WorkerControlError.candidateProtocolViolation(
                        description: "candidate emitted duplicate initial readiness")
                }
                readinessEvent = workerEvent
            case let .runtimeFeatureConfigurationApplied(configuration):
                if runtimeFeatureConfiguration != nil {
                    throw WorkerControlError.candidateProtocolViolation(
                        description: "candidate emitted duplicate runtime configuration")
                }
                if configuration.configurationGeneration != expectedConfigurationGeneration {
                    throw WorkerControlError.candidateConfigurationGenerationMismatch
                }
                runtimeFeatureConfiguration = configuration
            case .mlxMemorySample, .expertMemoryModeChanged, .persistentPromptCacheStats:
                deferredProcessEventCount += 1
                if deferredProcessEventCount >= maximumDeferredCandidateProcessEventCount {
                    throw WorkerControlError.candidateProtocolViolation(
                        description: "candidate emitted too many process events before acknowledgement")
                }
            default:
                throw WorkerControlError.unexpectedCandidateEvent(
                    unexpectedWorkerEventSummary: workerEvent.diagnosticSummary())
            }
            if let acknowledgedReadiness: WorkerEvent = readinessEvent,
               let acknowledgedConfiguration: WorkerRuntimeFeatureConfiguration = runtimeFeatureConfiguration {
                try WorkerTransactionalReplacement.validateCandidateModelBinding(
                    readinessEvent: acknowledgedReadiness,
                    runtimeConfiguration: acknowledgedConfiguration,
                    candidateModelPolicyCatalog: candidateModelPolicyCatalog);
                return CandidateWorkerAcknowledgement(
                    readinessEvent: acknowledgedReadiness,
                    runtimeFeatureConfiguration: acknowledgedConfiguration);
            }
        }
    }

    /// Rejects a candidate whose acknowledged model contradicts the candidate
    /// catalog: an idle candidate must carry no model, and a ready candidate
    /// must acknowledge exactly its catalog policy's runtime configuration.
    private static func validateCandidateModelBinding(
        readinessEvent: WorkerEvent,
        runtimeConfiguration: WorkerRuntimeFeatureConfiguration,
        candidateModelPolicyCatalog: Dictionary<String, RuntimeModelPolicy>
    ) throws -> Void {
        switch (readinessEvent) {
        case .idle where runtimeConfiguration.loadedModel == nil:
            return
        case let .ready(modelId, _):
            guard let loadedModel: WorkerLoadedModelRuntimeConfiguration = runtimeConfiguration.loadedModel else {
                throw WorkerControlError.candidateProtocolViolation(
                    description: "ready candidate did not acknowledge its loaded model policy")
            }
            guard let candidatePolicy: RuntimeModelPolicy = candidateModelPolicyCatalog[modelId] else {
                throw WorkerControlError.candidateProtocolViolation(
                    description: "ready candidate model is absent from the candidate catalog")
            }
            if loadedModel != candidatePolicy.workerModelConfiguration.runtimeConfiguration() {
                throw WorkerControlError.candidateProtocolViolation(
                    description: "ready candidate model disagrees with its acknowledged policy")
            }
            return
        case .idle:
            throw WorkerControlError.candidateProtocolViolation(
                description: "idle candidate acknowledged an unexpected loaded model")
        default:
            throw WorkerControlError.candidateProtocolViolation(
                description: "candidate readiness event is unsupported")
        }
    }

    /// Closes the trusted worker, swaps the candidate in, and publishes the
    /// candidate's acknowledged state, mirroring WorkerReplacement::execute's
    /// commit half. Any trusted-close failure contains both processes and
    /// publishes unavailable health before the error surfaces.
    fileprivate static func commitCandidate(
        trustedWorkerProcess: WorkerProcess,
        candidateWorkerProcess: WorkerProcess,
        candidateEventPump: WorkerEventPump,
        candidateModelPolicyCatalog: Dictionary<String, RuntimeModelPolicy>,
        candidateAcknowledgement: CandidateWorkerAcknowledgement,
        supervisor: WorkerSupervisor
    ) throws -> WorkerRuntimeFeatureConfiguration {
        do {
            _ = try trustedWorkerProcess.close();
        } catch let trustedCloseError {
            let cleanupDescription: String? = WorkerTransactionalReplacement.forceTerminateReporting(
                trustedWorkerProcess);
            supervisor.healthState.publish(.unavailable(.unavailable));
            if let cleanupDescription = cleanupDescription {
                throw WorkerControlError.operationAndCleanupFailed(
                    operationDescription: WorkerControlError.describe(trustedCloseError),
                    cleanupDescription: cleanupDescription);
            }
            throw trustedCloseError;
        }
        supervisor.stateLock.lock();
        supervisor.workerProcess = candidateWorkerProcess;
        supervisor.eventPump = candidateEventPump;
        supervisor.modelPolicyCatalog = candidateModelPolicyCatalog;
        supervisor.stateLock.unlock();
        do {
            try WorkerTransactionalReplacement.publishCandidateAcknowledgement(
                candidateAcknowledgement: candidateAcknowledgement,
                healthState: supervisor.healthState);
        } catch let publishError {
            let cleanupDescription: String? = WorkerTransactionalReplacement.forceTerminateReporting(
                candidateWorkerProcess);
            supervisor.healthState.publish(.unavailable(.unavailable));
            if let cleanupDescription = cleanupDescription {
                throw WorkerControlError.operationAndCleanupFailed(
                    operationDescription: WorkerControlError.describe(publishError),
                    cleanupDescription: cleanupDescription);
            }
            throw publishError;
        }
        return candidateAcknowledgement.runtimeFeatureConfiguration;
    }

    /// Contains one worker outright, returning a description when even the
    /// forced termination failed so callers can pair it with the operation
    /// error instead of masking it.
    private static func forceTerminateReporting(_ workerProcess: WorkerProcess) -> String? {
        do {
            _ = try workerProcess.forceTerminate();
            return nil;
        } catch {
            return String(describing: error);
        }
    }

    /// Rejects a candidate that failed its acknowledgement: close it, and
    /// pair the original error with a cleanup failure rather than masking it.
    fileprivate static func rejectCandidate(
        candidateWorkerProcess: WorkerProcess,
        candidateError: Error
    ) -> Error {
        do {
            _ = try candidateWorkerProcess.close();
            return candidateError;
        } catch {
            _ = try? candidateWorkerProcess.forceTerminate();
            return WorkerControlError.operationAndCleanupFailed(
                operationDescription: WorkerControlError.describe(candidateError),
                cleanupDescription: WorkerControlError.describe(error));
        }
    }

    /// Publishes the committed candidate's readiness, runtime policy, and
    /// acknowledged memory facts into health state. The lifecycle flag stays
    /// acknowledged from the previous worker; the candidate snapshot is
    /// published directly, exactly like the recovery path after a relaunch.
    private static func publishCandidateAcknowledgement(
        candidateAcknowledgement: CandidateWorkerAcknowledgement,
        healthState: WorkerHealthState
    ) throws -> Void {
        var candidateSnapshot: WorkerHealthSnapshot;
        switch (candidateAcknowledgement.readinessEvent) {
        case let .idle(
            machineMlxMemoryCeilingBytes,
            effectiveMlxMemoryCeilingBytes,
            minimumMlxMemoryCeilingBytes):
            candidateSnapshot = WorkerHealthSnapshot.readyWithoutModel(
                machineMlxMemoryCeilingBytes: machineMlxMemoryCeilingBytes,
                effectiveMlxMemoryCeilingBytes: effectiveMlxMemoryCeilingBytes,
                minimumMlxMemoryCeilingBytes: minimumMlxMemoryCeilingBytes);
        case let .ready(modelId, capabilities):
            candidateSnapshot = WorkerHealthSnapshot.readyWithModel(
                modelId: modelId,
                capabilities: capabilities);
        default:
            throw WorkerControlError.workerProtocolViolation(
                description: "candidate acknowledgement lost readiness");
        }
        candidateSnapshot.workerRuntimeFeatureConfiguration =
            candidateAcknowledgement.runtimeFeatureConfiguration;
        healthState.publish(candidateSnapshot);
    }
}
