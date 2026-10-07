import Foundation;

import IpcProtocol;

/// Applies the InitializeWorker runtime-policy acknowledgement before any
/// further worker command is legal.
///
/// Migrates apps/supervisor/src/worker_startup_runtime.rs: launch writes
/// InitializeWorker and returns as soon as the process is spawned, so a
/// worker can report lifecycle events while that acknowledgement is still in
/// the pipe. Waiting for the acknowledgement here — once per worker process —
/// keeps a later swap from treating a missing health generation as "policy
/// already ready" and then containing the worker for an illegal loaded-model
/// change. Live memory updates and rejected swaps must not wait again, which
/// is why the applied flag latches.
public enum WorkerStartupRuntime {

    /// Blocks until the worker's readiness and runtime-policy acknowledgement
    /// are recorded in health, the wait already happened for this process, or
    /// the worker launched with no startup policy at all — in which case
    /// readiness alone is awaited, exactly as the Rust worker loop consumes
    /// an idle event without a policy wait.
    public static func waitForStartupRuntimeConfiguration(
        workerProcess: WorkerProcess,
        eventPump: WorkerEventPump,
        healthState: WorkerHealthState,
        modelLoadTimeout: TimeInterval
    ) throws -> Void {
        if workerProcess.isStartupRuntimeConfigurationApplied() {
            return;
        }
        let awaitsRuntimeConfiguration: Bool = workerProcess.expectedConfigurationGeneration() != nil;
        if awaitsRuntimeConfiguration && healthState.hasRuntimeFeatureConfiguration() {
            workerProcess.markStartupRuntimeConfigurationApplied();
            return;
        }
        if !awaitsRuntimeConfiguration && healthState.hasAcknowledgedLifecycle() {
            workerProcess.markStartupRuntimeConfigurationApplied();
            return;
        }
        let waitDeadline: Date = Date().addingTimeInterval(modelLoadTimeout);
        while true {
            if awaitsRuntimeConfiguration && healthState.hasRuntimeFeatureConfiguration() {
                workerProcess.markStartupRuntimeConfigurationApplied();
                return;
            }
            if !awaitsRuntimeConfiguration && healthState.hasAcknowledgedLifecycle() {
                workerProcess.markStartupRuntimeConfigurationApplied();
                return;
            }
            let remainingWait: TimeInterval = waitDeadline.timeIntervalSinceNow;
            if remainingWait <= 0 {
                throw WorkerControlError.modelLoadTimeout(
                    modelLoadTimeoutMillis: UInt64((modelLoadTimeout * 1000).rounded()));
            }
            // `nil` is exclusively a bounded-wait expiry: stream closure and
            // read failures surface as thrown errors from the pump.
            guard let workerEvent: WorkerEvent = try eventPump.nextEvent(within: remainingWait) else {
                throw WorkerControlError.modelLoadTimeout(
                    modelLoadTimeoutMillis: UInt64((modelLoadTimeout * 1000).rounded()));
            }
            try WorkerEventHandler.handle(workerEvent, healthState: healthState);
        }
    }
}
