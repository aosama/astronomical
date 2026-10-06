import Foundation;

import IpcProtocol;

/// Generation admission preparation: swap the worker onto the requested
/// model on demand, then the generate command goes out.
///
/// Migrates apps/supervisor/src/worker_generate.rs's model-selection core. A
/// request whose model is already resident goes straight to the command; a
/// request for a cataloged model triggers SwapModel and its bounded
/// acknowledgement wait; a request for an unmapped model with no resident
/// model is rejected before any command. A clean swap rejection becomes a
/// modelLoadFailed start error; every control failure is contained by the
/// owning supervisor before it surfaces as workerUnavailable.
public enum WorkerGenerate {

    public static func prepareResidentModel(
        _ generationCommand: ChatGenerationCommand,
        workerProcess: WorkerProcess,
        eventPump: WorkerEventPump,
        healthState: WorkerHealthState,
        modelPolicyCatalog: Dictionary<String, RuntimeModelPolicy>,
        modelLoadTimeout: TimeInterval,
        containment: (Error) -> Void
    ) throws -> Void {
        let residentModelId: String? = healthState.currentSnapshot().readyModelId;
        if residentModelId == generationCommand.model {
            return;
        }
        guard let modelPolicy: RuntimeModelPolicy = modelPolicyCatalog[generationCommand.model] else {
            if residentModelId == nil {
                // The worker cannot serve an unmapped model from cold, and no
                // resident model protects the request either.
                throw GenerationStartError.workerUnavailable;
            }
            // A resident model stays in place; the worker itself rejects the
            // generation for the wrong model identity.
            return;
        }
        let expectedConfigurationGeneration: String? = healthState
            .currentSnapshot()
            .workerRuntimeFeatureConfiguration?
            .configurationGeneration;
        let expectedModelRuntimeConfiguration: WorkerLoadedModelRuntimeConfiguration =
            modelPolicy.workerModelConfiguration.runtimeConfiguration();
        do {
            try workerProcess.sendCommand(.swapModel(
                modelDirectory: modelPolicy.modelDirectory.string,
                modelConfiguration: modelPolicy.workerModelConfiguration));
        } catch {
            containment(error);
            throw GenerationStartError.workerUnavailable;
        }
        let modelSwapOutcome: ModelSwapWaitOutcome;
        do {
            modelSwapOutcome = try WorkerModelSwap.waitForModelSwap(
                eventPump: eventPump,
                healthState: healthState,
                expectedConfigurationGeneration: expectedConfigurationGeneration,
                expectedModelRuntimeConfiguration: expectedModelRuntimeConfiguration,
                modelLoadTimeout: modelLoadTimeout);
        } catch {
            containment(error);
            throw GenerationStartError.workerUnavailable;
        }
        switch (modelSwapOutcome) {
        case .loaded:
            return;
        case let .rejected(modelLoadFailureReason):
            throw GenerationStartError.modelLoadFailed(modelLoadFailureReason: modelLoadFailureReason);
        }
    }
}
