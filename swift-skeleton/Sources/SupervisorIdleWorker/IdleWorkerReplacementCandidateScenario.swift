import Foundation;

import IpcProtocol;

/**
 * The scripted handshake of a transactional-replacement candidate, migrating
 * apps/supervisor/tests/fixtures/replacement_ready_worker.rs: the candidate
 * acknowledges the supervisor's InitializeWorker in shapes deliberately
 * keyed by the requested configuration generation, proving the supervisor
 * commits a swap only on the exact acknowledged pair it demanded.
 */
enum IdleWorkerReplacementCandidateScenario {

    static let CONFIGURATION_BEFORE_READY_GENERATION: String =
        String(repeating: "4", count: 64);
    static let GENERATION_EVENT_GENERATION: String = String(repeating: "3", count: 64);
    static let INCONSISTENT_READY_GENERATION: String = String(repeating: "5", count: 64);
    static let MATCHING_FLUX_GENERATION: String = String(repeating: "6", count: 64);
    static let MISMATCHED_ACKNOWLEDGED_GENERATION: String = String(repeating: "f", count: 64);
    static let FLUX_MODEL_ID: String = "FLUX.2-klein-4B";
    static let FLUX_ACKNOWLEDGED_REVISION: String = "reviewed-revision";
    static let UNACKNOWLEDGED_READY_MODEL_ID: String = "astronomical/unacknowledged-ready-model";
    static let CANDIDATE_PID_FILE_NAME: String = "replacement-candidate.pid";

    enum FixtureFailure: Error, CustomStringConvertible {

        case unexpectedFirstWorkerCommand

        var description: String {
            switch (self) {
            case .unexpectedFirstWorkerCommand:
                return "replacement candidate expected InitializeWorker as its first command";
            }
        }
    }

    /// Runs the generation-keyed candidate handshake, then consumes the
    /// remaining command stream silently until the supervisor closes it.
    static func runFixture(
        commandReader: ProtocolReader,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        guard let firstCommand: WorkerCommand = try commandReader.nextCommand() else {
            return;
        }
        guard case let .initializeWorker(startupConfiguration) = firstCommand else {
            throw FixtureFailure.unexpectedFirstWorkerCommand;
        }
        IdleWorkerReplacementCandidateScenario.writeCandidatePidFile(
            startupConfiguration.loggingDirectory);
        let acknowledgedGeneration: String = startupConfiguration.configurationGeneration;
        if (acknowledgedGeneration
            == IdleWorkerReplacementCandidateScenario.CONFIGURATION_BEFORE_READY_GENERATION) {
            try IdleWorkerReplacementCandidateScenario.sendRuntimeConfiguration(
                startupConfiguration,
                acknowledgedGeneration: acknowledgedGeneration,
                loadedModel: nil,
                eventWriter: eventWriter);
        }
        if (acknowledgedGeneration == IdleWorkerReplacementCandidateScenario.MATCHING_FLUX_GENERATION) {
            try eventWriter.sendEvent(.ready(
                modelId: IdleWorkerReplacementCandidateScenario.FLUX_MODEL_ID,
                capabilities: .imageGeneration(
                    imageGeneration: IdleWorkerReplacementCandidateScenario.fluxImageCapabilities())));
        } else if (acknowledgedGeneration
            == IdleWorkerReplacementCandidateScenario.INCONSISTENT_READY_GENERATION) {
            try eventWriter.sendEvent(.ready(
                modelId: IdleWorkerReplacementCandidateScenario.UNACKNOWLEDGED_READY_MODEL_ID,
                capabilities: .from(chatCapabilities: ChatModelCapabilities(
                    supportsReasoning: false,
                    supportsToolCalls: false,
                    hasVision: false,
                    maxInputTokens: 1,
                    maxOutputTokens: 1,
                    contextWindow: 2))));
        } else {
            try eventWriter.sendEvent(.idle(
                machineMlxMemoryCeilingBytes: 40_000_000_000,
                effectiveMlxMemoryCeilingBytes: 40_000_000_000,
                minimumMlxMemoryCeilingBytes: 1));
        }
        if (acknowledgedGeneration == IdleWorkerReplacementCandidateScenario.GENERATION_EVENT_GENERATION) {
            try eventWriter.sendEvent(.completed(
                requestId: RequestId(rawRequestId: 1),
                promptTokenCount: 1,
                generatedTokenCount: 0,
                reasoningTokenCount: 0,
                cachedTokenCount: 0,
                persistentPromptCacheDiagnostics: nil,
                reason: ChatGenerationCompletionReason.endOfSequence));
        } else if (acknowledgedGeneration == IdleWorkerReplacementCandidateScenario.MATCHING_FLUX_GENERATION) {
            try IdleWorkerReplacementCandidateScenario.sendRuntimeConfiguration(
                startupConfiguration,
                acknowledgedGeneration: acknowledgedGeneration,
                loadedModel: .flux2Klein(WorkerFlux2KleinModelConfiguration(
                    modelId: IdleWorkerReplacementCandidateScenario.FLUX_MODEL_ID,
                    modelFamily: .flux2Klein,
                    artifactRevision: IdleWorkerReplacementCandidateScenario.FLUX_ACKNOWLEDGED_REVISION)),
                eventWriter: eventWriter);
        } else if (acknowledgedGeneration
            == IdleWorkerReplacementCandidateScenario.INCONSISTENT_READY_GENERATION) {
            try IdleWorkerReplacementCandidateScenario.sendRuntimeConfiguration(
                startupConfiguration,
                acknowledgedGeneration: acknowledgedGeneration,
                loadedModel: nil,
                eventWriter: eventWriter);
        } else if (acknowledgedGeneration
            != IdleWorkerReplacementCandidateScenario.CONFIGURATION_BEFORE_READY_GENERATION) {
            try IdleWorkerReplacementCandidateScenario.sendRuntimeConfiguration(
                startupConfiguration,
                acknowledgedGeneration:
                    IdleWorkerReplacementCandidateScenario.MISMATCHED_ACKNOWLEDGED_GENERATION,
                loadedModel: nil,
                eventWriter: eventWriter);
        }
        while let _ = try commandReader.nextCommand() {}
    }

    /// Records the candidate's process identifier where the journeys read
    /// it to prove a rejected candidate was reaped.
    private static func writeCandidatePidFile(_ loggingDirectory: String) -> Void {
        let pidFilePath: String = loggingDirectory + "/"
            + IdleWorkerReplacementCandidateScenario.CANDIDATE_PID_FILE_NAME;
        try? String(ProcessInfo.processInfo.processIdentifier).write(
            toFile: pidFilePath,
            atomically: true,
            encoding: String.Encoding.utf8);
    }

    private static func sendRuntimeConfiguration(
        _ startupConfiguration: WorkerStartupConfiguration,
        acknowledgedGeneration: String,
        loadedModel: WorkerLoadedModelRuntimeConfiguration?,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        try eventWriter.sendEvent(.runtimeFeatureConfigurationApplied(
            WorkerRuntimeFeatureConfiguration(
                configurationGeneration: acknowledgedGeneration,
                persistentPromptCacheEnabled: startupConfiguration.persistentPromptCacheEnabled,
                promptCacheMaximumSizeBytes: startupConfiguration.globalPromptCacheMaximumSizeBytes,
                loadedModel: loadedModel)));
    }

    private static func fluxImageCapabilities() -> ImageGenerationCapabilities {
        return ImageGenerationCapabilities(
            minimumWidthPixels: 64,
            maximumWidthPixels: 1_024,
            minimumHeightPixels: 64,
            maximumHeightPixels: 1_024,
            dimensionMultiplePixels: 16,
            maximumSteps: 50,
            maximumGuidanceThousandths: 10_000,
            outputMimeTypes: ["image/png"]);
    }
}
