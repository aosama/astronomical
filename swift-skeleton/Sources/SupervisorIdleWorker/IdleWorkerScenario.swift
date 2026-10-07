import Foundation

import IpcProtocol

/**
 * The scripted command loop of the supervisor test worker, migrating
 * apps/supervisor/tests/fixtures/idle_worker.rs. Model identities and
 * directory suffixes select scripted protocol behavior over the real framed
 * transport, so the supervisor journey suites exercise queueing, swap
 * ordering, and event handling without loading MLX or local artifacts.
 */
enum IdleWorkerScenario {

    private static let DELAYED_COMPLETION_MODEL_ID: String = "astronomical/delayed-completion-model"
    private static let GENERATION_EVENT_BEFORE_SWAP_MODEL_ID: String =
        "astronomical/generation-event-before-swap-model"
    private static let TELEMETRY_BEFORE_SWAP_MODEL_ID: String = "astronomical/telemetry-before-swap-model"
    private static let DELAYED_POLICY_ACK_MODEL_ID: String = "astronomical/delayed-policy-ack-model"
    private static let DELAYED_IMAGE_POLICY_ACK_MODEL_ID: String =
        "astronomical/delayed-image-policy-ack-model"
    private static let DELAYED_STARTUP_RUNTIME_CONFIGURATION_GENERATION: String =
        "delayed-startup-runtime-configuration"
    private static let MACHINE_MLX_MEMORY_CEILING_BYTES: UInt64 = 40_000_000_000
    private static let MINIMUM_MLX_MEMORY_CEILING_BYTES: UInt64 = 1
    private static let SWAPPED_MODEL_MINIMUM_MLX_MEMORY_CEILING_BYTES: UInt64 = 3_000_000_000
    private static let DISCONNECT_TRIPWIRE_PROMPT: String = "must-not-dispatch-after-disconnect"
    private static let DISCONNECT_TRIPWIRE_MARKER_FILE_NAME: String = "dispatched_after_disconnect"

    /**
     * The mutable acknowledgement state the scripted behaviors build on:
     * the loaded model identity and the last acknowledged runtime policy,
     * exactly the two facts the Rust fixture tracks.
     */
    private final class AcknowledgedState {
        var loadedModelId: String? = nil
        var runtimeConfiguration: WorkerRuntimeFeatureConfiguration? = nil
        var pendingCancellation: IdleWorkerCancellationScenario.PendingCancellation? = nil
    }

    enum FixtureFailure: Error, CustomStringConvertible {
        case imageGenerationDispatchedAfterDisconnect

        var description: String {
            switch (self) {
            case .imageGenerationDispatchedAfterDisconnect:
                return "image generation dispatched after client disconnect"
            }
        }
    }

    static func runFixture(
        commandReader: ProtocolReader,
        eventWriter: ProtocolWriter,
        controlDirectoryPath: String?
    ) throws -> Void {
        let acknowledgedState: AcknowledgedState = AcknowledgedState()
        try eventWriter.sendEvent(.idle(
            machineMlxMemoryCeilingBytes: IdleWorkerScenario.MACHINE_MLX_MEMORY_CEILING_BYTES,
            effectiveMlxMemoryCeilingBytes: IdleWorkerScenario.MACHINE_MLX_MEMORY_CEILING_BYTES,
            minimumMlxMemoryCeilingBytes: IdleWorkerScenario.MINIMUM_MLX_MEMORY_CEILING_BYTES))
        while let workerCommand: WorkerCommand = try commandReader.nextCommand() {
            try IdleWorkerScenario.handleCommand(
                workerCommand,
                eventWriter: eventWriter,
                acknowledgedState: acknowledgedState,
                controlDirectoryPath: controlDirectoryPath)
        }
    }

    private static func handleCommand(
        _ workerCommand: WorkerCommand,
        eventWriter: ProtocolWriter,
        acknowledgedState: AcknowledgedState,
        controlDirectoryPath: String?
    ) throws -> Void {
        switch (workerCommand) {
        case let .initializeWorker(startupConfiguration):
            try IdleWorkerScenario.handleInitializeWorker(
                startupConfiguration,
                eventWriter: eventWriter,
                acknowledgedState: acknowledgedState)
        case let .swapModel(modelDirectory, modelConfiguration):
            try IdleWorkerScenario.handleSwapModel(
                modelDirectory,
                modelConfiguration: modelConfiguration,
                eventWriter: eventWriter,
                acknowledgedState: acknowledgedState)
        case let .generateEmbeddings(embeddingsCommand):
            try IdleWorkerScenario.handleGenerateEmbeddings(
                embeddingsCommand,
                eventWriter: eventWriter)
        case let .generateImage(generationCommand):
            try IdleWorkerScenario.handleGenerateImage(
                generationCommand,
                eventWriter: eventWriter,
                controlDirectoryPath: controlDirectoryPath)
        case let .generate(generationCommand):
            if let pendingCancellation: IdleWorkerCancellationScenario.PendingCancellation =
                IdleWorkerCancellationScenario.pendingCancellation(for: generationCommand) {
                acknowledgedState.pendingCancellation = pendingCancellation
                try IdleWorkerCancellationScenario.emitOnGenerate(
                    pendingCancellation,
                    eventWriter: eventWriter)
                return
            }
            if try IdleWorkerChatScenario.emitScriptedSequence(
                modelId: generationCommand.model,
                requestId: generationCommand.requestId,
                eventWriter: eventWriter,
                maximumOutputTokens: generationCommand.settings.maxOutputTokens) {
                return
            }
            if (acknowledgedState.loadedModelId == IdleWorkerScenario.DELAYED_COMPLETION_MODEL_ID) {
                Thread.sleep(forTimeInterval: 0.25)
            }
            try eventWriter.sendEvent(IdleWorkerScenario.chatCompleted(generationCommand.requestId))
        case let .cancel(requestId):
            let pendingCancellation: IdleWorkerCancellationScenario.PendingCancellation? =
                acknowledgedState.pendingCancellation
            acknowledgedState.pendingCancellation = nil
            try IdleWorkerCancellationScenario.acknowledgeCancellation(
                pendingCancellation,
                requestId: requestId,
                eventWriter: eventWriter)
        case .sampleMlxMemory:
            try IdleWorkerScenario.emitMemorySnapshot(
                eventWriter,
                snapshotSource: .idlePoll)
        case let .updateMlxMemoryLimit(effectiveMlxMemoryCeilingBytes, configurationGeneration):
            try IdleWorkerScenario.handleUpdateMlxMemoryLimit(
                effectiveMlxMemoryCeilingBytes,
                configurationGeneration: configurationGeneration,
                eventWriter: eventWriter,
                acknowledgedState: acknowledgedState)
        case let .clearPromptCache(modelId):
            try IdleWorkerScenario.handleClearPromptCache(
                modelId,
                eventWriter: eventWriter)
        }
    }

    private static func handleInitializeWorker(
        _ startupConfiguration: WorkerStartupConfiguration,
        eventWriter: ProtocolWriter,
        acknowledgedState: AcknowledgedState
    ) throws -> Void {
        if (startupConfiguration.configurationGeneration
            == IdleWorkerScenario.DELAYED_STARTUP_RUNTIME_CONFIGURATION_GENERATION) {
            Thread.sleep(forTimeInterval: 0.15)
        }
        let runtimeConfiguration: WorkerRuntimeFeatureConfiguration = WorkerRuntimeFeatureConfiguration(
            configurationGeneration: startupConfiguration.configurationGeneration,
            persistentPromptCacheEnabled: startupConfiguration.persistentPromptCacheEnabled,
            promptCacheMaximumSizeBytes: startupConfiguration.globalPromptCacheMaximumSizeBytes,
            loadedModel: nil)
        try eventWriter.sendEvent(.runtimeFeatureConfigurationApplied(runtimeConfiguration))
        acknowledgedState.runtimeConfiguration = runtimeConfiguration
    }

    private static func handleSwapModel(
        _ modelDirectory: String,
        modelConfiguration: WorkerModelConfiguration,
        eventWriter: ProtocolWriter,
        acknowledgedState: AcknowledgedState
    ) throws -> Void {
        if (modelDirectory.hasSuffix("hanging-model")) {
            return
        }
        if (modelDirectory.hasSuffix("invalid-model")) {
            try eventWriter.sendEvent(.modelSwapFailed(
                loadedModelRemainsReady: false,
                modelLoadFailureReason: "model artifact validation failed: OptiQ metadata uses unsupported 2-bit quantization"))
            return
        }
        let replacementModelId: String = modelConfiguration.modelId()
        // These two branches deliberately violate the assumption that
        // ModelSwapped is always the next frame after SwapModel.
        if (replacementModelId == IdleWorkerScenario.TELEMETRY_BEFORE_SWAP_MODEL_ID) {
            try IdleWorkerScenario.emitMemorySnapshot(eventWriter, snapshotSource: .idlePoll)
        } else if (replacementModelId == IdleWorkerScenario.GENERATION_EVENT_BEFORE_SWAP_MODEL_ID) {
            try eventWriter.sendEvent(IdleWorkerScenario.chatCompleted(RequestId(rawRequestId: 999)))
        }
        try eventWriter.sendEvent(.modelSwapped(
            modelId: replacementModelId,
            capabilities: IdleWorkerScenario.swappedModelCapabilities(modelConfiguration),
            expertMemoryMode: ExpertMemoryMode.resident,
            minimumMlxMemoryCeilingBytes: IdleWorkerScenario.SWAPPED_MODEL_MINIMUM_MLX_MEMORY_CEILING_BYTES))
        if let acknowledgedRuntimeConfiguration: WorkerRuntimeFeatureConfiguration = acknowledgedState.runtimeConfiguration {
            if (replacementModelId == IdleWorkerScenario.DELAYED_POLICY_ACK_MODEL_ID
                || replacementModelId == IdleWorkerScenario.DELAYED_IMAGE_POLICY_ACK_MODEL_ID) {
                Thread.sleep(forTimeInterval: 0.2)
            }
            let runtimeConfigurationWithModel: WorkerRuntimeFeatureConfiguration =
                WorkerRuntimeFeatureConfiguration(
                    configurationGeneration: acknowledgedRuntimeConfiguration.configurationGeneration,
                    persistentPromptCacheEnabled: acknowledgedRuntimeConfiguration.persistentPromptCacheEnabled,
                    promptCacheMaximumSizeBytes: acknowledgedRuntimeConfiguration.promptCacheMaximumSizeBytes,
                    loadedModel: modelConfiguration.runtimeConfiguration())
            try eventWriter.sendEvent(.runtimeFeatureConfigurationApplied(runtimeConfigurationWithModel))
            acknowledgedState.runtimeConfiguration = runtimeConfigurationWithModel
        }
        acknowledgedState.loadedModelId = replacementModelId
        try IdleWorkerScenario.emitMemorySnapshot(eventWriter, snapshotSource: .modelLoaded)
    }

    private static func swappedModelCapabilities(
        _ modelConfiguration: WorkerModelConfiguration
    ) -> WorkerModelCapabilities {
        guard let autoregressiveConfiguration: WorkerAutoregressiveModelConfiguration =
            modelConfiguration.autoregressive() else {
            return WorkerModelCapabilities.imageGeneration(
                imageGeneration: IdleWorkerScenario.imageCapabilities())
        }
        return WorkerModelCapabilities.from(chatCapabilities: ChatModelCapabilities(
            supportsReasoning: true,
            supportsToolCalls: true,
            hasVision: false,
            maxInputTokens: autoregressiveConfiguration.maximumContextTokens > 0
                ? autoregressiveConfiguration.maximumContextTokens - 1
                : 0,
            maxOutputTokens: autoregressiveConfiguration.maximumOutputTokens,
            contextWindow: autoregressiveConfiguration.maximumContextTokens))
    }

    private static func handleGenerateEmbeddings(
        _ embeddingsCommand: EmbeddingsCommand,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        let embeddingVectors: Array<Array<Float>> = Array<Array<Float>>(
            repeating: [1.0, 0.0],
            count: embeddingsCommand.inputs.count)
        let inputTokenCounts: Array<UInt32> = Array<UInt32>(
            repeating: 1,
            count: embeddingsCommand.inputs.count)
        try eventWriter.sendEvent(.embeddingsCompleted(
            requestId: embeddingsCommand.requestId,
            embeddings: embeddingVectors,
            inputTokenCounts: inputTokenCounts,
            elapsedMillis: 1))
        try eventWriter.sendEvent(.embeddingsFinalized(
            requestId: embeddingsCommand.requestId,
            elapsedMillis: 2,
            mlxMemorySnapshot: nil))
    }

    private static func handleGenerateImage(
        _ generationCommand: ImageGenerationCommand,
        eventWriter: ProtocolWriter,
        controlDirectoryPath: String?
    ) throws -> Void {
        if (generationCommand.prompt == IdleWorkerScenario.DISCONNECT_TRIPWIRE_PROMPT) {
            // The supervisor must never dispatch this command after its
            // requester went away; the marker plus the nonzero exit make a
            // silent violation observable by the journey that armed it.
            IdleWorkerScenario.writeDisconnectTripwireMarker(controlDirectoryPath)
            throw FixtureFailure.imageGenerationDispatchedAfterDisconnect
        }
        let encodedPngBytes: Array<UInt8> = try IdleWorkerPngEncoder.encodeTruecolorPng(
            widthPixels: generationCommand.settings.widthPixels,
            heightPixels: generationCommand.settings.heightPixels)
        try eventWriter.sendEvent(.imageGenerationProgress(
            requestId: generationCommand.requestId,
            phase: ImageGenerationPhase.denoising,
            completedSteps: generationCommand.settings.steps,
            totalSteps: generationCommand.settings.steps,
            elapsedMillis: 20,
            mlxMemorySnapshot: nil))
        try eventWriter.sendEvent(.imageGenerationCompleted(
            requestId: generationCommand.requestId,
            generatedImage: GeneratedImage(
                mimeType: "image/png",
                encodedBytes: encodedPngBytes),
            resultMetadata: ImageGenerationResultMetadata(
                widthPixels: generationCommand.settings.widthPixels,
                heightPixels: generationCommand.settings.heightPixels,
                steps: generationCommand.settings.steps,
                guidanceThousandths: generationCommand.settings.guidanceThousandths,
                seed: generationCommand.settings.seed,
                elapsedMillis: 25)))
        try eventWriter.sendEvent(.imageGenerationFinalized(
            requestId: generationCommand.requestId,
            elapsedMillis: 30,
            mlxMemorySnapshot: WorkerMlxMemorySnapshot(
                source: MlxMemorySnapshotSource.finalized,
                activeMemoryBytes: 96_000_000,
                allocatorCacheMemoryBytes: 0,
                peakMemoryBytes: 512_000_000,
                expertPayloadBytes: 0,
                modelCorePayloadBytes: 96_000_000,
                contextStatePayloadBytes: 0,
                memoryCeilingUtilization: nil)))
    }

    private static func handleUpdateMlxMemoryLimit(
        _ effectiveMlxMemoryCeilingBytes: UInt64,
        configurationGeneration: String,
        eventWriter: ProtocolWriter,
        acknowledgedState: AcknowledgedState
    ) throws -> Void {
        if (effectiveMlxMemoryCeilingBytes == 30_000_000_000) {
            return
        }
        if (effectiveMlxMemoryCeilingBytes == 31_000_000_000) {
            Thread.sleep(forTimeInterval: 0.5)
            try eventWriter.sendEvent(.mlxMemoryLimitRejected(
                requestedMlxMemoryCeilingBytes: effectiveMlxMemoryCeilingBytes,
                minimumMlxMemoryCeilingBytes: IdleWorkerScenario.MINIMUM_MLX_MEMORY_CEILING_BYTES,
                machineMlxMemoryCeilingBytes: IdleWorkerScenario.MACHINE_MLX_MEMORY_CEILING_BYTES,
                reason: "fixture rejected the requested limit"))
            return
        }
        try eventWriter.sendEvent(.mlxMemoryLimitChanged(
            effectiveMlxMemoryCeilingBytes: effectiveMlxMemoryCeilingBytes,
            minimumMlxMemoryCeilingBytes: IdleWorkerScenario.MINIMUM_MLX_MEMORY_CEILING_BYTES,
            expertMemoryMode: ExpertMemoryMode.resident,
            mlxMemorySnapshot: nil,
            expertResidency: nil))
        if let currentConfiguration: WorkerRuntimeFeatureConfiguration = acknowledgedState.runtimeConfiguration {
            acknowledgedState.runtimeConfiguration = WorkerRuntimeFeatureConfiguration(
                configurationGeneration: configurationGeneration,
                persistentPromptCacheEnabled: currentConfiguration.persistentPromptCacheEnabled,
                promptCacheMaximumSizeBytes: currentConfiguration.promptCacheMaximumSizeBytes,
                loadedModel: currentConfiguration.loadedModel)
        }
    }

    private static func handleClearPromptCache(
        _ modelId: String?,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        let acknowledgedModelId: String? =
            (modelId == "astronomical/mismatched-clear-model") ? "astronomical/different-model" : modelId
        try eventWriter.sendEvent(.persistentPromptCacheStats(
            WorkerPersistentPromptCacheStats(
                persistentPromptCacheHits: 0,
                persistentPromptCacheMisses: 0,
                persistentPromptCacheTokensSaved: 0,
                persistentPromptCachePartialTailHits: 0,
                persistentPromptCacheBlockTokenCount: 2_048,
                persistentPromptCacheSequenceStateBlockCount: 0,
                persistentPromptCacheBoundaryStateSnapshotCount: 0,
                persistentPromptCacheVisualEmbeddingCount: 0,
                persistentPromptCacheTotalSizeBytes: 0,
                persistentPromptCacheVisualEmbeddingTotalSizeBytes: 0,
                persistentPromptCacheMaximumSizeBytes: 50_000_000_000,
                persistentPromptCacheVisualEmbeddingHits: 0,
                persistentPromptCacheVisualEmbeddingMisses: 0,
                persistentPromptCacheVisualEmbeddingRowsLoaded: 0)))
        try eventWriter.sendEvent(.promptCacheCleared(
            modelId: acknowledgedModelId,
            blocksRemoved: 3,
            bytesFreed: 4_096))
    }

    private static func imageCapabilities() -> ImageGenerationCapabilities {
        return ImageGenerationCapabilities(
            minimumWidthPixels: 64,
            maximumWidthPixels: 1_024,
            minimumHeightPixels: 64,
            maximumHeightPixels: 1_024,
            dimensionMultiplePixels: 16,
            maximumSteps: 4,
            maximumGuidanceThousandths: 1_000,
            outputMimeTypes: ["image/png"])
    }

    private static func chatCompleted(_ requestId: RequestId) -> WorkerEvent {
        return .completed(
            requestId: requestId,
            promptTokenCount: 1,
            generatedTokenCount: 0,
            reasoningTokenCount: 0,
            cachedTokenCount: 0,
            persistentPromptCacheDiagnostics: nil,
            reason: ChatGenerationCompletionReason.endOfSequence)
    }

    private static func emitMemorySnapshot(
        _ eventWriter: ProtocolWriter,
        snapshotSource: MlxMemorySnapshotSource
    ) throws -> Void {
        try eventWriter.sendEvent(.mlxMemorySample(
            mlxMemorySnapshot: WorkerMlxMemorySnapshot(
                source: snapshotSource,
                activeMemoryBytes: 20_000_000_000,
                allocatorCacheMemoryBytes: 0,
                peakMemoryBytes: 20_000_000_000,
                expertPayloadBytes: 12_000_000_000,
                modelCorePayloadBytes: 8_000_000_000,
                contextStatePayloadBytes: 0,
                memoryCeilingUtilization: nil),
            expertResidency: nil))
    }

    private static func writeDisconnectTripwireMarker(_ controlDirectoryPath: String?) -> Void {
        guard let controlDirectoryPath: String = controlDirectoryPath else {
            return
        }
        let markerPath: String = controlDirectoryPath + "/"
            + IdleWorkerScenario.DISCONNECT_TRIPWIRE_MARKER_FILE_NAME
        try? "dispatched".write(toFile: markerPath, atomically: true, encoding: String.Encoding.utf8)
    }
}
