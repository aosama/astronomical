import Foundation;

import IpcProtocol;

extension EngineBackedWorker {

    /// Serves one command while no generation is active: model-less
    /// rejections, model swap, memory controls, and cache clearing.
    ///
    /// Mirrors crates/model-serving/src/engine_backed_worker/
    /// {idle_command,model_swap,memory_limit}.rs.
    func serveIdleCommand(
        _ workerCommand: WorkerCommand,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        switch workerCommand {
        case .initializeWorker:
            // A duplicate startup policy is ignored, exactly as the Rust idle
            // loop does: startup happens once per process.
            return;
        case let .generate(generationCommand):
            return try self.startChatGeneration(generationCommand, eventWriter: eventWriter);
        case let .generateImage(imageGenerationCommand):
            try eventWriter.sendEvent(.imageGenerationFailed(
                requestId: imageGenerationCommand.requestId,
                reason: .modelDoesNotSupportImageGeneration));
            return try eventWriter.sendEvent(.imageGenerationFinalized(
                requestId: imageGenerationCommand.requestId,
                elapsedMillis: 0,
                mlxMemorySnapshot: nil));
        case let .generateEmbeddings(embeddingsCommand):
            try eventWriter.sendEvent(.embeddingsFailed(
                requestId: embeddingsCommand.requestId,
                reason: .fatalExecution(reason: "the loaded model does not support embeddings")));
            return try eventWriter.sendEvent(.embeddingsFinalized(
                requestId: embeddingsCommand.requestId,
                elapsedMillis: 0,
                mlxMemorySnapshot: nil));
        case .cancel:
            return;
        case .sampleMlxMemory:
            try self.emitMlxMemorySample(source: .idlePoll, eventWriter: eventWriter);
            // The prompt-cache counters ride the same idle sample so the
            // supervisor's status view advances without a dedicated verb.
            if let persistentPromptCacheStats: WorkerPersistentPromptCacheStats = self
                .loadedChatRuntime?.engine.collectPersistentPromptCacheStats() {
                try eventWriter.sendEvent(.persistentPromptCacheStats(
                    persistentPromptCacheStats));
            }
            return;
        case let .updateMlxMemoryLimit(requestedMlxMemoryCeilingBytes, configurationGeneration):
            return try self.serveUpdateMlxMemoryLimit(
                requestedMlxMemoryCeilingBytes,
                configurationGeneration: configurationGeneration,
                eventWriter: eventWriter);
        case let .swapModel(modelDirectory, modelConfiguration):
            return try self.serveModelSwap(
                modelDirectory: modelDirectory,
                modelConfiguration: modelConfiguration,
                eventWriter: eventWriter);
        case let .clearPromptCache(modelId):
            // Engines with an attached store clear their persisted state;
            // the default engine reports the empty deletion.
            do {
                let clearOutcome: PersistentPromptCacheClearOutcome = try self
                    .loadedChatRuntime?.engine.clearPersistentPromptCache(modelId: modelId)
                    ?? PersistentPromptCacheClearOutcome(
                        modelId: modelId, blocksRemoved: 0, bytesFreed: 0);
                return try eventWriter.sendEvent(.promptCacheCleared(
                    modelId: modelId, blocksRemoved: clearOutcome.blocksRemoved,
                    bytesFreed: clearOutcome.bytesFreed));
            } catch {
                return try eventWriter.sendEvent(.promptCacheCleared(
                    modelId: modelId, blocksRemoved: 0, bytesFreed: 0));
            }
        }
    }

    private func serveUpdateMlxMemoryLimit(
        _ requestedMlxMemoryCeilingBytes: UInt64,
        configurationGeneration: String,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        if requestedMlxMemoryCeilingBytes == 0
            || requestedMlxMemoryCeilingBytes > self.machineMlxMemoryCeilingBytes {
            return try eventWriter.sendEvent(.mlxMemoryLimitRejected(
                requestedMlxMemoryCeilingBytes: requestedMlxMemoryCeilingBytes,
                minimumMlxMemoryCeilingBytes: self.minimumMlxMemoryCeilingBytes,
                machineMlxMemoryCeilingBytes: self.machineMlxMemoryCeilingBytes,
                reason: "requested memory ceiling is outside the worker machine limit"));
        }
        self.effectiveMlxMemoryCeilingBytes = requestedMlxMemoryCeilingBytes;
        self.acknowledgedConfigurationGeneration = configurationGeneration;
        if let loadedChatRuntime = self.loadedChatRuntime {
            // A loaded engine records the ceiling for its allocator policy;
            // an engine that cannot honor live limits fails the request
            // without mutating the worker's own accounting.
            do {
                try loadedChatRuntime.engine.applyMlxMemoryLimit(
                    requestedMlxMemoryCeilingBytes);
            } catch {
                return try eventWriter.sendEvent(.mlxMemoryLimitRejected(
                    requestedMlxMemoryCeilingBytes: requestedMlxMemoryCeilingBytes,
                    minimumMlxMemoryCeilingBytes: self.minimumMlxMemoryCeilingBytes,
                    machineMlxMemoryCeilingBytes: self.machineMlxMemoryCeilingBytes,
                    reason: "the loaded model rejected the memory ceiling"));
            }
        }
        let engineSnapshot: WorkerMlxMemorySnapshot? = self.loadedChatRuntime.map {
            loadedChatRuntime in loadedChatRuntime.engine.collectMlxMemorySnapshot()
        } ?? nil;
        // The governor's ownership decomposition rides the acknowledgement
        // so the status view reflects the new budget immediately.
        let memoryGovernor: WorkerMlxMemoryGovernor = WorkerMlxMemoryGovernor(
            effectiveMlxMemoryCeilingBytes: requestedMlxMemoryCeilingBytes);
        let utilizationSnapshot: WorkerMlxMemorySnapshot? = engineSnapshot.map({
            (observedSnapshot: WorkerMlxMemorySnapshot) -> WorkerMlxMemorySnapshot in
            let composedBudget: MlxRamBudgetSnapshot = memoryGovernor.composedBudget(
                modelCorePayloadBytes: observedSnapshot.activeMemoryBytes,
                contextWindowReserveBytes: 0,
                completeLayerStreamSlotBytes: 0);
            let ownershipBreakdown: WorkerMemoryCeilingUtilizationSnapshot =
                WorkerMemoryCeilingUtilizationSnapshot(
                    unusedHeadroomBytes: composedBudget.retainedExpertBudgetBytes,
                    reservedModelCoreSlackBytes: 0,
                    reservedContextGrowthBytes: 0,
                    reservedActivationAndWorkspaceBytes: memoryGovernor
                        .activationHeadroomBytes
                        + memoryGovernor.otherFixedBytes,
                    unseatedExpertEntitlementBytes: composedBudget.retainedExpertBudgetBytes,
                    unexplainedHeadroomBytes: 0,
                    ownerOverrunBytes: 0);
            return WorkerMlxMemorySnapshot(
                source: observedSnapshot.source,
                activeMemoryBytes: observedSnapshot.activeMemoryBytes,
                allocatorCacheMemoryBytes: observedSnapshot.allocatorCacheMemoryBytes,
                peakMemoryBytes: observedSnapshot.peakMemoryBytes,
                expertPayloadBytes: observedSnapshot.expertPayloadBytes,
                modelCorePayloadBytes: observedSnapshot.modelCorePayloadBytes,
                contextStatePayloadBytes: observedSnapshot.contextStatePayloadBytes,
                memoryCeilingUtilization: ownershipBreakdown);
        });
        return try eventWriter.sendEvent(.mlxMemoryLimitChanged(
            effectiveMlxMemoryCeilingBytes: requestedMlxMemoryCeilingBytes,
            minimumMlxMemoryCeilingBytes: self.minimumMlxMemoryCeilingBytes,
            expertMemoryMode: .resident,
            mlxMemorySnapshot: WorkerMemoryObservation.mlxMemorySnapshot(
                source: .memoryLimitAdjusted, observedSnapshot: utilizationSnapshot),
            expertResidency: nil));
    }

    func emitMlxMemorySample(
        source: MlxMemorySnapshotSource,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        let engineSnapshot: WorkerMlxMemorySnapshot? = self.loadedChatRuntime.map {
            loadedChatRuntime in loadedChatRuntime.engine.collectMlxMemorySnapshot()
        } ?? nil;
        return try eventWriter.sendEvent(.mlxMemorySample(
            mlxMemorySnapshot: WorkerMemoryObservation.mlxMemorySnapshot(
                source: source, observedSnapshot: engineSnapshot),
            expertResidency: nil));
    }

    /// Transactional model replacement: create first, drop the prior runtime
    /// before load, and keep the prior model ready on any failure.
    private func serveModelSwap(
        modelDirectory: String,
        modelConfiguration: WorkerModelConfiguration,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        guard let chatRuntimeFactory = self.chatRuntimeFactory else {
            return try eventWriter.sendEvent(.modelSwapFailed(
                loadedModelRemainsReady: self.loadedChatRuntime != nil,
                modelLoadFailureReason: "model swapping is unavailable"));
        }
        let replacementRuntime: LoadedChatRuntime;
        do {
            replacementRuntime = try chatRuntimeFactory.createChatRuntime(
                modelDirectory: modelDirectory,
                modelConfiguration: modelConfiguration);
        } catch {
            return try eventWriter.sendEvent(.modelSwapFailed(
                loadedModelRemainsReady: self.loadedChatRuntime != nil,
                modelLoadFailureReason: WorkerRuntimeError.boundedModelLoadFailureReason(error)));
        }
        // Creation preserves the prior runtime on a rejected selection; once
        // created, release it before load so model payloads never overlap.
        self.loadedChatRuntime = nil;
        let engineLoadResult: EngineLoadResult;
        do {
            engineLoadResult = try replacementRuntime.engine.load();
        } catch {
            return try eventWriter.sendEvent(.modelSwapFailed(
                loadedModelRemainsReady: false,
                modelLoadFailureReason: "model engine initialization failed"));
        }
        self.minimumMlxMemoryCeilingBytes =
            engineLoadResult.minimumMlxMemoryCeilingBytes;
        self.loadedChatRuntime = replacementRuntime;
        self.acknowledgedFeatureConfiguration = self.acknowledgedFeatureConfiguration.map {
            acknowledgedFeatureConfiguration in
            return WorkerRuntimeFeatureConfiguration(
                configurationGeneration: acknowledgedFeatureConfiguration.configurationGeneration,
                persistentPromptCacheEnabled:
                    acknowledgedFeatureConfiguration.persistentPromptCacheEnabled,
                promptCacheMaximumSizeBytes:
                    acknowledgedFeatureConfiguration.promptCacheMaximumSizeBytes,
                loadedModel: modelConfiguration.runtimeConfiguration());
        }
        // The swap event carries the loaded model's own identity and
        // capabilities, taken from the processor's ready event exactly as
        // the Rust model_swapped_from_ready_event does.
        let readyEvent: WorkerEvent = replacementRuntime.processor.readyEvent();
        let swappedModelId: String;
        let swappedCapabilities: WorkerModelCapabilities;
        switch readyEvent {
        case let .ready(modelId, capabilities):
            swappedModelId = modelId;
            swappedCapabilities = capabilities;
        default:
            swappedModelId = modelConfiguration.modelId();
            swappedCapabilities = WorkerModelCapabilities(
                chat: nil, imageGeneration: nil, embeddings: nil);
        }
        try eventWriter.sendEvent(.modelSwapped(
            modelId: swappedModelId,
            capabilities: swappedCapabilities,
            expertMemoryMode: engineLoadResult.expertMemoryMode,
            minimumMlxMemoryCeilingBytes: self.minimumMlxMemoryCeilingBytes));
        if let acknowledgedFeatureConfiguration = self.acknowledgedFeatureConfiguration {
            try eventWriter.sendEvent(
                .runtimeFeatureConfigurationApplied(acknowledgedFeatureConfiguration));
        }
        return try self.emitMlxMemorySample(source: .modelLoaded, eventWriter: eventWriter);
    }

    /// Starts one chat generation from the idle state; a request-scoped
    /// failure answers `failed` and leaves the worker serving.
    func startChatGeneration(
        _ generationCommand: ChatGenerationCommand,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        do {
            try generationCommand.validate();
        } catch {
            return try eventWriter.sendEvent(.failed(
                requestId: generationCommand.requestId,
                reason: .invalidRequest(reason: String(describing: error))));
        }
        guard let loadedChatRuntime = self.loadedChatRuntime else {
            return try eventWriter.sendEvent(.failed(
                requestId: generationCommand.requestId,
                reason: .invalidRequest(
                    reason: "the loaded model does not support chat generation")));
        }
        let preparedGeneration: any ActiveChatGeneration;
        do {
            preparedGeneration = try loadedChatRuntime.processor.prepareChatGeneration(
                generationCommand);
        } catch let preparationRejection as ChatPreparationRejection {
            return try eventWriter.sendEvent(.failed(
                requestId: generationCommand.requestId,
                reason: preparationRejection.reason));
        } catch {
            return try eventWriter.sendEvent(.failed(
                requestId: generationCommand.requestId,
                reason: .invalidRequest(reason: WorkerRuntimeError
                    .boundedModelLoadFailureReason(error))));
        }
        // The governor validates the request's context admission before the
        // engine starts: the context workspace charge must fit the wired
        // budget after the model core and the named reserves.
        let memoryGovernor: WorkerMlxMemoryGovernor = WorkerMlxMemoryGovernor(
            effectiveMlxMemoryCeilingBytes: self.effectiveMlxMemoryCeilingBytes);
        let modelCorePayloadBytes: UInt64 = UInt64(
            loadedChatRuntime.engine.collectMlxMemorySnapshot()?.modelCorePayloadBytes ?? 0);
        let contextTokenNeed: UInt64 = UInt64(clamping: preparedGeneration.promptTokenCount)
            &+ UInt64(clamping: generationCommand.settings.maxOutputTokens);
        let contextWindowReserveBytes: UInt64 = loadedChatRuntime.engine
            .contextWorkspaceBytesPerToken()
            .map({ (contextBytesPerToken: Int) -> UInt64 in
                return contextBytesPerToken > 0
                    ? contextTokenNeed &* UInt64(contextBytesPerToken) : 0;
            }) ?? 0;
        if case let .rejected(deficitBytes) = memoryGovernor.validateContextAdmission(
            modelCorePayloadBytes: modelCorePayloadBytes,
            contextWindowReserveBytes: contextWindowReserveBytes) {
            return try eventWriter.sendEvent(.failed(
                requestId: generationCommand.requestId,
                reason: .invalidRequest(reason: "the request context exceeds the wired "
                    + "memory budget by \(deficitBytes) bytes; lower the context need or "
                    + "raise the memory ceiling")));
        }
        let engineGenerationStart: EngineGenerationStart;
        do {
            engineGenerationStart = try loadedChatRuntime.engine.startGeneration(
                preparedGeneration.inferenceRequest);
        } catch let engineError as InferenceEngineError {
            return try eventWriter.sendEvent(.failed(
                requestId: generationCommand.requestId,
                reason: engineError.chatGenerationFailureReason()));
        } catch {
            throw WorkerRuntimeError.inferenceEngineGenerationFailed(
                reason: WorkerRuntimeError.boundedModelLoadFailureReason(error));
        }
        var activeGeneration = ActiveEngineGeneration(
            generationCommand: generationCommand,
            activeGeneration: preparedGeneration,
            restoredPromptPrefixTokenCount:
                engineGenerationStart.restoredPromptPrefixTokenCount);
        if engineGenerationStart.expertMemoryMode != nil {
            try eventWriter.sendEvent(.expertMemoryModeChanged(
                expertMemoryMode: engineGenerationStart.expertMemoryMode!));
        }
        // The pre-generation progress frame tells the client prompt work has
        // begun whenever more than one token needs processing.
        if activeGeneration.requiredPromptProcessingTokenCount > 1,
            let promptProcessingPhase = engineGenerationStart.promptProcessingPhase {
            try eventWriter.sendEvent(.prefillProgress(
                requestId: activeGeneration.requestId,
                promptProcessingPhase: promptProcessingPhase,
                processedTokens: 0,
                totalTokens: activeGeneration.requiredPromptProcessingTokenCount,
                elapsedMillis: 0,
                forwardPrefillChunkElapsedMillis: nil,
                completedPrefillChunkTokens: nil,
                mlxMemorySnapshot: nil,
                expertResidency: nil));
        }
        let keepsGenerating: Bool = try self.advanceGeneration(
            &activeGeneration, eventWriter: eventWriter);
        self.activeGeneration = keepsGenerating ? activeGeneration : nil;
    }
}

extension InferenceEngineError {

    /// The wire failure reason for one engine error.
    func chatGenerationFailureReason() -> ChatGenerationFailureReason {
        switch self {
        case .engineBusy: return .engineBusy;
        case let .invalidRequest(reason): return .invalidRequest(reason: reason);
        case let .modelLoad(reason): return .fatalExecution(reason: reason);
        case let .fatalExecution(reason): return .fatalExecution(reason: reason);
        }
    }
}
