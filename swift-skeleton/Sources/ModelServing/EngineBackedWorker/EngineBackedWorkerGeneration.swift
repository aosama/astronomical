import Foundation;

import IpcProtocol;

extension EngineBackedWorker {

    /// Advances one bounded engine boundary and emits its events.
    ///
    /// Mirrors crates/model-serving/src/engine_backed_worker/
    /// {generation_advance,output}.rs. Returns whether the generation stays
    /// active for another loop turn.
    func advanceGeneration(
        _ activeGeneration: inout ActiveEngineGeneration,
        eventWriter: ProtocolWriter
    ) throws -> Bool {
        if activeGeneration.generatedTokenCount >= activeGeneration.maximumOutputTokens {
            return try self.finishGeneration(
                &activeGeneration,
                completionReason: .maximumOutputTokens,
                eventWriter: eventWriter);
        }
        guard let loadedChatRuntime = self.loadedChatRuntime else {
            throw WorkerRuntimeError.inferenceEngineGenerationFailed(
                reason: "generation continued after the loaded model was removed");
        }
        let generatedToken: GeneratedToken;
        do {
            generatedToken = try loadedChatRuntime.engine.decodeNextToken(
                requestId: activeGeneration.requestId);
        } catch let engineError as InferenceEngineError {
            switch engineError {
            case let .invalidRequest(reason):
                try eventWriter.sendEvent(.failed(
                    requestId: activeGeneration.requestId,
                    reason: .invalidRequest(reason: reason)));
                return false;
            default:
                throw WorkerRuntimeError.inferenceEngineGenerationFailed(
                    reason: engineError.publicFailureReason);
            }
        } catch {
            throw WorkerRuntimeError.inferenceEngineGenerationFailed(
                reason: WorkerRuntimeError.boundedModelLoadFailureReason(error));
        }
        switch generatedToken {
        case let .tokenId(
            generatedTokenId, isReasoningToken, expertMemoryMode, mlxMemorySnapshot,
            firstDecodeForwardElapsedMillis, generationFinalization):
            return try self.advanceGeneratedToken(
                &activeGeneration,
                generatedTokenId: generatedTokenId,
                isReasoningToken: isReasoningToken,
                expertMemoryMode: expertMemoryMode,
                mlxMemorySnapshot: mlxMemorySnapshot,
                firstDecodeForwardElapsedMillis: firstDecodeForwardElapsedMillis,
                generationFinalization: generationFinalization,
                eventWriter: eventWriter);
        case .endOfSequence:
            return try self.finishGeneration(
                &activeGeneration,
                completionReason: .endOfSequence,
                eventWriter: eventWriter);
        case let .prefillProgress(
            processedTokenCount, elapsedMillis, forwardPrefillChunkElapsedMillis,
            completedPrefillChunkTokens, mlxMemorySnapshot, expertResidencyTelemetry,
            expertMemoryMode, promptWorkReuse):
            return try self.advancePrefillProgress(
                &activeGeneration,
                processedTokenCount: processedTokenCount,
                elapsedMillis: elapsedMillis,
                forwardPrefillChunkElapsedMillis: forwardPrefillChunkElapsedMillis,
                completedPrefillChunkTokens: completedPrefillChunkTokens,
                mlxMemorySnapshot: mlxMemorySnapshot,
                expertResidencyTelemetry: expertResidencyTelemetry,
                expertMemoryMode: expertMemoryMode,
                promptWorkReuse: promptWorkReuse,
                eventWriter: eventWriter);
        case let .promptProcessingPhaseStarted(promptProcessingPhase, totalTokenCount):
            try eventWriter.sendEvent(.prefillProgress(
                requestId: activeGeneration.requestId,
                promptProcessingPhase: promptProcessingPhase,
                processedTokens: 0,
                totalTokens: totalTokenCount,
                elapsedMillis: 0,
                forwardPrefillChunkElapsedMillis: nil,
                completedPrefillChunkTokens: nil,
                mlxMemorySnapshot: nil,
                expertResidency: nil));
            return true;
        case let .generationPreparationStarted(
            totalLayerCount, residentExpertCount, residentExpertPayloadBytes,
            mlxMemorySnapshot):
            try eventWriter.sendEvent(.generationPreparationStarted(
                requestId: activeGeneration.requestId,
                totalLayerCount: totalLayerCount,
                residentExpertCount: residentExpertCount,
                residentExpertPayloadBytes: residentExpertPayloadBytes,
                mlxMemorySnapshot: WorkerMemoryObservation.mlxMemorySnapshot(
                    source: .decodeSubmitted, observedSnapshot: mlxMemorySnapshot)));
            return true;
        }
    }

    private func advanceGeneratedToken(
        _ activeGeneration: inout ActiveEngineGeneration,
        generatedTokenId: UInt32,
        isReasoningToken: Bool,
        expertMemoryMode: ExpertMemoryMode?,
        mlxMemorySnapshot: WorkerMlxMemorySnapshot?,
        firstDecodeForwardElapsedMillis: UInt64?,
        generationFinalization: GenerationFinalization?,
        eventWriter: ProtocolWriter
    ) throws -> Bool {
        if let firstDecodeForwardElapsedMillis = firstDecodeForwardElapsedMillis {
            try eventWriter.sendEvent(.firstDecodeCompleted(
                requestId: activeGeneration.requestId,
                elapsedMillis: firstDecodeForwardElapsedMillis));
        }
        if let generationFinalization = generationFinalization {
            activeGeneration.engineHasFinalizedGeneration = true;
            try self.emitGenerationFinalization(
                &activeGeneration, generationFinalization: generationFinalization,
                eventWriter: eventWriter);
        }
        activeGeneration.recordGeneratedToken(isReasoningToken: isReasoningToken);
        let isEndOfSequence: Bool = activeGeneration.activeGeneration
            .isEndOfSequenceToken(generatedTokenId);
        let modelTranslation: ModelGeneratedTokenTranslation;
        do {
            modelTranslation = try activeGeneration.activeGeneration.translateGeneratedToken(
                generatedTokenId);
        } catch ModelGenerationOutputError.malformedOutput {
            try self.sendGenerationProgress(
                &activeGeneration, mlxMemorySnapshot: nil, eventWriter: eventWriter);
            return try self.failMalformedGeneration(&activeGeneration, eventWriter: eventWriter);
        } catch let fatalOutputError as ModelGenerationOutputError {
            if case let .fatalExecution(reason) = fatalOutputError {
                throw WorkerRuntimeError.inferenceEngineGenerationFailed(reason: reason);
            }
            throw WorkerRuntimeError.inferenceEngineGenerationFailed(
                reason: "unexpected model output failure");
        } catch {
            throw WorkerRuntimeError.inferenceEngineGenerationFailed(
                reason: WorkerRuntimeError.boundedModelLoadFailureReason(error));
        }
        if modelTranslation.publicOutputs.isEmpty {
            try self.sendGenerationProgress(
                &activeGeneration, mlxMemorySnapshot: mlxMemorySnapshot,
                eventWriter: eventWriter);
        }
        try self.emitModelOutputs(
            &activeGeneration,
            modelOutputs: modelTranslation.publicOutputs,
            mlxMemorySnapshot: mlxMemorySnapshot,
            eventWriter: eventWriter);
        if modelTranslation.modelFeedbackTokenIds.isEmpty == false {
            guard let loadedChatRuntime = self.loadedChatRuntime else {
                throw WorkerRuntimeError.inferenceEngineGenerationFailed(
                    reason: "generation continued after the loaded model was removed");
            }
            do {
                try loadedChatRuntime.engine.injectInputTokens(
                    requestId: activeGeneration.requestId,
                    inputTokenIds: modelTranslation.modelFeedbackTokenIds);
            } catch let engineError as InferenceEngineError {
                if case let .invalidRequest(reason) = engineError {
                    try eventWriter.sendEvent(.failed(
                        requestId: activeGeneration.requestId,
                        reason: .invalidRequest(reason: reason)));
                    return false;
                }
                throw WorkerRuntimeError.inferenceEngineGenerationFailed(
                    reason: engineError.publicFailureReason);
            }
        }
        if isEndOfSequence {
            return try self.finishGeneration(
                &activeGeneration,
                completionReason: .endOfSequence,
                eventWriter: eventWriter);
        }
        if activeGeneration.generatedTokenCount >= activeGeneration.maximumOutputTokens {
            return try self.finishGeneration(
                &activeGeneration,
                completionReason: .maximumOutputTokens,
                eventWriter: eventWriter);
        }
        return true;
    }

    private func advancePrefillProgress(
        _ activeGeneration: inout ActiveEngineGeneration,
        processedTokenCount: UInt32,
        elapsedMillis: UInt64,
        forwardPrefillChunkElapsedMillis: UInt64,
        completedPrefillChunkTokens: UInt32,
        mlxMemorySnapshot: WorkerMlxMemorySnapshot?,
        expertResidencyTelemetry: ExpertResidencyTelemetry?,
        expertMemoryMode: ExpertMemoryMode?,
        promptWorkReuse: WorkerPromptWorkReuse,
        eventWriter: ProtocolWriter
    ) throws -> Bool {
        if promptWorkReuse.targetEligibleTokenCount > 0 {
            activeGeneration.requiredPromptProcessingTokenCount = UInt32(clamping:
                promptWorkReuse.targetEligibleTokenCount
                    - promptWorkReuse.targetRestoredTokenCount);
        }
        activeGeneration.promptWorkReuse = promptWorkReuse;
        activeGeneration.prefillProcessedTokens = activeGeneration.prefillProcessedTokens
            .addingReportingOverflow(processedTokenCount).partialValue;
        activeGeneration.prefillElapsedMillis = activeGeneration.prefillElapsedMillis
            .addingReportingOverflow(elapsedMillis).partialValue;
        let requiredPromptProcessingTokenCount: UInt32 =
            activeGeneration.requiredPromptProcessingTokenCount;
        if requiredPromptProcessingTokenCount == 0 {
            return true;
        }
        try eventWriter.sendEvent(.prefillProgress(
            requestId: activeGeneration.requestId,
            promptProcessingPhase: .target,
            processedTokens: min(
                activeGeneration.prefillProcessedTokens,
                requiredPromptProcessingTokenCount),
            totalTokens: requiredPromptProcessingTokenCount,
            elapsedMillis: activeGeneration.prefillElapsedMillis,
            forwardPrefillChunkElapsedMillis: forwardPrefillChunkElapsedMillis,
            completedPrefillChunkTokens: completedPrefillChunkTokens,
            mlxMemorySnapshot: WorkerMemoryObservation.mlxMemorySnapshot(
                source: .prefill, observedSnapshot: mlxMemorySnapshot),
            expertResidency: WorkerMemoryObservation.expertResidencySnapshot(
                residencyTelemetry: expertResidencyTelemetry)));
        return true;
    }

    private func sendGenerationProgress(
        _ activeGeneration: inout ActiveEngineGeneration,
        mlxMemorySnapshot: WorkerMlxMemorySnapshot?,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        let elapsedMillis: UInt64 = activeGeneration.generationStartedAt.map {
            generationStartedAt in UInt64(
                ContinuousClock.now.duration(to: generationStartedAt).components.seconds * 1000)
        } ?? 0;
        try eventWriter.sendEvent(.generationProgress(
            requestId: activeGeneration.requestId,
            generatedTokenCount: activeGeneration.generatedTokenCount,
            maximumOutputTokens: activeGeneration.maximumOutputTokens,
            elapsedMillis: elapsedMillis,
            mlxMemorySnapshot: WorkerMemoryObservation.mlxMemorySnapshot(
                source: .decodeSubmitted, observedSnapshot: mlxMemorySnapshot),
            expertResidency: nil));
    }

    /// Batches public outputs into one ordered `output` frame and advances
    /// the sequence watermark; tool-call indexes must stay contiguous.
    private func emitModelOutputs(
        _ activeGeneration: inout ActiveEngineGeneration,
        modelOutputs: Array<ChatGenerationOutput>,
        mlxMemorySnapshot: WorkerMlxMemorySnapshot?,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        if modelOutputs.isEmpty {
            return;
        }
        for modelOutput: ChatGenerationOutput in modelOutputs {
            if case let .toolCall(toolCallIndex, _, _) = modelOutput {
                guard toolCallIndex == activeGeneration.nextToolCallIndex else {
                    throw WorkerRuntimeError.inferenceEngineGenerationFailed(
                        reason: "structured chat tool-call indexes were not contiguous");
                }
                activeGeneration.nextToolCallIndex = activeGeneration.nextToolCallIndex
                    .addingReportingOverflow(1).partialValue;
                activeGeneration.hasEmittedToolCall = true;
            }
        }
        let outputCount: UInt16 = UInt16(clamping: modelOutputs.count);
        try eventWriter.sendEvent(.output(
            requestId: activeGeneration.requestId,
            sequenceNumber: activeGeneration.nextSequenceNumber,
            generatedTokenCount: activeGeneration.generatedTokenCount,
            outputs: modelOutputs,
            mlxMemorySnapshot: WorkerMemoryObservation.mlxMemorySnapshot(
                source: .decodeSubmitted, observedSnapshot: mlxMemorySnapshot),
            expertResidency: nil));
        activeGeneration.nextSequenceNumber = activeGeneration.nextSequenceNumber
            .addingReportingOverflow(outputCount).partialValue;
    }

    /// Flushes final outputs, releases the engine request, and reports
    /// completion. Returns false: the generation is finished.
    func finishGeneration(
        _ activeGeneration: inout ActiveEngineGeneration,
        completionReason: ChatGenerationCompletionReason,
        eventWriter: ProtocolWriter
    ) throws -> Bool {
        let finalOutputs: Array<ChatGenerationOutput>;
        do {
            finalOutputs = try activeGeneration.activeGeneration.finishOutputs();
        } catch ModelGenerationOutputError.malformedOutput {
            return try self.failMalformedGeneration(&activeGeneration, eventWriter: eventWriter);
        } catch let fatalOutputError as ModelGenerationOutputError {
            if case let .fatalExecution(reason) = fatalOutputError {
                throw WorkerRuntimeError.inferenceEngineGenerationFailed(reason: reason);
            }
            throw WorkerRuntimeError.inferenceEngineGenerationFailed(
                reason: "unexpected model output failure");
        } catch {
            throw WorkerRuntimeError.inferenceEngineGenerationFailed(
                reason: WorkerRuntimeError.boundedModelLoadFailureReason(error));
        }
        try self.emitModelOutputs(
            &activeGeneration, modelOutputs: finalOutputs, mlxMemorySnapshot: nil,
            eventWriter: eventWriter);
        if activeGeneration.engineHasFinalizedGeneration == false {
            let generationFinalization: GenerationFinalization = self.cancelEngineRequest(
                activeGeneration.requestId);
            try self.emitGenerationFinalization(
                &activeGeneration, generationFinalization: generationFinalization,
                eventWriter: eventWriter);
        }
        let finalReason: ChatGenerationCompletionReason =
            activeGeneration.hasEmittedToolCall ? .toolCalls : completionReason;
        try self.sendCompletion(&activeGeneration, completionReason: finalReason,
            eventWriter: eventWriter);
        return false;
    }

    private func failMalformedGeneration(
        _ activeGeneration: inout ActiveEngineGeneration,
        eventWriter: ProtocolWriter
    ) throws -> Bool {
        _ = self.cancelEngineRequest(activeGeneration.requestId);
        try eventWriter.sendEvent(.failed(
            requestId: activeGeneration.requestId,
            reason: .malformedModelOutput));
        return false;
    }

    private func emitGenerationFinalization(
        _ activeGeneration: inout ActiveEngineGeneration,
        generationFinalization: GenerationFinalization,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        guard generationFinalization.hasReportableState else {
            return;
        }
        try eventWriter.sendEvent(.generationFinalized(
            requestId: activeGeneration.requestId,
            expertMemoryMode: generationFinalization.expertMemoryMode,
            mlxMemorySnapshot: WorkerMemoryObservation.mlxMemorySnapshot(
                source: .finalized,
                observedSnapshot: generationFinalization.mlxMemorySnapshot),
            expertResidency: WorkerMemoryObservation.expertResidencySnapshot(
                residencyTelemetry: generationFinalization.expertResidencyTelemetry)));
    }

    private func sendCompletion(
        _ activeGeneration: inout ActiveEngineGeneration,
        completionReason: ChatGenerationCompletionReason,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        if activeGeneration.promptWorkReuse.targetEligibleTokenCount > 0 {
            try eventWriter.sendEvent(.promptWorkReuse(
                requestId: activeGeneration.requestId,
                promptWorkReuse: activeGeneration.promptWorkReuse));
        }
        try eventWriter.sendEvent(.completed(
            requestId: activeGeneration.requestId,
            promptTokenCount: UInt32(clamping: activeGeneration.activeGeneration
                .promptTokenCount),
            generatedTokenCount: activeGeneration.generatedTokenCount,
            reasoningTokenCount: activeGeneration.reasoningTokenCount,
            cachedTokenCount: activeGeneration.restoredPromptPrefixTokenCount,
            persistentPromptCacheDiagnostics: nil,
            reason: completionReason));
    }
}
