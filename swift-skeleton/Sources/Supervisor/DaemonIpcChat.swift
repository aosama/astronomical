import Foundation;

import AstronomicalConfig;
import IpcProtocol;

/// Chat generation serving over the daemon IPC: schema validation and
/// instruction pairing, worker-health gating, default filling, and stream
/// relaying from the worker executor to the CLI client.
///
/// Migrates apps/supervisor/src/daemon_ipc_chat.rs. A known model that is not
/// resident is admitted on purpose: the worker loop swaps or loads it on
/// demand, which is how the CLI auto-loads from a cold daemon.
enum DaemonIpcChat {

    private static let workerUnavailableRejectionReason: String =
        "the daemon worker is not ready to serve chat generation";

    /// Runs one IPC chat generation: gates on worker readiness, fills
    /// defaults, and streams events to the client until a terminal frame.
    static func serve(
        _ chatRequest: DaemonRequest,
        chatContext: DaemonIpcChatContext,
        requestIdAllocator: ChatRequestIdAllocator,
        streamingResponseWriter: StreamingResponseWriter
    ) throws -> Void {
        guard case let .chatGenerate(model, messages, settings, schemaJson) = chatRequest else {
            return;
        }
        let attributionStart: ContinuousClock.Instant? = DaemonIpcPerformanceAttribution.startedOperation(
            operationName: "daemon_ipc_chat_generate",
            performanceAttributionEnabled: chatContext.resolvedRuntimeConfig.performanceAttributionEnabled);
        defer {
            DaemonIpcPerformanceAttribution.finishedOperation(
                operationName: "daemon_ipc_chat_generate",
                operationStart: attributionStart,
                operationOutcome: "served",
                performanceAttributionEnabled: chatContext.resolvedRuntimeConfig.performanceAttributionEnabled);
        }
        // A malformed schema is a request-shape error, so it rejects before
        // the worker-health gate: the request cannot be served regardless of
        // state.
        var chatMessages: Array<ChatMessage> = messages;
        var structuredGenerationConstraint: StructuredGenerationConstraint?;
        if let schemaJson = schemaJson {
            do {
                let validatedConstraint: StructuredGenerationConstraint = try ChatSchemaConstraint.validated(schemaJson);
                // The token mask clamps the visible channel but never tells
                // the model what shape to plan, so the enforced schema
                // constraint pairs with a system instruction. The validated
                // constraint exists exactly when the schema text does, so the
                // schema text drives the injection.
                ChatSchemaConstraint.insertJsonOutputInstruction(
                    &chatMessages,
                    jsonOutputInstruction: ChatSchemaConstraint.enforcedSchemaOutputInstruction(
                        schemaJson: schemaJson));
                structuredGenerationConstraint = validatedConstraint;
            } catch let schemaRejection as ChatSchemaConstraint.ChatSchemaRejectionError {
                return try DaemonIpcChat.sendTerminalResponse(
                    streamingResponseWriter,
                    .generationRejected(reason: schemaRejection.reason));
            }
        }
        let healthSnapshot: WorkerHealthSnapshot = chatContext.chatExecutor.workerHealthSnapshot();
        if let rejectionReason: String = DaemonIpcChat.generationRejectionReason(
            healthSnapshot: healthSnapshot,
            resolvedRuntimeConfig: chatContext.resolvedRuntimeConfig,
            requestedModelId: model) {
            return try DaemonIpcChat.sendTerminalResponse(
                streamingResponseWriter,
                .generationRejected(reason: rejectionReason));
        }
        guard let requestIdentifier: UInt64 = requestIdAllocator.allocate() else {
            return try DaemonIpcChat.sendTerminalResponse(
                streamingResponseWriter,
                .generationRejected(reason: "the local request identifier space is exhausted"));
        }
        let settingsPresence: RequestGenerationSettingsPresence = RequestGenerationSettingsPresence(
            generationSettings: settings);
        var generationSettings: ChatGenerationSettings = RequestGenerationDefaults.apply(
            resolvedRuntimeConfig: chatContext.resolvedRuntimeConfig,
            modelId: model,
            settingsPresence: settingsPresence,
            generationSettings: settings);
        if generationSettings.maxOutputTokens == 0,
           let readyModelCapabilities: WorkerModelCapabilities = healthSnapshot.readyModelCapabilities,
           let chatCapabilities: ChatModelCapabilities = readyModelCapabilities.chat {
            // The worker advertises u32 token counts; the wire settings field is u16.
            generationSettings = DaemonIpcChat.withClampedMaximumOutputTokens(
                generationSettings,
                maximumOutputTokens: chatCapabilities.maxOutputTokens);
        }
        let generationCommand: ChatGenerationCommand = ChatGenerationCommand(
            requestId: RequestId(rawRequestId: requestIdentifier),
            model: model,
            messages: chatMessages,
            tools: [],
            toolChoice: .auto,
            settings: generationSettings,
            qwenThinkingChannelSeed: QwenThinkingChannelSeed.load(
                resolvedRuntimeConfig: chatContext.resolvedRuntimeConfig,
                instancePaths: chatContext.instancePaths,
                modelId: model),
            structuredGeneration: structuredGenerationConstraint);
        let streamEvents: Array<ChatGenerationStreamEvent>;
        do {
            streamEvents = try chatContext.chatExecutor.startChatGeneration(generationCommand);
        } catch let startError as GenerationStartError {
            return try DaemonIpcChat.sendTerminalResponse(
                streamingResponseWriter,
                .generationRejected(reason: DaemonIpcChat.generationStartRejectionReason(startError)));
        }
        try DaemonIpcChat.relayStreamEvents(streamEvents, streamingResponseWriter: streamingResponseWriter);
    }

    /// Rejects a generation the daemon cannot serve: an unavailable worker, or
    /// a requested model that is unknown to the live model policy catalog
    /// (checked as either a full catalog key or a leaf alias).
    private static func generationRejectionReason(
        healthSnapshot: WorkerHealthSnapshot,
        resolvedRuntimeConfig: ResolvedRuntimeConfig,
        requestedModelId: String
    ) -> String? {
        if healthSnapshot.status == .unavailable {
            return DaemonIpcChat.workerUnavailableRejectionReason;
        }
        let knownModelIds: Array<String> = Array(resolvedRuntimeConfig.modelPolicyCatalog.keys);
        let requestedIsKnown: Bool = knownModelIds.contains(requestedModelId)
            || knownModelIds.contains(ModelIdentity.leafModelId(modelId: requestedModelId));
        if !requestedIsKnown {
            let suggestedModelIds: Array<String> = ModelIdentity.nearModelMatches(
                requestedModelId: requestedModelId,
                candidateModelIds: knownModelIds);
            return DaemonIpcChat.unknownModelRejectionReason(
                requestedModelId: requestedModelId,
                suggestedModelIds: suggestedModelIds,
                baseMessage: "the daemon knows no such model; run `astronomical models list` "
                    + "to see installed models or `astronomical models supported` to see downloadable ones");
        }
        return nil;
    }

    /// Builds the shared unknown-model rejection text with near-match suggestions.
    static func unknownModelRejectionReason(
        requestedModelId: String,
        suggestedModelIds: Array<String>,
        baseMessage: String
    ) -> String {
        if suggestedModelIds.isEmpty {
            return "the model \(requestedModelId) is unknown — \(baseMessage)";
        }
        return "the model \(requestedModelId) is unknown — \(baseMessage); did you mean: "
            + suggestedModelIds.joined(separator: ", ");
    }

    static func generationStartRejectionReason(_ startError: GenerationStartError) -> String {
        switch (startError) {
        case .capacityUnavailable:
            return "no generation capacity is available";
        case let .modelLoadFailed(modelLoadFailureReason):
            return "the model could not be loaded: \(modelLoadFailureReason)";
        case let .requestTooLarge(actualIpcMessageBytes, maximumIpcMessageBytes):
            return "the request is \(actualIpcMessageBytes) bytes but the IPC limit is "
                + "\(maximumIpcMessageBytes) bytes";
        case .workerUnavailable:
            return "the worker is unavailable";
        }
    }

    private static func withClampedMaximumOutputTokens(
        _ generationSettings: ChatGenerationSettings,
        maximumOutputTokens: UInt32
    ) -> ChatGenerationSettings {
        let clampedMaximumOutputTokens: UInt16 = maximumOutputTokens > UInt32(UInt16.max)
            ? UInt16.max
            : UInt16(maximumOutputTokens);
        return ChatGenerationSettings(
            maxOutputTokens: clampedMaximumOutputTokens,
            temperatureThousandths: generationSettings.temperatureThousandths,
            topPThousandths: generationSettings.topPThousandths,
            seed: generationSettings.seed,
            thinkingBudget: generationSettings.thinkingBudget);
    }

    private static func relayStreamEvents(
        _ streamEvents: Array<ChatGenerationStreamEvent>,
        streamingResponseWriter: StreamingResponseWriter
    ) throws -> Void {
        for streamEvent: ChatGenerationStreamEvent in streamEvents {
            let daemonResponse: DaemonResponse? = DaemonIpcChat.streamEventToDaemonResponse(streamEvent);
            guard let daemonResponse = daemonResponse else {
                // PrefillProgress is worker-internal progress with no CLI presentation.
                continue;
            }
            try streamingResponseWriter.sendResponse(daemonResponse);
            if streamEvent.isTerminal {
                try streamingResponseWriter.close();
                return;
            }
        }
        let eofFailureResponse: DaemonResponse = .chatGenerationFailed(reason: .fatalExecution(
            reason: "the worker stream ended before completing the generation"));
        return try DaemonIpcChat.sendTerminalResponse(streamingResponseWriter, eofFailureResponse);
    }

    private static func streamEventToDaemonResponse(
        _ streamEvent: ChatGenerationStreamEvent
    ) -> DaemonResponse? {
        switch (streamEvent) {
        case let .textFragment(text):
            return .chatGenerationText(text: text);
        case let .reasoningFragment(text):
            return .chatGenerationReasoning(text: text);
        case let .toolCall(toolCallIndex, functionName, argumentsJson):
            return .chatGenerationToolCall(
                toolCallIndex: toolCallIndex,
                functionName: functionName,
                argumentsJson: argumentsJson);
        case .prefillProgress:
            return nil;
        case let .completed(promptTokenCount, generatedTokenCount, reasoningTokenCount, cachedTokenCount, reason):
            return .chatGenerationCompleted(
                promptTokenCount: promptTokenCount,
                generatedTokenCount: generatedTokenCount,
                reasoningTokenCount: reasoningTokenCount,
                cachedTokenCount: cachedTokenCount,
                reason: reason);
        case let .failed(reason):
            return .chatGenerationFailed(reason: reason);
        case .streamError:
            return .chatGenerationFailed(reason: .fatalExecution(
                reason: "the worker became unavailable during the generation"));
        }
    }

    private static func sendTerminalResponse(
        _ streamingResponseWriter: StreamingResponseWriter,
        _ daemonResponse: DaemonResponse
    ) throws -> Void {
        try streamingResponseWriter.sendResponse(daemonResponse);
        try streamingResponseWriter.close();
    }
}
