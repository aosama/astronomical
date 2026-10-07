import Foundation

import AstronomicalConfig;
import IpcProtocol;

/**
 * Embeddings routing for the ephemeral daemon IPC service, porting
 * apps/supervisor/src/daemon_ipc_embeddings.rs: resolves the target model,
 * dispatches one embeddings command to the worker executor, and answers with
 * exactly one terminal frame.
 */
enum DaemonIpcEmbeddings {

    /// Runs one IPC embeddings batch: resolves the resident-or-requested
    /// model, lets the worker swap models when they differ, and answers with
    /// exactly one terminal frame.
    static func serve(
        model requestedModelId: String?,
        inputs: Array<String>,
        dimensions: UInt32?,
        modelsContext: DaemonIpcModelsContext,
        requestIdAllocator: ChatRequestIdAllocator,
        streamingResponseWriter: StreamingResponseWriter
    ) throws -> Void {
        let attributionStart: ContinuousClock.Instant? = DaemonIpcPerformanceAttribution.startedOperation(
            operationName: "daemon_ipc_embed_generate",
            performanceAttributionEnabled: modelsContext.performanceAttributionEnabled());
        var attributionOutcomeText: String = "success";
        defer {
            DaemonIpcPerformanceAttribution.finishedOperation(
                operationName: "daemon_ipc_embed_generate",
                operationStart: attributionStart,
                operationOutcome: attributionOutcomeText,
                performanceAttributionEnabled: modelsContext.performanceAttributionEnabled());
        }
        let terminalResponse: DaemonResponse = DaemonIpcEmbeddings.resolveTerminalResponse(
            model: requestedModelId,
            inputs: inputs,
            dimensions: dimensions,
            modelsContext: modelsContext,
            requestIdAllocator: requestIdAllocator);
        attributionOutcomeText = DaemonIpcEmbeddings.isSuccessResponse(terminalResponse) ? "success" : "failure";
        try streamingResponseWriter.sendResponse(terminalResponse);
        try streamingResponseWriter.close();
    }

    private static func isSuccessResponse(_ daemonResponse: DaemonResponse) -> Bool {
        if case .embeddingsCompleted = daemonResponse {
            return true;
        }
        return false;
    }

    /**
     * Produces the single terminal frame for one embeddings batch: a
     * rejection before admission, or the worker's completed/failed outcome
     * after.
     */
    private static func resolveTerminalResponse(
        model requestedModelId: String?,
        inputs: Array<String>,
        dimensions: UInt32?,
        modelsContext: DaemonIpcModelsContext,
        requestIdAllocator: ChatRequestIdAllocator
    ) -> DaemonResponse {
        let healthSnapshot: WorkerHealthSnapshot = modelsContext.embeddingsExecutor.workerHealthSnapshot();
        // Embeddings target the resident model unless the caller names one.
        // Unlike chat generation, a requested model that differs from the
        // resident model is not a rejection: the worker swaps models for
        // embeddings itself.
        guard let resolvedModelId: String = requestedModelId ?? healthSnapshot.readyModelId else {
            return .generationRejected(
                reason: "no model is resident; load a model or pass --model");
        }
        guard let requestIdentifier: UInt64 = requestIdAllocator.allocate() else {
            return .generationRejected(reason: "the local request identifier space is exhausted");
        }
        let embeddingsCommand: EmbeddingsCommand = EmbeddingsCommand(
            requestId: RequestId(rawRequestId: requestIdentifier),
            model: resolvedModelId,
            inputs: inputs,
            encodingFormat: .float,
            dimensions: dimensions
        );
        do {
            try embeddingsCommand.validate();
        } catch let validationError {
            return .generationRejected(reason: String(describing: validationError));
        }
        let embeddingsOutput: EmbeddingsOutput;
        do {
            embeddingsOutput = try modelsContext.embeddingsExecutor.startEmbeddingsGeneration(
                embeddingsCommand
            );
        } catch let startError as GenerationStartError {
            return .generationRejected(
                reason: DaemonIpcChat.generationStartRejectionReason(startError));
        } catch let executionError as EmbeddingsExecutionError {
            switch (executionError) {
            case let .workerFailure(reason):
                return .embeddingsFailed(reason: reason);
            case .workerUnavailable:
                return .generationRejected(
                    reason: DaemonIpcChat.generationStartRejectionReason(.workerUnavailable));
            }
        } catch {
            return .generationRejected(
                reason: DaemonIpcChat.generationStartRejectionReason(.workerUnavailable));
        }
        return .embeddingsCompleted(
            model: resolvedModelId,
            vectors: embeddingsOutput.embeddings,
            inputTokenCounts: embeddingsOutput.inputTokenCounts
        );
    }
}
