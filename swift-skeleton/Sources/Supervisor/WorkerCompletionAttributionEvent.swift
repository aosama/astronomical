import Foundation

import IpcProtocol

/// Completion-event attribution fan-out, mirroring apps/supervisor/src/
/// worker_completion_event.rs: one completed request lands in both local
/// attribution logs — the performance row carries the cache diagnostics and
/// measured throughput, the completion row carries the emitted tool calls
/// when the operator enabled that toggle. Both are no-ops on a disabled log.
extension WorkerSupervisor {

    func recordCompletionAttribution(
        requestId: RequestId,
        requestStartedAt: Date,
        generationStartedAt: Date?,
        firstOutputAt: Date?,
        promptTokenCount: UInt32,
        cachedTokenCount: UInt32,
        generatedTokenCount: UInt16,
        prefillElapsedMillis: UInt64,
        maximumMlxPeakMemoryBytes: UInt64?,
        lastMlxActiveMemoryBytes: UInt64?,
        persistentPromptCacheDiagnostics: WorkerPersistentPromptCacheRequestDiagnostics?,
        completionReason: ChatGenerationCompletionReason,
        streamEvents: Array<ChatGenerationStreamEvent>
    ) -> Void {
        let totalElapsedMillis: UInt64 = UInt64(
            (Date().timeIntervalSince(requestStartedAt) * 1000).rounded());
        let generationElapsedMillis: UInt64 = generationStartedAt.map({ (startedAt: Date) -> UInt64 in
            return UInt64((Date().timeIntervalSince(startedAt) * 1000).rounded());
        }) ?? 0;
        let timeToFirstOutputMillis: UInt64? = firstOutputAt.map({ (firstOutputDate: Date) -> UInt64 in
            return UInt64((firstOutputDate.timeIntervalSince(requestStartedAt) * 1000).rounded());
        });
        let computedThroughput: (prefillTokPerSecond: Double?, generationTokPerSecond: Double?) =
            GenerationPerformanceRecord.computeThroughput(
                promptTokenCount: promptTokenCount,
                cachedTokenCount: cachedTokenCount,
                generatedTokenCount: generatedTokenCount,
                prefillElapsedMillis: prefillElapsedMillis,
                generationElapsedMillis: generationElapsedMillis);
        let readyModelId: String = self.healthState.currentSnapshot().readyModelId ?? "";
        let completionReasonName: String = WorkerSupervisor.completionReasonName(completionReason);
        self.generationPerformanceLog.record(GenerationPerformanceRecord(
            timestampMillis: PerformanceLogClock.unixEpochMillis(),
            requestId: requestId.value(),
            modelId: readyModelId,
            promptTokenCount: promptTokenCount,
            cachedTokenCount: cachedTokenCount,
            generatedTokenCount: generatedTokenCount,
            completionReason: completionReasonName,
            prefillElapsedMillis: prefillElapsedMillis,
            generationElapsedMillis: generationElapsedMillis,
            totalElapsedMillis: totalElapsedMillis,
            timeToFirstOutputMillis: timeToFirstOutputMillis,
            generationPreparationElapsedMillis: nil,
            firstDecodeForwardElapsedMillis: nil,
            generationPreparationExpertSourceReadByteCount: 0,
            finalResidentExpertCount: nil,
            finalResidentExpertPayloadBytes: nil,
            prefillTokPerSecond: computedThroughput.prefillTokPerSecond,
            generationTokPerSecond: computedThroughput.generationTokPerSecond,
            mlxPeakMemoryBytes: maximumMlxPeakMemoryBytes,
            mlxActiveMemoryBytes: lastMlxActiveMemoryBytes,
            persistentPromptCacheDiagnostics: persistentPromptCacheDiagnostics));
        // Attribute what the model emitted — function names and the arguments
        // JSON — so argument pollution and foreign-dialect regressions are
        // diagnosable instead of invisible. No-op unless the toggle is on.
        let completedToolCalls: Array<CompletedToolCall> = streamEvents.compactMap(
            { (streamEvent: ChatGenerationStreamEvent) -> CompletedToolCall? in
                if case let .toolCall(toolCallIndex, functionName, argumentsJson) = streamEvent {
                    return CompletedToolCall(
                        toolCallIndex: toolCallIndex,
                        functionName: functionName,
                        argumentsJson: argumentsJson);
                }
                return nil;
            });
        self.completionAttributionLog.recordCompletionAtNow(
            requestId: requestId.value(),
            modelId: readyModelId,
            completionReason: completionReasonName,
            completedToolCalls: completedToolCalls);
    }
}
