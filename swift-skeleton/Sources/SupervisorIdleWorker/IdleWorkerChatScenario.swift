import Foundation

import IpcProtocol

/**
 * The scripted chat command behaviors of the supervisor test worker,
 * migrating the generate arms of
 * apps/supervisor/tests/fixtures/scripted_worker.rs: each model identity
 * selects one exact event sequence — ordered happy paths, prefill
 * telemetry with finalized residency memory, and the protocol breaches
 * (out-of-order sequences, empty or invalid batches, duplicated
 * boundaries, over-budget completions, unsolicited cancellations) the
 * supervisor must contain.
 */
enum IdleWorkerChatScenario {

    static let ACCEPTED_CHAT_MODEL_ID: String = "astronomical/accepted-chat-fixture"
    static let PREFILL_PROGRESS_MODEL_ID: String = "astronomical/prefill-progress-fixture"
    static let ACTIVITY_TRANSITION_MODEL_ID: String = "astronomical/activity-transition-fixture"
    static let MALFORMED_OUTPUT_MODEL_ID: String = "astronomical/malformed-output-fixture"
    static let DUPLICATE_GENERATION_PREPARATION_MODEL_ID: String =
        "astronomical/duplicate-generation-preparation-fixture"
    static let OUT_OF_ORDER_MODEL_ID: String = "astronomical/out-of-order-chat-fixture"
    static let EMPTY_OUTPUT_BATCH_MODEL_ID: String = "astronomical/empty-output-batch-fixture"
    static let INVALID_OUTPUT_BATCH_MODEL_ID: String = "astronomical/invalid-output-batch-fixture"
    static let OVER_BUDGET_TOOL_COMPLETION_MODEL_ID: String =
        "astronomical/over-budget-tool-completion-fixture"
    static let UNSOLICITED_CANCELLATION_MODEL_ID: String =
        "astronomical/unsolicited-cancellation-fixture"
    static let BACKPRESSURE_MODEL_ID: String = "astronomical/backpressure-fixture"
    static let EXIT_AFTER_CHAT_ADMISSION_MODEL_ID: String =
        "astronomical/exit-after-chat-admission-fixture"

    /// The number of scripted output frames the backpressure fixture emits
    /// before its budget completion, matching the Rust fixture's 64.
    private static let BACKPRESSURE_OUTPUT_FRAME_COUNT: Int = 64
    /// The pause between backpressure output frames: long enough that the
    /// generation is provably still in flight when the journey asks for
    /// shutdown, short enough that the whole script stays inside the
    /// journey bounds without a draining client.
    private static let BACKPRESSURE_FRAME_INTERVAL_SECONDS: TimeInterval = 0.015

    /// Signals the fixture process to end cleanly after admitting a chat
    /// command; the entry point catches it and exits successfully, exactly
    /// like the Rust fixture's early return.
    enum ChatFixtureExit: Error {
        case processExitAfterChatAdmission
    }

    static func scriptedModelIds() -> Array<String> {
        return [
            IdleWorkerChatScenario.ACCEPTED_CHAT_MODEL_ID,
            IdleWorkerChatScenario.PREFILL_PROGRESS_MODEL_ID,
            IdleWorkerChatScenario.ACTIVITY_TRANSITION_MODEL_ID,
            IdleWorkerChatScenario.MALFORMED_OUTPUT_MODEL_ID,
            IdleWorkerChatScenario.DUPLICATE_GENERATION_PREPARATION_MODEL_ID,
            IdleWorkerChatScenario.OUT_OF_ORDER_MODEL_ID,
            IdleWorkerChatScenario.EMPTY_OUTPUT_BATCH_MODEL_ID,
            IdleWorkerChatScenario.INVALID_OUTPUT_BATCH_MODEL_ID,
            IdleWorkerChatScenario.OVER_BUDGET_TOOL_COMPLETION_MODEL_ID,
            IdleWorkerChatScenario.UNSOLICITED_CANCELLATION_MODEL_ID,
            IdleWorkerChatScenario.BACKPRESSURE_MODEL_ID,
            IdleWorkerChatScenario.EXIT_AFTER_CHAT_ADMISSION_MODEL_ID,
        ]
    }

    /**
     * Emits the scripted event sequence of one chat command. Returns true
     * when the model selected a scripted behavior; the caller falls back
     * to the default simple completion otherwise.
     */
    static func emitScriptedSequence(
        modelId: String,
        requestId: RequestId,
        eventWriter: ProtocolWriter,
        maximumOutputTokens: UInt16
    ) throws -> Bool {
        switch (modelId) {
        case IdleWorkerChatScenario.ACCEPTED_CHAT_MODEL_ID:
            try IdleWorkerChatScenario.emitAcceptedChat(requestId, eventWriter: eventWriter)
        case IdleWorkerChatScenario.PREFILL_PROGRESS_MODEL_ID:
            try IdleWorkerChatScenario.emitPrefillProgress(requestId, eventWriter: eventWriter)
        case IdleWorkerChatScenario.ACTIVITY_TRANSITION_MODEL_ID:
            try IdleWorkerChatScenario.emitActivityTransition(requestId, eventWriter: eventWriter)
        case IdleWorkerChatScenario.MALFORMED_OUTPUT_MODEL_ID:
            try eventWriter.sendEvent(.failed(
                requestId: requestId,
                reason: .malformedModelOutput))
        case IdleWorkerChatScenario.DUPLICATE_GENERATION_PREPARATION_MODEL_ID:
            try IdleWorkerChatScenario.emitDuplicateGenerationPreparation(
                requestId,
                eventWriter: eventWriter)
        case IdleWorkerChatScenario.OUT_OF_ORDER_MODEL_ID:
            try eventWriter.sendEvent(.output(
                requestId: requestId,
                sequenceNumber: 1,
                generatedTokenCount: 1,
                outputs: [.text(text: "out of order")],
                mlxMemorySnapshot: nil,
                expertResidency: nil))
        case IdleWorkerChatScenario.EMPTY_OUTPUT_BATCH_MODEL_ID:
            try eventWriter.sendEvent(.output(
                requestId: requestId,
                sequenceNumber: 0,
                generatedTokenCount: 1,
                outputs: [],
                mlxMemorySnapshot: nil,
                expertResidency: nil))
            try eventWriter.sendEvent(IdleWorkerChatScenario.chatCompleted(
                requestId,
                promptTokenCount: 1,
                generatedTokenCount: 1,
                reason: .endOfSequence))
        case IdleWorkerChatScenario.INVALID_OUTPUT_BATCH_MODEL_ID:
            try eventWriter.sendEvent(.output(
                requestId: requestId,
                sequenceNumber: 0,
                generatedTokenCount: 1,
                outputs: [
                    .text(text: "must not escape malformed batch"),
                    .toolCall(
                        toolCallIndex: 1,
                        functionName: "read",
                        argumentsJson: #"{"path":"AGENTS.md"}"#),
                ],
                mlxMemorySnapshot: nil,
                expertResidency: nil))
        case IdleWorkerChatScenario.OVER_BUDGET_TOOL_COMPLETION_MODEL_ID:
            try eventWriter.sendEvent(.output(
                requestId: requestId,
                sequenceNumber: 0,
                generatedTokenCount: 1,
                outputs: [.toolCall(
                    toolCallIndex: 0,
                    functionName: "read",
                    argumentsJson: #"{"path":"AGENTS.md"}"#)],
                mlxMemorySnapshot: nil,
                expertResidency: nil))
            try eventWriter.sendEvent(IdleWorkerChatScenario.chatCompleted(
                requestId,
                promptTokenCount: 1,
                generatedTokenCount: 17,
                reason: .toolCalls))
        case IdleWorkerChatScenario.UNSOLICITED_CANCELLATION_MODEL_ID:
            try eventWriter.sendEvent(IdleWorkerChatScenario.chatCompleted(
                requestId,
                promptTokenCount: 1,
                generatedTokenCount: 0,
                reason: .cancelled))
        case IdleWorkerChatScenario.BACKPRESSURE_MODEL_ID:
            try IdleWorkerChatScenario.emitBackpressureFragments(
                requestId,
                eventWriter: eventWriter,
                maximumOutputTokens: maximumOutputTokens)
        case IdleWorkerChatScenario.EXIT_AFTER_CHAT_ADMISSION_MODEL_ID:
            throw IdleWorkerChatScenario.ChatFixtureExit.processExitAfterChatAdmission
        default:
            return false
        }
        return true
    }

    /// The ordered happy-path sequence: reasoning, text, two tool calls,
    /// and a tool-call completion, exactly the Rust send_accepted_chat.
    private static func emitAcceptedChat(
        _ requestId: RequestId,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        let scriptedOutputs: Array<ChatGenerationOutput> = [
            .reasoning(text: "accepted chat reasoning"),
            .text(text: "accepted chat text"),
            .toolCall(
                toolCallIndex: 0,
                functionName: "read",
                argumentsJson: #"{"path":"AGENTS.md"}"#),
            .toolCall(
                toolCallIndex: 1,
                functionName: "glob",
                argumentsJson: #"{"pattern":"tests/**/*.rs"}"#),
        ]
        let scriptedGeneratedTokenCounts: Array<UInt16> = [1, 1, 3, 4]
        for outputIndex: Int in 0..<scriptedOutputs.count {
            try eventWriter.sendEvent(.output(
                requestId: requestId,
                sequenceNumber: UInt16(outputIndex),
                generatedTokenCount: scriptedGeneratedTokenCounts[outputIndex],
                outputs: [scriptedOutputs[outputIndex]],
                mlxMemorySnapshot: nil,
                expertResidency: nil))
        }
        try eventWriter.sendEvent(IdleWorkerChatScenario.chatCompleted(
            requestId,
            promptTokenCount: 2,
            generatedTokenCount: 4,
            reason: .toolCalls))
    }

    /// The prefill journey: one target-phase progress observation with
    /// prefill-era memory, the measured first decode, one text output, the
    /// finalized residency memory, then the completion whose totals prove
    /// the finalized telemetry replaced the prefill telemetry.
    private static func emitPrefillProgress(
        _ requestId: RequestId,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        try eventWriter.sendEvent(.prefillProgress(
            requestId: requestId,
            promptProcessingPhase: .target,
            processedTokens: 2_048,
            totalTokens: 50_000,
            elapsedMillis: 1_500,
            forwardPrefillChunkElapsedMillis: 1_400,
            completedPrefillChunkTokens: 2_048,
            mlxMemorySnapshot: WorkerMlxMemorySnapshot(
                source: .prefill,
                activeMemoryBytes: 11_000,
                allocatorCacheMemoryBytes: 12_000,
                peakMemoryBytes: 13_000,
                expertPayloadBytes: 4_000,
                modelCorePayloadBytes: 3_000,
                contextStatePayloadBytes: 2_000,
                memoryCeilingUtilization: nil),
            expertResidency: nil))
        try eventWriter.sendEvent(.firstDecodeCompleted(
            requestId: requestId,
            elapsedMillis: 321))
        try eventWriter.sendEvent(.output(
            requestId: requestId,
            sequenceNumber: 0,
            generatedTokenCount: 1,
            outputs: [.text(text: "done")],
            mlxMemorySnapshot: nil,
            expertResidency: nil))
        try eventWriter.sendEvent(.generationFinalized(
            requestId: requestId,
            expertMemoryMode: .resident,
            mlxMemorySnapshot: WorkerMlxMemorySnapshot(
                source: .finalized,
                activeMemoryBytes: 24_000,
                allocatorCacheMemoryBytes: 0,
                peakMemoryBytes: 25_000,
                expertPayloadBytes: 19_000,
                modelCorePayloadBytes: 3_000,
                contextStatePayloadBytes: 0,
                memoryCeilingUtilization: nil),
            expertResidency: nil))
        try eventWriter.sendEvent(IdleWorkerChatScenario.chatCompleted(
            requestId,
            promptTokenCount: 50_000,
            generatedTokenCount: 1,
            reason: .endOfSequence))
    }

    /// The activity journey: a pause in prompt processing, one output, a
    /// pause in generation, then completion — the timing that makes each
    /// activity transition observable on health state.
    private static func emitActivityTransition(
        _ requestId: RequestId,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        Thread.sleep(forTimeInterval: 0.1)
        try eventWriter.sendEvent(.output(
            requestId: requestId,
            sequenceNumber: 0,
            generatedTokenCount: 1,
            outputs: [.text(text: "activity transition")],
            mlxMemorySnapshot: nil,
            expertResidency: nil))
        Thread.sleep(forTimeInterval: 0.1)
        try eventWriter.sendEvent(IdleWorkerChatScenario.chatCompleted(
            requestId,
            promptTokenCount: 1,
            generatedTokenCount: 1,
            reason: .endOfSequence))
    }

    /// Twice the same once-only preparation boundary: the second frame is
    /// the breach the supervisor must contain.
    private static func emitDuplicateGenerationPreparation(
        _ requestId: RequestId,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        let generationPreparationEvent: WorkerEvent = .generationPreparationStarted(
            requestId: requestId,
            totalLayerCount: 40,
            residentExpertCount: 40,
            residentExpertPayloadBytes: 23_073_914_880,
            mlxMemorySnapshot: nil)
        try eventWriter.sendEvent(generationPreparationEvent)
        try eventWriter.sendEvent(generationPreparationEvent)
    }

    /// The undrained-stream journey: a long run of output frames the
    /// generation thread keeps producing, ending at an exact budget
    /// completion, so shutdown must interrupt the request rather than wait
    /// it out.
    private static func emitBackpressureFragments(
        _ requestId: RequestId,
        eventWriter: ProtocolWriter,
        maximumOutputTokens: UInt16
    ) throws -> Void {
        for outputFrameIndex: Int in 0..<IdleWorkerChatScenario.BACKPRESSURE_OUTPUT_FRAME_COUNT {
            if outputFrameIndex > 0 {
                Thread.sleep(forTimeInterval: IdleWorkerChatScenario.BACKPRESSURE_FRAME_INTERVAL_SECONDS)
            }
            try eventWriter.sendEvent(.output(
                requestId: requestId,
                sequenceNumber: UInt16(outputFrameIndex),
                generatedTokenCount: 1,
                outputs: [.text(text: "fragment")],
                mlxMemorySnapshot: nil,
                expertResidency: nil))
        }
        try eventWriter.sendEvent(IdleWorkerChatScenario.chatCompleted(
            requestId,
            promptTokenCount: 1,
            generatedTokenCount: maximumOutputTokens,
            reason: .maximumOutputTokens))
    }

    private static func chatCompleted(
        _ requestId: RequestId,
        promptTokenCount: UInt32,
        generatedTokenCount: UInt16,
        reason: ChatGenerationCompletionReason
    ) -> WorkerEvent {
        return .completed(
            requestId: requestId,
            promptTokenCount: promptTokenCount,
            generatedTokenCount: generatedTokenCount,
            reasoningTokenCount: 0,
            cachedTokenCount: 0,
            persistentPromptCacheDiagnostics: nil,
            reason: reason)
    }
}
