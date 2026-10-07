import Foundation

import IpcProtocol

/**
 * The per-request sequencing state the chat collection loop advances with
 * every accepted worker event, migrating the validation fields of the Rust
 * executor's ActiveGeneration: the next expected output sequence number and
 * tool-call index, the latest generated-token counts from both the output
 * and the progress path, and the once-only boundary markers.
 */
struct ChatGenerationRequestValidationState {

    var nextSequenceNumber: UInt16 = 0
    var nextToolCallIndex: UInt16 = 0
    var latestGeneratedTokenCount: UInt16 = 0
    var latestGenerationProgressTokenCount: UInt16 = 0
    var hasSeenGenerationPreparation: Bool = false
    var hasSeenFirstDecodeCompleted: Bool = false

    var latestKnownGeneratedTokenCount: UInt16 {
        return max(self.latestGeneratedTokenCount, self.latestGenerationProgressTokenCount)
    }
}

/**
 * Supervisor-side validation of one chat request's worker events, migrating
 * the correlation rules of apps/supervisor/src/worker_generation_output.rs,
 * worker_generation_preparation.rs, and worker_completion_event.rs. A
 * worker that breaks sequence ordering, emits an empty or internally
 * invalid output batch, duplicates a once-only boundary, or completes with
 * counts its history cannot support is a protocol breach the supervisor
 * contains; none of it ever reaches the public stream. Each validator
 * checks the whole event before the caller forwards any part of it.
 */
enum ChatGenerationEventValidation {

    /**
     * Validates one output frame in full — sequence position, generated
     * count bounds, batch non-emptiness, and tool-call index contiguity —
     * then advances the sequencing state so the next frame is judged
     * against the position this one established.
     */
    static func acceptOutputBatch(
        sequenceNumber: UInt16,
        generatedTokenCount: UInt16,
        outputs: [ChatGenerationOutput],
        maximumOutputTokens: UInt16,
        validationState: inout ChatGenerationRequestValidationState
    ) throws -> Void {
        if sequenceNumber != validationState.nextSequenceNumber
            || generatedTokenCount == 0
            || generatedTokenCount < validationState.latestKnownGeneratedTokenCount
            || generatedTokenCount > maximumOutputTokens {
            throw WorkerControlError.workerProtocolViolation(
                description: "output correlation or sequence mismatch")
        }
        if outputs.isEmpty {
            throw WorkerControlError.workerProtocolViolation(
                description: "output batch must not be empty")
        }
        var candidateToolCallIndex: UInt16 = validationState.nextToolCallIndex
        for workerOutput: ChatGenerationOutput in outputs {
            if case let .toolCall(toolCallIndex, _, _) = workerOutput {
                if toolCallIndex != candidateToolCallIndex {
                    throw WorkerControlError.workerProtocolViolation(
                        description: "tool-call index mismatch")
                }
                let (incrementedToolCallIndex, didOverflowToolCallIndex) =
                    candidateToolCallIndex.addingReportingOverflow(1)
                if didOverflowToolCallIndex {
                    throw WorkerControlError.workerProtocolViolation(
                        description: "tool-call index overflow")
                }
                candidateToolCallIndex = incrementedToolCallIndex
            }
        }
        guard let outputBatchCount: UInt16 = UInt16(exactly: outputs.count) else {
            throw WorkerControlError.workerProtocolViolation(
                description: "output batch count exceeds the u16 range")
        }
        let (incrementedSequenceNumber, didOverflowSequenceNumber) =
            validationState.nextSequenceNumber.addingReportingOverflow(outputBatchCount)
        if didOverflowSequenceNumber {
            throw WorkerControlError.workerProtocolViolation(
                description: "output sequence overflow")
        }
        validationState.nextSequenceNumber = incrementedSequenceNumber
        validationState.nextToolCallIndex = candidateToolCallIndex
        validationState.latestGeneratedTokenCount = generatedTokenCount
    }

    /**
     * Accepts the once-only prefill-to-decode preparation boundary,
     * rejecting duplicates and residency payloads whose count and byte
     * totals disagree about whether any expert is resident at all.
     */
    static func acceptGenerationPreparation(
        residentExpertCount: UInt32,
        residentExpertPayloadBytes: UInt64,
        validationState: inout ChatGenerationRequestValidationState
    ) throws -> Void {
        let isResidencyPayloadConsistent: Bool =
            (residentExpertCount == 0) == (residentExpertPayloadBytes == 0)
        if !isResidencyPayloadConsistent || validationState.hasSeenGenerationPreparation {
            throw WorkerControlError.workerProtocolViolation(
                description: "generation preparation was duplicated or had a correlation or payload mismatch")
        }
        validationState.hasSeenGenerationPreparation = true
    }

    /**
     * Accepts one decode-progress observation, rejecting zero, regressing,
     * or over-budget counts and a maximum that contradicts the admitted
     * command's bound.
     */
    static func acceptGenerationProgress(
        generatedTokenCount: UInt16,
        eventMaximumOutputTokens: UInt16,
        maximumOutputTokens: UInt16,
        validationState: inout ChatGenerationRequestValidationState
    ) throws -> Void {
        if generatedTokenCount == 0
            || generatedTokenCount < validationState.latestKnownGeneratedTokenCount
            || generatedTokenCount > maximumOutputTokens
            || eventMaximumOutputTokens != maximumOutputTokens {
            throw WorkerControlError.workerProtocolViolation(
                description: "generation progress correlation or count mismatch")
        }
        validationState.latestGenerationProgressTokenCount = generatedTokenCount
    }

    /** Accepts the measured first decode exactly once per request. */
    static func acceptFirstDecodeCompleted(
        validationState: inout ChatGenerationRequestValidationState
    ) throws -> Void {
        if validationState.hasSeenFirstDecodeCompleted {
            throw WorkerControlError.workerProtocolViolation(
                description: "first decode completion correlation or duplication mismatch")
        }
        validationState.hasSeenFirstDecodeCompleted = true
    }

    /**
     * Validates one completion against the request's history: the count
     * must not regress or exceed the budget, and the reason must be one
     * the history supports — end-of-sequence within budget, an exact
     * budget completion, tool calls only after a validated tool call, and
     * never an unsolicited cancellation.
     */
    static func acceptCompletion(
        generatedTokenCount: UInt16,
        reason: ChatGenerationCompletionReason,
        maximumOutputTokens: UInt16,
        validationState: inout ChatGenerationRequestValidationState
    ) throws -> Void {
        let isCompletionReasonSupported: Bool
        switch (reason) {
        case .endOfSequence:
            isCompletionReasonSupported = generatedTokenCount <= maximumOutputTokens
        case .maximumOutputTokens:
            isCompletionReasonSupported = generatedTokenCount == maximumOutputTokens
        case .toolCalls:
            isCompletionReasonSupported = validationState.nextToolCallIndex > 0
        case .cancelled:
            isCompletionReasonSupported = false
        }
        if generatedTokenCount < validationState.latestKnownGeneratedTokenCount
            || generatedTokenCount > maximumOutputTokens
            || !isCompletionReasonSupported {
            throw WorkerControlError.workerProtocolViolation(
                description: "completion correlation or count mismatch")
        }
    }
}
