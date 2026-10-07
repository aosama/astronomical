import Foundation

import Testing

import IpcProtocol

@testable import Supervisor

/**
 * Acceptance journeys for the supervisor's chat generation executor,
 * migrating apps/supervisor/tests/hermetic/chat_generation_executor.rs:
 * ordered streaming of a scripted happy path, prefill telemetry replaced
 * by finalized residency memory, activity transitions published without
 * worker phase events, malformed output reported while the worker stays
 * reusable, one-frame bounded IPC commands, and every protocol breach —
 * out-of-order sequences, duplicated preparation boundaries, empty or
 * invalid output batches, over-budget completions, unsolicited
 * cancellations, an undrained stream at shutdown, a disconnected stream's
 * capacity reuse, and a worker that exits mid-request — contained with the
 * stream's terminal error and the worker's health.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class ChatGenerationExecutorJourneyTests {

    @Test
    func should_stream_ordered_chat_outputs() throws {
        let journey: ChatExecutorJourney = try ChatExecutorJourney.launch()
        defer { journey.dispose() }

        let streamEvents: [ChatGenerationStreamEvent] = try journey.supervisor.startChatGeneration(
            journey.chatCommand(modelId: ChatExecutorFixture.ACCEPTED_CHAT_MODEL_ID))

        #expect(streamEvents == [
            .reasoningFragment("accepted chat reasoning"),
            .textFragment("accepted chat text"),
            .toolCall(
                toolCallIndex: 0,
                functionName: "read",
                argumentsJson: #"{"path":"AGENTS.md"}"#),
            .toolCall(
                toolCallIndex: 1,
                functionName: "glob",
                argumentsJson: #"{"pattern":"tests/**/*.rs"}"#),
            .completed(
                promptTokenCount: 2,
                generatedTokenCount: 4,
                reasoningTokenCount: 0,
                cachedTokenCount: 0,
                reason: .toolCalls),
        ])
        #expect(journey.supervisor.workerHealthSnapshot().status == .ready)
    }

    @Test
    func should_replace_prefill_memory_with_finalized_residency_memory() throws {
        let journey: ChatExecutorJourney = try ChatExecutorJourney.launch()
        defer { journey.dispose() }

        let streamEvents: [ChatGenerationStreamEvent] = try journey.supervisor.startChatGeneration(
            journey.chatCommand(modelId: ChatExecutorFixture.PREFILL_PROGRESS_MODEL_ID))

        #expect(streamEvents.first == .prefillProgress(
            processedTokens: 2_048,
            totalTokens: 50_000,
            elapsedMillis: 1_500,
            forwardPrefillChunkElapsedMillis: 1_400,
            completedPrefillChunkTokens: 2_048,
            mlxActiveMemoryBytes: 11_000,
            mlxAllocatorCacheMemoryBytes: 12_000,
            mlxPeakMemoryBytes: 13_000))
        guard case let .textFragment(prefillText)? = streamEvents.dropFirst().first else {
            Issue.record(Comment(stringLiteral: "expected a text fragment after prefill progress"))
            return
        }
        #expect(!prefillText.isEmpty)
        guard case .completed = streamEvents.last else {
            Issue.record(Comment(stringLiteral: "expected completion after the text fragment"))
            return
        }
        let healthSnapshot: WorkerHealthSnapshot = journey.supervisor.workerHealthSnapshot()
        let finalizedSnapshot: WorkerMlxMemorySnapshot = try #require(
            healthSnapshot.latestMlxMemorySnapshot,
            "finalized telemetry should replace prefill telemetry")
        #expect(finalizedSnapshot.source == .finalized)
        #expect(finalizedSnapshot.expertPayloadBytes == 19_000)
        #expect(finalizedSnapshot.modelCorePayloadBytes == 3_000)
        #expect(finalizedSnapshot.contextStatePayloadBytes == 0)
        #expect(finalizedSnapshot.activeMemoryBytes == 24_000)
        #expect(healthSnapshot.expertMemoryMode == .resident)
    }

    @Test
    func should_track_generation_activity_without_worker_phase_events() throws {
        let journey: ChatExecutorJourney = try ChatExecutorJourney.launch()
        defer { journey.dispose() }

        let eventCollector: ExecutorStreamedCollector = ExecutorStreamedCollector()
        let streamHandle: ChatGenerationStreamHandle = try journey.supervisor.startChatGenerationStream(
            journey.chatCommand(modelId: ChatExecutorFixture.ACTIVITY_TRANSITION_MODEL_ID),
            onEvent: { (streamEvent: ChatGenerationStreamEvent) -> Void in
                eventCollector.append(streamEvent)
            })
        journey.waitForActivity(.promptProcessing, journeyLabel: "prefill activity")
        journey.waitUntil(
            { () -> Bool in return eventCollector.hasTextFragment() },
            journeyLabel: "activity transition text fragment",
            timeoutSeconds: 2)
        journey.waitForActivity(.generating, journeyLabel: "generating activity")
        journey.waitUntil(
            { () -> Bool in return eventCollector.hasStreamEnded() },
            journeyLabel: "activity transition completion",
            timeoutSeconds: 2)
        journey.waitForActivity(.idle, journeyLabel: "idle activity")
        withExtendedLifetime(streamHandle) { () -> Void in return }
    }

    @Test
    func should_report_malformed_output_and_reuse_the_worker() throws {
        let journey: ChatExecutorJourney = try ChatExecutorJourney.launch()
        defer { journey.dispose() }

        let malformedEvents: [ChatGenerationStreamEvent] = try journey.collectUntilStreamEnd(
            modelId: ChatExecutorFixture.MALFORMED_OUTPUT_MODEL_ID)
        #expect(malformedEvents == [.failed(reason: .malformedModelOutput)])

        let followupEvents: [ChatGenerationStreamEvent] = try journey.supervisor.startChatGeneration(
            journey.chatCommand(modelId: ChatExecutorJourney.FOLLOWUP_MODEL_ID))
        IdleWorkerJourneySupport.assertGenerationCompleted(
            followupEvents,
            generationLabel: "follow-up request after malformed output")
        #expect(journey.supervisor.workerHealthSnapshot().status == .ready)
    }

    @Test
    func should_send_large_but_bounded_chat_commands_to_the_worker_in_one_ipc_frame() throws {
        let journey: ChatExecutorJourney = try ChatExecutorJourney.launch()
        defer { journey.dispose() }

        let largeCommand: ChatGenerationCommand = journey.chatCommand(
            modelId: ChatExecutorJourney.FOLLOWUP_MODEL_ID,
            requestId: 810,
            userContent: String(repeating: "x", count: ChatExecutorJourney.RETIRED_SMALL_FRAME_BYTES * 2))
        let serializedCommandBytes: Data = try MessageCodec.encodeCommand(.generate(largeCommand))
        #expect(serializedCommandBytes.count > ChatExecutorJourney.RETIRED_SMALL_FRAME_BYTES)
        #expect(serializedCommandBytes.count <= ChatExecutorJourney.MAXIMUM_IPC_FRAME_BYTES)

        let streamEvents: [ChatGenerationStreamEvent] = try journey.supervisor.startChatGeneration(
            largeCommand)
        #expect(streamEvents == [.completed(
            promptTokenCount: 1,
            generatedTokenCount: 0,
            reasoningTokenCount: 0,
            cachedTokenCount: 0,
            reason: .endOfSequence)])
    }

    @Test
    func should_terminate_worker_after_an_out_of_order_event() throws {
        let journey: ChatExecutorJourney = try ChatExecutorJourney.launch()
        defer { journey.dispose() }

        let breachEvents: [ChatGenerationStreamEvent] = try journey.collectUntilStreamEnd(
            modelId: ChatExecutorFixture.OUT_OF_ORDER_MODEL_ID)
        #expect(breachEvents == [.streamError(.workerUnavailable)])
        journey.waitForHealth(.unavailable, journeyLabel: "worker after out-of-order event")
    }

    @Test
    func should_terminate_worker_after_duplicate_generation_preparation() throws {
        let journey: ChatExecutorJourney = try ChatExecutorJourney.launch()
        defer { journey.dispose() }

        let breachEvents: [ChatGenerationStreamEvent] = try journey.collectUntilStreamEnd(
            modelId: ChatExecutorFixture.DUPLICATE_GENERATION_PREPARATION_MODEL_ID)
        #expect(breachEvents == [.streamError(.workerUnavailable)])
        journey.waitForHealth(.unavailable, journeyLabel: "worker after duplicated preparation")
    }

    @Test
    func should_terminate_worker_after_an_empty_output_batch() throws {
        let journey: ChatExecutorJourney = try ChatExecutorJourney.launch()
        defer { journey.dispose() }

        let breachEvents: [ChatGenerationStreamEvent] = try journey.collectUntilStreamEnd(
            modelId: ChatExecutorFixture.EMPTY_OUTPUT_BATCH_MODEL_ID)
        #expect(breachEvents == [.streamError(.workerUnavailable)])
        journey.waitForHealth(.unavailable, journeyLabel: "worker after empty output batch")
    }

    @Test
    func should_validate_an_entire_output_batch_before_forwarding_any_entry() throws {
        let journey: ChatExecutorJourney = try ChatExecutorJourney.launch()
        defer { journey.dispose() }

        let breachEvents: [ChatGenerationStreamEvent] = try journey.collectUntilStreamEnd(
            modelId: ChatExecutorFixture.INVALID_OUTPUT_BATCH_MODEL_ID)
        #expect(breachEvents == [.streamError(.workerUnavailable)])
        journey.waitForHealth(.unavailable, journeyLabel: "worker after invalid output batch")
    }

    @Test
    func should_terminate_worker_after_an_over_budget_tool_completion() throws {
        let journey: ChatExecutorJourney = try ChatExecutorJourney.launch()
        defer { journey.dispose() }

        let breachEvents: [ChatGenerationStreamEvent] = try journey.collectUntilStreamEnd(
            modelId: ChatExecutorFixture.OVER_BUDGET_TOOL_COMPLETION_MODEL_ID)
        #expect(breachEvents == [
            .toolCall(
                toolCallIndex: 0,
                functionName: "read",
                argumentsJson: #"{"path":"AGENTS.md"}"#),
            .streamError(.workerUnavailable),
        ])
        journey.waitForHealth(.unavailable, journeyLabel: "worker after over-budget completion")
    }

    @Test
    func should_terminate_worker_after_an_unsolicited_cancellation() throws {
        let journey: ChatExecutorJourney = try ChatExecutorJourney.launch()
        defer { journey.dispose() }

        let breachEvents: [ChatGenerationStreamEvent] = try journey.collectUntilStreamEnd(
            modelId: ChatExecutorFixture.UNSOLICITED_CANCELLATION_MODEL_ID)
        #expect(breachEvents == [.streamError(.workerUnavailable)])
        journey.waitForHealth(.unavailable, journeyLabel: "worker after unsolicited cancellation")
    }

    @Test
    func should_shut_down_when_the_http_stream_stops_consuming_output() throws {
        let journey: ChatExecutorJourney = try ChatExecutorJourney.launch()

        let backpressureCommand: ChatGenerationCommand = journey.chatCommand(
            modelId: ChatExecutorFixture.BACKPRESSURE_MODEL_ID,
            maximumOutputTokens: 1)
        let eventCollector: ExecutorStreamedCollector = ExecutorStreamedCollector()
        let streamHandle: ChatGenerationStreamHandle = try journey.supervisor.startChatGenerationStream(
            backpressureCommand,
            onEvent: { (streamEvent: ChatGenerationStreamEvent) -> Void in
                eventCollector.append(streamEvent)
            })
        Thread.sleep(forTimeInterval: 0.2)
        let shutdownOutcome: ThreadedShutdownOutcome = ThreadedShutdownOutcome()
        let supervisor: WorkerSupervisor = journey.supervisor
        let shutdownThread: Thread = Thread(block: { () -> Void in
            do {
                _ = try supervisor.shutdown()
                shutdownOutcome.recordSuccess()
            } catch let shutdownError {
                shutdownOutcome.recordFailure(shutdownError)
            }
        })
        shutdownThread.name = "astronomical-backpressure-shutdown"
        shutdownThread.start()
        let observedShutdownResult: Result<Void, Error>? = shutdownOutcome.awaitOutcome(deadlineSeconds: 2)
        withExtendedLifetime(streamHandle) { () -> Void in return }
        journey.dispose()
        guard case .success? = observedShutdownResult else {
            Issue.record(Comment(stringLiteral:
                "shutdown must not block behind an undrained HTTP stream: \(String(describing: observedShutdownResult))"))
            return
        }
    }

    @Test
    func should_cancel_a_disconnected_stream_and_reuse_capacity() throws {
        let journey: ChatExecutorJourney = try ChatExecutorJourney.launch()
        defer { journey.dispose() }

        let streamHandle: ChatGenerationStreamHandle = try journey.supervisor.startChatGenerationStream(
            journey.chatCommand(modelId: ChatExecutorFixture.DELAYED_FRAGMENT_CHAT_MODEL_ID),
            onEvent: { (_: ChatGenerationStreamEvent) -> Void in return })
        streamHandle.abandon()

        let followupOutcome: ThreadedChatOutcome = ThreadedChatOutcome()
        let supervisor: WorkerSupervisor = journey.supervisor
        let followupDispatch: ThreadedChatDispatch = ThreadedChatDispatch(
            chatCommand: journey.chatCommand(modelId: ChatExecutorJourney.FOLLOWUP_MODEL_ID))
        let followupThread: Thread = Thread(block: { () -> Void in
            do {
                followupOutcome.record(streamEvents: try supervisor.startChatGeneration(followupDispatch.chatCommand))
            } catch let followupError {
                followupOutcome.record(followupError: followupError)
            }
        })
        followupThread.name = "astronomical-disconnect-followup"
        followupThread.start()
        let observedOutcome: Result<[ChatGenerationStreamEvent], Error>? =
            followupOutcome.awaitOutcome(deadlineSeconds: 2)
        guard case let .success(followupEvents)? = observedOutcome else {
            Issue.record(Comment(stringLiteral:
                "follow-up request after disconnect failed: \(String(describing: observedOutcome))"))
            return
        }
        IdleWorkerJourneySupport.assertGenerationCompleted(
            followupEvents,
            generationLabel: "follow-up request after stream disconnect")
    }

    @Test
    func should_fail_one_stream_when_the_worker_exits() throws {
        let journey: ChatExecutorJourney = try ChatExecutorJourney.launch()
        defer { journey.dispose() }

        let breachEvents: [ChatGenerationStreamEvent] = try journey.collectUntilStreamEnd(
            modelId: ChatExecutorFixture.EXIT_AFTER_CHAT_ADMISSION_MODEL_ID)
        #expect(breachEvents == [.streamError(.workerUnavailable)])
    }
}
