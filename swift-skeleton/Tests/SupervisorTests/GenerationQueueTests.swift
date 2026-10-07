import Foundation;

import Testing;

import IpcProtocol;
import JourneyCategories;

@testable import Supervisor;

/**
 * Hermetic journeys for the supervisor's bounded FIFO generation queue and
 * its queued memory-limit interplay, migrating
 * apps/supervisor/tests/hermetic/request_queue.rs over the shared
 * marker-driven fake worker harness.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class GenerationQueueTests {

    @Test
    func should_queue_the_second_request_until_the_first_completes() throws {
        let harness: FakeWorkerJourneyHarness = try FakeWorkerJourneyHarness.launch();
        defer { harness.dispose() }

        let firstOutcome: GenerationJourneyOutcome = harness.startGenerationThread(requestId: 1);
        try harness.awaitQueueFill(expectedOutstandingCount: 1);

        // The second request must queue, not reject, while the first owns
        // the worker.
        let queuedOutcome: GenerationJourneyOutcome = harness.startGenerationThread(requestId: 2);
        try harness.awaitQueueFill(expectedOutstandingCount: 2);

        try harness.pokeCompletion(requestId: 1);
        GenerationQueueTests.join(firstOutcome, deadline: Date().addingTimeInterval(10));
        #expect(firstOutcome.isSuccessful);

        try harness.pokeCompletion(requestId: 2);
        GenerationQueueTests.join(queuedOutcome, deadline: Date().addingTimeInterval(10));
        guard case .success(let queuedEvents)? = queuedOutcome.observedOutcome else {
            Issue.record("the queued request should succeed, got \(queuedOutcome.observedOutcome)");
            return;
        }
        #expect(queuedEvents.last == .completed(
            promptTokenCount: 1,
            generatedTokenCount: 1,
            reasoningTokenCount: 0,
            cachedTokenCount: 0,
            reason: .endOfSequence));
    }

    @Test
    func should_apply_a_memory_limit_immediately_on_an_idle_worker() throws {
        let harness: FakeWorkerJourneyHarness = try FakeWorkerJourneyHarness.launch();
        defer { harness.dispose() }

        // The update blocks on the worker acknowledgement, so the marker the
        // fake answers with is placed from a side thread.
        let updateOutcome: GenerationJourneyOutcome = GenerationJourneyOutcome(workerThread: Thread());
        let updateThread: Thread = Thread {
            do {
                _ = try harness.supervisor.updateMlxMemoryLimit(
                    32_000_000_000,
                    configurationGeneration: "idle-apply");
                updateOutcome.record(streamEvents: []);
            } catch {
                updateOutcome.record(error: error);
            }
        };
        updateThread.name = "idle-memory-update";
        updateOutcome.workerThread = updateThread;
        updateThread.start();
        try harness.pokeMemoryRaise(32_000_000_000);
        GenerationQueueTests.join(updateOutcome, deadline: Date().addingTimeInterval(10));
        #expect(updateOutcome.isSuccessful);

        GenerationQueueTests.awaitRaiseApplied(harness: harness);
    }

    @Test
    func should_queue_a_memory_limit_while_generation_is_active() throws {
        let harness: FakeWorkerJourneyHarness = try FakeWorkerJourneyHarness.launch();
        defer { harness.dispose() }

        let activeOutcome: GenerationJourneyOutcome = harness.startGenerationThread(requestId: 1);
        try harness.awaitQueueFill(expectedOutstandingCount: 1);

        #expect(try harness.supervisor.updateMlxMemoryLimit(
            32_000_000_000,
            configurationGeneration: "queued-memory-generation") == .queued);
        #expect(harness.supervisor.workerHealthSnapshot().pendingMlxMemoryCeilingBytes == 32_000_000_000);

        try harness.pokeCompletion(requestId: 1);
        try harness.pokeMemoryRaise(32_000_000_000);
        GenerationQueueTests.join(activeOutcome, deadline: Date().addingTimeInterval(10));
        #expect(activeOutcome.isSuccessful);

        // The queued raise is applied when the active request finalizes.
        GenerationQueueTests.awaitRaiseApplied(harness: harness);
    }

    @Test
    func should_apply_a_queued_memory_limit_before_the_next_chat_starts() throws {
        let harness: FakeWorkerJourneyHarness = try FakeWorkerJourneyHarness.launch();
        defer { harness.dispose() }

        let activeOutcome: GenerationJourneyOutcome = harness.startGenerationThread(requestId: 1);
        try harness.awaitQueueFill(expectedOutstandingCount: 1);

        #expect(try harness.supervisor.updateMlxMemoryLimit(
            32_000_000_000,
            configurationGeneration: "raise-before-next-chat") == .queued);

        let nextOutcome: GenerationJourneyOutcome = harness.startGenerationThread(requestId: 2);
        try harness.awaitQueueFill(expectedOutstandingCount: 2);

        try harness.pokeCompletion(requestId: 1);
        try harness.pokeMemoryRaise(32_000_000_000);
        GenerationQueueTests.join(activeOutcome, deadline: Date().addingTimeInterval(10));
        #expect(activeOutcome.isSuccessful);

        // The release applies the raise before the queued chat is admitted,
        // so the next chat starts against the raised ceiling.
        GenerationQueueTests.awaitRaiseApplied(harness: harness);

        try harness.pokeCompletion(requestId: 2);
        GenerationQueueTests.join(nextOutcome, deadline: Date().addingTimeInterval(10));
        guard case .success(let nextEvents)? = nextOutcome.observedOutcome else {
            Issue.record("the next chat should succeed, got \(nextOutcome.observedOutcome)");
            return;
        }
        #expect(nextEvents.last == .completed(
            promptTokenCount: 1,
            generatedTokenCount: 1,
            reasoningTokenCount: 0,
            cachedTokenCount: 0,
            reason: .endOfSequence));
    }

    @Test
    func should_reject_the_request_beyond_the_queue_depth() throws {
        let harness: FakeWorkerJourneyHarness = try FakeWorkerJourneyHarness.launch();
        defer { harness.dispose() }

        let activeOutcome: GenerationJourneyOutcome = harness.startGenerationThread(requestId: 1);
        try harness.awaitQueueFill(expectedOutstandingCount: 1);

        var queuedOutcomes: Array<GenerationJourneyOutcome> = Array<GenerationJourneyOutcome>();
        for waiterIndex: Int in 0..<GenerationQueueDepth.maximumWaiterCount {
            // Each waiter starts only once the previous one holds a queued
            // ticket, so the FIFO order is the started request-id order and
            // the completion pokes can address the active request exactly.
            queuedOutcomes.append(harness.startGenerationThread(requestId: UInt64(waiterIndex) + 2));
            try harness.awaitQueueFill(expectedOutstandingCount: waiterIndex + 2);
        }

        do {
            _ = try harness.supervisor.startChatGeneration(
                FakeWorkerJourneyHarness.chatGenerationCommand(requestId: 10, modelId: "m1"));
            Issue.record("expected the request beyond the queue depth to be rejected");
        } catch let thrownError as GenerationStartError {
            #expect(thrownError == .capacityUnavailable);
        }

        // Every queued request still drains in arrival order once the active
        // one completes.
        for drainedRequestId: UInt64 in 1...UInt64(GenerationQueueDepth.maximumWaiterCount + 1) {
            let drainedOutcome: GenerationJourneyOutcome = drainedRequestId == 1
                ? activeOutcome
                : queuedOutcomes[Int(drainedRequestId) - 2];
            try harness.pokeCompletion(requestId: drainedRequestId);
            GenerationQueueTests.join(drainedOutcome, deadline: Date().addingTimeInterval(10));
            guard case .success? = drainedOutcome.observedOutcome else {
                Issue.record("the queued request \(drainedRequestId) should succeed, got \(drainedOutcome.observedOutcome)");
                return;
            }
        }
    }

    // MARK: - Journey plumbing

    /// Blocks until a queued memory raise is visible as applied: the
    /// effective ceiling raised and the pending flag cleared.
    private static func awaitRaiseApplied(harness: FakeWorkerJourneyHarness) -> Void {
        let raiseDeadline: Date = Date().addingTimeInterval(5);
        while true {
            let workerHealthSnapshot: WorkerHealthSnapshot = harness.supervisor.workerHealthSnapshot();
            if workerHealthSnapshot.effectiveMlxMemoryCeilingBytes == 32_000_000_000
                && workerHealthSnapshot.pendingMlxMemoryCeilingBytes == nil {
                break;
            }
            #expect(Date() < raiseDeadline,
                "the queued memory limit was not applied after finalization: \(workerHealthSnapshot)");
            if Date() >= raiseDeadline {
                return;
            }
            Thread.sleep(forTimeInterval: 0.01);
        }
    }

    /// Joins one generation thread, recording a failure when it outlives the
    /// deadline so every journey carries its own bounded wait.
    private static func join(_ outcome: GenerationJourneyOutcome, deadline: Date) -> Void {
        while outcome.workerThread.isFinished == false && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.01);
        }
        #expect(outcome.workerThread.isFinished, "the generation thread outlived its deadline");
    }
}
