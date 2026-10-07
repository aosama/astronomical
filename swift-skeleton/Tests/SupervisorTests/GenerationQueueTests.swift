import Foundation;

import Testing;

import AstronomicalConfig;

import IpcProtocol;
import JourneyCategories;

@testable import Supervisor;

/**
 * Hermetic journeys for the supervisor's bounded FIFO generation queue and
 * its queued memory-limit interplay, migrating
 * apps/supervisor/tests/hermetic/request_queue.rs. The fake worker is a
 * short shell script with a resident model acknowledged at startup, and the
 * journeys steer it through marker files in a temporary control directory:
 * one marker completes the pending generation with a chosen request id, the
 * other acknowledges a memory-ceiling raise.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class GenerationQueueTests {

    @Test
    func should_queue_the_second_request_until_the_first_completes() throws {
        let harness: GenerationQueueTestsHarness = try GenerationQueueTestsHarness.launch();
        defer { harness.dispose() }

        let firstOutcome: GenerationJourneyOutcome = harness.startGenerationThread(requestId: 1);
        Thread.sleep(forTimeInterval: 0.5);

        // The second request must queue, not reject, while the first owns
        // the worker.
        let queuedOutcome: GenerationJourneyOutcome = harness.startGenerationThread(requestId: 2);
        Thread.sleep(forTimeInterval: 0.4);

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
        let harness: GenerationQueueTestsHarness = try GenerationQueueTestsHarness.launch();
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
        GenerationQueueTests.awaitQueueFill(harness: harness, expectedOutstandingCount: 0);
        try harness.pokeMemoryRaise(32_000_000_000);
        GenerationQueueTests.join(updateOutcome, deadline: Date().addingTimeInterval(10));
        #expect(updateOutcome.isSuccessful);

        let raiseDeadline: Date = Date().addingTimeInterval(5);
        while true {
            let workerHealthSnapshot: WorkerHealthSnapshot = harness.supervisor.workerHealthSnapshot();
            if workerHealthSnapshot.effectiveMlxMemoryCeilingBytes == 32_000_000_000
                && workerHealthSnapshot.pendingMlxMemoryCeilingBytes == nil {
                break;
            }
            #expect(Date() < raiseDeadline,
                "the idle raise was not applied: \(workerHealthSnapshot)");
            if Date() >= raiseDeadline {
                return;
            }
            Thread.sleep(forTimeInterval: 0.01);
        }
    }

    @Test
    func should_queue_a_memory_limit_while_generation_is_active() throws {
        let harness: GenerationQueueTestsHarness = try GenerationQueueTestsHarness.launch();
        defer { harness.dispose() }

        let activeOutcome: GenerationJourneyOutcome = harness.startGenerationThread(requestId: 1);
        Thread.sleep(forTimeInterval: 0.5);

        #expect(try harness.supervisor.updateMlxMemoryLimit(
            32_000_000_000,
            configurationGeneration: "queued-memory-generation") == .queued);
        #expect(harness.supervisor.workerHealthSnapshot().pendingMlxMemoryCeilingBytes == 32_000_000_000);

        try harness.pokeCompletion(requestId: 1);
        try harness.pokeMemoryRaise(32_000_000_000);
        GenerationQueueTests.join(activeOutcome, deadline: Date().addingTimeInterval(10));
        #expect(activeOutcome.isSuccessful);
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

    @Test
    func should_apply_a_queued_memory_limit_before_the_next_chat_starts() throws {
        let harness: GenerationQueueTestsHarness = try GenerationQueueTestsHarness.launch();
        defer { harness.dispose() }

        let activeOutcome: GenerationJourneyOutcome = harness.startGenerationThread(requestId: 1);
        Thread.sleep(forTimeInterval: 0.5);

        #expect(try harness.supervisor.updateMlxMemoryLimit(
            32_000_000_000,
            configurationGeneration: "raise-before-next-chat") == .queued);

        let nextOutcome: GenerationJourneyOutcome = harness.startGenerationThread(requestId: 2);
        Thread.sleep(forTimeInterval: 0.3);

        try harness.pokeCompletion(requestId: 1);
        try harness.pokeMemoryRaise(32_000_000_000);
        GenerationQueueTests.join(activeOutcome, deadline: Date().addingTimeInterval(10));
        #expect(activeOutcome.isSuccessful);
        let raiseDeadline: Date = Date().addingTimeInterval(5);
        while true {
            let workerHealthSnapshot: WorkerHealthSnapshot = harness.supervisor.workerHealthSnapshot();
            if workerHealthSnapshot.effectiveMlxMemoryCeilingBytes == 32_000_000_000
                && workerHealthSnapshot.pendingMlxMemoryCeilingBytes == nil {
                break;
            }
            #expect(Date() < raiseDeadline,
                "the raise must apply before the next chat is admitted: \(workerHealthSnapshot)");
            if Date() >= raiseDeadline {
                return;
            }
            Thread.sleep(forTimeInterval: 0.01);
        }

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
        let harness: GenerationQueueTestsHarness = try GenerationQueueTestsHarness.launch();
        defer { harness.dispose() }

        let activeOutcome: GenerationJourneyOutcome = harness.startGenerationThread(requestId: 1);
        GenerationQueueTests.awaitQueueFill(harness: harness, expectedOutstandingCount: 1);

        var queuedOutcomes: Array<GenerationJourneyOutcome> = Array<GenerationJourneyOutcome>();
        for waiterIndex: Int in 0..<GenerationQueueDepth.maximumWaiterCount {
            // Each waiter starts only once the previous one holds a queued
            // ticket, so the FIFO order is the started request-id order and
            // the completion pokes can address the active request exactly.
            queuedOutcomes.append(harness.startGenerationThread(requestId: UInt64(waiterIndex) + 2));
            GenerationQueueTests.awaitQueueFill(
                harness: harness,
                expectedOutstandingCount: waiterIndex + 2);
        }

        do {
            _ = try harness.supervisor.startChatGeneration(
                GenerationQueueTests.chatGenerationCommand(requestId: 10, modelId: "m1"));
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

    /// Blocks until the issued admission tickets reach the expected
    /// outstanding count, proving the just-started requests are queued (or
    /// active) before the journey proceeds.
    private static func awaitQueueFill(harness: GenerationQueueTestsHarness, expectedOutstandingCount: Int) -> Void {
        let fillDeadline: Date = Date().addingTimeInterval(5);
        while harness.supervisor.outstandingAdmissionTicketCount < expectedOutstandingCount {
            if Date() >= fillDeadline {
                Issue.record("the queue never reached \(expectedOutstandingCount) outstanding requests");
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

    fileprivate static func chatGenerationCommand(requestId: UInt64, modelId: String) -> ChatGenerationCommand {
        return ChatGenerationCommand(
            requestId: RequestId(rawRequestId: requestId),
            model: modelId,
            messages: [.user(content: "hello", images: [])],
            tools: [],
            toolChoice: .auto,
            settings: ChatGenerationSettings(
                maxOutputTokens: 16,
                temperatureThousandths: nil,
                topPThousandths: nil,
                seed: nil,
                thinkingBudget: nil),
            qwenThinkingChannelSeed: nil,
            structuredGeneration: nil);
    }

    fileprivate static func modelPolicy(modelId: String) -> RuntimeModelPolicy {
        return RuntimeModelPolicy(
            modelDirectory: FilePath(string: "/fictional/models/\(modelId)"),
            generationDefaults: RuntimeModelGenerationDefaults(
                maximumOutputTokens: 512,
                configuredMaximumOutputTokens: 512,
                temperatureThousandths: nil,
                topPThousandths: nil),
            configuredMaximumContextTokens: 4096,
            defaultMaximumContextTokens: 8192,
            configuredChunkingFields: ConfiguredChunkingFields.inactive(),
            workerModelConfiguration: WorkerModelConfiguration.autoregressive(
                WorkerAutoregressiveModelConfiguration(
                    modelId: modelId,
                    maximumContextTokens: 4096,
                    maximumOutputTokens: 1024,
                    chunking: WorkerChunkingConfiguration(
                        fixedPromptProcessingChunkSizeTokens: 512,
                        fixedSsdStreamingPromptProcessingChunkSizeTokens: 512,
                        fullAttentionKeyValueGrowthTokens: 512,
                        prefillGraphSubmissionLayerInterval: 1,
                        experimentalSsdPagingPrefillGraphSubmissionLayerInterval: 1,
                        experimentalSsdPagingGenerationGraphSubmissionLayerInterval: 1,
                        promptCacheBlockTokens: nil,
                        promptCacheCommonPrefixStrideBlocks: 1,
                        experimentalDecodeStageAttributionEnabled: false,
                        experimentalQuantizedKvCacheEnabled: false,
                        experimentalFusedMoeDecodeEnabled: false))));
    }

    fileprivate static func startupConfiguration() -> WorkerStartupConfiguration {
        return WorkerStartupConfiguration(
            configurationGeneration: "gen-1",
            globalPromptCacheRootDirectory: "/tmp/astronomical-supervisor-cache",
            globalPromptCacheMaximumSizeBytes: 1073741824,
            persistentPromptCacheEnabled: true,
            configuredMaximumMlxMemoryBytes: nil,
            performanceAttributionEnabled: false,
            loggingDirectory: "/tmp/astronomical-supervisor-logs",
            loggingLevel: .info,
            retainedLogFileCount: 3);
    }

    /// The fake worker: startup acknowledges the resident model, then the
    /// polling loop completes whichever request id the journey last poked and
    /// acknowledges whichever ceiling raise the supervisor sent.
    fileprivate static func controlGatedWorkerScript(controlDirectoryPath: String) -> String {
        return FakeWorkerEventEmitter.frameEmitterFunction()
            + FakeWorkerEventEmitter.emitLine(payload: FakeWorkerEventEmitter.residentReadyEventPayload(modelId: "m1"))
            + FakeWorkerEventEmitter.emitLine(payload: FakeWorkerEventEmitter.residentRuntimePolicyPayload(modelId: "m1"))
            + "while true; do\n"
            + "  if [ -f \"\(controlDirectoryPath)/complete_request\" ]; then\n"
            + "    completion_request_id=$(cat \"\(controlDirectoryPath)/complete_request\")\n"
            + "    rm -f \"\(controlDirectoryPath)/complete_request\"\n"
            + "    emit_frame '{\"kind\":\"completed\",\"request_id\":'\"$completion_request_id\"',"
            + "\"prompt_token_count\":1,\"generated_token_count\":1,\"reasoning_token_count\":0,"
            + "\"cached_token_count\":0,\"persistent_prompt_cache_diagnostics\":null,\"reason\":\"end_of_sequence\"}'\n"
            + "  fi\n"
            + "  if [ -f \"\(controlDirectoryPath)/apply_memory_raise\" ]; then\n"
            + "    raised_ceiling_bytes=$(cat \"\(controlDirectoryPath)/apply_memory_raise\")\n"
            + "    rm -f \"\(controlDirectoryPath)/apply_memory_raise\"\n"
            + "    emit_frame '{\"kind\":\"mlx_memory_limit_changed\","
            + "\"effective_mlx_memory_ceiling_bytes\":'\"$raised_ceiling_bytes\"',"
            + "\"minimum_mlx_memory_ceiling_bytes\":1,\"expert_memory_mode\":\"resident\","
            + "\"mlx_memory_snapshot\":null,\"expert_residency\":null}'\n"
            + "  fi\n"
            + "  sleep 0.02\n"
            + "done\n";
    }
}

/// One in-flight generation executed on its own thread, with its terminal
/// outcome captured for the journey thread to inspect after a bounded join.
final class GenerationJourneyOutcome: @unchecked Sendable {

    var workerThread: Thread;
    private let outcomeLock: NSLock;
    private var observedOutcomeValue: Result<Array<ChatGenerationStreamEvent>, Error>?;

    init(workerThread: Thread) {
        self.workerThread = workerThread;
        self.outcomeLock = NSLock();
        self.observedOutcomeValue = nil;
    }

    var observedOutcome: Result<Array<ChatGenerationStreamEvent>, Error>? {
        self.outcomeLock.lock();
        defer { self.outcomeLock.unlock(); }
        return self.observedOutcomeValue;
    }

    var isSuccessful: Bool {
        guard case .success? = self.observedOutcome else {
            return false;
        }
        return true;
    }

    var thrownErrorAsGenerationStart: GenerationStartError? {
        guard case .failure(let capturedError as GenerationStartError)? = self.observedOutcome else {
            return nil;
        }
        return capturedError;
    }

    var streamEvents: Array<ChatGenerationStreamEvent>? {
        guard case .success(let capturedEvents)? = self.observedOutcome else {
            return nil;
        }
        return capturedEvents;
    }

    func record(streamEvents: Array<ChatGenerationStreamEvent>) -> Void {
        self.outcomeLock.lock();
        self.observedOutcomeValue = .success(streamEvents);
        self.outcomeLock.unlock();
    }

    func record(error: Error) -> Void {
        self.outcomeLock.lock();
        self.observedOutcomeValue = .failure(error);
        self.outcomeLock.unlock();
    }
}

/// One launched supervisor plus its marker-file-driven fake worker and
/// temporary control directory.
final class GenerationQueueTestsHarness {

    let supervisor: WorkerSupervisor;
    private let controlDirectoryPath: String;

    private init(supervisor: WorkerSupervisor, controlDirectoryPath: String) {
        self.supervisor = supervisor;
        self.controlDirectoryPath = controlDirectoryPath;
    }

    static func launch() throws -> GenerationQueueTestsHarness {
        let controlDirectoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("astronomical-queue-journey-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(at: controlDirectoryUrl, withIntermediateDirectories: true);
        let controlDirectoryPath: String = controlDirectoryUrl.path;
        let supervisor: WorkerSupervisor = try WorkerSupervisor.launch(
            workerExecutablePath: "/bin/bash",
            workerArguments: ["-c", GenerationQueueTests.controlGatedWorkerScript(
                controlDirectoryPath: controlDirectoryPath)],
            workerStartupConfiguration: GenerationQueueTests.startupConfiguration(),
            modelPolicyCatalog: ["m1": GenerationQueueTests.modelPolicy(modelId: "m1")],
            modelLoadTimeout: 10);
        return GenerationQueueTestsHarness(
            supervisor: supervisor,
            controlDirectoryPath: controlDirectoryPath);
    }

    func dispose() -> Void {
        _ = try? self.supervisor.shutdown();
        try? FileManager.default.removeItem(atPath: self.controlDirectoryPath);
    }

    func startGenerationThread(requestId: UInt64) -> GenerationJourneyOutcome {
        let outcome: GenerationJourneyOutcome = GenerationJourneyOutcome(workerThread: Thread());
        let supervisor: WorkerSupervisor = self.supervisor;
        let generationThread: Thread = Thread {
            do {
                outcome.record(streamEvents: try supervisor.startChatGeneration(
                    GenerationQueueTests.chatGenerationCommand(requestId: requestId, modelId: "m1")));
            } catch {
                outcome.record(error: error);
            }
        };
        generationThread.name = "queued-generation-\(requestId)";
        outcome.workerThread = generationThread;
        generationThread.start();
        return outcome;
    }

    func pokeCompletion(requestId: UInt64) throws -> Void {
        try String(requestId).write(
            toFile: self.controlDirectoryPath + "/complete_request",
            atomically: true,
            encoding: .utf8);
    }

    func pokeMemoryRaise(_ effectiveMlxMemoryCeilingBytes: UInt64) throws -> Void {
        try String(effectiveMlxMemoryCeilingBytes).write(
            toFile: self.controlDirectoryPath + "/apply_memory_raise",
            atomically: true,
            encoding: .utf8);
    }
}
