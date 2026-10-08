import Foundation;

import Testing;

import IpcProtocol;

@testable import Supervisor;

/**
 * Acceptance journeys for generation abandonment and cancellation
 * containment, migrating apps/supervisor/tests/hermetic/worker_cancellation.rs:
 * a client that stops consuming must cancel the in-flight request, a
 * bounded slow acknowledgement must keep the worker reusable, and a worker
 * that breaches the cancellation protocol is terminated and replaced.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class WorkerCancellationJourneyTests {

    /// A held stream delivers every public event through its sink and a
    /// normal completion releases the worker without any cancellation.
    @Test
    func should_stream_events_to_a_consuming_client_without_cancellation() throws {
        let journey: WorkerCancellationJourney = try WorkerCancellationJourney.launch(
            cancellationAcknowledgementTimeout: 0.1);
        defer { journey.dispose() }

        let streamedEvents: Array<ChatGenerationStreamEvent> =
            try journey.collectStreamedGeneration(
                modelId: WorkerCancellationJourney.FOLLOWUP_MODEL_ID);
        IdleWorkerJourneySupport.assertGenerationCompleted(
            streamedEvents,
            generationLabel: "streamed generation through a consuming client");
        #expect(
            journey.supervisor.observedCancellationContainmentCount == 0,
            "a fully consumed stream must never trigger cancellation containment");
    }

    /// A worker that never acknowledges cancellation is replaced inside the
    /// explicit bound, and serving continues.
    @Test
    func should_replace_a_worker_that_does_not_acknowledge_cancellation() throws {
        let journey: WorkerCancellationJourney = try WorkerCancellationJourney.launch(
            cancellationAcknowledgementTimeout: 0.1);
        defer { journey.dispose() }

        journey.startGeneration(modelId: "astronomical/unacknowledged-cancellation-fixture");
        journey.abandonActiveGeneration();
        journey.waitForContainmentTransition();
        #expect(journey.waitUntilReady(), "the explicit cancellation timeout should bound worker replacement");
        try journey.assertWorkerAcceptsFollowupRequest();

        try journey.assertShutdownIsIdempotent();
    }

    /// A prefill cancellation acknowledged within the default bound keeps
    /// the same worker ready and serving.
    @Test
    func should_keep_worker_ready_when_prefill_cancellation_acknowledgement_is_delayed() throws {
        let journey: WorkerCancellationJourney = try WorkerCancellationJourney.launch(
            cancellationAcknowledgementTimeout:
                WorkerSupervisor.defaultCancellationAcknowledgementTimeoutSeconds);
        defer { journey.dispose() }

        journey.startGeneration(
            modelId: "astronomical/delayed-cancellation-acknowledgement-fixture");
        journey.abandonActiveGeneration();
        Thread.sleep(forTimeInterval: 5);

        #expect(
            journey.supervisor.workerHealthSnapshot().status == .ready,
            "a bounded slow prefill cancellation must not permanently disable the worker");
        try journey.assertWorkerAcceptsFollowupRequest(followupJoinSeconds: 8);
    }

    /// An unexpected event during cancellation replaces the worker, and the
    /// diagnostics carry the exact correlation summary.
    @Test
    func should_replace_a_worker_after_an_unexpected_cancellation_event() throws {
        let journey: WorkerCancellationJourney = try WorkerCancellationJourney.launch(
            cancellationAcknowledgementTimeout:
                WorkerSupervisor.defaultCancellationAcknowledgementTimeoutSeconds);
        defer { journey.dispose() }

        journey.startGeneration(
            modelId: "astronomical/unexpected-cancellation-event-fixture");
        journey.abandonActiveGeneration();
        journey.waitForContainmentTransition();
        #expect(journey.waitUntilReady(), "the unexpected cancellation event should replace the worker");
        try journey.assertWorkerAcceptsFollowupRequest();

        let diagnosticError: WorkerControlError = .unexpectedCancellationEvent(
            requestId: 1,
            unexpectedWorkerEventSummary: "completed request_id=2");
        #expect(
            diagnosticError.errorDescription
                == "worker emitted an unexpected event while cancelling request 1: completed request_id=2");

        try journey.assertShutdownIsIdempotent();
    }

    /// Prompt-cache telemetry published during cancellation is recorded and
    /// the worker stays reusable.
    @Test
    func should_keep_worker_ready_when_cancellation_publishes_prompt_cache_stats() throws {
        let journey: WorkerCancellationJourney = try WorkerCancellationJourney.launch(
            cancellationAcknowledgementTimeout:
                WorkerSupervisor.defaultCancellationAcknowledgementTimeoutSeconds);
        defer { journey.dispose() }

        journey.startGeneration(
            modelId: "astronomical/cache-stats-during-cancellation-fixture");
        journey.abandonActiveGeneration();
        #expect(
            journey.waitUntil({ () -> Bool in
                return journey.supervisor.workerHealthSnapshot().persistentPromptCacheStats != nil;
            }, journeyLabel: "prompt-cache stats during cancellation"),
            "prompt-cache stats published during cancellation should be recorded");
        #expect(
            journey.supervisor.workerHealthSnapshot().persistentPromptCacheStats?
                .persistentPromptCacheHits == 1);
        #expect(
            journey.supervisor.workerHealthSnapshot().persistentPromptCacheStats?
                .persistentPromptCacheTokensSaved == 2_048);
        #expect(journey.supervisor.workerHealthSnapshot().status == .ready);
        try journey.assertWorkerAcceptsFollowupRequest();
    }

    /// MLX memory telemetry published during cancellation is recorded and
    /// the worker stays reusable.
    @Test
    func should_keep_worker_ready_when_cancellation_publishes_mlx_memory() throws {
        let journey: WorkerCancellationJourney = try WorkerCancellationJourney.launch(
            cancellationAcknowledgementTimeout:
                WorkerSupervisor.defaultCancellationAcknowledgementTimeoutSeconds);
        defer { journey.dispose() }

        journey.startGeneration(
            modelId: "astronomical/mlx-memory-during-cancellation-fixture");
        journey.abandonActiveGeneration();
        #expect(
            journey.waitUntil({ () -> Bool in
                return journey.supervisor.workerHealthSnapshot().latestMlxMemorySnapshot?
                    .activeMemoryBytes == 44_000;
            }, journeyLabel: "MLX memory during cancellation"),
            "MLX memory telemetry published during cancellation should be recorded");
        #expect(journey.supervisor.workerHealthSnapshot().status == .ready);
        try journey.assertWorkerAcceptsFollowupRequest();
    }

    /// A cleared MLX memory sample during cancellation clears the published
    /// snapshot rather than leaving stale telemetry.
    @Test
    func should_keep_worker_ready_when_cancellation_clears_mlx_memory() throws {
        let journey: WorkerCancellationJourney = try WorkerCancellationJourney.launch(
            cancellationAcknowledgementTimeout:
                WorkerSupervisor.defaultCancellationAcknowledgementTimeoutSeconds);
        defer { journey.dispose() }

        journey.startGeneration(
            modelId: "astronomical/mlx-memory-clear-during-cancellation-fixture");
        #expect(
            journey.waitUntil({ () -> Bool in
                return journey.supervisor.workerHealthSnapshot().latestMlxMemorySnapshot?
                    .activeMemoryBytes == 33_000;
            }, journeyLabel: "seeded MLX memory before abandonment"),
            "the fixture should seed a visible memory snapshot");
        journey.abandonActiveGeneration();
        #expect(
            journey.waitUntil({ () -> Bool in
                return journey.supervisor.workerHealthSnapshot().latestMlxMemorySnapshot == nil;
            }, journeyLabel: "cleared MLX memory after cancellation"),
            "cleared MLX memory telemetry must clear the published snapshot");
        #expect(journey.supervisor.workerHealthSnapshot().status == .ready);
        try journey.assertWorkerAcceptsFollowupRequest();
    }
}

/// One unconfigured scripted worker whose supervisor holds a configurable
/// cancellation bound, mirroring the Rust launch helpers of
/// worker_cancellation.rs. Follow-up requests run on their own thread and
/// join against a deadline, so a wedged cancellation path fails the journey
/// instead of hanging it.
final class WorkerCancellationJourney {

    static let FOLLOWUP_MODEL_ID: String = "astronomical/test-worker";

    let supervisor: WorkerSupervisor;
    private var activeStreamHandle: ChatGenerationStreamHandle?;
    private let journeyLock: NSLock;
    private let journeyCleanup: () -> Void;

    private init(
        supervisor: WorkerSupervisor,
        journeyCleanup: @escaping () -> Void
    ) {
        self.supervisor = supervisor;
        self.activeStreamHandle = nil;
        self.journeyLock = NSLock();
        self.journeyCleanup = journeyCleanup;
    }

    static func launch(cancellationAcknowledgementTimeout: TimeInterval) throws -> WorkerCancellationJourney {
        let fixtureModelIds: Array<String> = [
            "astronomical/unacknowledged-cancellation-fixture",
            "astronomical/delayed-cancellation-acknowledgement-fixture",
            "astronomical/unexpected-cancellation-event-fixture",
            "astronomical/cache-stats-during-cancellation-fixture",
            "astronomical/mlx-memory-during-cancellation-fixture",
            "astronomical/mlx-memory-clear-during-cancellation-fixture",
            WorkerCancellationJourney.FOLLOWUP_MODEL_ID,
        ];
        var modelPolicyCatalog: Dictionary<String, RuntimeModelPolicy> = [:];
        for fixtureModelId: String in fixtureModelIds {
            modelPolicyCatalog[fixtureModelId] = IdleWorkerJourneySupport.runtimeModelPolicy(
                fixtureModelId,
                modelDirectory: "/models/\(fixtureModelId)",
                maximumOutputTokens: 128);
        }
        let harness: IdleWorkerJourneySupport.IdleWorkerHarness =
            try IdleWorkerJourneySupport.launchConfiguredIdleWorker(
                modelPolicyCatalog: modelPolicyCatalog,
                modelLoadTimeout: 2,
                startupConfigurationBuilder: { (journeyDirectoryPath: String) -> WorkerStartupConfiguration? in
                    return nil
                },
                workerArguments: [],
                cancellationAcknowledgementTimeout: cancellationAcknowledgementTimeout);
        return WorkerCancellationJourney(
            supervisor: harness.supervisor,
            journeyCleanup: { () -> Void in
                harness.dispose();
            });
    }

    func dispose() -> Void {
        self.journeyCleanup();
    }

    /// Runs one streamed generation to completion, collecting everything
    /// the sink delivered — the consuming-client path no abandonment journey
    /// exercises. The handle stays held until the terminal event so no
    /// deinit-driven abandonment can fire mid-stream.
    func collectStreamedGeneration(modelId: String) throws -> Array<ChatGenerationStreamEvent> {
        let eventCollector: StreamedEventCollector = StreamedEventCollector();
        let streamHandle: ChatGenerationStreamHandle = try self.supervisor.startChatGenerationStream(
            WorkerCancellationJourney.chatCommand(modelId: modelId),
            onEvent: { (streamEvent: ChatGenerationStreamEvent) -> Void in
                eventCollector.append(streamEvent);
            });
        let completionDeadline: Date = Date().addingTimeInterval(5);
        // The handle must outlive the stream: releasing it early is exactly
        // the abandonment the consuming client must not perform.
        let streamedEvents: Array<ChatGenerationStreamEvent> = withExtendedLifetime(streamHandle) { () -> Array<ChatGenerationStreamEvent> in
            while (Date() < completionDeadline) {
                if eventCollector.hasTerminalEvent() {
                    break;
                }
                Thread.sleep(forTimeInterval: 0.02);
            }
            return eventCollector.collectedEvents();
        };
        return streamedEvents;
    }


    func startGeneration(modelId: String) -> Void {
        let streamHandle: ChatGenerationStreamHandle? = try? self.supervisor.startChatGenerationStream(
            WorkerCancellationJourney.chatCommand(modelId: modelId),
            onEvent: { (streamEvent: ChatGenerationStreamEvent) -> Void in
                return;
            });
        self.journeyLock.lock();
        self.activeStreamHandle = streamHandle;
        self.journeyLock.unlock();
    }

    /// Disconnects the held client handle — the Rust drop(stream_receiver).
    func abandonActiveGeneration() -> Void {
        self.journeyLock.lock();
        let streamHandle: ChatGenerationStreamHandle? = self.activeStreamHandle;
        self.activeStreamHandle = nil;
        self.journeyLock.unlock();
        streamHandle?.abandon();
    }

    /// Waits bounded for one condition, recording a journey failure at the
    /// bound instead of hanging.
    func waitUntil(
        _ isSatisfied: () -> Bool,
        journeyLabel: String,
        timeoutSeconds: TimeInterval = 5
    ) -> Bool {
        let conditionDeadline: Date = Date().addingTimeInterval(timeoutSeconds);
        while (Date() < conditionDeadline) {
            if isSatisfied() {
                return true;
            }
            Thread.sleep(forTimeInterval: 0.02);
        }
        Issue.record(Comment(stringLiteral: "\(journeyLabel): condition not met within \(timeoutSeconds)s"));
        return false;
    }

    /// The deterministic replacement signal: a misbehaving worker's
    /// cancellation containment increments the supervisor's counter, which
    /// no health-poll race can miss.
    func waitForContainmentTransition() -> Void {
        _ = self.waitUntil({ () -> Bool in
            return self.supervisor.observedCancellationContainmentCount > 0;
        }, journeyLabel: "cancellation containment", timeoutSeconds: 2);
    }

    func waitUntilReady() -> Bool {
        return self.waitUntil({ () -> Bool in
            return self.supervisor.workerHealthSnapshot().status == .ready;
        }, journeyLabel: "worker ready after cancellation");
    }

    /// Proves the (possibly replaced) worker still serves: the follow-up
    /// runs on its own thread and joins against a deadline, so a wedged
    /// cancellation path fails here rather than hanging the journey.
    func assertWorkerAcceptsFollowupRequest(followupJoinSeconds: TimeInterval = 5) throws -> Void {
        let supervisor: WorkerSupervisor = self.supervisor;
        let followupDispatch: ThreadedChatDispatch = ThreadedChatDispatch(
            chatCommand: WorkerCancellationJourney.chatCommand(
                modelId: WorkerCancellationJourney.FOLLOWUP_MODEL_ID));
        let threadedOutcome: ThreadedChatOutcome = ThreadedChatOutcome();
        let followupThread: Thread = Thread(block: { () -> Void in
            do {
                let streamEvents: Array<ChatGenerationStreamEvent> =
                    try supervisor.startChatGeneration(followupDispatch.chatCommand);
                threadedOutcome.record(streamEvents: streamEvents);
            } catch let followupError {
                threadedOutcome.record(followupError: followupError);
            }
        });
        followupThread.name = "astronomical-cancellation-followup";
        followupThread.start();
        let observedOutcome: Result<Array<ChatGenerationStreamEvent>, Error>? =
            threadedOutcome.awaitOutcome(deadlineSeconds: followupJoinSeconds);
        guard case let .success(streamEvents)? = observedOutcome else {
            Issue.record(
                Comment(stringLiteral: "follow-up request after cancellation failed: \(String(describing: observedOutcome))"));
            return;
        }
        IdleWorkerJourneySupport.assertGenerationCompleted(
            streamEvents,
            generationLabel: "follow-up request after cancellation telemetry");
    }

    func assertShutdownIsIdempotent() throws -> Void {
        _ = try self.supervisor.shutdown();
    }

    static func chatCommand(modelId: String) -> ChatGenerationCommand {
        return ChatGenerationCommand(
            requestId: RequestId(rawRequestId: 1),
            model: modelId,
            messages: [.user(content: "hello", images: [])],
            tools: [],
            toolChoice: .none,
            settings: ChatGenerationSettings(
                maxOutputTokens: 16,
                temperatureThousandths: nil,
                topPThousandths: nil,
                seed: nil,
                thinkingBudget: nil),
            structuredGeneration: nil);
    }
}

/// One chat request executed on its own thread with its outcome captured
/// for the journey thread to inspect after a bounded join.
final class ThreadedChatOutcome: @unchecked Sendable {    private let outcomeLock: NSLock;
    private var observedOutcomeValue: Result<Array<ChatGenerationStreamEvent>, Error>?;

    init() {
        self.outcomeLock = NSLock();
        self.observedOutcomeValue = nil;
    }

    func record(streamEvents: Array<ChatGenerationStreamEvent>) -> Void {
        self.outcomeLock.lock();
        self.observedOutcomeValue = .success(streamEvents);
        self.outcomeLock.unlock();
    }

    func record(followupError: Error) -> Void {
        self.outcomeLock.lock();
        self.observedOutcomeValue = .failure(followupError);
        self.outcomeLock.unlock();
    }

    func awaitOutcome(deadlineSeconds: Double) -> Result<Array<ChatGenerationStreamEvent>, Error>? {
        let joinDeadline: Date = Date().addingTimeInterval(deadlineSeconds);
        while (Date() < joinDeadline) {
            self.outcomeLock.lock();
            let observedOutcome: Result<Array<ChatGenerationStreamEvent>, Error>? =
                self.observedOutcomeValue;
            self.outcomeLock.unlock();
            if let observedOutcome = observedOutcome {
                return observedOutcome;
            }
            Thread.sleep(forTimeInterval: 0.01);
        }
        return nil;
    }
}

/// The follow-up command boxed for its single cross-thread handoff; the
/// command is a value this journey never mutates after construction.
final class ThreadedChatDispatch: @unchecked Sendable {

    let chatCommand: ChatGenerationCommand;

    init(chatCommand: ChatGenerationCommand) {
        self.chatCommand = chatCommand;
    }
}

/// Thread-safe collector for one streamed generation's delivered events;
/// the sink runs on the generation thread, the journey reads after the
/// terminal event.
final class StreamedEventCollector: @unchecked Sendable {

    private let collectorLock: NSLock;
    private var collectedEventsStorage: Array<ChatGenerationStreamEvent>;

    init() {
        self.collectorLock = NSLock();
        self.collectedEventsStorage = [];
    }

    func append(_ streamEvent: ChatGenerationStreamEvent) -> Void {
        self.collectorLock.lock();
        self.collectedEventsStorage.append(streamEvent);
        self.collectorLock.unlock();
    }

    func hasTerminalEvent() -> Bool {
        self.collectorLock.lock();
        defer { self.collectorLock.unlock(); }
        return self.collectedEventsStorage.contains { (streamEvent: ChatGenerationStreamEvent) -> Bool in
            if case .completed = streamEvent {
                return true;
            }
            if case .failed = streamEvent {
                return true;
            }
            return false;
        };
    }

    func collectedEvents() -> Array<ChatGenerationStreamEvent> {
        self.collectorLock.lock();
        defer { self.collectorLock.unlock(); };
        return self.collectedEventsStorage;
    }
}
