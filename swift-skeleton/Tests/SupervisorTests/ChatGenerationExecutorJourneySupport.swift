import Foundation

import Testing

import IpcProtocol

import AstronomicalConfig

@testable import Supervisor;


/**
 * The scripted chat fixture model identities, mirroring the constants the
 * SupervisorIdleWorker executable routes on; the test target cannot import
 * an executable module, so the shared identities live here as the single
 * journey-side catalog.
 */
enum ChatExecutorFixture {

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
    static let DELAYED_FRAGMENT_CHAT_MODEL_ID: String =
        "astronomical/delayed-fragment-chat-fixture"

    static func scriptedModelIds() -> Array<String> {
        return [
            ChatExecutorFixture.ACCEPTED_CHAT_MODEL_ID,
            ChatExecutorFixture.PREFILL_PROGRESS_MODEL_ID,
            ChatExecutorFixture.ACTIVITY_TRANSITION_MODEL_ID,
            ChatExecutorFixture.MALFORMED_OUTPUT_MODEL_ID,
            ChatExecutorFixture.DUPLICATE_GENERATION_PREPARATION_MODEL_ID,
            ChatExecutorFixture.OUT_OF_ORDER_MODEL_ID,
            ChatExecutorFixture.EMPTY_OUTPUT_BATCH_MODEL_ID,
            ChatExecutorFixture.INVALID_OUTPUT_BATCH_MODEL_ID,
            ChatExecutorFixture.OVER_BUDGET_TOOL_COMPLETION_MODEL_ID,
            ChatExecutorFixture.UNSOLICITED_CANCELLATION_MODEL_ID,
            ChatExecutorFixture.BACKPRESSURE_MODEL_ID,
            ChatExecutorFixture.EXIT_AFTER_CHAT_ADMISSION_MODEL_ID,
            ChatExecutorFixture.DELAYED_FRAGMENT_CHAT_MODEL_ID,
        ]
    }
}

/**
 * One launched scripted worker plus the waits and command shapes the
 * executor journeys share; the harness directory vanishes with `dispose()`.
 */
final class ChatExecutorJourney {

    static let FOLLOWUP_MODEL_ID: String = "astronomical/test-worker"
    static let RETIRED_SMALL_FRAME_BYTES: Int = 64 * 1_024
    static let MAXIMUM_IPC_FRAME_BYTES: Int = 32 * 1_024 * 1_024
    let supervisor: WorkerSupervisor
    private let journeyCleanup: () -> Void
    private var didRunCleanup: Bool

    private init(supervisor: WorkerSupervisor, journeyCleanup: @escaping () -> Void) {
        self.supervisor = supervisor
        self.journeyCleanup = journeyCleanup
        self.didRunCleanup = false
    }

    static func launch() throws -> ChatExecutorJourney {
        let fixtureModelIds: [String] = ChatExecutorFixture.scriptedModelIds()
            + [ChatExecutorJourney.FOLLOWUP_MODEL_ID]
        var modelPolicyCatalog: Dictionary<String, RuntimeModelPolicy> = [:]
        for fixtureModelId: String in fixtureModelIds {
            modelPolicyCatalog[fixtureModelId] = IdleWorkerJourneySupport.runtimeModelPolicy(
                fixtureModelId,
                modelDirectory: "/models/\(fixtureModelId)",
                maximumOutputTokens: 128)
        }
        let harness: IdleWorkerJourneySupport.IdleWorkerHarness =
            try IdleWorkerJourneySupport.launchConfiguredIdleWorker(
                modelPolicyCatalog: modelPolicyCatalog,
                modelLoadTimeout: 10,
                startupConfigurationBuilder: { (_: String) -> WorkerStartupConfiguration? in
                    return nil
                },
                workerArguments: [],
                cancellationAcknowledgementTimeout: WorkerSupervisor.defaultCancellationAcknowledgementTimeoutSeconds)
        return ChatExecutorJourney(
            supervisor: harness.supervisor,
            journeyCleanup: { () -> Void in
                harness.dispose()
            })
    }

    func dispose() -> Void {
        if self.didRunCleanup {
            return
        }
        self.didRunCleanup = true
        self.journeyCleanup()
    }

    func waitForHealth(
        _ expectedStatus: WorkerHealthStatus,
        journeyLabel: String
    ) -> Void {
        self.waitUntil(
            { () -> Bool in return self.supervisor.workerHealthSnapshot().status == expectedStatus },
            journeyLabel: journeyLabel,
            timeoutSeconds: 5)
    }

    func waitForActivity(
        _ expectedWorkerActivity: WorkerActivity,
        journeyLabel: String
    ) -> Void {
        self.waitUntil(
            { () -> Bool in return self.supervisor.workerHealthSnapshot().activity == expectedWorkerActivity },
            journeyLabel: journeyLabel,
            timeoutSeconds: 2)
    }

    func waitUntil(
        _ isSatisfied: () -> Bool,
        journeyLabel: String,
        timeoutSeconds: TimeInterval
    ) -> Void {
        let conditionDeadline: Date = Date().addingTimeInterval(timeoutSeconds)
        while (Date() < conditionDeadline) {
            if isSatisfied() {
                return
            }
            Thread.sleep(forTimeInterval: 0.02)
        }
        Issue.record(Comment(stringLiteral:
            "\(journeyLabel): condition not met within \(timeoutSeconds)s"))
    }

    /**
     * Starts one streamed generation and waits until its stream ends —
     * the terminal completion, failure, or the stream error a contained
     * breach delivers — collecting every event the sink produced.
     */
    func collectUntilStreamEnd(modelId: String) throws -> [ChatGenerationStreamEvent] {
        let eventCollector: ExecutorStreamedCollector = ExecutorStreamedCollector()
        let streamHandle: ChatGenerationStreamHandle = try self.supervisor.startChatGenerationStream(
            self.chatCommand(modelId: modelId),
            onEvent: { (streamEvent: ChatGenerationStreamEvent) -> Void in
                eventCollector.append(streamEvent)
            })
        self.waitUntil(
            { () -> Bool in return eventCollector.hasStreamEnded() },
            journeyLabel: "stream end for \(modelId)",
            timeoutSeconds: 5)
        return withExtendedLifetime(streamHandle) { () -> [ChatGenerationStreamEvent] in
            return eventCollector.collectedEvents()
        }
    }

    func chatCommand(
        modelId: String,
        requestId: UInt64 = 1,
        maximumOutputTokens: UInt16 = 16,
        userContent: String = "hello"
    ) -> ChatGenerationCommand {
        return ChatGenerationCommand(
            requestId: RequestId(rawRequestId: requestId),
            model: modelId,
            messages: [.user(content: userContent, images: [])],
            tools: [],
            toolChoice: .none,
            settings: ChatGenerationSettings(
                maxOutputTokens: maximumOutputTokens,
                temperatureThousandths: nil,
                topPThousandths: nil,
                seed: nil,
                thinkingBudget: nil),
            qwenThinkingChannelSeed: nil,
            structuredGeneration: nil)
    }
}

/// Thread-safe collector for one streamed generation's delivered events;
/// the sink runs on the generation thread, the journey reads after the
/// stream ended.
final class ExecutorStreamedCollector: @unchecked Sendable {

    private let collectorLock: NSLock
    private var collectedEventsStorage: [ChatGenerationStreamEvent]

    init() {
        self.collectorLock = NSLock()
        self.collectedEventsStorage = []
    }

    func append(_ streamEvent: ChatGenerationStreamEvent) -> Void {
        self.collectorLock.lock()
        self.collectedEventsStorage.append(streamEvent)
        self.collectorLock.unlock()
    }

    func hasTextFragment() -> Bool {
        self.collectorLock.lock()
        defer { self.collectorLock.unlock() }
        return self.collectedEventsStorage.contains { (streamEvent: ChatGenerationStreamEvent) -> Bool in
            if case .textFragment = streamEvent {
                return true
            }
            return false
        }
    }

    func hasStreamEnded() -> Bool {
        self.collectorLock.lock()
        defer { self.collectorLock.unlock() }
        return self.collectedEventsStorage.contains { (streamEvent: ChatGenerationStreamEvent) -> Bool in
            switch (streamEvent) {
            case .completed, .failed, .streamError:
                return true
            case .reasoningFragment, .textFragment, .toolCall, .prefillProgress:
                return false
            }
        }
    }

    func collectedEvents() -> [ChatGenerationStreamEvent] {
        self.collectorLock.lock()
        defer { self.collectorLock.unlock() }
        return self.collectedEventsStorage
    }
}

/// One shutdown executed on its own thread with its outcome captured, so
/// the journey can bound how long shutdown may take behind an active
/// generation.
final class ThreadedShutdownOutcome: @unchecked Sendable {

    private let outcomeLock: NSLock
    private var observedOutcome: Result<Void, Error>?

    init() {
        self.outcomeLock = NSLock()
        self.observedOutcome = nil
    }

    func recordSuccess() -> Void {
        self.outcomeLock.lock()
        self.observedOutcome = .success(())
        self.outcomeLock.unlock()
    }

    func recordFailure(_ shutdownError: Error) -> Void {
        self.outcomeLock.lock()
        self.observedOutcome = .failure(shutdownError)
        self.outcomeLock.unlock()
    }

    func awaitOutcome(deadlineSeconds: Double) -> Result<Void, Error>? {
        let joinDeadline: Date = Date().addingTimeInterval(deadlineSeconds)
        while (Date() < joinDeadline) {
            self.outcomeLock.lock()
            let observedOutcome: Result<Void, Error>? = self.observedOutcome
            self.outcomeLock.unlock()
            if let observedOutcome = observedOutcome {
                return observedOutcome
            }
            Thread.sleep(forTimeInterval: 0.01)
        }
        return nil
    }
}
