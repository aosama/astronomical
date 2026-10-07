import Foundation

import Testing

import AstronomicalConfig
import IpcProtocol

@testable import Supervisor

/**
 * Shared harness for the worker journey suites that script the real
 * supervisor test worker, migrating the fixture helpers of
 * apps/supervisor/tests/hermetic/worker_model_swap.rs: the launch catalog of
 * scripted model identities, the readiness and completion waits, and the
 * chat and image request shapes every one of those journeys reuses.
 */
enum IdleWorkerJourneySupport {

    static let TELEMETRY_BEFORE_SWAP_MODEL_ID: String = "astronomical/telemetry-before-swap-model"
    static let DELAYED_COMPLETION_MODEL_ID: String = "astronomical/delayed-completion-model"
    static let GENERATION_EVENT_BEFORE_SWAP_MODEL_ID: String =
        "astronomical/generation-event-before-swap-model"
    static let IMAGE_MODEL_ID: String = "astronomical/image-generation-model"
    static let INVALID_IMAGE_MODEL_ID: String = "astronomical/invalid-image-generation-model"
    static let DELAYED_POLICY_ACK_MODEL_ID: String = "astronomical/delayed-policy-ack-model"
    static let DELAYED_IMAGE_POLICY_ACK_MODEL_ID: String = "astronomical/delayed-image-policy-ack-model"
    static let DISCONNECT_TRIPWIRE_MARKER_FILE_NAME: String = "dispatched_after_disconnect"
    static let WORKER_EXECUTABLE_NAME: String = "SupervisorIdleWorker"

    enum IdleWorkerJourneyFailure: Error, CustomStringConvertible {
        case missingWorkerExecutable(path: String)
        case workerNeverBecameReady(lastStatus: String)

        var description: String {
            switch (self) {
            case let .missingWorkerExecutable(path):
                return "the built supervisor test worker was not found at \(path)"
            case let .workerNeverBecameReady(lastStatus):
                return "the idle worker did not become ready; last status was \(lastStatus)"
            }
        }
    }

    /**
     * One launched scripted worker together with the temporary directories
     * its startup configuration and tripwire markers live in. `dispose()`
     * reaps the worker and removes the directories; journeys always call it.
     */
    final class IdleWorkerHarness {
        let supervisor: WorkerSupervisor
        let controlDirectoryPath: String
        private let journeyDirectoryPath: String

        init(supervisor: WorkerSupervisor, controlDirectoryPath: String, journeyDirectoryPath: String) {
            self.supervisor = supervisor
            self.controlDirectoryPath = controlDirectoryPath
            self.journeyDirectoryPath = journeyDirectoryPath
        }

        func dispose() -> Void {
            _ = try? self.supervisor.shutdown()
            try? FileManager.default.removeItem(atPath: self.journeyDirectoryPath)
        }

        var disconnectTripwireMarkerPath: String {
            return self.controlDirectoryPath + "/" + IdleWorkerJourneySupport.DISCONNECT_TRIPWIRE_MARKER_FILE_NAME
        }
    }

    static func launchIdleWorkerFixture(
        configurationGeneration: String = "test-configuration-generation"
    ) throws -> IdleWorkerHarness {
        let workerExecutablePath: String = try IdleWorkerJourneySupport.locateIdleWorkerExecutable()
        let journeyDirectoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("astronomical-idle-worker-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: journeyDirectoryUrl, withIntermediateDirectories: true)
        let journeyDirectoryPath: String = journeyDirectoryUrl.path
        let controlDirectoryPath: String = journeyDirectoryPath + "/control"
        try FileManager.default.createDirectory(
            atPath: controlDirectoryPath,
            withIntermediateDirectories: true)
        let supervisor: WorkerSupervisor = try WorkerSupervisor.launch(
            workerExecutablePath: workerExecutablePath,
            workerArguments: [controlDirectoryPath],
            workerStartupConfiguration: WorkerStartupConfiguration(
                configurationGeneration: configurationGeneration,
                globalPromptCacheRootDirectory: journeyDirectoryPath + "/prompt-cache",
                globalPromptCacheMaximumSizeBytes: 50_000_000_000,
                persistentPromptCacheEnabled: true,
                configuredMaximumMlxMemoryBytes: nil,
                performanceAttributionEnabled: false,
                loggingDirectory: journeyDirectoryPath,
                loggingLevel: .warn,
                retainedLogFileCount: 7),
            modelPolicyCatalog: [
                IdleWorkerJourneySupport.DELAYED_COMPLETION_MODEL_ID:
                    IdleWorkerJourneySupport.runtimeModelPolicy(
                        IdleWorkerJourneySupport.DELAYED_COMPLETION_MODEL_ID,
                        modelDirectory: "/models/delayed-completion-model",
                        maximumOutputTokens: 128),
                IdleWorkerJourneySupport.TELEMETRY_BEFORE_SWAP_MODEL_ID:
                    IdleWorkerJourneySupport.runtimeModelPolicy(
                        IdleWorkerJourneySupport.TELEMETRY_BEFORE_SWAP_MODEL_ID,
                        modelDirectory: "/models/telemetry-before-swap-model",
                        maximumOutputTokens: 64),
                IdleWorkerJourneySupport.GENERATION_EVENT_BEFORE_SWAP_MODEL_ID:
                    IdleWorkerJourneySupport.runtimeModelPolicy(
                        IdleWorkerJourneySupport.GENERATION_EVENT_BEFORE_SWAP_MODEL_ID,
                        modelDirectory: "/models/generation-event-before-swap-model",
                        maximumOutputTokens: 32),
                IdleWorkerJourneySupport.IMAGE_MODEL_ID:
                    IdleWorkerJourneySupport.imageRuntimeModelPolicy(
                        IdleWorkerJourneySupport.IMAGE_MODEL_ID,
                        modelDirectory: "/models/image-generation-model"),
                IdleWorkerJourneySupport.INVALID_IMAGE_MODEL_ID:
                    IdleWorkerJourneySupport.imageRuntimeModelPolicy(
                        IdleWorkerJourneySupport.INVALID_IMAGE_MODEL_ID,
                        modelDirectory: "/models/invalid-model"),
                IdleWorkerJourneySupport.DELAYED_POLICY_ACK_MODEL_ID:
                    IdleWorkerJourneySupport.runtimeModelPolicy(
                        IdleWorkerJourneySupport.DELAYED_POLICY_ACK_MODEL_ID,
                        modelDirectory: "/models/delayed-policy-ack-model",
                        maximumOutputTokens: 64),
                IdleWorkerJourneySupport.DELAYED_IMAGE_POLICY_ACK_MODEL_ID:
                    IdleWorkerJourneySupport.imageRuntimeModelPolicy(
                        IdleWorkerJourneySupport.DELAYED_IMAGE_POLICY_ACK_MODEL_ID,
                        modelDirectory: "/models/delayed-image-policy-ack-model"),
            ],
            modelLoadTimeout: 10,
            generationPerformanceLog: GenerationPerformanceLog.open(
                logDirectory: FilePath(string: journeyDirectoryPath)))
        let harness: IdleWorkerHarness = IdleWorkerHarness(
            supervisor: supervisor,
            controlDirectoryPath: controlDirectoryPath,
            journeyDirectoryPath: journeyDirectoryPath)
        try IdleWorkerJourneySupport.waitForReadyWorker(harness.supervisor)
        return harness
    }

    /// The fixture binary built by this package; SwiftPM builds every
    /// executable target before tests run, and gives tests no
    /// CARGO_BIN_EXE-style variable, so the executable is located from this
    /// source file's package root, checking both build layouts.
    static func locateIdleWorkerExecutable() throws -> String {
        let packageRootUrl: URL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let buildLayouts: Array<String> = [
            ".build/out/Products/Debug",
            ".build/debug"
        ]
        for buildLayout: String in buildLayouts {
            let workerExecutableUrl: URL = packageRootUrl
                .appendingPathComponent(buildLayout)
                .appendingPathComponent(IdleWorkerJourneySupport.WORKER_EXECUTABLE_NAME)
            if (FileManager.default.isExecutableFile(atPath: workerExecutableUrl.path)) {
                return workerExecutableUrl.path
            }
        }
        throw IdleWorkerJourneyFailure.missingWorkerExecutable(
            path: packageRootUrl.appendingPathComponent(".build/...").path)
    }

    static func waitForReadyWorker(_ supervisor: WorkerSupervisor) throws -> Void {
        let readinessDeadline: Date = Date().addingTimeInterval(10)
        while (true) {
            let workerHealthStatus: WorkerHealthStatus = supervisor.workerHealthSnapshot().status
            if (workerHealthStatus == .ready) {
                return
            }
            if (Date() >= readinessDeadline) {
                throw IdleWorkerJourneyFailure.workerNeverBecameReady(
                    lastStatus: workerHealthStatus.readinessText())
            }
            Thread.sleep(forTimeInterval: 0.01)
        }
    }

    /// Asserts the generation's terminal event is a clean end-of-sequence
    /// completion, the shared success shape of every scripted chat journey.
    static func assertGenerationCompleted(
        _ streamEvents: Array<ChatGenerationStreamEvent>,
        generationLabel: String
    ) -> Void {
        let terminalCompletion: ChatGenerationStreamEvent? = streamEvents.first(
            where: { (streamEvent: ChatGenerationStreamEvent) -> Bool in
                if case .completed = streamEvent {
                    return true
                }
                return false
            })
        guard case let .completed(_, _, _, _, completionReason)? = terminalCompletion else {
            Issue.record(
                Comment(stringLiteral: "\(generationLabel): expected EndOfSequence completion, received \(streamEvents)"))
            return
        }
        #expect(completionReason == ChatGenerationCompletionReason.endOfSequence)
    }

    static func chatCommand(modelId: String, requestId: UInt64) -> ChatGenerationCommand {
        return ChatGenerationCommand(
            requestId: RequestId(rawRequestId: requestId),
            model: modelId,
            messages: [.user(content: "Wherefore art thou Romeo?", images: [])],
            tools: [],
            toolChoice: .none,
            settings: ChatGenerationSettings(
                maxOutputTokens: 1,
                temperatureThousandths: nil,
                topPThousandths: nil,
                seed: nil,
                thinkingBudget: nil),
            qwenThinkingChannelSeed: nil,
            structuredGeneration: nil)
    }

    static func imageGenerationCommand(modelId: String, requestId: UInt64) -> ImageGenerationCommand {
        return ImageGenerationCommand(
            requestId: RequestId(rawRequestId: requestId),
            model: modelId,
            prompt: "A moonlit balcony scene from Romeo and Juliet",
            settings: IdleWorkerJourneySupport.imageGenerationSettings())
    }

    static func imageGenerationSettings() -> ImageGenerationSettings {
        return ImageGenerationSettings(
            widthPixels: 1_024,
            heightPixels: 1_024,
            steps: 4,
            guidanceThousandths: 1_000,
            seed: 7)
    }

    static func runtimeModelPolicy(
        _ modelId: String,
        modelDirectory: String,
        maximumOutputTokens: UInt32
    ) -> RuntimeModelPolicy {
        return RuntimeModelPolicy(
            modelDirectory: FilePath(string: modelDirectory),
            generationDefaults: RuntimeModelGenerationDefaults(
                maximumOutputTokens: UInt16(clamping: maximumOutputTokens),
                configuredMaximumOutputTokens: nil,
                temperatureThousandths: nil,
                topPThousandths: nil),
            configuredMaximumContextTokens: nil,
            defaultMaximumContextTokens: 2_048,
            configuredChunkingFields: ConfiguredChunkingFields.inactive(),
            workerModelConfiguration: .autoregressive(
                WorkerAutoregressiveModelConfiguration(
                    modelId: modelId,
                    maximumContextTokens: 2_048,
                    maximumOutputTokens: maximumOutputTokens,
                    chunking: IdleWorkerJourneySupport.chatChunkingConfiguration())))
    }

    static func imageRuntimeModelPolicy(
        _ modelId: String,
        modelDirectory: String
    ) -> RuntimeModelPolicy {
        return RuntimeModelPolicy(
            modelDirectory: FilePath(string: modelDirectory),
            generationDefaults: RuntimeModelGenerationDefaults.inert(),
            configuredMaximumContextTokens: nil,
            defaultMaximumContextTokens: 0,
            configuredChunkingFields: ConfiguredChunkingFields.inactive(),
            workerModelConfiguration: .flux2Klein(
                WorkerFlux2KleinModelConfiguration(
                    modelId: modelId,
                    modelFamily: .flux2Klein,
                    artifactRevision: "fixture-revision")))
    }

    private static func chatChunkingConfiguration() -> WorkerChunkingConfiguration {
        return WorkerChunkingConfiguration(
            fixedPromptProcessingChunkSizeTokens: 256,
            fixedSsdStreamingPromptProcessingChunkSizeTokens: 2_048,
            fullAttentionKeyValueGrowthTokens: 256,
            prefillGraphSubmissionLayerInterval: 0,
            experimentalSsdPagingPrefillGraphSubmissionLayerInterval: 1,
            experimentalSsdPagingGenerationGraphSubmissionLayerInterval: 3,
            promptCacheBlockTokens: nil,
            promptCacheCommonPrefixStrideBlocks: 4,
            experimentalDecodeStageAttributionEnabled: false,
            experimentalQuantizedKvCacheEnabled: false,
            experimentalFusedMoeDecodeEnabled: false)
    }
}

/// One in-flight image generation executed on its own thread, with its
/// terminal outcome captured for the journey thread to inspect after a
/// bounded join; the image mirror of `GenerationJourneyOutcome`.
final class ImageGenerationJourneyOutcome: @unchecked Sendable {

    var workerThread: Thread
    private let outcomeLock: NSLock
    private var observedOutcomeValue: Result<ImageGenerationOutput, Error>?

    init(workerThread: Thread) {
        self.workerThread = workerThread
        self.outcomeLock = NSLock()
        self.observedOutcomeValue = nil
    }

    var observedOutcome: Result<ImageGenerationOutput, Error>? {
        self.outcomeLock.lock()
        defer { self.outcomeLock.unlock() }
        return self.observedOutcomeValue
    }

    func record(output: ImageGenerationOutput) -> Void {
        self.outcomeLock.lock()
        self.observedOutcomeValue = .success(output)
        self.outcomeLock.unlock()
    }

    func record(error: Error) -> Void {
        self.outcomeLock.lock()
        self.observedOutcomeValue = .failure(error)
        self.outcomeLock.unlock()
    }

    /// Joins the worker thread, failing the journey when the outcome never
    /// arrived inside the bound.
    func awaitOutcome(deadlineSeconds: Double, journeyLabel: String) -> Result<ImageGenerationOutput, Error>? {
        let joinDeadline: Date = Date().addingTimeInterval(deadlineSeconds)
        while (Date() < joinDeadline) {
            if let observedOutcome: Result<ImageGenerationOutput, Error> = self.observedOutcome {
                return observedOutcome
            }
            Thread.sleep(forTimeInterval: 0.01)
        }
        Issue.record(Comment(stringLiteral: "\(journeyLabel): the image generation never finished"))
        return nil
    }
}
