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
    static let STDERR_PROBE_EXECUTABLE_NAME: String = "SupervisorStderrProbeWorker"
    static let REQUESTED_MODEL_ID: String = "astronomical/requested-model"
    static let INVALID_MODEL_ID: String = "astronomical/invalid-model"
    static let HANGING_MODEL_ID: String = "astronomical/hanging-model"

    enum IdleWorkerJourneyFailure: Error, CustomStringConvertible {
        case missingBuiltExecutable(name: String, path: String)
        case workerNeverBecameReady(lastStatus: String)

        var description: String {
            switch (self) {
            case let .missingBuiltExecutable(name, path):
                return "the built \(name) executable was not found at \(path)"
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
        let journeyDirectoryPath: String

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
        return try IdleWorkerJourneySupport.launchWorkerHarness(
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
            startupConfigurationBuilder: { (journeyDirectoryPath: String) -> WorkerStartupConfiguration? in
                return WorkerStartupConfiguration(
                    configurationGeneration: configurationGeneration,
                    globalPromptCacheRootDirectory: journeyDirectoryPath + "/prompt-cache",
                    globalPromptCacheMaximumSizeBytes: 50_000_000_000,
                    persistentPromptCacheEnabled: true,
                    configuredMaximumMlxMemoryBytes: nil,
                    performanceAttributionEnabled: false,
                    loggingDirectory: journeyDirectoryPath,
                    loggingLevel: .warn,
                    retainedLogFileCount: 7)
            })
    }

    /// Launches the fixture worker with no startup configuration at all —
    /// the lazy model-load world of worker_launch.rs, where the supervisor
    /// waits only for the worker's idle readiness.
    static func launchUnconfiguredIdleWorker(
        modelPolicyCatalog: Dictionary<String, RuntimeModelPolicy>,
        modelLoadTimeout: TimeInterval
    ) throws -> IdleWorkerHarness {
        return try IdleWorkerJourneySupport.launchWorkerHarness(
            modelPolicyCatalog: modelPolicyCatalog,
            modelLoadTimeout: modelLoadTimeout,
            startupConfigurationBuilder: { (journeyDirectoryPath: String) -> WorkerStartupConfiguration? in
                return nil
            })
    }

    /// Launches the fixture worker with full caller control over the
    /// supervisor inputs: the scenario argv the worker runs under, the
    /// startup configuration it must acknowledge (built against the journey
    /// directory the harness created), and the cancellation bound its
    /// abandonments are held to.
    static func launchConfiguredIdleWorker(
        modelPolicyCatalog: Dictionary<String, RuntimeModelPolicy>,
        modelLoadTimeout: TimeInterval,
        startupConfigurationBuilder: (String) -> WorkerStartupConfiguration?,
        workerArguments: Array<String>,
        cancellationAcknowledgementTimeout: TimeInterval
    ) throws -> IdleWorkerHarness {
        return try IdleWorkerJourneySupport.launchWorkerHarness(
            modelPolicyCatalog: modelPolicyCatalog,
            modelLoadTimeout: modelLoadTimeout,
            startupConfigurationBuilder: startupConfigurationBuilder,
            workerArguments: workerArguments,
            cancellationAcknowledgementTimeout: cancellationAcknowledgementTimeout)
    }

    private static func launchWorkerHarness(
        modelPolicyCatalog: Dictionary<String, RuntimeModelPolicy>,
        modelLoadTimeout: TimeInterval,
        startupConfigurationBuilder: (String) -> WorkerStartupConfiguration?,
        workerArguments: Array<String> = [],
        cancellationAcknowledgementTimeout: TimeInterval = WorkerSupervisor.defaultCancellationAcknowledgementTimeoutSeconds
    ) throws -> IdleWorkerHarness {
        let workerExecutablePath: String = try IdleWorkerJourneySupport.locateBuiltExecutable(
            executableName: IdleWorkerJourneySupport.WORKER_EXECUTABLE_NAME)
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
            workerArguments: workerArguments.isEmpty ? [controlDirectoryPath] : workerArguments,
            workerStartupConfiguration: startupConfigurationBuilder(journeyDirectoryPath),
            modelPolicyCatalog: modelPolicyCatalog,
            modelLoadTimeout: modelLoadTimeout,
            cancellationAcknowledgementTimeout: cancellationAcknowledgementTimeout,
            generationPerformanceLog: GenerationPerformanceLog.open(
                logDirectory: FilePath(string: journeyDirectoryPath)))
        let harness: IdleWorkerHarness = IdleWorkerHarness(
            supervisor: supervisor,
            controlDirectoryPath: controlDirectoryPath,
            journeyDirectoryPath: journeyDirectoryPath)
        try IdleWorkerJourneySupport.waitForReadyWorker(harness.supervisor)
        return harness
    }

    /// A fixture binary built by this package; SwiftPM builds every
    /// executable target before tests run, and gives tests no
    /// CARGO_BIN_EXE-style variable, so the executable is located from this
    /// source file's package root, checking both build layouts.
    static func locateBuiltExecutable(executableName: String) throws -> String {
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
                .appendingPathComponent(executableName)
            if (FileManager.default.isExecutableFile(atPath: workerExecutableUrl.path)) {
                return workerExecutableUrl.path
            }
        }
        throw IdleWorkerJourneyFailure.missingBuiltExecutable(
            name: executableName,
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

    /// Waits until the worker is ready with one exact effective
    /// configuration generation acknowledged, mirroring the Rust
    /// wait_for_effective_generation loop of worker_replacement.rs.
    static func waitForEffectiveGeneration(
        _ supervisor: WorkerSupervisor,
        expectedGeneration: String
    ) throws -> Void {
        let acknowledgementDeadline: Date = Date().addingTimeInterval(2)
        while (true) {
            let healthSnapshot: WorkerHealthSnapshot = supervisor.workerHealthSnapshot()
            let acknowledgedGeneration: String? =
                healthSnapshot.workerRuntimeFeatureConfiguration?.configurationGeneration
            if (healthSnapshot.status == .ready && acknowledgedGeneration == expectedGeneration) {
                return
            }
            if (Date() >= acknowledgementDeadline) {
                throw IdleWorkerJourneyFailure.workerNeverBecameReady(
                    lastStatus: healthSnapshot.status.readinessText()
                        + " (generation \(acknowledgedGeneration ?? "none"))")
            }
            Thread.sleep(forTimeInterval: 0.01)
        }
    }

    /// The five-thousand-word Romeo and Juliet corpus every real-text LLM
    /// journey uses, read from the shared repository fixture.
    static func romeoAndJulietFiveThousandWords() throws -> String {
        let repositoryRootUrl: URL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let corpusUrl: URL = repositoryRootUrl
            .appendingPathComponent("apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt")
        return try String(contentsOf: corpusUrl, encoding: String.Encoding.utf8)
    }

    /// Proves a rejected replacement candidate was reaped: the candidate
    /// fixture records its process identifier at initialization, and the
    /// probe must find no living process behind it afterwards.
    static func assertCandidateWasReaped(journeyDirectoryPath: String) -> Void {
        let pidFilePath: String = journeyDirectoryPath + "/replacement-candidate.pid"
        let processIdText: String =
            (try? String(contentsOfFile: pidFilePath, encoding: String.Encoding.utf8))
            ?? "<missing pid file>"
        let candidateProcessId: pid_t = pid_t(processIdText.trimmingCharacters(
            in: CharacterSet.whitespacesAndNewlines)) ?? -1
        #expect(candidateProcessId > 0, "the candidate fixture should record its process identifier")
        let probeOutcome: Int32 = kill(candidateProcessId, 0)
        #expect(probeOutcome == -1 && errno == ESRCH, "the rejected candidate must be reaped")
    }

    /// The FLUX runtime policy with a caller-chosen artifact revision, the
    /// exact policy shape the tagged-revision replacement journeys compare
    /// the candidate acknowledgement against.
    static func fluxRuntimeModelPolicy(artifactRevision: String) -> RuntimeModelPolicy {
        return RuntimeModelPolicy(
            modelDirectory: FilePath(string: "/fictional/models/FLUX.2-klein-4B"),
            generationDefaults: RuntimeModelGenerationDefaults.inert(),
            configuredMaximumContextTokens: nil,
            defaultMaximumContextTokens: 0,
            configuredChunkingFields: ConfiguredChunkingFields.inactive(),
            workerModelConfiguration: .flux2Klein(
                WorkerFlux2KleinModelConfiguration(
                    modelId: "FLUX.2-klein-4B",
                    modelFamily: .flux2Klein,
                    artifactRevision: artifactRevision)))
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
