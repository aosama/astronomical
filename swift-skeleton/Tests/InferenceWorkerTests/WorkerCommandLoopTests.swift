import Foundation;

import Testing;

import IpcProtocol;
import ModelServingTestSupport;
import JourneyCategories;

@testable import InferenceWorker;

/**
 * Hermetic journeys for the model-less worker command loop: bootstrap over
 * real pipes, the idle lifecycle emission, and every idle command answer the
 * supervisor can receive before an engine lands. CPU only — no model load.
 *
 * The suite runs serialized: every journey owns real pipe descriptors whose
 * lifecycle spans the test, and a parallel neighbor could otherwise reopen
 * and then lose a reused descriptor number mid-journey.
 */
@Suite(.serialized, .tags(.hermeticMlxJourney))
final class WorkerCommandLoopTests {

    /**
     * A closed harness pipe must surface as a write error inside the loop
     * under test, not kill the test runner.
     */
    init() {
        signal(SIGPIPE, SIG_IGN);
    }

    @Test
    func should_reject_a_non_initialize_worker_first_command() throws {
        let loopRun: WorkerLoopRun = try self.startLoop(preLoopConfiguration: { (commandWriter: ProtocolWriter) throws -> Void in
            try commandWriter.sendCommand(.sampleMlxMemory);
        });
        let loopFailure: (any Error)? = loopRun.joinWithin(seconds: 10);
        let startupFailure: WorkerProcessError? = loopFailure as? WorkerProcessError;
        #expect(startupFailure == WorkerProcessError.startup(.initializeWorkerNotFirst(
            description: "expected InitializeWorker as the first worker command")));
    }

    @Test
    func should_reject_end_of_stream_before_the_startup_policy() throws {
        let loopRun: WorkerLoopRun = try self.startLoop(preLoopConfiguration: nil);
        loopRun.finish();
        let loopFailure: (any Error)? = loopRun.joinWithin(seconds: 10);
        let startupFailure: WorkerProcessError? = loopFailure as? WorkerProcessError;
        #expect(startupFailure == WorkerProcessError.startup(.initializeWorkerNotFirst(
            description: "expected InitializeWorker as the first worker command")));
    }

    @Test(arguments: [nil, UInt64(1)])
    func should_emit_idle_then_its_startup_policy_with_resolved_ceilings(
        configuredMaximumMlxMemoryBytes: UInt64?
    ) throws {
        try self.assertBootstrappedLifecycle(
            configuredMaximumMlxMemoryBytes: configuredMaximumMlxMemoryBytes,
            expectedConfigurationGeneration: configuredMaximumMlxMemoryBytes == nil ? "gen-1" : "gen-2");
    }

    @Test
    func should_reject_chat_generate_as_invalid_request_and_survive_duplicate_startup() throws {
        let loopRun: WorkerLoopRun = try self.startBootstrappedLoop(
            startupConfiguration: self.startupConfiguration(
                configurationGeneration: "gen-1",
                configuredMaximumMlxMemoryBytes: nil));
        defer { loopRun.finish(); }
        try loopRun.commandWriter.sendCommand(.initializeWorker(
            self.startupConfiguration(
                configurationGeneration: "gen-1",
                configuredMaximumMlxMemoryBytes: nil)));
        // The bootstrapped lifecycle events precede every command answer.
        _ = try self.expectEvent(loopRun.eventReader);
        _ = try self.expectEvent(loopRun.eventReader);

        let generateCommand: ChatGenerationCommand = self.chatGenerationCommand(requestId: 7);
        try loopRun.commandWriter.sendCommand(.generate(generateCommand));

        try loopRun.commandWriter.sendCommand(.generateImage(
            self.imageGenerationCommand(requestId: 8)));
        try loopRun.commandWriter.sendCommand(.generateEmbeddings(
            self.embeddingsCommand(requestId: 9)));

        let rejectedEvent: WorkerEvent = try self.expectEvent(loopRun.eventReader);
        #expect(rejectedEvent == .failed(
            requestId: RequestId(rawRequestId: 7),
            reason: .invalidRequest(reason: "the loaded model does not support chat generation")));

        let imageEvents: Array<WorkerEvent> = try self.expectEvents(loopRun.eventReader, count: 2);
        #expect(imageEvents[0] == .imageGenerationFailed(
            requestId: RequestId(rawRequestId: 8),
            reason: .modelDoesNotSupportImageGeneration));
        #expect(imageEvents[1] == .imageGenerationFinalized(
            requestId: RequestId(rawRequestId: 8), elapsedMillis: 0, mlxMemorySnapshot: nil));

        let embeddingsEvents: Array<WorkerEvent> = try self.expectEvents(loopRun.eventReader, count: 2);
        #expect(embeddingsEvents[0] == .embeddingsFailed(
            requestId: RequestId(rawRequestId: 9),
            reason: .fatalExecution(reason: "the loaded model does not support embeddings")));
        #expect(embeddingsEvents[1] == .embeddingsFinalized(
            requestId: RequestId(rawRequestId: 9), elapsedMillis: 0, mlxMemorySnapshot: nil));

        // The duplicate InitializeWorker was ignored: the loop is still
        // answering commands, proving no second bootstrap happened.
        try loopRun.commandWriter.sendCommand(.sampleMlxMemory);
        let pollAnswer: WorkerEvent = try self.expectEvent(loopRun.eventReader);
        guard case .mlxMemorySample = pollAnswer else {
            Issue.record("expected an idle memory sample, got \(pollAnswer)");
            return;
        }
    }

    @Test
    func should_accept_memory_limit_updates_inside_the_machine_and_reject_outside() throws {
        let loopRun: WorkerLoopRun = try self.startBootstrappedLoop(
            startupConfiguration: self.startupConfiguration(
                configurationGeneration: "gen-1",
                configuredMaximumMlxMemoryBytes: nil));
        defer { loopRun.finish(); }
        let machineMlxMemoryCeilingBytes: UInt64 = try self.observedMachineCeiling(loopRun.eventReader);

        try loopRun.commandWriter.sendCommand(.updateMlxMemoryLimit(
            effectiveMlxMemoryCeilingBytes: 0,
            configurationGeneration: "gen-1"));
        let zeroReject: WorkerEvent = try self.expectEvent(loopRun.eventReader);
        #expect(zeroReject == .mlxMemoryLimitRejected(
            requestedMlxMemoryCeilingBytes: 0,
            minimumMlxMemoryCeilingBytes: 1,
            machineMlxMemoryCeilingBytes: machineMlxMemoryCeilingBytes,
            reason: "requested memory ceiling is outside the worker machine limit"));

        let beyondMachine: UInt64 = machineMlxMemoryCeilingBytes + 1;
        try loopRun.commandWriter.sendCommand(.updateMlxMemoryLimit(
            effectiveMlxMemoryCeilingBytes: beyondMachine,
            configurationGeneration: "gen-1"));
        let beyondReject: WorkerEvent = try self.expectEvent(loopRun.eventReader);
        #expect(beyondReject == .mlxMemoryLimitRejected(
            requestedMlxMemoryCeilingBytes: beyondMachine,
            minimumMlxMemoryCeilingBytes: 1,
            machineMlxMemoryCeilingBytes: machineMlxMemoryCeilingBytes,
            reason: "requested memory ceiling is outside the worker machine limit"));

        try loopRun.commandWriter.sendCommand(.updateMlxMemoryLimit(
            effectiveMlxMemoryCeilingBytes: machineMlxMemoryCeilingBytes,
            configurationGeneration: "gen-3"));
        let acceptedChange: WorkerEvent = try self.expectEvent(loopRun.eventReader);
        #expect(acceptedChange == .mlxMemoryLimitChanged(
            effectiveMlxMemoryCeilingBytes: machineMlxMemoryCeilingBytes,
            minimumMlxMemoryCeilingBytes: 1,
            expertMemoryMode: .resident,
            mlxMemorySnapshot: nil,
            expertResidency: nil));

        let loweredCeiling: UInt64 = machineMlxMemoryCeilingBytes / 2;
        try loopRun.commandWriter.sendCommand(.updateMlxMemoryLimit(
            effectiveMlxMemoryCeilingBytes: loweredCeiling,
            configurationGeneration: "gen-4"));
        let loweredChange: WorkerEvent = try self.expectEvent(loopRun.eventReader);
        #expect(loweredChange == .mlxMemoryLimitChanged(
            effectiveMlxMemoryCeilingBytes: loweredCeiling,
            minimumMlxMemoryCeilingBytes: 1,
            expertMemoryMode: .resident,
            mlxMemorySnapshot: nil,
            expertResidency: nil));
    }

    @Test
    func should_fail_a_model_less_swap_while_the_worker_stays_responsive() throws {
        let loopRun: WorkerLoopRun = try self.startBootstrappedLoop(
            startupConfiguration: self.startupConfiguration(
                configurationGeneration: "gen-1",
                configuredMaximumMlxMemoryBytes: nil));
        defer { loopRun.finish(); }
        _ = try self.expectEvent(loopRun.eventReader);
        _ = try self.expectEvent(loopRun.eventReader);

        try loopRun.commandWriter.sendCommand(.swapModel(
            modelDirectory: "/fictional/models/qwen3.5",
            modelConfiguration: self.autoregressiveModelConfiguration()));

        let swapFailure: WorkerEvent = try self.expectEvent(loopRun.eventReader);
        guard case let .modelSwapFailed(loadedModelRemainsReady, modelLoadFailureReason) = swapFailure else {
            Issue.record("expected a model swap failure, got \(swapFailure)");
            return;
        }
        #expect(loadedModelRemainsReady == false);
        #expect(modelLoadFailureReason.isEmpty == false);

        // The worker answered the swap failure and stays responsive.
        try loopRun.commandWriter.sendCommand(.sampleMlxMemory);
        let pollAnswer: WorkerEvent = try self.expectEvent(loopRun.eventReader);
        guard case .mlxMemorySample = pollAnswer else {
            Issue.record("expected a memory sample after the failed swap");
            return;
        }
    }

    @Test
    func should_acknowledge_a_prompt_cache_clear_with_zero_footprint() throws {
        let loopRun: WorkerLoopRun = try self.startBootstrappedLoop(
            startupConfiguration: self.startupConfiguration(
                configurationGeneration: "gen-1",
                configuredMaximumMlxMemoryBytes: nil));
        defer { loopRun.finish(); }
        _ = try self.expectEvent(loopRun.eventReader);
        _ = try self.expectEvent(loopRun.eventReader);

        try loopRun.commandWriter.sendCommand(.clearPromptCache(modelId: nil));
        #expect(try self.expectEvent(loopRun.eventReader) == .promptCacheCleared(
            modelId: nil, blocksRemoved: 0, bytesFreed: 0));

        try loopRun.commandWriter.sendCommand(.clearPromptCache(modelId: "qwen3.5"));
        #expect(try self.expectEvent(loopRun.eventReader) == .promptCacheCleared(
            modelId: "qwen3.5", blocksRemoved: 0, bytesFreed: 0));
    }

    @Test
    func should_exit_cleanly_at_end_of_stream_after_bootstrap() throws {
        let loopRun: WorkerLoopRun = try self.startBootstrappedLoop(
            startupConfiguration: self.startupConfiguration(
                configurationGeneration: "gen-1",
                configuredMaximumMlxMemoryBytes: nil));
        _ = try self.expectEvent(loopRun.eventReader);
        _ = try self.expectEvent(loopRun.eventReader);
        loopRun.finish();
        let loopFailure: (any Error)? = loopRun.joinWithin(seconds: 10);
        #expect(loopFailure == nil);
    }

    // MARK: - Harness

    private func startupConfiguration(
        configurationGeneration: String,
        configuredMaximumMlxMemoryBytes: UInt64?
    ) -> WorkerStartupConfiguration {
        return WorkerStartupConfiguration(
            configurationGeneration: configurationGeneration,
            globalPromptCacheRootDirectory: "/tmp/astronomical-worker-loop-cache",
            globalPromptCacheMaximumSizeBytes: 1073741824,
            persistentPromptCacheEnabled: true,
            configuredMaximumMlxMemoryBytes: configuredMaximumMlxMemoryBytes,
            performanceAttributionEnabled: false,
            loggingDirectory: "/tmp/astronomical-worker-loop-logs",
            loggingLevel: .info,
            retainedLogFileCount: 3);
    }

    private func chatGenerationCommand(requestId: UInt64) -> ChatGenerationCommand {
        return ChatGenerationCommand(
            requestId: RequestId(rawRequestId: requestId),
            model: "qwen3.5",
            messages: [.system(content: "answer plainly"), .user(content: "hello", images: [])],
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

    private func imageGenerationCommand(requestId: UInt64) -> ImageGenerationCommand {
        return ImageGenerationCommand(
            requestId: RequestId(rawRequestId: requestId),
            model: "flux2-klein",
            prompt: "a paper boat",
            settings: ImageGenerationSettings(
                widthPixels: 1024,
                heightPixels: 1024,
                steps: 4,
                guidanceThousandths: 3500,
                seed: 11));
    }

    private func embeddingsCommand(requestId: UInt64) -> EmbeddingsCommand {
        return EmbeddingsCommand(
            requestId: RequestId(rawRequestId: requestId),
            model: "modernbert",
            inputs: ["one", "two"],
            encodingFormat: .float,
            dimensions: nil);
    }

    private func autoregressiveModelConfiguration() -> WorkerModelConfiguration {
        return WorkerModelConfiguration.autoregressive(WorkerAutoregressiveModelConfiguration(
            modelId: "qwen3.5",
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
                experimentalFusedMoeDecodeEnabled: false)));
    }

    /**
     * Starts the loop over a real pipe pair without bootstrapping it first;
     * `preLoopConfiguration` writes whatever the journey needs before the
     * loop thread starts.
     */
    private func startLoop(
        preLoopConfiguration: ((ProtocolWriter) throws -> Void)?
    ) throws -> WorkerLoopRun {
        let commandPipe: Pipe = Pipe();
        let eventPipe: Pipe = Pipe();
        let commandWriter: ProtocolWriter = ProtocolWriter(transport: PipeFrameTransport(
            fileDescriptor: commandPipe.fileHandleForWriting.fileDescriptor,
            isWriteEnd: true));
        let eventReader: ProtocolReader = ProtocolReader(transport: PipeFrameTransport(
            fileDescriptor: eventPipe.fileHandleForReading.fileDescriptor,
            isWriteEnd: false));
        if let preLoopConfiguration = preLoopConfiguration {
            try preLoopConfiguration(commandWriter);
        }
        return WorkerLoopRun(
            commandPipe: commandPipe,
            eventPipe: eventPipe,
            commandWriter: commandWriter,
            eventReader: eventReader,
            loopBody: { [commandPipe, eventPipe] in
                try WorkerCommandLoop.runBootstrappedWorker(
                    readTransport: PipeFrameTransport(
                        fileDescriptor: commandPipe.fileHandleForReading.fileDescriptor,
                        isWriteEnd: false),
                    writeTransport: PipeFrameTransport(
                        fileDescriptor: eventPipe.fileHandleForWriting.fileDescriptor,
                        isWriteEnd: true));
            });
    }

    private func startBootstrappedLoop(
        startupConfiguration: WorkerStartupConfiguration
    ) throws -> WorkerLoopRun {
        let loopRun: WorkerLoopRun = try self.startLoop(preLoopConfiguration: nil);
        try loopRun.commandWriter.sendCommand(.initializeWorker(startupConfiguration));
        return loopRun;
    }

    private func expectEvent(_ eventReader: ProtocolReader) throws -> WorkerEvent {
        guard let workerEvent: WorkerEvent = try eventReader.nextEvent() else {
            Issue.record("the worker event stream ended before the expected event");
            throw WorkerCommandLoopTests.harnessFailure();
        }
        return workerEvent;
    }

    private func expectEvents(
        _ eventReader: ProtocolReader,
        count expectedEventCount: Int
    ) throws -> Array<WorkerEvent> {
        var collectedEvents: Array<WorkerEvent> = Array<WorkerEvent>();
        collectedEvents.reserveCapacity(expectedEventCount);
        for _ in 0..<expectedEventCount {
            collectedEvents.append(try self.expectEvent(eventReader));
        }
        return collectedEvents;
    }

    /**
     * Reads the bootstrapped lifecycle (idle, runtime policy) and returns the
     * machine ceiling the worker reported.
     */
    private func observedMachineCeiling(_ eventReader: ProtocolReader) throws -> UInt64 {
        let idleEvent: WorkerEvent = try self.expectEvent(eventReader);
        guard case let .idle(machineMlxMemoryCeilingBytes, _, _) = idleEvent else {
            Issue.record("expected the idle lifecycle event, got \(idleEvent)");
            throw WorkerCommandLoopTests.harnessFailure();
        }
        _ = try self.expectEvent(eventReader);
        return machineMlxMemoryCeilingBytes;
    }

    private func assertBootstrappedLifecycle(
        configuredMaximumMlxMemoryBytes: UInt64?,
        expectedConfigurationGeneration: String
    ) throws {
        let loopRun: WorkerLoopRun = try self.startBootstrappedLoop(
            startupConfiguration: self.startupConfiguration(
                configurationGeneration: expectedConfigurationGeneration,
                configuredMaximumMlxMemoryBytes: configuredMaximumMlxMemoryBytes));
        defer { loopRun.finish(); }

        let idleEvent: WorkerEvent = try self.expectEvent(loopRun.eventReader);
        guard case let .idle(
            machineMlxMemoryCeilingBytes,
            effectiveMlxMemoryCeilingBytes,
            minimumMlxMemoryCeilingBytes) = idleEvent else {
            Issue.record("expected the idle lifecycle event, got \(idleEvent)");
            return;
        }
        #expect(machineMlxMemoryCeilingBytes > 0);
        #expect(effectiveMlxMemoryCeilingBytes == min(
            configuredMaximumMlxMemoryBytes ?? machineMlxMemoryCeilingBytes,
            machineMlxMemoryCeilingBytes));
        #expect(minimumMlxMemoryCeilingBytes == 1);

        let acknowledgedPolicy: WorkerEvent = try self.expectEvent(loopRun.eventReader);
        #expect(acknowledgedPolicy == .runtimeFeatureConfigurationApplied(
            WorkerRuntimeFeatureConfiguration(
                configurationGeneration: expectedConfigurationGeneration,
                persistentPromptCacheEnabled: true,
                promptCacheMaximumSizeBytes: 1073741824,
                loadedModel: nil)));
        loopRun.finish();
        let loopFailure: (any Error)? = loopRun.joinWithin(seconds: 10);
        #expect(loopFailure == nil);
    }

    private static func harnessFailure() -> Error {
        return WorkerProcessError.startup(.initializeWorkerNotFirst(description: "test harness failure"));
    }
}

/**
 * One loop run under test: the thread, the harness-side transports, and the
 * captured loop failure. `finish()` closes the command side, which ends the
 * loop through its normal EOF path; `joinWithin` must run before the run
 * drops so no pipe closes behind the loop thread's back.
 */
final class WorkerLoopRun {
    let failureStore: FailureStore = FailureStore()

    let commandWriter: ProtocolWriter;
    let eventReader: ProtocolReader;
    private let loopThread: Thread;
    // The pipe objects stay owned for the whole run: their file descriptors
    // back the loop's transports, and a deallocated Pipe would close them
    // behind the loop thread's back.
    private let commandPipe: Pipe;
    private let eventPipe: Pipe;
    private let finishLock: NSLock;
    private var isFinishCalled: Bool;

    init(
        commandPipe: Pipe,
        eventPipe: Pipe,
        commandWriter: ProtocolWriter,
        eventReader: ProtocolReader,
        loopBody: @escaping @Sendable () throws -> Void
    ) {
        self.commandPipe = commandPipe;
        self.eventPipe = eventPipe;
        self.commandWriter = commandWriter;
        self.eventReader = eventReader;
        self.finishLock = NSLock();
        self.isFinishCalled = false;
        let failureStore: FailureStore = self.failureStore;
        self.loopThread = Thread {
            do {
                try loopBody();
            } catch let loopFailure {
                failureStore.record(loopFailure);
            }
        };
        self.loopThread.name = "worker-command-loop-under-test";
        self.loopThread.start();
    }

    /**
     * Closes the command side so the loop observes EOF.
     */
    func finish() -> Void {
        self.finishLock.lock();
        defer { self.finishLock.unlock(); }
        if self.isFinishCalled {
            return;
        }
        self.isFinishCalled = true;
        self.commandWriter.closeTransportFileDescriptor();
    }

    /**
     * Waits for the loop thread and returns the captured loop failure, if any.
     * The wait keys on `isFinished` alone: `isExecuting` is false in the
     * window between `start()` and the closure's first instruction, and
     * joining on it races the record below.
     */
    func joinWithin(seconds: TimeInterval) -> Error? {
        let joinDeadline: Date = Date().addingTimeInterval(seconds);
        while self.loopThread.isFinished == false && Date() < joinDeadline {
            Thread.sleep(forTimeInterval: 0.01);
        }
        if self.loopThread.isFinished == false {
            Issue.record("the worker loop thread did not exit within \(seconds)s");
        }
        return self.failureStore.takeCapturedFailure();
    }

    final class FailureStore: @unchecked Sendable {
        private let lock: NSLock = NSLock();
        private var capturedFailure: Error?;

        func record(_ loopFailure: Error) -> Void {
            self.lock.lock();
            self.capturedFailure = loopFailure;
            self.lock.unlock();
        }

        func takeCapturedFailure() -> Error? {
            self.lock.lock();
            defer { self.lock.unlock(); }
            let capturedFailure: Error? = self.capturedFailure;
            self.capturedFailure = nil;
            return capturedFailure;
        }
    }
}

extension WorkerCommandLoopTests {

    /**
     * The real model family factory classifies a real directory, streams
     * the validated artifact weights into the dense engine, and the swap
     * completes with the model's capabilities; the worker stays responsive
     * for the next command.
     */
    @Test(.timeLimit(.minutes(2)))
    func should_swap_a_real_qwen35_directory_through_the_artifact_weight_path() throws {
        MLXMetallibLocator.overrideMetallibPathIfNecessary();
        let (modelDirectoryUrl, _): (URL, TinyDenseArtifactFixture.SynthesizedLayout) =
            try TinyDenseArtifactFixture.writeModelDirectory(includeTokenizerFiles: true);
        defer { try? FileManager.default.removeItem(at: modelDirectoryUrl); }

        let loopRun: WorkerLoopRun = try self.startBootstrappedLoop(
            startupConfiguration: self.startupConfiguration(
                configurationGeneration: "gen-1",
                configuredMaximumMlxMemoryBytes: nil));
        defer { loopRun.finish(); }
        _ = try self.expectEvent(loopRun.eventReader);
        _ = try self.expectEvent(loopRun.eventReader);

        try loopRun.commandWriter.sendCommand(.swapModel(
            modelDirectory: modelDirectoryUrl.path,
            modelConfiguration: self.autoregressiveModelConfiguration()));
        let swapAnswer: WorkerEvent = try self.expectEvent(loopRun.eventReader);
        guard case let .modelSwapped(swappedModelId, capabilities, expertMemoryMode, minimumMlxMemoryCeilingBytes) = swapAnswer else {
            Issue.record("expected a completed swap, got \(swapAnswer)");
            return;
        }
        #expect(swappedModelId == "qwen3.5");
        #expect(capabilities.chat != nil);
        #expect(capabilities.chat?.contextWindow == 4096);
        #expect(expertMemoryMode == nil);
        #expect(minimumMlxMemoryCeilingBytes == 1);

        // A successful swap publishes the runtime feature configuration
        // bound to the freshly loaded model before the worker answers the
        // next command; the fail-closed path never emits it.
        let featureConfigurationEvent: WorkerEvent = try self.expectEvent(loopRun.eventReader);
        guard case let .runtimeFeatureConfigurationApplied(featureConfiguration) = featureConfigurationEvent else {
            Issue.record("expected the runtime feature configuration after the swap, got \\(featureConfigurationEvent)");
            return;
        }
        #expect(featureConfiguration.loadedModel?.modelId() == "qwen3.5");

        try loopRun.commandWriter.sendCommand(.sampleMlxMemory);
        let pollAnswer: WorkerEvent = try self.expectEvent(loopRun.eventReader);
        guard case .mlxMemorySample = pollAnswer else {
            Issue.record("expected a memory sample after the swap, got \(pollAnswer)");
            return;
        }
    }
}
