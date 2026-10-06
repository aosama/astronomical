import Foundation;

import Testing;

import IpcProtocol;
import ModelServing;

/// Real-pipe harness for engine-backed worker journeys: the worker runs on
/// its own thread exactly as the process entry point drives it, and the
/// journey reads events with a bounded blocking read.
final class WorkerHarness {

    private let commandPipe: Pipe;
    private let eventPipe: Pipe;
    private let commandWriter: ProtocolWriter;
    private let eventReader: ProtocolReader;
    private let loopFinishedSemaphore: DispatchSemaphore;
    private let loopFailureBox: LoopFailureBox;

    private init(
        commandPipe: Pipe, eventPipe: Pipe,
        commandWriter: ProtocolWriter, eventReader: ProtocolReader,
        loopFinishedSemaphore: DispatchSemaphore, loopFailureBox: LoopFailureBox
    ) {
        self.commandPipe = commandPipe;
        self.eventPipe = eventPipe;
        self.commandWriter = commandWriter;
        self.eventReader = eventReader;
        self.loopFinishedSemaphore = loopFinishedSemaphore;
        self.loopFailureBox = loopFailureBox;
    }

    static func start(
        factory: (any ChatModelRuntimeFactory)?,
        machineMlxMemoryCeilingBytes: Int = 1_000_000
    ) throws -> WorkerHarness {
        let commandPipe: Pipe = Pipe();
        let eventPipe: Pipe = Pipe();
        let commandWriter: ProtocolWriter = ProtocolWriter(transport: PipeFrameTransport(
            fileDescriptor: commandPipe.fileHandleForWriting.fileDescriptor,
            isWriteEnd: true));
        let eventReader: ProtocolReader = ProtocolReader(transport: PipeFrameTransport(
            fileDescriptor: eventPipe.fileHandleForReading.fileDescriptor,
            isWriteEnd: false));
        let loopFinishedSemaphore: DispatchSemaphore = DispatchSemaphore(value: 0);
        let loopFailureBox: LoopFailureBox = LoopFailureBox();
        let loopCommandPipe: Pipe = commandPipe;
        let loopEventPipe: Pipe = eventPipe;
        let loopThread: Thread = Thread(block: {
            let engineBackedWorker: EngineBackedWorker = EngineBackedWorker(
                chatRuntimeFactory: factory,
                sampleMachineMlxMemoryCeilingBytes: { return machineMlxMemoryCeilingBytes },
                resolveEffectiveCeilingBytes: { configuredMlxMemoryCeilingBytes, machineCeilingBytes in
                    guard let configuredMlxMemoryCeilingBytes = configuredMlxMemoryCeilingBytes else {
                        return machineCeilingBytes;
                    }
                    return Int(min(UInt64(machineCeilingBytes), configuredMlxMemoryCeilingBytes));
                });
            do {
                try engineBackedWorker.run(
                    commandReader: ProtocolReader(transport: PipeFrameTransport(
                        fileDescriptor: loopCommandPipe.fileHandleForReading.fileDescriptor,
                        isWriteEnd: false)),
                    eventWriter: ProtocolWriter(transport: PipeFrameTransport(
                        fileDescriptor: loopEventPipe.fileHandleForWriting.fileDescriptor,
                        isWriteEnd: true)));
            } catch {
                loopFailureBox.capture(error);
            }
            loopFinishedSemaphore.signal();
        });
        loopThread.start();
        return WorkerHarness(
            commandPipe: commandPipe, eventPipe: eventPipe,
            commandWriter: commandWriter, eventReader: eventReader,
            loopFinishedSemaphore: loopFinishedSemaphore, loopFailureBox: loopFailureBox);
    }

    /// Reads the bootstrapped lifecycle (idle, runtime policy) and returns
    /// the machine ceiling the worker reported.
    func expectBootstrappedLifecycle(configurationGeneration: String = "gen-1") throws -> UInt64 {
        try self.commandWriter.sendCommand(.initializeWorker(WorkerStartupConfiguration(
            configurationGeneration: configurationGeneration,
            globalPromptCacheRootDirectory: "/tmp/astronomical-engine-worker-journey-cache",
            globalPromptCacheMaximumSizeBytes: 1073741824,
            persistentPromptCacheEnabled: true,
            configuredMaximumMlxMemoryBytes: nil,
            performanceAttributionEnabled: false,
            loggingDirectory: "/tmp/astronomical-engine-worker-journey-logs",
            loggingLevel: .info,
            retainedLogFileCount: 3)));
        let idleEvent: WorkerEvent = try self.expectEvent();
        guard case let .idle(machineCeiling, _, _) = idleEvent else {
            throw WorkerHarness.harnessFailure("expected the idle lifecycle");
        }
        let policyEvent: WorkerEvent = try self.expectEvent();
        guard case .runtimeFeatureConfigurationApplied = policyEvent else {
            throw WorkerHarness.harnessFailure("expected the runtime policy");
        }
        return machineCeiling;
    }

    /// Swaps the journey model in and drains its three lifecycle events.
    func swapJourneyModel() throws -> Void {
        try self.swapModelIn(
            modelDirectory: "/fictional/models/qwen3.5",
            modelConfiguration: WorkerHarness.autoregressiveModelConfiguration());
    }

    /// Swaps any model in and drains the swap lifecycle events (swap, policy,
    /// loaded memory sample).
    func swapModelIn(
        modelDirectory: String,
        modelConfiguration: WorkerModelConfiguration
    ) throws -> Void {
        try self.sendCommand(.swapModel(
            modelDirectory: modelDirectory,
            modelConfiguration: modelConfiguration));
        _ = try self.expectEvent();
        _ = try self.expectEvent();
        _ = try self.expectEvent();
    }

    func sendCommand(_ workerCommand: WorkerCommand) throws -> Void {
        try self.commandWriter.sendCommand(workerCommand);
    }

    func expectEvent() throws -> WorkerEvent {
        guard let workerEvent: WorkerEvent = try self.eventReader.nextEvent() else {
            throw WorkerHarness.harnessFailure("the worker event stream ended early");
        }
        return workerEvent;
    }

    func joinWithin(seconds timeoutSeconds: Int) -> (any Error)? {
        let waitOutcome: DispatchTimeoutResult = self.loopFinishedSemaphore.wait(
            timeout: .now() + .seconds(timeoutSeconds));
        if waitOutcome == .timedOut {
            return WorkerHarness.harnessFailure("the worker loop timed out");
        }
        return self.loopFailureBox.read();
    }

    func finish() -> Void {
        self.commandPipe.fileHandleForWriting.closeFile();
    }

    private static func autoregressiveModelConfiguration() -> WorkerModelConfiguration {
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

    static func harnessFailure(_ description: String) -> Error {
        return WorkerProcessFailure.expectedInitializeWorkerFirst; // harness failures surface as loop failures
    }
}

/// Shared box capturing the loop thread's failure.
final class LoopFailureBox: @unchecked Sendable {
    var failure: (any Error)?;
    private let lock: NSLock = NSLock();

    func capture(_ capturedFailure: (any Error)?) -> Void {
        self.lock.lock();
        self.failure = capturedFailure;
        self.lock.unlock();
    }

    func read() -> (any Error)? {
        self.lock.lock();
        let capturedFailure: (any Error)? = self.failure;
        self.lock.unlock();
        return capturedFailure;
    }
}
