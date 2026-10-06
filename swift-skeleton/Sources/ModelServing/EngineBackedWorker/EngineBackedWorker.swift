import Foundation;

import IpcProtocol;

/// The engine-backed worker: bootstrap, model swap, and the stepwise command
/// loop that interleaves supervisor commands between engine decode steps.
///
/// Mirrors crates/model-serving/src/engine_backed_worker/{protocol.rs,
/// construction.rs,idle_command.rs,model_swap.rs,memory_limit.rs}. The loop
/// is the single MLX owner; commands interleave at token boundaries through
/// `ProtocolReader.pollNextCommand`, exactly as the Rust loop's biased
/// `tokio::select!` interleaves them. A nil chat-runtime factory keeps the
/// model-less worker answering idle commands only.
public final class EngineBackedWorker {

    let chatRuntimeFactory: (any ChatModelRuntimeFactory)?;
    var loadedChatRuntime: LoadedChatRuntime?;
    var activeGeneration: ActiveEngineGeneration?;

    var machineMlxMemoryCeilingBytes: UInt64 = 0;
    var effectiveMlxMemoryCeilingBytes: UInt64 = 0;
    var minimumMlxMemoryCeilingBytes: UInt64 = 1;
    var acknowledgedConfigurationGeneration: String = "";
    var acknowledgedFeatureConfiguration: WorkerRuntimeFeatureConfiguration?;

    private let performanceAttributionEnabled: Bool;
    private let sampleMachineMlxMemoryCeilingBytes: () -> Int;
    private let resolveEffectiveCeilingBytes:
        (_ configuredMlxMemoryCeilingBytes: UInt64?, _ machineMlxMemoryCeilingBytes: Int) -> Int;

    public init(
        chatRuntimeFactory: (any ChatModelRuntimeFactory)?,
        performanceAttributionEnabled: Bool = false,
        sampleMachineMlxMemoryCeilingBytes: @escaping () -> Int = { return 1 },
        resolveEffectiveCeilingBytes: @escaping (
            _ configuredMlxMemoryCeilingBytes: UInt64?, _ machineMlxMemoryCeilingBytes: Int
        ) -> Int = { (configuredMlxMemoryCeilingBytes, machineMlxMemoryCeilingBytes) in
            guard let configuredMlxMemoryCeilingBytes = configuredMlxMemoryCeilingBytes else {
                return machineMlxMemoryCeilingBytes;
            }
            return Int(min(
                UInt64(machineMlxMemoryCeilingBytes), configuredMlxMemoryCeilingBytes));
        }
    ) {
        self.chatRuntimeFactory = chatRuntimeFactory;
        self.performanceAttributionEnabled = performanceAttributionEnabled;
        self.sampleMachineMlxMemoryCeilingBytes = sampleMachineMlxMemoryCeilingBytes;
        self.resolveEffectiveCeilingBytes = resolveEffectiveCeilingBytes;
    }

    /// Bootstraps the worker over the framed transports and serves commands
    /// until the supervisor closes the command side.
    public func run(
        commandReader: ProtocolReader,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        try self.bootstrap(commandReader: commandReader, eventWriter: eventWriter);
        try self.serveCommandLoop(commandReader: commandReader, eventWriter: eventWriter);
    }

    /// Serves commands with an already-read startup policy; the worker
    /// process entry point validates the first command before the engine
    /// seams exist, then hands the policy over.
    public func run(
        workerStartupConfiguration: WorkerStartupConfiguration,
        commandReader: ProtocolReader,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        try self.applyStartupPolicy(
            workerStartupConfiguration, eventWriter: eventWriter);
        try self.serveCommandLoop(commandReader: commandReader, eventWriter: eventWriter);
    }

    /// Reads the startup policy and emits the model-less lifecycle events.
    private func bootstrap(
        commandReader: ProtocolReader,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        guard case let .initializeWorker(workerStartupConfiguration) =
            try commandReader.nextCommand() else {
            throw WorkerProcessFailure.expectedInitializeWorkerFirst;
        }
        return try self.applyStartupPolicy(
            workerStartupConfiguration, eventWriter: eventWriter);
    }

    /// Applies the startup policy and emits the model-less lifecycle events.
    private func applyStartupPolicy(
        _ workerStartupConfiguration: WorkerStartupConfiguration,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        self.machineMlxMemoryCeilingBytes = UInt64(self.sampleMachineMlxMemoryCeilingBytes());
        self.effectiveMlxMemoryCeilingBytes = UInt64(self.resolveEffectiveCeilingBytes(
            workerStartupConfiguration.configuredMaximumMlxMemoryBytes,
            Int(self.machineMlxMemoryCeilingBytes)));
        self.acknowledgedConfigurationGeneration =
            workerStartupConfiguration.configurationGeneration;
        self.acknowledgedFeatureConfiguration = WorkerRuntimeFeatureConfiguration(
            configurationGeneration: workerStartupConfiguration.configurationGeneration,
            persistentPromptCacheEnabled: workerStartupConfiguration.persistentPromptCacheEnabled,
            promptCacheMaximumSizeBytes: workerStartupConfiguration.globalPromptCacheMaximumSizeBytes,
            loadedModel: nil);
        try eventWriter.sendEvent(.idle(
            machineMlxMemoryCeilingBytes: self.machineMlxMemoryCeilingBytes,
            effectiveMlxMemoryCeilingBytes: self.effectiveMlxMemoryCeilingBytes,
            minimumMlxMemoryCeilingBytes: self.minimumMlxMemoryCeilingBytes));
        if let acknowledgedFeatureConfiguration = self.acknowledgedFeatureConfiguration {
            try eventWriter.sendEvent(
                .runtimeFeatureConfigurationApplied(acknowledgedFeatureConfiguration));
        }
    }

    /// The stepwise command loop: one engine step per turn while a
    /// generation is active, bounded polls between steps. A fatal runtime
    /// error reports `failed` to the supervisor before the process exits.
    private func serveCommandLoop(
        commandReader: ProtocolReader,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        do {
            return try self.serveCommandLoopBody(
                commandReader: commandReader, eventWriter: eventWriter);
        } catch let fatalRuntimeError as WorkerRuntimeError {
            let fatalReason: String;
            let activeRequestId: RequestId?;
            switch fatalRuntimeError {
            case let .inferenceEngineGenerationFailed(reason):
                fatalReason = reason;
                activeRequestId = self.activeGeneration?.requestId;
            case let .modelSwapFailed(modelLoadFailureReason):
                fatalReason = modelLoadFailureReason;
                activeRequestId = nil;
            }
            try? eventWriter.sendEvent(.failed(
                requestId: activeRequestId ?? RequestId(rawRequestId: 0),
                reason: .fatalExecution(reason: fatalReason)));
            throw fatalRuntimeError;
        }
    }

    private func serveCommandLoopBody(
        commandReader: ProtocolReader,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        while true {
            if var activeGeneration = self.activeGeneration {
                let polledOutcome: ProtocolReader.PolledCommand =
                    try commandReader.pollNextCommand(
                        timeoutMilliseconds: EngineBackedWorker
                            .activeGenerationPollTimeoutMilliseconds);
                switch polledOutcome {
                case let .command(polledCommand):
                    let keepsGenerating: Bool = try self.serveActiveGenerationCommand(
                        polledCommand,
                        activeGeneration: &activeGeneration,
                        eventWriter: eventWriter);
                    self.activeGeneration = keepsGenerating ? activeGeneration : nil;
                case .none:
                    let keepsGenerating: Bool = try self.advanceGeneration(
                        &activeGeneration, eventWriter: eventWriter);
                    self.activeGeneration = keepsGenerating ? activeGeneration : nil;
                case .endOfStream:
                    return;
                }
                continue;
            }
            guard let idleCommand: WorkerCommand = try commandReader.nextCommand() else {
                return;
            }
            try self.serveIdleCommand(idleCommand, eventWriter: eventWriter);
        }
    }

    /// Handles one command while a generation owns the engine.
    ///
    /// Mirrors the Rust `select!` command arms: a matching cancel finalizes,
    /// every other admission is answered without disturbing the active
    /// request. Returns whether the generation keeps running.
    private func serveActiveGenerationCommand(
        _ workerCommand: WorkerCommand,
        activeGeneration: inout ActiveEngineGeneration,
        eventWriter: ProtocolWriter
    ) throws -> Bool {
        switch workerCommand {
        case .initializeWorker:
            return true;
        case let .cancel(cancelledRequestId):
            guard cancelledRequestId == activeGeneration.requestId else {
                return true;
            }
            return try self.finishGeneration(
                &activeGeneration,
                completionReason: .cancelled,
                eventWriter: eventWriter);
        case let .generate(generationCommand):
            try eventWriter.sendEvent(.failed(
                requestId: generationCommand.requestId,
                reason: .engineBusy));
            return true;
        case let .generateImage(imageGenerationCommand):
            try eventWriter.sendEvent(.imageGenerationFailed(
                requestId: imageGenerationCommand.requestId,
                reason: .engineBusy));
            try eventWriter.sendEvent(.imageGenerationFinalized(
                requestId: imageGenerationCommand.requestId,
                elapsedMillis: 0,
                mlxMemorySnapshot: nil));
            return true;
        case let .generateEmbeddings(embeddingsCommand):
            try eventWriter.sendEvent(.embeddingsFailed(
                requestId: embeddingsCommand.requestId,
                reason: .fatalExecution(
                    reason: "the inference engine is busy with another generation")));
            try eventWriter.sendEvent(.embeddingsFinalized(
                requestId: embeddingsCommand.requestId,
                elapsedMillis: 0,
                mlxMemorySnapshot: nil));
            return true;
        case .swapModel:
            return true;
        case .sampleMlxMemory:
            return true;
        case let .updateMlxMemoryLimit(requestedMlxMemoryCeilingBytes, _):
            try eventWriter.sendEvent(.mlxMemoryLimitRejected(
                requestedMlxMemoryCeilingBytes: requestedMlxMemoryCeilingBytes,
                minimumMlxMemoryCeilingBytes: self.minimumMlxMemoryCeilingBytes,
                machineMlxMemoryCeilingBytes: self.machineMlxMemoryCeilingBytes,
                reason: "memory limits cannot change during generation"));
            return true;
        case .clearPromptCache:
            return true;
        }
    }

    private static let activeGenerationPollTimeoutMilliseconds: Int32 = 10;

    /// Cancels the engine request and reports the released state.
    func cancelEngineRequest(
        _ requestId: RequestId
    ) -> GenerationFinalization {
        guard let loadedChatRuntime = self.loadedChatRuntime else {
            return GenerationFinalization();
        }
        do {
            return try loadedChatRuntime.engine.cancelGeneration(requestId: requestId);
        } catch {
            // Cancellation releases state opportunistically; a failing
            // release never escalates past the finalization event.
            return GenerationFinalization();
        }
    }
}

/// Bootstrap failure surfaced to the worker process entry point.
public enum WorkerProcessFailure: Error, Equatable {
    case expectedInitializeWorkerFirst;
}
