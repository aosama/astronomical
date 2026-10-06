import Foundation;

import IpcProtocol;

import RuntimeIntegration;

/// The model-less worker command loop: bootstrap, the idle lifecycle
/// emission, and every idle command answer.
///
/// Mirrors the no-loaded-runtime arms of
/// apps/inference-worker/src/worker_startup.rs and
/// crates/model-serving/src/engine_backed_worker/{protocol,idle_command,
/// generation_start,image_generation,embeddings,memory_limit}.rs. The
/// InitializeWorker write at launch means this process's first command must
/// be the startup policy; commands arriving after bootstrap are answered
/// from the model-less state until the engine slices plug family runtimes
/// in. End of the command stream is the clean exit.
public enum WorkerCommandLoop {

    private static let initializeWorkerNotFirstDescription: String =
        "expected InitializeWorker as the first worker command";

    /// Runs the bootstrapped worker over the framed transports and returns
    /// when the supervisor closes the command side. The first command must
    /// carry the startup policy; anything else is a startup failure.
    public static func runBootstrappedWorker(
        readTransport: any FrameTransport,
        writeTransport: any FrameTransport
    ) throws -> Void {
        let commandReader: ProtocolReader = ProtocolReader(transport: readTransport);
        let eventWriter: ProtocolWriter = ProtocolWriter(transport: writeTransport);
        let firstCommand: WorkerCommand? = try commandReader.nextCommand();
        guard case let .initializeWorker(workerStartupConfiguration) = firstCommand else {
            throw WorkerProcessError.startup(.initializeWorkerNotFirst(
                description: WorkerCommandLoop.initializeWorkerNotFirstDescription));
        }
        return try WorkerCommandLoop.runInitializedWorker(
            workerStartupConfiguration,
            commandReader: commandReader,
            eventWriter: eventWriter);
    }

    private static func runInitializedWorker(
        _ workerStartupConfiguration: WorkerStartupConfiguration,
        commandReader: ProtocolReader,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        let machineMlxMemoryCeilingBytes: Int = try GpuWiredMemoryLimit.sampleIogpuWiredLimitBytes();
        let effectiveMlxMemoryCeilingBytes: Int = GpuWiredMemoryLimit.resolveEffectiveMlxMemoryCeilingBytes(
            configuredMlxMemoryCeilingBytes: workerStartupConfiguration.configuredMaximumMlxMemoryBytes,
            machineMlxMemoryCeilingBytes: machineMlxMemoryCeilingBytes);
        // A model-less worker can never hold resident model state, so its
        // minimum ceiling is the one-byte floor the engine slices also start
        // from before a load raises it.
        let minimumMlxMemoryCeilingBytes: UInt64 = 1;
        let acknowledgedFeatureConfiguration: WorkerRuntimeFeatureConfiguration =
            WorkerRuntimeFeatureConfiguration(
                configurationGeneration: workerStartupConfiguration.configurationGeneration,
                persistentPromptCacheEnabled: workerStartupConfiguration.persistentPromptCacheEnabled,
                promptCacheMaximumSizeBytes: workerStartupConfiguration.globalPromptCacheMaximumSizeBytes,
                loadedModel: nil);
        try eventWriter.sendEvent(.idle(
            machineMlxMemoryCeilingBytes: UInt64(machineMlxMemoryCeilingBytes),
            effectiveMlxMemoryCeilingBytes: UInt64(effectiveMlxMemoryCeilingBytes),
            minimumMlxMemoryCeilingBytes: minimumMlxMemoryCeilingBytes));
        try eventWriter.sendEvent(.runtimeFeatureConfigurationApplied(acknowledgedFeatureConfiguration));

        var idleWorker: IdleWorkerState = IdleWorkerState(
            machineMlxMemoryCeilingBytes: UInt64(machineMlxMemoryCeilingBytes),
            effectiveMlxMemoryCeilingBytes: UInt64(effectiveMlxMemoryCeilingBytes),
            minimumMlxMemoryCeilingBytes: minimumMlxMemoryCeilingBytes,
            configurationGeneration: workerStartupConfiguration.configurationGeneration);
        while true {
            guard let workerCommand: WorkerCommand = try commandReader.nextCommand() else {
                return;
            }
            try idleWorker.serve(workerCommand, eventWriter: eventWriter);
        }
    }
}

/// The mutable state of the model-less worker: its live memory ceilings and
/// the configuration generation the last accepted memory policy carried.
private struct IdleWorkerState {

    let machineMlxMemoryCeilingBytes: UInt64;
    var effectiveMlxMemoryCeilingBytes: UInt64;
    let minimumMlxMemoryCeilingBytes: UInt64;
    var configurationGeneration: String;

    mutating func serve(
        _ workerCommand: WorkerCommand,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        switch (workerCommand) {
        case .initializeWorker:
            // A duplicate startup policy is ignored, exactly as the Rust idle
            // loop does: startup happens once per process.
            return;
        case let .generate(generationCommand):
            return try IdleWorkerState.rejectChatGeneration(
                generationCommand.requestId,
                eventWriter: eventWriter);
        case let .generateImage(imageGenerationCommand):
            return try IdleWorkerState.rejectImageGeneration(
                imageGenerationCommand.requestId,
                eventWriter: eventWriter);
        case .generateEmbeddings(let embeddingsCommand):
            return try IdleWorkerState.rejectEmbeddings(
                embeddingsCommand.requestId,
                eventWriter: eventWriter);
        case .cancel:
            return;
        case .sampleMlxMemory:
            // No MLX allocator exists in the model-less worker, so the sample
            // carries no observation; it still clears supervisor-side state.
            return try eventWriter.sendEvent(.mlxMemorySample(
                mlxMemorySnapshot: nil,
                expertResidency: nil));
        case let .updateMlxMemoryLimit(requestedMlxMemoryCeilingBytes, configurationGeneration):
            return try self.serveUpdateMlxMemoryLimit(
                requestedMlxMemoryCeilingBytes,
                configurationGeneration: configurationGeneration,
                eventWriter: eventWriter);
        case .swapModel:
            // No family runtime is linked in this worker build, so every swap
            // fails while the worker stays responsive and model-less.
            return try eventWriter.sendEvent(.modelSwapFailed(
                loadedModelRemainsReady: false,
                modelLoadFailureReason: "this worker build loads no model families yet"));
        case let .clearPromptCache(modelId):
            // No persistent cache exists in the model-less worker; the
            // acknowledgement reports the empty deletion.
            return try eventWriter.sendEvent(.promptCacheCleared(
                modelId: modelId,
                blocksRemoved: 0,
                bytesFreed: 0));
        }
    }

    private mutating func serveUpdateMlxMemoryLimit(
        _ requestedMlxMemoryCeilingBytes: UInt64,
        configurationGeneration: String,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        if requestedMlxMemoryCeilingBytes == 0
            || requestedMlxMemoryCeilingBytes > self.machineMlxMemoryCeilingBytes {
            return try eventWriter.sendEvent(.mlxMemoryLimitRejected(
                requestedMlxMemoryCeilingBytes: requestedMlxMemoryCeilingBytes,
                minimumMlxMemoryCeilingBytes: self.minimumMlxMemoryCeilingBytes,
                machineMlxMemoryCeilingBytes: self.machineMlxMemoryCeilingBytes,
                reason: "requested memory ceiling is outside the worker machine limit"));
        }
        self.effectiveMlxMemoryCeilingBytes = requestedMlxMemoryCeilingBytes;
        self.configurationGeneration = configurationGeneration;
        return try eventWriter.sendEvent(.mlxMemoryLimitChanged(
            effectiveMlxMemoryCeilingBytes: requestedMlxMemoryCeilingBytes,
            minimumMlxMemoryCeilingBytes: self.minimumMlxMemoryCeilingBytes,
            expertMemoryMode: .resident,
            mlxMemorySnapshot: nil,
            expertResidency: nil));
    }

    private static func rejectChatGeneration(
        _ requestId: RequestId,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        return try eventWriter.sendEvent(.failed(
            requestId: requestId,
            reason: .invalidRequest(reason: "the loaded model does not support chat generation")));
    }

    private static func rejectImageGeneration(
        _ requestId: RequestId,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        try eventWriter.sendEvent(.imageGenerationFailed(
            requestId: requestId,
            reason: .modelDoesNotSupportImageGeneration));
        return try eventWriter.sendEvent(.imageGenerationFinalized(
            requestId: requestId,
            elapsedMillis: 0,
            mlxMemorySnapshot: nil));
    }

    private static func rejectEmbeddings(
        _ requestId: RequestId,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        try eventWriter.sendEvent(.embeddingsFailed(
            requestId: requestId,
            reason: .fatalExecution(reason: "the loaded model does not support embeddings")));
        return try eventWriter.sendEvent(.embeddingsFinalized(
            requestId: requestId,
            elapsedMillis: 0,
            mlxMemorySnapshot: nil));
    }
}
