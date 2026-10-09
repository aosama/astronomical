import Foundation;

import IpcProtocol;
import ModelServing;

/// The worker command loop entry point: bootstrap and delegation into the
/// engine-backed worker.
///
/// Mirrors apps/inference-worker/src/worker_startup.rs: the first command
/// must carry the startup policy; end of the command stream is the clean
/// exit. A nil chat-runtime factory keeps the worker model-less (the
/// hermetic loop journeys exercise exactly that behavior); the process
/// entry point passes the real family factory.
public enum WorkerCommandLoop {

    private static let initializeWorkerNotFirstDescription: String =
        "expected InitializeWorker as the first worker command";

    /// Runs the bootstrapped worker over the framed transports and returns
    /// when the supervisor closes the command side.
    public static func runBootstrappedWorker(
        readTransport: any FrameTransport,
        writeTransport: any FrameTransport,
        chatRuntimeFactoryProvider: ((WorkerStartupConfiguration) -> (any ChatModelRuntimeFactory)?)? = nil
    ) throws -> Void {
        let commandReader: ProtocolReader = ProtocolReader(transport: readTransport);
        let eventWriter: ProtocolWriter = ProtocolWriter(transport: writeTransport);
        guard case let .initializeWorker(workerStartupConfiguration) =
            try commandReader.nextCommand() else {
            throw WorkerProcessError.startup(.initializeWorkerNotFirst(
                description: WorkerCommandLoop.initializeWorkerNotFirstDescription));
        }
        // The GPU ceiling sample happens here so a sampler failure still
        // fails the worker startup exactly as the previous loop did.
        let sampledMachineMlxMemoryCeilingBytes: Int =
            try GpuWiredMemoryLimit.sampleIogpuWiredLimitBytes();
        // Production builds the real family factory bound to the policy's
        // attribution switch; journeys inject their own provider.
        let persistentPromptCachePolicy: Qwen35MoePromptCacheSpawnPolicy?;
        if workerStartupConfiguration.persistentPromptCacheEnabled {
            persistentPromptCachePolicy = Qwen35MoePromptCacheSpawnPolicy(
                globalPromptCacheRootDirectory: URL(fileURLWithPath:
                    workerStartupConfiguration.globalPromptCacheRootDirectory),
                globalPromptCacheMaximumSizeBytes:
                    workerStartupConfiguration.globalPromptCacheMaximumSizeBytes,
                effectiveMlxMemoryCeilingBytes: UInt64(
                    GpuWiredMemoryLimit.resolveEffectiveMlxMemoryCeilingBytes(
                        configuredMlxMemoryCeilingBytes:
                            workerStartupConfiguration.configuredMaximumMlxMemoryBytes,
                        machineMlxMemoryCeilingBytes:
                            sampledMachineMlxMemoryCeilingBytes)));
        } else {
            persistentPromptCachePolicy = nil;
        }
        let chatRuntimeFactory: (any ChatModelRuntimeFactory)? =
            chatRuntimeFactoryProvider?(workerStartupConfiguration)
            ?? ModelFamilyFactory(
                performanceAttributionEnabled:
                    workerStartupConfiguration.performanceAttributionEnabled,
                persistentPromptCachePolicy: persistentPromptCachePolicy);
        let engineBackedWorker: EngineBackedWorker = EngineBackedWorker(
            chatRuntimeFactory: chatRuntimeFactory,
            sampleMachineMlxMemoryCeilingBytes: {
                return sampledMachineMlxMemoryCeilingBytes;
            },
            resolveEffectiveCeilingBytes: { (configuredMlxMemoryCeilingBytes, machineMlxMemoryCeilingBytes) in
                return GpuWiredMemoryLimit.resolveEffectiveMlxMemoryCeilingBytes(
                    configuredMlxMemoryCeilingBytes: configuredMlxMemoryCeilingBytes,
                    machineMlxMemoryCeilingBytes: machineMlxMemoryCeilingBytes);
            });
        return try engineBackedWorker.run(
            workerStartupConfiguration: workerStartupConfiguration,
            commandReader: commandReader, eventWriter: eventWriter);
    }
}
