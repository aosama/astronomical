import Foundation;

import AstronomicalConfig;
import IpcProtocol;

@testable import Supervisor;

/**
 * Shared marker-driven fake worker and supervisor harness for the
 * supervisor journey suites. The fake worker is a short `/bin/bash` script
 * with a resident model acknowledged at startup; journeys steer it through
 * marker files in a temporary control directory: one marker completes the
 * pending generation with a chosen request id, one acknowledges a memory
 * raise, and one acknowledges a prompt-cache clear with a chosen scope.
 */
final class FakeWorkerJourneyHarness: @unchecked Sendable {

    let supervisor: WorkerSupervisor;
    private let controlDirectoryPath: String;

    private init(supervisor: WorkerSupervisor, controlDirectoryPath: String) {
        self.supervisor = supervisor;
        self.controlDirectoryPath = controlDirectoryPath;
    }

    static func launch(residentModelId: String = "m1") throws -> FakeWorkerJourneyHarness {
        let controlDirectoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("astronomical-journey-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(at: controlDirectoryUrl, withIntermediateDirectories: true);
        let controlDirectoryPath: String = controlDirectoryUrl.path;
        let supervisor: WorkerSupervisor = try WorkerSupervisor.launch(
            workerExecutablePath: "/bin/bash",
            workerArguments: ["-c", FakeWorkerJourneyHarness.controlGatedWorkerScript(
                controlDirectoryPath: controlDirectoryPath,
                residentModelId: residentModelId)],
            workerStartupConfiguration: FakeWorkerJourneyHarness.startupConfiguration(),
            modelPolicyCatalog: [
                residentModelId: FakeWorkerJourneyHarness.modelPolicy(modelId: residentModelId),
            ],
            modelLoadTimeout: 10);
        // The resident ready event zeroes the memory ceilings, so the
        // journey publishes the fixture worker's startup probe (40 GB
        // machine, 8 GB effective, 1 byte minimum) directly into health
        // state, exactly what the Rust idle fixture reports.
        try supervisor.ownedHealthState().apply({ (snapshot: inout WorkerHealthSnapshot) in
            snapshot.machineMlxMemoryCeilingBytes = 40_000_000_000;
            snapshot.effectiveMlxMemoryCeilingBytes = 8_000_000_000;
            snapshot.minimumMlxMemoryCeilingBytes = 1;
        });
        return FakeWorkerJourneyHarness(
            supervisor: supervisor,
            controlDirectoryPath: controlDirectoryPath);
    }

    func dispose() -> Void {
        _ = try? self.supervisor.shutdown();
        try? FileManager.default.removeItem(atPath: self.controlDirectoryPath);
    }

    // MARK: Generation journeys

    func startGenerationThread(requestId: UInt64) -> GenerationJourneyOutcome {
        let outcome: GenerationJourneyOutcome = GenerationJourneyOutcome(workerThread: Thread());
        let supervisor: WorkerSupervisor = self.supervisor;
        let generationThread: Thread = Thread {
            do {
                outcome.record(streamEvents: try supervisor.startChatGeneration(
                    FakeWorkerJourneyHarness.chatGenerationCommand(
                        requestId: requestId,
                        modelId: "m1")));
            } catch {
                outcome.record(error: error);
            }
        };
        generationThread.name = "journey-generation-\(requestId)";
        outcome.workerThread = generationThread;
        generationThread.start();
        return outcome;
    }

    // MARK: Marker pokes

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

    /// Arms the next memory-ceiling command to answer with a rejection, so
    /// journeys can steer the worker's refusal of a raised ceiling.
    func pokeMemoryRaiseRejection(_ requestedMlxMemoryCeilingBytes: UInt64) throws -> Void {
        try String(requestedMlxMemoryCeilingBytes).write(
            toFile: self.controlDirectoryPath + "/reject_memory_raise",
            atomically: true,
            encoding: .utf8);
    }

    /// Arms the next prompt-cache clear acknowledgement. A `nil` scope
    /// acknowledges the global clear; any other scope is echoed as the
    /// cleared `model_id`, so mismatch journeys can arm a wrong scope.
    func pokeCacheClearAck(
        scopeModelId: String?,
        blocksRemoved: UInt64,
        bytesFreed: UInt64
    ) throws -> Void {
        let scopeField: String = scopeModelId ?? "global";
        let ackContent: String = "\(scopeField)|\(blocksRemoved)|\(bytesFreed)";
        try ackContent.write(
            toFile: self.controlDirectoryPath + "/cache_clear_ack",
            atomically: true,
            encoding: .utf8);
    }

    // MARK: Bounded waits

    /// Blocks until the issued admission tickets reach the expected
    /// outstanding count, proving the just-started requests are queued (or
    /// active) before the journey proceeds.
    func awaitQueueFill(expectedOutstandingCount: Int) -> Void {
        let fillDeadline: Date = Date().addingTimeInterval(5);
        while supervisor.outstandingAdmissionTicketCount < expectedOutstandingCount {
            if Date() >= fillDeadline {
                assertionFailure("the queue never reached \(expectedOutstandingCount) outstanding requests");
                return;
            }
            Thread.sleep(forTimeInterval: 0.01);
        }
    }

    // MARK: Fake worker script and fixtures

    /// The fake worker: startup acknowledges the resident model, then the
    /// polling loop answers whichever markers the journey armed.
    private static func controlGatedWorkerScript(
        controlDirectoryPath: String,
        residentModelId: String
    ) -> String {
        return FakeWorkerEventEmitter.frameEmitterFunction()
            + FakeWorkerEventEmitter.emitLine(payload: FakeWorkerEventEmitter.residentReadyEventPayload(
                modelId: residentModelId))
            + FakeWorkerEventEmitter.emitLine(payload: FakeWorkerEventEmitter.residentRuntimePolicyPayload(
                modelId: residentModelId))
            + "while true; do\n"
            + "  if [ -f \"\(controlDirectoryPath)/complete_request\" ]; then\n"
            + "    completion_request_id=$(cat \"\(controlDirectoryPath)/complete_request\")\n"
            + "    rm -f \"\(controlDirectoryPath)/complete_request\"\n"
            + "    emit_frame '{\"kind\":\"completed\",\"request_id\":'\"$completion_request_id\"',"
            + "\"prompt_token_count\":1,\"generated_token_count\":1,\"reasoning_token_count\":0,"
            + "\"cached_token_count\":0,\"persistent_prompt_cache_diagnostics\":null,"
            + "\"reason\":\"end_of_sequence\"}'\n"
            + "  fi\n"
            + "  if [ -f \"\(controlDirectoryPath)/apply_memory_raise\" ]; then\n"
            + "    raised_ceiling_bytes=$(cat \"\(controlDirectoryPath)/apply_memory_raise\")\n"
            + "    rm -f \"\(controlDirectoryPath)/apply_memory_raise\"\n"
            + "    emit_frame '{\"kind\":\"mlx_memory_limit_changed\","
            + "\"effective_mlx_memory_ceiling_bytes\":'\"$raised_ceiling_bytes\"',"
            + "\"minimum_mlx_memory_ceiling_bytes\":1,\"expert_memory_mode\":\"resident\","
            + "\"mlx_memory_snapshot\":null,\"expert_residency\":null}'\n"
            + "  fi\n"
            + "  if [ -f \"\(controlDirectoryPath)/reject_memory_raise\" ]; then\n"
            + "    rejected_ceiling_bytes=$(cat \"\(controlDirectoryPath)/reject_memory_raise\")\n"
            + "    rm -f \"\(controlDirectoryPath)/reject_memory_raise\"\n"
            + "    emit_frame '{\"kind\":\"mlx_memory_limit_rejected\","
            + "\"requested_mlx_memory_ceiling_bytes\":'\"$rejected_ceiling_bytes\"',"
            + "\"minimum_mlx_memory_ceiling_bytes\":1,"
            + "\"machine_mlx_memory_ceiling_bytes\":40000000000,"
            + "\"reason\":\"fixture rejected the memory raise\"}'\n"
            + "  fi\n"
            + "  if [ -f \"\(controlDirectoryPath)/cache_clear_ack\" ]; then\n"
            + "    clear_ack_content=$(cat \"\(controlDirectoryPath)/cache_clear_ack\")\n"
            + "    rm -f \"\(controlDirectoryPath)/cache_clear_ack\"\n"
            + "    clear_scope=$(printf '%s' \"$clear_ack_content\" | cut -d'|' -f1)\n"
            + "    clear_blocks=$(printf '%s' \"$clear_ack_content\" | cut -d'|' -f2)\n"
            + "    clear_bytes=$(printf '%s' \"$clear_ack_content\" | cut -d'|' -f3)\n"
            + "    if [ \"$clear_scope\" = \"global\" ]; then\n"
            + "      cleared_model_json='\"model_id\":null'\n"
            + "    else\n"
            + "      cleared_model_json=\"\\\"model_id\\\":\\\"$clear_scope\\\"\"\n"
            + "    fi\n"
            + "    emit_frame \"{\\\"kind\\\":\\\"prompt_cache_cleared\\\",$cleared_model_json,"
            + "\\\"blocks_removed\\\":$clear_blocks,\\\"bytes_freed\\\":$clear_bytes}\"\n"
            + "  fi\n"
            + "  sleep 0.02\n"
            + "done\n";
    }

    static func chatGenerationCommand(requestId: UInt64, modelId: String) -> ChatGenerationCommand {
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

    static func modelPolicy(modelId: String) -> RuntimeModelPolicy {
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

    static func startupConfiguration() -> WorkerStartupConfiguration {
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
