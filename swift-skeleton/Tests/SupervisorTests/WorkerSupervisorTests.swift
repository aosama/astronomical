import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import JourneyCategories;

@testable import Supervisor;

/**
 * Hermetic journeys for the supervisor's worker ownership: launch with a
 * bootstrapped child, serialized chat execution over the command pipe, the
 * swap-on-demand path, containment, and shutdown. The fake workers are short
 * shell scripts that emit genuine framed events over real pipes.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class WorkerSupervisorTests {

    @Test
    func should_launch_to_a_ready_model_less_worker_through_the_startup_acknowledgement() throws {
        let supervisor: WorkerSupervisor = try WorkerSupervisor.launch(
            workerExecutablePath: "/bin/bash",
            workerArguments: ["-c", WorkerSupervisorTests.modelLessWorkerScript()],
            workerStartupConfiguration: WorkerSupervisorTests.startupConfiguration(),
            modelPolicyCatalog: [:],
            modelLoadTimeout: 10);
        defer { _ = try? supervisor.shutdown() }

        #expect(supervisor.workerHealthSnapshot().status == .ready);
        #expect(supervisor.workerHealthSnapshot().readyModelId == nil);
        #expect(supervisor.workerHealthSnapshot().machineMlxMemoryCeilingBytes == 17179869184);
        #expect(supervisor.workerHealthSnapshot().effectiveMlxMemoryCeilingBytes == 8589934592);
        #expect(supervisor.workerHealthSnapshot().workerRuntimeFeatureConfiguration?.configurationGeneration == "gen-1");
        #expect(supervisor.isAvailable());
    }

    @Test
    func should_relay_a_worker_request_failure_to_the_caller() throws {
        let residentModelScript: String = FakeWorkerEventEmitter.frameEmitterFunction()
            + FakeWorkerEventEmitter.emitLine(payload: WorkerSupervisorTests.readyEventPayload(modelId: "m123"))
            + FakeWorkerEventEmitter.emitLine(payload: WorkerSupervisorTests.loadedRuntimePolicyPayload(modelId: "m123"))
            + FakeWorkerEventEmitter.emitLine(payload: WorkerSupervisorTests.failedEventPayload(
                requestId: 3,
                reason: "engine cold"))
            + "read _\nexit 0\n";
        let supervisor: WorkerSupervisor = try WorkerSupervisor.launch(
            workerExecutablePath: "/bin/bash",
            workerArguments: ["-c", residentModelScript],
            workerStartupConfiguration: WorkerSupervisorTests.startupConfiguration(),
            modelPolicyCatalog: [
                "m123": WorkerSupervisorTests.modelPolicy(modelId: "m123"),
            ],
            modelLoadTimeout: 10);
        defer { _ = try? supervisor.shutdown() }
        #expect(supervisor.workerHealthSnapshot().readyModelId == "m123");

        let streamEvents: Array<ChatGenerationStreamEvent> = try supervisor.startChatGeneration(
            WorkerSupervisorTests.chatGenerationCommand(requestId: 3, modelId: "m123"));

        #expect(streamEvents == [
            .failed(reason: .invalidRequest(reason: "engine cold")),
        ]);
    }

    @Test
    func should_map_a_rejected_swap_on_demand_to_a_model_load_failure() throws {
        let swapRejectingScript: String = FakeWorkerEventEmitter.frameEmitterFunction()
            + FakeWorkerEventEmitter.emitLine(payload: FakeWorkerEventEmitter.idleEventPayload())
            + FakeWorkerEventEmitter.emitLine(payload: FakeWorkerEventEmitter.modelLessRuntimePolicyPayload())
            + FakeWorkerEventEmitter.emitLine(payload: WorkerSupervisorTests.modelSwapFailedPayload(
                modelLoadFailureReason: "artifact unsupported"))
            + "read _\nexit 0\n";
        let supervisor: WorkerSupervisor = try WorkerSupervisor.launch(
            workerExecutablePath: "/bin/bash",
            workerArguments: ["-c", swapRejectingScript],
            workerStartupConfiguration: WorkerSupervisorTests.startupConfiguration(),
            modelPolicyCatalog: [
                "m9": WorkerSupervisorTests.modelPolicy(modelId: "m9"),
            ],
            modelLoadTimeout: 10);

        do {
            _ = try supervisor.startChatGeneration(
                WorkerSupervisorTests.chatGenerationCommand(requestId: 5, modelId: "m9"));
            Issue.record("expected a model load failure");
        } catch let thrownError as GenerationStartError {
            #expect(thrownError == .modelLoadFailed(modelLoadFailureReason: "artifact unsupported"));
        }
        // A rejected first load leaves a healthy model-less worker.
        #expect(supervisor.workerHealthSnapshot().status == .ready);
        #expect(supervisor.workerHealthSnapshot().readyModelId == nil);
        _ = try? supervisor.shutdown()
    }

    @Test
    func should_reject_generation_for_an_unmapped_model_without_a_resident_model() throws {
        let supervisor: WorkerSupervisor = try WorkerSupervisor.launch(
            workerExecutablePath: "/bin/bash",
            workerArguments: ["-c", WorkerSupervisorTests.modelLessWorkerScript()],
            workerStartupConfiguration: WorkerSupervisorTests.startupConfiguration(),
            modelPolicyCatalog: [:],
            modelLoadTimeout: 10);
        defer { _ = try? supervisor.shutdown() }

        do {
            _ = try supervisor.startChatGeneration(
                WorkerSupervisorTests.chatGenerationCommand(requestId: 6, modelId: "mystery"));
            Issue.record("expected a worker-unavailable rejection");
        } catch let thrownError as GenerationStartError {
            #expect(thrownError == .workerUnavailable);
        }
    }

    @Test
    func should_surface_a_start_worker_error_when_the_executable_is_missing() throws {
        do {
            _ = try WorkerSupervisor.launch(
                workerExecutablePath: "/nonexistent/astronomical-worker",
                workerArguments: [],
                workerStartupConfiguration: WorkerSupervisorTests.startupConfiguration(),
                modelPolicyCatalog: [:],
                modelLoadTimeout: 10);
            Issue.record("expected a start-worker failure");
        } catch let thrownError as WorkerControlError {
            guard case .startWorker = thrownError else {
                Issue.record("expected a start-worker failure, got \(thrownError)");
                return;
            }
        }
    }

    @Test
    func should_close_a_worker_that_exits_at_end_of_stream_on_shutdown() throws {
        let supervisor: WorkerSupervisor = try WorkerSupervisor.launch(
            workerExecutablePath: "/bin/bash",
            workerArguments: ["-c", WorkerSupervisorTests.modelLessWorkerScript()],
            workerStartupConfiguration: WorkerSupervisorTests.startupConfiguration(),
            modelPolicyCatalog: [:],
            modelLoadTimeout: 10);

        let terminationOutcome: WorkerTerminationOutcome = try supervisor.shutdown();
        #expect(terminationOutcome == .graceful(processExitSuccessful: true));
        #expect(supervisor.isAvailable() == false);
    }

    @Test
    func should_queue_a_concurrent_generation_and_abandon_it_at_shutdown() throws {
        // The fake worker never answers the swap, so the first generate stays
        // inside its bounded model-load wait while the second request queues
        // behind it.
        let silentWorkerScript: String = FakeWorkerEventEmitter.frameEmitterFunction()
            + FakeWorkerEventEmitter.emitLine(payload: FakeWorkerEventEmitter.idleEventPayload())
            + FakeWorkerEventEmitter.emitLine(payload: FakeWorkerEventEmitter.modelLessRuntimePolicyPayload())
            + "read _\nexit 0\n";
        let supervisor: WorkerSupervisor = try WorkerSupervisor.launch(
            workerExecutablePath: "/bin/bash",
            workerArguments: ["-c", silentWorkerScript],
            workerStartupConfiguration: WorkerSupervisorTests.startupConfiguration(),
            modelPolicyCatalog: [
                "m1": WorkerSupervisorTests.modelPolicy(modelId: "m1"),
            ],
            modelLoadTimeout: 4);
        defer { _ = try? supervisor.shutdown() }

        let blockedOutcome: GenerationJourneyOutcome = GenerationJourneyOutcome(workerThread: Thread());
        let blockedGeneration: Thread = Thread {
            do {
                _ = try supervisor.startChatGeneration(
                    WorkerSupervisorTests.chatGenerationCommand(requestId: 8, modelId: "m1"));
            } catch {
                // The bounded model-load wait or the shutdown close ends this
                // request; its outcome is not the subject of this journey.
            }
        };
        blockedGeneration.name = "blocked-generate";
        blockedOutcome.workerThread = blockedGeneration;
        blockedGeneration.start();
        // Let the blocked generation take the admission slot before queueing.
        Thread.sleep(forTimeInterval: 0.5);

        let queuedOutcome: GenerationJourneyOutcome = GenerationJourneyOutcome(workerThread: Thread());
        let queuedGeneration: Thread = Thread {
            do {
                _ = try supervisor.startChatGeneration(
                    WorkerSupervisorTests.chatGenerationCommand(requestId: 9, modelId: "m1"));
                queuedOutcome.record(streamEvents: []);
            } catch {
                queuedOutcome.record(error: error);
            }
        };
        queuedGeneration.name = "queued-generate";
        queuedOutcome.workerThread = queuedGeneration;
        queuedGeneration.start();
        Thread.sleep(forTimeInterval: 0.5);

        // The queued request must not wedge the supervisor: shutdown wakes
        // it, waits a bounded time for the active request, and still closes
        // the worker.
        let terminationOutcome: WorkerTerminationOutcome = try supervisor.shutdown();
        #expect(supervisor.isAvailable() == false);
        WorkerSupervisorTests.join(blockedGeneration, deadline: Date().addingTimeInterval(10));
        WorkerSupervisorTests.join(queuedGeneration, deadline: Date().addingTimeInterval(10));
        // The queued waiter abandons its ticket once shutdown is visible.
        #expect(queuedOutcome.thrownErrorAsGenerationStart == .workerUnavailable);
        // The blocked request's containment force-terminated the fake before
        // shutdown ran, so shutdown finds no living process and reports a
        // graceful close.
        #expect(terminationOutcome == .graceful(processExitSuccessful: true));
    }

    // MARK: - Fake worker payloads

    private static func modelLessWorkerScript() -> String {
        return FakeWorkerEventEmitter.frameEmitterFunction()
            + FakeWorkerEventEmitter.emitLine(payload: FakeWorkerEventEmitter.idleEventPayload())
            + FakeWorkerEventEmitter.emitLine(payload: FakeWorkerEventEmitter.modelLessRuntimePolicyPayload())
            + "read _\nexit 0\n";
    }

    private static func readyEventPayload(modelId: String) -> String {
        return FakeWorkerEventEmitter.residentReadyEventPayload(modelId: modelId);
    }

    private static func loadedRuntimePolicyPayload(modelId: String) -> String {
        return FakeWorkerEventEmitter.residentRuntimePolicyPayload(modelId: modelId);
    }

    private static func failedEventPayload(requestId: UInt64, reason: String) -> String {
        return "{\"kind\":\"failed\",\"request_id\":\(requestId),"
            + "\"reason\":{\"invalid_request\":{\"reason\":\"\(reason)\"}}}";
    }

    private static func modelSwapFailedPayload(modelLoadFailureReason: String) -> String {
        return "{\"kind\":\"model_swap_failed\",\"loaded_model_remains_ready\":false,"
            + "\"model_load_failure_reason\":\"\(modelLoadFailureReason)\"}";
    }

    private static func join(_ workerThread: Thread, deadline: Date) -> Void {
        while workerThread.isFinished == false && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.01);
        }
    }

    // MARK: - Fixtures

    private static func startupConfiguration() -> WorkerStartupConfiguration {
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

    private static func chatGenerationCommand(requestId: UInt64, modelId: String) -> ChatGenerationCommand {
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
            structuredGeneration: nil);
    }

    private static func modelPolicy(modelId: String) -> RuntimeModelPolicy {
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
}
