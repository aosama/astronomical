import Testing;

import Foundation;

import IpcProtocol;
import JourneyCategories;

@testable import Supervisor;

/// Hermetic coverage of the worker startup handshake: the InitializeWorker
/// write at launch, the bounded wait for the runtime-policy acknowledgement,
/// and the process-scoped event handling that feeds real worker state into
/// the daemon status verb. The fake workers are short shell scripts that emit
/// genuine framed events over a real pipe pair.
@Suite(.serialized, .tags(.hermeticJourney))
final class WorkerStartupHandshakeTests {

    @Test
    func should_start_health_state_loading_and_reach_ready_through_lifecycle_events() throws {
        let healthState: WorkerHealthState = WorkerHealthState();
        #expect(
            healthState.daemonStatusReport()
                == DaemonStatusReport(workerStatus: .loading, readyModelId: nil));

        try WorkerEventHandler.handle(
            .idle(
                machineMlxMemoryCeilingBytes: 17179869184,
                effectiveMlxMemoryCeilingBytes: 8589934592,
                minimumMlxMemoryCeilingBytes: 1),
            healthState: healthState);
        #expect(
            healthState.daemonStatusReport()
                == DaemonStatusReport(workerStatus: .ready, readyModelId: nil));
        #expect(
            healthState.currentSnapshot().machineMlxMemoryCeilingBytes
                == 17179869184);

        do {
            try WorkerEventHandler.handle(
                .idle(
                    machineMlxMemoryCeilingBytes: 1,
                    effectiveMlxMemoryCeilingBytes: 1,
                    minimumMlxMemoryCeilingBytes: 1),
                healthState: healthState);
            Issue.record("expected a protocol violation");
        } catch let workerControlError as WorkerControlError {
            guard case let .workerProtocolViolation(description) = workerControlError else {
                Issue.record(Comment(stringLiteral: "expected a protocol violation, got \(workerControlError)"));
                return;
            }
            #expect(description == "duplicate worker idle event");
        }
    }

    @Test
    func should_publish_the_model_on_the_ready_event_and_reject_duplicates() throws {
        let healthState: WorkerHealthState = WorkerHealthState();
        try WorkerEventHandler.handle(
            .ready(modelId: "qwen3-model", capabilities: WorkerStartupHandshakeTests.chatCapabilities()),
            healthState: healthState);
        #expect(
            healthState.daemonStatusReport()
                == DaemonStatusReport(workerStatus: .ready, readyModelId: "qwen3-model"));
        #expect(
            healthState.currentSnapshot().readyModelCapabilities
                == WorkerStartupHandshakeTests.chatCapabilities());

        do {
            try WorkerEventHandler.handle(
                .ready(modelId: "qwen3-model", capabilities: WorkerStartupHandshakeTests.chatCapabilities()),
                healthState: healthState);
            Issue.record("expected a protocol violation");
        } catch let workerControlError as WorkerControlError {
            guard case let .workerProtocolViolation(description) = workerControlError else {
                Issue.record(Comment(stringLiteral: "expected a protocol violation, got \(workerControlError)"));
                return;
            }
            #expect(description == "duplicate worker readiness");
        }
    }

    @Test
    func should_require_the_runtime_policy_acknowledgement_to_match_the_published_model() throws {
        let healthState: WorkerHealthState = WorkerHealthState();
        try WorkerEventHandler.handle(
            .ready(modelId: "qwen3-model", capabilities: WorkerStartupHandshakeTests.chatCapabilities()),
            healthState: healthState);

        let foreignAcknowledgement: WorkerRuntimeFeatureConfiguration = WorkerRuntimeFeatureConfiguration(
            configurationGeneration: "gen-1",
            persistentPromptCacheEnabled: true,
            promptCacheMaximumSizeBytes: 1073741824,
            loadedModel: WorkerStartupHandshakeTests.loadedModel(modelId: "other-model"));
        do {
            try WorkerEventHandler.handle(
                .runtimeFeatureConfigurationApplied(foreignAcknowledgement),
                healthState: healthState);
            Issue.record("expected a protocol violation");
        } catch let workerControlError as WorkerControlError {
            guard case let .workerProtocolViolation(description) = workerControlError else {
                Issue.record(Comment(stringLiteral: "expected a protocol violation, got \(workerControlError)"));
                return;
            }
            #expect(
                description
                    == "runtime policy acknowledgement does not match the published model");
        }

        let matchingAcknowledgement: WorkerRuntimeFeatureConfiguration = WorkerRuntimeFeatureConfiguration(
            configurationGeneration: "gen-1",
            persistentPromptCacheEnabled: true,
            promptCacheMaximumSizeBytes: 1073741824,
            loadedModel: WorkerStartupHandshakeTests.loadedModel(modelId: "qwen3-model"));
        try WorkerEventHandler.handle(
            .runtimeFeatureConfigurationApplied(matchingAcknowledgement),
            healthState: healthState);
        #expect(
            healthState.currentSnapshot().workerRuntimeFeatureConfiguration
                == matchingAcknowledgement);
        #expect(healthState.hasRuntimeFeatureConfiguration());

        // A refreshed generation with an identical loaded model is the legal
        // live-memory-update shape; a changed loaded-model payload without a
        // model transition is not.
        let silentlyChangedPolicy: WorkerRuntimeFeatureConfiguration = WorkerRuntimeFeatureConfiguration(
            configurationGeneration: "gen-2",
            persistentPromptCacheEnabled: true,
            promptCacheMaximumSizeBytes: 1073741824,
            loadedModel: WorkerStartupHandshakeTests.loadedModel(modelId: "qwen3-model", maximumOutputTokens: 2048));
        do {
            try WorkerEventHandler.handle(
                .runtimeFeatureConfigurationApplied(silentlyChangedPolicy),
                healthState: healthState);
            Issue.record("expected a protocol violation");
        } catch let workerControlError as WorkerControlError {
            guard case let .workerProtocolViolation(description) = workerControlError else {
                Issue.record(Comment(stringLiteral: "expected a protocol violation, got \(workerControlError)"));
                return;
            }
            #expect(
                description
                    == "runtime policy changed without an atomic model transition");
        }

        let refreshedGeneration: WorkerRuntimeFeatureConfiguration = WorkerRuntimeFeatureConfiguration(
            configurationGeneration: "gen-2",
            persistentPromptCacheEnabled: true,
            promptCacheMaximumSizeBytes: 1073741824,
            loadedModel: WorkerStartupHandshakeTests.loadedModel(modelId: "qwen3-model"));
        try WorkerEventHandler.handle(
            .runtimeFeatureConfigurationApplied(refreshedGeneration),
            healthState: healthState);
        #expect(
            healthState.currentSnapshot().workerRuntimeFeatureConfiguration
                == refreshedGeneration);
    }

    @Test
    func should_publish_and_clear_memory_sample_observations() throws {
        let healthState: WorkerHealthState = WorkerHealthState();
        let memorySnapshot: WorkerMlxMemorySnapshot = WorkerStartupHandshakeTests.memorySnapshot();
        let expertResidency: WorkerExpertResidencySnapshot = WorkerExpertResidencySnapshot(
            totalLayerCount: 48,
            residentExpertCount: 12,
            residentExpertPayloadBytes: 1073741824);
        try WorkerEventHandler.handle(
            .mlxMemorySample(mlxMemorySnapshot: memorySnapshot, expertResidency: expertResidency),
            healthState: healthState);
        #expect(healthState.currentSnapshot().latestMlxMemorySnapshot == memorySnapshot);
        #expect(healthState.currentSnapshot().expertResidency == expertResidency);

        try WorkerEventHandler.handle(
            .mlxMemorySample(mlxMemorySnapshot: nil, expertResidency: nil),
            healthState: healthState);
        #expect(healthState.currentSnapshot().latestMlxMemorySnapshot == nil);
        // Residency is concrete topology and survives a cleared sample.
        #expect(healthState.currentSnapshot().expertResidency == expertResidency);
    }

    @Test
    func should_await_readiness_when_a_worker_launches_without_a_startup_configuration() throws {
        // A no-policy launch still consumes the fixture's idle event, exactly
        // as the Rust worker loop does; a silent worker times the wait out.
        let fakeWorkerScript: String = FakeWorkerEventEmitter.frameEmitterFunction()
            + FakeWorkerEventEmitter.emitLine(payload: FakeWorkerEventEmitter.idleEventPayload())
            + "exec sleep 30\n";
        let workerProcess: WorkerProcess = try WorkerProcess.launch(
            workerExecutablePath: "/bin/bash",
            arguments: ["-c", fakeWorkerScript]);
        defer {
            _ = try? workerProcess.close();
        }
        let healthState: WorkerHealthState = WorkerHealthState();
        let eventPump: WorkerEventPump = WorkerEventPump(workerProcess: workerProcess);
        try WorkerStartupRuntime.waitForStartupRuntimeConfiguration(
            workerProcess: workerProcess,
            eventPump: eventPump,
            healthState: healthState,
            modelLoadTimeout: 2);
        #expect(workerProcess.isStartupRuntimeConfigurationApplied());
        #expect(healthState.currentSnapshot().status == .ready);
        #expect(healthState.currentSnapshot().workerRuntimeFeatureConfiguration == nil);
    }

    @Test
    func should_complete_the_startup_wait_when_the_fake_worker_acknowledges_its_policy() throws {
        let fakeWorkerScript: String = FakeWorkerEventEmitter.frameEmitterFunction()
            + FakeWorkerEventEmitter.emitLine(payload: FakeWorkerEventEmitter.idleEventPayload())
            + FakeWorkerEventEmitter.emitLine(payload: FakeWorkerEventEmitter.modelLessRuntimePolicyPayload())
            + "exec sleep 30\n";
        let workerProcess: WorkerProcess = try WorkerProcess.launch(
            workerExecutablePath: "/bin/bash",
            arguments: ["-c", fakeWorkerScript],
            workerStartupConfiguration: WorkerStartupHandshakeTests.startupConfiguration());
        defer {
            _ = try? workerProcess.close();
        }
        let healthState: WorkerHealthState = WorkerHealthState();
        let eventPump: WorkerEventPump = WorkerEventPump(workerProcess: workerProcess);

        try WorkerStartupRuntime.waitForStartupRuntimeConfiguration(
            workerProcess: workerProcess,
            eventPump: eventPump,
            healthState: healthState,
            modelLoadTimeout: 10);

        #expect(workerProcess.isStartupRuntimeConfigurationApplied());
        #expect(workerProcess.expectedConfigurationGeneration() == "gen-1");
        #expect(
            healthState.daemonStatusReport()
                == DaemonStatusReport(workerStatus: .ready, readyModelId: nil));
        #expect(
            healthState.currentSnapshot().machineMlxMemoryCeilingBytes
                == 17179869184);
        #expect(
            healthState.currentSnapshot().workerRuntimeFeatureConfiguration?.configurationGeneration
                == "gen-1");
    }

    @Test
    func should_time_out_the_startup_wait_when_the_worker_never_acknowledges() throws {
        let workerProcess: WorkerProcess = try WorkerProcess.launch(
            workerExecutablePath: "/bin/bash",
            arguments: ["-c", "exec sleep 30\n"],
            workerStartupConfiguration: WorkerStartupHandshakeTests.startupConfiguration());
        defer {
            _ = try? workerProcess.close();
        }
        let eventPump: WorkerEventPump = WorkerEventPump(workerProcess: workerProcess);

        do {
            try WorkerStartupRuntime.waitForStartupRuntimeConfiguration(
                workerProcess: workerProcess,
                eventPump: eventPump,
                healthState: WorkerHealthState(),
                modelLoadTimeout: 1);
            Issue.record("expected a model-load timeout");
        } catch let workerControlError as WorkerControlError {
            guard case let .modelLoadTimeout(modelLoadTimeoutMillis) = workerControlError else {
                Issue.record(Comment(stringLiteral: "expected a model-load timeout, got \(workerControlError)"));
                return;
            }
            #expect(modelLoadTimeoutMillis == 1000);
        }
        #expect(!workerProcess.isStartupRuntimeConfigurationApplied());
    }

    @Test
    func should_surface_a_stream_closure_instead_of_a_timeout_from_the_startup_wait() throws {
        let fakeWorkerScript: String = FakeWorkerEventEmitter.frameEmitterFunction()
            + FakeWorkerEventEmitter.emitLine(payload: FakeWorkerEventEmitter.idleEventPayload());
        let workerProcess: WorkerProcess = try WorkerProcess.launch(
            workerExecutablePath: "/bin/bash",
            arguments: ["-c", fakeWorkerScript],
            workerStartupConfiguration: WorkerStartupHandshakeTests.startupConfiguration());
        defer {
            _ = try? workerProcess.close();
        }
        let eventPump: WorkerEventPump = WorkerEventPump(workerProcess: workerProcess);

        do {
            try WorkerStartupRuntime.waitForStartupRuntimeConfiguration(
                workerProcess: workerProcess,
                eventPump: eventPump,
                healthState: WorkerHealthState(),
                modelLoadTimeout: 10);
            Issue.record("expected a stream closure");
        } catch let workerControlError as WorkerControlError {
            // A real subprocess closing its output carries process
            // diagnostics, exactly as the Rust worker's next_event composes
            // them; the bare stream-closed case is reserved for non-process
            // fixtures.
            guard case WorkerControlError.workerProcessExited = workerControlError else {
                Issue.record(Comment(stringLiteral: "expected a stream closure, got \(workerControlError)"));
                return;
            }
        }
    }

    static func startupConfiguration() -> WorkerStartupConfiguration {
        return WorkerStartupConfiguration(
            configurationGeneration: "gen-1",
            globalPromptCacheRootDirectory: "/tmp/astronomical-prompt-cache",
            globalPromptCacheMaximumSizeBytes: 1073741824,
            persistentPromptCacheEnabled: true,
            configuredMaximumMlxMemoryBytes: nil,
            performanceAttributionEnabled: false,
            loggingDirectory: "/tmp/astronomical-worker-logs",
            loggingLevel: .info,
            retainedLogFileCount: 3);
    }

    private static func chatCapabilities() -> WorkerModelCapabilities {
        return WorkerModelCapabilities.from(chatCapabilities: ChatModelCapabilities(
            supportsReasoning: true,
            supportsToolCalls: true,
            hasVision: false,
            maxInputTokens: 4096,
            maxOutputTokens: 1024,
            contextWindow: 8192));
    }

    private static func loadedModel(
        modelId: String,
        maximumOutputTokens: UInt32 = 1024
    ) -> WorkerLoadedModelRuntimeConfiguration {
        return WorkerLoadedModelRuntimeConfiguration.autoregressive(
            WorkerLoadedAutoregressiveModelRuntimeConfiguration(
                modelId: modelId,
                maximumContextTokens: 8192,
                maximumOutputTokens: maximumOutputTokens,
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

    private static func memorySnapshot() -> WorkerMlxMemorySnapshot {
        return WorkerMlxMemorySnapshot(
            source: .idlePoll,
            activeMemoryBytes: 4294967296,
            allocatorCacheMemoryBytes: 268435456,
            peakMemoryBytes: 5368709120,
            expertPayloadBytes: 1073741824,
            modelCorePayloadBytes: 3221225472,
            contextStatePayloadBytes: 134217728,
            memoryCeilingUtilization: nil);
    }
}
