import XCTest;

import Foundation;

import IpcProtocol;

@testable import Supervisor;

/// Hermetic coverage of the worker startup handshake: the InitializeWorker
/// write at launch, the bounded wait for the runtime-policy acknowledgement,
/// and the process-scoped event handling that feeds real worker state into
/// the daemon status verb. The fake workers are short shell scripts that emit
/// genuine framed events over a real pipe pair.
final class WorkerStartupHandshakeTests: XCTestCase {

    func testHealthStateStartsLoadingAndReachesReadyThroughLifecycleEvents() throws {
        let healthState: WorkerHealthState = WorkerHealthState();
        XCTAssertEqual(
            healthState.daemonStatusReport(),
            DaemonStatusReport(workerStatus: .loading, readyModelId: nil));

        try WorkerEventHandler.handle(
            .idle(
                machineMlxMemoryCeilingBytes: 17179869184,
                effectiveMlxMemoryCeilingBytes: 8589934592,
                minimumMlxMemoryCeilingBytes: 1),
            healthState: healthState);
        XCTAssertEqual(
            healthState.daemonStatusReport(),
            DaemonStatusReport(workerStatus: .ready, readyModelId: nil));
        XCTAssertEqual(
            healthState.currentSnapshot().machineMlxMemoryCeilingBytes,
            17179869184);

        XCTAssertThrowsError(try WorkerEventHandler.handle(
            .idle(
                machineMlxMemoryCeilingBytes: 1,
                effectiveMlxMemoryCeilingBytes: 1,
                minimumMlxMemoryCeilingBytes: 1),
            healthState: healthState)) { (thrownError: any Error) in
            guard case let WorkerControlError.workerProtocolViolation(description) = thrownError else {
                return XCTFail("expected a protocol violation, got \(thrownError)");
            }
            XCTAssertEqual(description, "duplicate worker idle event");
        }
    }

    func testReadyEventPublishesTheModelAndRejectsDuplicates() throws {
        let healthState: WorkerHealthState = WorkerHealthState();
        try WorkerEventHandler.handle(
            .ready(modelId: "qwen3-model", capabilities: WorkerStartupHandshakeTests.chatCapabilities()),
            healthState: healthState);
        XCTAssertEqual(
            healthState.daemonStatusReport(),
            DaemonStatusReport(workerStatus: .ready, readyModelId: "qwen3-model"));
        XCTAssertEqual(
            healthState.currentSnapshot().readyModelCapabilities,
            WorkerStartupHandshakeTests.chatCapabilities());

        XCTAssertThrowsError(try WorkerEventHandler.handle(
            .ready(modelId: "qwen3-model", capabilities: WorkerStartupHandshakeTests.chatCapabilities()),
            healthState: healthState)) { (thrownError: any Error) in
            guard case let WorkerControlError.workerProtocolViolation(description) = thrownError else {
                return XCTFail("expected a protocol violation, got \(thrownError)");
            }
            XCTAssertEqual(description, "duplicate worker readiness");
        }
    }

    func testRuntimePolicyAcknowledgementMustMatchThePublishedModel() throws {
        let healthState: WorkerHealthState = WorkerHealthState();
        try WorkerEventHandler.handle(
            .ready(modelId: "qwen3-model", capabilities: WorkerStartupHandshakeTests.chatCapabilities()),
            healthState: healthState);

        let foreignAcknowledgement: WorkerRuntimeFeatureConfiguration = WorkerRuntimeFeatureConfiguration(
            configurationGeneration: "gen-1",
            persistentPromptCacheEnabled: true,
            promptCacheMaximumSizeBytes: 1073741824,
            loadedModel: WorkerStartupHandshakeTests.loadedModel(modelId: "other-model"));
        XCTAssertThrowsError(try WorkerEventHandler.handle(
            .runtimeFeatureConfigurationApplied(foreignAcknowledgement),
            healthState: healthState)) { (thrownError: any Error) in
            guard case let WorkerControlError.workerProtocolViolation(description) = thrownError else {
                return XCTFail("expected a protocol violation, got \(thrownError)");
            }
            XCTAssertEqual(
                description,
                "runtime policy acknowledgement does not match the published model");
        }

        let matchingAcknowledgement: WorkerRuntimeFeatureConfiguration = WorkerRuntimeFeatureConfiguration(
            configurationGeneration: "gen-1",
            persistentPromptCacheEnabled: true,
            promptCacheMaximumSizeBytes: 1073741824,
            loadedModel: WorkerStartupHandshakeTests.loadedModel(modelId: "qwen3-model"));
        try WorkerEventHandler.handle(
            .runtimeFeatureConfigurationApplied(matchingAcknowledgement),
            healthState: healthState);
        XCTAssertEqual(
            healthState.currentSnapshot().workerRuntimeFeatureConfiguration,
            matchingAcknowledgement);
        XCTAssertTrue(healthState.hasRuntimeFeatureConfiguration());

        // A refreshed generation with an identical loaded model is the legal
        // live-memory-update shape; a changed loaded-model payload without a
        // model transition is not.
        let silentlyChangedPolicy: WorkerRuntimeFeatureConfiguration = WorkerRuntimeFeatureConfiguration(
            configurationGeneration: "gen-2",
            persistentPromptCacheEnabled: true,
            promptCacheMaximumSizeBytes: 1073741824,
            loadedModel: WorkerStartupHandshakeTests.loadedModel(modelId: "qwen3-model", maximumOutputTokens: 2048));
        XCTAssertThrowsError(try WorkerEventHandler.handle(
            .runtimeFeatureConfigurationApplied(silentlyChangedPolicy),
            healthState: healthState)) { (thrownError: any Error) in
            guard case let WorkerControlError.workerProtocolViolation(description) = thrownError else {
                return XCTFail("expected a protocol violation, got \(thrownError)");
            }
            XCTAssertEqual(
                description,
                "runtime policy changed without an atomic model transition");
        }

        let refreshedGeneration: WorkerRuntimeFeatureConfiguration = WorkerRuntimeFeatureConfiguration(
            configurationGeneration: "gen-2",
            persistentPromptCacheEnabled: true,
            promptCacheMaximumSizeBytes: 1073741824,
            loadedModel: WorkerStartupHandshakeTests.loadedModel(modelId: "qwen3-model"));
        try WorkerEventHandler.handle(
            .runtimeFeatureConfigurationApplied(refreshedGeneration),
            healthState: healthState);
        XCTAssertEqual(
            healthState.currentSnapshot().workerRuntimeFeatureConfiguration,
            refreshedGeneration);
    }

    func testMemorySamplesPublishAndClearTheirObservations() throws {
        let healthState: WorkerHealthState = WorkerHealthState();
        let memorySnapshot: WorkerMlxMemorySnapshot = WorkerStartupHandshakeTests.memorySnapshot();
        let expertResidency: WorkerExpertResidencySnapshot = WorkerExpertResidencySnapshot(
            totalLayerCount: 48,
            residentExpertCount: 12,
            residentExpertPayloadBytes: 1073741824);
        try WorkerEventHandler.handle(
            .mlxMemorySample(mlxMemorySnapshot: memorySnapshot, expertResidency: expertResidency),
            healthState: healthState);
        XCTAssertEqual(healthState.currentSnapshot().latestMlxMemorySnapshot, memorySnapshot);
        XCTAssertEqual(healthState.currentSnapshot().expertResidency, expertResidency);

        try WorkerEventHandler.handle(
            .mlxMemorySample(mlxMemorySnapshot: nil, expertResidency: nil),
            healthState: healthState);
        XCTAssertNil(healthState.currentSnapshot().latestMlxMemorySnapshot);
        // Residency is concrete topology and survives a cleared sample.
        XCTAssertEqual(healthState.currentSnapshot().expertResidency, expertResidency);
    }

    func testStartupWaitReturnsImmediatelyWithoutAStartupConfiguration() throws {
        let workerProcess: WorkerProcess = try WorkerProcess.launch(
            workerExecutablePath: "/bin/sleep",
            arguments: ["30"]);
        defer {
            _ = try? workerProcess.close();
        }
        let eventPump: WorkerEventPump = WorkerEventPump(workerProcess: workerProcess);
        try WorkerStartupRuntime.waitForStartupRuntimeConfiguration(
            workerProcess: workerProcess,
            eventPump: eventPump,
            healthState: WorkerHealthState(),
            modelLoadTimeout: 1);
        XCTAssertFalse(workerProcess.isStartupRuntimeConfigurationApplied());
    }

    func testStartupWaitCompletesWhenTheFakeWorkerAcknowledgesItsPolicy() throws {
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

        XCTAssertTrue(workerProcess.isStartupRuntimeConfigurationApplied());
        XCTAssertEqual(workerProcess.expectedConfigurationGeneration(), "gen-1");
        XCTAssertEqual(
            healthState.daemonStatusReport(),
            DaemonStatusReport(workerStatus: .ready, readyModelId: nil));
        XCTAssertEqual(
            healthState.currentSnapshot().machineMlxMemoryCeilingBytes,
            17179869184);
        XCTAssertEqual(
            healthState.currentSnapshot().workerRuntimeFeatureConfiguration?.configurationGeneration,
            "gen-1");
    }

    func testStartupWaitTimesOutWhenTheWorkerNeverAcknowledges() throws {
        let workerProcess: WorkerProcess = try WorkerProcess.launch(
            workerExecutablePath: "/bin/bash",
            arguments: ["-c", "exec sleep 30\n"],
            workerStartupConfiguration: WorkerStartupHandshakeTests.startupConfiguration());
        defer {
            _ = try? workerProcess.close();
        }
        let eventPump: WorkerEventPump = WorkerEventPump(workerProcess: workerProcess);

        XCTAssertThrowsError(try WorkerStartupRuntime.waitForStartupRuntimeConfiguration(
            workerProcess: workerProcess,
            eventPump: eventPump,
            healthState: WorkerHealthState(),
            modelLoadTimeout: 1)) { (thrownError: any Error) in
            guard case let WorkerControlError.modelLoadTimeout(modelLoadTimeoutMillis) = thrownError else {
                return XCTFail("expected a model-load timeout, got \(thrownError)");
            }
            XCTAssertEqual(modelLoadTimeoutMillis, 1000);
        }
        XCTAssertFalse(workerProcess.isStartupRuntimeConfigurationApplied());
    }

    func testStartupWaitSurfacesAStreamClosureInsteadOfATimeout() throws {
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

        XCTAssertThrowsError(try WorkerStartupRuntime.waitForStartupRuntimeConfiguration(
            workerProcess: workerProcess,
            eventPump: eventPump,
            healthState: WorkerHealthState(),
            modelLoadTimeout: 10)) { (thrownError: any Error) in
            guard case WorkerControlError.workerEventStreamClosed = thrownError else {
                return XCTFail("expected a stream closure, got \(thrownError)");
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
