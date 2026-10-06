import Testing;

import Foundation;

import IpcProtocol;
import JourneyCategories;

@testable import Supervisor;

/// Hermetic coverage of the worker model-swap wait: staged swap and policy
/// acknowledgements, capability and identity validation, rejection handling,
/// and interleaved process-scoped events, driven by fake `/bin/bash` workers
/// emitting genuine framed events over real pipes.
@Suite(.serialized, .tags(.hermeticJourney))
final class WorkerModelSwapTests {

    private static let CHAT_CAPABILITIES_PAYLOAD: String =
        "{\"chat\":{\"supports_reasoning\":true,\"supports_tool_calls\":false,"
        + "\"has_vision\":false,\"max_input_tokens\":8191,\"max_output_tokens\":1024,"
        + "\"context_window\":8192},\"image_generation\":null,\"embeddings\":null}";

    private static func modelSwappedPayload(modelId: String) -> String {
        return "{\"kind\":\"model_swapped\",\"model_id\":\"\(modelId)\","
            + "\"capabilities\":\(WorkerModelSwapTests.CHAT_CAPABILITIES_PAYLOAD),"
            + "\"expert_memory_mode\":null,\"minimum_mlx_memory_ceiling_bytes\":1048576}";
    }

    private static let RUNTIME_POLICY_PAYLOAD: String =
        "{\"kind\":\"runtime_feature_configuration_applied\","
        + "\"worker_runtime_feature_configuration\":{\"configuration_generation\":\"gen-1\","
        + "\"persistent_prompt_cache_enabled\":true,\"prompt_cache_maximum_size_bytes\":1073741824,"
        + "\"loaded_model\":{\"kind\":\"autoregressive\",\"configuration\":{\"model_id\":\"m1\","
        + "\"maximum_context_tokens\":8192,\"maximum_output_tokens\":1024,"
        + "\"chunking\":{\"fixed_prompt_processing_chunk_size_tokens\":512,"
        + "\"fixed_ssd_streaming_prompt_processing_chunk_size_tokens\":512,"
        + "\"full_attention_key_value_growth_tokens\":512,"
        + "\"prefill_graph_submission_layer_interval\":1,"
        + "\"experimental_ssd_paging_prefill_graph_submission_layer_interval\":1,"
        + "\"experimental_ssd_paging_generation_graph_submission_layer_interval\":1,"
        + "\"prompt_cache_block_tokens\":null,"
        + "\"prompt_cache_common_prefix_stride_blocks\":1,"
        + "\"experimental_decode_stage_attribution_enabled\":false,"
        + "\"experimental_quantized_kv_cache_enabled\":false,"
        + "\"experimental_fused_moe_decode_enabled\":false}}}}}";

    private static let MODEL_SWAP_FAILED_PAYLOAD: String =
        "{\"kind\":\"model_swap_failed\",\"loaded_model_remains_ready\":false,"
        + "\"model_load_failure_reason\":\"model files unreadable\"}";

    @Test
    func should_publish_the_model_and_policy_together_through_the_swap_wait() throws {
        let fakeWorkerScript: String = FakeWorkerEventEmitter.frameEmitterFunction()
            + FakeWorkerEventEmitter.emitLine(payload: FakeWorkerEventEmitter.idleEventPayload())
            + FakeWorkerEventEmitter.emitLine(payload: WorkerModelSwapTests.modelSwappedPayload(modelId: "m1"))
            + FakeWorkerEventEmitter.emitLine(payload: WorkerModelSwapTests.RUNTIME_POLICY_PAYLOAD)
            + "exec sleep 30\n";
        let workerProcess: WorkerProcess = try WorkerProcess.launch(
            workerExecutablePath: "/bin/bash",
            arguments: ["-c", fakeWorkerScript]);
        defer {
            _ = try? workerProcess.close();
        }
        let healthState: WorkerHealthState = WorkerHealthState();
        let eventPump: WorkerEventPump = WorkerEventPump(workerProcess: workerProcess);

        let swapOutcome: ModelSwapWaitOutcome = try WorkerModelSwap.waitForModelSwap(
            eventPump: eventPump,
            healthState: healthState,
            expectedConfigurationGeneration: "gen-1",
            expectedModelRuntimeConfiguration: WorkerModelSwapTests.expectedModelConfiguration(),
            modelLoadTimeout: 10);

        #expect(swapOutcome == ModelSwapWaitOutcome.loaded);
        #expect(
            healthState.daemonStatusReport()
                == DaemonStatusReport(workerStatus: .ready, readyModelId: "m1"));
        #expect(
            healthState.currentSnapshot().workerRuntimeFeatureConfiguration?.configurationGeneration
                == "gen-1");
        #expect(
            healthState.currentSnapshot().minimumMlxMemoryCeilingBytes
                == 1048576);
    }

    @Test
    func should_surface_a_swap_rejection_and_reset_to_a_model_less_worker() throws {
        let fakeWorkerScript: String = FakeWorkerEventEmitter.frameEmitterFunction()
            + FakeWorkerEventEmitter.emitLine(payload: FakeWorkerEventEmitter.idleEventPayload())
            + FakeWorkerEventEmitter.emitLine(payload: WorkerModelSwapTests.MODEL_SWAP_FAILED_PAYLOAD)
            + "exec sleep 30\n";
        let workerProcess: WorkerProcess = try WorkerProcess.launch(
            workerExecutablePath: "/bin/bash",
            arguments: ["-c", fakeWorkerScript]);
        defer {
            _ = try? workerProcess.close();
        }
        let healthState: WorkerHealthState = WorkerHealthState();
        let eventPump: WorkerEventPump = WorkerEventPump(workerProcess: workerProcess);

        let swapOutcome: ModelSwapWaitOutcome = try WorkerModelSwap.waitForModelSwap(
            eventPump: eventPump,
            healthState: healthState,
            expectedConfigurationGeneration: "gen-1",
            expectedModelRuntimeConfiguration: WorkerModelSwapTests.expectedModelConfiguration(),
            modelLoadTimeout: 10);

        #expect(
            swapOutcome
                == ModelSwapWaitOutcome.rejected(modelLoadFailureReason: "model files unreadable"));
        #expect(
            healthState.daemonStatusReport()
                == DaemonStatusReport(workerStatus: .ready, readyModelId: nil));
    }

    @Test
    func should_reject_a_swap_acknowledgement_with_an_identity_mismatch() throws {
        let fakeWorkerScript: String = FakeWorkerEventEmitter.frameEmitterFunction()
            + FakeWorkerEventEmitter.emitLine(payload: FakeWorkerEventEmitter.idleEventPayload())
            + FakeWorkerEventEmitter.emitLine(payload: WorkerModelSwapTests.modelSwappedPayload(modelId: "other"))
            + "exec sleep 30\n";
        let workerProcess: WorkerProcess = try WorkerProcess.launch(
            workerExecutablePath: "/bin/bash",
            arguments: ["-c", fakeWorkerScript]);
        defer {
            _ = try? workerProcess.close();
        }
        let eventPump: WorkerEventPump = WorkerEventPump(workerProcess: workerProcess);

        do {
            _ = try WorkerModelSwap.waitForModelSwap(
                eventPump: eventPump,
                healthState: WorkerHealthState(),
                expectedConfigurationGeneration: "gen-1",
                expectedModelRuntimeConfiguration: WorkerModelSwapTests.expectedModelConfiguration(),
                modelLoadTimeout: 10);
            Issue.record("expected a protocol violation");
        } catch let workerControlError as WorkerControlError {
            guard case let .workerProtocolViolation(description) = workerControlError else {
                Issue.record(Comment(stringLiteral: "expected a protocol violation, got \(workerControlError)"));
                return;
            }
            #expect(description == "model swap identity acknowledgement mismatch");
        }
    }

    @Test
    func should_reject_a_swap_acknowledgement_with_a_capability_geometry_mismatch() throws {
        // The acknowledged context window is 4096 while the requested policy
        // says 8192, so the worker's capability advertisement cannot
        // constrain requests under that policy.
        let mismatchedCapabilitiesPayload: String =
            "{\"kind\":\"model_swapped\",\"model_id\":\"m1\","
            + "\"capabilities\":{\"chat\":{\"supports_reasoning\":true,\"supports_tool_calls\":false,"
            + "\"has_vision\":false,\"max_input_tokens\":4095,\"max_output_tokens\":1024,"
            + "\"context_window\":4096},\"image_generation\":null,\"embeddings\":null},"
            + "\"expert_memory_mode\":null,\"minimum_mlx_memory_ceiling_bytes\":1048576}";
        let fakeWorkerScript: String = FakeWorkerEventEmitter.frameEmitterFunction()
            + FakeWorkerEventEmitter.emitLine(payload: FakeWorkerEventEmitter.idleEventPayload())
            + FakeWorkerEventEmitter.emitLine(payload: mismatchedCapabilitiesPayload)
            + "exec sleep 30\n";
        let workerProcess: WorkerProcess = try WorkerProcess.launch(
            workerExecutablePath: "/bin/bash",
            arguments: ["-c", fakeWorkerScript]);
        defer {
            _ = try? workerProcess.close();
        }
        let eventPump: WorkerEventPump = WorkerEventPump(workerProcess: workerProcess);

        do {
            _ = try WorkerModelSwap.waitForModelSwap(
                eventPump: eventPump,
                healthState: WorkerHealthState(),
                expectedConfigurationGeneration: "gen-1",
                expectedModelRuntimeConfiguration: WorkerModelSwapTests.expectedModelConfiguration(),
                modelLoadTimeout: 10);
            Issue.record("expected a protocol violation");
        } catch let workerControlError as WorkerControlError {
            guard case let .workerProtocolViolation(description) = workerControlError else {
                Issue.record(Comment(stringLiteral: "expected a protocol violation, got \(workerControlError)"));
                return;
            }
            #expect(description == "model swap capabilities acknowledgement mismatch");
        }
    }

    @Test
    func should_reject_duplicate_swap_acknowledgements() throws {
        let fakeWorkerScript: String = FakeWorkerEventEmitter.frameEmitterFunction()
            + FakeWorkerEventEmitter.emitLine(payload: FakeWorkerEventEmitter.idleEventPayload())
            + FakeWorkerEventEmitter.emitLine(payload: WorkerModelSwapTests.modelSwappedPayload(modelId: "m1"))
            + FakeWorkerEventEmitter.emitLine(payload: WorkerModelSwapTests.modelSwappedPayload(modelId: "m1"))
            + "exec sleep 30\n";
        let workerProcess: WorkerProcess = try WorkerProcess.launch(
            workerExecutablePath: "/bin/bash",
            arguments: ["-c", fakeWorkerScript]);
        defer {
            _ = try? workerProcess.close();
        }
        let eventPump: WorkerEventPump = WorkerEventPump(workerProcess: workerProcess);

        do {
            _ = try WorkerModelSwap.waitForModelSwap(
                eventPump: eventPump,
                healthState: WorkerHealthState(),
                expectedConfigurationGeneration: "gen-1",
                expectedModelRuntimeConfiguration: WorkerModelSwapTests.expectedModelConfiguration(),
                modelLoadTimeout: 10);
            Issue.record("expected a protocol violation");
        } catch let workerControlError as WorkerControlError {
            guard case let .workerProtocolViolation(description) = workerControlError else {
                Issue.record(Comment(stringLiteral: "expected a protocol violation, got \(workerControlError)"));
                return;
            }
            #expect(description == "duplicate model swap acknowledgement");
        }
    }

    private static func expectedModelConfiguration() -> WorkerLoadedModelRuntimeConfiguration {
        return WorkerModelSwapTests.loadedConfiguration(modelId: "m1");
    }

    private static func loadedConfiguration(modelId: String) -> WorkerLoadedModelRuntimeConfiguration {
        return WorkerModelSwapTests.loadedConfiguration(modelId: modelId, maximumContextTokens: 8192);
    }

    private static func loadedConfiguration(
        modelId: String,
        maximumContextTokens: UInt32
    ) -> WorkerLoadedModelRuntimeConfiguration {
        return WorkerLoadedModelRuntimeConfiguration.autoregressive(
            WorkerLoadedAutoregressiveModelRuntimeConfiguration(
                modelId: modelId,
                maximumContextTokens: maximumContextTokens,
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
}
