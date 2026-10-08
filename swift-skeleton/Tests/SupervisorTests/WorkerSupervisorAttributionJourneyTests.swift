import Foundation

import Testing

import AstronomicalConfig
import IpcProtocol
import JourneyCategories

@testable import Supervisor

/**
 * End-to-end journey for the completion-event attribution fan-out, migrating
 * the serving-path slice of apps/supervisor/src/worker_completion_event.rs:
 * one chat generation driven through a scripted resident-model worker lands
 * one row in `performance.jsonl` (measured counters, computed throughput,
 * cache diagnostics) and one row in `completion.jsonl` (the emitted tool
 * calls with bounded arguments) when the operator enabled that toggle.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class WorkerSupervisorAttributionJourneyTests {

    @Test
    func should_record_both_attribution_rows_when_one_generation_completes() throws -> Void {
        let loggingDirectory: FilePath = try AttributionJourneySupport.freshTemporaryDirectory(
            named: "worker-completion")
        defer { try? FileManager.default.removeItem(atPath: loggingDirectory.string); }
        let generationPerformanceLog: GenerationPerformanceLog = try GenerationPerformanceLog.open(
            logDirectory: loggingDirectory);
        let completionAttributionLog: CompletionAttributionLog = try CompletionAttributionLog.open(
            logDirectory: loggingDirectory,
            completionAttributionEnabled: true);

        // The scripted worker emits the generation's events up front — the
        // pump buffers them — and blocks in `read _` until shutdown's EOF
        // half-close, because supervisor commands are length-prefixed frames
        // with no newline for a line read to synchronize on.
        let scriptedWorker: String = FakeWorkerEventEmitter.frameEmitterFunction()
            + FakeWorkerEventEmitter.emitLine(payload: FakeWorkerEventEmitter.residentReadyEventPayload(modelId: "m123"))
            + FakeWorkerEventEmitter.emitLine(payload: FakeWorkerEventEmitter.residentRuntimePolicyPayload(modelId: "m123"))
            + FakeWorkerEventEmitter.emitLine(payload: WorkerSupervisorAttributionJourneyTests.toolCallOutputPayload(requestId: 9))
            + FakeWorkerEventEmitter.emitLine(payload: WorkerSupervisorAttributionJourneyTests.toolCallsCompletedPayload(requestId: 9))
            + "read _\n"
            + "exit 0\n";
        let supervisor: WorkerSupervisor = try WorkerSupervisor.launch(
            workerExecutablePath: "/bin/bash",
            workerArguments: ["-c", scriptedWorker],
            workerStartupConfiguration: WorkerSupervisorAttributionJourneyTests.startupConfiguration(),
            modelPolicyCatalog: [:],
            modelLoadTimeout: 10,
            generationPerformanceLog: generationPerformanceLog,
            completionAttributionLog: completionAttributionLog);
        defer { _ = try? supervisor.shutdown() }
        #expect(supervisor.workerHealthSnapshot().readyModelId == "m123");

        let streamEvents: Array<ChatGenerationStreamEvent> = try supervisor.startChatGeneration(
            WorkerSupervisorAttributionJourneyTests.chatGenerationCommand(requestId: 9, modelId: "m123"));

        #expect(streamEvents.contains(where: { (streamEvent: ChatGenerationStreamEvent) -> Bool in
            if case let .toolCall(toolCallIndex, functionName, _) = streamEvent {
                return toolCallIndex == 0 && functionName == "read";
            }
            return false;
        }));
        #expect(streamEvents.contains(where: { (streamEvent: ChatGenerationStreamEvent) -> Bool in
            if case let .completed(_, generatedTokenCount, _, _, reason) = streamEvent {
                return generatedTokenCount == 1 && reason == .toolCalls;
            }
            return false;
        }));

        let performanceRow: [String: Any] = try WorkerSupervisorAttributionJourneyTests.readOnlyRow(
            loggingDirectory: loggingDirectory,
            fileName: "performance.jsonl");
        #expect(performanceRow["request_id"] as? Int == 9);
        #expect(performanceRow["model_id"] as? String == "m123");
        #expect(performanceRow["completion_reason"] as? String == "tool_calls");
        #expect(performanceRow["prompt_token_count"] as? Int == 12);
        #expect(performanceRow["cached_token_count"] as? Int == 0);
        #expect(performanceRow["generated_token_count"] as? Int == 1);
        #expect((performanceRow["total_elapsed_millis"] as? NSNumber) != nil);
        #expect(performanceRow["time_to_first_output_millis"] != nil);
        // No prefill_progress event fired: a zero prefill budget serializes a
        // null throughput rather than a divide-by-zero.
        #expect(performanceRow["prefill_tok_per_second"] is NSNull);
        #expect(performanceRow["persistent_prompt_cache_diagnostics"] is NSNull);

        let completionRow: [String: Any] = try WorkerSupervisorAttributionJourneyTests.readOnlyRow(
            loggingDirectory: loggingDirectory,
            fileName: "completion.jsonl");
        #expect(completionRow["request_id"] as? Int == 9);
        #expect(completionRow["model_id"] as? String == "m123");
        #expect(completionRow["completion_reason"] as? String == "tool_calls");
        let recordedToolCalls: Array<Any> = completionRow["tool_calls"] as! Array<Any>
        #expect(recordedToolCalls.count == 1);
        let recordedToolCall: [String: Any] = recordedToolCalls[0] as! [String: Any]
        #expect(recordedToolCall["function_name"] as? String == "read");
        let recordedArguments: [String: Any] = recordedToolCall["arguments"] as! [String: Any]
        #expect(recordedArguments["truncated"] as? Bool == false);
        #expect(recordedArguments["json"] as? String == "{\"path\":\"romeo-and-juliet.md\"}");
    }

    @Test
    func should_persist_worker_cache_diagnostics_for_a_completed_user_request() throws -> Void {
        let loggingDirectory: FilePath = try AttributionJourneySupport.freshTemporaryDirectory(
            named: "worker-cache-diagnostics");
        defer { try? FileManager.default.removeItem(atPath: loggingDirectory.string); }
        let generationPerformanceLog: GenerationPerformanceLog = try GenerationPerformanceLog.open(
            logDirectory: loggingDirectory);

        let scriptedWorker: String = FakeWorkerEventEmitter.frameEmitterFunction()
            + FakeWorkerEventEmitter.emitLine(payload: FakeWorkerEventEmitter.residentReadyEventPayload(modelId: "m123"))
            + FakeWorkerEventEmitter.emitLine(payload: FakeWorkerEventEmitter.residentRuntimePolicyPayload(modelId: "m123"))
            + FakeWorkerEventEmitter.emitLine(payload: WorkerSupervisorAttributionJourneyTests.cacheDiagnosticsCompletedPayload(requestId: 9))
            + "read _\n"
            + "exit 0\n";
        let supervisor: WorkerSupervisor = try WorkerSupervisor.launch(
            workerExecutablePath: "/bin/bash",
            workerArguments: ["-c", scriptedWorker],
            workerStartupConfiguration: WorkerSupervisorAttributionJourneyTests.startupConfiguration(),
            modelPolicyCatalog: [:],
            modelLoadTimeout: 10,
            generationPerformanceLog: generationPerformanceLog);
        defer { _ = try? supervisor.shutdown() }
        #expect(supervisor.workerHealthSnapshot().readyModelId == "m123");

        let streamEvents: Array<ChatGenerationStreamEvent> = try supervisor.startChatGeneration(
            WorkerSupervisorAttributionJourneyTests.chatGenerationCommand(requestId: 9, modelId: "m123"));
        #expect(streamEvents.contains(where: { (streamEvent: ChatGenerationStreamEvent) -> Bool in
            if case let .completed(_, generatedTokenCount, _, _, reason) = streamEvent {
                return generatedTokenCount == 1 && reason == .endOfSequence;
            }
            return false;
        }));

        let performanceRow: [String: Any] = try WorkerSupervisorAttributionJourneyTests.readOnlyRow(
            loggingDirectory: loggingDirectory,
            fileName: "performance.jsonl");
        let cacheDiagnostics: [String: Any] = try #require(
            performanceRow["persistent_prompt_cache_diagnostics"] as? [String: Any],
            "the completed request should persist its cache diagnostics");
        #expect(cacheDiagnostics["lookup_outcome"] as? String == "miss");
        #expect(cacheDiagnostics["block_token_count"] as? Int == 2_048);
        #expect(cacheDiagnostics["matched_sequence_state_block_count"] as? Int == 0);
        #expect(cacheDiagnostics["restored_block_count"] as? Int == 0);
        #expect(cacheDiagnostics["first_missing_sequence_state_block_index"] as? Int == 0);
        #expect(cacheDiagnostics["miss_reason"] as? String == "root_sequence_state_block_missing");
        let startupCleanupEvidence: [String: Any] = try #require(
            cacheDiagnostics["startup_cleanup_evidence"] as? [String: Any]);
        let obsoleteFormat: [String: Any] = try #require(startupCleanupEvidence["obsolete_format"] as? [String: Any]);
        #expect(obsoleteFormat["artifact_count"] as? Int == 2);
        let corruptCurrentFormat: [String: Any] = try #require(startupCleanupEvidence["corrupt_current_format"] as? [String: Any]);
        #expect(corruptCurrentFormat["block_count"] as? Int == 1);
        #expect(cacheDiagnostics["published_block_count"] as? Int == 1);

        let performanceLogDocument: String = try String(
            contentsOf: URL(fileURLWithPath: loggingDirectory.string + "/performance.jsonl"),
            encoding: String.Encoding.utf8);
        #expect(performanceLogDocument.contains("/fictional/") == false);
        #expect(performanceLogDocument.contains("model_directory") == false);
    }

    // MARK: Journey fixtures

    private static func cacheDiagnosticsCompletedPayload(requestId: UInt64) -> String {
        return "{\"kind\":\"completed\",\"request_id\":\(requestId),"
            + "\"prompt_token_count\":12,\"generated_token_count\":1,\"reasoning_token_count\":0,"
            + "\"cached_token_count\":0,"
            + "\"persistent_prompt_cache_diagnostics\":{"
            + "\"lookup_outcome\":\"miss\",\"block_token_count\":2048,"
            + "\"complete_prompt_block_count\":1,\"maximum_restorable_block_count\":1,"
            + "\"matched_sequence_state_block_count\":0,\"restored_block_count\":0,"
            + "\"partial_tail_block_token_count\":null,"
            + "\"first_missing_sequence_state_block_index\":0,"
            + "\"miss_reason\":\"root_sequence_state_block_missing\","
            + "\"expected_block_hash_prefix\":null,"
            + "\"startup_cleanup_evidence\":{"
            + "\"interrupted_transaction_recovery\":{\"artifact_count\":0,\"block_count\":0,\"byte_count\":0},"
            + "\"obsolete_format\":{\"artifact_count\":2,\"block_count\":0,\"byte_count\":3},"
            + "\"corrupt_current_format\":{\"artifact_count\":0,\"block_count\":1,\"byte_count\":4},"
            + "\"quota_eviction\":{\"artifact_count\":0,\"block_count\":0,\"byte_count\":0}},"
            + "\"published_block_count\":1,\"allocator_bytes_cleared_for_publication\":0,"
            + "\"expert_bytes_reclaimed_for_publication\":0,\"expert_bytes_reclaimed_for_restore\":0},"
            + "\"reason\":\"end_of_sequence\"}";
    }

    private static func toolCallOutputPayload(requestId: UInt64) -> String {
        return "{\"kind\":\"output\",\"request_id\":\(requestId),\"sequence_number\":0,"
            + "\"generated_token_count\":1,"
            + "\"outputs\":[{\"kind\":\"tool_call\",\"tool_call_index\":0,"
            + "\"function_name\":\"read\",\"arguments_json\":\"{\\\"path\\\":\\\"romeo-and-juliet.md\\\"}\"}],"
            + "\"mlx_memory_snapshot\":null,\"expert_residency\":null}";
    }

    private static func toolCallsCompletedPayload(requestId: UInt64) -> String {
        return "{\"kind\":\"completed\",\"request_id\":\(requestId),"
            + "\"prompt_token_count\":12,\"generated_token_count\":1,\"reasoning_token_count\":0,"
            + "\"cached_token_count\":0,\"persistent_prompt_cache_diagnostics\":null,"
            + "\"reason\":\"tool_calls\"}";
    }

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

    private static func readOnlyRow(
        loggingDirectory: FilePath,
        fileName: String
    ) throws -> [String: Any] {
        let logFileUrl: URL = URL(
            fileURLWithPath: loggingDirectory.appending(component: fileName).string)
        let logContents: String = try String(contentsOf: logFileUrl, encoding: .utf8)
        let logLines: Array<String> = logContents.trimmingCharacters(in: .whitespacesAndNewlines)
            .split(separator: "\n")
            .map({ (logLine: Substring) -> String in return String(logLine) })
        guard logLines.count == 1 else {
            throw JourneyAssertionFailure(problem: "exactly one \(fileName) row must be written")
        }
        let parsedDocument: Any = try JSONSerialization.jsonObject(
            with: Data(logLines[0].utf8),
            options: [])
        guard let parsedObject: [String: Any] = parsedDocument as? [String: Any] else {
            throw JourneyAssertionFailure(problem: "the \(fileName) row must serialize as a JSON object")
        }
        return parsedObject
    }
}
