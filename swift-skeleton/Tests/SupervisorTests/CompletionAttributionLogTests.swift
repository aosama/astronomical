import Foundation

import CryptoKit
import Testing

import AstronomicalConfig

@testable import Supervisor

/**
 * Hermetic coverage for the completion attribution log, migrating the
 * journeys of apps/supervisor/tests/hermetic/completion_attribution_log.rs:
 * the arguments bounding logic, the enabled/disabled gating, and the JSONL
 * serialization contract of `completion.jsonl`, mirroring the generation
 * performance log journeys.
 */
@Suite(.tags(.hermeticJourney))
final class CompletionAttributionLogTests {

    @Test
    func should_record_full_arguments_when_under_the_size_cap() throws -> Void {
        let toolCallRecord: CompletionToolCallRecord = CompletionToolCallRecord.fromArguments(
            toolCallIndex: 0,
            functionName: "read",
            argumentsJson: "{\"path\":\"romeo-and-juliet.md\"}")
        let boundedArguments: CompletionArgumentsRecord = toolCallRecord.arguments
        #expect(boundedArguments.truncated == false)
        #expect(boundedArguments.json == "{\"path\":\"romeo-and-juliet.md\"}")
        #expect(boundedArguments.sha256.isEmpty == false)
        #expect(boundedArguments.sizeBytes == "{\"path\":\"romeo-and-juliet.md\"}".utf8.count)
    }

    @Test
    func should_truncate_and_hash_arguments_when_over_the_size_cap() throws -> Void {
        let oversizedArguments: String = AttributionJourneySupport.romeoArgumentsPayload(repeats: 1_000)
        let toolCallRecord: CompletionToolCallRecord = CompletionToolCallRecord.fromArguments(
            toolCallIndex: 1,
            functionName: "find_character",
            argumentsJson: oversizedArguments)
        let boundedArguments: CompletionArgumentsRecord = toolCallRecord.arguments
        #expect(boundedArguments.truncated == true)
        #expect(boundedArguments.sizeBytes == oversizedArguments.utf8.count)
        #expect(boundedArguments.json.utf8.count < oversizedArguments.utf8.count)
        #expect(boundedArguments.sha256.isEmpty == false)
        // The sha256 must be of the full original arguments, not the
        // truncation, so two identical over-cap calls correlate regardless of
        // truncation.
        let expectedDigest: SHA256.Digest = SHA256.hash(data: Data(oversizedArguments.utf8))
        let expectedSha256: String = expectedDigest.map({ (digestByte: UInt8) -> String in
            return String(format: "%02x", digestByte)
        }).joined()
        #expect(boundedArguments.sha256 == expectedSha256)
    }

    @Test
    func should_write_one_completion_row_with_tool_calls_when_enabled() throws -> Void {
        let logDirectory: FilePath = try AttributionJourneySupport.freshTemporaryDirectory(
            named: "completion-enabled")
        defer { try? FileManager.default.removeItem(atPath: logDirectory.string); }
        let completionLog: CompletionAttributionLog = try CompletionAttributionLog.open(
            logDirectory: logDirectory,
            completionAttributionEnabled: true)

        completionLog.recordCompletion(
            timestampMillis: 1_700_000_000_000,
            requestId: 42,
            modelId: "ornith-1.5-35b",
            completionReason: "tool_calls",
            completedToolCalls: [
                CompletedToolCall(
                    toolCallIndex: 0,
                    functionName: "read",
                    argumentsJson: "{\"path\":\"romeo-and-juliet.md\"}"),
                CompletedToolCall(
                    toolCallIndex: 1,
                    functionName: "find_character",
                    argumentsJson: "{\"name\":\"Romeo\"}"),
            ])

        let completionLines: Array<String> = try CompletionAttributionLogTests.readLogLines(logDirectory: logDirectory)
        #expect(completionLines.count == 1)
        let parsedRecord: [String: Any] = try CompletionAttributionLogTests.parseJsonObject(completionLines[0])
        #expect(parsedRecord["request_id"] as? Int == 42)
        #expect(parsedRecord["model_id"] as? String == "ornith-1.5-35b")
        #expect(parsedRecord["completion_reason"] as? String == "tool_calls")
        let parsedToolCalls: Array<Any> = parsedRecord["tool_calls"] as! Array<Any>
        let firstToolCall: [String: Any] = parsedToolCalls[0] as! [String: Any]
        #expect(firstToolCall["tool_call_index"] as? Int == 0)
        #expect(firstToolCall["function_name"] as? String == "read")
        let firstArguments: [String: Any] = firstToolCall["arguments"] as! [String: Any]
        #expect(firstArguments["truncated"] as? Bool == false)
        #expect(firstArguments["json"] as? String == "{\"path\":\"romeo-and-juliet.md\"}")
        let secondToolCall: [String: Any] = parsedToolCalls[1] as! [String: Any]
        #expect(secondToolCall["function_name"] as? String == "find_character")
    }

    @Test
    func should_write_an_empty_tool_call_list_for_end_of_sequence() throws -> Void {
        let logDirectory: FilePath = try AttributionJourneySupport.freshTemporaryDirectory(
            named: "completion-end-of-sequence")
        defer { try? FileManager.default.removeItem(atPath: logDirectory.string); }
        let completionLog: CompletionAttributionLog = try CompletionAttributionLog.open(
            logDirectory: logDirectory,
            completionAttributionEnabled: true)

        completionLog.recordCompletion(
            timestampMillis: 1_700_000_000_000,
            requestId: 7,
            modelId: "ornith-1.5-35b",
            completionReason: "end_of_sequence",
            completedToolCalls: [])

        let completionLines: Array<String> = try CompletionAttributionLogTests.readLogLines(logDirectory: logDirectory)
        #expect(completionLines.count == 1)
        let parsedRecord: [String: Any] = try CompletionAttributionLogTests.parseJsonObject(completionLines[0])
        #expect(parsedRecord["completion_reason"] as? String == "end_of_sequence")
        let parsedToolCalls: Array<Any> = parsedRecord["tool_calls"] as! Array<Any>
        #expect(parsedToolCalls.count == 0)
    }

    @Test
    func should_write_nothing_when_attribution_is_disabled() throws -> Void {
        let logDirectory: FilePath = try AttributionJourneySupport.freshTemporaryDirectory(
            named: "completion-disabled")
        defer { try? FileManager.default.removeItem(atPath: logDirectory.string); }
        let disabledLog: CompletionAttributionLog = try CompletionAttributionLog.open(
            logDirectory: logDirectory,
            completionAttributionEnabled: false)
        disabledLog.recordCompletion(
            timestampMillis: 1_700_000_000_000,
            requestId: 99,
            modelId: "ornith-1.5-35b",
            completionReason: "tool_calls",
            completedToolCalls: [
                CompletedToolCall(
                    toolCallIndex: 0,
                    functionName: "read",
                    argumentsJson: "{\"path\":\"romeo-and-juliet.md\"}"),
            ])
        let completionFileUrl: URL = URL(
            fileURLWithPath: logDirectory.appending(component: "completion.jsonl").string)
        #expect(FileManager.default.fileExists(atPath: completionFileUrl.path) == false)
    }

    @Test
    func should_never_leak_local_paths_into_completion_rows() throws -> Void {
        let logDirectory: FilePath = try AttributionJourneySupport.freshTemporaryDirectory(
            named: "completion-paths")
        defer { try? FileManager.default.removeItem(atPath: logDirectory.string); }
        let completionLog: CompletionAttributionLog = try CompletionAttributionLog.open(
            logDirectory: logDirectory,
            completionAttributionEnabled: true)

        completionLog.recordCompletion(
            timestampMillis: 1_700_000_000_000,
            requestId: 5,
            modelId: "ornith-1.5-35b",
            completionReason: "tool_calls",
            completedToolCalls: [
                CompletedToolCall(
                    toolCallIndex: 0,
                    functionName: "read",
                    argumentsJson: "{\"path\":\"romeo-and-juliet.md\"}"),
            ])

        let serializedRow: String = try CompletionAttributionLogTests.readLogLines(logDirectory: logDirectory)[0]
        #expect(serializedRow.contains("/fictional/") == false)
        #expect(serializedRow.contains("Users") == false)
    }

    // MARK: Journey fixtures

    private static func readLogLines(logDirectory: FilePath) throws -> Array<String> {
        let completionFileUrl: URL = URL(
            fileURLWithPath: logDirectory.appending(component: "completion.jsonl").string)
        let logContents: String = try String(contentsOf: completionFileUrl, encoding: .utf8)
        return logContents.trimmingCharacters(in: .whitespacesAndNewlines)
            .split(separator: "\n")
            .map({ (logLine: Substring) -> String in return String(logLine) })
    }

    private static func parseJsonObject(_ serializedLine: String) throws -> [String: Any] {
        let parsedDocument: Any = try JSONSerialization.jsonObject(
            with: Data(serializedLine.utf8),
            options: [])
        guard let parsedObject: [String: Any] = parsedDocument as? [String: Any] else {
            throw JourneyAssertionFailure(problem: "the completion row must serialize as a JSON object")
        }
        return parsedObject
    }
}
