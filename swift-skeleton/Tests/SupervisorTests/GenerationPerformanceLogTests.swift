import Foundation

import Testing

import AstronomicalConfig
import IpcProtocol

@testable import Supervisor

/**
 * Hermetic coverage for the generation performance log, migrating the
 * journeys of apps/supervisor/tests/hermetic/generation_performance_log.rs:
 * the throughput math's edge cases (fully cached prompts, zero elapsed,
 * zero tokens) and the append-only JSONL contract of `performance.jsonl`
 * with its serde-shaped field set.
 */
@Suite(.tags(.hermeticJourney))
final class GenerationPerformanceLogTests {

    @Test
    func should_create_generation_performance_record_with_computed_throughput() -> Void {
        let computedThroughput: (prefillTokPerSecond: Double?, generationTokPerSecond: Double?) =
            GenerationPerformanceRecord.computeThroughput(
                promptTokenCount: 100_000,
                cachedTokenCount: 50_000,
                generatedTokenCount: 500,
                prefillElapsedMillis: 1_000,
                generationElapsedMillis: 5_000)
        // 50,000 uncached tokens / 1.0 seconds = 50,000 tok/s
        #expect(computedThroughput.prefillTokPerSecond == 50_000.0)
        // 500 tokens / 5.0 seconds = 100 tok/s
        #expect(computedThroughput.generationTokPerSecond == 100.0)
    }

    @Test
    func should_return_none_for_prefill_tok_per_second_when_fully_cached() -> Void {
        // When all tokens are cached (0 ms prefill elapsed), prefill TPS is None.
        let computedThroughput: (prefillTokPerSecond: Double?, generationTokPerSecond: Double?) =
            GenerationPerformanceRecord.computeThroughput(
                promptTokenCount: 10_000,
                cachedTokenCount: 10_000,
                generatedTokenCount: 200,
                prefillElapsedMillis: 0,
                generationElapsedMillis: 2_000)
        #expect(computedThroughput.prefillTokPerSecond == nil)
        #expect(computedThroughput.generationTokPerSecond == 100.0)
    }

    @Test
    func should_return_none_for_generation_tok_per_second_when_zero_elapsed() -> Void {
        let computedThroughput: (prefillTokPerSecond: Double?, generationTokPerSecond: Double?) =
            GenerationPerformanceRecord.computeThroughput(
                promptTokenCount: 5_000,
                cachedTokenCount: 0,
                generatedTokenCount: 1,
                prefillElapsedMillis: 500,
                generationElapsedMillis: 0)
        #expect(computedThroughput.prefillTokPerSecond == 10_000.0)
        #expect(computedThroughput.generationTokPerSecond == nil)
    }

    @Test
    func should_return_none_for_prefill_tps_when_uncached_tokens_is_zero() -> Void {
        // If prompt_token_count == cached_token_count, uncached = 0, so TPS
        // is None even if prefill_elapsed_millis > 0: the math stays robust.
        let computedThroughput: (prefillTokPerSecond: Double?, generationTokPerSecond: Double?) =
            GenerationPerformanceRecord.computeThroughput(
                promptTokenCount: 1_000,
                cachedTokenCount: 1_000,
                generatedTokenCount: 100,
                prefillElapsedMillis: 500,
                generationElapsedMillis: 1_000)
        #expect(computedThroughput.prefillTokPerSecond == nil)
    }

    @Test
    func should_return_none_for_generation_tps_when_token_count_is_zero() -> Void {
        let computedThroughput: (prefillTokPerSecond: Double?, generationTokPerSecond: Double?) =
            GenerationPerformanceRecord.computeThroughput(
                promptTokenCount: 1_000,
                cachedTokenCount: 0,
                generatedTokenCount: 0,
                prefillElapsedMillis: 100,
                generationElapsedMillis: 1_000)
        #expect(computedThroughput.generationTokPerSecond == nil)
    }

    @Test
    func should_open_and_append_to_performance_log() throws -> Void {
        let logDirectory: FilePath = try AttributionJourneySupport.freshTemporaryDirectory(
            named: "performance-open")
        defer { try? FileManager.default.removeItem(atPath: logDirectory.string); }
        let performanceLog: GenerationPerformanceLog = try GenerationPerformanceLog.open(
            logDirectory: logDirectory)

        let performanceRecord: GenerationPerformanceRecord = GenerationPerformanceRecord(
            timestampMillis: 1_700_000_000_000,
            requestId: 42,
            modelId: "test-model",
            promptTokenCount: 10_000,
            cachedTokenCount: 5_000,
            generatedTokenCount: 200,
            completionReason: "end_of_sequence",
            prefillElapsedMillis: 1_000,
            generationElapsedMillis: 2_000,
            totalElapsedMillis: 3_500,
            timeToFirstOutputMillis: 1_750,
            generationPreparationElapsedMillis: 250,
            firstDecodeForwardElapsedMillis: 83,
            generationPreparationExpertSourceReadByteCount: 0,
            finalResidentExpertCount: 40,
            finalResidentExpertPayloadBytes: 11_000_000_000,
            prefillTokPerSecond: 5_000.0,
            generationTokPerSecond: 100.0,
            mlxPeakMemoryBytes: 40_000_000_000,
            mlxActiveMemoryBytes: 31_000_000_000,
            persistentPromptCacheDiagnostics: GenerationPerformanceLogTests.hitDiagnostics())

        performanceLog.record(performanceRecord)

        let performanceLogLines: Array<String> = try GenerationPerformanceLogTests.readLogLines(
            logDirectory: logDirectory,
            fileName: "performance.jsonl")
        #expect(performanceLogLines.count == 1)
        let parsedRecord: [String: Any] = try GenerationPerformanceLogTests.parseJsonObject(
            performanceLogLines[0])
        #expect(parsedRecord["request_id"] as? Int == 42)
        #expect(parsedRecord["model_id"] as? String == "test-model")
        #expect(parsedRecord["prompt_token_count"] as? Int == 10_000)
        #expect(parsedRecord["cached_token_count"] as? Int == 5_000)
        #expect(parsedRecord["generated_token_count"] as? Int == 200)
        #expect(parsedRecord["completion_reason"] as? String == "end_of_sequence")
        #expect(parsedRecord["prefill_elapsed_millis"] as? Int == 1_000)
        #expect(parsedRecord["generation_elapsed_millis"] as? Int == 2_000)
        #expect(parsedRecord["total_elapsed_millis"] as? Int == 3_500)
        #expect(parsedRecord["time_to_first_output_millis"] as? Int == 1_750)
        #expect(parsedRecord["generation_preparation_elapsed_millis"] as? Int == 250)
        #expect(parsedRecord["generation_preparation_expert_source_read_byte_count"] as? Int == 0)
        #expect((parsedRecord["prefill_tok_per_second"] as? Double) == 5_000.0)
        #expect((parsedRecord["generation_tok_per_second"] as? Double) == 100.0)
        #expect(parsedRecord["mlx_peak_memory_bytes"] as? Int == 40_000_000_000)
        #expect(parsedRecord["mlx_active_memory_bytes"] as? Int == 31_000_000_000)
        let parsedDiagnostics: [String: Any] = parsedRecord["persistent_prompt_cache_diagnostics"] as! [String: Any]
        #expect(parsedDiagnostics["matched_sequence_state_block_count"] as? Int == 4)
        #expect(parsedDiagnostics["published_block_count"] as? Int == 1)
        #expect(parsedDiagnostics["expert_bytes_reclaimed_for_restore"] as? Int == 16_384)
        let parsedCleanupEvidence: [String: Any] = parsedDiagnostics["startup_cleanup_evidence"] as! [String: Any]
        let obsoleteFormatEvidence: [String: Any] = parsedCleanupEvidence["obsolete_format"] as! [String: Any]
        #expect(obsoleteFormatEvidence["artifact_count"] as? Int == 2)
        let serializedLine: String = performanceLogLines[0]
        #expect(serializedLine.contains("/fictional/") == false)
        #expect(serializedLine.contains("model_directory") == false)
    }

    @Test
    func should_append_multiple_records_to_performance_log() throws -> Void {
        let logDirectory: FilePath = try AttributionJourneySupport.freshTemporaryDirectory(
            named: "performance-append")
        defer { try? FileManager.default.removeItem(atPath: logDirectory.string); }
        let performanceLog: GenerationPerformanceLog = try GenerationPerformanceLog.open(
            logDirectory: logDirectory)

        let firstRecord: GenerationPerformanceRecord = GenerationPerformanceRecord(
            timestampMillis: 1_700_000_000_000,
            requestId: 1,
            modelId: "model-a",
            promptTokenCount: 1_000,
            cachedTokenCount: 0,
            generatedTokenCount: 50,
            completionReason: "end_of_sequence",
            prefillElapsedMillis: 200,
            generationElapsedMillis: 1_000,
            totalElapsedMillis: 1_500,
            timeToFirstOutputMillis: 400,
            generationPreparationElapsedMillis: 100,
            firstDecodeForwardElapsedMillis: 10,
            generationPreparationExpertSourceReadByteCount: 0,
            finalResidentExpertCount: 2,
            finalResidentExpertPayloadBytes: 1_100,
            prefillTokPerSecond: 5_000.0,
            generationTokPerSecond: 50.0,
            mlxPeakMemoryBytes: nil,
            mlxActiveMemoryBytes: nil,
            persistentPromptCacheDiagnostics: nil)

        let secondRecord: GenerationPerformanceRecord = GenerationPerformanceRecord(
            timestampMillis: 1_700_000_005_000,
            requestId: 2,
            modelId: "model-a",
            promptTokenCount: 50_000,
            cachedTokenCount: 49_000,
            generatedTokenCount: 500,
            completionReason: "tool_calls",
            prefillElapsedMillis: 100,
            generationElapsedMillis: 10_000,
            totalElapsedMillis: 11_000,
            timeToFirstOutputMillis: 700,
            generationPreparationElapsedMillis: 200,
            firstDecodeForwardElapsedMillis: 15,
            generationPreparationExpertSourceReadByteCount: 0,
            finalResidentExpertCount: 4,
            finalResidentExpertPayloadBytes: 2_200,
            prefillTokPerSecond: 10_000.0,
            generationTokPerSecond: 50.0,
            mlxPeakMemoryBytes: 35_000_000_000,
            mlxActiveMemoryBytes: 28_000_000_000,
            persistentPromptCacheDiagnostics: nil)

        performanceLog.record(firstRecord)
        performanceLog.record(secondRecord)

        let performanceLogLines: Array<String> = try GenerationPerformanceLogTests.readLogLines(
            logDirectory: logDirectory,
            fileName: "performance.jsonl")
        #expect(performanceLogLines.count == 2)
        let firstParsedRecord: [String: Any] = try GenerationPerformanceLogTests.parseJsonObject(
            performanceLogLines[0])
        #expect(firstParsedRecord["request_id"] as? Int == 1)
        let secondParsedRecord: [String: Any] = try GenerationPerformanceLogTests.parseJsonObject(
            performanceLogLines[1])
        #expect(secondParsedRecord["request_id"] as? Int == 2)
        #expect((secondParsedRecord["mlx_peak_memory_bytes"] as? NSNumber) != nil)
        #expect((secondParsedRecord["mlx_active_memory_bytes"] as? NSNumber) != nil)
    }

    @Test
    func should_serialize_null_for_optional_fields() throws -> Void {
        let performanceRecord: GenerationPerformanceRecord = GenerationPerformanceRecord(
            timestampMillis: 1_700_000_000_000,
            requestId: 1,
            modelId: "model-a",
            promptTokenCount: 1_000,
            cachedTokenCount: 1_000,
            generatedTokenCount: 50,
            completionReason: "end_of_sequence",
            prefillElapsedMillis: 0,
            generationElapsedMillis: 1_000,
            totalElapsedMillis: 1_500,
            timeToFirstOutputMillis: nil,
            generationPreparationElapsedMillis: nil,
            firstDecodeForwardElapsedMillis: nil,
            generationPreparationExpertSourceReadByteCount: 0,
            finalResidentExpertCount: nil,
            finalResidentExpertPayloadBytes: nil,
            prefillTokPerSecond: nil,
            generationTokPerSecond: 50.0,
            mlxPeakMemoryBytes: nil,
            mlxActiveMemoryBytes: nil,
            persistentPromptCacheDiagnostics: GenerationPerformanceLogTests.missDiagnostics())

        var wireWriter: JsonWireWriter = JsonWireWriter()
        try wireWriter.appendValue(performanceRecord.jsonlWireValue())
        let serializedRecord: String = wireWriter.serializedText
        #expect(serializedRecord.contains("\"prefill_tok_per_second\":null"))
        #expect(serializedRecord.contains("\"mlx_peak_memory_bytes\":null"))
        #expect(serializedRecord.contains("\"mlx_active_memory_bytes\":null"))
        #expect(serializedRecord.contains("\"generation_tok_per_second\":50.0"))
    }

    // MARK: Journey fixtures

    private static func hitDiagnostics() -> WorkerPersistentPromptCacheRequestDiagnostics {
        return WorkerPersistentPromptCacheRequestDiagnostics(
            lookupOutcome: .hit,
            blockTokenCount: 2_048,
            completePromptBlockCount: 5,
            maximumRestorableBlockCount: 4,
            matchedSequenceStateBlockCount: 4,
            restoredBlockCount: 4,
            partialTailBlockTokenCount: nil,
            firstMissingSequenceStateBlockIndex: nil,
            missReason: nil,
            expectedBlockHashPrefix: nil,
            startupCleanupEvidence: WorkerPersistentPromptCacheStartupCleanupEvidence(
                interruptedTransactionRecovery: WorkerPersistentPromptCacheStartupCleanupCategory(
                    artifactCount: 1, blockCount: 0, byteCount: 128),
                obsoleteFormat: WorkerPersistentPromptCacheStartupCleanupCategory(
                    artifactCount: 2, blockCount: 0, byteCount: 256),
                corruptCurrentFormat: WorkerPersistentPromptCacheStartupCleanupCategory(
                    artifactCount: 0, blockCount: 1, byteCount: 512),
                quotaEviction: WorkerPersistentPromptCacheStartupCleanupCategory(
                    artifactCount: 0, blockCount: 2, byteCount: 1_024)),
            publishedBlockCount: 1,
            allocatorBytesClearedForPublication: 4_096,
            expertBytesReclaimedForPublication: 8_192,
            expertBytesReclaimedForRestore: 16_384)
    }

    private static func missDiagnostics() -> WorkerPersistentPromptCacheRequestDiagnostics {
        return WorkerPersistentPromptCacheRequestDiagnostics(
            lookupOutcome: .miss,
            blockTokenCount: 2_048,
            completePromptBlockCount: 1,
            maximumRestorableBlockCount: 1,
            matchedSequenceStateBlockCount: 0,
            restoredBlockCount: 0,
            partialTailBlockTokenCount: nil,
            firstMissingSequenceStateBlockIndex: 0,
            missReason: .rootSequenceStateBlockMissing,
            expectedBlockHashPrefix: WorkerPersistentPromptCacheExpectedBlockHashPrefix.fromBlockHash(
                blockHash: Array<UInt8>(repeating: 1, count: 32)),
            startupCleanupEvidence: nil,
            publishedBlockCount: 1,
            allocatorBytesClearedForPublication: 0,
            expertBytesReclaimedForPublication: 0,
            expertBytesReclaimedForRestore: 0)
    }

    private static func readLogLines(logDirectory: FilePath, fileName: String) throws -> Array<String> {
        let logFileUrl: URL = URL(fileURLWithPath: logDirectory.appending(component: fileName).string)
        let logContents: String = try String(contentsOf: logFileUrl, encoding: .utf8)
        return logContents.trimmingCharacters(in: .whitespacesAndNewlines)
            .split(separator: "\n")
            .map({ (logLine: Substring) -> String in return String(logLine) })
    }

    private static func parseJsonObject(_ serializedLine: String) throws -> [String: Any] {
        let parsedDocument: Any = try JSONSerialization.jsonObject(
            with: Data(serializedLine.utf8),
            options: [])
        guard let parsedObject: [String: Any] = parsedDocument as? [String: Any] else {
            throw JourneyAssertionFailure(problem: "the log line must serialize as a JSON object")
        }
        return parsedObject
    }
}
