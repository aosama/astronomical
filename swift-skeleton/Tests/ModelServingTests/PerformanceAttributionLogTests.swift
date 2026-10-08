import Foundation

import Testing

import ModelServing

/// Append-only report log contracts, port of
/// crates/model-serving/tests/hermetic/performance_attribution/log.rs.
@Suite
final class PerformanceAttributionLogTests {

    @Test
    func shouldAppendOneParseableModelLoadingJsonRecord() throws {
        let temporaryLogDirectory = try PerformanceAttributionTestSupport
            .makeTemporaryDirectory();
        defer { try? FileManager.default.removeItem(at: temporaryLogDirectory) }
        let performanceAttributionLogPath = temporaryLogDirectory
            .appendingPathComponent("performance-attribution.jsonl");
        let performanceAttribution = PerformanceAttribution.enabled();
        performanceAttribution.recordCompletedOperation(
            .artifactValidation,
            startedOffsetNanoseconds: 5,
            endedOffsetNanoseconds: 25);
        let performanceAttributionReport = try #require(
            performanceAttribution.finishModelLoading(
                PerformanceAttributionTestSupport.modelLoadingMetadata(outcome: .success)),
            "enabled attribution should produce one model-loading report");
        let performanceAttributionLog = try PerformanceAttributionLog.open(
            at: performanceAttributionLogPath,
            performanceAttributionEnabled: true);

        try performanceAttributionLog.record(performanceAttributionReport);

        let performanceAttributionJsonl = try String(
            contentsOf: performanceAttributionLogPath,
            encoding: .utf8);
        let performanceAttributionJson = try #require(
            JSONSerialization.jsonObject(with: Data(performanceAttributionJsonl.utf8))
                as? [String: Any],
            "attribution report should be valid JSON");
        #expect(
            PerformanceAttributionTestSupport.text(performanceAttributionJson, "report_kind")
                == "model_loading");
        #expect(
            PerformanceAttributionTestSupport.text(performanceAttributionJson, "outcome")
                == "success");
        let operationRows = PerformanceAttributionTestSupport.rows(
            performanceAttributionJson,
            "operations");
        #expect(
            PerformanceAttributionTestSupport.text(operationRows[0], "operation")
                == "artifact_validation");
        #expect(
            PerformanceAttributionTestSupport.integer(
                operationRows[0],
                "total_elapsed_nanoseconds") == 20);
    }

    @Test
    func shouldNotCreateAttributionLogWhenDisabled() throws {
        let temporaryLogDirectory = try PerformanceAttributionTestSupport
            .makeTemporaryDirectory();
        defer { try? FileManager.default.removeItem(at: temporaryLogDirectory) }
        let performanceAttributionLogPath = temporaryLogDirectory
            .appendingPathComponent("performance-attribution.jsonl");

        _ = try PerformanceAttributionLog.open(
            at: performanceAttributionLogPath,
            performanceAttributionEnabled: false);

        #expect(!FileManager.default.fileExists(atPath: performanceAttributionLogPath.path));
    }

    @Test
    func shouldAppendMultipleReportsToOneJsonlFile() throws {
        let temporaryLogDirectory = try PerformanceAttributionTestSupport
            .makeTemporaryDirectory();
        defer { try? FileManager.default.removeItem(at: temporaryLogDirectory) }
        let performanceAttributionLogPath = temporaryLogDirectory
            .appendingPathComponent("performance-attribution.jsonl");
        let performanceAttributionLog = try PerformanceAttributionLog.open(
            at: performanceAttributionLogPath,
            performanceAttributionEnabled: true);

        for modelLoadingOutcome in [PerformanceAttributionOutcome.success, .failed] {
            let performanceAttributionReport = try #require(
                PerformanceAttribution.enabled().finishModelLoading(
                    PerformanceAttributionTestSupport.modelLoadingMetadata(
                        outcome: modelLoadingOutcome)),
                "enabled attribution should produce a model-loading report");
            try performanceAttributionLog.record(performanceAttributionReport);
        }

        let performanceAttributionJsonl = try String(
            contentsOf: performanceAttributionLogPath,
            encoding: .utf8);
        let performanceAttributionReportLines = try performanceAttributionJsonl
            .split(separator: "\n")
            .map { line -> [String: Any] in
                try #require(
                    JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
                    "every JSON Lines record should parse");
            };

        #expect(performanceAttributionReportLines.count == 2);
        #expect(
            PerformanceAttributionTestSupport.text(
                performanceAttributionReportLines[0],
                "outcome") == "success");
        #expect(
            PerformanceAttributionTestSupport.text(
                performanceAttributionReportLines[1],
                "outcome") == "failed");
    }

    @Test
    func shouldReopenAttributionLogAfterRotationReplacesTheFile() throws {
        let temporaryLogDirectory = try PerformanceAttributionTestSupport
            .makeTemporaryDirectory();
        defer { try? FileManager.default.removeItem(at: temporaryLogDirectory) }
        let performanceAttributionLogPath = temporaryLogDirectory
            .appendingPathComponent("performance-attribution.jsonl");
        let rotatedPerformanceAttributionLogPath = temporaryLogDirectory
            .appendingPathComponent("performance-attribution.previous.jsonl");
        let performanceAttributionLog = try PerformanceAttributionLog.open(
            at: performanceAttributionLogPath,
            performanceAttributionEnabled: true);
        let firstPerformanceAttributionReport = try #require(
            PerformanceAttribution.enabled().finishModelLoading(
                PerformanceAttributionTestSupport.modelLoadingMetadata(outcome: .success)),
            "enabled attribution should produce the first report");
        try performanceAttributionLog.record(firstPerformanceAttributionReport);
        try FileManager.default.moveItem(
            at: performanceAttributionLogPath,
            to: rotatedPerformanceAttributionLogPath);
        FileManager.default.createFile(
            atPath: performanceAttributionLogPath.path,
            contents: nil);

        let secondPerformanceAttributionReport = try #require(
            PerformanceAttribution.enabled().finishModelLoading(
                PerformanceAttributionTestSupport.modelLoadingMetadata(outcome: .failed)),
            "enabled attribution should produce the second report");
        try performanceAttributionLog.record(secondPerformanceAttributionReport);

        let activePerformanceAttributionLog = try String(
            contentsOf: performanceAttributionLogPath,
            encoding: .utf8);
        let activePerformanceAttributionJson = try #require(
            JSONSerialization.jsonObject(
                with: Data(activePerformanceAttributionLog.utf8)) as? [String: Any],
            "active attribution report should be valid JSON");
        #expect(
            PerformanceAttributionTestSupport.text(
                activePerformanceAttributionJson,
                "outcome") == "failed");
        let rotatedPerformanceAttributionLog = try String(
            contentsOf: rotatedPerformanceAttributionLogPath,
            encoding: .utf8);
        #expect(rotatedPerformanceAttributionLog.split(separator: "\n").count == 1);
    }
}
