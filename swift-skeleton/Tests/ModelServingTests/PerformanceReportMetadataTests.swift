import Foundation

import Testing

import ModelServing

/// Serialized metadata contracts for finished attribution reports, port of
/// crates/model-serving/tests/hermetic/performance_attribution/report_metadata.rs.
@Suite
final class PerformanceReportMetadataTests {

    @Test
    func shouldSerializeRejectedGenerationOutcomeAndFailureDescription() throws {
        let performanceAttributionJson = try PerformanceAttributionTestSupport
            .serializeGenerationReport(PerformanceAttribution.enabled(), .rejected);

        #expect(
            PerformanceAttributionTestSupport.text(performanceAttributionJson, "report_kind")
                == "generation");
        #expect(
            PerformanceAttributionTestSupport.text(performanceAttributionJson, "outcome")
                == "rejected");
        #expect(
            PerformanceAttributionTestSupport.integer(performanceAttributionJson, "request_id")
                == 42);
        #expect(
            PerformanceAttributionTestSupport.text(
                performanceAttributionJson,
                "failure_description") == "simulated generation failure");
    }

    @Test
    func shouldSerializeUnobservedPrefillStateInModelLoadingReport() throws {
        let performanceAttributionJson = try PerformanceAttributionTestSupport
            .serializeModelLoadingReport(PerformanceAttribution.enabled(), .success);

        #expect(
            PerformanceAttributionTestSupport.boolean(
                performanceAttributionJson,
                "prefill_transient_observation_completed") == false);
        #expect(
            PerformanceAttributionTestSupport.integer(
                performanceAttributionJson,
                "prefill_observed_transient_high_water_bytes") == 0);
    }

    @Test
    func shouldSerializePrefillEvidenceInGenerationReport() throws {
        let performanceAttributionJson = try PerformanceAttributionTestSupport
            .serializeGenerationReport(PerformanceAttribution.enabled(), .success);

        #expect(
            PerformanceAttributionTestSupport.boolean(
                performanceAttributionJson,
                "prefill_transient_observation_completed") == true);
        #expect(
            PerformanceAttributionTestSupport.integer(
                performanceAttributionJson,
                "prefill_observed_transient_high_water_bytes")
                == Int(PerformanceAttributionTestSupport
                    .attributedPrefillTransientHighWaterBytes));
    }

    @Test
    func shouldSerializeProcessIoAsAnAllOrUnavailablePair() throws {
        let performanceAttributionJson = try PerformanceAttributionTestSupport
            .serializeGenerationReport(PerformanceAttribution.enabled(), .success);

        let physicalDiskReadBytes = PerformanceAttributionTestSupport.integer(
            performanceAttributionJson,
            "process_physical_disk_read_bytes");
        let physicalDiskWrittenBytes = PerformanceAttributionTestSupport.integer(
            performanceAttributionJson,
            "process_physical_disk_written_bytes");
        let unavailabilityReason = PerformanceAttributionTestSupport.text(
            performanceAttributionJson,
            "process_io_unavailability_reason");

        #expect(
            (physicalDiskReadBytes != nil) == (physicalDiskWrittenBytes != nil),
            "read and write deltas must come from the same process-I/O interval");
        #expect(
            (physicalDiskReadBytes != nil) == (unavailabilityReason == nil),
            "available byte deltas and an unavailability reason are mutually exclusive");
    }

    @Test
    func shouldExcludeOuterDiagnosticSpansFromAttributedElapsedTime() throws {
        let performanceAttribution = PerformanceAttribution.enabled();
        performanceAttribution.recordCompletedOperation(
            .promptPrefillAdvanceSpan,
            startedOffsetNanoseconds: 0,
            endedOffsetNanoseconds: 10_000_000_000);
        performanceAttribution.recordCompletedOperation(
            .decodeAdvanceSpan,
            startedOffsetNanoseconds: 0,
            endedOffsetNanoseconds: 5_000_000_000);
        performanceAttribution.recordCompletedOperation(
            .promptTokenization,
            startedOffsetNanoseconds: 0,
            endedOffsetNanoseconds: 7);

        let performanceAttributionJson = try PerformanceAttributionTestSupport
            .serializeGenerationReport(performanceAttribution, .success);

        #expect(
            PerformanceAttributionTestSupport.integer(
                performanceAttributionJson,
                "attributed_elapsed_nanoseconds") == 7);
    }
}
