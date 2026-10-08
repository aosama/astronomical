import Foundation

import Testing

import ModelServing

/// Measurement aggregation and disabled-path contracts, port of
/// crates/model-serving/tests/hermetic/performance_attribution/measurement.rs.
/// Stands in for the Rust test's `String` failure payload; Swift requires a
/// genuine Error type for Result failures.
private enum MeasuredOperationFailure: Error, Equatable {
    case diskReadFailed;
}

@Suite
final class PerformanceOperationMeasurementTests {

    @Test
    func shouldKeepTheDisabledAttributionHandlePointerSized() {
        #expect(
            MemoryLayout<PerformanceAttribution>.size <= MemoryLayout<Int>.size,
            "disabled attribution travels through every engine command and must not enlarge the command queue");
    }

    @Test
    func shouldExecuteMeasuredWorkWithoutRecordingWhenAttributionIsDisabled() {
        let performanceAttribution = PerformanceAttribution.disabled();

        let operationOutput = performanceAttribution.measureOperation(
            .promptTokenization,
            { _ in 42 });

        #expect(operationOutput == 42);
        #expect(
            performanceAttribution.operationMeasurement(.promptTokenization) == nil);
    }

    @Test
    func shouldRecordAnErrorReturningOperation() {
        let performanceAttribution = PerformanceAttribution.enabled();

        let operationOutcome: Result<Int, MeasuredOperationFailure> = performanceAttribution
            .measureOperation(
                .persistentPromptCacheOpenAndScan,
                { _ in .failure(.diskReadFailed) });

        #expect(operationOutcome == .failure(.diskReadFailed));
        #expect(
            performanceAttribution.operationMeasurement(.persistentPromptCacheOpenAndScan)?
                .occurrenceCount == 1);
    }

    @Test
    func shouldAggregateRepeatedOperationMeasurements() throws {
        let performanceAttribution = PerformanceAttribution.enabled();
        performanceAttribution.recordCompletedOperation(
            .promptTokenization,
            startedOffsetNanoseconds: 5,
            endedOffsetNanoseconds: 25);
        performanceAttribution.recordCompletedOperation(
            .promptTokenization,
            startedOffsetNanoseconds: 50,
            endedOffsetNanoseconds: 100);

        let operationMeasurement = try #require(
            performanceAttribution.operationMeasurement(.promptTokenization),
            "two recorded operations should have an aggregate");

        #expect(operationMeasurement.occurrenceCount == 2);
        #expect(operationMeasurement.totalElapsedNanoseconds == 70);
        #expect(operationMeasurement.minimumElapsedNanoseconds == 20);
        #expect(operationMeasurement.maximumElapsedNanoseconds == 50);
        #expect(operationMeasurement.firstStartedOffsetNanoseconds == 5);
        #expect(operationMeasurement.lastEndedOffsetNanoseconds == 100);
    }

    @Test
    func shouldSaturateRepeatedOperationElapsedTime() throws {
        let performanceAttribution = PerformanceAttribution.enabled();
        let maximumElapsedNanoseconds = UInt64.max;
        for _ in 0..<2 {
            performanceAttribution.recordCompletedOperation(
                .promptRendering,
                startedOffsetNanoseconds: 0,
                endedOffsetNanoseconds: maximumElapsedNanoseconds);
        }

        let operationMeasurement = try #require(
            performanceAttribution.operationMeasurement(.promptRendering),
            "saturated operations should still retain an aggregate");

        #expect(operationMeasurement.totalElapsedNanoseconds == UInt64.max);
        #expect(operationMeasurement.maximumElapsedNanoseconds == UInt64.max);
    }

    @Test
    func shouldAggregatePerformanceCountersWithSaturation() {
        let performanceAttribution = PerformanceAttribution.enabled();
        performanceAttribution.recordCounter(.generatedTokenCount, amount: UInt64.max);
        performanceAttribution.recordCounter(.generatedTokenCount, amount: 1);

        #expect(
            performanceAttribution.counterValue(.generatedTokenCount) == UInt64.max);
    }

    @Test
    func shouldRetainTheLargestMaximumCounterObservation() {
        let performanceAttribution = PerformanceAttribution.enabled();
        let maximumCounter = PerformanceCounter.positionalFileReadMaximumElapsedNanoseconds;
        performanceAttribution.recordMaximumCounter(maximumCounter, amount: 40);
        performanceAttribution.recordMaximumCounter(maximumCounter, amount: 10);
        performanceAttribution.recordMaximumCounter(maximumCounter, amount: 80);

        #expect(performanceAttribution.counterValue(maximumCounter) == 80);
    }
}
