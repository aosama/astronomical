import Foundation;

/// Bounded aggregate for every occurrence of one operation in one report.
public struct PerformanceOperationMeasurement: Equatable, Sendable {

    private var occurrenceCountValue: UInt64;
    private var totalElapsedNanosecondsValue: UInt64;
    private var minimumElapsedNanosecondsValue: UInt64;
    private var maximumElapsedNanosecondsValue: UInt64;
    private var firstStartedOffsetNanosecondsValue: UInt64;
    private var lastEndedOffsetNanosecondsValue: UInt64;

    /// Aggregate before any occurrence has been recorded.
    public static let empty = PerformanceOperationMeasurement(
        occurrenceCountValue: 0,
        totalElapsedNanosecondsValue: 0,
        minimumElapsedNanosecondsValue: UInt64.max,
        maximumElapsedNanosecondsValue: 0,
        firstStartedOffsetNanosecondsValue: 0,
        lastEndedOffsetNanosecondsValue: 0);

    public var occurrenceCount: UInt64 {
        occurrenceCountValue;
    }

    public var totalElapsedNanoseconds: UInt64 {
        totalElapsedNanosecondsValue;
    }

    public var minimumElapsedNanoseconds: UInt64 {
        minimumElapsedNanosecondsValue;
    }

    public var maximumElapsedNanoseconds: UInt64 {
        maximumElapsedNanosecondsValue;
    }

    public var firstStartedOffsetNanoseconds: UInt64 {
        firstStartedOffsetNanosecondsValue;
    }

    public var lastEndedOffsetNanoseconds: UInt64 {
        lastEndedOffsetNanosecondsValue;
    }

    /// Folds one occurrence into the aggregate with saturating arithmetic, so
    /// overlapping or clock-edge evidence can never wrap into a small number.
    public mutating func record(
        startedOffsetNanoseconds: UInt64,
        endedOffsetNanoseconds: UInt64
    ) -> Void {
        let elapsedNanoseconds = endedOffsetNanoseconds
            &- startedOffsetNanoseconds;
        if occurrenceCountValue == 0 {
            firstStartedOffsetNanosecondsValue = startedOffsetNanoseconds;
        }
        let (nextOccurrenceCount, occurrenceOverflow) = occurrenceCountValue
            .addingReportingOverflow(1);
        occurrenceCountValue = occurrenceOverflow ? UInt64.max : nextOccurrenceCount;
        let (nextTotal, totalOverflow) = totalElapsedNanosecondsValue
            .addingReportingOverflow(elapsedNanoseconds);
        totalElapsedNanosecondsValue = totalOverflow ? UInt64.max : nextTotal;
        minimumElapsedNanosecondsValue = min(
            minimumElapsedNanosecondsValue,
            elapsedNanoseconds);
        maximumElapsedNanosecondsValue = max(
            maximumElapsedNanosecondsValue,
            elapsedNanoseconds);
        lastEndedOffsetNanosecondsValue = endedOffsetNanoseconds;
    }
}
