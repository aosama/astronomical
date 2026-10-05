import Foundation;
import os;

/**
 * Lightweight performance attribution for discovery filesystem walks and
 * parses. Opt-in per entry point through the `attributionEnabled` flag so
 * production discovery stays telemetry-free by default while slow paths stay
 * attributable when diagnosing regressions.
 */
internal enum DiscoveryPerformanceTrace {
    private static let discoveryLogger: Logger = Logger(subsystem: "com.astronomical.config", category: "ModelDiscovery");

    internal static func measure<MeasuredValue>(
        operationName: String,
        attributionEnabled: Bool,
        operation: () throws -> MeasuredValue
    ) rethrows -> MeasuredValue {
        guard attributionEnabled else {
            return try operation();
        }
        let operationStartNanoseconds: UInt64 = DispatchTime.now().uptimeNanoseconds;
        let measuredValue: MeasuredValue = try operation();
        let operationEndNanoseconds: UInt64 = DispatchTime.now().uptimeNanoseconds;
        let operationDurationNanoseconds: UInt64 = operationEndNanoseconds - operationStartNanoseconds;
        self.discoveryLogger.notice("\(operationName, privacy: .public) took \(operationDurationNanoseconds, privacy: .public) ns");
        return measuredValue;
    }
}
