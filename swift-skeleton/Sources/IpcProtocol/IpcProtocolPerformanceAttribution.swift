import Foundation;
import os;

/// Small switchable timing pair for the IPC transport critical path (frame
/// reads, frame writes, daemon bind/accept, client connect). Every transport
/// operation must be attributable to its exact call site, so the start and end
/// of each operation are captured separately and only logged when the caller
/// enables performance attribution.
internal enum IpcProtocolPerformanceAttribution {

    private static let performanceAttributionLog: Logger = Logger(subsystem: "dev.astronomical.ipc", category: "performance");

    internal static func startedOperation(operationName: String, performanceAttributionEnabled: Bool) -> ContinuousClock.Instant? {
        if performanceAttributionEnabled == false {
            return nil;
        }
        return ContinuousClock.now;
    }

    internal static func finishedOperation(operationName: String, operationStart: ContinuousClock.Instant?, operationOutcome: String, performanceAttributionEnabled: Bool) {
        if performanceAttributionEnabled == false {
            return;
        }
        guard let unwrappedOperationStart = operationStart else {
            return;
        }
        let elapsedDuration = ContinuousClock.now - unwrappedOperationStart;
        let elapsedMilliseconds = Double(elapsedDuration.components.seconds) * 1000.0 + Double(elapsedDuration.components.attoseconds) / 1_000_000_000_000_000.0;
        performanceAttributionLog.info("operation=\(operationName, privacy: .public) outcome=\(operationOutcome, privacy: .public) elapsed_ms=\(elapsedMilliseconds, format: .fixed(precision: 3))");
    }
}
