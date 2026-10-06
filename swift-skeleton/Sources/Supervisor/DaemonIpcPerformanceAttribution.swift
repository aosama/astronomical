import Foundation;
import os;

/// Switchable timing pair for the daemon IPC serving path (chat gates,
/// executor dispatch, and frame relay). Every serving operation must be
/// attributable to its exact call site, so the start and end of each
/// operation are captured separately and only logged when the caller enables
/// performance attribution — the same contract as the REST transport log.
enum DaemonIpcPerformanceAttribution {

    private static let performanceAttributionLog: Logger = Logger(subsystem: "dev.astronomical.daemon-ipc", category: "performance");

    static func startedOperation(operationName: String, performanceAttributionEnabled: Bool) -> ContinuousClock.Instant? {
        if performanceAttributionEnabled == false {
            return nil;
        }
        return ContinuousClock.now;
    }

    static func finishedOperation(operationName: String, operationStart: ContinuousClock.Instant?, operationOutcome: String, performanceAttributionEnabled: Bool) -> Void {
        if performanceAttributionEnabled == false {
            return;
        }
        guard let unwrappedOperationStart: ContinuousClock.Instant = operationStart else {
            return;
        }
        let elapsedDuration: ContinuousClock.Duration = ContinuousClock.now - unwrappedOperationStart;
        let elapsedMilliseconds: Double = Double(elapsedDuration.components.seconds) * 1000.0
            + Double(elapsedDuration.components.attoseconds) / 1_000_000_000_000_000.0;
        performanceAttributionLog.info("operation=\(operationName, privacy: .public) outcome=\(operationOutcome, privacy: .public) elapsed_ms=\(elapsedMilliseconds, format: .fixed(precision: 3))");
    }
}
