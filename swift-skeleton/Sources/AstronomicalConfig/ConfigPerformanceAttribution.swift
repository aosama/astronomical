import Foundation;
import os;

/// Small switchable timing pair for configuration parse and migration passes.
/// Every configuration pass must be attributable to its exact call site, so the
/// start and end of each pass are captured separately and only logged when the
/// caller enables performance attribution.
internal enum ConfigPerformanceAttribution {

    private static let performanceAttributionLog: Logger = Logger(subsystem: "dev.astronomical.config", category: "performance");

    internal static func startedPass(operationName: String, performanceAttributionEnabled: Bool) -> ContinuousClock.Instant? {
        if performanceAttributionEnabled == false {
            return nil;
        }
        return ContinuousClock.now;
    }

    internal static func finishedPass(operationName: String, passStart: ContinuousClock.Instant?, passOutcome: String, performanceAttributionEnabled: Bool) {
        if performanceAttributionEnabled == false {
            return;
        }
        guard let unwrappedPassStart = passStart else {
            return;
        }
        let elapsedDuration = ContinuousClock.now - unwrappedPassStart;
        let elapsedMilliseconds = Double(elapsedDuration.components.seconds) * 1000.0 + Double(elapsedDuration.components.attoseconds) / 1_000_000_000_000_000.0;
        performanceAttributionLog.info("operation=\(operationName, privacy: .public) outcome=\(passOutcome, privacy: .public) elapsed_ms=\(elapsedMilliseconds, format: .fixed(precision: 3))");
    }
}
