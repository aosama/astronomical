import Foundation;

/// Switchable start/end performance attribution for serving-side operations.
///
/// Replaces the inert skeleton for the engine tracks: the full measurement
/// catalog port stays with the performance-attribution end-to-end track,
/// while model load, prompt processing, and token generation carry
/// attributable spans from their first working slice, per the repository
/// attribution mandate.
public enum ServingPerformanceAttribution {

    /// Opens one attributed operation; returns the start instant only when
    /// attribution is enabled, so disabled runs pay one branch.
    public static func startedOperation(
        operationName: String,
        attributionEnabled: Bool
    ) -> ContinuousClock.Instant? {
        guard attributionEnabled else {
            return nil;
        }
        return ContinuousClock.now;
    }

    /// Closes one attributed operation and records its wall duration.
    public static func endedOperation(
        operationName: String,
        operationStart: ContinuousClock.Instant?,
        attributionEnabled: Bool
    ) -> Void {
        guard attributionEnabled, let operationStart = operationStart else {
            return;
        }
        let elapsed: Duration = ContinuousClock.now.duration(to: operationStart);
        let elapsedMilliseconds: Int64 = elapsed.components.seconds * 1000
            + elapsed.components.attoseconds / 1_000_000_000_000_000;
        // The attribution log stream is stderr until the measurement-catalog
        // port lands; operators toggle it through configuration.
        FileHandle.standardError.write(Data(
            "attribution operation=\(operationName) elapsed_ms=\(elapsedMilliseconds)\n".utf8));
    }
}
