import Foundation;

import IpcProtocol;

/// Compact lifetime-of-daemon serving summary rendered by local interfaces,
/// porting apps/supervisor/src/serving_session_snapshot.rs: completed-request
/// counts, prompt-token totals with their reused slice, target prompt-work
/// reuse, and rolling throughput averages.
public struct ServingSessionSnapshot: Equatable {

    public var completedRequestCount: UInt64;
    public var totalPromptTokenCount: UInt64;
    public var totalReusedPromptTokenCount: UInt64;
    public var targetPromptWorkTokenCount: UInt64;
    public var targetReusedPromptWorkTokenCount: UInt64;
    public var averagePrefillTokPerSecond: Double;
    public var averageGenerationTokPerSecond: Double;
    var prefillMeasurementCount: UInt64;
    var generationMeasurementCount: UInt64;

    public static func empty() -> ServingSessionSnapshot {
        return ServingSessionSnapshot(
            completedRequestCount: 0,
            totalPromptTokenCount: 0,
            totalReusedPromptTokenCount: 0,
            targetPromptWorkTokenCount: 0,
            targetReusedPromptWorkTokenCount: 0,
            averagePrefillTokPerSecond: 0,
            averageGenerationTokPerSecond: 0,
            prefillMeasurementCount: 0,
            generationMeasurementCount: 0);
    }

    /// Records one completed request's token totals and throughput
    /// measurements; absent rates leave their averages untouched.
    public mutating func recordCompletedRequest(
        promptTokenCount: UInt32,
        cachedTokenCount: UInt32,
        prefillTokPerSecond: Double?,
        generationTokPerSecond: Double?
    ) -> Void {
        self.completedRequestCount &+= 1;
        self.totalPromptTokenCount &+= UInt64(promptTokenCount);
        self.totalReusedPromptTokenCount &+= UInt64(min(cachedTokenCount, promptTokenCount));
        if let prefillTokPerSecond = prefillTokPerSecond {
            self.averagePrefillTokPerSecond = ServingSessionSnapshot.rollingAverage(
                currentAverage: self.averagePrefillTokPerSecond,
                priorCount: self.prefillMeasurementCount,
                newMeasurement: prefillTokPerSecond);
            self.prefillMeasurementCount &+= 1;
        }
        if let generationTokPerSecond = generationTokPerSecond {
            self.averageGenerationTokPerSecond = ServingSessionSnapshot.rollingAverage(
                currentAverage: self.averageGenerationTokPerSecond,
                priorCount: self.generationMeasurementCount,
                newMeasurement: generationTokPerSecond);
            self.generationMeasurementCount &+= 1;
        }
    }

    /// Records one request's target prompt-work reuse observation.
    public mutating func recordPromptWorkReuse(
        _ promptWorkReuse: WorkerPromptWorkReuse
    ) -> Void {
        self.targetPromptWorkTokenCount &+= promptWorkReuse.targetEligibleTokenCount;
        self.targetReusedPromptWorkTokenCount &+= min(
            promptWorkReuse.targetRestoredTokenCount,
            promptWorkReuse.targetEligibleTokenCount);
    }

    /// Derives the prefill and generation throughput for one completed
    /// request, porting GenerationPerformanceRecord::compute_throughput:
    /// rates exist only when tokens were processed in measured time.
    public static func computeThroughput(
        promptTokenCount: UInt32,
        cachedTokenCount: UInt32,
        generatedTokenCount: UInt16,
        prefillElapsedMillis: UInt64,
        generationElapsedMillis: UInt64
    ) -> (prefillTokPerSecond: Double?, generationTokPerSecond: Double?) {
        let uncachedPromptTokens: UInt32 = promptTokenCount &- min(cachedTokenCount, promptTokenCount);
        let prefillTokPerSecond: Double?;
        if prefillElapsedMillis > 0 && uncachedPromptTokens > 0 {
            prefillTokPerSecond = Double(uncachedPromptTokens) / (Double(prefillElapsedMillis) / 1000.0);
        } else {
            prefillTokPerSecond = nil;
        }
        let generationTokPerSecond: Double?;
        if generationElapsedMillis > 0 && generatedTokenCount > 0 {
            generationTokPerSecond = Double(generatedTokenCount) / (Double(generationElapsedMillis) / 1000.0);
        } else {
            generationTokPerSecond = nil;
        }
        return (prefillTokPerSecond, generationTokPerSecond);
    }

    private static func rollingAverage(
        currentAverage: Double,
        priorCount: UInt64,
        newMeasurement: Double
    ) -> Double {
        let priorTotal: Double = currentAverage * Double(priorCount);
        return (priorTotal + newMeasurement) / Double(priorCount &+ 1);
    }
}
