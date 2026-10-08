import Foundation;

/// Fixed-size enabled accumulator backing one report, mirrored from the Rust
/// `EnabledPerformanceAttribution`. Arrays are indexed by catalog position, so
/// recording stays allocation-free on inference paths.
struct EnabledPerformanceAttribution: Sendable {

    var reportStartedUptimeNanoseconds: UInt64;
    var reportStartedAtUnixMillis: UInt64;
    var operationMeasurements: [PerformanceOperationMeasurement];
    var counterValues: [UInt64];
    var previousTokenSelectedExpertIdsByLayer: [[Int]?];
    /// The immediately preceding observed decode token's compact route for the
    /// current request. Request-owned so a new request starts its prediction
    /// chain without cross-request state (issue #536).
    var routeObservationPreviousRoute: ObservedExpertRoute?;
    var previousTokenExpertRouteReuseByLayer: [PreviousTokenExpertRouteReuseMeasurement];
    var expertStreamingSourceSummaries: [ExpertStreamingSourceSummary?];
    /// Cumulative process input/output captured at the same boundary as the
    /// monotonic report clock. Keeping the Result preserves sampling failure as
    /// explicit evidence; replacing failure with zero would falsely claim that
    /// macOS served no disk input/output.
    var processIoStart: Result<MacosProcessIoSnapshot, MacosProcessIoError>;
}

/// Per-layer previous-token route reuse aggregate, serialized without expert
/// identifiers so diagnostics never carry route contents.
struct PreviousTokenExpertRouteReuseMeasurement: Sendable {

    var predictedExpertCount: UInt64;
    var matchedExpertCount: UInt64;
    var completelyMatchedLayerCount: UInt64;
    var examinedLayerCount: UInt64;

    static let empty = PreviousTokenExpertRouteReuseMeasurement(
        predictedExpertCount: 0,
        matchedExpertCount: 0,
        completelyMatchedLayerCount: 0,
        examinedLayerCount: 0);
}

/// Memory snapshot boundary supplied when an image-generation report finishes.
public struct PerformanceMemoryObservation: Sendable {

    public let mlxActiveMemoryBytes: UInt64?;
    public let mlxAllocatorCacheMemoryBytes: UInt64?;
    public let mlxPeakMemoryBytes: UInt64?;

    public init(
        mlxActiveMemoryBytes: UInt64?,
        mlxAllocatorCacheMemoryBytes: UInt64?,
        mlxPeakMemoryBytes: UInt64?
    ) {
        self.mlxActiveMemoryBytes = mlxActiveMemoryBytes;
        self.mlxAllocatorCacheMemoryBytes = mlxAllocatorCacheMemoryBytes;
        self.mlxPeakMemoryBytes = mlxPeakMemoryBytes;
    }
}

/// Switchable, request-owned attribution with a zero-allocation disabled path.
///
/// Enabled reports aggregate fixed operation and counter arrays, avoiding maps
/// and per-event records on inference paths. Disabled reports hold only `nil`,
/// skip clock reads, and execute measured closures directly.
public final class PerformanceAttribution {

    var enabledAttribution: EnabledPerformanceAttribution?;

    /// Creates a no-op accumulator that performs no clock reads.
    public static func disabled() -> PerformanceAttribution {
        PerformanceAttribution(enabledAttribution: nil);
    }

    /// Creates an enabled accumulator and captures its monotonic start boundary.
    public static func enabled() -> PerformanceAttribution {
        PerformanceAttribution(
            enabledAttribution: EnabledPerformanceAttribution(
                reportStartedUptimeNanoseconds: DispatchTime.now().uptimeNanoseconds,
                reportStartedAtUnixMillis: performanceAttributionUnixEpochMillis(),
                operationMeasurements: Array(
                    repeating: PerformanceOperationMeasurement.empty,
                    count: PerformanceOperation.allCases.count),
                counterValues: Array(repeating: 0, count: PerformanceCounter.count),
                previousTokenSelectedExpertIdsByLayer: [],
                routeObservationPreviousRoute: nil,
                previousTokenExpertRouteReuseByLayer: [],
                expertStreamingSourceSummaries: [],
                processIoStart: MacosProcessIo.sampleCurrentProcessIo()));
    }

    init(enabledAttribution: EnabledPerformanceAttribution?) {
        self.enabledAttribution = enabledAttribution;
    }

    /// Returns whether this accumulator records measurements.
    public var isEnabled: Bool {
        enabledAttribution != nil;
    }

    /// Measures one operation, including error-returning operations, when enabled.
    public func measureOperation<OperationOutput>(
        _ operation: PerformanceOperation,
        _ measuredOperation: (PerformanceAttribution) throws -> OperationOutput
    ) rethrows -> OperationOutput {
        guard let reportStartedUptimeNanoseconds = enabledAttribution?
            .reportStartedUptimeNanoseconds
        else {
            return try measuredOperation(self);
        }
        let operationStartedUptimeNanoseconds = DispatchTime.now().uptimeNanoseconds;
        let operationOutput = try measuredOperation(self);
        let operationEndedUptimeNanoseconds = DispatchTime.now().uptimeNanoseconds;
        recordCompletedOperation(
            operation,
            startedOffsetNanoseconds: saturatingElapsedNanoseconds(
                operationStartedUptimeNanoseconds,
                since: reportStartedUptimeNanoseconds),
            endedOffsetNanoseconds: saturatingElapsedNanoseconds(
                operationEndedUptimeNanoseconds,
                since: reportStartedUptimeNanoseconds));
        return operationOutput;
    }

    /// Records deterministic offsets; exposed for aggregation tests and
    /// pre-measured boundaries.
    public func recordCompletedOperation(
        _ operation: PerformanceOperation,
        startedOffsetNanoseconds: UInt64,
        endedOffsetNanoseconds: UInt64
    ) -> Void {
        guard var enabledAttribution = enabledAttribution else {
            return;
        }
        enabledAttribution.operationMeasurements[operation.rawValue]
            .record(
                startedOffsetNanoseconds: startedOffsetNanoseconds,
                endedOffsetNanoseconds: endedOffsetNanoseconds);
        self.enabledAttribution = enabledAttribution;
    }

    /// Returns the nonempty aggregate for one operation.
    public func operationMeasurement(
        _ operation: PerformanceOperation
    ) -> PerformanceOperationMeasurement? {
        guard let enabledAttribution = enabledAttribution else {
            return nil;
        }
        let operationMeasurement = enabledAttribution.operationMeasurements[operation.rawValue];
        return operationMeasurement.occurrenceCount > 0 ? operationMeasurement : nil;
    }

    /// Adds a bounded amount to one report counter.
    public func recordCounter(
        _ counter: PerformanceCounter,
        amount: UInt64
    ) -> Void {
        guard var enabledAttribution = enabledAttribution else {
            return;
        }
        let summedAmount = enabledAttribution.counterValues[counter.rawValue]
            .addingReportingOverflow(amount);
        enabledAttribution.counterValues[counter.rawValue] = summedAmount.overflow
            ? UInt64.max
            : summedAmount.partialValue;
        self.enabledAttribution = enabledAttribution;
    }

    /// Replaces one report counter with an exact amount.
    ///
    /// Peak-coherent snapshots need this: recording each term of a split with
    /// `recordMaximumCounter` lets terms from different instants mix, and a
    /// mixed split reconstructs no real moment. The caller gates on which
    /// instant wins and replaces every term of that instant together.
    public func recordSnapshotCounter(
        _ counter: PerformanceCounter,
        amount: UInt64
    ) -> Void {
        guard var enabledAttribution = enabledAttribution else {
            return;
        }
        enabledAttribution.counterValues[counter.rawValue] = amount;
        self.enabledAttribution = enabledAttribution;
    }

    /// Retains the largest observed amount for one report counter.
    public func recordMaximumCounter(
        _ counter: PerformanceCounter,
        amount: UInt64
    ) -> Void {
        guard var enabledAttribution = enabledAttribution else {
            return;
        }
        enabledAttribution.counterValues[counter.rawValue] = max(
            enabledAttribution.counterValues[counter.rawValue],
            amount);
        self.enabledAttribution = enabledAttribution;
    }

    /// Returns one accumulated counter, or zero when attribution is disabled.
    public func counterValue(_ counter: PerformanceCounter) -> UInt64 {
        guard let enabledAttribution = enabledAttribution else {
            return 0;
        }
        return enabledAttribution.counterValues[counter.rawValue];
    }

    /// Returns the cumulative logical expert-streaming payload bytes recorded so
    /// far for this request. Sampling it before and after one forward yields
    /// that forward's mandatory expert-page stream, which the completed-forward
    /// learning must exclude from activation evidence (issue #691).
    public func expertStreamingPayloadByteCount() -> UInt64 {
        counterValue(.rustExpertStreamingPayloadByteCount);
    }

    /// Produces the model-loading report and ends attribution for this request.
    public func finishModelLoading(
        _ modelLoadingMetadata: ModelLoadingPerformanceAttributionMetadata
    ) -> PerformanceAttributionReport? {
        guard let enabledAttribution = takeEnabledAttribution() else {
            return nil;
        }
        return PerformanceAttributionReport.modelLoading(
            ModelLoadingPerformanceAttributionReport(
                common: enabledAttribution.finishCommonReport(
                    outcome: modelLoadingMetadata.outcome),
                modelId: modelLoadingMetadata.modelId,
                modelRevision: modelLoadingMetadata.modelRevision,
                prefillTransientObservationCompleted: modelLoadingMetadata
                    .prefillTransientObservationCompleted,
                prefillObservedTransientHighWaterBytes: modelLoadingMetadata
                    .prefillObservedTransientHighWaterBytes,
                totalArtifactPayloadBytes: modelLoadingMetadata.totalArtifactPayloadBytes,
                residentModelPayloadBytes: modelLoadingMetadata.residentModelPayloadBytes,
                modelShardCount: modelLoadingMetadata.modelShardCount,
                mlxActiveMemoryBytes: modelLoadingMetadata.mlxActiveMemoryBytes,
                mlxAllocatorCacheMemoryBytes: modelLoadingMetadata
                    .mlxAllocatorCacheMemoryBytes,
                mlxPeakMemoryBytes: modelLoadingMetadata.mlxPeakMemoryBytes,
                failureDescription: modelLoadingMetadata.failureDescription));
    }

    /// Produces the generation report and ends attribution for this request.
    public func finishGeneration(
        _ generationMetadata: GenerationPerformanceAttributionMetadata
    ) -> PerformanceAttributionReport? {
        guard let enabledAttribution = takeEnabledAttribution() else {
            return nil;
        }
        let routeReuseRows = enabledAttribution.previousTokenExpertRouteReuseByLayer
            .enumerated()
            .filter { _, layerMeasurement in layerMeasurement.examinedLayerCount > 0 }
            .map { layerIndex, layerMeasurement in
                PreviousTokenExpertRouteReuseByLayerReport(
                    layerIndex: layerIndex,
                    predictedExpertCount: layerMeasurement.predictedExpertCount,
                    matchedExpertCount: layerMeasurement.matchedExpertCount,
                    completelyMatchedLayerCount: layerMeasurement
                        .completelyMatchedLayerCount,
                    examinedLayerCount: layerMeasurement.examinedLayerCount);
            };
        let sourceSummaries = enabledAttribution.expertStreamingSourceSummaries
            .compactMap { $0 };
        return PerformanceAttributionReport.generation(
            GenerationPerformanceAttributionReport(
                common: enabledAttribution.finishCommonReport(
                    outcome: generationMetadata.outcome),
                modelId: generationMetadata.modelId,
                modelRevision: generationMetadata.modelRevision,
                prefillTransientObservationCompleted: generationMetadata
                    .prefillTransientObservationCompleted,
                prefillObservedTransientHighWaterBytes: generationMetadata
                    .prefillObservedTransientHighWaterBytes,
                requestId: generationMetadata.requestId,
                configuredMaximumOutputTokens: generationMetadata
                    .configuredMaximumOutputTokens,
                mlxActiveMemoryBytes: generationMetadata.mlxActiveMemoryBytes,
                mlxAllocatorCacheMemoryBytes: generationMetadata
                    .mlxAllocatorCacheMemoryBytes,
                mlxPeakMemoryBytes: generationMetadata.mlxPeakMemoryBytes,
                failureDescription: generationMetadata.failureDescription,
                previousTokenExpertRouteReuseByLayer: routeReuseRows,
                expertStreamingSourceSummaries: sourceSummaries));
    }

    /// Produces the image-generation report and ends attribution for this request.
    public func finishImageGeneration(
        outcome: PerformanceAttributionOutcome,
        requestId: UInt64,
        modelId: String,
        modelRevision: String,
        widthPixels: UInt32,
        heightPixels: UInt32,
        steps: UInt16,
        guidanceThousandths: UInt32,
        seed: UInt64,
        encodedBytes: UInt64?,
        requestStartMemory: PerformanceMemoryObservation,
        finalCleanupMemory: PerformanceMemoryObservation,
        failureDescription: String?
    ) -> PerformanceAttributionReport? {
        guard let enabledAttribution = takeEnabledAttribution() else {
            return nil;
        }
        return PerformanceAttributionReport.imageGeneration(
            ImageGenerationPerformanceAttributionReport(
                common: enabledAttribution.finishCommonReport(outcome: outcome),
                requestId: requestId,
                modelId: modelId,
                modelRevision: modelRevision,
                widthPixels: widthPixels,
                heightPixels: heightPixels,
                steps: steps,
                guidanceThousandths: guidanceThousandths,
                seed: seed,
                encodedBytes: encodedBytes,
                memorySnapshots: [
                    imageMemorySnapshot("request_start", requestStartMemory),
                    imageMemorySnapshot("final_cleanup", finalCleanupMemory),
                ],
                failureDescription: failureDescription));
    }

    /// Produces the embeddings report and ends attribution for this request.
    public func finishEmbeddings(
        outcome: PerformanceAttributionOutcome,
        requestId: UInt64,
        modelId: String,
        inputCount: Int,
        totalInputTokens: UInt32,
        vectorWidth: UInt32,
        failureDescription: String?
    ) -> PerformanceAttributionReport? {
        guard let enabledAttribution = takeEnabledAttribution() else {
            return nil;
        }
        return PerformanceAttributionReport.embeddings(
            EmbeddingsPerformanceAttributionReport(
                common: enabledAttribution.finishCommonReport(outcome: outcome),
                requestId: requestId,
                modelId: modelId,
                inputCount: inputCount,
                totalInputTokens: totalInputTokens,
                vectorWidth: vectorWidth,
                failureDescription: failureDescription));
    }

    /// Ends attribution exactly once, mirroring the Rust report functions
    /// consuming `self`.
    private func takeEnabledAttribution() -> EnabledPerformanceAttribution? {
        let takenEnabledAttribution = enabledAttribution;
        enabledAttribution = nil;
        return takenEnabledAttribution;
    }
}

extension EnabledPerformanceAttribution {

    /// Assembles the fields shared by every report kind, then samples the
    /// closing process input/output boundary.
    func finishCommonReport(
        outcome: PerformanceAttributionOutcome
    ) -> CommonPerformanceAttributionReport {
        let reportElapsedNanoseconds = saturatingElapsedNanoseconds(
            DispatchTime.now().uptimeNanoseconds,
            since: reportStartedUptimeNanoseconds);
        var operationReports: [PerformanceOperationReport] = [];
        var attributedElapsedNanoseconds = UInt64(0);
        // Zip by catalog order: the arrays intentionally avoid a map allocation
        // on every measured operation.
        for (performanceOperation, operationMeasurement) in zip(
            PerformanceOperation.allCases,
            operationMeasurements)
        {
            if operationMeasurement.occurrenceCount == 0 {
                continue;
            }
            if performanceOperation.contributesToAttributedElapsed {
                attributedElapsedNanoseconds = attributedElapsedNanoseconds
                    &+ operationMeasurement.totalElapsedNanoseconds;
            }
            operationReports.append(
                PerformanceOperationReport(
                    operation: performanceOperation.identifier,
                    occurrenceCount: operationMeasurement.occurrenceCount,
                    totalElapsedNanoseconds: operationMeasurement.totalElapsedNanoseconds,
                    minimumElapsedNanoseconds: operationMeasurement.minimumElapsedNanoseconds,
                    maximumElapsedNanoseconds: operationMeasurement.maximumElapsedNanoseconds,
                    firstStartedOffsetNanoseconds: operationMeasurement
                        .firstStartedOffsetNanoseconds,
                    lastEndedOffsetNanoseconds: operationMeasurement
                        .lastEndedOffsetNanoseconds));
        }
        var counterReports: [PerformanceCounterReport] = [];
        for (performanceCounter, counterAmount) in zip(
            PerformanceCounter.allCases,
            counterValues)
        {
            if counterAmount == 0 {
                continue;
            }
            counterReports.append(
                PerformanceCounterReport(
                    counter: performanceCounter.identifier,
                    amount: counterAmount));
        }
        // Saturation keeps overlapping or clock-edge diagnostics from wrapping;
        // correctly classified leaf totals should remain within report elapsed.
        let unattributedElapsedNanoseconds = reportElapsedNanoseconds
            >= attributedElapsedNanoseconds
            ? reportElapsedNanoseconds - attributedElapsedNanoseconds
            : 0;
        let attributedPercent = reportElapsedNanoseconds == 0
            ? 0.0
            : Double(attributedElapsedNanoseconds)
                / Double(reportElapsedNanoseconds) * 100.0;
        // This end sample intentionally occurs after all report counters and
        // timings are finalized. The resulting interval therefore covers the
        // complete user-visible operation represented by this JSON row.
        let processIoEvidence = finishProcessIoEvidence(
            processIoStart: processIoStart);
        return CommonPerformanceAttributionReport(
            startedAtUnixMillis: reportStartedAtUnixMillis,
            endedAtUnixMillis: performanceAttributionUnixEpochMillis(),
            reportElapsedNanoseconds: reportElapsedNanoseconds,
            attributedElapsedNanoseconds: attributedElapsedNanoseconds,
            unattributedElapsedNanoseconds: unattributedElapsedNanoseconds,
            attributedPercent: attributedPercent,
            outcome: outcome,
            operations: operationReports,
            counters: counterReports,
            processPhysicalDiskReadBytes: processIoEvidence.physicalDiskReadBytes,
            processPhysicalDiskWrittenBytes: processIoEvidence.physicalDiskWrittenBytes,
            processIoUnavailabilityReason: processIoEvidence.unavailabilityReason);
    }
}

/// Process input/output evidence serialized as an all-or-unavailable triple.
func finishProcessIoEvidence(
    processIoStart: Result<MacosProcessIoSnapshot, MacosProcessIoError>
) -> (
    physicalDiskReadBytes: UInt64?,
    physicalDiskWrittenBytes: UInt64?,
    unavailabilityReason: String?
) {
    // Preserve one invariant in the serialized contract: either both byte
    // deltas are present, or both are null with a reason. Consumers must never
    // mistake an unavailable operating-system sample for measured zero traffic.
    switch processIoStart {
    case .failure(let startSamplingError):
        return (nil, nil, startSamplingError.description);
    case .success(let startSnapshot):
        switch MacosProcessIo.sampleCurrentProcessIo() {
        case .failure(let endSamplingError):
            return (nil, nil, endSamplingError.description);
        case .success(let endSnapshot):
            switch endSnapshot.deltaSince(startSnapshot) {
            case .failure(let deltaError):
                return (nil, nil, deltaError.description);
            case .success(let processIoDelta):
                return (
                    processIoDelta.physicalDiskReadBytes,
                    processIoDelta.physicalDiskWrittenBytes,
                    nil);
            }
        }
    }
}

func imageMemorySnapshot(
    _ phase: String,
    _ memory: PerformanceMemoryObservation
) -> ImageGenerationMemorySnapshot {
    ImageGenerationMemorySnapshot(
        phase: phase,
        mlxActiveMemoryBytes: memory.mlxActiveMemoryBytes,
        mlxAllocatorCacheMemoryBytes: memory.mlxAllocatorCacheMemoryBytes,
        mlxPeakMemoryBytes: memory.mlxPeakMemoryBytes);
}

/// Monotonic offset that can never wrap, mirroring Rust's saturating
/// `duration_since` on `Instant`.
func saturatingElapsedNanoseconds(
    _ laterUptimeNanoseconds: UInt64,
    since earlierUptimeNanoseconds: UInt64
) -> UInt64 {
    laterUptimeNanoseconds >= earlierUptimeNanoseconds
        ? laterUptimeNanoseconds - earlierUptimeNanoseconds
        : 0;
}

func performanceAttributionUnixEpochMillis() -> UInt64 {
    let secondsSinceEpoch = Date().timeIntervalSince1970;
    return secondsSinceEpoch <= 0
        ? 0
        : UInt64(secondsSinceEpoch * 1000);
}

