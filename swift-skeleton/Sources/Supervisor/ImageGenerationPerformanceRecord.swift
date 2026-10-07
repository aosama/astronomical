import Foundation

import IpcProtocol

/**
 * One finalized image request's end-to-end performance attribution,
 * mirroring the Rust record from apps/supervisor/src/generation_performance_log.rs.
 */
public struct ImageGenerationPerformanceRecord: Equatable {

    public let operation: String

    public let timestampMillis: UInt64

    public let requestId: UInt64

    public let modelId: String

    public let widthPixels: UInt32

    public let heightPixels: UInt32

    public let steps: UInt16

    public let completionOutcome: String

    public let totalElapsedMillis: UInt64

    public let queueWaitElapsedMillis: UInt64

    public let swapLoadElapsedMillis: UInt64

    public let executionElapsedMillis: UInt64

    public let finalizationElapsedMillis: UInt64

    public let workerReportedElapsedMillis: UInt64

    public let encodedImageBytes: UInt64?

    public let mlxPeakMemoryBytes: UInt64?

    public let mlxActiveMemoryBytes: UInt64?

    public init(
        operation: String,
        timestampMillis: UInt64,
        requestId: UInt64,
        modelId: String,
        widthPixels: UInt32,
        heightPixels: UInt32,
        steps: UInt16,
        completionOutcome: String,
        totalElapsedMillis: UInt64,
        queueWaitElapsedMillis: UInt64,
        swapLoadElapsedMillis: UInt64,
        executionElapsedMillis: UInt64,
        finalizationElapsedMillis: UInt64,
        workerReportedElapsedMillis: UInt64,
        encodedImageBytes: UInt64?,
        mlxPeakMemoryBytes: UInt64?,
        mlxActiveMemoryBytes: UInt64?
    ) {
        self.operation = operation
        self.timestampMillis = timestampMillis
        self.requestId = requestId
        self.modelId = modelId
        self.widthPixels = widthPixels
        self.heightPixels = heightPixels
        self.steps = steps
        self.completionOutcome = completionOutcome
        self.totalElapsedMillis = totalElapsedMillis
        self.queueWaitElapsedMillis = queueWaitElapsedMillis
        self.swapLoadElapsedMillis = swapLoadElapsedMillis
        self.executionElapsedMillis = executionElapsedMillis
        self.finalizationElapsedMillis = finalizationElapsedMillis
        self.workerReportedElapsedMillis = workerReportedElapsedMillis
        self.encodedImageBytes = encodedImageBytes
        self.mlxPeakMemoryBytes = mlxPeakMemoryBytes
        self.mlxActiveMemoryBytes = mlxActiveMemoryBytes
    }

    /// The serde-shaped JSON object written to `performance.jsonl`.
    public func jsonlWireValue() -> JsonWireValue {
        var wireObject: JsonWireObject = JsonWireObject(entries: [])
        wireObject.appendEntry(key: "operation", value: .string(self.operation))
        wireObject.appendEntry(key: "timestamp_millis", value: .unsignedInteger(self.timestampMillis))
        wireObject.appendEntry(key: "request_id", value: .unsignedInteger(self.requestId))
        wireObject.appendEntry(key: "model_id", value: .string(self.modelId))
        wireObject.appendEntry(key: "width_pixels", value: .unsignedInteger(UInt64(self.widthPixels)))
        wireObject.appendEntry(key: "height_pixels", value: .unsignedInteger(UInt64(self.heightPixels)))
        wireObject.appendEntry(key: "steps", value: .unsignedInteger(UInt64(self.steps)))
        wireObject.appendEntry(key: "completion_outcome", value: .string(self.completionOutcome))
        wireObject.appendEntry(key: "total_elapsed_millis", value: .unsignedInteger(self.totalElapsedMillis))
        wireObject.appendEntry(key: "queue_wait_elapsed_millis", value: .unsignedInteger(self.queueWaitElapsedMillis))
        wireObject.appendEntry(key: "swap_load_elapsed_millis", value: .unsignedInteger(self.swapLoadElapsedMillis))
        wireObject.appendEntry(key: "execution_elapsed_millis", value: .unsignedInteger(self.executionElapsedMillis))
        wireObject.appendEntry(key: "finalization_elapsed_millis", value: .unsignedInteger(self.finalizationElapsedMillis))
        wireObject.appendEntry(key: "worker_reported_elapsed_millis", value: .unsignedInteger(self.workerReportedElapsedMillis))
        wireObject.appendEntry(key: "encoded_image_bytes", value: ImageGenerationPerformanceRecord.optionalUInteger(self.encodedImageBytes))
        wireObject.appendEntry(key: "mlx_peak_memory_bytes", value: ImageGenerationPerformanceRecord.optionalUInteger(self.mlxPeakMemoryBytes))
        wireObject.appendEntry(key: "mlx_active_memory_bytes", value: ImageGenerationPerformanceRecord.optionalUInteger(self.mlxActiveMemoryBytes))
        return .object(wireObject)
    }

    private static func optionalUInteger(_ optionalValue: UInt64?) -> JsonWireValue {
        guard let unwrappedValue: UInt64 = optionalValue else {
            return .null
        }
        return .unsignedInteger(unwrappedValue)
    }
}
