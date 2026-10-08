import Foundation

import RuntimeIntegration

/**
 * Typed failure while sampling or rejecting an MLX allocation, port of
 * the Rust `MlxAllocationAdmissionError` payload surface. The
 * runtime-backed admission owner that throws it lands with the worker
 * seam; the classification oracle needs the payload enum first because
 * expert paging surfaces these rejections as recoverable capacity
 * pressure.
 */
public enum MlxAllocationAdmissionError: Error, Equatable, Sendable {

    /// The pending allocation cannot be admitted at the stated boundary.
    case rejected(
        stage: String,
        boundary: MemoryBoundary,
        shortfallBytes: UInt64,
        activeMemoryBytes: UInt64,
        pendingAllocationBytes: UInt64,
        activeMemoryCeilingBytes: UInt64)

    /// Reading the MLX memory counters failed before an admission could
    /// be decided.
    case mlxRuntime(MlxRuntimeError)
}
