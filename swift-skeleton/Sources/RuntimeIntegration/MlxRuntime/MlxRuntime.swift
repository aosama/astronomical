import Foundation

import MLX

/**
 * The SafeTensors I/O surface of the MLX runtime, continuing the Rust
 * `MlxRuntime` namespace at the Swift boundary. The GPU device facts stay
 * on the enum in `GpuDeviceInfo.swift`; this extension carries the I/O
 * API: retained whole-file loads, bounded-range (expert-paged) loads,
 * URL saves, and (bounded) in-memory serialization — all delegating to
 * MLX's stock I/O primitives instead of custom serialization, per the
 * repo's stock-layer-first rule.
 */
extension MlxRuntime {

    /**
     * Loads a whole SafeTensors weights file eagerly into one retained
     * buffer.
     *
     * - Parameters:
     *   - weightsFile: the open weights file descriptor to read from.
     *   - positionalFileReadMetrics: optional instrumentation for the read.
     * - Returns: the decoded tensors plus the retained assembled bytes.
     */
    public static func loadSafetensors(
        weightsFile: FileHandle,
        positionalFileReadMetrics: PositionalFileReadMetrics? = nil
    ) throws -> SafetensorsFile {
        return try RetainedFileSafetensorsLoader.load(
            weightsFile: weightsFile,
            positionalFileReadMetrics: positionalFileReadMetrics)
    }

    /**
     * Loads a SafeTensors payload from explicit source-file ranges — the
     * expert-paged read path — gathered at the machine-adaptive
     * concurrency ceiling.
     *
     * - Parameters:
     *   - sourceFile: the open weights file the intervals read from.
     *   - syntheticHeaderBytes: the SafeTensors header synthesized for the
     *     virtual payload.
     *   - intervals: the source ranges composing the virtual payload.
     *   - totalPayloadBytes: the exact virtual payload byte count.
     *   - expertFileReadMetrics: optional instrumentation for the reads.
     * - Returns: the decoded tensors plus the retained assembled bytes.
     */
    public static func loadSafetensorsFromBoundedRanges(
        sourceFile: FileHandle,
        syntheticHeaderBytes: Data,
        intervals: [BoundedReadInterval],
        totalPayloadBytes: UInt64,
        expertFileReadMetrics: PositionalFileReadMetrics? = nil
    ) throws -> SafetensorsFile {
        return try BoundedRangeSafetensorsLoader.load(
            sourceFile: sourceFile,
            syntheticHeaderBytes: syntheticHeaderBytes,
            intervals: intervals,
            totalPayloadBytes: totalPayloadBytes,
            expertFileReadMetrics: expertFileReadMetrics)
    }

    /**
     * Writes named arrays to a `.safetensors` file through MLX's stock
     * saver, probing the destination descriptor first.
     *
     * - Returns: the measured on-disk byte count of the completed write.
     */
    public static func saveSafetensors(
        arrays: [String: MLXArray],
        metadata: [String: String],
        to destinationUrl: URL
    ) throws -> SafetensorsWriteOutcome {
        return try SafetensorsFileWriter.save(
            arrays: arrays,
            metadata: metadata,
            to: destinationUrl)
    }

    /**
     * Serializes named arrays to SafeTensors bytes in memory, unbounded.
     *
     * - Returns: the serialized bytes.
     */
    public static func serializeSafetensors(
        arrays: [String: MLXArray],
        metadata: [String: String]
    ) throws -> Data {
        return try BoundedSafetensorsSerializer.serialize(arrays: arrays, metadata: metadata)
    }

    /**
     * Serializes named arrays to SafeTensors bytes in memory, refusing
     * byte counts above the caller's ceiling.
     *
     * - Returns: the serialized bytes when they fit `maximumByteCount`.
     */
    public static func serializeSafetensors(
        arrays: [String: MLXArray],
        metadata: [String: String],
        maximumByteCount: Int
    ) throws -> Data {
        return try BoundedSafetensorsSerializer.serialize(
            arrays: arrays,
            metadata: metadata,
            maximumByteCount: maximumByteCount)
    }
}
