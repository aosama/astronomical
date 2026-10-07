import Foundation

import MLX

/**
 * Loads a SafeTensors payload from explicit source-file ranges — the
 * expert-paged read path — continuing the Rust
 * `BoundedRangeSafetensorsLoader` contract: validate that the intervals
 * tile the virtual payload exactly and never overlap in the source file,
 * gather them positionally at the machine-adaptive concurrency ceiling,
 * then decode header-plus-payload with MLX's stock in-memory loader.
 */
enum BoundedRangeSafetensorsLoader {

    /**
     * - Parameters:
     *   - sourceFile: the open weights file the intervals read from.
     *   - syntheticHeaderBytes: the SafeTensors header (length prefix plus
     *     JSON) synthesized for the virtual payload.
     *   - intervals: the source ranges composing the virtual payload.
     *   - totalPayloadBytes: the exact virtual payload byte count.
     *   - expertFileReadMetrics: optional instrumentation; every chunk read
     *     is measured for overlap, latency, and volume when attached.
     * - Returns: the decoded tensors plus the retained assembled bytes.
     * - Throws: `MlxRuntimeError.boundedIntervalValidation` when the
     *   intervals do not tile the payload or overlap in the source;
     *   `MlxRuntimeError.positionalReadFailed` when the read fails; and
     *   MLX's native decode error for malformed SafeTensors bytes.
     */
    static func load(
        sourceFile: FileHandle,
        syntheticHeaderBytes: Data,
        intervals: [BoundedReadInterval],
        totalPayloadBytes: UInt64,
        expertFileReadMetrics: PositionalFileReadMetrics?
    ) throws -> SafetensorsFile {
        try validateIntervals(intervals: intervals, totalPayloadBytes: totalPayloadBytes)
        let payloadRequests: [PositionalFileReader.ReadRequest] = intervals.map({ (interval: BoundedReadInterval) -> PositionalFileReader.ReadRequest in
            return PositionalFileReader.ReadRequest(
                sourceFileOffset: interval.sourceFileOffset,
                destinationBufferOffset: syntheticHeaderBytes.count + Int(interval.virtualPayloadOffset),
                byteCount: interval.sourceByteCount)
        })
        let assembledBytes: Data = try PositionalFileReader.read(
            fileDescriptor: sourceFile.fileDescriptor,
            headerBytes: syntheticHeaderBytes,
            payloadRequests: payloadRequests,
            maximumConcurrentReadCount: PositionalFileReader.DEFAULT_MAXIMUM_CONCURRENT_READ_COUNT,
            metrics: expertFileReadMetrics)
        let tensorsByName: [String: MLXArray] = try MLX.loadArrays(data: assembledBytes)
        return SafetensorsFile(tensorsByName: tensorsByName, assembledBytes: assembledBytes)
    }

    /**
     * Enforces both tiling invariants: sorted by virtual payload offset
     * the chain must start at zero, be contiguous, and end exactly at
     * `totalPayloadBytes`; sorted by source offset no two ranges may
     * overlap inside the source file.
     *
     * - Throws: `MlxRuntimeError.boundedIntervalValidation` naming the
     *   violated invariant.
     */
    static func validateIntervals(
        intervals: [BoundedReadInterval],
        totalPayloadBytes: UInt64
    ) throws {
        let byVirtualOffset: [BoundedReadInterval] = intervals.sorted(by: { (leftInterval: BoundedReadInterval, rightInterval: BoundedReadInterval) -> Bool in
            return leftInterval.virtualPayloadOffset < rightInterval.virtualPayloadOffset
        })
        guard let firstInterval: BoundedReadInterval = byVirtualOffset.first else {
            throw MlxRuntimeError.boundedIntervalValidation(
                description: "at least one bounded read interval is required")
        }
        guard firstInterval.virtualPayloadOffset == 0 else {
            throw MlxRuntimeError.boundedIntervalValidation(
                description: "the virtual payload chain must start at offset 0, not \(firstInterval.virtualPayloadOffset)")
        }
        var previousInterval: BoundedReadInterval = firstInterval
        for currentInterval: BoundedReadInterval in byVirtualOffset.dropFirst() {
            let expectedVirtualOffset: UInt64 =
                previousInterval.virtualPayloadOffset + UInt64(previousInterval.sourceByteCount)
            guard currentInterval.virtualPayloadOffset == expectedVirtualOffset else {
                throw MlxRuntimeError.boundedIntervalValidation(
                    description: "the virtual payload chain is not contiguous: expected offset \(expectedVirtualOffset), found \(currentInterval.virtualPayloadOffset)")
            }
            previousInterval = currentInterval
        }
        let chainedPayloadBytes: UInt64 =
            previousInterval.virtualPayloadOffset + UInt64(previousInterval.sourceByteCount)
        guard chainedPayloadBytes == totalPayloadBytes else {
            throw MlxRuntimeError.boundedIntervalValidation(
                description: "the virtual payload chain ends at \(chainedPayloadBytes) bytes, expected \(totalPayloadBytes)")
        }

        let bySourceOffset: [BoundedReadInterval] = intervals.sorted(by: { (leftInterval: BoundedReadInterval, rightInterval: BoundedReadInterval) -> Bool in
            return leftInterval.sourceFileOffset < rightInterval.sourceFileOffset
        })
        var previousSourceInterval: BoundedReadInterval = bySourceOffset[0]
        for currentInterval: BoundedReadInterval in bySourceOffset.dropFirst() {
            let previousSourceEnd: UInt64 =
                previousSourceInterval.sourceFileOffset + UInt64(previousSourceInterval.sourceByteCount)
            guard currentInterval.sourceFileOffset >= previousSourceEnd else {
                throw MlxRuntimeError.boundedIntervalValidation(
                    description: "source ranges overlap at offset \(currentInterval.sourceFileOffset), previous range ends at \(previousSourceEnd)")
            }
            previousSourceInterval = currentInterval
        }
    }
}
