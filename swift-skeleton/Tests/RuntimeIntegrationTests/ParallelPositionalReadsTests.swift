import Foundation

import MLX
import Testing
import JourneyCategories
import ModelServingTestSupport
import RuntimeIntegration

/**
 * Env-gated SafeTensors positional-read concurrency journey, continuing
 * the Rust `parallel_positional_reads` test: four 40 MiB tensor partitions
 * of one weights file are gathered through bounded read intervals at the
 * machine-adaptive concurrency ceiling, and the read instrumentation must
 * show overlapping reads on any machine with more than one active core.
 *
 * Gated behind `ASTRONOMICAL_SAFETENSORS_CONCURRENCY_JOURNEY` because the
 * journey writes and reads 160 MiB of disk traffic on demand.
 */
@Suite(
    .serialized,
    .tags(.hermeticMlxJourney, .realModelJourney),
    .enabled(if: RealModelJourneyGate.safetensorsConcurrencyJourneyEnabled()))
final class ParallelPositionalReadsTests {

    init() {
        signal(SIGPIPE, SIG_IGN)
        MLXMetallibLocator.overrideMetallibPathIfNecessary()
    }

    @Test(.timeLimit(.minutes(2)))
    func should_overlap_positional_reads_across_four_tensor_partitions() throws {
        let partitionFileUrl: URL = SafetensorsFixtureSupport.temporaryFileUrl("parallel-positional-reads.safetensors")
        let headerJson: String = SafetensorsConcurrencyFixtureSupport.fourTensorHeaderJson()
        let headerJsonBytes: [UInt8] = Array(headerJson.utf8)
        var syntheticHeaderBytes: [UInt8] = SafetensorsFixtureSupport.littleEndianLengthPrefix(of: UInt64(headerJsonBytes.count))
        syntheticHeaderBytes.append(contentsOf: headerJsonBytes)
        let payloadOffsetBytes: UInt64 = UInt64(syntheticHeaderBytes.count)

        FileManager.default.createFile(atPath: partitionFileUrl.path, contents: Data(syntheticHeaderBytes))
        defer {
            try? FileManager.default.removeItem(at: partitionFileUrl)
        }

        let writeHandle: FileHandle = try FileHandle(forWritingTo: partitionFileUrl)
        try writeHandle.seekToEnd()
        try writeHandle.write(
            contentsOf: Data(count: SafetensorsConcurrencyFixtureSupport.CONCURRENT_READ_TENSOR_BYTE_COUNT * 4))
        try SafetensorsConcurrencyFixtureSupport.writePartitionBoundaryMarkers(
            to: writeHandle,
            payloadOffsetBytes: payloadOffsetBytes)
        try writeHandle.close()

        let readHandle: FileHandle = try FileHandle(forReadingFrom: partitionFileUrl)
        defer {
            readHandle.closeFile()
        }

        let readIntervals: [BoundedReadInterval] = (0..<4).map({ (tensorIndex: Int) -> BoundedReadInterval in
            let partitionOffsetBytes: UInt64 =
                UInt64(tensorIndex * SafetensorsConcurrencyFixtureSupport.CONCURRENT_READ_TENSOR_BYTE_COUNT)
            return BoundedReadInterval(
                virtualPayloadOffset: partitionOffsetBytes,
                sourceFileOffset: payloadOffsetBytes + partitionOffsetBytes,
                sourceByteCount: SafetensorsConcurrencyFixtureSupport.CONCURRENT_READ_TENSOR_BYTE_COUNT)
        })
        let expertFileReadMetrics: PositionalFileReadMetrics = PositionalFileReadMetrics()
        let weightsFile: SafetensorsFile = try MlxRuntime.loadSafetensorsFromBoundedRanges(
            sourceFile: readHandle,
            syntheticHeaderBytes: Data(syntheticHeaderBytes),
            intervals: readIntervals,
            totalPayloadBytes: UInt64(SafetensorsConcurrencyFixtureSupport.CONCURRENT_READ_TENSOR_BYTE_COUNT * 4),
            expertFileReadMetrics: expertFileReadMetrics)

        var firstTensorHolder: MLXArray? = nil
        var secondTensorHolder: MLXArray? = nil
        var thirdTensorHolder: MLXArray? = nil
        var fourthTensorHolder: MLXArray? = nil
        do {
            let firstTensor: MLXArray = try weightsFile.tensor("first.weight")
            MLX.asyncEval([firstTensor])
            firstTensorHolder = firstTensor
            let secondTensor: MLXArray = try weightsFile.tensor("second.weight")
            MLX.asyncEval([secondTensor])
            secondTensorHolder = secondTensor
            let thirdTensor: MLXArray = try weightsFile.tensor("third.weight")
            MLX.asyncEval([thirdTensor])
            thirdTensorHolder = thirdTensor
            let fourthTensor: MLXArray = try weightsFile.tensor("fourth.weight")
            MLX.asyncEval([fourthTensor])
            fourthTensorHolder = fourthTensor
        }
        guard let firstTensor: MLXArray = firstTensorHolder,
              let secondTensor: MLXArray = secondTensorHolder,
              let thirdTensor: MLXArray = thirdTensorHolder,
              let fourthTensor: MLXArray = fourthTensorHolder else {
            Issue.record("all four partition tensors must load under their header names")
            return
        }

        let evaluationStartedAtNanoseconds: UInt64 = DispatchTime.now().uptimeNanoseconds
        MLX.eval([firstTensor, secondTensor, thirdTensor, fourthTensor])
        let evaluationElapsedNanoseconds: Int =
            Int(DispatchTime.now().uptimeNanoseconds - evaluationStartedAtNanoseconds)

        // Every boundary marker lands inside the first partition, so the
        // first tensor alone carries all four marker values at the element
        // positions implied by their byte offsets.
        let firstTensorValues: [Float] = firstTensor.asArray(Float.self)
        let elementCountPerTensor: Int = SafetensorsConcurrencyFixtureSupport.CONCURRENT_READ_TENSOR_BYTE_COUNT / 4
        #expect(firstTensorValues[0] == 1)
        #expect(firstTensorValues[elementCountPerTensor / 2 - 1] == 2)
        #expect(firstTensorValues[elementCountPerTensor / 2] == 3)
        #expect(firstTensorValues[elementCountPerTensor - 1] == 4)

        let readSnapshot: PositionalFileReadSnapshot = expertFileReadMetrics.snapshot()
        let availableReadParallelism: Int = min(4, ProcessInfo.processInfo.activeProcessorCount)
        let evidenceLine: String = "[safetensors-positional-read] status=success tensors=4 reads=\(readSnapshot.readCallCount) bytes=\(readSnapshot.readByteCount) maximum_concurrent_reads=\(readSnapshot.maximumConcurrentReadCount) summed_read_milliseconds=\(readSnapshot.totalReadElapsedNanoseconds / 1_000_000) evaluation_milliseconds=\(evaluationElapsedNanoseconds / 1_000_000)"
        FileHandle.standardError.write(Data(evidenceLine.utf8))

        if (availableReadParallelism > 1) {
            #expect(readSnapshot.readCallCount > 4)
            #expect(readSnapshot.maximumConcurrentReadCount >= 2)
        }
        #expect(readSnapshot.readByteCount == 167_772_160)
        #expect(readSnapshot.maximumConcurrentReadCount <= availableReadParallelism)
        #expect(readSnapshot.readFailureCount == 0)
    }
}

private enum SafetensorsConcurrencyFixtureSupport {

    static let CONCURRENT_READ_TENSOR_BYTE_COUNT: Int = 40 * 1024 * 1024

    static func fourTensorHeaderJson() -> String {
        let elementCountPerTensor: Int = CONCURRENT_READ_TENSOR_BYTE_COUNT / 4
        let tensorNames: [String] = ["first.weight", "second.weight", "third.weight", "fourth.weight"]
        let tensorHeaderEntries: [String] = tensorNames.enumerated().map({ (nameAndIndex: (offset: Int, element: String)) -> String in
            let tensorStartByteOffset: Int = nameAndIndex.offset * CONCURRENT_READ_TENSOR_BYTE_COUNT
            let tensorEndByteOffset: Int = tensorStartByteOffset + CONCURRENT_READ_TENSOR_BYTE_COUNT
            return "\"\(nameAndIndex.element)\":{\"dtype\":\"F32\",\"shape\":[\(elementCountPerTensor)],\"data_offsets\":[\(tensorStartByteOffset),\(tensorEndByteOffset)]}"
        })
        return "{" + tensorHeaderEntries.joined(separator: ",") + "}"
    }

    /**
     * Plants four distinct marker floats at partition boundaries — start,
     * both sides of the half-partition seam, and the partition end — all
     * inside the first partition's byte range, so one array read proves
     * the assembled payload kept its ordering.
     */
    static func writePartitionBoundaryMarkers(
        to writeHandle: FileHandle,
        payloadOffsetBytes: UInt64
    ) throws {
        let halfPartitionByteCount: Int = CONCURRENT_READ_TENSOR_BYTE_COUNT / 2
        let boundaryMarkers: [(byteOffsetInPayload: Int, markerValue: Float)] = [
            (byteOffsetInPayload: 0, markerValue: 1),
            (byteOffsetInPayload: halfPartitionByteCount - 4, markerValue: 2),
            (byteOffsetInPayload: halfPartitionByteCount, markerValue: 3),
            (byteOffsetInPayload: CONCURRENT_READ_TENSOR_BYTE_COUNT - 4, markerValue: 4),
        ]
        for boundaryMarker in boundaryMarkers {
            let absoluteFileOffset: UInt64 = payloadOffsetBytes + UInt64(boundaryMarker.byteOffsetInPayload)
            try writeHandle.seek(toOffset: absoluteFileOffset)
            try writeHandle.write(contentsOf: Data(SafetensorsFixtureSupport.littleEndianBytes(of: [boundaryMarker.markerValue])))
        }
    }
}
