import Foundation

/// Groups selected expert reads by shard file and builds the compact
/// per-shard manifests one expert page assembles from. Port of the Rust
/// `source_manifests` module: per-tensor page placements plus contiguous
/// expert runs become bounded intervals validated against both tilings —
/// no overlap in the shard file and no gaps in the virtual payload.
public enum QuantizedExpertSourceManifestBuilder {

    /**
     * Builds one shard manifest per shard file the tensor sources name.
     *
     * - Parameters:
     *   - tensorSources: The validated layer tensor sources.
     *   - selectedExpertIds: The validated ascending expert ids the page seats.
     * - Returns: One manifest per contributing shard, shard-name ordered.
     */
    public static func buildSourceManifests(
        tensorSources: [QuantizedTensorSource],
        selectedExpertIds: [Int]
    ) throws -> [QuantizedExpertShardManifest] {
        let sortedSourceFileNames: [String] = Set(tensorSources.map({ (tensorSource: QuantizedTensorSource) -> String in
            return tensorSource.sourceFileName
        })).sorted()
        var sourceManifests: [QuantizedExpertShardManifest] = []
        sourceManifests.reserveCapacity(sortedSourceFileNames.count)
        for sourceFileName: String in sortedSourceFileNames {
            var tensorRanges: [QuantizedExpertTensorRange] = []
            var sourceIntervals: [QuantizedExpertSourceInterval] = []
            var virtualPayloadOffsetBytes: UInt64 = 0
            for tensorSource: QuantizedTensorSource in tensorSources
            where tensorSource.sourceFileName == sourceFileName {
                let selectedTensorByteCount: Int = selectedExpertIds.count * tensorSource.bytesPerExpert
                tensorRanges.append(QuantizedExpertTensorRange(
                    tensorName: "\(tensorSource.projectionName).\(tensorSource.parameterName)",
                    projectionName: tensorSource.projectionName,
                    parameterName: tensorSource.parameterName,
                    dtype: tensorSource.dtype,
                    shape: [selectedExpertIds.count] + tensorSource.fullShape.dropFirst(),
                    virtualPayloadOffsetBytes: virtualPayloadOffsetBytes,
                    byteCount: selectedTensorByteCount))
                for contiguousRun: (expertStart: Int, expertCount: Int, firstPageSlot: Int)
                in QuantizedExpertSourceManifestBuilder.contiguousSelectedRuns(
                    expertIds: selectedExpertIds) {
                    sourceIntervals.append(QuantizedExpertSourceInterval(
                        tensorName: tensorSource.tensorName,
                        expertStart: contiguousRun.expertStart,
                        expertCount: contiguousRun.expertCount,
                        sourceFileOffsetBytes: tensorSource.tensorPayloadOffsetBytes
                            + UInt64(contiguousRun.expertStart * tensorSource.bytesPerExpert),
                        sourceByteCount: contiguousRun.expertCount * tensorSource.bytesPerExpert,
                        virtualPayloadOffsetBytes: virtualPayloadOffsetBytes
                            + UInt64(contiguousRun.firstPageSlot * tensorSource.bytesPerExpert)))
                }
                virtualPayloadOffsetBytes += UInt64(selectedTensorByteCount)
            }
            let orderedSourceIntervals: [QuantizedExpertSourceInterval] = sourceIntervals.sorted(by: {
                (leftInterval: QuantizedExpertSourceInterval,
                 rightInterval: QuantizedExpertSourceInterval) -> Bool in
                return leftInterval.sourceFileOffsetBytes < rightInterval.sourceFileOffsetBytes
            })
            try QuantizedExpertManifestValidation.validatedSourceIntervals(
                sourceIntervals: orderedSourceIntervals)
            try QuantizedExpertManifestValidation.validatedVirtualIntervals(
                sourceIntervals: orderedSourceIntervals,
                virtualPayloadByteCount: virtualPayloadOffsetBytes)
            sourceManifests.append(QuantizedExpertShardManifest(
                sourceFileName: sourceFileName,
                tensorRanges: tensorRanges,
                sourceIntervals: orderedSourceIntervals,
                payloadByteCount: virtualPayloadOffsetBytes))
        }
        return sourceManifests
    }

    /**
     * Groups adjacent selected experts into sequential file runs while
     * retaining each run's first compact page slot, so one run becomes one
     * bounded read.
     *
     * - Returns: `(expertStart, expertCount, firstPageSlot)` tuples in
     *   expert order.
     */
    public static func contiguousSelectedRuns(
        expertIds: [Int]
    ) -> [(expertStart: Int, expertCount: Int, firstPageSlot: Int)] {
        var selectedRuns: [(expertStart: Int, expertCount: Int, firstPageSlot: Int)] = []
        guard let firstExpertId: Int = expertIds.first else {
            return selectedRuns
        }
        var runStartExpertId: Int = firstExpertId
        var runFirstPageSlot: Int = 0
        for pageSlotIndex: Int in 1...expertIds.count {
            let runIsComplete: Bool = pageSlotIndex == expertIds.count
            if runIsComplete == false,
                expertIds[pageSlotIndex] == expertIds[pageSlotIndex - 1] + 1 {
                continue
            }
            let runEndExpertId: Int = expertIds[pageSlotIndex - 1]
            selectedRuns.append((
                expertStart: runStartExpertId,
                expertCount: runEndExpertId - runStartExpertId + 1,
                firstPageSlot: runFirstPageSlot))
            if runIsComplete == false {
                runStartExpertId = expertIds[pageSlotIndex]
                runFirstPageSlot = pageSlotIndex
            }
        }
        return selectedRuns
    }
}
