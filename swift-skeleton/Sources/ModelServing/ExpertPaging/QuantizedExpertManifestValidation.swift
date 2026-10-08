import Foundation

/// Pure startup validation for quantized expert manifest construction.
/// Port of the Rust `quantized_expert_validation` module: no I/O, no
/// model knowledge — every function operates solely on its arguments and
/// fails closed through `ExpertPagingError.manifestValidationFailure`.
public enum QuantizedExpertManifestValidation {

    /**
     * Validates that expert ids are non-empty, strictly ascending, and
     * inside the layer's expert capacity.
     *
     * - Returns: The validated expert ids unchanged.
     */
    public static func validatedExpertIds(
        expertIds: [Int],
        expertCapacity: Int
    ) throws -> [Int] {
        guard expertIds.isEmpty == false else {
            throw ExpertPagingError.manifestValidationFailure(
                description: "quantized expert page must select at least one expert")
        }
        if let maximumSelectedExpertId: Int = expertIds.max(),
            maximumSelectedExpertId >= expertCapacity {
            throw ExpertPagingError.manifestValidationFailure(
                description: "selected expert ids exceed the layer's expert capacity "
                    + "of \(expertCapacity): max selected \(maximumSelectedExpertId)")
        }
        for expertIndex: Int in 1..<expertIds.count
        where expertIds[expertIndex] <= expertIds[expertIndex - 1] {
            throw ExpertPagingError.manifestValidationFailure(
                description: "expert ids must be unique and strictly ascending")
        }
        return expertIds
    }

    /**
     * Validates the affine quantization contract: positive bits and group
     * size whose packed group occupies whole U32 elements.
     */
    public static func validatedQuantizationContract(
        quantizationBits: Int32,
        quantizationGroupSize: Int32
    ) throws {
        guard quantizationBits > 0 else {
            throw ExpertPagingError.manifestValidationFailure(
                description: "quantization bits must be a positive integer")
        }
        guard quantizationGroupSize > 0 else {
            throw ExpertPagingError.manifestValidationFailure(
                description: "quantization group_size must be a positive integer")
        }
        let (packedGroupBitCount, multiplyOverflowed): (Int, Bool) = Int(quantizationGroupSize)
            .multipliedReportingOverflow(by: Int(quantizationBits))
        if multiplyOverflowed || packedGroupBitCount % 32 != 0 {
            throw ExpertPagingError.manifestValidationFailure(
                description: "quantized expert groups must pack into whole U32 elements: "
                    + "bits=\(quantizationBits), group_size=\(quantizationGroupSize)")
        }
    }

    /**
     * Rejects zero-length or overflowing source intervals, and intervals
     * that overlap inside the source file when iterated in file order.
     */
    public static func validatedSourceIntervals(
        sourceIntervals: [QuantizedExpertSourceInterval]
    ) throws {
        var previousSourceEndBytes: UInt64? = nil
        for sourceInterval: QuantizedExpertSourceInterval in sourceIntervals {
            guard sourceInterval.sourceByteCount > 0 else {
                throw ExpertPagingError.manifestValidationFailure(
                    description: "source interval at offset "
                        + "\(sourceInterval.sourceFileOffsetBytes) is empty")
            }
            let (sourceEndBytes, endOverflowed): (UInt64, Bool) = sourceInterval.sourceFileOffsetBytes
                .addingReportingOverflow(UInt64(sourceInterval.sourceByteCount))
            guard endOverflowed == false else {
                throw ExpertPagingError.manifestValidationFailure(
                    description: "source interval offset \(sourceInterval.sourceFileOffsetBytes) "
                        + "overflows past the file end")
            }
            if let previousSourceEndBytes: UInt64 = previousSourceEndBytes,
                sourceInterval.sourceFileOffsetBytes < previousSourceEndBytes {
                throw ExpertPagingError.manifestValidationFailure(
                    description: "source intervals overlap: interval at offset "
                        + "\(sourceInterval.sourceFileOffsetBytes) overlaps previous")
            }
            previousSourceEndBytes = sourceEndBytes
        }
    }

    /**
     * Requires exact compact virtual coverage: sorted by virtual offset the
     * intervals chain from zero without gaps and end exactly at the
     * declared payload byte count.
     */
    public static func validatedVirtualIntervals(
        sourceIntervals: [QuantizedExpertSourceInterval],
        virtualPayloadByteCount: UInt64
    ) throws {
        let sortedSourceIntervals: [QuantizedExpertSourceInterval] = sourceIntervals.sorted(by: {
            (leftInterval: QuantizedExpertSourceInterval,
             rightInterval: QuantizedExpertSourceInterval) -> Bool in
            return leftInterval.virtualPayloadOffsetBytes < rightInterval.virtualPayloadOffsetBytes
        })
        var expectedVirtualOffsetBytes: UInt64 = 0
        for sourceInterval: QuantizedExpertSourceInterval in sortedSourceIntervals {
            guard sourceInterval.virtualPayloadOffsetBytes == expectedVirtualOffsetBytes else {
                throw ExpertPagingError.manifestValidationFailure(
                    description: "virtual intervals are not contiguous: expected offset "
                        + "\(expectedVirtualOffsetBytes), found "
                        + "\(sourceInterval.virtualPayloadOffsetBytes)")
            }
            let (chainedOffsetBytes, addOverflowed): (UInt64, Bool) = expectedVirtualOffsetBytes
                .addingReportingOverflow(UInt64(sourceInterval.sourceByteCount))
            if addOverflowed {
                throw ExpertPagingError.manifestValidationFailure(
                    description: "virtual intervals overflow the declared payload "
                        + "\(virtualPayloadByteCount)")
            }
            expectedVirtualOffsetBytes = chainedOffsetBytes
        }
        guard expectedVirtualOffsetBytes == virtualPayloadByteCount else {
            throw ExpertPagingError.manifestValidationFailure(
                description: "virtual intervals cover \(expectedVirtualOffsetBytes) bytes "
                    + "but the page declares \(virtualPayloadByteCount)")
        }
    }
}
