import Foundation

/// Static tensor contract for one named decoder-cache state component,
/// port of the Rust `DecoderCacheTensorLayout`.
public struct DecoderCacheTensorLayout: Equatable, Hashable, Sendable {

    public let tensorRoleName: String

    public let dtype: DecoderCacheTensorDtype

    /// Static tensor dimensions; a sequence dimension is represented by zero.
    public let dimensions: [Int]

    /// The token axis for a sequence-sliceable tensor.
    public let sequenceAxis: Int?

    /// Creates a fixed-shape tensor restored only from a boundary snapshot.
    public static func fixed(
        tensorRoleName: String,
        dtype: DecoderCacheTensorDtype,
        dimensions: [Int]
    ) -> DecoderCacheTensorLayout {
        DecoderCacheTensorLayout(
            tensorRoleName: tensorRoleName,
            dtype: dtype,
            dimensions: dimensions,
            sequenceAxis: nil)
    }

    /// Creates a tensor sliced and concatenated along one token axis.
    public static func sequence(
        tensorRoleName: String,
        dtype: DecoderCacheTensorDtype,
        dimensions: [Int],
        sequenceAxis: Int
    ) -> DecoderCacheTensorLayout {
        DecoderCacheTensorLayout(
            tensorRoleName: tensorRoleName,
            dtype: dtype,
            dimensions: dimensions,
            sequenceAxis: sequenceAxis)
    }

    private init(
        tensorRoleName: String,
        dtype: DecoderCacheTensorDtype,
        dimensions: [Int],
        sequenceAxis: Int?
    ) {
        self.tensorRoleName = tensorRoleName
        self.dtype = dtype
        self.dimensions = dimensions
        self.sequenceAxis = sequenceAxis
    }

    /// Returns the checked payload bytes for a fixed-shape tensor.
    public func fixedPayloadByteCount() throws -> Int {
        if sequenceAxis != nil || dimensions.contains(0) {
            throw DecoderCacheLayoutError.invalidTensorPayloadGeometry(
                tensorRoleName: tensorRoleName,
                description: "a fixed tensor must not contain a sequence axis or dynamic dimension")
        }
        return try Self.checkedTensorPayloadByteCount(
            tensorRoleName: tensorRoleName,
            dimensions: dimensions,
            dtype: dtype)
    }

    /// Returns checked payload bytes for one token of a sequence tensor.
    public func sequencePayloadByteCountPerToken() throws -> Int {
        guard let sequenceAxis else {
            throw DecoderCacheLayoutError.invalidTensorPayloadGeometry(
                tensorRoleName: tensorRoleName,
                description: "a sequence tensor must declare a sequence axis")
        }
        if sequenceAxis >= dimensions.count || dimensions[sequenceAxis] != 0 {
            throw DecoderCacheLayoutError.invalidTensorPayloadGeometry(
                tensorRoleName: tensorRoleName,
                description: "the sequence axis must contain the dynamic dimension")
        }
        var oneTokenDimensions = dimensions
        oneTokenDimensions[sequenceAxis] = 1
        return try Self.checkedTensorPayloadByteCount(
            tensorRoleName: tensorRoleName,
            dimensions: oneTokenDimensions,
            dtype: dtype)
    }

    private static func checkedTensorPayloadByteCount(
        tensorRoleName: String,
        dimensions: [Int],
        dtype: DecoderCacheTensorDtype
    ) throws -> Int {
        var elementCount = 1
        for dimension in dimensions {
            let (multipliedElementCount, multiplicationOverflowed) =
                elementCount.multipliedReportingOverflow(by: dimension)
            if multiplicationOverflowed {
                throw DecoderCacheLayoutError.tensorPayloadByteCountOverflow(
                    tensorRoleName: tensorRoleName)
            }
            elementCount = multipliedElementCount
        }
        let (payloadByteCount, multiplicationOverflowed) =
            elementCount.multipliedReportingOverflow(by: dtype.scalarByteCount)
        if multiplicationOverflowed {
            throw DecoderCacheLayoutError.tensorPayloadByteCountOverflow(
                tensorRoleName: tensorRoleName)
        }
        return payloadByteCount
    }
}
