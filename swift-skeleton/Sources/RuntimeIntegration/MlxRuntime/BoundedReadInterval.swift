import Foundation

/**
 * One source-file range mapped into the virtual payload of a bounded
 * (expert-paged) SafeTensors load, continuing the Rust
 * `BoundedReadInterval` contract.
 *
 * The pair of offsets is the mapping itself: bytes at
 * `sourceFileOffset ..< sourceFileOffset + sourceByteCount` in the weights
 * file belong at `virtualPayloadOffset` of the payload that follows the
 * synthetic header. Validators guarantee the intervals tile the virtual
 * payload exactly and never overlap in the source file.
 */
public struct BoundedReadInterval: Equatable, Sendable {

    /// Where this range's bytes belong in the virtual payload.
    public let virtualPayloadOffset: UInt64

    /// Where this range's bytes start in the source weights file.
    public let sourceFileOffset: UInt64

    /// How many bytes this range contributes.
    public let sourceByteCount: Int

    public init(virtualPayloadOffset: UInt64, sourceFileOffset: UInt64, sourceByteCount: Int) {
        self.virtualPayloadOffset = virtualPayloadOffset
        self.sourceFileOffset = sourceFileOffset
        self.sourceByteCount = sourceByteCount
    }
}
