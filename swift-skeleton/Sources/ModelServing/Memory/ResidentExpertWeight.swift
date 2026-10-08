import Foundation

/// Opaque per-expert payload the decode cache stores but never inspects.
///
/// The cache uses only the byte count. Families own tensors, stacking,
/// and I/O.
public protocol ResidentExpertWeight {

    /// Bytes this expert's weights occupy in wired memory.
    var payloadBytes: UInt64 { get }
}
