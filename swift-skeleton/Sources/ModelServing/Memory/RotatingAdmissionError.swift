import Foundation

/// Errors from rotating-cache admission arithmetic, port of the Rust
/// `RotatingAdmissionError`.
public enum RotatingAdmissionError: Error, Equatable, Sendable {

    /// Window size was zero.
    case zeroWindowSize

    /// Window plus chunk overflowed the destination integer.
    case transientTokenCountOverflow(windowSize: UInt32, promptChunkTokenCount: UInt32)
}
