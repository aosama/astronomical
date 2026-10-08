import Foundation

/// Errors from sliding-window visibility construction, port of the Rust
/// `SlidingWindowVisibilityError`.
public enum SlidingWindowVisibilityError: Error, Equatable, Sendable {

    /// Window size was zero.
    case zeroWindowSize

    /// Query or key token count was zero.
    case zeroTokenCount(description: String)
}
