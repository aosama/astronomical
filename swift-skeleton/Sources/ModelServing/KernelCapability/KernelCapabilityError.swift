import Foundation;

/// The failure evidence a probe reports, port of the Rust
/// `KernelCapabilityError`.
public enum KernelCapabilityError: Equatable, Error, Sendable {

    case compilation(description: String);
    case execution(description: String);
    case outputMismatch(description: String);

    /// The retained evidence restated as the reason stored in a verdict, so
    /// probe failures lose no description when they become unsupported
    /// verdicts.
    var unsupportedReason: KernelUnsupportedReason {
        switch self {
        case .compilation(let description):
            return .compilation(description: description);
        case .execution(let description):
            return .execution(description: description);
        case .outputMismatch(let description):
            return .outputMismatch(description: description);
        }
    }
}
