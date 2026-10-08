import Foundation;

/// Why one custom kernel family cannot run on this GPU, port of the Rust
/// `KernelUnsupportedReason`.
public enum KernelUnsupportedReason: Equatable, Sendable {

    /// The kernel source failed to compile on this device.
    case compilation(description: String);

    /// The bounded representative launch failed to execute.
    case execution(description: String);

    /// The launch executed but produced values that fail expected-value
    /// validation; a silently dropped dispatch returning zeros is a known
    /// historical failure signature on this platform.
    case outputMismatch(description: String);

    /// The family was never probed in this worker process. Fail closed.
    case unprobed;
}
