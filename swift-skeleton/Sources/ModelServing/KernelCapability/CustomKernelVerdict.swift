import Foundation;

/// The capability verdict for one custom kernel family, port of the Rust
/// `CustomKernelVerdict`.
public enum CustomKernelVerdict: Equatable, Sendable {

    /// The probe passed; production dispatch may use the custom kernel.
    case supported;

    /// The probe failed or was never run; production dispatch must use the
    /// equivalent public MLX API for the rest of the worker process.
    case unsupported(KernelUnsupportedReason);
}
