import Foundation;

/// A bounded capability probe for one custom kernel family, port of the Rust
/// `CustomMetalKernelProbe` trait.
///
/// Implementations compile the kernel and execute a representative launch on
/// minimal inputs, validating the output against fixed deterministic expected
/// values. The runtime binding lives inside each implementation so hermetic
/// tests can drive the capability owner with fake probes.
public protocol CustomMetalKernelProbe: AnyObject {

    var family: CustomMetalKernelFamily { get }

    /// Runs the bounded probe; `.success(())` proves this GPU can run the
    /// kernel.
    func probe() -> Result<Void, KernelCapabilityError>;
}
